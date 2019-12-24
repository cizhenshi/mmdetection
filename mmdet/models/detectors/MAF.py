from ..registry import DETECTORS
from .two_stage import TwoStageDetector
import torch
import torch.nn as nn

from mmdet.core import bbox2result, bbox2roi, build_assigner, build_sampler
from .. import builder
from .test_mixins import BBoxTestMixin, MaskTestMixin, RPNTestMixin
from ..domain_classifier import DC_img
import numpy as np
from mmdet.core.bbox.assigners.assign_result import AssignResult
from icecream import ic

@DETECTORS.register_module
class MAF(TwoStageDetector):

    def __init__(self,
                 backbone,
                 rpn_head,
                 bbox_roi_extractor,
                 bbox_head,
                 train_cfg,
                 test_cfg,
                 da_img=None,
                 da_ins=None,
                 da_cons=None,
                 da_scale=None,
                 neck=None,
                 shared_head=None,
                 pretrained=None):
        super(MAF, self).__init__(
            backbone=backbone,
            neck=neck,
            shared_head=shared_head,
            rpn_head=rpn_head,
            bbox_roi_extractor=bbox_roi_extractor,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained)
        self.da_img = da_img
        if da_img is not None:
            self.da_img = builder.build_head(da_img)
        self.da_ins = da_ins
        if da_ins is not None:
            self.da_ins = builder.build_head(da_ins)
        self.da_cons = da_cons
        if da_cons is not None:
            assert da_ins is not None and da_img is not None
            self.da_cons = builder.build_head(da_cons)
        self.da_scale = da_scale
        if da_scale is not None:
            self.da_scale = builder.build_head(da_scale)

    def inverse_pixel_shuffle(self, x, scale_factor):
        N, C, H, W = x.shape
        oh = H / scale_factor
        ow = W / scale_factor
        out = x.new(N, C*scale_factor*scale_factor, oh, ow)
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = x[:, :, i:i+scale_factor, j:j+scale_factor].reshape(N, -1)
        return out

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self, source, target):
        loss1 = self.forward_source(**source)
        loss2 = self.forward_target(**target)
        losses = dict()
        for key in loss1:
            if key not in loss2:
                losses[key] = loss1[key]
            else:
                losses[key] = (loss1[key] + loss2[key]) / 2.0
        return losses

    def forward_source(self,
                      img,
                      img_meta,
                      gt_bboxes=None,
                      gt_labels=None,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      source=True):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()
        # comput dc loss
        dc_score = None
        ins_score = None
        if self.da_img is not None:
            dc_score = self.da_img(x)
            dc_loss = self.da_img.loss(dc_score, source)
            losses.update(dc_loss)
        # RPN forward and loss
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            rpn_loss_inputs = rpn_outs + (gt_bboxes, img_meta,
                                          self.train_cfg.rpn)
            rpn_losses = self.rpn_head.loss(
                *rpn_loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
            losses.update(rpn_losses)

            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
            proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)

        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            bbox_assigner = build_assigner(self.train_cfg.rcnn.assigner)
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = bbox_assigner.assign(proposal_list[i],
                                                     gt_bboxes[i],
                                                     gt_bboxes_ignore[i],
                                                     gt_labels[i])
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        # bbox head forward and loss
        if self.with_bbox:
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            cls_score, bbox_pred = self.bbox_head(bbox_feats)

            # compute for target
            if self.da_ins is not None:
                ins_score = self.da_ins(bbox_feats)
                loss_da_ins = self.da_ins.loss(ins_score, source)
                losses.update(loss_da_ins)

            if self.da_scale is not None:
                scale_score = self.da_scale(bbox_feats)
                loss_da_scale = self.da_scale.loss(scale_score, rois)
                losses.update(loss_da_scale)

            bbox_targets = self.bbox_head.get_target(sampling_results,
                                                     gt_bboxes, gt_labels,
                                                     self.train_cfg.rcnn)
            loss_bbox = self.bbox_head.loss(cls_score, bbox_pred,
                                            *bbox_targets)
            losses.update(loss_bbox)

        # mask head forward and loss
        if self.with_mask:
            if not self.share_roi_extractor:
                pos_rois = bbox2roi(
                    [res.pos_bboxes for res in sampling_results])
                mask_feats = self.mask_roi_extractor(
                    x[:self.mask_roi_extractor.num_inputs], pos_rois)
                if self.with_shared_head:
                    mask_feats = self.shared_head(mask_feats)
            else:
                pos_inds = []
                device = bbox_feats.device
                for res in sampling_results:
                    pos_inds.append(
                        torch.ones(
                            res.pos_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                    pos_inds.append(
                        torch.zeros(
                            res.neg_bboxes.shape[0],
                            device=device,
                            dtype=torch.uint8))
                pos_inds = torch.cat(pos_inds)
                mask_feats = bbox_feats[pos_inds]
            mask_pred = self.mask_head(mask_feats)

            mask_targets = self.mask_head.get_target(sampling_results,
                                                     gt_masks,
                                                     self.train_cfg.rcnn)
            pos_labels = torch.cat(
                [res.pos_gt_labels for res in sampling_results])
            loss_mask = self.mask_head.loss(mask_pred, mask_targets,
                                            pos_labels)
            losses.update(loss_mask)

        if self.da_cons is not None:
            loss_cons = self.da_cons.loss(dc_score, ins_score)
            losses.update(loss_cons)

        return losses

    def forward_target(self,
                       img,
                       img_meta,
                       source=False):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_meta (list[dict]): list of image info dict where each dict has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()
        # comput dc loss
        dc_score = None
        ins_score = None
        if self.da_img is not None:
            dc_score = self.da_img(x)
            dc_loss = self.da_img.loss(dc_score, source)
            losses.update(dc_loss)
        # RPN forward without loss
        if self.da_ins is not None:
            with torch.no_grad():
                rpn_outs = self.rpn_head(x)
                proposal_cfg = self.train_cfg.get('target_rpn_proposal',
                                                  self.test_cfg.rpn)
                proposal_inputs = rpn_outs + (img_meta, proposal_cfg)
                proposal_list = self.rpn_head.get_bboxes(*proposal_inputs)
            # bbox head forward and loss
            sampling_results = []
            bbox_sampler = build_sampler(
                self.train_cfg.rcnn.sampler, context=self)
            num_imgs = img.size(0)
            for i in range(num_imgs):
                bboxes = proposal_list[i]
                num_bboxes = bboxes.size(0)
                assigned_gt_inds = bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
                max_overlaps = bboxes.new_full((num_bboxes,), 0, dtype=torch.long)
                assign_result = AssignResult(0, assigned_gt_inds, max_overlaps, labels=None)
                pesudo_gt = bboxes.new(0, 4)
                pesudo_label = bboxes.new(0,)
                sampling_result = bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    pesudo_gt,
                    pesudo_label,
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)
            rois = bbox2roi([res.bboxes for res in sampling_results])
            # TODO: a more flexible way to decide which feature maps to use
            bbox_feats = self.bbox_roi_extractor(
                x[:self.bbox_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                bbox_feats = self.shared_head(bbox_feats)
            # cls_score, bbox_pred = self.bbox_head(bbox_feats)

            # compute for target
            ins_score = self.da_ins(bbox_feats)
            loss_da_ins = self.da_ins.loss(ins_score, source)
            losses.update(loss_da_ins)

        if self.da_cons is not None:
            loss_cons = self.da_cons.loss(dc_score, ins_score)
            losses.update(loss_cons)
        return losses

    def random_choice(self, gallery, num):
        """Random select some elements from the gallery.

        It seems that Pytorch's implementation is slower than numpy so we use
        numpy to randperm the indices.
        """
        assert len(gallery) >= num
        if isinstance(gallery, list):
            gallery = np.array(gallery)
        cands = np.arange(len(gallery))
        np.random.seed(1)
        np.random.shuffle(cands)
        rand_inds = cands[:num]
        if not isinstance(gallery, np.ndarray):
            rand_inds = torch.from_numpy(rand_inds).long().to(gallery.device)
        return gallery[rand_inds]

    def simple_test(self, img, img_meta, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        x = self.extract_feat(img)

        proposal_list = self.simple_test_rpn(
            x, img_meta, self.test_cfg.rpn) if proposals is None else proposals

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_meta, proposal_list, self.test_cfg.rcnn, rescale=rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.simple_test_mask(
                x, img_meta, det_bboxes, det_labels, rescale=rescale)
            return bbox_results, segm_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        # recompute feats to save memory
        proposal_list = self.aug_test_rpn(
            self.extract_feats(imgs), img_metas, self.test_cfg.rpn)
        det_bboxes, det_labels = self.aug_test_bboxes(
            self.extract_feats(imgs), img_metas, proposal_list,
            self.test_cfg.rcnn)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)

        # det_bboxes always keep the original scale
        if self.with_mask:
            segm_results = self.aug_test_mask(
                self.extract_feats(imgs), img_metas, det_bboxes, det_labels)
            return bbox_results, segm_results
        else:
            return bbox_results
