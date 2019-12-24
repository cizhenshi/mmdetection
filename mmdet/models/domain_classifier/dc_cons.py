import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from mmdet.core import (auto_fp16, bbox_target, delta2bbox, force_fp32,
                        multiclass_nms)
from ..builder import build_loss
from ..losses import accuracy
from ..registry import HEADS
from ..utils import ConvModule, GradientScalarLayer
from icecream import ic


@HEADS.register_module
class DAConsHead(nn.Module):
    """instance level domain classifier"""

    def __init__(self, loss_weight=0.1):
        super(DAConsHead, self).__init__()
        self.loss_weight = loss_weight

    @force_fp32(apply_to='cls_score')
    def loss(self,
             img_score,
             ins_score):
        losses = dict()
        N, C, H, W = img_score.shape
        ins_score = ins_score.squeeze().view(N, -1)
        ins_score = F.sigmoid(ins_score)
        _, number = ins_score.shape

        img_score = img_score.view(N, -1)
        img_score = F.sigmoid(img_score)
        avg_img_score = img_score.mean(dim=1).unsqueeze(1)
        avg_img_score = avg_img_score.repeat(1, number)
        losses['loss_cons'] = self.loss_weight * ((ins_score - avg_img_score)**2).mean()
        return losses


# def consistency_loss(img_feas, ins_fea, ins_labels, size_average=True):
#     """
#     Consistency regularization as stated in the paper
#     `Domain Adaptive Faster R-CNN for Object Detection in the Wild`
#     L_cst = \sum_{i,j}||\frac{1}{|I|}\sum_{u,v}p_i^{(u,v)}-p_{i,j}||_2
#     """
#     loss = []
#     len_ins = ins_fea.size(0)
#     intervals = [torch.nonzero(ins_labels).size(0), len_ins-torch.nonzero(ins_labels).size(0)]
#     for img_fea_per_level in img_feas:
#         N, A, H, W = img_fea_per_level.shape
#         img_fea_per_level = torch.mean(img_fea_per_level.reshape(N, -1), 1)
#         img_feas_per_level = []
#         assert N==2, \
#             "only batch size=2 is supported for consistency loss now, received batch size: {}".format(N)
#         for i in range(N):
#             img_fea_mean = img_fea_per_level[i].view(1, 1).repeat(intervals[i], 1)
#             img_feas_per_level.append(img_fea_mean)
#         img_feas_per_level = torch.cat(img_feas_per_level, dim=0)
#         loss_per_level = torch.abs(img_feas_per_level - ins_fea)
#         loss.append(loss_per_level)
#     loss = torch.cat(loss, dim=1)
#     if size_average:
#         return loss.mean()
#     return loss.sum()