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
class DC_img(nn.Module):
    """img level domain classifier"""

    def __init__(self,
                 in_channels=512,
                 feat_channels=512,
                 grl_weight=-0.1,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.1),):
        super(DC_img, self).__init__()
        self.da_conv = nn.Conv2d(in_channels, feat_channels, kernel_size=1, stride=1)
        self.da_cls = nn.Conv2d(feat_channels, 1, kernel_size=1, stride=1)
        self.loss_cls = build_loss(loss_cls)
        self.grl_img = GradientScalarLayer(grl_weight)
        self.init_weights()

    def init_weights(self):
        for l in [self.da_conv, self.da_cls]:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    @auto_fp16()
    def forward(self, x):
        features = self.grl_img(x[0])
        features = F.relu(self.da_conv(features))
        dc_scores = self.da_cls(features)
        return dc_scores

    @force_fp32(apply_to='cls_score')
    def loss(self,
             cls_score,
             source,
             reduction_override=None):
        losses = dict()
        N, C, H, W = cls_score.shape
        domain_label = 0 if source else 1
        labels = torch.zeros_like(cls_score, dtype=torch.float32)
        labels[:, :] = domain_label
        cls_score = cls_score.view(N, -1)
        labels = labels.view(N, -1)
        losses['loss_dc_img'] = self.loss_cls(
                    cls_score,
                    labels,
                    reduction_override=reduction_override)
        return losses


class MultiDAImg(nn.Module):
    """img level domain classifier"""

    def __init__(self,
                 in_channels=[128, 256, 512],
                 out_channels=[64, 256, 512],
                 feat_channels=512,
                 grl_weight=-0.1,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.1),):
        super(MultiDAImg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.da_conv1 = nn.ModuleList([
            ConvModule(in_channels[0], out_channels[0], 1),
            ConvModule(out_channels[0], feat_channels, 1),
        ])
        self.da_conv2 = nn.ModuleList([
            ConvModule(in_channels[1], out_channels[1], 1),
            ConvModule(out_channels[1], feat_channels, 1),
        ])
        self.da_conv3 = nn.ModuleList([
            ConvModule(in_channels[2], out_channels[2], 1),
            ConvModule(out_channels[2], feat_channels, 1),
        ])

        self.da_cls = nn.ModuleList([
            nn.Conv2d(feat_channels, 2, kernel_size=1),
            nn.Conv2d(feat_channels, 2, kernel_size=1),
            nn.Conv2d(feat_channels, 2, kernel_size=1),
        ])
        self.loss_cls = build_loss(loss_cls)
        self.grl_img = GradientScalarLayer(grl_weight)
        self.init_weights()

    def init_weights(self):
        for l in self.da_cls:
            torch.nn.init.normal_(l.weight, std=0.001)
            torch.nn.init.constant_(l.bias, 0)

    def inverse_pixel_shuffle(self, x, scale_factor):
        N, C, H, W = x.shape
        oh = H / scale_factor
        ow = W / scale_factor
        out = x.new(N, C*scale_factor*scale_factor, oh, ow)
        for i in range(oh):
            for j in range(ow):
                out[:, :, i, j] = x[:, :, i:i+scale_factor, j:j+scale_factor].reshape(N, -1)
        return out

    @auto_fp16()
    def forward(self, x):
        c3 = x[0]
        c4 = x[1]
        c5 = x[2]
        c3 = self.da_conv1[0](c3)
        c3 = self.da_conv1[1](c3)

        c4 = self.da_conv2[0](c4)
        c4 = self.da_conv2[0](c4)

        c5 = self.da_conv3[0](c5)
        c5 = self.da_conv3[1](c5)

        return score_list

    def loss_single(self, score, source):
        N, C, H, W = score.shape
        domain_label = 0 if source else 1
        labels = torch.zeros_like(score, dtype=torch.float32)
        labels[:, :] = domain_label
        cls_score = score.view(N, -1)
        labels = labels.view(N, -1)
        losses = self.loss_cls(
            cls_score,
            labels,
            reduction_override=reduction_override)
        return losses

    @force_fp32(apply_to='cls_score')
    def loss(self,
             score_list,
             source,
             reduction_override=None):
        loss_imgs = multi_apply(
            self.loss_single,
            score_list,
            source)
        return dict(loss_dc_img=loss_imgs)

