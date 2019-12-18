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
                 in_channel=512,
                 out_channel=512,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.1),):
        super(DC_img, self).__init__()
        self.da_conv = nn.Conv2d(in_channel, 512, kernel_size=1, stride=1)
        self.da_cls = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.loss_cls = build_loss(loss_cls)
        self.grl_img = GradientScalarLayer(-0.1)
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
        losses['loss_dc_img'] = 0.1*F.binary_cross_entropy_with_logits(
            cls_score, labels
        )
        # losses['loss_dc_img'] = self.loss_cls(
        #             cls_score,
        #             labels,
        #             reduction_override=reduction_override)
        return losses
