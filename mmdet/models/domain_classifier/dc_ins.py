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
class DAInsHead(nn.Module):
    """instance level domain classifier"""

    def __init__(self,
                 in_channels=25088,
                 feat_channels=1024,
                 grl_weight=-0.1,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=0.1),):
        super(DAInsHead, self).__init__()
        self.da_fc1 = nn.Linear(in_channels, feat_channels)
        self.da_fc2 = nn.Linear(feat_channels, feat_channels)
        self.da_cls = nn.Linear(feat_channels, 1)
        self.loss_cls = build_loss(loss_cls)
        self.grl_img = GradientScalarLayer(grl_weight)
        self.init_weights()

    def init_weights(self):
        for l in [self.da_fc1, self.da_fc2]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        nn.init.normal_(self.da_cls.weight, std=0.05)
        nn.init.constant_(self.da_cls.bias, 0)

    @auto_fp16()
    def forward(self, x):
        x = self.grl_img(x)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.da_fc1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.da_fc2(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.da_cls(x)
        return x

    @force_fp32(apply_to='cls_score')
    def loss(self,
             cls_score,
             source,
             reduction_override=None):
        losses = dict()
        N = cls_score.shape
        domain_label = 0 if source else 1
        labels = torch.zeros_like(cls_score, dtype=torch.float32)
        labels[:, :] = domain_label
        cls_score = cls_score.view(N, -1)
        labels = labels.view(N, -1)
        losses['loss_dc_ins'] = self.loss_cls(
                    cls_score,
                    labels,
                    reduction_override=reduction_override)
        return losses
