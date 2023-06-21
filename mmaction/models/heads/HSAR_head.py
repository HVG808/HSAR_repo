# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import trunc_normal_init

from ..builder import HEADS
from .base import BaseHead
import ipdb


@HEADS.register_module()
class HSARHead(BaseHead):
    """Classification head for TimeSformer.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to `dict(type='CrossEntropyLoss')`.
        init_std (float): Std value for Initiation. Defaults to 0.02.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 init_std=0.02,
                 dropout_ratio=0.1,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)
        self.init_std = init_std
        dropout = 0.1
        # self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        hidden_dim = self.in_channels * 4
        self.pool = nn.MaxPool1d(3, stride=1, padding=1)
        self.fc_cls = nn.Sequential(
            nn.Linear(self.in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_classes)
        )


    def init_weights(self):
        """Initiate the parameters from scratch."""
        trunc_normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        # [N, in_channels]
        x = self.pool(x)
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
