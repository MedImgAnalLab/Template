#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Template 
@File    ：ResNet.py
@Author  ：Yu Hui
@Date    ：2023/7/26 15:11 
"""
import torch
from torch import nn
from torch.nn import functional as F

from utils.utils import resnet_type


class ResNet(nn.Module):
    def __init__(self, backbone='resnet50', weights=None):
        super(ResNet, self).__init__()
        self.backbone = backbone
        self.weights = weights
        self.resnet = resnet_type[backbone](weights=self.weights)

    def forward(self, x):
        return self.resnet(x)


if __name__ == '__main__':
    model = ResNet()
    print(model)
    print(model(torch.randn(1, 3, 224, 224)).shape)
