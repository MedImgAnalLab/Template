#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Template 
@File    ：utils.py
@Author  ：Yu Hui
@Date    ：2023/7/26 15:09 
"""
import os
from random import random

import numpy as np
import torch
from torchvision import models

resnet_type = {
    'resnet18': models.resnet18,
    'resnet34': models.resnet34,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101,
    'resnet152': models.resnet152
}


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def seed_everything(seed=42):
    """
    Seeds basic parameters for reproductibility of results

    Args:
        seed (int, optional): Number of the seed. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
