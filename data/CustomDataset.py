#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：Template 
@File    ：CustomDataset.py
@Author  ：Yu Hui
@Date    ：2023/7/26 15:15 
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def load_custom_data(_path):
    """
    Load the data from the np file
    :param _path:
    :return:
    """
    _data = np.load(_path, allow_pickle=True)
    _img = _data.item()['img']
    _mask = _data.item()['mask']
    _label = _data.item()['label']
    return _img, _mask, _label


class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None, train=True, test=False):
        super(CustomDataset, self).__init__()
        self.data_path = data_path
        self.transform = transform
        self.train = train
        self.test = test

        if self.train:
            self.data_path = os.path.join(self.data_path, 'train')

        if self.test:
            self.data_path = os.path.join(self.data_path, 'test')

        self.data_paths = [os.path.join(self.data_path, i) for i in os.listdir(self.data_path)]

    def __getitem__(self, item):
        this_path = self.data_paths[item]
        img, mask, label = load_custom_data(this_path)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if self.transform is not None:
            img = self.transform(img)
            mask = self.transform(mask)
        return img, mask, label

    def __len__(self):
        return len(self.data_paths)


if __name__ == '__main__':
    pass
