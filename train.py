# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年12月12日
"""

import torch
import torch.nn.functional as F
import importlib
import os
import argparse
import numpy as np


def to_categorical(y, num_classes=10):
    """
    对一个张量标签进行独热编码
    :param y: 标签张量
    :param num_classes: 分类数，默认为10
    :return: 独热编码的新标签
    """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy()]
    if y.is_cuda():
        return new_y.cuda()
    return new_y


