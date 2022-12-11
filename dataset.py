# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年12月11日
"""

import os
from torch.utils.data import Dataset


def create_data_list(data_root: str, save_fn='data_list'):
    """
    在数据集根目录生成训练集和测试集数据读取列表，要求MNIST数据集
    :param data_root: MNIST数据集根目录
    :param save_fn: 列表名
    :return: NULL
    """
    with open(data_root + '\\' + save_fn + '_train.txt', 'wt') as f_train:
        train_dir = os.path.join(data_root, 'train_images')
        for label in os.listdir(train_dir):
            temp = os.listdir(os.path.join(train_dir, label))
            for image_name in temp[:-1]:
                f_train.write('%s %s\n' % ('train_images\\'+label+'\\'+image_name, label))
            f_train.write('%s %s' % ('train_images\\' + label + '\\' + temp[-1], label))

    with open(data_root + '\\' + save_fn + '_test.txt', 'wt') as f_test:
        test_dir = os.path.join(data_root, 'test_images')
        temp = os.listdir(test_dir)
        for image_name in temp[:-1]:
            f_test.write('%s\n' % ('test_images\\'+image_name))
        f_test.write('%s' % ('test_images\\' + temp[-1]))


def train_val_shuffle_split(data_root: str, list_fn: str, train_ratio=0.7, save_fn='data', dir_name='train_val_split'):
    """
    对create_data_list函数生成的训练数据列表按比例随机拆分为训练集和验证集
    :param data_root: MNIST数据集根目录
    :param list_fn: 读取的训练数据列表名
    :param train_ratio: 训练集占比
    :param save_fn: 保存的训练和验证集的列表名
    :param dir_name: 生成的列表保存的文件夹名
    :return: NULL
    """
