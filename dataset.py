# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年12月11日
"""

import os
from torch.utils.data import Dataset
import random
from PIL import Image


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
            for image_name in os.listdir(os.path.join(train_dir, label)):
                f_train.write('%s %s\n' % ('train_images\\'+label+'\\'+image_name, label))

    with open(data_root + '\\' + save_fn + '_test.txt', 'wt') as f_test:
        test_dir = os.path.join(data_root, 'test_images')
        for image_name in os.listdir(test_dir):
            f_test.write('%s %s\n' % ('test_images\\'+image_name, image_name[0]))


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
    with open(data_root + '\\' + list_fn) as f:
        data = f.readlines()
    random.shuffle(data)
    train_num = train_ratio * len(data)
    if not os.path.exists(data_root + '\\' + dir_name):
        os.mkdir(data_root + '\\' + dir_name)
    with open(data_root + '\\' + dir_name + '\\' + save_fn + '_train.txt', 'wt') as f_train:
        f_train.writelines(data[:train_num])
    with open(data_root + '\\' + dir_name + '\\' + save_fn + '_val.txt', 'wt') as f_val:
        f_val.writelines(data[train_num:])


class MnistDataset(Dataset):
    def __init__(self, root: str, data_list: str, transform=None):
        """
        按照数据列表读取图片并存储是否执行变换信息
        :param root: MNIST数据集根目录
        :param data_list: 传入数据列表的地址
        :param transform: 是否对图片进行变换，传入变换函数
        """
        images = []
        with open(data_list) as f:
            for line in f:
                line = line.split()
                images.append((line[0], int(line[1])))
        self.root = root
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.images[index]
        img = Image.open(self.root + '\\' + fn)
        if self.transform is not None:
            img = self.transform(img)
        img = img.resize((28, 28))
        return img, label

    def __len__(self):
        return len(self.images)




