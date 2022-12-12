# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年12月12日
"""
import sys
import logging
import torch
import torch.nn.functional as F
import importlib
import os
import argparse
import numpy as np
import datetime
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))


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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['MiniVGG', 'MLP'], default='MLP', help='从MiniVGG和MLP中选择你所希望使用的网络模型（默认MLP）')
    parser.add_argument('--epoch', type=int, default=250, help='输入训练次数（默认250）')
    parser.add_argument('--gpu', type=str, default='0', help='选择所使用的gpu（默认GPU 0）')
    parser.add_argument('--cpu', action='store_true', help='是否使用cpu训练（默认否，不需传参）')
    parser.add_argument('--root', type=str, required=True, help='传入数据集根目录')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='设置batch大小')
    parser.add_argument('--save_name', type=str, default=None, help='模型文件与日志保存名')

    return parser.parse_args()


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # HYPER PARAMETER

    """创建文件夹"""
    time_log = str(datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm"))  # 日志文件名
    save_dir = Path('./save/')
    save_dir.mkdir(exist_ok=True)
    if args.model == 'MLP':
        save_dir = save_dir.joinpath('MLP')
    else:
        save_dir = save_dir.joinpath('MiniVGG')
    if args.log_name is None:
        save_dir = save_dir.joinpath(time_log)
    else:
        save_dir = save_dir.joinpath(args.log_name)
    checkpoints_dir = save_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = save_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    """保存日志"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y/%m/%d %H:%M:%S')
    file_handler =

