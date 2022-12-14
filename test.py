# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年12月14日
"""

import argparse
import os
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import dataset
from torchvision import transforms
from pathlib import Path
import datetime
from torch.utils.data import DataLoader
import shutil

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
    if y.is_cuda:
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['MiniVGG', 'MLP'], default='MLP', help='从MiniVGG和MLP中选择你所希望使用的网络模型（默认MLP）')
    parser.add_argument('--pth_path', type=str, required=True, help='传入所使用的模型参数pth文件地址')
    parser.add_argument('--gpu', type=str, default='0', help='选择所使用的gpu（默认GPU 0）')
    parser.add_argument('--cpu', action='store_true', default=False, help='是否使用cpu训练（默认否，不需传参）')
    parser.add_argument('--root', type=str, required=True, help='传入数据集根目录')
    parser.add_argument('--batch_size', type=int, default=128, help='设置batch大小')
    parser.add_argument('--save_name', type=str, default=None, help='日志存放文件夹名')
    parser.add_argument('--copy_model', action='store_true', default=False, help='是否复制模型文件到log目录（默认否，不需传参）')

    return parser.parse_args()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    """创建文件夹"""
    time_log = str(datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm"))
    save_dir = Path('./save/test')
    save_dir.mkdir(exist_ok=True, parents=True)

    if args.save_name is not None:
        save_dir = save_dir.joinpath(args.save_name)
    else:
        save_dir = save_dir.joinpath(time_log)
    save_dir.mkdir(exist_ok=True)
    """保存日志"""
    logger = logging.getLogger('TEST')
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (save_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(info):
        logger.info(info)
        print(info)

    """复制模型文件"""
    if args.copy_model:
        shutil.copy(args.pth_path, save_dir)
        log_string('复制模型文件到：%s' % save_dir)

    log_string('参数：')
    log_string(args)
    MODEL = importlib.import_module(args.model)

    test_list = dataset.create_data_list(args.root, train=False)
    test_data = dataset.MnistDataset(args.root, test_list, transform=transforms.ToTensor(), label_transform=to_categorical)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=False)
    model = MODEL.MyModel()

    log_string('测试集图片数量：%d' % len(test_data))

    if not args.cpu:
        model.cuda()
        log_string('使用gpu测试......')

    checkpoint = torch.load(args.pth_path)
    model.load_state_dict(checkpoint['model.state_dict'])

    test_acc = []
    model.eval()
    with torch.no_grad():
        for batch, (batch_x, batch_y) in tqdm(enumerate(test_loader), total=len(test_loader)):
            if not args.cpu:
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()

            out = model(batch_x)
            pred = to_categorical(torch.max(out, dim=1)[1])
            num_correct = torch.min(pred.eq(batch_y), dim=1)[0].data.cpu().sum()
            test_acc.append(num_correct / args.batch_size)

    global_test_acc = np.mean(test_acc)
    log_string('测试Accuracy：%.5f' % global_test_acc)
    log_string(5 * '*' + '测试结束' + 5 * '*')
    log_string('模型其他参数：')
    log_string('模型训练epoch：%d' % checkpoint['epoch'])
    log_string('模型训练Accuracy：%.5f' % checkpoint['train_instance_acc'])
    log_string('模型验证集最高Accuracy：%.5f' % checkpoint['best_val_acc'])
    log_string('\n')


if __name__ == '__main__':
    args = parse_args()
    main(args)
