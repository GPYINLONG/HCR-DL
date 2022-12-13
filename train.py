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
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import datetime
from pathlib import Path
import dataset
import tqdm

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
    parser.add_argument('--cpu', action='store_true', default=False,help='是否使用cpu训练（默认否，不需传参）')
    parser.add_argument('--root', type=str, required=True, help='传入数据集根目录')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=16, help='设置batch大小')
    parser.add_argument('--save_name', type=str, default=None, help='模型文件与日志存放文件夹名')
    parser.add_argument('--split_ratio', type=float, default=0.8, help='训练集/(训练集+验证集)的比例')

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
    if args.save_name is None:
        save_dir = save_dir.joinpath(time_log)
    else:
        save_dir = save_dir.joinpath(args.save_name)
    checkpoints_dir = save_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = save_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    """保存日志"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%Y/%m/%d %H:%M:%S')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    def log_string(info):
        logger.info(info)
        print(info)

    log_string('参数：')
    log_string(args)

    """Dataset与Dataloader"""
    train_list = dataset.create_data_list(args.root, train=True)
    train_list, val_list = dataset.train_val_shuffle_split(train_list, train_ratio=args.split_ratio)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomCrop(24),
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)
    ])
    train_data = dataset.MnistDataset(args.root, train_list, transform=transform, label_transform=to_categorical)
    val_data = dataset.MnistDataset(args.root, val_list, transform=transforms.ToTensor(), label_transform=to_categorical)
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=args.batch_size)

    log_string('训练集图片数量：%d' % len(train_data))
    log_string('验证集图片数量：%d' % len(val_data))

    """加载模型"""
    MODEL = importlib.import_module(args.model)
    model = MODEL.MyModel()
    if not args.cpu:
        model.cuda()
        log_string('使用gpu训练......')

    """继续训练"""
    try:
        checkpoint = torch.load(str(save_dir) + '\\checkpoints\\best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('使用预训练模型继续训练......')
    except:
        start_epoch = 0
        log_string(args.save_name + '下未检测到已存在模型，进行新的训练')

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-03
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [80, 150], 0.1)

    loss_func = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    global_epoch = 0
    global_train_loss = []
    global_val_loss = []
    for epoch in range(start_epoch, args.epoch):
        train_acc = []
        train_loss = 0
        log_string('Epoch %d (%d/%s)' % (global_epoch + 1, epoch + 1, args.epoch))
        model.train()
        for batch, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            if not args.cpu:
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()

            out = model(batch_x)

            pred = torch.max(out, 1)[1]
            num_correct = pred.eq(batch_y.data).cpu().sum()
            train_acc.append(num_correct / args.batch_size)

            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_instance_acc = np.mean(train_acc).data
        global_train_loss.append(train_loss)
        log_string('第%d个epoch训练loss：%.6f，Accuracy：%.5f' % (start_epoch + 1, train_loss, train_instance_acc))


        global_epoch += 1
