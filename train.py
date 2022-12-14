# -*- coding:utf-8 -*-
"""
作者：机智的枫树
日期：2022年12月12日
"""
import sys
import logging
import torch
import importlib
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import argparse
import numpy as np
import datetime
from pathlib import Path
import dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

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
    parser.add_argument('--epoch', type=int, default=250, help='输入训练次数（默认250）')
    parser.add_argument('--gpu', type=str, default='0', help='选择所使用的gpu（默认GPU 0）')
    parser.add_argument('--cpu', action='store_true', default=False, help='是否使用cpu训练（默认否，不需传参）')
    parser.add_argument('--root', type=str, required=True, help='传入数据集根目录')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--batch_size', type=int, default=128, help='设置batch大小')
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
    checkpoints_dir.mkdir(exist_ok=True, parents=True)
    log_dir = save_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True, parents=True)

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
        transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
        transforms.Resize(28)
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

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=1e-03
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [int(0.3 * args.epoch), int(0.7 * args.epoch)], 0.1)

    """继续训练"""
    try:
        checkpoint = torch.load(str(save_dir) + '\\checkpoints\\best_model.pth')
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint['best_val_acc']
        global_train_loss = checkpoint['global_train_loss']
        global_val_loss = checkpoint['global_val_loss']
        global_train_acc = checkpoint['global_train_acc']
        global_val_acc = checkpoint['global_val_acc']
        model.load_state_dict(checkpoint['model.state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer.state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler.state_dict'])

        log_string('使用预训练模型继续训练......')
    except:
        start_epoch = 0
        best_val_acc = 0
        global_train_loss = [0]
        global_val_loss = [0]
        global_train_acc = [0]
        global_val_acc = [0]
        log_string(str(save_dir) + '下未检测到已存在模型，进行新的训练')

    loss_func = torch.nn.CrossEntropyLoss()

    global_epoch = 0
    for epoch in range(start_epoch, args.epoch):
        train_acc = []
        train_loss = 0
        val_acc = []
        val_loss = 0
        log_string('Epoch %d (%d/%s)' % (global_epoch + 1, epoch + 1, args.epoch))
        model.train()
        for batch_train, (batch_x, batch_y) in tqdm(enumerate(train_loader), total=len(train_loader)):
            if not args.cpu:
                batch_x = batch_x.float().cuda()
                batch_y = batch_y.float().cuda()

            out = model(batch_x)

            pred = to_categorical(torch.max(out, 1)[1])
            num_correct = torch.min(pred.eq(batch_y), dim=1)[0].data.cpu().sum()
            train_acc.append(num_correct / args.batch_size)

            loss = loss_func(out, batch_y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()
        train_instance_acc = np.mean(train_acc)
        global_train_acc.append(train_instance_acc)
        global_train_loss.append(train_loss)
        log_string('训练loss：%.6f，Accuracy：%.5f' % (train_loss, train_instance_acc))

        model.eval()
        with torch.no_grad():
            for batch_val, (batch_x, batch_y) in tqdm(enumerate(val_loader), total=len(val_loader)):
                if not args.cpu:
                    batch_x = batch_x.float().cuda()
                    batch_y = batch_y.float().cuda()

                out = model(batch_x)

                pred = to_categorical(torch.max(out, 1)[1])
                num_correct = torch.min(pred.eq(batch_y), dim=1)[0].data.cpu().sum()
                val_acc.append(num_correct / args.batch_size)
                loss = loss_func(pred, batch_y)
                val_loss += loss.item()
        val_instance_acc = np.mean(val_acc)
        global_val_acc.append(val_instance_acc)
        global_val_loss.append(val_loss)
        log_string('验证loss：%.6f，Accuracy：%.5f' % (val_loss, val_instance_acc))
        if val_instance_acc > best_val_acc:
            best_val_acc = val_instance_acc
            log_string('保存模型中......')
            state = {
                'epoch': epoch + 1,
                'model.state_dict': model.state_dict(),
                'optimizer.state_dict': optimizer.state_dict(),
                'scheduler.state_dict': scheduler.state_dict(),
                'train_instance_acc': train_instance_acc,
                'best_val_acc': best_val_acc,
                'global_train_loss': global_train_loss,
                'global_val_loss': global_val_loss,
                'global_train_acc': global_train_acc,
                'global_val_acc': global_val_acc
            }
            torch.save(state, str(checkpoints_dir) + '\\best_model.pth')
            log_string('模型保存完成，路径：%s' % str(checkpoints_dir))
        if epoch < args.epoch - 1:
            log_string(10 * '*' + '进入下一个epoch' + 10 * '*' + '\n')
        global_epoch += 1
    log_string('训练结束......')
    log_string('全局最高验证集正确率：%.5f' % best_val_acc)

    """保存loss图"""
    plt.figure('loss')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(global_train_loss, label='train_loss')
    plt.plot(global_val_loss, label='val_loss')
    plt.xlim(1,)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=0)
    plt.savefig(str(log_dir)+'\\loss.jpg')
    log_string('loss图保存完成，路径：%s' % str(log_dir))

    """保存Accuracy图"""
    plt.figure('accuracy')
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.plot(global_train_acc, label='train_acc')
    plt.plot(global_val_acc, label='val_acc')
    plt.xlim(1,)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc=0)
    plt.savefig(str(log_dir)+'\\Accuracy.jpg')
    log_string('Accuracy图保存完成，路径：%s' % str(log_dir))


if __name__ == '__main__':
    args = parse_args()
    main(args)
