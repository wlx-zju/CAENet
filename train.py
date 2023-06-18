import argparse
import time
import os
from tqdm import tqdm
import torch
import torch.optim as optim

from models import CAENet
from dataloader import get_data_loader
from loss import LossSum, init_losses, update_losses
from metrics import CriteriaDict, init_criteria_ave
from logger_utils import init_logger, inform_logger, close_logger, show
from evaluation import test_process


def train_process(args):
    torch.backends.cudnn.benchmark = True
    # ------------------------------ initialization for other training utils ------------------------------ #
    logger = init_logger(args)

    train_loader, test_loader, class_num = get_data_loader(args.tasks, args.dataset, args.augmentation, args.batch_size)
    train_batch_num = len(train_loader)
    show('{} batches in an epoch'.format(train_batch_num), logger)

    # criteria dictionary for specific tasks
    cd_test = CriteriaDict(args.tasks)  # for the test data
    c_average = init_criteria_ave(args.tasks)  # for the test data, during the last n epochs
    net  = CAENet(tasks=args.tasks,
                  class_num=class_num,
                  pretrained=args.pretrained,
                  backbone=args.backbone).cuda()

    show('initial learning rate is {}'.format(args.lr), logger)
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9)
    # ------------------------------ initialization for loss weight related ------------------------------ #
    if args.dataset == 'city':
        loss_sum = LossSum(tasks=args.tasks, loss_weights=[1, 0.5]).cuda()
    else:
        loss_sum = LossSum(tasks=args.tasks).cuda()

    # ------------------------------ Let's start! The real training process! ------------------------------ #
    for epoch in range(args.epochs):
        epoch_train_time = time.time()

        # reset the criteria dictionary and loss
        cd_test.reset()
        train_losses = init_losses(args.tasks)
        scheduler.step()
        net.train()

        train_data = iter(train_loader)
        loss_value = 0
        for k in tqdm(range(train_batch_num), position=0, leave=True, ncols=50, bar_format='{desc}{percentage:3.0f}%|{bar}|{n_fmt}/{total_fmt}[{elapsed}<{remaining}]'):
            train_img_b, train_gt_b = train_data.next()
            train_pred_b = net(train_img_b)
            loss, task_losses = loss_sum(train_pred_b, train_gt_b)
            loss_value += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            update_losses(train_losses, task_losses, train_batch_num)

        epoch_test_time = time.time()
        test_losses = test_process(net, test_loader, loss_sum, cd_test, c_average, epoch, args.epochs, last_epochs=5)
        epoch_end_time = time.time()
        time_list = [epoch_train_time, epoch_test_time, epoch_end_time]
        inform_logger(logger, epoch, cd_test, train_losses, args.tasks, time_list)
        loss_value /= train_batch_num
        show('label_loss: {:1.4f}'.format(loss_value), logger)
        show('', logger)

    close_logger(logger, c_average, args.tasks)
    if args.save_dir != 'no_save':
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        torch.save(net.state_dict(), os.path.join(args.save_dir, args.log_dir_specific+'.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train process')
    # model related
    parser.add_argument('--backbone', default='mobilenetv2', type=str, help='mobilenetv2 or resnet18 or stdcnet')
    parser.add_argument('--tasks', nargs='+', default=['semantic', 'depth'])
    parser.add_argument('--pretrained', action='store_true', help='use pretrained backbone weights')
    parser.add_argument('--lr', default=5e-4, type=float)
    # dataset related
    parser.add_argument('--dataset', default='nyuv2', type=str, help='nyuv2 or city (stands for cityscapes)')
    parser.add_argument('--augmentation', action='store_true', help='apply augmentation to the dataset')
    parser.add_argument('--batch_size', default=8, type=int)
    # training related
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--log_dir_common', default='./logs/nyuv2', type=str)
    parser.add_argument('--log_dir_specific', default='caenet_mobilenetv2_x1', type=str)
    parser.add_argument('--save_dir', default='no_save', type=str)
    parser.add_argument('--comment', default='none', type=str)

    args = parser.parse_args()

    train_process(args)

