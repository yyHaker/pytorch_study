# -*- coding: utf-8 -*-
"""
use cnn_finetune to finetune our network.
"""
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

from cnn_finetune import make_model

import pandas as pd
import logging
import random
from dataUtils import ImageSceneData, ImageSceneTestData
from myutils import write_data_to_file

# reproduce
# random.seed(1)
# torch.manual_seed(1)
# device
device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch fine tune scene data Training')
# parser.add_argument('--data', metavar='DIR', default='image_scene_data/data', help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet34)')
parser.add_argument('--num_classes', default=20, type=int,
                    help="num of classes to classify")
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning_rate', default=0.0001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument("--optim", '--op', default='momentum', type=str,
                    help='use what optimizer ')
parser.add_argument('--print_freq', '-p', default=104, type=int,
                    metavar='N', help='print frequency (default: 100 batch)')

parser.add_argument('--resume', default='result/res34/checkpoint.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--log_path', default='result/res34/log.log', type=str,
                    help="path to save logs")
parser.add_argument('--test_dir', default='test_a', type=str,
                    help='test data dir')

parser.add_argument('--pretrained', dest='pretrained', default=True, action='store_true',
                    help='use pre-trained model')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--predict', dest='predict', action='store_true',
                    help='use  model to do prediction')

best_prec1 = 0  # best precision
best_prec3 = 0

# remember some data to plot
losses_dict = {"train_loss": [], "valid_loss": []}
prec_dict = {"train_p1": [], "train_p3": [], "valid_p1": [], "valid_p3": []}


def main():
    global args, best_prec1, best_prec3, losses_dict, prec_dict
    args = parser.parse_args()
    logger = logging.getLogger("scene classification")

    # create model
    model = make_model(args.arch, num_classes=args.num_classes, pretrained=args.pretrained)
    model.to(device)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim == "momentum":
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                     weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            best_prec3 = checkpoint['best_prec3']
            losses_dict = checkpoint['losses_dict']
            prec_dict = checkpoint['prec_dict']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_dataset = ImageSceneData(categories_csv='image_scene_data/categories.csv',
                                   list_csv='image_scene_data/train_list.csv',
                                   data_root='image_scene_data/data',
                                   transform=transforms.Compose([
                                    transforms.Resize((random.randint(256, 480), random.randint(256, 480))),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    transforms.RandomAffine(random.randint(1, 90)),
                                    transforms.RandomRotation(random.randint(1, 90)),
                                    transforms.RandomCrop(224),
                                    transforms.ToTensor(),
                                ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    logger.info("train data analyze:{}".format(train_dataset.analyze_data()))

    valid_dataset = ImageSceneData(categories_csv='image_scene_data/categories.csv',
                                   list_csv='image_scene_data/valid_list.csv',
                                   data_root='image_scene_data/data',
                                   transform=transforms.Compose([
                                       transforms.Resize((random.randint(256, 480), random.randint(256, 480))),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.FiveCrop(224),
                                       transforms.Lambda(lambda crops: torch.stack([
                                           transforms.ToTensor()(crop) for crop in crops]))
                                   ]))
    val_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    logger.info("valid data analyze: {}".format(valid_dataset.analyze_data()))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    for epoch in range(args.start_epoch, args.epochs):

        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss, train_p1, train_p3 = train(train_loader, model, criterion, optimizer, epoch)
        losses_dict["train_loss"].append(train_loss)
        prec_dict["train_p1"].append(train_p1)
        prec_dict["train_p3"].append(train_p3)

        # evaluate on validation set
        valid_loss, valid_prec1, valid_prec3 = validate(val_loader, model, criterion)
        losses_dict["valid_loss"].append(valid_loss)
        prec_dict["valid_p1"].append(valid_prec1)
        prec_dict["valid_p3"].append(valid_prec3)

        # test teh type
        # print("type(train_loss): {}, type(train_p1): {}, type(train_p3): {}".format(
        #    type(train_loss), type(train_p1), type(train_p3)))
        # print("type(valid_loss): {}, type(valid_p1): {}, type(valid_p3): {}".format(
        #    type(valid_loss), type(valid_prec1), type(valid_prec3)))
        # print("type(loss_dict): {}, type(losses_dict['train_loss']): {})".format(
        #     type(losses_dict), type(losses_dict["train_loss"])))

        # remember best prec@1 and save checkpoint
        is_best = valid_prec1 > best_prec1
        if is_best:
            best_prec3 = valid_prec3
        best_prec1 = max(valid_prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'best_prec3': best_prec3,
            'losses_dict': losses_dict,
            'prec_dict': prec_dict,
            'optimizer': optimizer.state_dict(),
        }, is_best)
        logger.info("epoch {}, current the best model valid prec1: {}, prec3: {}, cur learning rate: {}".format(
            epoch, best_prec1, best_prec3, optimizer.param_groups[0]['lr']))
    logger.info("training is done!")
    # save the plot data
    logger.info("save result to plot")
    write_data_to_file(losses_dict, "result/res34/loss_dict.pkl")
    write_data_to_file(prec_dict, "result/res34/prec_dict.pkl")


# training on one epoch
def train(train_loader, model, criterion, optimizer, epoch):
    logger = logging.getLogger("scene classification")
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, sample in enumerate(train_loader):
        input = Variable(sample['image']).to(device)
        target = Variable(sample['label']).to(device)
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec3 = accuracy(output, target, topk=(1, 3))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top3.update(prec3.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top3=top3))
    return losses.val, top1.val, top3.val


def validate(val_loader, model, criterion):
    logger = logging.getLogger("scene classification")
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top3 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, sample in enumerate(val_loader):
            input = Variable(sample['image']).to(device)
            target = Variable(sample['label']).to(device)
            # 5-crop cope with input is a 5d tensor, target is 2d
            bs, ncrops, c, h, w = input.size()
            # compute output
            output = model(input.view(-1, c, h, w))  # fuse batch size and ncrops
            output = output.view(bs, ncrops, -1).mean(1)  # avg over crops
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec3 = accuracy(output, target, topk=(1, 3))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top3.update(prec3.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logger.info('Valid: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))

        logger.info(' Valid:  * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'.format(
            top1=top1, top3=top3))

    return losses.val, top1.avg, top3.avg


def predict():
    """predict the result"""
    args = parser.parse_args()
    logger = logging.getLogger("scene classification")
    test_dir = args.test_dir

    # load model
    logger.info("load models....")
    # create model
    model = make_model(args.arch, num_classes=args.num_classes, pretrained=False)
    model.to(device)
    # torch.load('my_file.pt', map_location=lambda storage, loc: storage)
    if os.path.isfile(args.resume):
        logger.info("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage)
        best_prec1 = checkpoint['best_prec1']
        best_prec3 = checkpoint['best_prec3']
        model.load_state_dict(checkpoint['state_dict'])
        logger.info("About the model, the best valid prec1: {}, prec3: {}".format(best_prec1, best_prec3))
        logger.info("=> loaded models done!")
    else:
        logger.info("=> no checkpoint found at '{}'".format(args.resume))
    # load data
    logger.info("load test data....")
    list_frame = pd.read_csv(os.path.join(test_dir, 'list.csv'))
    list_frame['CATEGORY_ID0'] = ""
    list_frame['CATEGORY_ID1'] = ""
    list_frame['CATEGORY_ID2'] = ""

    test_dataset = ImageSceneTestData(categories_csv=os.path.join(test_dir, "categories.csv"),
                                  list_csv=os.path.join(test_dir, 'list.csv'),
                                  data_root=os.path.join(test_dir, "data"),
                                  transform=transforms.Compose([
                                       transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                   ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    with torch.no_grad():
        cur_id = 0
        for i, sample in enumerate(test_loader):
            input = Variable(sample['image']).to(device)
            # compute output(use top3)
            output = model(input)  # [batch, 20]
            # write predict result to file
            _, pred = output.topk(3, 1, sorted=True, largest=True)  # [batch, 3]
            batch_size = pred.size(0)
            pred = pred.to("cpu").numpy()
            # logger.info("type(pred): {}, shape(pred): {}".format(type(pred), pred.shape))
            # logger.info("pred[0][0]: {}".format(pred[0][0]))
            for idx in range(batch_size):
                list_frame.iloc[idx+cur_id, 1] = pred[idx][0]
                list_frame.iloc[idx+cur_id, 2] = pred[idx][1]
                list_frame.iloc[idx+cur_id, 3] = pred[idx][2]
            cur_id += batch_size
            logger.info("batch {} test data predict done!".format(i))
    list_frame.to_csv(os.path.join(test_dir, 'list_predict.csv'), index=False)
    logger.info("predict all data done!")


def run():
    """run the system"""
    args = parser.parse_args()
    logger = logging.getLogger("scene classification")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    if args.log_path:  # logging 不会自己创建目录，但是会自己创建文件
        log_dir = os.path.dirname(args.log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(args.log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    # 默认输出到console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info("Running with args: {}".format(args))
    # run
    if args.predict:
        predict()
    else:
        main()


def save_checkpoint(state, is_best, filename='result/res34/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'result/res34/model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """adjust the learning rate according to the training process"""
    lr = optimizer.param_groups[0]['lr']
    if epoch <= 20 and epoch % 10 == 0:
        lr = lr * 1.0
    elif epoch <= 40 and epoch % 5 == 0:
        lr = lr * 1.0
    elif epoch <= 50 and epoch % 5 == 0:
        lr = lr * 1.0
    elif epoch % 5 == 0:
        lr = lr * 1.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    run()
