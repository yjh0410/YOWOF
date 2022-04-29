from __future__ import division
from cmath import e

import os
import argparse
import time
import random
from copy import deepcopy

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from dataset.voc import VOCDetection
from dataset.coco import COCODataset
from dataset.transforms import TrainTransforms, ValTransforms, BaseTransforms

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, get_total_grad_norm
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator

from config import build_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='Object Detection Benchmark')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size on single GPU')
    parser.add_argument('--schedule', type=str, default='1x',
                        help='training schedule.')
    parser.add_argument('-lr', '--base_lr', type=float, default=0.01,
                        help='base learning rate')
    parser.add_argument('-lr_bk', '--backbone_lr', type=float, default=0.01,
                        help='backbone learning rate')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=2, help='interval between evaluations')
    parser.add_argument('--grad_clip_norm', type=float, default=-1.,
                        help='grad clip.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')

    # model
    parser.add_argument('-v', '--version', default='yolof18', type=str,
                        help='build yolof')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # train trick
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='Mosaic augmentation')
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    cfg = build_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(cfg, args, device)

    # dataloader
    dataloader = build_dataloader(args, dataset, CollateFunc())

    # build model
    net = build_model(args=args, 
                      cfg=cfg,
                      device=device, 
                      num_classes=num_classes, 
                      trainable=True,
                      coco_pretrained=args.coco_pretrained)
    model = net
    model = model.to(device).train()

    # SyncBatchNorm
    if args.sybn and args.distributed:
        print('use SyncBatchNorm ...')
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # DDP
    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_copy = deepcopy(model_without_ddp)
        model_copy.trainable = False
        model_copy.eval()
        FLOPs_and_Params(model=model_copy, 
                         min_size=cfg['test_min_size'], 
                         max_size=cfg['test_max_size'], 
                         device=device)
        model_copy.trainable = True
        model_copy.train()
    if args.distributed:
        # wait for all processes to synchronize
        dist.barrier()

    # optimizer
    optimizer = build_optimizer(model=model_without_ddp,
                                base_lr=args.base_lr,
                                backbone_lr=args.backbone_lr,
                                name=cfg['optimizer'],
                                momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, 
                                                        milestones=cfg['epoch'][args.schedule]['lr_epoch'])

    # warmup scheduler
    warmup_scheduler = build_warmup(name=cfg['warmup'],
                                    base_lr=args.base_lr,
                                    wp_iter=cfg['wp_iter'],
                                    warmup_factor=cfg['warmup_factor'])

    # training configuration
    max_epoch = cfg['epoch'][args.schedule]['max_epoch']
    epoch_size = len(dataloader)
    best_map = -1.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(max_epoch):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (images, targets, masks) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if ni < cfg['wp_iter'] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == cfg['wp_iter'] and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, args.base_lr, args.base_lr)

            # to device
            images = images.to(device)
            masks = masks.to(device)

            # inference
            loss_dict = model(images, mask=masks, targets=targets)
            losses = loss_dict['losses']

            # reduce            
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(losses):
                print('loss is NAN !!')
                continue

            # Backward and Optimize
            losses.backward()
            if args.grad_clip_norm > 0.:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                total_norm = get_total_grad_norm(model.parameters())
            optimizer.step()
            optimizer.zero_grad()

            # display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                cur_lr_dict = {'lr': cur_lr[0], 'lr_bk': cur_lr[1]}
                log = dict(
                    lr=round(cur_lr_dict['lr'], 6),
                    lr_bk=round(cur_lr_dict['lr_bk'], 6)
                )
                # basic infor
                log =  '[Epoch: {}/{}]'.format(epoch+1, max_epoch)
                log += '[Iter: {}/{}]'.format(iter_i, epoch_size)
                log += '[lr: {:.6f}][lr_bk: {:.6f}]'.format(cur_lr_dict['lr'], cur_lr_dict['lr_bk'])
                # loss infor
                for k in loss_dict_reduced.keys():
                    log += '[{}: {:.2f}]'.format(k, loss_dict[k])

                # other infor
                log += '[time: {:.2f}]'.format(t1 - t0)
                log += '[gnorm: {:.2f}]'.format(total_norm)
                log += '[size: [{}, {}]]'.format(cfg['train_min_size'], cfg['train_max_size'])

                # print log infor
                print(log, flush=True)
                
                t0 = time.time()

        lr_scheduler.step()
        
        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            # check evaluator
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print('No evaluator ... save model and go on training.')
                    print('Saving state, epoch: {}'.format(epoch + 1))
                    weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
                    checkpoint_path = os.path.join(path_to_save, weight_name)
                    torch.save({'model': model_without_ddp.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'args': args}, 
                                checkpoint_path)                      
                    
                else:
                    print('eval ...')
                    # set eval mode
                    model_without_ddp.trainable = False
                    model_without_ddp.eval()

                    # evaluate
                    evaluator.evaluate(model_without_ddp)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map*100)
                        checkpoint_path = os.path.join(path_to_save, weight_name)
                        torch.save({'model': model_without_ddp.state_dict(),
                                    'optimizer': optimizer.state_dict(),
                                    'lr_scheduler': lr_scheduler.state_dict(),
                                    'epoch': epoch,
                                    'args': args}, 
                                    checkpoint_path)                      

                    # set train mode.
                    model_without_ddp.trainable = True
                    model_without_ddp.train()
        
            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()

        # close mosaic augmentation
        if args.mosaic and max_epoch - epoch == 5:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False


def build_dataset(cfg, args, device):
    # transform
    trans_config = cfg['transforms'][args.schedule]
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    train_transform = TrainTransforms(trans_config=trans_config,
                                      min_size=cfg['train_min_size'],
                                      max_size=cfg['train_max_size'],
                                      random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                      pixel_mean=cfg['pixel_mean'],
                                      pixel_std=cfg['pixel_std'],
                                      format=cfg['format'])
    val_transform = ValTransforms(min_size=cfg['test_min_size'],
                                  max_size=cfg['test_max_size'],
                                  pixel_mean=cfg['pixel_mean'],
                                  pixel_std=cfg['pixel_std'],
                                  format=cfg['format'],
                                  padding=cfg['val_padding'])
    color_augment = BaseTransforms(min_size=cfg['train_min_size'],
                                   max_size=cfg['train_max_size'],
                                   random_size=cfg['epoch'][args.schedule]['multi_scale'],
                                   pixel_mean=cfg['pixel_mean'],
                                   pixel_std=cfg['pixel_std'],
                                   format=cfg['format'])
    # dataset
    
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        # dataset
        dataset = VOCDetection(img_size=cfg['train_max_size'],
                               data_dir=data_dir, 
                               transform=train_transform,
                               color_augment=color_augment,
                               mosaic=args.mosaic)
        # evaluator
        evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                    device=device,
                                    transform=val_transform)

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        # dataset
        dataset = COCODataset(img_size=cfg['train_max_size'],
                              data_dir=data_dir,
                              image_set='train2017',
                              transform=train_transform,
                              color_augment=color_augment,
                              mosaic=args.mosaic)
        # evaluator
        evaluator = COCOAPIEvaluator(data_dir=data_dir,
                                     device=device,
                                     transform=val_transform)
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                        args.batch_size, 
                                                        drop_last=True)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_sampler=batch_sampler_train,
                                             collate_fn=collate_fn, 
                                             num_workers=args.num_workers)
    
    return dataloader
    

if __name__ == '__main__':
    train()
