from __future__ import division

import os
import math
import time
import argparse
from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from utils import distributed_utils
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, build_dataset, build_dataloader, get_total_grad_norm
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from config import build_dataset_config, build_model_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOF')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--grad_clip_norm', type=float, default=-1.,
                        help='grad clip.')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')
    parser.add_argument('--eval_epoch', default=2, type=int, 
                        help='after eval epoch, the model is evaluated on val dataset.')
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='use tensorboard')

    # model
    parser.add_argument('-v', '--version', default='baseline', type=str,
                        help='build spatio-temporal action detector')
    parser.add_argument('--topk', default=40, type=int,
                        help='topk candidates for evaluation')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')

    # dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, jhmdb')
    
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
    print('World size: {}'.format(distributed_utils.get_world_size()))
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
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(d_cfg, m_cfg, args, is_train=True)

    # dataloader
    batch_size = d_cfg['batch_size'] * distributed_utils.get_world_size()
    dataloader = build_dataloader(args, dataset, batch_size, CollateFunc())

    # build model
    net = build_model(args=args, 
                      cfg=m_cfg,
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

    # optimizer
    base_lr = d_cfg['base_lr'] * batch_size
    optimizer = build_optimizer(model=model_without_ddp,
                                base_lr=base_lr,
                                bk_lr_ratio=d_cfg['bk_lr_ratio'],
                                name=m_cfg['optimizer'],
                                momentum=m_cfg['momentum'],
                                weight_decay=m_cfg['weight_decay'])

    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer, milestones=d_cfg['lr_epoch'])

    # warmup scheduler
    warmup_scheduler = build_warmup(name=d_cfg['warmup'],
                                    base_lr=base_lr,
                                    wp_iter=d_cfg['wp_iter'],
                                    warmup_factor=d_cfg['warmup_factor'])

    # training configuration
    max_epoch = d_cfg['max_epoch']
    epoch_size = len(dataloader)
    best_map = -1.
    warmup = True

    t0 = time.time()
    for epoch in range(max_epoch):
        if args.distributed:
            dataloader.batch_sampler.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (video_clips, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if ni < d_cfg['wp_iter'] and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == d_cfg['wp_iter'] and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, lr=base_lr, base_lr=base_lr)

            # to device
            video_clips = [video_clip.to(device) for video_clip in video_clips]

            # inference
            loss_dict = model(video_clips, targets=targets, vis_data=args.vis_data)
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
                log += '[size: {}]'.format(m_cfg['train_size'])

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


if __name__ == '__main__':
    train()
