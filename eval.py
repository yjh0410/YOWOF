import argparse
import cv2
import os
import time
import numpy as np
import torch

from evaluator.jhmdb_evaluator import JHMDBEvaluator

from dataset.transforms import ValTransforms

from utils.misc import load_weight

from config import build_dataset_config, build_model_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOF')

    # basic
    parser.add_argument('-size', '--img_size', default=320, type=int,
                        help='the size of input frame')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('-mt', '--metrics', default=['frame_map', 'video_map'], type=str,
                        help='evaluation metrics')
    parser.add_argument('--save_dir', default='inference_results/',
                        type=str, help='Trained state_dict file path to open')

    # model
    parser.add_argument('-v', '--version', default='baseline', type=str,
                        help='build yowof')
    parser.add_argument('--weight', default='weight/',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc.')

    return parser.parse_args()


def jhmdb_eval(d_cfg, model, transform, save_dir, metrics):
    evaluator = JHMDBEvaluator(
        cfg=d_cfg,
        len_clip=model.len_clip,
        img_size=args.img_size,
        thresh=0.5,
        transform=transform,
        save_dir=save_dir)

    for metric in metrics:
        evaluator.metric = metric
        evaluator.evaluate(model)


if __name__ == '__main__':
    args = parse_args()
    # dataset
    if args.dataset == 'ucf24':
        num_classes = 24

    elif args.dataset == 'jhmdb':
        num_classes = 21
    
    else:
        print('unknow dataset.')
        exit(0)

    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)


    # build model
    model = build_model(
        args=args, 
        cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False,
        inference='stream'
        )

    # load trained weight
    model = load_weight(
        device=device,
        model=model, 
        path_to_ckpt=args.weight
        )

    # to eval
    model = model.to(device).eval()

    # transform
    transform = ValTransforms(
        img_size=args.img_size,
        pixel_mean=m_cfg['pixel_mean'],
        pixel_std=m_cfg['pixel_std'],
        format=m_cfg['format']
        )

    # path to save inference results
    save_path = os.path.join(args.save_dir, args.dataset)

    # run
    if args.dataset == 'jhmdb':
        jhmdb_eval(
            d_cfg=d_cfg,
            model=model,
            transform=transform,
            save_dir=save_path,
            metrics=args.metrics
            )
