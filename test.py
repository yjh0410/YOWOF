import argparse
import cv2
import os
import time
import numpy as np
import torch

from dataset.ucf24 import UCF24, UCF24_CLASSES
from dataset.jhmdb import JHMDB, JHMDB_CLASSES
from dataset.transforms import ValTransforms

from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import vis_detection

from config import build_dataset_config, build_model_config
from models.detector import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOWOF')

    # basic
    parser.add_argument('-size', '--img_size', default=320, type=int,
                        help='the size of input frame')
    parser.add_argument('--show', action='store_true', default=False,
                        help='show the visulization results.')
    parser.add_argument('--cuda', action='store_true', default=False, 
                        help='use cuda.')
    parser.add_argument('--save', action='store_true', default=False, 
                        help='save detection results.')
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.35, type=float,
                        help='threshold for visualization')

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



@torch.no_grad()
def inference(args, model, device, dataset, transform=None, class_names=None, class_colors=None):
    # path to save 
    save_path = os.path.join(
        args.save_folder, args.dataset, 
        args.version, 'video_clips')
    os.makedirs(save_path, exist_ok=True)

    # inference
    for index in range(len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        image_list, _ = dataset[index]

        orig_h, orig_w, _ = image_list[0].shape
        orig_size = [orig_w, orig_h]

        # prepare
        image_list, _ = transform(image_list)
        video_clip = torch.stack(image_list, dim=1)     # [3, T, H, W]
        video_clip = video_clip.unsqueeze(0).to(device) # [B, 3, T, H, W], B=1

        t0 = time.time()
        # inference
        scores, labels, bboxes = model(video_clip)
        print("inference time ", time.time() - t0, "s")
        
        # rescale
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame
        vis_results = vis_detection(
            frame=image_list[-1],
            scores=scores,
            labels=labels,
            bboxes=bboxes,
            vis_thresh=args.vis_thresh,
            class_names=class_names,
            class_colors=class_colors
            )

        if args.show:
            cv2.imshow('key-frame detection', vis_results)
            cv2.waitKey(0)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path,
            '{:0>5}.jpg'.format(index)), vis_results)
        

if __name__ == '__main__':
    args = parse_args()
    # cuda
    if args.cuda:
        print('use cuda')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # config
    d_cfg = build_dataset_config(args)
    m_cfg = build_model_config(args)

    # dataset
    if args.dataset == 'ucf24':
        num_classes = 24
        class_names = UCF24_CLASSES
        # dataset
        dataset = UCF24(
            cfg=d_cfg,
            img_size=args.img_size,
            len_clip=d_cfg['len_clip'],
            is_train=False,
            transform=None,
            debug=False
            )

    elif args.dataset == 'jhmdb':
        num_classes = 21
        class_names = JHMDB_CLASSES
        # dataset
        dataset = JHMDB(
            cfg=d_cfg,
            img_size=args.img_size,
            len_clip=d_cfg['len_clip'],
            is_train=False,
            transform=None,
            debug=False
            )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    np.random.seed(100)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # build model
    model = build_model(
        args=args,
        d_cfg=d_cfg,
        m_cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False
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
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std'],
        format=d_cfg['format']
        )

    # run
    inference(args=args,
              model=model,
              device=device,
              dataset=dataset,
              transform=transform,
              class_names=class_names,
              class_colors=class_colors)
