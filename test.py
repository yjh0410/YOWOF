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
from utils.box_ops import rescale_bboxes, rescale_bboxes_list
from utils.vis_tools import vis_video_clip, vis_video_frame

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
    parser.add_argument('--save_folder', default='det_results/', type=str,
                        help='Dir to save results')
    parser.add_argument('-vs', '--vis_thresh', default=0.35, type=float,
                        help='threshold for visualization')
    parser.add_argument('-inf', '--inference', default='clip', type=str,
                        help='clip: infer with video clip; stream: infer with a video stream.')
    parser.add_argument('-if', '--img_format', default='jpg', type=str,
                        help='format of image.')

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
def inference_with_video_stream(args, model, device, transform=None, class_names=None, class_colors=None):
    # path to save 
    save_path = os.path.join(
        args.save_folder, args.version, 
        args.dataset, 'video_streams')
    os.makedirs(save_path, exist_ok=True)

    # inference
    num_videos = dataset.num_videos
    for index in range(num_videos):
        # load a video
        video_name = dataset.load_video(index)
        num_frames = dataset.nframes[video_name]
        print('Video {:d}/{:d}: {}'.format(index+1, num_videos, video_name))
        video_path = os.path.join(dataset.image_path, video_name)
        
        # prepare
        model.initialization = True
        frame_index = 0
        init_video_clip = []

        # inference with video stream
        for fid in range(1, num_frames + 1):
            image_file = os.path.join(video_path, '{:0>5}.{}'.format(fid, args.img_format))
            cur_frame = cv2.imread(image_file)
            
            assert cur_frame is not None

            orig_h, orig_w = cur_frame.shape[:2]
            orig_size = [orig_w, orig_h]
            # update frame index
            frame_index += 1

            if model.initialization:
                if frame_index < model.len_clip:
                    init_video_clip.append(cur_frame)
                else:
                    init_video_clip.append(cur_frame)
                    # preprocess
                    xs, _ = transform(init_video_clip)

                    # to device
                    xs = [x.unsqueeze(0).to(device) for x in xs] 

                    # inference with an init video clip
                    init_scores, init_labels, init_bboxes = model(xs)

                    # rescale
                    init_bboxes = rescale_bboxes(init_bboxes, orig_size)

                    # vis init detection
                    vis_results = vis_video_frame(
                        frame=init_video_clip[-1],
                        scores=init_scores,
                        labels=init_labels,
                        bboxes=init_bboxes,
                        vis_thresh=args.vis_thresh,
                        class_names=class_names,
                        class_colors=class_colors
                        )

                    # save result
                    cv2.imwrite(os.path.join(save_path, video_name,
                        'init_video_clip.jpg'), vis_results)
                    
                    model.initialization = False
                    del init_video_clip

            else:
                # preprocess
                xs, _ = transform([cur_frame])

                # to device
                xs = [x.unsqueeze(0).to(device) for x in xs] 

                # inference with the current frame
                t0 = time.time()
                cur_score, cur_label, cur_bboxes = model(xs[0])
                t1 = time.time()

                print('inference time: {:.3f}'.format(t1 - t0))
                
                # rescale
                cur_bboxes = rescale_bboxes(cur_bboxes, orig_size)

                # vis current detection results
                vis_results = vis_video_frame(
                    cur_frame, cur_score, cur_label, cur_bboxes, 
                    args.vis_thresh, class_names, class_colors
                )

                cv2.imshow('current frame', vis_results)
                cv2.waitKey(0)

                # save result
                cv2.imwrite(os.path.join(save_path, 
                    '{:0>5}.jpg'.format(index)), vis_results)


@torch.no_grad()
def inference_with_video_clip(args, model, device, dataset, transform=None, class_names=None, class_colors=None):
    # path to save 
    save_path = os.path.join(
        args.save_folder, args.dataset, 
        args.version, 'video_clips')
    os.makedirs(save_path, exist_ok=True)

    # inference
    for index in range(len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        video_clip, _ = dataset[index]

        orig_h, orig_w, _ = video_clip[0].shape
        orig_size = [orig_w, orig_h]

        # prepare
        xs, _ = transform(video_clip)
        xs = [x.unsqueeze(0).to(device) for x in xs] 

        t0 = time.time()
        # inference
        scores, labels, bboxes = model(xs)
        print("inference time ", time.time() - t0, "s")
        
        # rescale
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame
        vis_results = vis_video_frame(
            frame=video_clip[-1],
            scores=scores,
            labels=labels,
            bboxes=bboxes,
            vis_thresh=args.vis_thresh,
            class_names=class_names,
            class_colors=class_colors
            )

        cv2.imshow('key-frame frame', vis_results)
        cv2.waitKey(0)

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
            len_clip=m_cfg['len_clip'],
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
            len_clip=m_cfg['len_clip'],
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
        cfg=m_cfg,
        device=device, 
        num_classes=num_classes, 
        trainable=False,
        inference=args.inference
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

    # run
    if args.inference == 'clip':
        inference_with_video_clip(
            args=args, model=model, device=device, dataset=dataset, transform=transform, 
            class_names=class_names, class_colors=class_colors)
    elif args.inference == 'stream':
        inference_with_video_stream(
            args=args, model=model, device=device, transform=transform, 
            class_names=class_names, class_colors=class_colors)
