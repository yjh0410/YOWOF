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
    parser.add_argument('-vs', '--visual_threshold', default=0.35, type=float,
                        help='Final confidence threshold')
    parser.add_argument('-inf', '--inference', default='clip', type=str,
                        help='clip: infer with video clip; stream: infer with a video stream.')

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


def rescale_bboxes(bboxes, orig_size):
    orig_w, orig_h = orig_size[0], orig_size[1]
    rescale_bboxes = bboxes * orig_size    
    rescale_bboxes[..., [0, 2]] = np.clip(
        rescale_bboxes[..., [0, 2]], a_min=0., a_max=orig_w
        )
    rescale_bboxes[..., [1, 3]] = np.clip(
        rescale_bboxes[..., [1, 3]], a_min=0., a_max=orig_h
        )
    
    return rescale_bboxes


def rescale_bboxes_list(bboxes_list, orig_size):
    orig_w, orig_h = orig_size[0], orig_size[1]
    rescale_bboxes_list = []
    for bboxes in bboxes_list:
        rescale_bboxes = bboxes * orig_size    
        rescale_bboxes[..., [0, 2]] = np.clip(
            rescale_bboxes[..., [0, 2]], a_min=0., a_max=orig_w
            )
        rescale_bboxes[..., [1, 3]] = np.clip(
            rescale_bboxes[..., [1, 3]], a_min=0., a_max=orig_h
            )
        rescale_bboxes_list.append(rescale_bboxes)
    
    return rescale_bboxes_list


def inference_with_video_stream(args, net, device, transform=None, class_names=None):
    # inference
    num_videos = dataset.num_videos
    for index in range(num_videos):
        print('Testing video {:d}/{:d}....'.format(index+1, num_videos))
        video_name = dataset.load_video(index)
        video_path = os.path.join(dataset.image_path, video_name)
        num_frames = len(os.listdir(video_path))

        initialization = True
        frame_count = 0
        init_video_clip = []
        scores_list = []
        labels_list = []
        bboxes_list = []
        for fid in range(1, num_frames + 1):
            image_file = os.path.join(video_path, '{:0>5}.jpg'.format(fid))
            cur_frame = cv2.imread(image_file)
            orig_h, orig_w = cur_frame.shape[:2]
            orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])
            frame_count += 1

            if initialization:
                if frame_count < net.len_clip:
                    init_video_clip.append(cur_frame)
                    continue
                else:
                    xs, _ = transform(init_video_clip)
                    xs = [x.unsqueeze(0).to(device) for x in xs] 
                    init_scores, init_labels, init_bboxes = net(xs)

                    # rescale
                    init_bboxes = rescale_bboxes_list(init_bboxes, orig_size)

                    scores_list.extend(init_scores)
                    labels_list.extend(init_labels)
                    bboxes_list.extend(init_bboxes)
                    initialization = False
                    
            else:
                xs, _ = transform([cur_frame])
                xs = [x.unsqueeze(0).to(device) for x in xs] 

                t0 = time.time()
                cur_score, cur_label, cur_bboxes = net(xs[0])
                t1 = time.time()
                
                # rescale
                cur_bboxes = rescale_bboxes(cur_bboxes, orig_size)

                scores_list.append(cur_score)
                labels_list.append(cur_label)
                bboxes_list.append(cur_bboxes)
                # vis current detection results
                vis_results = vis_video_frame(
                    cur_frame, cur_score, cur_label, cur_bboxes, 
                    args.vis_threshold, class_names
                )


def inference_with_video_clip(args, net, device, dataset, transform=None, class_names=None):
    # inference
    for index in range(len(dataset)):
        print('Testing video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        video_clip, _ = dataset[index]

        orig_h, orig_w, _ = video_clip[0].shape
        orig_size = np.array([[orig_w, orig_h, orig_w, orig_h]])

        # prepare
        xs, _ = transform(video_clip)
        xs = [x.unsqueeze(0).to(device) for x in xs] 

        t0 = time.time()
        # inference
        scores_list, labels_list, bboxes_list = net(xs)
        print("detection time used ", time.time() - t0, "s")
        
        # rescale
        bboxes_list = rescale_bboxes_list(bboxes_list, orig_size)

        # vis detection
        vis_results = vis_video_clip(
            video_clip, scores_list, labels_list, rescale_bboxes_list, 
            args.vis_threshold, class_names
        )


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
        dataset = UCF24(cfg=d_cfg,
                        img_size=m_cfg['train_size'],
                        len_clip=m_cfg['len_clip'],
                        is_train=False,
                        transform=None,
                        debug=False)

    elif args.dataset == 'jhmdb':
        num_classes = 24
        class_names = UCF24_CLASSES
        # dataset
        dataset = JHMDB(cfg=d_cfg,
                        img_size=m_cfg['train_size'],
                        len_clip=m_cfg['len_clip'],
                        is_train=False,
                        transform=None,
                        debug=False)
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    np.random.seed(0)
    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # build model
    model = build_model(args=args, 
                        cfg=m_cfg,
                        device=device, 
                        num_classes=num_classes, 
                        trainable=False)

    # load trained weight
    model = load_weight(device=device, 
                        model=model, 
                        path_to_ckpt=args.weight)

    # transform
    transform = ValTransforms(img_size=args.img_size,
                              pixel_mean=m_cfg['pixel_mean'],
                              pixel_std=m_cfg['pixel_std'],
                              format=m_cfg['format'])

    # run
    test(args=args,
        net=model, 
        device=device, 
        dataset=dataset,
        transform=transform,
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        show=args.show,
        dataset_name=args.dataset)
