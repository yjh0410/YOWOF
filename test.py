import argparse
import cv2
import os
import time
import numpy as np
import torch

from dataset.ucf_jhmdb import UCF_JHMDB_Dataset
from dataset.ava import AVA_Dataset
from dataset.ava_pose import AVA_Pose_Dataset
from dataset.transforms import BaseTransform

from utils.misc import load_weight
from utils.box_ops import rescale_bboxes
from utils.vis_tools import convert_tensor_to_cv2img, vis_detection

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
    parser.add_argument('-sid', '--start_index', default=0, type=int,
                        help='start index to test.')

    # model
    parser.add_argument('-v', '--version', default='yowof-r18', type=str,
                        help='build yowof')
    parser.add_argument('--weight', default=None,
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--topk', default=40, type=int,
                        help='NMS threshold')
    parser.add_argument('-inf', '--inf_mode', default='clip', type=str, choices=['clip', 'stream', 'semi_stream'],
                        help='inference mode')

    # dataset
    parser.add_argument('-d', '--dataset', default='ucf24',
                        help='ucf24, jhmdb21, ava_v2.2.')

    return parser.parse_args()


@torch.no_grad()
def inference_ucf_jhmdb_stream(args, d_cfg, model, device, dataset, class_names=None, class_colors=None):
    # path to save 
    if args.save:
        save_path = os.path.join(
            args.save_folder, args.dataset, 
            args.version, 'video_stream')
        os.makedirs(save_path, exist_ok=True)

    # initalize model
    model.initialization = True

    # inference
    prev_frame_id = ''
    for index in range(args.start_index, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        frame_id, video_clip, target = dataset[index]
        orig_size = target['orig_size'].tolist()  # width, height

        # ex: frame_id = Basketball_v_Basketball_g01_c01_00048.txt
        if index == 0:
            prev_frame_id = frame_id[:-10]
            model.initialization = True

        if frame_id[:-10] != prev_frame_id:
            # a new video
            print(prev_frame_id)
            prev_frame_id = frame_id[:-10]
            model.initialization = True

        # prepare
        video_clip = video_clip.unsqueeze(0).to(device) # [B, T, 3, H, W], B=1

        t0 = time.time()
        # inference
        scores, labels, bboxes = model(video_clip)
        print("inference time ", time.time() - t0, "s")
        
        # rescale
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame
        key_frame_tensor = video_clip[0, -1, :, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])

        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)

        vis_results = vis_detection(
            frame=key_frame,
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
        

@torch.no_grad()
def inference_ucf_jhmdb_clip(args, d_cfg, model, device, dataset, class_names=None, class_colors=None):
    # path to save 
    if args.save:
        save_path = os.path.join(
            args.save_folder, args.dataset, 
            args.version, 'video_clips')
        os.makedirs(save_path, exist_ok=True)

    # inference
    for index in range(args.start_index, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        frame_id, video_clip, target = dataset[index]
        orig_size = target['orig_size'].tolist()  # width, height

        # prepare
        video_clip = video_clip.unsqueeze(0).to(device) # [B, T, 3, H, W], B=1

        t0 = time.time()
        # inference
        batch_outputs = model(video_clip)
        print("inference time ", time.time() - t0, "s")
        
        scores, labels, bboxes = batch_outputs[0]
        # rescale
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame
        key_frame_tensor = video_clip[0, -1, :, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])

        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)

        vis_results = vis_detection(
            frame=key_frame,
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
        

@torch.no_grad()
def inference_ucf_jhmdb_semi_stream(args, d_cfg, model, device, dataset, class_names=None, class_colors=None):
    # path to save 
    if args.save:
        save_path = os.path.join(
            args.save_folder, args.dataset, 
            args.version, 'video_clips')
        os.makedirs(save_path, exist_ok=True)

    # initalize model
    model.initialization = True

    # inference
    for index in range(args.start_index, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        frame_id, video_clip, target = dataset[index]
        orig_size = target['orig_size'].tolist()  # width, height

        # prepare
        video_clip = video_clip.unsqueeze(0).to(device) # [B, T, 3, H, W], B=1

        t0 = time.time()
        # inference
        scores, labels, bboxes = model(video_clip)
        print("inference time ", time.time() - t0, "s")
        
        # rescale
        bboxes = rescale_bboxes(bboxes, orig_size)

        # vis results of key-frame
        key_frame_tensor = video_clip[0, -1, :, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])

        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)

        vis_results = vis_detection(
            frame=key_frame,
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
        

@torch.no_grad()
def inference_ava_stream(args, d_cfg, model, device, dataset, class_names=None):
    # path to save 
    if args.save:
        save_path = os.path.join(
            args.save_folder, args.dataset, 
            args.version, 'video_stream')
        os.makedirs(save_path, exist_ok=True)

    # initalize model
    model.initialization = True

    # inference
    prev_video_id = ''
    prev_video_sec = ''
    for index in range(args.start_index, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        key_frame_info, video_clip, target = dataset[index]
        
        orig_size = target['orig_size'].tolist()  # width, height

        # ex: video_id: 1204, sec: 900
        if index == 0:
            prev_video_id = key_frame_info[0]
            prev_video_sec = key_frame_info[1]
            model.initialization = True

        if key_frame_info[0] != prev_video_id:
            # a new video
            prev_video_id = key_frame_info[0]
            prev_video_sec = key_frame_info[1]
            model.initialization = True

        # prepare
        video_clip = video_clip.unsqueeze(0).to(device) # [B, T, 3, H, W], B=1

        t0 = time.time()
        # inference
        bboxes = model(video_clip)
        print("inference time ", time.time() - t0, "s")
        
        # vis results of key-frame
        key_frame_tensor = video_clip[0, -1, :, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])
        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)

        # visualize detection results
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            cls_out = bbox[4:]
        
            # rescale bbox
            x1, x2 = int(x1 * orig_size[0]), int(x2 * orig_size[0])
            y1, y2 = int(y1 * orig_size[1]), int(y2 * orig_size[1])

            cls_scores = np.array(cls_out)
            indices = np.where(cls_scores > args.vis_thresh)
            scores = cls_scores[indices]
            indices = list(indices[0])
            scores = list(scores)

            cv2.rectangle(key_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if len(scores) > 0:
                blk   = np.zeros(key_frame.shape, np.uint8)
                font  = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text  = []
                text_size = []
                # scores, indices  = [list(a) for a in zip(*sorted(zip(scores,indices), reverse=True))] # if you want, you can sort according to confidence level
                for _, cls_ind in enumerate(indices):
                    text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                    text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                    coord.append((x1+3, y1+7+10*_))
                    cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                key_frame = cv2.addWeighted(key_frame, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(key_frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)
        
        if args.show:
            cv2.imshow('key-frame detection', key_frame)
            cv2.waitKey(0)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path,
            '{:0>5}.jpg'.format(index)), key_frame)
        

@torch.no_grad()
def inference_ava_clip(args, d_cfg, model, device, dataset, class_names=None):
    # path to save 
    if args.save:
        save_path = os.path.join(
            args.save_folder, args.dataset, 
            args.version, 'video_clips')
        os.makedirs(save_path, exist_ok=True)

    # inference
    for index in range(args.start_index, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        key_frame_info, video_clip, target = dataset[index]
        
        orig_size = target['orig_size'].tolist()  # width, height

        # prepare
        video_clip = video_clip.unsqueeze(0).to(device) # [B, T, 3, H, W], B=1

        t0 = time.time()
        # inference
        batch_outputs = model(video_clip)
        print("inference time ", time.time() - t0, "s")
        
        out_bboxes = batch_outputs[0]
        
        # vis results of key-frame
        key_frame_tensor = video_clip[0, -1, :, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])
        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)

        # visualize detection results
        for bbox in out_bboxes:
            x1, y1, x2, y2 = bbox[:4]
            cls_out = bbox[4:]
        
            # rescale bbox
            x1, x2 = int(x1 * orig_size[0]), int(x2 * orig_size[0])
            y1, y2 = int(y1 * orig_size[1]), int(y2 * orig_size[1])

            cls_scores = np.array(cls_out)
            indices = np.where(cls_scores > args.vis_thresh)
            scores = cls_scores[indices]
            indices = list(indices[0])
            scores = list(scores)

            cv2.rectangle(key_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if len(scores) > 0:
                blk   = np.zeros(key_frame.shape, np.uint8)
                font  = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text  = []
                text_size = []
                # scores, indices  = [list(a) for a in zip(*sorted(zip(scores,indices), reverse=True))] # if you want, you can sort according to confidence level
                for _, cls_ind in enumerate(indices):
                    text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                    text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                    coord.append((x1+3, y1+7+10*_))
                    cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                key_frame = cv2.addWeighted(key_frame, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(key_frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)
        
        if args.show:
            cv2.imshow('key-frame detection', key_frame)
            cv2.waitKey(0)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path,
            '{:0>5}.jpg'.format(index)), key_frame)
        

@torch.no_grad()
def inference_ava_semi_stream(args, d_cfg, model, device, dataset, class_names=None):
    # path to save 
    if args.save:
        save_path = os.path.join(
            args.save_folder, args.dataset, 
            args.version, 'video_clips')
        os.makedirs(save_path, exist_ok=True)

    # initalize model
    model.initialization = True

    # inference
    for index in range(args.start_index, len(dataset)):
        print('Video clip {:d}/{:d}....'.format(index+1, len(dataset)))
        key_frame_info, video_clip, target = dataset[index]
        
        orig_size = target['orig_size'].tolist()  # width, height

        # prepare
        video_clip = video_clip.unsqueeze(0).to(device) # [B, T, 3, H, W], B=1

        t0 = time.time()
        # inference
        out_bboxes = model(video_clip)
        print("inference time ", time.time() - t0, "s")
                
        # vis results of key-frame
        key_frame_tensor = video_clip[0, -1, :, :, :]
        key_frame = convert_tensor_to_cv2img(key_frame_tensor, d_cfg['pixel_mean'], d_cfg['pixel_std'])
        # resize key_frame to orig size
        key_frame = cv2.resize(key_frame, orig_size)

        # visualize detection results
        for bbox in out_bboxes:
            x1, y1, x2, y2 = bbox[:4]
            cls_out = bbox[4:]
        
            # rescale bbox
            x1, x2 = int(x1 * orig_size[0]), int(x2 * orig_size[0])
            y1, y2 = int(y1 * orig_size[1]), int(y2 * orig_size[1])

            cls_scores = np.array(cls_out)
            indices = np.where(cls_scores > args.vis_thresh)
            scores = cls_scores[indices]
            indices = list(indices[0])
            scores = list(scores)

            cv2.rectangle(key_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            if len(scores) > 0:
                blk   = np.zeros(key_frame.shape, np.uint8)
                font  = cv2.FONT_HERSHEY_SIMPLEX
                coord = []
                text  = []
                text_size = []
                # scores, indices  = [list(a) for a in zip(*sorted(zip(scores,indices), reverse=True))] # if you want, you can sort according to confidence level
                for _, cls_ind in enumerate(indices):
                    text.append("[{:.2f}] ".format(scores[_]) + str(class_names[cls_ind]))
                    text_size.append(cv2.getTextSize(text[-1], font, fontScale=0.25, thickness=1)[0])
                    coord.append((x1+3, y1+7+10*_))
                    cv2.rectangle(blk, (coord[-1][0]-1, coord[-1][1]-6), (coord[-1][0]+text_size[-1][0]+1, coord[-1][1]+text_size[-1][1]-4), (0, 255, 0), cv2.FILLED)
                key_frame = cv2.addWeighted(key_frame, 1.0, blk, 0.25, 1)
                for t in range(len(text)):
                    cv2.putText(key_frame, text[t], coord[t], font, 0.25, (0, 0, 0), 1)
        
        if args.show:
            cv2.imshow('key-frame detection', key_frame)
            cv2.waitKey(0)

        if args.save:
            # save result
            cv2.imwrite(os.path.join(save_path,
            '{:0>5}.jpg'.format(index)), key_frame)
        

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

    # transform
    basetransform = BaseTransform(
        img_size=d_cfg['test_size'],
        pixel_mean=d_cfg['pixel_mean'],
        pixel_std=d_cfg['pixel_std'])

    # dataset
    if args.dataset in ['ucf24', 'jhmdb21']:
        dataset = UCF_JHMDB_Dataset(
            data_root=d_cfg['data_root'],
            dataset=args.dataset,
            img_size=d_cfg['test_size'],
            transform=basetransform,
            is_train=False,
            len_clip=d_cfg['len_clip'],
            sampling_rate=d_cfg['sampling_rate']
            )
        class_names = d_cfg['label_map']
        num_classes = dataset.num_classes

    elif args.dataset == 'ava_v2.2':
        dataset = AVA_Dataset(
            cfg=d_cfg,
            is_train=False,
            img_size=d_cfg['test_size'],
            transform=basetransform,
            len_clip=d_cfg['len_clip'],
            sampling_rate=d_cfg['sampling_rate']
        )
        class_names = d_cfg['label_map']
        num_classes = dataset.num_classes

    elif args.dataset == 'ava_pose':
        dataset = AVA_Pose_Dataset(
            cfg=d_cfg,
            is_train=False,
            img_size=d_cfg['test_size'],
            transform=basetransform,
            len_clip=d_cfg['len_clip'],
            sampling_rate=d_cfg['sampling_rate']
        )
        class_names = d_cfg['label_map']
        num_classes = dataset.num_classes

    else:
        print('unknow dataset !! Only support ucf24 & jhmdb21 & ava_v2.2 and coco !!')
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
    model = load_weight(model=model, path_to_ckpt=args.weight)

    # to eval
    model = model.to(device).eval()

    # set inference mode
    model.set_inference_mode(mode=args.inf_mode)

    # run
    if args.dataset in ['ucf24', 'jhmdb21']:
        if args.inf_mode == 'stream':
            inference_ucf_jhmdb_stream(
                args=args,
                d_cfg=d_cfg,
                model=model,
                device=device,
                dataset=dataset,
                class_names=class_names,
                class_colors=class_colors
                )
        elif args.inf_mode == 'clip':
            inference_ucf_jhmdb_clip(
                args=args,
                d_cfg=d_cfg,
                model=model,
                device=device,
                dataset=dataset,
                class_names=class_names,
                class_colors=class_colors
                )
        elif args.inf_mode == 'semi_stream':
            inference_ucf_jhmdb_semi_stream(
                args=args,
                d_cfg=d_cfg,
                model=model,
                device=device,
                dataset=dataset,
                class_names=class_names,
                class_colors=class_colors
                )

    elif args.dataset in ['ava_v2.2']:
        if args.inf_mode == 'stream':
            inference_ava_stream(
                d_cfg=d_cfg,
                args=args,
                model=model,
                device=device,
                dataset=dataset,
                class_names=class_names
                )
        elif args.inf_mode == 'clip':
            inference_ava_clip(
                d_cfg=d_cfg,
                args=args,
                model=model,
                device=device,
                dataset=dataset,
                class_names=class_names
                )
        elif args.inf_mode == 'semi_stream':
            inference_ava_semi_stream(
                d_cfg=d_cfg,
                args=args,
                model=model,
                device=device,
                dataset=dataset,
                class_names=class_names
                )
