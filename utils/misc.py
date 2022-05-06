import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import cv2

from dataset.ucf24 import UCF24
from dataset.jhmdb import JHMDB
from dataset.transforms import TrainTransforms, ValTransforms


def build_dataset(d_cfg, m_cfg, args, is_train=False):
    """
        d_cfg: dataset config
        m_cfg: model config
    """
    # transform
    trans_config = m_cfg['transforms']
    print('==============================')
    print('TrainTransforms: {}'.format(trans_config))
    train_transform = TrainTransforms(trans_config=trans_config,
                                      img_size=m_cfg['train_size'],
                                      pixel_mean=m_cfg['pixel_mean'],
                                      pixel_std=m_cfg['pixel_std'],
                                      format=m_cfg['format'])
    val_transform = ValTransforms(img_size=m_cfg['test_size'],
                                  pixel_mean=m_cfg['pixel_mean'],
                                  pixel_std=m_cfg['pixel_std'],
                                  format=m_cfg['format'])
    # dataset
    
    if args.dataset == 'ucf24':
        num_classes = 24
        # dataset
        dataset = UCF24(cfg=d_cfg,
                        img_size=m_cfg['train_size'],
                        len_clip=m_cfg['len_clip'],
                        is_train=is_train,
                        transform=train_transform,
                        debug=False)
        # evaluator
        evaluator = None

    elif args.dataset == 'jhmdb':
        num_classes = 21
        # dataset
        dataset = JHMDB(cfg=d_cfg,
                        img_size=m_cfg['train_size'],
                        len_clip=m_cfg['len_clip'],
                        is_train=is_train,
                        transform=train_transform,
                        debug=False)
        # evaluator
        evaluator = None
    
    else:
        print('unknow dataset !! Only support UCF24 and JHMDB !!')
        exit(0)

    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, batch_size, collate_fn=None):
    # distributed
    if args.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = torch.utils.data.RandomSampler(dataset)

    batch_sampler_train = torch.utils.data.BatchSampler(sampler, 
                                                        batch_size, 
                                                        drop_last=True)

    dataloader = torch.utils.data.DataLoader(dataset, 
                                             batch_sampler=batch_sampler_train,
                                             collate_fn=collate_fn, 
                                             num_workers=args.num_workers)
    
    return dataloader
    

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2.0, reduction='none'):
    p = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(input=logits, 
                                                    target=targets, 
                                                    reduction="none")
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    loss = ce_loss * ((1.0 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()

    elif reduction == "sum":
        loss = loss.sum()

    return loss


def get_total_grad_norm(parameters, norm_type=2):
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]),
                            norm_type)
    return total_norm


def load_weight(device, model, path_to_ckpt):
    checkpoint = torch.load(path_to_ckpt, map_location='cpu')
    # checkpoint state dict
    checkpoint_state_dict = checkpoint.pop("model")
    # model state dict
    model_state_dict = model.state_dict()
    # check
    for k in list(checkpoint_state_dict.keys()):
        if k in model_state_dict:
            shape_model = tuple(model_state_dict[k].shape)
            shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
            if shape_model != shape_checkpoint:
                checkpoint_state_dict.pop(k)
        else:
            checkpoint_state_dict.pop(k)
            print(k)

    model.load_state_dict(checkpoint_state_dict)
    model = model.to(device).eval()
    print('Finished loading model!')

    return model


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


class CollateFunc(object):
    def __call__(self, batch):
        batch_target_clips = []
        batch_video_clips = []

        len_clip = len(batch[0][0])
        for fid in range(len_clip):
            cur_fid_video_clip = []
            cur_fid_target_clip = []
            for sample in batch:
                cur_fid_video_clip.append(sample[0][fid])
                cur_fid_target_clip.append(sample[1][fid])
            cur_fid_video_clip = torch.stack(cur_fid_video_clip, dim=0)  # [B, C, H, W]
            batch_video_clips.append(cur_fid_video_clip)
            batch_target_clips.append(cur_fid_target_clip)

        # batch_clip_images: List[Tensor] -> [Tensor[B, C, H, W],
        #                                     ..., 
        #                                     Tensor[B, C, H, W]]
        # batch_clip_targets: List[List] -> [List[B, N, 6],
        #                                    ...,
        #                                    List[B, N, 6]]
        return batch_video_clips, batch_target_clips
