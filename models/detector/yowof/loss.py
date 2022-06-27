import torch
import torch.nn as nn
from .matcher import UniformMatcher
from utils.box_ops import *
from utils.misc import sigmoid_focal_loss
from utils.vis_tools import vis_targets
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device, 
                 alpha=0.25,
                 gamma=2.0,
                 loss_cls_weight=1.0, 
                 loss_reg_weight=1.0, 
                 num_classes=80):
        self.cfg = cfg
        self.device = device
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        if cfg['matcher'] == 'uniform_matcher':
            self.matcher = UniformMatcher(match_times=cfg['topk'])


    def __call__(self, 
                 outputs, 
                 targets, 
                 video_clips=None, 
                 vis_data=False):
        """
            outputs['cls_preds']: List[Tensor] -> [[B, M, C], ...]
            outputs['box_preds']: List[Tensor] -> [[B, M, 4], ...]
            outputs['strides']: Int -> stride of the model output
            anchor_boxes: (Tensor) [M, 4]
            targets: List[List] -> [List[B, N, 6], 
                                    ...,
                                    List[B, N, 6]],
            video_clips: Lits[Tensor] -> [Tensor[B, C, H, W], 
                                          ..., 
                                          Tensor[B, C, H, W]]
        """
        if vis_data:
            # To DO: 
            # vis video clip and targets
            vis_targets(video_clips, targets)

        # target of key-frame
        kf_target = targets[-1]
            
        box_pred = outputs['box_preds']
        cls_pred = outputs['cls_preds']
        anchor_boxes = outputs['anchors']
        batch_size = len(kf_target)

        # reformat target
        kf_target = [{
            'boxes': t[:, :4].float(),  # [Ni, 4]
            'labels': t[:, 4].long(),   # [Ni,]
        } for t in kf_target]

        # copy pred data
        box_pred_copy = box_pred.detach().clone().cpu()
        anchor_boxes_copy = anchor_boxes.clone().cpu()

        # Matcher for this frame
        indices = self.matcher(
            pred_boxes = box_pred_copy,
            anchor_boxes = anchor_boxes_copy,
            targets = kf_target)

        # convert cxcywh to x1y1x2y2
        anchor_boxes_copy = box_cxcywh_to_xyxy(anchor_boxes_copy)

        # [M, 4] -> [1, M, 4] -> [B, M, 4]
        anchor_boxes_copy = anchor_boxes_copy[None].repeat(batch_size, 1, 1)

        ious = []
        pos_ious = []
        for batch_index in range(batch_size):
            src_idx, tgt_idx = indices[batch_index]
            # iou between predbox and tgt box
            iou, _ = box_iou(box_pred_copy[batch_index, ...], 
                                kf_target[batch_index]['boxes'].clone())
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]

            # iou between anchorbox and tgt box
            a_iou, _ = box_iou(anchor_boxes_copy[batch_index], 
                                kf_target[batch_index]['boxes'].clone())
            if a_iou.numel() == 0:
                pos_iou = a_iou.new_full((0,), 0)
            else:
                pos_iou = a_iou[src_idx, tgt_idx]
            ious.append(max_iou)
            pos_ious.append(pos_iou)

        ious = torch.cat(ious)
        ignore_idx = ious > self.cfg['igt']
        pos_ious = torch.cat(pos_ious)
        pos_ignore_idx = pos_ious < self.cfg['iou_t']

        src_idx = torch.cat(
            [src + idx * anchor_boxes_copy[0].shape[0] for idx, (src, _) in
            enumerate(indices)])
        # [BM,]
        cls_pred = cls_pred.view(-1, self.num_classes)
        gt_cls = torch.full(cls_pred.shape[:1],
                            self.num_classes,
                            dtype=torch.int64,
                            device=self.device)
        gt_cls[ignore_idx] = -1
        tgt_cls_o = torch.cat([t['labels'][J] for t, (_, J) in zip(kf_target, indices)])
        tgt_cls_o[pos_ignore_idx] = -1

        gt_cls[src_idx] = tgt_cls_o.to(self.device)

        foreground_idxs = (gt_cls >= 0) & (gt_cls != self.num_classes)
        num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        # class loss
        valid_idxs = gt_cls >= 0
        gt_cls_target = torch.zeros_like(cls_pred)
        gt_cls_target[foreground_idxs, gt_cls[foreground_idxs]] = 1
        key_frame_loss_labels = sigmoid_focal_loss(
                            cls_pred[valid_idxs], 
                            gt_cls_target[valid_idxs], 
                            self.alpha, 
                            self.gamma, 
                            reduction='none'
                            )
        loss_labels = key_frame_loss_labels.sum() / num_foreground

        # bbox loss
        tgt_boxes = torch.cat([t['boxes'][i]
                                    for t, (_, i) in zip(kf_target, indices)], dim=0).to(self.device)
        tgt_boxes = tgt_boxes[~pos_ignore_idx]
        matched_pred_box = box_pred.reshape(-1, 4)[src_idx[~pos_ignore_idx]]
        # giou
        gious = generalized_box_iou(matched_pred_box, tgt_boxes)  # [N, M]
        # giou loss
        key_frame_loss_bboxes = 1. - torch.diag(gious)
        loss_bboxes = key_frame_loss_bboxes.sum() / num_foreground

        # total loss
        losses = self.loss_cls_weight * loss_labels + \
                 self.loss_reg_weight * loss_bboxes

        loss_dict = dict(
                loss_labels = loss_labels,
                loss_bboxes = loss_bboxes,
                losses = losses
        )

        return loss_dict


if __name__ == "__main__":
    pass
