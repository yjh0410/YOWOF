import torch
import torch.nn as nn
from .matcher import UniformMatcher
from utils.box_ops import box_iou, generalized_box_iou, box_cxcywh_to_xyxy
from utils.misc import Sigmoid_FocalLoss
from utils.vis_tools import vis_targets
from utils.distributed_utils import get_world_size, is_dist_avail_and_initialized



class Criterion(object):
    def __init__(self, 
                 cfg, 
                 device,
                 img_size=224,
                 alpha=0.25,
                 gamma=2.0,
                 loss_cls_weight=1.0, 
                 loss_reg_weight=1.0, 
                 num_classes=80,
                 multi_hot=False):
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.multi_hot = multi_hot
        # loss weight
        self.loss_cls_weight = loss_cls_weight
        self.loss_reg_weight = loss_reg_weight
        # matcher
        self.matcher = UniformMatcher(match_times=cfg['topk'])
        # loss
        if multi_hot:
            pass
        else:
            self.cls_loss = Sigmoid_FocalLoss(alpha=alpha, gamma=gamma, reduction='none')


    def prepare_targets(self, cls_pred_shape, targets, indices, src_idx, ignore_idx, pos_ignore_idx):
        if self.multi_hot:
            # [BM, C]
            gt_cls = torch.zeros(cls_pred_shape, device=self.device).float()
            gt_cls[ignore_idx, :] = -1
            # [BN, C]
            tgt_cls_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)]).float()
            tgt_cls_o[pos_ignore_idx, :] = -1
            gt_cls[src_idx] = tgt_cls_o.to(self.device)

            foreground_idxs = (torch.sum(gt_cls, dim=-1) >= 0)
            num_foreground = foreground_idxs.sum()

        else:
            # [BM,]
            gt_cls = torch.full(cls_pred_shape[:1],
                                self.num_classes,
                                dtype=torch.int64,
                                device=self.device)
            gt_cls[ignore_idx] = -1
            tgt_cls_o = torch.cat([t['labels'][J] for t, (_, J) in zip(targets, indices)])
            tgt_cls_o[pos_ignore_idx] = -1

            gt_cls[src_idx] = tgt_cls_o.to(self.device)

            foreground_idxs = (gt_cls >= 0) & (gt_cls != self.num_classes)
            num_foreground = foreground_idxs.sum()

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_foreground)
        num_foreground = torch.clamp(num_foreground / get_world_size(), min=1).item()

        return gt_cls, foreground_idxs, num_foreground


    def loss_labels(self, cls_pred, tgt_labels, num_foreground):
        # class loss
        loss_labels = self.cls_loss(cls_pred, tgt_labels)
        loss_labels = loss_labels.sum() / num_foreground

        return loss_labels


    def loss_bboxes(self, pred_box, tgt_boxes, num_boxes):
        # giou
        gious = generalized_box_iou(pred_box, tgt_boxes)  # [N, M]
        # giou loss
        loss_bboxes = 1. - torch.diag(gious)
        loss_bboxes = loss_bboxes.sum() / num_boxes

        return loss_bboxes


    def __call__(self, outputs, targets):
        """
            outputs['cls_preds']: List[Tensor] -> [[B, M, C], ...]
            outputs['box_preds']: List[Tensor] -> [[B, M, 4], ...]
            outputs['strides']: Int -> stride of the model output
            targets: List[List] -> [List[B, N, 6], 
                                    ...,
                                    List[B, N, 6]],
        """            
        box_pred = outputs['box_preds']
        cls_pred = outputs['cls_preds']
        anchor_boxes = outputs['anchors']
        batch_size = len(targets)
        targets = [{k: v.to(box_pred.device) for k, v in t.items()} for t in targets]

        # rescale tgt box
        rescale_tgt = []
        for tgt in targets:
            tgt['boxes'] = tgt['boxes'] * self.img_size
            rescale_tgt.append(tgt)
        targets = rescale_tgt
        
        # Matcher for this frame
        indices = self.matcher(
            pred_boxes = box_pred,
            anchor_boxes = anchor_boxes,
            targets = targets)

        # convert cxcywh to x1y1x2y2
        anchor_boxes = box_cxcywh_to_xyxy(anchor_boxes)

        # [M, 4] -> [1, M, 4] -> [B, M, 4]
        anchor_boxes = anchor_boxes[None].repeat(batch_size, 1, 1)

        ious = []
        pos_ious = []
        for batch_index in range(batch_size):
            src_idx, tgt_idx = indices[batch_index]
            # iou between predbox and tgt box
            iou, _ = box_iou(box_pred[batch_index, ...], 
                                targets[batch_index]['boxes'].clone())
            if iou.numel() == 0:
                max_iou = iou.new_full((iou.size(0),), 0)
            else:
                max_iou = iou.max(dim=1)[0]

            # iou between anchorbox and tgt box
            a_iou, _ = box_iou(anchor_boxes[batch_index], 
                                targets[batch_index]['boxes'].clone())
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
            [src + idx * anchor_boxes[0].shape[0] for idx, (src, _) in
            enumerate(indices)])

        # [B, M, 4] -> [BM, 4]
        cls_pred = cls_pred.view(-1, self.num_classes)

        # prepare targets
        (
            gt_cls,
            foreground_idxs,
            num_foreground
        ) = self.prepare_targets(
            cls_pred_shape=cls_pred.shape, 
            targets=targets, 
            indices=indices, 
            src_idx=src_idx, 
            ignore_idx=ignore_idx,
            pos_ignore_idx=pos_ignore_idx
            )

        # class loss
        valid_idxs = gt_cls >= 0
        tgt_labels = torch.zeros_like(cls_pred)
        tgt_labels[foreground_idxs, gt_cls[foreground_idxs]] = 1
        valid_cls_pred = cls_pred[valid_idxs]
        valid_tgt_labels = tgt_labels[valid_idxs]
        loss_labels = self.loss_labels(valid_cls_pred, valid_tgt_labels, num_foreground)

        # bbox loss
        tgt_boxes = torch.cat([t['boxes'][i]
                                    for t, (_, i) in zip(targets, indices)], dim=0).to(self.device)
        tgt_boxes = tgt_boxes[~pos_ignore_idx]
        matched_pred_box = box_pred.reshape(-1, 4)[src_idx[~pos_ignore_idx]]
        loss_bboxes = self.loss_bboxes(matched_pred_box, tgt_boxes, num_foreground)

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