# YOWOF

import numpy as np
import math
import torch
import torch.nn as nn
from ..backbone import build_backbone
from ..neck import build_neck
from ..head.decoupled_head import DecoupledHead


DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)


class YOWOF(nn.Module):
    def __init__(self, 
                 cfg,
                 device, 
                 img_size = 320,
                 num_classes = 20, 
                 conf_thresh = 0.05,
                 nms_thresh = 0.6,
                 topk = 1000,
                 trainable = False):
        super(YOWOF, self).__init__()
        self.cfg = cfg
        self.device = device
        self.img_size = img_size
        self.stride = cfg['stride']
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.num_anchors = len(cfg['anchor_size'])
        self.anchor_size = torch.as_tensor(cfg['anchor_size'])

        # backbone
        self.backbone, bk_dim = build_backbone(model_name=cfg['backbone'], 
                                               pretrained=trainable,
                                               norm_type=cfg['norm_type'])

        # neck
        self.neck = build_neck(cfg=cfg, 
                               in_dim=bk_dim, 
                               out_dim=cfg['head_dim'])
                                     
        # head
        self.head = DecoupledHead(head=cfg['head'],
                                  head_dim=cfg['head_dim'],
                                  kernel_size=3,
                                  padding=1,
                                  num_classes=num_classes,
                                  trainable=trainable,
                                  num_anchors=self.num_anchors,
                                  act_type=cfg['act_type'])

        # create grid
        self.anchor_boxes = self.generate_anchors(img_size)
        

    def generate_anchors(self, img_size):
        # generate grid cells
        fmp_h = fmp_w = img_size // self.stride
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
        anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1).to(self.device)
        anchor_xy *= self.stride

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1).to(self.device)

        # [HW, KA, 4] -> [N, 4]
        anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4)

        return anchor_boxes
        

    def reset_anchors(self, img_size):
        self.img_size = img_size
        self.anchor_boxes = self.generate_anchors(img_size)

        
    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[tensor]) [1, M, 4]
            pred_reg: (List[tensor])     [B, M, 4]
        """
        # dxdy -> cxcy
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
        pred_ctr_offset = torch.clamp(pred_ctr_offset,
                                      max=self.cfg['ctr_clamp'],
                                      min=-self.cfg['ctr_clamp'])
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset

        # dwdh -> bwbh
        pred_dwdh = pred_reg[..., 2:]
        pred_dwdh = torch.clamp(pred_dwdh, 
                                max=DEFAULT_SCALE_CLAMP)
        pred_wh = anchor_boxes[..., 2:] * pred_dwdh.exp()

        # convert [x, y, w, h] -> [x1, y1, x2, y2]
        pred_x1y1 = pred_ctr_xy - 0.5 * pred_wh
        pred_x2y2 = pred_ctr_xy + 0.5 * pred_wh
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box


    def nms(self, dets, scores):
        """"Pure Python NMS."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-10)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    @torch.no_grad()
    def inference_single_image(self, x):
        # backbone
        feat = self.backbone(x)

        # neck
        feat = self.neck(feat)

        # head
        cls_pred, reg_pred = self.head(feat)
        cls_pred = cls_pred[0]                              # [M, C]
        reg_pred = reg_pred[0]                              # [M, 4]
                    
        # scores
        scores, labels = torch.max(cls_pred.sigmoid(), dim=-1)

        # topk
        anchor_boxes = self.anchor_boxes
        if scores.shape[0] > self.topk:
            scores, indices = torch.topk(scores, self.topk)
            labels = labels[indices]
            reg_pred = reg_pred[indices]
            anchor_boxes = anchor_boxes[indices]

        # decode box
        bboxes = self.decode_boxes(anchor_boxes[None], reg_pred[None])[0] # [N, 4]
        # normalize box
        bboxes = torch.clamp(bboxes / self.img_size, 0., 1.)
        
        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # threshold
        keep = np.where(scores >= self.conf_thresh)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores)
            keep[inds[c_keep]] = 1

        keep = np.where(keep > 0)
        bboxes = bboxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        return bboxes, scores, labels


    def forward(self, x):
        if not self.trainable:
            return self.inference_single_image(x)
        else:
            # backbone
            feat = self.backbone(x)

            # neck
            feat = self.neck(feat)

            # head
            cls_pred, reg_pred = self.head(feat)

            # decode box
            box_pred = self.decode_boxes(self.anchor_boxes[None], reg_pred)
                        
            return cls_pred, box_pred
