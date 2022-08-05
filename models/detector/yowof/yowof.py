# This is a frame-level model which is set as the Baseline
from os import truncate
import numpy as np
import math
import torch
import torch.nn as nn

from ...backbone import build_backbone_2d
from ...backbone import build_backbone_3d
from ...head.decoupled_head import DecoupledHead
from .encoder import ChannelEncoder
from .loss import Criterion


DEFAULT_SCALE_CLAMP = math.log(1000.0 / 16)
DEFAULT_EXP_CLAMP = math.log(1e8)


class YOWOF(nn.Module):
    def __init__(self, 
                 cfg,
                 device,
                 img_size,
                 len_clip = 16,
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
        self.len_clip = len_clip
        self.num_classes = num_classes
        self.trainable = trainable
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.topk = topk
        self.num_anchors = len(cfg['anchor_size'])
        self.anchor_size = torch.as_tensor(cfg['anchor_size'])
        self.stream_infernce = False
        self.initialization = False

        # ------------------ Anchor Box --------------------
        # [M, 4]
        self.anchor_boxes = self.generate_anchors(img_size)

        # ------------------ Network ---------------------
        # 2D backbone
        self.backbone_2d, bk_dim_2d = build_backbone_2d(
            model_name=cfg['backbone_2d'], 
            pretrained=cfg['pretrained_2d'] and trainable
            )
        # 3D backbone
        self.backbone_3d, bk_dim_3d = build_backbone_3d(
            model_name=cfg['backbone_3d'],
            pretrained=cfg['pretrained_3d'] and trainable,
            part=cfg['backbone_3d_part']
        )

        # Channel Encoder
        self.channel_encoder = ChannelEncoder(
            in_dim=bk_dim_2d + bk_dim_3d,
            out_dim=cfg['head_dim'],
            act_type=cfg['head_act'],
            norm_type=cfg['head_norm']
        )

        # head
        self.head = DecoupledHead(
            head_dim=cfg['head_dim'],
            num_cls_heads=cfg['num_cls_heads'],
            num_reg_heads=cfg['num_reg_heads'],
            act_type=cfg['head_act'],
            norm_type=cfg['head_norm']
            )

        # pred
        self.obj_pred_ = nn.Conv2d(cfg['head_dim'], 1 * self.num_anchors, kernel_size=3, padding=1)
        self.cls_pred_ = nn.Conv2d(cfg['head_dim'], self.num_classes * self.num_anchors, kernel_size=3, padding=1)
        self.reg_pred_ = nn.Conv2d(cfg['head_dim'], 4 * self.num_anchors, kernel_size=3, padding=1)


        if trainable:
            # init bias
            self._init_pred_layers()

        # ------------------ Criterion ---------------------
        if self.trainable:
            self.criterion = Criterion(
                cfg=cfg,
                device=device,
                alpha=cfg['alpha'],
                gamma=cfg['gamma'],
                loss_cls_weight=cfg['loss_cls_weight'],
                loss_reg_weight=cfg['loss_reg_weight'],
                num_classes=num_classes
                )


    def _init_pred_layers(self):  
        # init cls pred
        nn.init.normal_(self.cls_pred_.weight, mean=0, std=0.01)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred_.bias, bias_value)
        # init reg pred
        nn.init.normal_(self.reg_pred_.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred_.bias, 0.0)


    def generate_anchors(self, img_size):
        """fmp_size: list -> [H, W] \n
           stride: int -> output stride
        """
        # generate grid cells
        img_h = img_w = img_size
        fmp_h, fmp_w = img_h // self.stride, img_w // self.stride
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2) + 0.5
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
        anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
        anchor_xy *= self.stride

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

        # [HW, KA, 4] -> [M, 4]
        anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

        return anchor_boxes
        

    def decode_boxes(self, anchor_boxes, pred_reg):
        """
            anchor_boxes: (List[tensor]) [1, M, 4] or [M, 4]
            pred_reg: (List[tensor]) [B, M, 4] or [M, 4]
        """
        # x = x_anchor + dx * w_anchor
        # y = y_anchor + dy * h_anchor
        pred_ctr_offset = pred_reg[..., :2] * anchor_boxes[..., 2:]
        if self.cfg['ctr_clamp'] is not None:
            pred_ctr_offset = torch.clamp(pred_ctr_offset,
                                        max=self.cfg['ctr_clamp'],
                                        min=-self.cfg['ctr_clamp'])
        pred_ctr_xy = anchor_boxes[..., :2] + pred_ctr_offset

        # w = w_anchor * exp(tw)
        # h = h_anchor * exp(th)
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

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def post_process(self, scores, labels, bboxes):
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
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]

        return scores, labels, bboxes
    

    def set_inference_mode(self, mode='stream'):
        if mode == 'stream':
            self.stream_infernce = True
        elif mode == 'clip':
            self.stream_infernce = False


    def inference_video_clip(self, video_clips):
        # prepare
        backbone_2d_feats = []
        img_size = video_clips[0].shape[-1]

        # 2D backbone
        for i in range(len(video_clips)):
            if i == len(video_clips) - 1:
                feat, feat_2d = self.backbone_2d(video_clips[i], truncate=False)
            else:
                feat = self.backbone_2d(video_clips[i], truncate=True)

            backbone_2d_feats.append(feat)

        # 3D backbone
        feat_3d = self.backbone_3d(torch.stack(backbone_2d_feats, dim=2))

        # channel encoder
        feat = self.channel_encoder(torch.cat([feat_2d, feat_3d], dim=1))

        # head
        cls_feats, reg_feats = self.head(feat)

        obj_pred = self.obj_pred_(reg_feats)
        cls_pred = self.cls_pred_(cls_feats)
        reg_pred = self.reg_pred_(reg_feats)

        # implicit objectness
        B, _, H, W = obj_pred.size()
        obj_pred = obj_pred.view(B, -1, 1, H, W)
        cls_pred = cls_pred.view(B, -1, self.num_classes, H, W)
        normalized_cls_pred = cls_pred + obj_pred - torch.log(
            1. + 
            torch.clamp(cls_pred, max=DEFAULT_EXP_CLAMP).exp() + 
            torch.clamp(obj_pred, max=DEFAULT_EXP_CLAMP).exp())
        # [B, KA, C, H, W] -> [B, H, W, KA, C] -> [B, M, C], M = HxWxKA
        normalized_cls_pred = normalized_cls_pred.permute(0, 3, 4, 1, 2).contiguous()
        normalized_cls_pred = normalized_cls_pred.view(B, -1, self.num_classes)

        # [B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
        reg_pred =reg_pred.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
        reg_pred = reg_pred.view(B, -1, 4)

        # [1, M, C] -> [M, C]
        cls_pred = normalized_cls_pred[0]
        reg_pred = reg_pred[0]
                    
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
        bboxes = self.decode_boxes(anchor_boxes, reg_pred) # [N, 4]
        # normalize box
        bboxes = torch.clamp(bboxes / img_size, 0., 1.)
        
        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # post-process
        scores, labels, bboxes = self.post_process(scores, labels, bboxes)

        return backbone_2d_feats, scores, labels, bboxes


    def inference_single_frame(self, x):
        img_size = x.shape[-1]
        # backbone
        cur_feat, cur_feat_2d = self.backbone_2d(x, truncate=False)

        # push the current feature
        self.clip_feats.append(cur_feat)
        # delete the oldest feature
        del self.clip_feats[0]

        # 3D backbone
        cur_feat_3d = self.backbone_3d(torch.stack(self.clip_feats, dim=2))

        # channel encoder
        cur_feat = self.channel_encoder(torch.cat([cur_feat_2d, cur_feat_3d], dim=1))

        # head
        cls_feats, reg_feats = self.head(cur_feat)

        obj_pred = self.obj_pred_(reg_feats)
        cls_pred = self.cls_pred_(cls_feats)
        reg_pred = self.reg_pred_(reg_feats)

        # implicit objectness
        B, _, H, W = obj_pred.size()
        obj_pred = obj_pred.view(B, -1, 1, H, W)
        cls_pred = cls_pred.view(B, -1, self.num_classes, H, W)
        normalized_cls_pred = cls_pred + obj_pred - torch.log(
            1. + 
            torch.clamp(cls_pred, max=DEFAULT_EXP_CLAMP).exp() + 
            torch.clamp(obj_pred, max=DEFAULT_EXP_CLAMP).exp())
        # [B, KA, C, H, W] -> [B, H, W, KA, C] -> [B, M, C], M = HxWxKA
        normalized_cls_pred = normalized_cls_pred.permute(0, 3, 4, 1, 2).contiguous()
        normalized_cls_pred = normalized_cls_pred.view(B, -1, self.num_classes)

        # [B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
        reg_pred =reg_pred.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
        reg_pred = reg_pred.view(B, -1, 4)

        # [1, M, C] -> [M, C]
        cls_pred = normalized_cls_pred[0]
        reg_pred = reg_pred[0]

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
        bboxes = torch.clamp(bboxes / img_size, 0., 1.)
        
        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # post-process
        scores, labels, bboxes = self.post_process(scores, labels, bboxes)

        return cur_feat_2d, scores, labels, bboxes


    @torch.no_grad()
    def inference(self, x):
        # Init inference, model processes a video clip
        if not self.stream_infernce:
            bk_feats, scores, labels, bboxes = self.inference_video_clip(x)

            return scores, labels, bboxes
            
        else:
            self.scores_list = []
            self.bboxes_list = []
            self.labels_list = []

            if self.initialization:
                # Init stage, detector process a video clip
                # and output results of per frame
                (   
                    clip_feats,
                    init_scores, 
                    init_labels, 
                    init_bboxes
                    ) = self.inference_video_clip(x)
                self.initialization = False
                self.clip_feats = clip_feats

                return init_scores, init_labels, init_bboxes
            else:
                # After init stage, detector process current frame
                (
                    cur_feats,
                    cur_scores,
                    cur_labels,
                    cur_bboxes
                    ) = self.inference_single_frame(x)

                return cur_scores, cur_labels, cur_bboxes


    def forward(self, video_clips, targets=None, vis_data=False):
        """
            video_clips: List[Tensor] -> [Tensor[B, C, H, W], ..., Tensor[B, C, H, W]].
        """                        
        if not self.trainable:
            return self.inference(video_clips)
        else:
            backbone_2d_feats = []
            # 2D backbone
            for i in range(len(video_clips)):
                if i == len(video_clips) - 1:
                    feat, feat_2d = self.backbone_2d(video_clips[i], truncate=False)
                else:
                    feat = self.backbone_2d(video_clips[i], truncate=True)

                backbone_2d_feats.append(feat)

            # 3D backbone
            print(torch.stack(backbone_2d_feats, dim=2).device)
            feat_3d = self.backbone_3d(torch.stack(backbone_2d_feats, dim=2))

            # channel encoder
            feat = self.channel_encoder(torch.cat([feat_2d, feat_3d], dim=1))

            # detection head
            cls_feats, reg_feats = self.head(feat)

            obj_pred = self.obj_pred_(reg_feats)
            cls_pred = self.cls_pred_(cls_feats)
            reg_pred = self.reg_pred_(reg_feats)

            # implicit objectness
            B, _, H, W = obj_pred.size()
            obj_pred = obj_pred.view(B, -1, 1, H, W)
            cls_pred = cls_pred.view(B, -1, self.num_classes, H, W)
            normalized_cls_pred = cls_pred + obj_pred - torch.log(
                1. + 
                torch.clamp(cls_pred, max=DEFAULT_EXP_CLAMP).exp() + 
                torch.clamp(obj_pred, max=DEFAULT_EXP_CLAMP).exp())
            # [B, KA, C, H, W] -> [B, H, W, KA, C] -> [B, M, C], M = HxWxKA
            normalized_cls_pred = normalized_cls_pred.permute(0, 3, 4, 1, 2).contiguous()
            normalized_cls_pred = normalized_cls_pred.view(B, -1, self.num_classes)

            # [B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
            reg_pred =reg_pred.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
            reg_pred = reg_pred.view(B, -1, 4)

            # decode box
            box_pred = self.decode_boxes(self.anchor_boxes[None], reg_pred)

            outputs = {"cls_preds": normalized_cls_pred,
                       "box_preds": box_pred,
                       "anchors": self.anchor_boxes,
                       'strides': self.stride}

            # loss
            loss_dict = self.criterion(
                outputs=outputs, 
                targets=targets, 
                video_clips=video_clips,
                vis_data=vis_data
                )

            return loss_dict 
