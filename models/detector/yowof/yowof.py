# This is a frame-level model which is set as the Baseline
import numpy as np
import torch
import torch.nn as nn

from models.basic.conv import Conv2d

from ...backbone import build_backbone_2d
from ...backbone import build_backbone_3d
from ...head.decoupled_head import DecoupledHead
from ...basic.convlstm import ConvLSTM
from .encoder import ChannelEncoder
from .loss import Criterion



class YOWOF(nn.Module):
    def __init__(self, 
                 cfg,
                 device,
                 img_size,
                 anchor_size,
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
        self.num_anchors = len(anchor_size)
        self.anchor_size = torch.as_tensor(anchor_size)
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

        # temporal encoder
        self.temporal_encoder = ConvLSTM(
            in_dim=bk_dim_2d,
            hidden_dim=cfg['conv_lstm_hdm'],
            kernel_size=cfg['conv_lstm_ks'],
            num_layers=cfg['conv_lstm_nl'],
            return_all_layers=False,
            inf_full_seq=trainable
        )

        # head
        self.head = Conv2d(cfg['conv_lstm_hdm'], cfg['head_dim'], k=3, p=1, act_type=cfg['head_act'], norm_type=cfg['head_norm'])

        # output
        self.pred = nn.Conv2d(cfg['head_dim'], self.num_anchors * (1 + num_classes + 4), kernel_size=3, padding=1)


        if trainable:
            # init bias
            self._init_bias()

        # ------------------ Criterion ---------------------
        if self.trainable:
            self.criterion = Criterion(
                cfg=cfg,
                device=device,
                anchor_size=self.anchor_size,
                num_anchors=self.num_anchors,
                num_classes=self.num_classes,
                loss_obj_weight=cfg['loss_obj_weight'],
                loss_noobj_weight=cfg['loss_noobj_weight'],
                loss_cls_weight=cfg['loss_cls_weight'],
                loss_reg_weight=cfg['loss_reg_weight']
                )


    def _init_bias(self):  
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        # init bias
        nn.init.constant_(self.pred.bias[..., :self.num_anchors], bias_value)
        nn.init.constant_(self.pred.bias[..., self.num_anchors:self.num_anchors*(1 + self.num_classes)], bias_value)


    def generate_anchors(self, img_size):
        """fmp_size: list -> [H, W] \n
           stride: int -> output stride
        """
        # generate grid cells
        img_h = img_w = img_size
        fmp_h, fmp_w = img_h // self.stride, img_w // self.stride
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)])
        # [H, W, 2] -> [HW, 2]
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)
        # [HW, 2] -> [HW, 1, 2] -> [HW, KA, 2] 
        anchor_xy = anchor_xy[:, None, :].repeat(1, self.num_anchors, 1)
        anchor_xy *= self.stride

        # [KA, 2] -> [1, KA, 2] -> [HW, KA, 2]
        anchor_wh = self.anchor_size[None, :, :].repeat(fmp_h*fmp_w, 1, 1)

        # [HW, KA, 4] -> [M, 4]
        anchor_boxes = torch.cat([anchor_xy, anchor_wh], dim=-1)
        anchor_boxes = anchor_boxes.view(-1, 4).to(self.device)

        return anchor_boxes
        

    def decode_bbox(self, anchors, reg_pred):
        """
        Input:
            anchors:  [B, M, 4] or [M, 4]
            reg_pred: [B, M, 4] or [M, 4]
        Output:
            box_pred: [B, M, 4] or [M, 4]
        """
        # txty -> cxcy
        xy_pred = reg_pred[..., :2].sigmoid() * self.stride + anchors[..., :2]
        # twth -> wh
        wh_pred = reg_pred[..., 2:].exp() * anchors[..., 2:]

        # xywh -> x1y1x2y2
        x1y1_pred = xy_pred - wh_pred * 0.5
        x2y2_pred = xy_pred + wh_pred * 0.5
        box_pred = torch.cat([x1y1_pred, x2y2_pred], dim=-1)

        return box_pred


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
            self.temporal_encoder.inf_full_seq = False
        elif mode == 'clip':
            self.stream_infernce = False
            self.temporal_encoder.inf_full_seq = True


    def inference_video_clip(self, x):
        # prepare
        backbone_feats = []
        img_size = x[0].shape[-1]

        # backbone
        for i in range(len(x)):
            feat = self.backbone_2d(x[i])

            backbone_feats.append(feat)

        # temporal encoder
        feats, _ = self.temporal_encoder(backbone_feats)
        feat = feats[-1][-1]

        # head
        feat = self.head(feat)

        # pred
        pred = self.pred(feat)

        B, K, C = pred.size(0), self.num_anchors, self.num_classes
        # [B, K*C, H, W] -> [B, H, W, K*C] -> [B, M, C], M = HWK
        conf_pred = pred[:, :K, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
        cls_pred = pred[:, K:K*(1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        reg_pred = pred[:, K*(1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

        # [1, M, C] -> [M, C]
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]
        reg_pred = reg_pred[0]
                    
        # scores
        scores, labels = torch.max(torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1), dim=-1)

        # topk
        anchor_boxes = self.anchor_boxes
        if scores.shape[0] > self.topk:
            scores, indices = torch.topk(scores, self.topk)
            labels = labels[indices]
            reg_pred = reg_pred[indices]
            anchor_boxes = anchor_boxes[indices]

        # decode box
        bboxes = self.decode_bbox(anchor_boxes, reg_pred) # [N, 4]
        # normalize box
        bboxes = torch.clamp(bboxes / img_size, 0., 1.)
        
        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # post-process
        scores, labels, bboxes = self.post_process(scores, labels, bboxes)

        return backbone_feats, scores, labels, bboxes


    def inference_single_frame(self, x):
        img_size = x.shape[-1]
        # backbone
        cur_bk_feat = self.backbone_2d(x)

        # push the current feature
        self.clip_feats.append(cur_bk_feat)
        # delete the oldest feature
        del self.clip_feats[0]

        # temporal encoder
        cur_feats, _ = self.temporal_encoder(self.clip_feats)
        cur_feat = cur_feats[-1]

        # head
        cur_feat = self.head(cur_feat)

        # pred
        pred = self.pred(cur_feat)

        B, K, C = pred.size(0), self.num_anchors, self.num_classes
        # [B, K*C, H, W] -> [B, H, W, K*C] -> [B, M, C], M = HWK
        conf_pred = pred[:, :K, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
        cls_pred = pred[:, K:K*(1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        reg_pred = pred[:, K*(1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

        # [1, M, C] -> [M, C]
        conf_pred = conf_pred[0]
        cls_pred = cls_pred[0]
        reg_pred = reg_pred[0]
                    
        # scores
        scores, labels = torch.max(torch.sigmoid(conf_pred) * torch.softmax(cls_pred, dim=-1), dim=-1)

        # topk
        anchor_boxes = self.anchor_boxes
        if scores.shape[0] > self.topk:
            scores, indices = torch.topk(scores, self.topk)
            labels = labels[indices]
            reg_pred = reg_pred[indices]
            anchor_boxes = anchor_boxes[indices]

        # decode box
        bboxes = self.decode_bbox(anchor_boxes, reg_pred) # [N, 4]
        # normalize box
        bboxes = torch.clamp(bboxes / img_size, 0., 1.)
        
        # to cpu
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # post-process
        scores, labels, bboxes = self.post_process(scores, labels, bboxes)

        return cur_bk_feat, scores, labels, bboxes


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
                self.temporal_encoder.initialization = True
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
            backbone_feats = []
            # backbone
            for i in range(len(video_clips)):
                feat = self.backbone_2d(video_clips[i])

                backbone_feats.append(feat)

            # temporal encoder
            feats, _ = self.temporal_encoder(backbone_feats)
            feat = feats[-1][-1]

            # head
            feat = self.head(feat)

            # pred
            pred = self.pred(feat)

            B, K, C = pred.size(0), self.num_anchors, self.num_classes
            # [B, K*C, H, W] -> [B, H, W, K*C] -> [B, M, C], M = HWK
            conf_pred = pred[:, :K, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 1)
            cls_pred = pred[:, K:K*(1 + C), :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, C)
            reg_pred = pred[:, K*(1 + C):, :, :].permute(0, 2, 3, 1).contiguous().view(B, -1, 4)

            # decode box
            box_pred = self.decode_bbox(self.anchor_boxes[None], reg_pred)

            outputs = {"conf_pred": conf_pred,
                       "cls_pred": cls_pred,
                       "box_pred": box_pred,
                       "anchor_size": self.anchor_size,
                       "img_size": self.img_size,
                       "stride": self.stride}

            # loss
            loss_dict = self.criterion(
                outputs=outputs, 
                targets=targets, 
                video_clips=video_clips,
                vis_data=vis_data
                )

            return loss_dict
