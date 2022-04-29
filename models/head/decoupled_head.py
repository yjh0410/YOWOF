import torch
import torch.nn as nn

from ..basic.conv import Conv


class NaiveHead(nn.Module):
    def __init__(self, 
                 head_dim=256,
                 kernel_size=3,
                 padding=1,
                 act_type='relu'):
        super().__init__()
        self.head_dim = head_dim

        self.cls_feats = nn.Sequential(
            Conv(head_dim, head_dim, k=kernel_size, p=padding, act_type=act_type),
            Conv(head_dim, head_dim, k=kernel_size, p=padding, act_type=act_type)
        )
        self.reg_feats = nn.Sequential(
            Conv(head_dim, head_dim, k=kernel_size, p=padding, act_type=act_type),
            Conv(head_dim, head_dim, k=kernel_size, p=padding, act_type=act_type),
            Conv(head_dim, head_dim, k=kernel_size, p=padding, act_type=act_type),
            Conv(head_dim, head_dim, k=kernel_size, p=padding, act_type=act_type)
        )

        self._init_weight()


    def _init_weight(self):
        # init weight of detection head
        for m in [self.cls_feats, self.reg_feats]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """
            in_feats: (List of Tensor) [C3, C4, C5]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats


class DecoupledHead(nn.Module):
    def __init__(self, 
                 head='naive_head',
                 head_dim=256,
                 kernel_size=3,
                 padding=1,
                 num_classes=80, 
                 trainable=False,
                 num_anchors=1,
                 act_type='relu'):
        super().__init__()
        self.num_classes = num_classes
        self.head_dim = head_dim

        print('==============================')
        print('Head: {}'.format(head))

        # feature stage
        if head == 'naive_head':
            self.head = NaiveHead(head_dim, kernel_size, padding, act_type)

        # prediction stage
        self.obj_pred = nn.Conv2d(head_dim, 1 * num_anchors, kernel_size=3, padding=1)
        self.cls_pred = nn.Conv2d(head_dim, self.num_classes * num_anchors, kernel_size=3, padding=1)
        self.reg_pred = nn.Conv2d(head_dim, 4 * num_anchors, kernel_size=3, padding=1)

        if trainable:
            # init bias
            self._init_pred_layers()


    def _init_pred_layers(self):  
        # init cls pred
        nn.init.normal_(self.cls_pred.weight, mean=0, std=0.01)
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred.bias, bias_value)
        # init reg pred
        nn.init.normal_(self.reg_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.reg_pred.bias, 0.0)
        # init obj pred
        nn.init.normal_(self.obj_pred.weight, mean=0, std=0.01)
        nn.init.constant_(self.obj_pred.bias, 0.0)
        

    def forward(self, x):
        """
            in_feats: (List of Tensor) [C3, C4, C5]
        """
        cls_feats, reg_feats = self.head(x)

        obj_pred = self.obj_pred(reg_feats)
        cls_pred = self.cls_pred(cls_feats)
        reg_pred = self.reg_pred(reg_feats)

        # implicit objectness
        B, _, H, W = obj_pred.size()
        obj_pred = obj_pred.view(B, -1, 1, H, W)
        cls_pred = cls_pred.view(B, -1, self.num_classes, H, W)
        normalized_cls_pred = cls_pred + obj_pred - torch.log(
            1. + torch.clamp(cls_pred.exp(), max=1e8) + torch.clamp(
                obj_pred.exp(), max=1e8))
        # [B, KA, C, H, W] -> [B, H, W, KA, C] -> [B, M, C], M = HxWxKA
        normalized_cls_pred = normalized_cls_pred.permute(0, 3, 4, 1, 2).contiguous()
        normalized_cls_pred = normalized_cls_pred.view(B, -1, self.num_classes)

        # [B, KA*4, H, W] -> [B, KA, 4, H, W] -> [B, H, W, KA, 4] -> [B, M, 4]
        reg_pred =reg_pred.view(B, -1, 4, H, W).permute(0, 3, 4, 1, 2).contiguous()
        reg_pred = reg_pred.view(B, -1, 4)

        return normalized_cls_pred, reg_pred
