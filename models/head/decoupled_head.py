import torch
import torch.nn as nn

from ..basic.conv import Conv2d


class DecoupledHead(nn.Module):
    def __init__(self, 
                 head_dim=256,
                 num_cls_heads=4,
                 num_reg_heads=4,
                 act_type='relu',
                 norm_type='',
                 depthwise=False):
        super().__init__()

        print('==============================')
        print('Head: Decoupled Head')
        self.cls_feats = nn.Sequential(*[Conv2d(head_dim, 
                                            head_dim, 
                                            k=3, p=1, s=1, 
                                            act_type=act_type, 
                                            norm_type=norm_type,
                                            depthwise=depthwise) for _ in range(num_cls_heads)])
        self.reg_feats = nn.Sequential(*[Conv2d(head_dim, 
                                            head_dim, 
                                            k=3, p=1, s=1, 
                                            act_type=act_type, 
                                            norm_type=norm_type,
                                            depthwise=depthwise) for _ in range(num_reg_heads)])

        self._init_weight()


    def _init_weight(self):
        # init weight of detection head
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        """
            in_feats: (Tensor) [B, C, H, W]
        """
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)

        return cls_feats, reg_feats


# build head
def build_head(cfg):
    head = DecoupledHead(
        head_dim=cfg['head_dim'],
        num_cls_heads=cfg['num_cls_heads'],
        num_reg_heads=cfg['num_reg_heads'],
        act_type=cfg['head_act'],
        norm_type=cfg['head_norm']
        )

    return head
