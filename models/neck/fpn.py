import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import weight_init
from ..basic.conv import Conv


class BasicFPN(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 2048],
                 out_dim=256,
                 ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim

        # latter layers
        self.input_projs = nn.ModuleList()
        self.smooth_layers = nn.ModuleList()
        
        for in_dim in in_dims[::-1]:
            self.input_projs.append(nn.Conv2d(in_dim, out_dim, kernel_size=1))
            self.smooth_layers.append(nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1))


        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)


    def forward(self, feats):
        """
            feats: (List of Tensor) [C3, C4, C5], C_i ∈ R^(B x C_i x H_i x W_i)
        """
        outputs = []
        # [C3, C4, C5] -> [C5, C4, C3]
        feats = feats[::-1]
        top_level_feat = feats[0]
        prev_feat = self.input_projs[0](top_level_feat)
        outputs.append(self.smooth_layers[0](prev_feat))

        for feat, input_proj, smooth_layer in zip(feats[1:], self.input_projs[1:], self.smooth_layers[1:]):
            feat = input_proj(feat)
            top_down_feat = F.interpolate(prev_feat, size=feat.shape[2:], mode='nearest')
            prev_feat = feat + top_down_feat
            outputs.insert(0, smooth_layer(prev_feat))

        # [P3, P4, P5]
        return outputs


class PaFPN(nn.Module):
    def __init__(self, 
                 in_dims=[512, 1024, 2048], # [..., C3, C4, C5, ...]
                 out_dim=256,
                 norm_type=''
                 ):
        super().__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.num_fpn_feats = len(in_dims)

        # input projection layers
        self.input_projs = nn.ModuleList()
        
        for in_dim in in_dims:
            self.input_projs.append(Conv(in_dim, out_dim, k=1, p=0, s=1, act_type=None, norm_type=norm_type))

        # top_down_smooth_layers
        self.top_down_smooth_layers = nn.ModuleList()
        for i in range(self.num_fpn_feats):
            self.top_down_smooth_layers.append(
                Conv(out_dim, out_dim, 
                     k=3, p=1, s=1, 
                     act_type=None, 
                     norm_type=norm_type)
                     )

        # bottom_up_smooth_layers
        self.bottom_up_smooth_layers = nn.ModuleList()
        self.bottom_up_downsample_layers = nn.ModuleList()
        for i in range(self.num_fpn_feats):
            self.bottom_up_smooth_layers.append(
                Conv(out_dim, out_dim, 
                     k=3, p=1, s=1, 
                     act_type=None, 
                     norm_type=norm_type)
                     )
            if i > 0:
                self.bottom_up_downsample_layers.append(
                Conv(out_dim, out_dim, 
                     k=3, p=1, s=2, 
                     act_type=None, 
                     norm_type=norm_type)
                     )
        self._init_weight()


    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_init.c2_xavier_fill(m)


    def forward(self, feats):
        """
            feats: (List of Tensor) [C3, C4, C5], C_i ∈ R^(B x C_i x H_i x W_i)
        """
        in_feats = []
        for feat, layer in zip(feats, self.input_projs):
            in_feats.append(layer(feat))

        # top down fpn
        inter_feats = []
        in_feats = in_feats[::-1]    # [..., C3, C4, C5, ...] -> [..., C5, C4, C3, ...]
        top_level_feat = in_feats[0]
        prev_feat = top_level_feat
        inter_feats.append(self.top_down_smooth_layers[0](prev_feat))

        for feat, smooth in zip(in_feats[1:], self.top_down_smooth_layers[1:]):
            # upsample
            top_down_feat = F.interpolate(prev_feat, size=feat.shape[2:], mode='nearest')
            # sum
            prev_feat = feat + top_down_feat
            inter_feats.insert(0, smooth(prev_feat))

        # Finally, inter_feats contains [P3_inter, P4_inter, P5_inter, P6_inter, P7_inter]
        # bottom up fpn
        out_feats = []
        bottom_level_feat = inter_feats[0]
        prev_feat = bottom_level_feat
        out_feats.append(self.bottom_up_smooth_layers[0](prev_feat))
        for inter_feat, smooth, downsample in zip(inter_feats[1:], 
                                                  self.bottom_up_smooth_layers[1:], 
                                                  self.bottom_up_downsample_layers):
            # downsample
            bottom_up_feat = downsample(prev_feat)
            # sum
            prev_feat = inter_feat + bottom_up_feat
            out_feats.append(smooth(prev_feat))

        return out_feats


def build_fpn(cfg, in_dims, out_dim):
    model = cfg['fpn']
    print('==============================')
    print('FPN: {}'.format(model))
    # build neck
    if model == 'basic_fpn':
        fpn_net = BasicFPN(in_dims=in_dims, out_dim=out_dim)

    elif model == 'pafpn':
        fpn_net = PaFPN(in_dims=in_dims, out_dim=out_dim)
                            
    return fpn_net
