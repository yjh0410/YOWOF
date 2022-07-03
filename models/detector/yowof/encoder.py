import torch
import torch.nn as nn
from ...basic.conv import Conv


# Channel Self Attetion Module
class CSAM(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.scale = in_dim ** -0.5

        # query, key, value
        self.qkv_conv = nn.Conv2d(in_dim, 3*in_dim, kernel_size=1, bias=False)

        # attention
        self.attend = nn.Softmax(dim = -1)
        self.linear = Conv(in_dim, in_dim, k=1, act_type=None, norm_type='BN')

        # output
        self.out = nn.Sequential(
            nn.Dropout2d(dropout),
            Conv(in_dim, in_dim, k=1, act_type=None, norm_type='BN')
            )


    def forward(self, x):
        """
            x: (Tensor) [B, C, H, W]
        """
        B, C, H, W = x.size()

        qkv = self.qkv_conv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=1)
        # [B, C, H, W] -> [B, C, N], N = HW
        q, k, v = q.flatten(-2), k.flatten(-2), v.flatten(-2)

        # [B, C, N] x [B, N, C] -> [B, C, C]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attention: [B, C, C] x [B, C, N] -> [B, C, N]
        attn = self.attend(dots)
        y = torch.matmul(attn, v)

        y = y.view(B, C, H, W)
        y = y + self.linear(y)

        return y + self.out(y)


# Motion Encoder
class MotionEncoder(nn.Module):
    def __init__(self, in_dim, expand_ratio=0.5, len_clip=1, dropout=0.):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.expand_ratio = expand_ratio
        self.len_clip = len_clip

        # input projection
        inter_dim = int(in_dim * expand_ratio)
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_dim, inter_dim, kernel_size=1)
            for _ in range(len_clip)
        ])

        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(inter_dim, inter_dim, kernel_size=1)
            for _ in range(len_clip - 1)
        ])

        # fuse conv
        self.fuse_conv = nn.Sequential(
            Conv(inter_dim * (len_clip - 1), in_dim, k=1, act_type='relu', norm_type='BN'),
            Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')
        )

        # CSAM
        self.csam = CSAM(in_dim, dropout)

        # output conv
        self.out_conv = Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')

    
    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        input_feats = []
        # List[K, B, C, H, W] -> List[K, B, C', H, W]
        for feat, layer in zip(feats, self.input_proj):
            input_feats.append(layer(feat))

        motion_feats = []
        for idx in range(len(feats) - 1):
            feat_t1 = input_feats[idx]
            feat_t2 = input_feats[idx + 1]
            smooth = self.smooth_layers[idx]
            motion_feat = smooth(feat_t2) - feat_t1
            motion_feats.append(motion_feat)

        # List[K-1, B, C', H, W] -> [B, (K-1)C', H, W]
        mfeats = torch.cat(motion_feats, dim=1)
        # [B, (K-1)C', H, W] -> [B, C, H, W]
        feats = self.fuse_conv(mfeats)

        # Channel Self Attention Module
        feats = self.csam(feats)

        # output
        out_feats = self.out_conv(feats)

        return out_feats


# Spatio-Temporal Encoder
class SpatioTemporalEncoder(nn.Module):
    def __init__(self, in_dim, expand_ratio=0.5, len_clip=1, dropout=0.):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.expand_ratio = expand_ratio
        self.len_clip = len_clip

        # spatial avgpool
        self.spa_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # coefficient
        self.coeff = nn.Sequential(
            nn.Linear(in_dim * len_clip, len_clip),
            nn.ReLU(inplace=True),
            nn.Linear(len_clip, len_clip),
            nn.Softmax()
        )

        # smooth
        self.smooth = Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')

        # CSAM
        self.csam = CSAM(in_dim, dropout)

        # output conv
        self.out_conv = Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')


    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        # List[K, B, C, H, W] -> [B, KC, H, W]
        spatio_feats = torch.cat(feats, dim=1)

        # [B, KC, H, W] -> [B, KC, 1, 1] -> [B, KC]
        spatio_vectors = self.spa_avgpool(spatio_feats)
        spatio_vectors = spatio_vectors.flatten(1)

        # [B, KC] -> [B, K]
        coeffs = self.coeff(spatio_vectors)

        # Weighted summation: List[K, B, C, H, W] -> [B, C, H, W]
        feats = sum([feats[k] * coeffs[..., k] for k in range(self.len_clip)])
        feats = self.smooth(feats)

        # Channel Self Attention Module
        feats = self.csam(feats)

        # output
        out_feats = self.out_conv(feats)

        return out_feats


# STM Encoder
class STMEncoder(nn.Module):
    def __init__(self, in_dim, expand_ratio=0.5, len_clip=1, dropout=0.):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.expand_ratio = expand_ratio
        self.len_clip = len_clip

        # input proj
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=1)
            for _ in range(len_clip)
        ])

        # SpatioTemporalEncoder
        self.st_encoder = SpatioTemporalEncoder(in_dim, expand_ratio, len_clip, dropout)

        # Motion Encoder
        self.mt_encoder = MotionEncoder(in_dim, expand_ratio, len_clip, dropout)

        # CSAM
        self.csam = CSAM(in_dim, dropout)

        # fuse layer
        self.fuse_conv = nn.Sequential(
            Conv(in_dim * 3, in_dim, k=1, act_type='relu', norm_type='BN'),
            Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')
        )

        # out layer
        self.out_conv = Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        input_feats = []
        for feat, layer in zip(feats, self.input_proj):
            input_feats.append(layer(feat))

        # spat-temp encoder
        spattemp_feats = self.st_encoder(input_feats)

        # motion encoder
        motion_feats = self.mt_encoder(input_feats)

        # fuse conv
        feats = torch.cat([input_feats[-1], spattemp_feats, motion_feats], dim=1)
        feats = self.fuse_conv(feats)

        # Channel Self Attention Module
        feats = self.csam(feats)

        # output
        out_feats = self.out_conv(feats)

        return out_feats
