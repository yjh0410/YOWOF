import torch
import torch.nn as nn
from ...basic.conv import Conv


# Motion Encoder
class MotionEncoder(nn.Module):
    def __init__(self, in_dim, expand_ratio=0.5, len_clip=1):
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

        # output conv
        self.out_conv = nn.Sequential(
            Conv(inter_dim,
                 in_dim,
                 k=1,
                 act_type='relu',
                 norm_type='BN')
        )

    
    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        input_feats = []
        # List[K, B, C, H, W] -> List[K, B, C', H, W]
        for feat, layer in zip(feats, self.input_proj):
            input_feats.append(layer(feat))

        motion_feats = [torch.zeros_like(input_feats[0])]
        for idx in range(len(feats) - 1):
            feat_t1 = input_feats[idx]
            feat_t2 = input_feats[idx + 1]
            smooth = self.smooth_layers[idx]
            motion_feat = smooth(feat_t2) - feat_t1
            motion_feats.append(motion_feat)

        # List[K, B, C', H, W] -> [B, K, C', H, W]
        mfeats = torch.stack(motion_feats, dim=1)
        B, K, C, H, W = mfeats.size()
        # [B, K, C', H, W] -> [BK, C', H, W]
        mfeats = mfeats.view(-1, C, H, W)
        # [BK, C', H, W] -> [BK, C, H, W]
        out_feats = self.out_conv(mfeats)

        # [BK, C, H, W] -> [B, K, C, H, W]
        out_feats = out_feats.view(B, K, -1, H, W)
        # [B, K, C, H, W] -> [B, C, K, H, W]
        out_feats = out_feats.permute(0, 2, 1, 3, 4).contiguous()

        return out_feats


# Spatio-Temporal Encoder
class SpatioEncoder(nn.Module):
    def __init__(self, in_dim, expand_ratio=0.5, len_clip=1):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.expand_ratio = expand_ratio
        self.len_clip = len_clip

        # input projection
        inter_dim = int(in_dim / expand_ratio)

        # Spatio Attention: [BK, C, H, W] shape required
        self.spatio_attn = nn.Sequential(
            Conv(in_dim, inter_dim, k=1, act_type='relu', norm_type='BN'),
            Conv(inter_dim, inter_dim, k=3, p=1, act_type='relu', norm_type='BN'),
            Conv(inter_dim, in_dim, k=1, act_type='relu', norm_type='BN')
        )

    
    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        # List[K, B, C, H, W] -> [B, K, C, H, W]
        spatio_feats = torch.stack(feats, dim=1)
        B, K, C, H, W = spatio_feats.size()
        # [B, K, C, H, W] -> [BK, C, H, W]
        spatio_feats = spatio_feats.view(-1, C, H, W)

        # Spatio attention
        spatio_feats = self.spatio_attn(spatio_feats) * spatio_feats + spatio_feats

        # [BK, C, H, W] -> [B, K, C, H, W]
        out_feats = spatio_feats.view(B, K, C, H, W)
        # [B, K, C, H, W] -> [B, C, K, H, W]
        out_feats = out_feats.permute(0, 2, 1, 3, 4).contiguous()

        return out_feats


# Temporal Encoder
class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, expand_ratio=0.5, len_clip=1):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.expand_ratio = expand_ratio
        self.len_clip = len_clip

        # Temporal Attention: [B, C, K, HW] shape required
        self.temporal_attn = nn.Sequential(
            Conv(in_dim, in_dim, k=(3, 1), p=(1, 0), act_type='relu', norm_type='BN'),
            Conv(in_dim, in_dim, k=1, act_type=None, norm_type='BN'),
            nn.Sigmoid()
        )

    
    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        B, C, H, W = feats[0].size()
        # List[K, B, C, H, W] -> [B, C, K, H, W] -> [B, C, K, HW]
        temporal_feats = torch.stack(feats, dim=2).flatten(-2)

        # Temporal attention
        temporal_feats = self.temporal_attn(temporal_feats) * temporal_feats + temporal_feats

        # [B, C, K, HW] -> [B, C, K, H, W]
        out_feats = temporal_feats.view(B, -1, self.len_clip, H, W)

        return out_feats


# STM Encoder
class STMEncoder(nn.Module):
    def __init__(self, in_dim, expand_ratio=0.5, len_clip=1, depth=1):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.expand_ratio = expand_ratio
        self.len_clip = len_clip
        self.depth = depth

        self.spatio_encoders = nn.ModuleList([
            SpatioEncoder(in_dim, expand_ratio, len_clip)
            for _ in range(depth)
        ])

        self.temporal_encoders = nn.ModuleList([
            TemporalEncoder(in_dim, expand_ratio, len_clip)
            for _ in range(depth)
        ])

        self.motion_encoders = nn.ModuleList([
            MotionEncoder(in_dim, expand_ratio, len_clip)
            for _ in range(depth)
        ])

        # fuse layer
        self.fuse_layers = nn.ModuleList([
            Conv(in_dim, in_dim , k=1, act_type='relu', norm_type='BN')
            for _ in range(depth)
        ])

        # out layer
        self.out_layer = Conv(in_dim * len_clip, in_dim, k=3, p=1, act_type='relu', norm_type='BN')


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        B, C, H, W = feats[0].size()
        for se, te, me, fl in zip(self.spatio_encoders, self.temporal_encoders, self.motion_encoders, self.fuse_layers):
            # out shape: [B, C, K, H, W]
            spatio_feats = se(feats)
            temporal_feats = te(feats)
            motion_feats = me(feats)

            feats = spatio_feats + temporal_feats + motion_feats
            # [B, C, K, H, W] -> [B, C, K, HW]
            feats = fl(feats.flatten(-2))
            # [B, C, K, HW] -> [B, C, K, H, W]
            feats = feats.view(B, C, self.len_clip, H, W)
            # [B, C, K, H, W] -> List[K, B, C, H, W]
            feats = [feats[:, :, k, :, :] for k in range(self.len_clip)]

        # output: [B, C, H, W]
        out_feat = self.out_layer(torch.cat(feats, dim=1))

        return out_feat

