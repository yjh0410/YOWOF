import math
import torch
import torch.nn as nn
from ...basic.conv import Conv


# Channel Self Attetion Module
class CSAM(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.scale = in_dim ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim*4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_dim*4, in_dim, kernel_size=1),
            nn.Dropout2d(dropout)
        )
        self.norm = nn.InstanceNorm2d(in_dim)


    def forward(self, x):
        """
            x: (Tensor) [B, C, H, W]
        """
        B, C, H, W = x.size()

        # query: [B, C, N]
        q = x.view(B, C, -1)
        # key: [B, C, N] -> [B, N, C]
        k = x.view(B, C, -1).transpose(-1, -2)
        # value: [B, C, N]
        v = x.view(B, C, -1)

        # self attention
        attn = self.attend(torch.matmul(q, k) * self.scale)
        out = torch.matmul(attn, v)

        # [B, C, N] -> [B, C, H, W]
        out = out.view(B, C, H, W)

        # output
        out = self.ffn(out) + x

        return out


# Spatial Self Attetion Module
class SSAM(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.scale = in_dim ** -0.5

        self.gamma = nn.Parameter(torch.zeros(1))

        # attention
        self.attend = nn.Softmax(dim = -1)
        self.ffn = nn.Sequential(
            nn.Conv2d(in_dim, in_dim*4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_dim*4, in_dim, kernel_size=1),
            nn.Dropout2d(dropout)
        )
        self.norm = nn.InstanceNorm2d(in_dim)

    def forward(self, x):
        """
            x: (Tensor) [B, C, H, W]
        """
        B, C, H, W = x.size()

        # query: [B, N, C]
        q = x.view(B, C, -1).transpose(-1, -2)
        # key: [B, C, N]
        k = x.view(B, C, -1) 
        # value: [B, N, C]
        v = x.view(B, C, -1).transpose(-1, -2)

        # self attention: [B, N, N]
        attn = self.attend(torch.matmul(q, k) * self.scale)
        out = torch.matmul(attn, v)

        # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        out = out.transpose(-1, -2).view(B, C, H, W)

        # output
        out = self.ffn(out) * self.gamma + x

        return out


class ChannelFuseConv(nn.Module):
    def __init__(self, c1, c2, dropout=0.):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv(c1 + c2, c2, k=1, act_type='relu', norm_type='BN'),
            Conv(c2, c2, k=3, p=1, act_type='relu', norm_type='BN'),
            CSAM(c2, dropout),
            Conv(c2, c2, k=3, p=1, act_type='relu', norm_type='BN')
        )

    def forward(self, x):
        """
            x: [B, CN, H, W]
        """
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x


class TemporalShuffle(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, feats):
        """
        Inputs:
            feats: (List) [K, B, C, H, W]
        Outputs:
            x: (Tensor) [B, CK, H, W]
        """
        x = torch.stack(feats, dim=1)

        # [B, K, C, H, W] -> [B, C, K, H, W]
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten: [B, CK, H, W]
        x = x.flatten(1, 2)

        return x


class TemporalUnShuffle(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, feats):
        """
        Inputs:
            feats: (Tensor) [B, CK, H, W]
        Outputs:
            x: (List) [K, B, C, H, W]
        """
        B, C, H, W = feats.size()
        G = self.len_clip
        S = C // G
        # [B, CK, H, W] -> [B, C, K, H, W]
        x = feats.view(B, S, G, H, W)

        # [B, C, K, H, W] -> [B, K, C, H, W]
        x = torch.transpose(x, 1, 2).contiguous()

        # [B, K, C, H, W] -> (List) [K, B, C, H, W]
        x = [x[:, k, :, :, :] for k in range(G)]

        return x


# STC Encoder
class STCEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, len_clip=1, depth=1, dropout=0.):
        """
            in_dim: (Int) -> dim of single feature
            len_clip: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.len_clip = len_clip

        # input proj
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, kernel_size=1)
            for _ in range(len_clip)
            ])

        # input proj
        self.temporal_shuffle = nn.ModuleList([
            nn.Sequential(
                TemporalShuffle(),
                Conv(out_dim * len_clip, out_dim, k=1, g=out_dim, act_type=None, norm_type=None),
                Conv(out_dim, out_dim, k=3, p=1, act_type=None, norm_type='BN')
            )
        ])

        # CSAM
        self.csam = nn.ModuleList([CSAM(out_dim, dropout) for _ in range(depth)])

        # SSAM
        self.ssam = nn.ModuleList([SSAM(out_dim, dropout) for _ in range(depth)])


        self.channel_fuse_convs = nn.ModuleList([
            ChannelFuseConv(out_dim, out_dim, dropout=dropout)
            for _ in range(depth)
        ])


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        # input proj
        feats = [layer(x) for x, layer in zip(feats, self.input_proj)]

        for ts, ssam, csam, fuse in zip(self.temporal_shuffle, self.ssam, self.csam, self.channel_fuse_convs):
            kf_feat = feats[-1]
            # temporal shuffle: [B, C, H, W]
            x = ts(feats)

            # SSAM & CSAM
            x = ssam(x)
            x = csam(x)
            # channel fuse
            kf_feat = fuse(torch.cat([kf_feat, x], dim=1))

        return kf_feat
