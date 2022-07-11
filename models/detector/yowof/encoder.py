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


class TemporalGroupConv(nn.Module):
    def __init__(self, in_dim, in_len_clip=1, out_len_clip=1):
        super().__init__()
        self.group_convs = nn.ModuleList([
            nn.Sequential(
                Conv(in_dim * in_len_clip, in_dim, k=1, g=in_dim, act_type=None, norm_type=None),
                Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')
            )
            for _ in range(out_len_clip)
        ])

    def forward(self, x):
        """
            x: [B, CK, H, W]
        """
        feats = [layer(x) for layer in self.group_convs]

        return feats


class FuseConv(nn.Module):
    def __init__(self, in_dim, N=1, dropout=0.):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv(in_dim*N, in_dim, k=1, act_type='relu', norm_type='BN'),
            Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN'),
            CSAM(in_dim, dropout),
            Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')
        )

    def forward(self, x):
        """
            x: [B, CN, H, W]
        """
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x


class STCEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, len_clip=1, depth=1, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip

        # input proj
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, kernel_size=1)
            for _ in range(len_clip)
            ])

        self.temp_group_convs = nn.ModuleList([
            TemporalGroupConv(out_dim, len_clip // 2**(i), len_clip // 2**(i+1))
            for i in range(int(math.log2(len_clip)))
        ])

        self.channel_fuse_convs = nn.ModuleList([
            FuseConv(out_dim, N=len_clip // 2**(i+1) + 1, dropout=dropout)
            for i in range(int(math.log2(len_clip)))
        ])


    def temporal_shuffle(self, feats):
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


    def temporal_unshuffle(self, feats):
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


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        # print(feats[0].shape)
        feats = [layer(feat) for feat, layer in zip(feats, self.input_proj)]
        kf_feat = feats[-1]

        for tgconv, cfconv in zip(self.temp_group_convs, self.channel_fuse_convs):
            # [B, K, C, H, W] -> [B, CK, H, W]
            x = self.temporal_shuffle(feats)
            # List[K, B, C, H, W]
            feats = tgconv(x)
            kf_feat = torch.cat([kf_feat, *feats], dim=1)
            kf_feat = cfconv(kf_feat)

        return kf_feat
