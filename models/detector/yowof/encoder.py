from turtle import forward
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

        self.group_conv = nn.Sequential(
            Conv(out_dim * len_clip, out_dim, k=1, g=out_dim, act_type=None, norm_type=None),
            Conv(out_dim, out_dim, k=3, p=1, act_type='relu', norm_type='BN')
        )

        self.fuse = nn.Sequential(
            Conv(out_dim*2, out_dim, k=1, act_type='relu', norm_type='BN'),
            Conv(out_dim, out_dim, k=3, p=1, act_type='relu', norm_type='BN'),
            CSAM(out_dim, dropout),
            Conv(out_dim, out_dim, k=3, p=1, act_type='relu', norm_type='BN')
        )


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        feats = [layer(feat) for feat, layer in zip(feats, self.input_proj)]
        kf_feat = feats[-1]
        # (List) [K, B, C, H, W] -> [B, K, C, H, W]
        x = torch.stack(feats, dim=1)
        B, K, C, H, W = x.size()

        # [B, K, C, H, W] -> [B, C, K, H, W]
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten: [B, CK, H, W]
        x = x.view(B, -1, H, W)

        # [B, CK, H, W] -> [B, C, H, W]
        x = self.group_conv(x)

        # fuse
        x = self.fuse(torch.cat([kf_feat, x], dim=1))

        return x
