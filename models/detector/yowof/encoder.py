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


# Spatial Self Attetion Module
class SSAM(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.scale = in_dim ** -0.5

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
        out = self.ffn(out) + x

        return out


class STCEncoder(nn.Module):
    def __init__(self, in_dim, en_dim, out_dim, len_clip=1, depth=1, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip

        # input proj
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_dim, out_dim, kernel_size=1)
            for _ in range(len_clip)
            ])

        self.group_conv = nn.Conv2d(out_dim * len_clip, out_dim, kernel_size=3, padding=1, groups=out_dim)


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        feats = [layer(feat) for feat, layer in zip(feats, self.input_proj)]
        # (List) [K, B, C, H, W] -> [B, K, C, H, W]
        x = torch.stack(feats, dim=1)
        B, K, C, H, W = x.size()

        # [B, K, C, H, W] -> [B, C, K, H, W]
        x = torch.transpose(x, 1, 2).contiguous()

        # flatten: [B, CK, H, W]
        x = x.view(B, -1, H, W)

        # [B, CK, H, W] -> [B, C, H, W]
        x = self.group_conv(x)

        return x
