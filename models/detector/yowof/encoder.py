import torch
import torch.nn as nn
from ...basic.conv import Conv


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., act='relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU() if act =='gelu' else nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
            x: [B, N, C]
        """
        return self.norm(self.net(x)) + x


# Channel Self Attetion Module
class CSAM(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.scale = in_dim ** -0.5

        # query, key, value
        self.qkv_conv = nn.Conv2d(in_dim, 3*in_dim, kernel_size=1, bias=False)

        # attention
        self.attend = nn.Softmax(dim = -1)

        # output
        self.out = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.InstanceNorm2d(in_dim)
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

        # [B, C, N] -> [B, C, H, W]
        y = y.view(B, C, H, W)

        # output
        y = y + self.out(y)

        return y


# Spatial Self Attetion Module
class SSAM(nn.Module):
    def __init__(self, in_dim, dropout=0.1):
        super().__init__()
        self.in_dim = in_dim
        self.scale = in_dim ** -0.5

        # query, key, value
        self.qkv = nn.Linear(in_dim, 3*in_dim, bias=False)

        # attention
        self.attend = nn.Softmax(dim = -1)
        self.ffn = FeedForward(in_dim, in_dim*4, dropout=dropout)

        # output
        self.out = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(in_dim)
        )

    def forward(self, x):
        """
            x: (Tensor) [B, C, H, W]
        """
        B, C, H, W = x.size()
        # [B, C, H, W] -> [B, C, N] -> [B, N, C]
        x = x.flatten(-2).permute(0, 2, 1).contiguous()

        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, chunks=3, dim=-1)

        # [B, N, C] x [B, C, N] -> [B, N, N]
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        # attention: [B, N, N] x [B, N, C] -> [B, N, C]
        attn = self.attend(dots)
        y = torch.matmul(attn, v)
        y = self.ffn(y)

        # output
        y = y + self.out(y)

        # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        y = y.permute(0, 2, 1).contiguous().view(B, C, H, W)

        return y


# Temporal Self Attetion Module
class TSAM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.in_dim = in_dim

        # query, key, value
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
        )
        self.attend = nn.Sigmoid()

        # output
        self.out = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
            x: (Tensor) [B, KC, H, W]
        """
        # [B, KC, H, W] -> [B, KC, 1, 1]
        attn = self.attend(self.attn(x))
        x = x * attn

        return x + self.out(x)


# STC Encoder
class STCEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, len_clip=1, dropout=0., depth=1):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip
        self.out_dim = out_dim


        self.input_proj = nn.Conv2d(in_dim * len_clip, out_dim, kernel_size=1)

        # TSAM
        self.tsam = nn.ModuleList([
            TSAM(out_dim)
            for _ in range(depth)
        ])

        # SSAM
        self.ssam = nn.ModuleList([
            SSAM(out_dim, dropout)
            for _ in range(depth)
        ])

        # CSAM
        self.csam = nn.ModuleList([
            CSAM(out_dim, dropout)
            for _ in range(depth)
        ])

        # fuse SSAM & CSAM output
        self.fuse_convs = nn.ModuleList([
            Conv(out_dim*2, out_dim, k=1, act_type='relu', norm_type='BN')
            for _ in range(depth)
        ])

        # output
        self.output = Conv(out_dim, out_dim, k=3, p=1, act_type='relu', norm_type='BN')


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        # (List)[K, B, C, H, W] -> [B, KC, H, W]
        input_feats = torch.cat(feats, dim=1)
        # [B, KC, H, W] -> [B, C, H, W]
        x = self.input_proj(input_feats)

        for tsam, ssam, csam, fuse in zip(self.tsam, self.ssam, self.csam, self.fuse_convs):
            # TSAM
            x = tsam(x)
            # SSAM & CSAM
            x1 = ssam(x)
            x2 = csam(x)
            x = fuse(torch.cat([x1, x2], dim=1))

        return self.output(x)
