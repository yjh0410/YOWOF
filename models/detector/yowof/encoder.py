import torch
import torch.nn as nn
from ...basic.conv import Conv


# Channel Self Attetion Module
class CSAM(nn.Module):
    def __init__(self, in_dim, dropout=0.):
        super().__init__()
        self.in_dim = in_dim
        self.scale = in_dim ** -0.5

        self.gamma = nn.Parameter(torch.zeros(1))

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
        out = self.ffn(out) * self.gamma + x

        return out


# Channel Cross Attetion Module
class CCAM(nn.Module):
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


    def forward(self, query, key, value):
        """
            query: (Tensor) [B, C, H, W]
            key:   (Tensor) [B, C, H, W]
            value: (Tensor) [B, C, H, W]
        """
        B, C, H, W = query.size()

        # query: [B, C, N]
        q = query.view(B, C, -1)
        # key: [B, N, C]
        k = key.view(B, C, -1).transpose(-1, -2)
        # value: [B, C, N]
        v = value.view(B, C, -1)

        # self attention
        attn = self.attend(torch.matmul(q, k) * self.scale)
        out = torch.matmul(attn, v)

        # [B, C, N] -> [B, C, H, W]
        out = out.view(B, C, H, W)

        # output
        out = self.norm(out) * self.gamma + query

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


# Spatial Cross Attetion Module
class SCAM(nn.Module):
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


    def forward(self, query, key, value):
        """
            query: (Tensor) [B, C, H, W]
            key:   (Tensor) [B, C, H, W]
            value: (Tensor) [B, C, H, W]
        """
        B, C, H, W = query.size()

        # query: [B, N, C]
        q = query.view(B, C, -1).transpose(-1, -2)
        # key: [B, C, N]
        k = key.view(B, C, -1) 
        # value: [B, N, C]
        v = value.view(B, C, -1).transpose(-1, -2)

        # self attention: [B, N, N]
        attn = self.attend(torch.matmul(q, k) * self.scale)
        out = torch.matmul(attn, v)

        # [B, N, C] -> [B, C, N] -> [B, C, H, W]
        out = out.transpose(-1, -2).view(B, C, H, W)

        # output
        out = self.ffn(out) * self.gamma + query

        return out


# STC Encoder
class STCEncoder(nn.Module):
    def __init__(self, in_dim, len_clip=1, depth=1, dropout=0.):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip

        # input proj
        self.input_proj = nn.Conv2d(in_dim * len_clip, in_dim, kernel_size=1)

        # SSAM
        self.ssam = nn.ModuleList([SSAM(in_dim, dropout) for _ in range(depth)])

        # CSAM
        self.csam = nn.ModuleList([CSAM(in_dim, dropout) for _ in range(depth)])

        # SCAM
        self.scam = nn.ModuleList([SCAM(in_dim, dropout) for _ in range(depth)])

        # CCAM
        self.ccam = nn.ModuleList([CCAM(in_dim, dropout) for _ in range(depth)])

        self.smooth = nn.ModuleList([
                Conv(in_dim, in_dim, k=3, p=1, act_type='relu', norm_type='BN')
            for _ in range(depth)
        ])


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        # (List)[K, B, C, H, W] -> [B, KC, H, W]
        input_feats = torch.cat(feats, dim=1)
        # [B, KC, H, W] -> [B, C, H, W]
        x = self.input_proj(input_feats)
        kf_feats = feats[-1]

        for ssam, csam, scam, ccam, smooth in zip(self.ssam, self.csam, self.scam, self.ccam, self.smooth):
            # SSAM & CSAM
            x = ssam(x)
            x = csam(x)
            # SCAM & CCAM
            kf_feats = scam(kf_feats, x, x)
            kf_feats = ccam(kf_feats, x, x)
            # smooth
            kf_feats = smooth(kf_feats)

        return kf_feats
