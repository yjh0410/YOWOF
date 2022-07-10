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


# STC Encoder
class STCEncoder(nn.Module):
    def __init__(self, in_dim, en_dim, out_dim, len_clip=1, depth=1, dropout=0.):
        """
            in_dim: (Int) -> dim of single feature
            len_clip: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip

        # input proj

        # Spatio-Temporal
        self.st_input_proj = nn.Conv2d(in_dim * len_clip, en_dim, kernel_size=1)
        self.st_ssam = nn.ModuleList([SSAM(en_dim, dropout) for _ in range(depth)])
        self.st_csam = nn.ModuleList([CSAM(en_dim, dropout) for _ in range(depth)])
        self.st_smooth = nn.ModuleList([
            Conv(en_dim, en_dim, k=3, p=1, act_type='relu', norm_type='BN')
            for _ in range(depth)
            ])

        # Motion
        self.mt_input_proj = nn.Conv2d(in_dim * (len_clip - 1), en_dim, kernel_size=1)
        self.mt_ssam = nn.ModuleList([SSAM(en_dim, dropout) for _ in range(depth)])
        self.mt_csam = nn.ModuleList([CSAM(en_dim, dropout) for _ in range(depth)])
        self.mt_smooth = nn.ModuleList([
            Conv(en_dim, en_dim, k=3, p=1, act_type='relu', norm_type='BN')
            for _ in range(depth)
            ])

        # fuse
        self.fuse = nn.Sequential(
                Conv(in_dim + en_dim * 2, out_dim, k=1, act_type='relu', norm_type='BN'),
                Conv(out_dim, out_dim, k=3, p=1, act_type='relu', norm_type='BN'),
                CSAM(out_dim, dropout)
                )


    def forward(self, feats):
        """
            feats: (List) [K, B, C, H, W]
        """
        # key frame feature
        kf_feats = feats[-1]
        # (List)[K, B, C, H, W] -> [B, KC, H, W]
        x_st = torch.cat(feats, dim=1)
        # [B, KC, H, W] -> [B, C, H, W]
        x_st = self.st_input_proj(x_st)

        # (List)[K, B, C, H, W] -> [K, B, C, H, W]
        x_mt = torch.stack(feats)
        x_mt = x_mt[1:] - x_mt[:-1]
        x_mt = x_mt.transpose(1, 0).flatten(1,2)
        x_mt = self.mt_input_proj(x_mt)

        for st_ssam, st_csam, st_smooth,\
            mt_ssam, mt_csam, mt_smooth in zip(self.st_ssam, self.st_csam, self.st_smooth,\
                                               self.mt_ssam, self.mt_csam, self.mt_smooth):
            # spatio-temporal
            x_s = st_ssam(x_st)
            x_c = st_csam(x_st)
            x_st = st_smooth(x_s + x_c)

            # motion
            m_s = mt_ssam(x_mt)
            m_c = mt_csam(x_mt)
            x_mt = mt_smooth(m_s + m_c)

        # fuse
        kf_feats = self.fuse(torch.cat([kf_feats, x_st, x_mt], dim=1))

        return kf_feats
