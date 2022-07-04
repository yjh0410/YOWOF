import torch
import torch.nn as nn
from ...basic.conv import Conv


# FFN
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0., act='relu'):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=1),
            nn.GELU() if act =='gelu' else nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, dim, kernel_size=1),
            nn.Dropout(dropout)
        )
        self.norm = nn.InstanceNorm1d(dim)

    def forward(self, x):
        """
            x: [B, C, N]
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
        self.ffn = FeedForward(in_dim, in_dim*4, dropout=dropout)

        # output
        self.out = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(in_dim, in_dim, kernel_size=1)
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
        y = self.ffn(y)

        # [B, C, N] -> [B, C, H, W]
        y = y.view(B, C, H, W)

        # output
        y = y + self.out(y)

        return y


# Spatio-Temporal Attention Module
class STAM(nn.Module):
    def __init__(self, in_dim, spatial_size, len_clip=1, dropout=0.1):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip
        self.scale = in_dim ** -0.5

        # spatio-temporal query: [C, N], N=HW
        self.st_query = nn.Embedding(in_dim, spatial_size[0]*spatial_size[1])

        # attention
        self.attend = nn.Softmax(dim = -1)
        self.ffn = FeedForward(in_dim, in_dim*4, dropout=dropout)

        # output
        self.out = nn.Sequential(
            nn.Dropout2d(dropout),
            nn.Conv2d(in_dim, in_dim, kernel_size=1)
        )


    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        # List[K, B, C, H, W] -> [B, KC, H, W] -> [B, KC, N]
        sfeats = torch.cat(feats, dim=1)
        B, _, H, W = sfeats.size()
        sfeats = sfeats.flatten(-2)

        # [C, N] -> [B, C, N] -> [B, C, N]
        st_query = self.st_query.weight.unsqueeze(0).repeat(B, 1, 1)

        # [B, C, N] x [B, N, KC] -> [B, C, KC]
        dots = torch.matmul(st_query, sfeats.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        # attention: [B, C, KC] x [B, KC, N] -> [B, C, N]
        st_query = torch.matmul(attn, sfeats)
        st_query = self.ffn(st_query)
 
        # [B, C, N] -> [B, C, H, W]
        st_query = st_query.view(B, -1, H, W)

        # output
        st_query = st_query + self.out(st_query)

        return st_query


# STM Encoder
class STMEncoder(nn.Module):
    def __init__(self, in_dim, spatial_size=[10, 10], len_clip=1, dropout=0.):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip

        # input proj
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=1)
            for _ in range(len_clip)
        ])

        # STAM
        self.stam = STAM(in_dim, spatial_size, len_clip, dropout)

        # CSAM
        self.csam = CSAM(in_dim, dropout)

        # fuse layer
        self.fuse_conv = nn.Sequential(
            Conv(in_dim * 2, in_dim, k=1, act_type='relu', norm_type='BN'),
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

        # STAM: [B, C, H, W]
        spattemp_feats = self.stam(input_feats)

        # fuse conv: [B, 2C, H, W] -> [B, C, H, W]
        feats = torch.cat([input_feats[-1], spattemp_feats], dim=1)
        feats = self.fuse_conv(feats)

        # CSAM: [B, C, H, W]
        feats = self.csam(feats)

        # output: [B, C, H, W]
        out_feats = self.out_conv(feats)

        return out_feats
