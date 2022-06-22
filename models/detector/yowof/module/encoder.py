import torch
import torch.nn as nn


# Temporal Feature Aggregator
class TemporalAggregator(nn.Module):
    def __init__(self, in_dim, len_clip=1):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.len_clip = len_clip
        self.coeff_conv = nn.Conv2d(
            in_dim * len_clip,
            len_clip,
            kernel_size=3,
            padding=1
            )
        self.coeff_act = nn.Softmax(dim=1)
        self.out_conv = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=3,
            padding=1
            )

    
    def forward(self, feats):
        # List[K, B, C, H, W] -> [B, KC, H, W]
        feat = torch.cat(feats, dim=1)
        # [B, KC, H, W] -> [B, K, H, W]
        coeff_feat = self.coeff_conv(feat)
        coeffs = self.coeff_act(coeff_feat)
        # [B, K, H, W] -> List[K, B, 1, H, W]
        coeffs_list = torch.chunk(coeffs, chunks=self.len_clip, dim=1)
        # adapatively aggregation
        feat = sum([f * a for f, a in zip(feats, coeffs_list)])

        return self.out_conv(feat)


# Motion Feature Aggregator
class MotionAggregator(nn.Module):
    def __init__(self, in_dim, len_clip=1):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.len_clip = len_clip
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
            for _ in range(len_clip - 1)
            ])
        self.coeff_conv = nn.Conv2d(
            in_dim * (len_clip - 1),
            len_clip,
            kernel_size=3,
            padding=1
            )
        self.coeff_act = nn.Softmax(dim=1)
        self.out_conv = nn.Conv2d(
            in_dim,
            in_dim,
            kernel_size=3,
            padding=1
            )

    
    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        # List[K, B, C, H, W] -> [B, K, C, H, W]
        feats = torch.stack(feats, dim=1)
        # [B, K-1, C, H, W]
        diff_feat = feats[:, 1:, :, :, :] - feats[:, :-1, :, :, :]
        # List[K-1, B, C, H, W]
        diff_feat = [layer(diff_feat[:, i, :, :, :]) for i, layer in enumerate(self.smooth_layers)]
        # [B, (K-1)C, H, W] -> [B, K-1, H, W]
        coeff_feat = self.coeff_conv(torch.cat(diff_feat, dim=1))
        coeffs = self.coeff_act(coeff_feat)
        # [B, K-1, H, W] -> List[K-1, B, 1, H, W]
        coeffs_list = torch.chunk(coeffs, chunks=self.len_clip, dim=1)
        # adapatively aggregation
        feat = sum([f * a for f, a in zip(diff_feat, coeffs_list)])

        return self.out_conv(feat)


# BasicEncoder
class BasicEncoder(nn.Module):
    def __init__(self, in_dim, len_clip=1):
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip
        
        self.temp_encoder = TemporalAggregator(in_dim=in_dim, len_clip=len_clip)
        self.motn_encoder = MotionAggregator(in_dim=in_dim, len_clip=len_clip)

        self.smooth_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_dim*3, in_dim, kernel_size=3, padding=1),
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True)
            )
         for _ in range(len_clip)
         ])


    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        # [B, C, H, W]
        temp_aggr = self.temp_encoder(feats)
        motn_aggr = self.motn_encoder(feats)

        # List[K, B, 3C, H, W] -> List[K, B, C, H, W]
        out_feats = [
            layer(torch.cat([x, temp_aggr, motn_aggr], dim=1))
            for x, layer in zip(feats, self.smooth_layers)
            ]

        return out_feats


# Temporal-Motino Encoder
class TempMotionEncoder(nn.Module):
    def __init__(self, in_dim, len_clip=1, depth=1):
        super().__init__()
        self.in_dim = in_dim
        self.len_clip = len_clip
        self.depth = depth

        # input proj
        self.input_proj = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=1)
            for _ in range(len_clip)
            ])

        # temp-motion encoder
        self.encoders = nn.ModuleList([
            BasicEncoder(in_dim=in_dim, len_clip=len_clip)
            for _ in range(depth)
        ])


    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        # input proj
        feats = [layer(x) for x, layer in zip(feats, self.input_proj)]

        # temp-motion encoder
        for encoder in self.encoders:
            feats = encoder(feats)

        return feats
