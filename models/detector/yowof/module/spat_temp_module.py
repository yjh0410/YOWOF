import torch
import torch.nn as nn


# Adaptively Spatio-Temporal Feature Aggregator
class ASTFAggregator(nn.Module):
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
        feat = torch.sum([f * a for f, a in zip(feats, coeffs_list)], dim=1)

        return self.out_conv(feat)


# Adaptively Motion Feature Aggregator
class AMFAggregator(nn.Module):
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
        # List[K, B, C, H, W] -> [B, K, C, H, W]
        feat = torch.stack(feats, dim=1)
        # [B, K-1, C, H, W]
        diff_feat = feat[:, 1:, :, :, :] - feat[:, :-1, :, :, :]
        # [B, (K-1)C, H, W] -> [B, K-1, H, W]
        coeff_feat = self.coeff_conv(diff_feat)
        coeffs = self.coeff_act(coeff_feat)
        # [B, K-1, H, W] -> List[K-1, B, 1, H, W]
        coeffs_list = torch.chunk(coeffs, chunks=self.len_clip, dim=1)
        # adapatively aggregation
        feat = torch.sum([f * a for f, a in zip(diff_feat, coeffs_list)], dim=1)

        return self.out_conv(feat)

