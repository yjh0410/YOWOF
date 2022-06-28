import torch
import torch.nn as nn


# Motion Feature Aggregator
class MotionEncoder(nn.Module):
    def __init__(self, in_dim, len_clip=1):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.len_clip = len_clip
        # smooth layer
        self.smooth_layers = nn.ModuleList([
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1)
            for _ in range(len_clip - 1)
        ])
        # coefficient conv
        dims = in_dim * (len_clip - 1)
        self.coeff_convs = nn.Sequential(
            nn.Conv2d(dims, dims, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(dims, len_clip - 1, kernel_size=1),
            nn.Sigmoid()
        )
        # fusion conv
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_dim * 2, in_dim, kernel_size=1),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    
    def forward(self, feats):
        """
            feats: List(Tensor) [K, B, C, H, W]
        """
        motion_feats = []
        for idx in range(len(feats) - 1):
            feat_t1 = feats[idx]
            feat_t2 = feats[idx + 1]
            smooth = self.smooth_layers[idx]
            motion_feat = smooth(feat_t2) - feat_t1
            motion_feats.append(motion_feat)

        # List[K-1, B, C, H, W] -> [B, (K-1)C, H, W]
        motion_feats_ = torch.cat(motion_feats, dim=1)

        # [B, (K-1)C, H, W] -> [B, K-1, H, W]
        coeffs = self.coeff_convs(motion_feats_)

        # [B, K-1, H, W] -> List[K-1, B, 1, H, W]
        coeffs_list = torch.chunk(coeffs, chunks=self.len_clip, dim=1)

        # motion features
        mfeat = sum([f * a for f, a in zip(motion_feats, coeffs_list)]) / (self.len_clip - 1)

        # fuse motion features and key-frame features
        feat = torch.cat([feats[-1], mfeat], dim=1)
        feat = self.fuse_conv(feat)

        return feat
