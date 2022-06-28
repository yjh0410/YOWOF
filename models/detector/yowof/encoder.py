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

        # fusion conv
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_dim * len_clip, in_dim, kernel_size=1),
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_dim),
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
        mfeat = torch.cat(motion_feats, dim=1)

        # fuse motion features and key-frame features
        feat = torch.cat([feats[-1], mfeat], dim=1)
        feat = self.fuse_conv(feat)

        return feat
