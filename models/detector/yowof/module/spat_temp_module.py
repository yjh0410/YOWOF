import torch
import torch.nn as nn

from ....basic.conv import Conv


# Spatio-Temporal Feature Aggregator
class STFAggregator(nn.Module):
    def __init__(self, in_dim, K):
        """
            in_dim: (Int) -> dim of single feature
            K: (Int) -> length of video clip
        """
        super().__init__()
        self.spatio_conv = Conv(in_dim * K, )