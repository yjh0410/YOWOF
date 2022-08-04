import math
import torch
import torch.nn as nn
from ...basic.conv import Conv2d, Conv3d


# Channel Self Attetion Module
class CSAM(nn.Module):
    """ Channel attention module """
    def __init__(self):
        super(CSAM, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1)
        key = x.view(B, C, -1).permute(0, 2, 1)
        value = x.view(B, C, -1)

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.view(B, C, H, W)

        # output
        out = self.gamma*out + x

        return out


# Spatial Self Attetion Module
class SSAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B x C x H x W )
            returns :
                out : attention value + input feature
                attention: B X HW X HW
        """
        B, C, H, W = x.size()
        # query / key / value
        query = x.view(B, C, -1).permute(0, 2, 1).contiguous()
        key = x.view(B, C, -1)
        value = x.view(B, C, -1).permute(0, 2, 1).contiguous()

        # attention matrix
        energy = torch.bmm(query, key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # attention
        out = torch.bmm(attention, value)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)

        # output
        out = self.gamma*out + x

        return out


class ChannelEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            CSAM(),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        """
            x: [B, CN, H, W]
        """
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x


class SpatialEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='', norm_type=''):
        super().__init__()
        self.fuse_convs = nn.Sequential(
            Conv2d(in_dim, out_dim, k=1, act_type=act_type, norm_type=norm_type),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SSAM(),
            Conv2d(out_dim, out_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            nn.Dropout(0.1, inplace=False),
            nn.Conv2d(out_dim, out_dim, kernel_size=1)
        )

    def forward(self, x):
        """
            x: [B, CN, H, W]
        """
        # [B, CN, H, W] -> [B, C, H, W]
        x = self.fuse_convs(x)

        return x


class TemporalEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, act_type='relu', norm_type='BN'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.stem = nn.Sequential(
            Conv3d(in_dim, out_dim, k=3, p=1, s=(2, 1, 1), act_type=act_type, norm_type=norm_type),
            Conv3d(out_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        )

        self.max_pool_1 = nn.MaxPool3d((2, 1, 1))
        self.layer_1 = nn.Sequential(
            Conv3d(out_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type),
            Conv3d(out_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        )

        self.max_pool_2 = nn.MaxPool3d((2, 1, 1))
        self.layer_2 = nn.Sequential(
            Conv3d(out_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type),
            Conv3d(out_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        )

        self.max_pool_3 = nn.MaxPool3d((2, 1, 1))
        self.layer_3 = nn.Sequential(
            Conv3d(out_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type),
            Conv3d(out_dim, out_dim, k=3, p=1, s=1, act_type=act_type, norm_type=norm_type)
        )


    def forward(self, x):
        """
        Input:
            x: (Tensor) [B, C_in, T, H, W]
        Output:
            y: (Tensor) [B, C_out, 1, H, W]
        """

        x = self.stem(x)

        x = self.max_pool_1(x)
        x = self.layer_1(x) + x

        x = self.max_pool_2(x)
        x = self.layer_2(x) + x

        x = self.max_pool_3(x)
        x = self.layer_3(x) + x

        y = torch.mean(x, dim=2)


        return y.squeeze(2) # [B, C_out, H, W]
