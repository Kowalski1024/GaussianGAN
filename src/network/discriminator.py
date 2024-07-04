import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from conf.main_config import DiscriminatorConfig
from src.network.layers import MappingNetwork


class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(DiscriminatorBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if x.size(2) % 2 != 0:
            x = F.pad(x, (0, 1, 0, 1))
        y = self.skip(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return y.add_(x)


class MinibatchStdLayer(nn.Module):
    def __init__(self, group_size=4, num_channels=1):
        super(MinibatchStdLayer, self).__init__()
        self.group_size = group_size
        self.num_channels = num_channels

    def forward(self, x):
        N, C, H, W = x.shape
        G = min(self.group_size, N) if self.group_size is not None else N
        F = self.num_channels
        c = C // F

        y = x.reshape(
            G, -1, F, c, H, W
        )  # [GnFcHW] Split minibatch N into n groups of size G, and channels C into F groups of size c.
        y = y - y.mean(dim=0)  # [GnFcHW] Subtract mean over group.
        y = y.square().mean(dim=0)  # [nFcHW]  Calc variance over group.
        y = (y + 1e-8).sqrt()  # [nFcHW]  Calc stddev over group.
        y = y.mean(dim=[2, 3, 4])  # [nF]     Take average over channels and pixels.
        y = y.reshape(-1, F, 1, 1)  # [nF11]   Add missing dimensions.
        y = y.repeat(G, 1, H, W)  # [NFHW]   Replicate over group and pixels.
        x = torch.cat([x, y], dim=1)  # [NCHW]   Append to input as new channels.
        return x


class DiscriminatorEpilogue(nn.Module):
    def __init__(self, in_channels, cmap_channels, out_channels=1, resolution=4):
        super(DiscriminatorEpilogue, self).__init__()
        self.minibatch_std = MinibatchStdLayer()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_channels * resolution**2 + cmap_channels, in_channels)
        self.out = nn.Linear(in_channels, out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x, cmap):
        x = self.minibatch_std(x)
        x = self.conv(x)
        x = self.act(x)
        x = x.flatten(1)
        x = torch.cat([x, cmap], dim=1)
        x = self.fc(x)
        x = self.act(x)
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, config: DiscriminatorConfig, image_resolution: int=128):
        super(Discriminator, self).__init__()
        self.block_resolutions = [2**i for i in range(int(np.log2(image_resolution)), 2, -1)]
        self.channels_dict = {
            res: min(8192 // res, 128) for res in self.block_resolutions + [4]
        }
        self.epilogue = DiscriminatorEpilogue(self.channels_dict[4], config.mapping_out_channels, resolution=4)

        self.mapping = MappingNetwork(
            in_dim=config.mappping_in_channels,
            hidden_dim=config.mapping_hidden_channels,
            out_dim=config.mapping_out_channels,
            num_layers=config.mapping_layers
        )

        self.blocks = nn.ModuleList()
        for res in self.block_resolutions:
            in_channels = 3 if res == 128 else self.channels_dict[res]
            hidden_channels = self.channels_dict[res]
            out_channels = self.channels_dict[res // 2]
            self.blocks.append(
                DiscriminatorBlock(in_channels, hidden_channels, out_channels)
            )

    def forward(self, x, c):
        for block in self.blocks:
            x = block(x)
        cmap = self.mapping(c)
        return self.epilogue(x, cmap)