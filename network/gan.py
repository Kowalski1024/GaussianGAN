import torch
from torch import nn
from network.gaussian import GaussianDecoder, render
from network.camera import extract_cameras, generate_cameras
import numpy as np
from torch.nn import functional as F
from torch_geometric.nn import knn_graph, PointGNNConv, global_max_pool
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import rff


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm1d(in_channel)
        self.style = nn.Linear(style_dim, in_channel * 2)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = (gamma * out).add_(beta)

        return out


class GNNConv(nn.Module):
    def __init__(self, channels, aggr="sum"):
        super().__init__()

        self.mlp_h = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 3),
            nn.Tanh(),
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.ReLU(inplace=True),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.ReLU(inplace=True),
        )

        self.network = PointGNNConv(self.mlp_h, self.mlp_f, self.mlp_g, aggr=aggr)

    def forward(self, x, pos, edge_index):
        return self.network(x, pos, edge_index)


class SyntheticBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gnn_conv_1 = GNNConv(channels)
        self.gnn_conv_2 = GNNConv(channels)
        self.adaptive_norm_1 = AdaptivePointNorm(channels, 128)
        self.adaptive_norm_2 = AdaptivePointNorm(channels, 128)
        self.leaky_relu = nn.LeakyReLU(inplace=True)

    def forward(self, h, pos, edge_index, style):
        h = self.gnn_conv_1(h, pos, edge_index)
        h = self.adaptive_norm_1(h, style)
        h = self.leaky_relu(h)

        h = self.gnn_conv_2(h, pos, edge_index)
        h = self.adaptive_norm_2(h, style)
        h = self.leaky_relu(h)

        return h


class CloudGenerator(nn.Module):
    def __init__(self, channels=128, z_dim=128, blocks=2):
        super().__init__()
        self.z_dim = z_dim

        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=channels // 2
        )

        self.style = nn.Sequential(
            nn.Linear(z_dim + 3, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
        )

        self.global_conv = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 3),
            nn.Tanh(),
        )

        self.synthetic_blocks = nn.ModuleList()
        for _ in range(blocks):
            self.synthetic_blocks.append(SyntheticBlock(channels))

    def forward(self, pos, edge_index, batch, z):
        z = z.view(-1, self.z_dim)
        style = torch.cat([z, pos], dim=-1)
        style = self.style(style)

        x = self.encoder(pos)

        for block in self.synthetic_blocks:
            x = block(x, pos, edge_index, style)

        h = global_max_pool(x, batch)
        h = self.global_conv(h)
        h = h[batch]

        x = torch.cat([x, h], dim=-1)
        return self.tail(x), x


class GaussiansGenerator(nn.Module):
    def __init__(self, channels=128, z_dim=128, blocks=1):
        super().__init__()
        self.z_dim = z_dim

        self.style = nn.Sequential(
            nn.Linear(z_dim + 3, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
        )

        self.feature_encoder = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.synthetic_blocks = nn.ModuleList()
        for _ in range(blocks):
            self.synthetic_blocks.append(SyntheticBlock(channels))

    def forward(self, h, pos, edge_index, batch, z):
        z = z.view(-1, self.z_dim)
        style = torch.cat([z, pos], dim=-1)
        style = self.style(style)

        h = self.feature_encoder(h)

        for block in self.synthetic_blocks:
            h = block(h, pos, edge_index, style)

        return h


class ImageGenerator(nn.Module):
    def __init__(self, z_dim=128, **kwargs):
        super(ImageGenerator, self).__init__()
        self.point_encoder = CloudGenerator(z_dim=z_dim)
        self.gaussians = GaussiansGenerator(z_dim=z_dim)
        self.decoder = GaussianDecoder(128)

        self.register_buffer("background", torch.ones(3, dtype=torch.float32))
        self.register_buffer("sphere", self._fibonacci_sphere(1024, 0.3))

    @staticmethod
    def _fibonacci_sphere(samples=1000, scale=1.0):
        phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))

        indices = torch.arange(samples)
        y = 1 - (indices / float(samples - 1)) * 2
        radius = torch.sqrt(1 - y * y)

        theta = phi * indices

        x = torch.cos(theta) * radius
        z = torch.sin(theta) * radius

        points = torch.stack([x, y, z], dim=-1)

        return points * scale

    def forward(self, z, camera=None, *args, **kwargs):
        B, _, _ = z.size()
        with torch.no_grad():
            if camera is None:
                cameras = generate_cameras(B, z.device)
            else:
                poses = camera[:, :16].view(-1, 4, 4).detach()
                intrinsics = camera[:, 16:25].view(-1, 3, 3).detach() * 512
                cameras = extract_cameras(poses, intrinsics)

            spheres = Batch.from_data_list([Data(pos=self.sphere) for _ in range(B)])
            pos, batch = spheres.pos, spheres.batch
            edge_index = knn_graph(spheres.pos, k=8, batch=spheres.batch)

        point_clouds, points_features = self.point_encoder(pos, edge_index, batch, z)
        gaussians = self.gaussians(points_features, point_clouds, edge_index, batch, z)

        gaussians = to_dense_batch(gaussians, batch)[0]
        point_clouds = to_dense_batch(point_clouds, batch)[0]

        images = []
        for gaussian, camera, point_cloud in zip(gaussians, cameras, point_clouds):
            gaussian_model = self.decoder(gaussian, point_cloud)
            image = render(camera, gaussian_model, self.background, use_rgb=True)
            images.append(image)

        return torch.stack(images, dim=0).contiguous()


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
    def __init__(self, in_channels, out_channels=1, resolution=4):
        super(DiscriminatorEpilogue, self).__init__()
        self.minibatch_std = MinibatchStdLayer()
        self.conv = nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(in_channels * resolution**2, in_channels)
        self.out = nn.Linear(in_channels, out_channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.minibatch_std(x)
        x = self.conv(x)
        x = self.act(x)
        x = self.fc(x.flatten(1))
        x = self.act(x)
        x = self.out(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_size=128):
        super(Discriminator, self).__init__()
        self.block_resolutions = [2**i for i in range(int(np.log2(img_size)), 2, -1)]
        self.channels_dict = {
            res: min(8192 // res, 512) for res in self.block_resolutions + [4]
        }
        self.epilogue = DiscriminatorEpilogue(self.channels_dict[4], resolution=4)
        self.blocks = nn.ModuleList()
        for res in self.block_resolutions:
            in_channels = 3 if res == 128 else self.channels_dict[res]
            hidden_channels = self.channels_dict[res]
            out_channels = self.channels_dict[res // 2]
            self.blocks.append(
                DiscriminatorBlock(in_channels, hidden_channels, out_channels)
            )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.epilogue(x)
