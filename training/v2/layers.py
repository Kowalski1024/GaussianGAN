import torch
from torch import nn
# from .layers import Attention, EdgeBlock, AdaptivePointNorm
from .gaussian import GaussianDecoder, render
from .camera import extract_cameras, generate_cameras
from training import networks_stylegan2 as stylegan2
from torch_geometric.nn import knn_graph, PointGNNConv, global_max_pool, InstanceNorm
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import rff


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = InstanceNorm(in_channel)
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
        )

        self.network = PointGNNConv(self.mlp_h, self.mlp_f, self.mlp_g, aggr=aggr)

    def forward(self, x, pos, edge_index):
        return self.network(x, pos, edge_index)


class SyntheticBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.gnn_conv = GNNConv(channels)
        self.adaptive_norm = AdaptivePointNorm(channels, 128)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, h, pos, edge_index, style):
        h = self.gnn_conv(h, pos, edge_index)
        h = self.leaky_relu(h)
        h = self.adaptive_norm(h, style)
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
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
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
        return self.tail(x) * 0.75, x


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
            nn.LeakyReLU(inplace=True),
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
        self.num_ws = 4096 * 2

        self.register_buffer("background", torch.ones(3, dtype=torch.float32))
        self.register_buffer("sphere", self._fibonacci_sphere(self.num_ws, 0.3))

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
            poses = camera[:, :16].view(-1, 4, 4)
            intrinsics = camera[:, 16:25].view(-1, 3, 3) * 512
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


class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs = {}, **kwargs):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = 0
        self.w_dim = 128
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = ImageGenerator(**kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = stylegan2.MappingNetwork(z_dim=z_dim, c_dim=0, w_dim=128, num_ws=self.num_ws, **mapping_kwargs)


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c)
        

