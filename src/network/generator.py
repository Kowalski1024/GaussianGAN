import torch
import torch.nn as nn
import rff

from torch_geometric.nn import global_max_pool, knn_graph, LayerNorm, global_mean_pool
from torch_geometric.data import Data

from .layers import SyntheticBlock, StyleLINKX
from .gaussian_decoder import GaussianDecoder
from src.utils.render import render
from src.utils.camera import extract_cameras, generate_cameras
from conf.main_config import GeneratorConfig


class CloudGenerator(nn.Module):
    def __init__(self, channels=256, style_channels=256, num_layers=2):
        super().__init__()
        self.style_channels = style_channels

        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=channels // 2
        )

        self.global_conv = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels // 2, 3),
            nn.Tanh(),
        )

        self.synthetic_blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.synthetic_blocks.append(
                SyntheticBlock(channels, channels, style_channels)
            )

    def forward(self, pos, edge_index, batch, style):
        """
        Args:
            pos: [N, 3]
            edge_index: [2, E]
            batch: [N]
            style: [N, z_channels]

        Returns:
            pos: [N, 3]
            x: [N, channels]
        """
        x = self.encoder(pos)

        for block in self.synthetic_blocks:
            x = block(x, pos, edge_index, style)

        h = global_max_pool(x, batch)
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)

        x = torch.cat([x, h], dim=-1)
        return self.tail(x), x


class GaussiansGenerator(nn.Module):
    def __init__(
        self,
        points,
        in_channels,
        hidden_channels,
        out_channels,
        style_channels,
        num_layers,
        **linkx_kwargs,
    ):
        super().__init__()

        self.synthetic_block = StyleLINKX(
            num_nodes=points,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            style_channels=style_channels,
            num_layers=num_layers,
            **linkx_kwargs,
        )
        # self.synthetic_block = SyntheticBlock(in_channels, out_channels, style_channels)

    def forward(self, x, pos, edge_index, batch, style):
        """
        Args:
            x: [N, in_channels]
            pos: [N, 3]
            edge_index: [2, E]
            batch: [N]
            style: [N, style_channels]

        Returns:
            x: [N, out_channels]
        """
        return self.synthetic_block(x, edge_index, style)


class ImageGenerator(nn.Module):
    def __init__(
        self,
        config: GeneratorConfig,
        image_resolution: int,
    ):
        super().__init__()
        self.config = config
        self.image_resolution = image_resolution

        self.point_encoder = CloudGenerator(
            channels=config.cloud_channels,
            num_layers=config.cloud_layers,
            style_channels=config.z_dim,
        )

        self.gaussians = GaussiansGenerator(
            points=config.points,
            in_channels=config.cloud_channels * 2,
            hidden_channels=config.gaussian_channels,
            out_channels=config.gaussian_channels,
            style_channels=config.z_dim,
            num_layers=config.gaussian_layers,
        )

        self.decoder = GaussianDecoder(
            in_channels=config.gaussian_channels,
            shs_degree=config.shs_degree,
            use_rgb=config.use_rgb,
            xyz_offset=config.xyz_offset,
            restrict_offset=config.restrict_offset,
        )

        self.style_head = nn.Sequential(
            nn.Linear(config.z_dim + 3, config.z_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(config.z_dim, config.z_dim),
            nn.LeakyReLU(inplace=True),
        )

        self.background: torch.Tensor
        self.register_buffer("background", torch.ones(3, dtype=torch.float32))

    def forward(self, zs, sphere, camera=None):
        with torch.no_grad():
            if camera is None:
                cameras = generate_cameras(len(zs), zs.device)
            else:
                poses = camera[:, :16].view(-1, 4, 4)
                fovx = camera[:, 16]
                fovy = camera[:, 17]
                cameras = extract_cameras(poses, fovx, fovy, self.image_resolution)

            sphere = Data(pos=sphere)
            pos, batch = sphere.pos, sphere.batch
            edge_index = knn_graph(pos, k=self.config.knn, batch=batch)

        images = []
        for camera, z in zip(cameras, zs):
            style = torch.cat([pos, z], dim=-1)
            style = self.style_head(style)

            point_cloud, points_features = self.point_encoder(
                pos, edge_index, batch, style
            )
            edge_index = knn_graph(point_cloud, k=self.config.knn, batch=batch)
            gaussian = self.gaussians(
                points_features, point_cloud, edge_index, batch, style
            )

            gaussian_model = self.decoder(gaussian, point_cloud)
            image = render(camera, gaussian_model, self.background, use_rgb=self.config.use_rgb)
            images.append(image)

        return torch.stack(images, dim=0).contiguous()
