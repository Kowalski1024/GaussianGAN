from itertools import pairwise

import rff
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.typing import Adj
from torch_geometric.nn import knn_graph

from training.v3.models import GeneratorConfig, GlobalPoolingConfig, LINKXConfig, PointGNNConfig
from training.v3.layers import (
    LINKX,
    GlobalPoolingLayer,
    PointGNNConv,
    trunc_exp,
)
from training.networks_stylegan2 import MappingNetwork
from training.v3.camera import extract_cameras
from training.v3.render import render, GaussianModel
from dataclasses import asdict


class CloudNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        out_channels: int,
        style_channels: int,
        layers_config: PointGNNConfig,
        pooling_config: GlobalPoolingConfig,
    ):
        super().__init__()
        self.synthethic_layers = 0
        self.styles_per_layer = layers_config["synthethic_layers"]
        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=in_channels // 2
        )

        self.global_pooling = GlobalPoolingLayer(
            **pooling_config, channels=out_channels
        )

        self.position_decoder = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(out_channels, out_channels // 2),
            nn.LeakyReLU(inplace=True),
            nn.Linear(out_channels // 2, 3),
            nn.Tanh(),
        )

        layers = [in_channels] + hidden_channels + [out_channels]
        self.graph_layers = nn.ModuleList()
        for in_c, out_c in pairwise(layers):
            self.synthethic_layers += layers_config["synthethic_layers"]
            self.graph_layers.append(
                PointGNNConv(
                    in_channels=in_c,
                    out_channels=out_c,
                    style_channels=style_channels,
                    **layers_config,
                )
            )

    def forward(
        self, pos: Tensor, edge_index: Adj, batch: Batch, styles: Tensor
    ) -> tuple[Tensor, Tensor]:
        x = self.encoder(pos)

        fragmented_styles = torch.split(styles, self.styles_per_layer)
        for graph_block, style in zip(self.graph_layers, fragmented_styles):
            x = graph_block(x, pos, edge_index, style)

        h = self.global_pooling(x, batch)
        x = torch.cat([x, h], dim=-1)

        return self.position_decoder(x) * 1.5, x


class FeatureNetwork(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        out_channels: int,
        style_channels: int,
        num_points: int,
        layers_config: LINKXConfig,
    ):
        super().__init__()
        self.synthethic_layers = 0
        self.styles_per_layer = layers_config["synthethic_layers"]

        layers = [in_channels] + hidden_channels + [out_channels]
        self.graph_layers = nn.ModuleList()
        for in_c, out_c in pairwise(layers):
            self.synthethic_layers += layers_config["synthethic_layers"]
            self.graph_layers.append(
                LINKX(
                    num_nodes=num_points,
                    in_channels=in_c,
                    hidden_channels=out_c,
                    out_channels=out_c,
                    style_channels=style_channels,
                    **layers_config,
                )
            )

    def forward(
        self, x: Tensor, pos: Tensor, edge_index: Adj, batch: Batch, styles: Tensor
    ) -> Tensor:
        x_ = x

        fragmented_styles = torch.split(styles, self.styles_per_layer)
        for graph_block, style in zip(self.graph_layers, fragmented_styles):
            x = graph_block(x, pos, edge_index, style)

        return torch.cat([x, x_], dim=-1)


class GaussianDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list[int],
        max_scale: float = 0.02,
        shs_degree: int = 3,
        use_rgb: bool = True,
        xyz_offset: bool = True,
        restrict_offset: bool = True,
    ):
        super(GaussianDecoder, self).__init__()
        self.max_scale = max_scale
        self.use_rgb = use_rgb
        self.xyz_offset = xyz_offset
        self.restrict_offset = restrict_offset

        self.feature_channels = {
            "scaling": 3,
            "rotation": 4,
            "opacity": 1,
            "shs": shs_degree,
            "xyz": 3,
        }

        layers = [in_channels] + hidden_channels
        self.mlp = nn.ModuleList()
        for in_c, out_c in pairwise(layers):
            self.mlp.append(nn.Sequential(nn.Linear(in_c, out_c), nn.LeakyReLU(inplace=True)))

        self.decoders = torch.nn.ModuleList()
        for key, channels in self.feature_channels.items():
            layer = nn.Linear(hidden_channels[-1], channels)

            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, -2.2)

            self.decoders.append(layer)

    def forward(self, x: Tensor, point_cloud: Tensor | None = None) -> GaussianModel:
        for layer in self.mlp:
            x = layer(x)

        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.decoders):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                v = torch.clamp(v, min=0, max=self.max_scale)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                if point_cloud is None:
                    v = torch.tanh(v)
                else:
                    if self.restrict_offset:
                        max_step = 1.2 / 32
                        v = (torch.sigmoid(v) - 0.5) * max_step
                    v = v + point_cloud if self.xyz_offset else point_cloud
            ret[k] = v

        return GaussianModel(**ret)


class GaussianGenerator(nn.Module):
    def __init__(self, generator_config: GeneratorConfig):
        super().__init__()
        self.config = generator_config
        noise_channels = self.config.noise_channels

        self.cloud_network = CloudNetwork(
            style_channels=noise_channels,
            **asdict(self.config.cloud_network),
        )
        self.feature_network = FeatureNetwork(
            style_channels=noise_channels,
            num_points=self.config.points,
            **asdict(self.config.feature_network),
        )
        self.decoder = GaussianDecoder(**asdict(self.config.decoder))

        self.synthethic_layers = (
            self.cloud_network.synthethic_layers
            + self.feature_network.synthethic_layers
        )

    def forward(self, styles: Tensor, sphere: Data) -> GaussianModel:
        pos, edge_index, batch = sphere.pos, sphere.edge_index, sphere.batch

        pos, cloud_features = self.cloud_network(pos, edge_index, batch, styles)
        features = self.feature_network(cloud_features, pos, edge_index, batch, styles)
        return self.decoder(features, pos)


class ImageGenerator(nn.Module):
    def __init__(
        self,
        generator_config: GeneratorConfig,
        image_size: int,
        background: tuple[int, int, int],
    ):
        super().__init__()
        self.image_size = image_size
        self.gaussian_generator = GaussianGenerator(generator_config)
        self.use_rgb = generator_config.decoder.use_rgb
        self.config = generator_config

        self.background: torch.Tensor
        self.sphere: torch.Tensor
        self.register_buffer(
            "background", torch.tensor(background, dtype=torch.float32)
        )
        self.register_buffer("sphere", self._fibonacci_sphere(generator_config.points))

    @staticmethod
    def _fibonacci_sphere(samples=1000):
        phi = torch.pi * (3.0 - torch.sqrt(torch.tensor(5.0)))  # golden angle in radians

        indices = torch.arange(samples)
        y = 1 - (indices / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = torch.sqrt(1 - y * y)  # radius at y

        theta = phi * indices  # golden angle increment

        x = torch.cos(theta) * radius
        z = torch.sin(theta) * radius

        points = torch.stack([x, y, z], dim=-1)

        return points

    def forward(self, ws: Tensor, cameras: Tensor, *args, **kwargs) -> Tensor:
        with torch.no_grad():
            poses = cameras[:, :16].view(-1, 4, 4)
            intrinsics = cameras[:, 16:25].view(-1, 3, 3) * 512
            cameras = extract_cameras(poses, intrinsics)

            edge_index = knn_graph(self.sphere, k=self.config.knn)
            sphere = Data(pos=self.sphere, edge_index=edge_index)

        images = torch.empty(
            len(ws), 3, self.image_size, self.image_size, device=ws.device
        )

        for i, (w, camera) in enumerate(zip(ws, cameras)):
            gaussian_model = self.gaussian_generator(w, sphere)
            img = render(camera, gaussian_model, self.background, use_rgb=self.use_rgb)
            images[i] = img

        return images
    


class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs = {}, **kwargs):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = 0
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = ImageGenerator(GeneratorConfig(), 128, (1,1,1))
        self.num_ws = self.synthesis.gaussian_generator.synthethic_layers
        self.mapping = MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, num_ws=self.num_ws, **mapping_kwargs)


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas)
