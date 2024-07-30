from itertools import pairwise

import rff
import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.typing import Adj

from conf.layers import GlobalPoolingConfig, LINKXConfig, PointGNNConfig
from conf.models import GeneratorConfig
from src.network.layers import (
    LINKX,
    GlobalPoolingLayer,
    PointGNNConv,
    trunc_exp,
)
from src.utils.camera import Camera, extract_cameras
from src.utils.render import GaussianModel, render
from src.network.networks_stylegan2 import MappingNetwork


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
        self.styles_per_layer = layers_config.synthethic_layers
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
            self.synthethic_layers += layers_config.synthethic_layers
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
        self.styles_per_layer = layers_config.synthethic_layers

        layers = [in_channels] + hidden_channels + [out_channels]
        self.graph_layers = nn.ModuleList()
        for in_c, out_c in pairwise(layers):
            self.synthethic_layers += layers_config.synthethic_layers
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
            **self.config.cloud_network,
        )
        self.feature_network = FeatureNetwork(
            style_channels=noise_channels,
            num_points=self.config.points,
            **self.config.feature_network,
        )
        self.decoder = GaussianDecoder(**self.config.decoder)

        self.synthethic_layers = (
            self.cloud_network.synthethic_layers
            + self.feature_network.synthethic_layers
        )

        self.mapping_network = MappingNetwork(512, 0, 512, self.synthethic_layers)

    def forward(self, noise: Tensor, sphere: Data) -> GaussianModel:
        pos, edge_index, batch = sphere.pos, sphere.edge_index, sphere.batch

        noise = noise.unsqueeze(0)
        styles = self.mapping_network(noise, None).squeeze(0)
        # styles = styles.repeat(self.synthethic_layers, 1)

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

        self.background: torch.Tensor
        self.register_buffer(
            "background", torch.tensor(background, dtype=torch.float32)
        )

    def forward(self, noises: Tensor, sphere: Data, cameras: Tensor) -> Tensor:
        with torch.no_grad():
            pose = cameras[:, :16].reshape(-1, 4, 4)
            fovx = cameras[:, 16]
            fovy = cameras[:, 17]
            cameras = extract_cameras(pose, fovx, fovy, self.image_size)

        images = torch.empty(
            len(noises), 3, self.image_size, self.image_size, device=noises.device
        )

        for i, (noise, camera) in enumerate(zip(noises, cameras)):
            gaussian_model = self.gaussian_generator(noise, sphere)
            img = render(camera, gaussian_model, self.background, use_rgb=self.use_rgb)
            images[i] = img

        return images
