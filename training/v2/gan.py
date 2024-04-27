import torch
from torch import nn
# from .layers import Attention, EdgeBlock, AdaptivePointNorm
from .gaussian import GaussianDecoder, render
from .camera import extract_cameras, generate_cameras
from training import networks_stylegan2 as stylegan2
from torch_geometric.nn import knn_graph, PointGNNConv, global_max_pool, InstanceNorm
from torch_geometric.nn.norm import LayerNorm
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch
import rff
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch.nn import BatchNorm1d
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.linkx import SparseLinear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import InstanceNorm
from torch_utils.ops.fma import fma
import numpy as np



POINTS = 8192

def modulated_linear(
    x,  # Input tensor of shape [batch_size, in_features].
    weight,  # Weight tensor of shape [out_features, in_features].
    styles,  # Modulation coefficients of shape [features].
    noise=None,  # Optional noise tensor to add to the output activations.
    demodulate=True,  # Apply weight demodulation?
):
    _, in_features = weight.shape
    styles = styles[0]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (
            1 / np.sqrt(in_features) / weight.norm(float("inf"), dim=1, keepdim=True)
        )  # max_I
        styles = styles / styles.norm(float("inf"), keepdim=True)  # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate:
        w = weight.unsqueeze(0)  # [NOI]
        w = w * styles.unsqueeze(0).unsqueeze(0)  # [1, 1, I]
        dcoefs = (w.square().sum(dim=2) + 1e-8).rsqrt()  # [NO]
        w = w * dcoefs.unsqueeze(0)  # [1, NO, I]

    # Execute by scaling the activations before and after the linear layer.
    x = x * styles.to(x.dtype).unsqueeze(0)
    x = torch.nn.functional.linear(x, weight.to(x.dtype))
    if demodulate and noise is not None:
        dcoefs = dcoefs.expand_as(x)
        x = fma(x, dcoefs.to(x.dtype), noise.to(x.dtype))
    elif demodulate:
        dcoefs = dcoefs.expand_as(x)
        x = x * dcoefs.to(x.dtype)
    elif noise is not None:
        x = x.add_(noise.to(x.dtype))
    return x


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = InstanceNorm(in_channel)
        self.affine = nn.Linear(style_dim, in_channel * 2)

        self.affine.weight.data.normal_()
        self.affine.bias.data.zero_()

        self.affine.bias.data[:in_channel] = 1
        self.affine.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.affine(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = (gamma * out).add_(beta)

        return out
    

class StyleLinearLayer(nn.Module):
    def __init__(self, in_dim, w_dim, out_dim, noise=True):
        super().__init__()
        self.in_dim = in_dim
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.noise = noise
        self.activation = nn.LeakyReLU(inplace=True)

        self.linear = nn.Linear(in_dim, out_dim)
        self.adain = AdaptivePointNorm(out_dim, w_dim)
        self.noise_strength = nn.Parameter(torch.zeros(1)) if noise else None
        self.affine = nn.Linear(w_dim, in_dim)

    def forward(self, x, w):
        x = self.linear(x)

        if self.noise:
            noise = torch.randn(1, x.size(1), device=x.device)
            noise = noise * self.noise_strength
            x.add_(noise)

        x = self.adain(x, w)

        return self.activation(x)

    
# class StyleLinearLayer(nn.Module):
#     def __init__(self, in_dim, w_dim, out_dim, noise=True):
#         super().__init__()
#         self.in_dim = in_dim
#         self.w_dim = w_dim
#         self.out_dim = out_dim
#         self.noise = noise
#         self.activation = nn.LeakyReLU(inplace=True)

#         self.weight = nn.Parameter(torch.randn(out_dim, in_dim))
#         self.bias = nn.Parameter(torch.zeros(out_dim))
#         self.noise_strength = nn.Parameter(torch.zeros(1)) if noise else None
#         self.affine = nn.Linear(w_dim, in_dim)

#     def forward(self, x, w):
#         if self.noise:
#             noise = torch.randn(1, x.size(1), device=x.device)
#             noise = noise * self.noise_strength
#         else:
#             noise = None

#         x = modulated_linear(x, self.weight, w, noise)

#         return self.activation(x.add_(self.bias))
    

class StyleMLP(nn.Module):
    def __init__(
        self,
        channels,
        w_dim,
        noise=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(channels[:-1], channels[1:]):
            self.layers.append(
                StyleLinearLayer(in_dim, w_dim, out_dim, noise)
            )

    def forward(self, x, w):
        for layer in self.layers:
            x = layer(x, w)
        return x


class StyleLINKX(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        w_dim: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.w_dim = w_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers
        self.num_node_layers = num_node_layers
        self.num_layers = num_layers

        self.edge_lin = SparseLinear(num_nodes, hidden_channels)

        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0.0, act_first=True)
        else:
            self.edge_norm = None
            self.edge_mlp = None

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = StyleMLP(channels, w_dim)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = StyleMLP(channels, w_dim)
        self.num_ws = num_node_layers + num_layers

    def forward(
        self,
        x: OptTensor,
        edge_index: Adj,
        w: Tensor,
    ) -> Tensor:
        out = self.edge_lin(edge_index, None)

        if self.edge_norm is not None and self.edge_mlp is not None:
            out = out.relu_()
            out = self.edge_norm(out)
            out = self.edge_mlp(out)

        out = out + self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x, w)
            out = out + x
            out = out + self.cat_lin2(x)

        return self.final_mlp(out.relu_(), w)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_nodes={self.num_nodes}, "
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_edge_layers={self.num_edge_layers}, "
            f"num_node_layers={self.num_node_layers})"
        )


class GNNConv(nn.Module):
    def __init__(self, channels, aggr="sum"):
        super().__init__()

        self.mlp_h = nn.Sequential(
            nn.Linear(channels, channels),
            # LayerNorm(channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, 3),
            nn.Tanh(),
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(channels + 3, channels),
            # LayerNorm(channels),
            nn.LeakyReLU(inplace=True),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(channels, channels),
            # LayerNorm(channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, channels),
        )

        self.network = PointGNNConv(self.mlp_h, self.mlp_f, self.mlp_g, aggr=aggr)

    def forward(self, x, pos, edge_index):
        return self.network(x, pos, edge_index)


class SyntheticBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.add_noise = True

        self.gnn_conv = GNNConv(channels)
        self.adaptive_norm = AdaptivePointNorm(channels, 128)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        if self.add_noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

    def forward(self, h, pos, edge_index, style):
        h = self.gnn_conv(h, pos, edge_index)

        if self.add_noise:
            noise = torch.randn_like(h) * self.noise_strength
            h = h + noise

        h = self.leaky_relu(h)
        h = self.adaptive_norm(h, style)
        return h


class CloudGenerator(nn.Module):
    def __init__(self, channels=128, z_dim=128, blocks=2):
        super().__init__()
        self.z_dim = z_dim
        self.blocks = blocks

        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=channels // 2
        )

        self.global_conv = nn.Sequential(
            nn.Linear(128, 128),
            # LayerNorm(128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            # LayerNorm(128),
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

        self.synthetic_block = StyleLINKX(POINTS, 128, 128, 128, 128, 2)

    def forward(self, pos, edge_index, batch, style):
        x = self.encoder(pos)

        x = self.synthetic_block(x, edge_index, style)

        h = global_max_pool(x, batch)
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)

        x = torch.cat([x, h], dim=-1)
        return self.tail(x) * 0.75, x


class GaussiansGenerator(nn.Module):
    def __init__(self, channels=128, z_dim=128, blocks=1):
        super().__init__()
        self.z_dim = z_dim

        self.feature_encoder = nn.Sequential(
            nn.Linear(256, channels),
            nn.LeakyReLU(inplace=True),
        )

        self.synthetic_block = StyleLINKX(POINTS, 128, 128, 128, 128, 3)

    def forward(self, h, pos, edge_index, batch, style):
        h = self.feature_encoder(h)

        h = self.synthetic_block(h, edge_index, style)

        return h


class ImageGenerator(nn.Module):
    def __init__(self, z_dim=128, **kwargs):
        super(ImageGenerator, self).__init__()
        c = 128
        self.point_encoder = CloudGenerator(z_dim=z_dim)
        self.gaussians = GaussiansGenerator(c, z_dim=z_dim)
        self.decoder = GaussianDecoder(c)
        self.num_ws = POINTS
        self.z_dim = z_dim

        self.style = nn.Sequential(
            nn.Linear(z_dim + 3, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 128),
            nn.LeakyReLU(inplace=True),
        )

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

    def forward(self, ws, camera=None, *args, **kwargs):
        with torch.no_grad():
            poses = camera[:, :16].view(-1, 4, 4)
            intrinsics = camera[:, 16:25].view(-1, 3, 3) * 512
            cameras = extract_cameras(poses, intrinsics)

            sphere = Data(pos=self.sphere)
            pos, batch = sphere.pos, sphere.batch
            edge_index = knn_graph(sphere.pos, k=8, batch=sphere.batch)

        images = []
        for camera, z in zip(cameras, ws):
            style = torch.cat([z, pos], dim=-1)
            style = self.style(style)

            point_cloud, points_features = self.point_encoder(pos, edge_index, batch, style)
            gaussian = self.gaussians(points_features, point_cloud, edge_index, batch, style)

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
        self.mapping = stylegan2.MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, num_ws=self.num_ws, **mapping_kwargs)


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c)
        

