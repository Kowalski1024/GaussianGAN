import torch
from torch import nn
# from .layers import Attention, EdgeBlock, AdaptivePointNorm
from .gaussian import GaussianDecoder, render
from .camera import extract_cameras, generate_cameras
from training import networks_stylegan2 as stylegan2
from torch_geometric.nn import knn_graph, PointGNNConv, global_max_pool, InstanceNorm, MessagePassing, global_mean_pool
import rff
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from torch import Tensor
from torch.nn import BatchNorm1d
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.linkx import SparseLinear
from torch_geometric.typing import Adj, OptTensor
from torch_geometric import nn as gnn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj
from torch import Tensor
from itertools import pairwise



POINTS = 8192 * 2

def fmm_modulate_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    styles: torch.Tensor,
    activation: str = "demod",
) -> torch.Tensor:
    points_num, c_in = x.shape
    c_out, c_in = weight.shape
    rank = styles.shape[0] // (c_in + c_out)

    assert styles.shape[0] % (c_in + c_out) == 0
    assert len(styles.shape) == 1

    # Now, we need to construct a [c_out, c_in] matrix
    left_matrix = styles[: c_out * rank]  # [left_matrix_size]
    right_matrix = styles[c_out * rank :]  # [right_matrix_size]

    left_matrix = left_matrix.view(c_out, rank)  # [c_out, rank]
    right_matrix = right_matrix.view(rank, c_in)  # [c_out, rank]

    # Imagine, that the output of `self.affine` (in SynthesisLayer) is N(0, 1)
    # Then, std of weights is sqrt(rank). Converting it back to N(0, 1)
    modulation = left_matrix @ right_matrix / np.sqrt(rank)  # [c_out, c_in]

    if activation == "tanh":
        modulation = modulation.tanh()
    elif activation == "sigmoid":
        modulation = modulation.sigmoid() - 0.5

    W = weight * (modulation + 1.0)  # [c_out, c_in]
    if activation == "demod":
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)  # [c_out, c_in]
    W = W.to(dtype=x.dtype)

    x = x.view(points_num, c_in, 1)
    out = torch.matmul(W, x)  # [num_rays, c_out, 1]
    out = out.view(points_num, c_out)  # [num_rays, c_out]

    return out


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        w_dim,
        channels_last=False,
        activation=nn.LeakyReLU(inplace=True),
        rank=10,
    ):
        super().__init__()

        self.w_dim = w_dim
        self.affine = nn.Linear(self.w_dim, (in_channels + out_channels) * rank)

        memory_format = (
            torch.channels_last if channels_last else torch.contiguous_format
        )
        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels]).to(memory_format=memory_format)
        )
        self.bias = torch.nn.Parameter(torch.zeros([1, out_channels]))
        self.activation = activation

    def forward(self, x, w):
        styles = self.affine(w).squeeze(0)

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles, activation="demod"
        )

        x = self.activation(x.add_(self.bias))
        return x
    

class EdgeConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, z_dim: int):
        super().__init__(aggr="add")

        self.mlp = nn.ModuleList(
            [
                SynthesisLayer(in_channels + 3, out_channels, z_dim),
                SynthesisLayer(out_channels, out_channels, z_dim),
            ]
        )

    def forward(
        self,
        h,
        pos,
        edge_index,
        w
    ):
        return self.propagate(edge_index, h=h, pos=pos, w=w)

    def message(self, h_j, pos_j, pos_i, w):
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        for layer in self.mlp:
            edge_feat = layer(edge_feat, w)
        return edge_feat
    

class LINKX(torch.nn.Module):
    r"""The LINKX model from the `"Large Scale Learning on Non-Homophilous
    Graphs: New Benchmarks and Strong Simple Methods"
    <https://arxiv.org/abs/2110.14446>`_ paper.
    """
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        w_dim: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers
        self.num_layers = num_layers

        self.edge_lin = SparseLinear(num_nodes, hidden_channels)

        if self.num_edge_layers > 1:
            self.edge_norm = BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = MLP(channels, dropout=0., act_first=True, act="leakyrelu")
        else:
            self.edge_norm = None
            self.edge_mlp = None

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = MLP(channels, dropout=0., act_first=True, act="leakyrelu")

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = nn.ModuleList()
        for channel_in, channel_out in pairwise(channels):
            self.final_mlp.append(SynthesisLayer(channel_in, channel_out, w_dim))

        self.leakyrelu = nn.LeakyReLU(inplace=True)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.edge_lin.reset_parameters()
        if self.edge_norm is not None:
            self.edge_norm.reset_parameters()
        if self.edge_mlp is not None:
            self.edge_mlp.reset_parameters()
        self.node_mlp.reset_parameters()
        self.cat_lin1.reset_parameters()
        self.cat_lin2.reset_parameters()

    def forward(
        self,
        x: OptTensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        w = None,
    ) -> Tensor:
        """"""  # noqa: D419
        out = self.edge_lin(edge_index, edge_weight)

        if self.edge_norm is not None and self.edge_mlp is not None:
            out = self.leakyrelu(out)
            out = self.edge_norm(out)
            out = self.edge_mlp(out)

        out = out + self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x)
            out = out + x
            out = out + self.cat_lin2(x)

        out = self.leakyrelu(out)
        for i, layer in enumerate(self.final_mlp):
            out = layer(out, w[i])
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(num_nodes={self.num_nodes}, '
                f'layers={self.num_layers}, '
                f'in_channels={self.in_channels}, '
                f'out_channels={self.out_channels})')


class PointGNNConv(gnn.MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper.
    """

    def __init__(
        self,
        channels,
        z_dim,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.mlp_h = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, 3),
            nn.Tanh(),
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(channels + 3, channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.LeakyReLU(inplace=True),
        )

        self.mlp_g = nn.ModuleList(
            [
                SynthesisLayer(channels, channels, z_dim),
                SynthesisLayer(channels, channels, z_dim),
            ]
        )

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_h)
        reset(self.mlp_f)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, w: Tensor) -> Tensor:
        # propagate_type: (x: Tensor, pos: Tensor)
        out = self.propagate(edge_index, x=x, pos=pos)
        for i, layer in enumerate(self.mlp_g):
            out = layer(out, w[i])
        return x + out

    def message(self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor, x_j: Tensor) -> Tensor:
        delta = self.mlp_h(x_i)
        e = torch.cat([pos_j - pos_i + delta, x_j], dim=-1)
        return self.mlp_f(e)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  mlp_h={self.mlp_h},\n"
            f"  mlp_f={self.mlp_f},\n"
            f"  mlp_g={self.mlp_g},\n"
            f")"
        )


class CloudGenerator(nn.Module):
    def __init__(self, channels=128, z_dim=128, blocks=2):
        super().__init__()
        self.z_dim = z_dim
        self.blocks = blocks

        self.encoder = rff.layers.GaussianEncoding(
            sigma=10.0, input_size=3, encoded_size=channels // 2
        )
        # self.layer_norm = nn.LayerNorm(channels)

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

        self.synthetic_block1 = PointGNNConv(128, z_dim)
        self.synthetic_block2 = PointGNNConv(128, z_dim)

        self.synthetic_block3 = LINKX(POINTS, 128, 128, 128, 2, z_dim)
        self.synthetic_block4 = LINKX(POINTS, 128, 128, 128, 2, z_dim)

    def forward(self, pos, edge_index, batch, w):
        x = self.encoder(pos)

        x = self.synthetic_block1(x, pos, edge_index, w[:2])
        # x = self.synthetic_block3(x, edge_index, w=w)
        # x = self.synthetic_block4(x, edge_index, w=w)
        x = self.synthetic_block2(x, pos, edge_index, w[2:])
        
        h = global_max_pool(x, batch)
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)

        x = torch.cat([x, h], dim=-1)
        return self.tail(x), x


class GaussiansGenerator(nn.Module):
    def __init__(self, channels=128, z_dim=128, blocks=1):
        super().__init__()
        self.z_dim = z_dim

        self.synthetic_block1 = LINKX(POINTS, 256, 256, 256, 2, z_dim)
        self.synthetic_block2 = LINKX(POINTS, 256, 256, 256, 2, z_dim)
        self.synthetic_block3 = LINKX(POINTS, 256, 256, 256, 2, z_dim)
        self.synthetic_block4 = LINKX(POINTS, 256, 256, 256, 2, z_dim)
        self.synthetic_block5 = LINKX(POINTS, 256, 256, 256, 2, z_dim)
        self.synthetic_block6 = LINKX(POINTS, 256, 256, 256, 2, z_dim)
        self.synthetic_block7 = LINKX(POINTS, 256, 256, 256, 2, z_dim)
        self.synthetic_block8 = LINKX(POINTS, 256, 256, 256, 2, z_dim)


    def forward(self, x, pos, edge_index, batch, w):
        x_ = x
        x = self.synthetic_block1(x, edge_index, w=w)
        x = self.synthetic_block2(x, edge_index, w=w)
        x = self.synthetic_block3(x, edge_index, w=w)
        x = self.synthetic_block4(x, edge_index, w=w)
        x = self.synthetic_block5(x, edge_index, w=w)
        x = self.synthetic_block6(x, edge_index, w=w)
        x = self.synthetic_block7(x, edge_index, w=w)
        x = self.synthetic_block8(x, edge_index, w=w)

        return torch.cat([x, x_], dim=-1)


class ImageGenerator(nn.Module):
    def __init__(self, z_dim=512, **kwargs):
        super(ImageGenerator, self).__init__()
        c = 128
        self.point_encoder = CloudGenerator(z_dim=z_dim)
        self.gaussians = GaussiansGenerator(c, z_dim=z_dim)
        self.decoder = GaussianDecoder(256 + 256)
        self.num_ws = 10
        self.z_dim = z_dim

        self.register_buffer("background", torch.ones(3, dtype=torch.float32))
        self.register_buffer("sphere", self._fibonacci_sphere(POINTS, 1.0))
        self.register_buffer("dist", torch.cdist(self.sphere, self.sphere))
                             
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
            edge_index = knn_graph(sphere.pos, k=6, batch=sphere.batch)

        images = []
        for camera, w in zip(cameras, ws):
            point_cloud, points_features = self.point_encoder(pos, edge_index, batch, w[:4])
            new_edge_index = knn_graph(point_cloud, k=6, batch=sphere.batch)

            gaussian = self.gaussians(points_features, point_cloud, new_edge_index, batch, w[4:])
            gaussian_model = self.decoder(gaussian, point_cloud)
            image = render(camera, gaussian_model, self.background, use_rgb=True)
            images.append(image)

        return torch.stack(images, dim=0).contiguous()


class Generator(nn.Module):
    def __init__(self, z_dim, c_dim, w_dim, img_resolution, img_channels, mapping_kwargs = {}, **kwargs):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = 0
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = ImageGenerator(z_dim=w_dim, **kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = stylegan2.MappingNetwork(z_dim=self.z_dim, c_dim=self.c_dim, w_dim=self.w_dim, num_ws=self.num_ws, **mapping_kwargs)


    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, None, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas)

