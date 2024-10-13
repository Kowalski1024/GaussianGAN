from itertools import pairwise

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch_geometric import nn as gnn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor

from src.utils.training import normalize_2nd_moment


def fmm_modulate_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    styles: torch.Tensor,
    noise: OptTensor = None,
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

    if noise is not None:
        out = out.add_(noise)

    return out


class SynthesisLayer(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        style_channels: int,
        use_noise: bool = False,
        use_bias: bool = True,
        rank: int = 10,
        activation=nn.LeakyReLU(inplace=True),
    ):
        super().__init__()
        self.use_noise = use_noise
        self.out_channels = out_channels
        self.affine = nn.Linear(style_channels, (in_channels + out_channels) * rank)

        self.weight = torch.nn.Parameter(
            torch.randn([out_channels, in_channels]).to(memory_format=torch.contiguous_format)
        )
        if use_bias:
            self.bias = torch.nn.Parameter(torch.zeros([1, out_channels]))
        else:
            self.register_buffer("bias", None)

        self.noise_strength = None
        if use_noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros([]))

        self.activation = activation

    def forward(self, x, w):
        styles = self.affine(w).squeeze(0)

        if self.use_noise:
            noise = torch.randn(self.out_channels, device=x.device) * self.noise_strength
        else:
            noise = None

        x = fmm_modulate_linear(
            x=x, weight=self.weight, styles=styles, activation="demod", noise=noise
        )

        if self.bias is not None:
            x.add_(self.bias)

        x = self.activation(x)
        return x


class SynthesisBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        style_channels: int,
        synthethic_layers: int,
        use_noise: bool = False,
        rank: int = 10,
    ):
        super().__init__()

        self.synthethic_layers = synthethic_layers

        channels = [in_channels] + [hidden_channels] * synthethic_layers + [out_channels]
        self.final_mlp = nn.ModuleList()
        for in_c, out_c in pairwise(channels):
            self.final_mlp.append(
                SynthesisLayer(
                    in_channels=in_c,
                    out_channels=out_c,
                    style_channels=style_channels,
                    use_noise=use_noise,
                    rank=rank,
                )
            )

    def forward(self, x, w):
        out = x
        for i, layer in enumerate(self.final_mlp):
            out = layer(out, w)
        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(synthethic_layers={self.synthethic_layers})"


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
        style_channels: int,
        synthethic_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
        use_noise: bool = False,
        rank: int = 10,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_layers = num_edge_layers
        self.synthethic_layers = synthethic_layers

        self.edge_lin = gnn.models.linkx.SparseLinear(num_nodes, hidden_channels)

        if self.num_edge_layers > 1:
            self.edge_norm = nn.BatchNorm1d(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = gnn.models.MLP(channels, dropout=0.0, act_first=True, act="leakyrelu")
        else:
            self.edge_norm = None
            self.edge_mlp = None

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = gnn.models.MLP(channels, dropout=0.0, act_first=True, act="leakyrelu")

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * synthethic_layers + [out_channels]
        self.final_mlp = nn.ModuleList()
        for in_c, out_c in pairwise(channels):
            self.final_mlp.append(
                SynthesisLayer(
                    in_channels=in_c,
                    out_channels=out_c,
                    style_channels=style_channels,
                    use_noise=use_noise,
                    rank=rank,
                )
            )

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
        pos: OptTensor,
        edge_index: Adj,
        w=None,
    ) -> Tensor:
        """"""  # noqa: D419
        out = self.edge_lin(edge_index, None)

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
        return (
            f"{self.__class__.__name__}(num_nodes={self.num_nodes}, "
            f"synthethic_layers={self.synthethic_layers}, "
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels})"
        )


class PointGNNConv(gnn.MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        style_channels,
        synthethic_layers=2,
        use_noise=False,
        rank: int = 10,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "max")
        super().__init__(**kwargs)

        self.mlp_h = nn.ModuleList(
            [
                SynthesisLayer(in_channels, in_channels // 2, style_channels),
                SynthesisLayer(in_channels // 2, 3, style_channels, activation=nn.Tanh()),
            ]
        )

        layers = [in_channels + 3] + [in_channels] * synthethic_layers
        self.mlp_g = nn.ModuleList()
        for in_c, out_c in pairwise(layers):
            self.mlp_g.append(
                SynthesisLayer(
                    in_channels=in_c,
                    out_channels=out_c,
                    style_channels=style_channels,
                    use_noise=use_noise,
                    rank=rank,
                )
            )

        if in_channels != out_channels:
            self.lin = nn.Linear(in_channels, out_channels)
        else:
            self.lin = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_h)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj, w: Tensor) -> Tensor:
        delta = x
        for i, layer in enumerate(self.mlp_h):
            delta = layer(delta, w[i])
        out = self.propagate(edge_index, x=x, pos=pos, delta=delta)

        for i, layer in enumerate(self.mlp_g):
            out = layer(out, w[i])

        x = x + out

        return self.lin(x)

    def message(
        self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor, x_j: Tensor, delta_i: Tensor
    ) -> Tensor:
        return torch.cat([pos_j - pos_i + delta_i, x_j], dim=-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  mlp_h={self.mlp_h},\n"
            f"  mlp_g={self.mlp_g},\n"
            f"  lin={self.lin}\n"
            f")"
        )


class MappingNetwork(nn.Module):
    def __init__(self, channels: int, num_layers: int):
        super().__init__()
        self.mapping = nn.Sequential(
            *[
                nn.Linear(channels, channels),
                nn.LeakyReLU(inplace=True),
            ]
            * num_layers
        )

    def forward(self, x):
        x = normalize_2nd_moment(x)
        return self.mapping(x)


class GlobalPoolingLayer(nn.Module):
    def __init__(self, type: str, channels: int, layers: int = 2):
        super().__init__()

        if type == "max":
            self.pool = gnn.global_max_pool
        elif type == "mean":
            self.pool = gnn.global_mean_pool
        elif type == "sum":
            self.pool = gnn.global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {type}")

        self.global_layers = nn.Sequential(
            *[
                nn.Linear(channels, channels),
                nn.LeakyReLU(inplace=True),
            ]
            * layers
        )

    def forward(self, x, batch):
        h = self.pool(x, batch)
        h = self.global_layers(h)
        h = h.repeat(x.size(0), 1)
        return h


class _TruncExp(torch.autograd.Function):  # pylint: disable=abstract-method
    # Implementation from torch-ngp:
    # https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):  # pylint: disable=arguments-differ
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, g):  # pylint: disable=arguments-differ
        x = ctx.saved_tensors[0]
        return g * torch.exp(torch.clamp(x, max=15))


trunc_exp = _TruncExp.apply
