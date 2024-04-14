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

from .fma import fma


def modulated_linear(
    x,                          # Input tensor of shape [batch_size, in_features].
    weight,                     # Weight tensor of shape [out_features, in_features].
    styles,                     # Modulation coefficients of shape [features].
    noise=None,                 # Optional noise tensor to add to the output activations.
    demodulate=True,            # Apply weight demodulation?
):
    _, in_features = weight.shape

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == torch.float16 and demodulate:
        weight = weight * (1 / np.sqrt(in_features) / weight.norm(float('inf'), dim=[1], keepdim=True)) # max_I
        styles = styles / styles.norm(float('inf'), keepdim=True) # max_I

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate:
        w = weight.unsqueeze(0) # [NOI]
        w = w * styles.unsqueeze(0).unsqueeze(0) # [1, 1, I]
        dcoefs = (w.square().sum(dim=2) + 1e-8).rsqrt() # [NO]
        w = w * dcoefs.unsqueeze(0) # [1, NO, I]

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
        if len(style.shape) == 1:
            style = style.unsqueeze(0)

        style = self.affine(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = (gamma * out).add_(beta)

        return out


class StyleLinearLayer(nn.Module):
    def __init__(self, in_dim, w_dim, out_dim, noise=True, activation=F.leaky_relu):
        super().__init__()
        self.in_dim = in_dim
        self.w_dim = w_dim
        self.out_dim = out_dim
        self.noise = noise
        self.activation = activation

        self.linear = nn.Linear(in_dim, out_dim)
        self.adain = AdaptivePointNorm(out_dim, w_dim)
        self.noise_strength = nn.Parameter(torch.zeros(1)) if noise else None
        self.affine = nn.Linear(w_dim, in_dim)

    def forward(self, x, w):
        x = self.linear(x)

        if self.noise:
            noise = torch.randn(1, x.size(1), device=x.device)
            noise = noise * self.noise_strength
        
        x = modulated_linear(x, self.linear.weight, w, noise)

        return self.activation(x.add_(self.linear.bias))


class StyleMLP(nn.Module):
    def __init__(
        self,
        channels,
        w_dim,
        noise=True,
        activation=F.leaky_relu,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(channels[:-1], channels[1:]):
            self.layers.append(
                StyleLinearLayer(in_dim, w_dim, out_dim, noise, activation)
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
            f"out_channels={self.out_channels})"
        )
