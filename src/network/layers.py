import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric import nn as gnn
from src.utils.training import normalize_2nd_moment


class MappingNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers):
        super().__init__()

        self.mapping = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        for _ in range(num_layers - 2):
            self.mapping.add_module(
                f"linear_{_}",
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.mapping.add_module(
                f"act_{_}",
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.mapping.add_module(
            f"linear_{num_layers - 1}",
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        """
        Args:
            x: [N, style_dim]

        Returns:
            x: [N, style_dim]
        """
        x = normalize_2nd_moment(x)
        return self.mapping(x)


class AdaptivePointNorm(nn.Module):
    def __init__(self, in_channels, style_dim):
        super().__init__()

        self.norm = gnn.InstanceNorm(in_channels)
        self.style = nn.Linear(style_dim, in_channels * 2)

        self.style.weight.data.normal_()
        self.style.bias.data.zero_()

        self.style.bias.data[:in_channels] = 1
        self.style.bias.data[in_channels:] = 0

    def forward(self, x, style):
        """
        Args:
            x: [N, C]
            style: [N, style_dim]

        Returns:
            x: [N, C]
        """
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        x = self.norm(x)
        x = (gamma * x).add_(beta)

        return x


class EdgeConv(gnn.MessagePassing):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")

        self.mlp = torch.nn.Sequential(
            nn.Linear(in_channels + 3, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(
        self,
        h,
        pos,
        edge_index,
    ):
        return self.propagate(edge_index, h=h, pos=pos)

    def message(self, h_j, pos_j, pos_i):
        edge_feat = torch.cat([h_j, pos_j - pos_i], dim=-1)
        return self.mlp(edge_feat)


class PointGNNConv(nn.Module):
    def __init__(self, in_channels, out_channels, aggr="sum"):
        super().__init__()

        self.mlp_h = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_channels, 3),
            nn.Tanh(),
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(in_channels + 3, in_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

        self.network = gnn.PointGNNConv(self.mlp_h, self.mlp_f, self.mlp_g, aggr=aggr)

    def forward(self, x, pos, edge_index):
        """
        Args:
            x: [N, in_channels]
            pos: [N, 3]
            edge_index: [2, E]

        Returns:
            out: [N, out_channels]
        """
        return self.network(x, pos, edge_index)


class SyntheticBlock(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, use_noise=True):
        super().__init__()
        self.add_noise = use_noise

        self.gnn_conv = PointGNNConv(in_channels, out_channels)
        self.adaptive_norm = AdaptivePointNorm(out_channels, style_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        if self.add_noise:
            self.noise_strength = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, pos, edge_index, style):
        """
        Args:
            x: [N, in_channels]
            pos: [N, 3]
            edge_index: [2, E]
            style: [N, style_channels]

        Returns:
            x: [N, out_channels]
        """
        x = self.gnn_conv(x, pos, edge_index)

        if self.add_noise:
            noise = torch.randn(1, x.size(1), device=x.device) * self.noise_strength
            x = x.add_(noise)

        x = self.leaky_relu(x)
        x = self.adaptive_norm(x, style)
        return x


class StyleLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels, style_channels, use_noise=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_channels = style_channels
        self.use_noise = use_noise
        self.activation = nn.LeakyReLU(inplace=True)

        self.linear = nn.Linear(in_channels, out_channels)
        self.adain = AdaptivePointNorm(out_channels, style_channels)
        self.noise_strength = nn.Parameter(torch.zeros(1)) if use_noise else None

    def forward(self, x, style):
        """
        Args:
            x: [N, in_channels]
            style: [N, style_channels]

        Returns:
            x: [N, out_channels]
        """
        x = self.linear(x)

        if self.use_noise:
            noise = torch.randn(1, x.size(1), device=x.device) * self.noise_strength
            x.add_(noise)

        x = self.adain(x, style)

        return self.activation(x)


class StyleMLP(nn.Module):
    def __init__(
        self,
        channels,
        style_channels,
        use_noise=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            self.layers.append(
                StyleLinearLayer(in_channels, style_channels, out_channels, use_noise)
            )

    def forward(self, x, style):
        """
        Args:
            x: [N, in_channels]
            style: [N, style_channels]

        Returns:
            x: [N, out_channels]
        """
        for layer in self.layers:
            x = layer(x, style)
        return x


class StyleLINKX(torch.nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        style_channels: int,
        num_layers: int,
        num_edge_layers: int = 1,
        num_node_layers: int = 1,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.style_channels = style_channels
        self.num_edge_layers = num_edge_layers
        self.num_node_layers = num_node_layers
        self.num_layers = num_layers

        self.edge_lin = gnn.models.linkx.SparseLinear(num_nodes, hidden_channels)

        if self.num_edge_layers > 1:
            self.edge_norm = gnn.LayerNorm(hidden_channels)
            channels = [hidden_channels] * num_edge_layers
            self.edge_mlp = gnn.MLP(channels, dropout=0.0, act_first=True)
        else:
            self.edge_norm = None
            self.edge_mlp = None

        channels = [in_channels] + [hidden_channels] * num_node_layers
        self.node_mlp = StyleMLP(channels, style_channels)

        self.cat_lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.cat_lin2 = torch.nn.Linear(hidden_channels, hidden_channels)

        channels = [hidden_channels] * num_layers + [out_channels]
        self.final_mlp = StyleMLP(channels, style_channels)
        self.num_ws = num_node_layers + num_layers

    def forward(self, x, edge_index, style):
        """
        Args:
            x: [N, in_channels]
            edge_index: [2, E]
            style: [N, style_channels]

        Returns:
            x: [N, out_channels]
        """

        out = self.edge_lin(edge_index, None)

        if self.edge_norm is not None and self.edge_mlp is not None:
            out = out.relu_()
            out = self.edge_norm(out)
            out = self.edge_mlp(out)

        out = out + self.cat_lin1(out)

        if x is not None:
            x = self.node_mlp(x, style)
            out = out + x
            out = out + self.cat_lin2(x)

        return self.final_mlp(out.relu_(), style)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_nodes={self.num_nodes}, "
            f"in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}, "
            f"num_layers={self.num_layers}, "
            f"num_edge_layers={self.num_edge_layers}, "
            f"num_node_layers={self.num_node_layers})"
        )
