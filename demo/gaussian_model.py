import os
from typing import NamedTuple

import numpy as np
from plyfile import PlyData, PlyElement
import rff
import torch
from torch import Tensor, nn
from torch_geometric import nn as gnn
from torch_geometric.data import Data
from torch_geometric.nn.inits import reset
from torch_geometric.nn.models import LINKX
from torch_geometric.typing import Adj


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


class GaussianModel(NamedTuple):
    xyz: torch.Tensor
    opacity: torch.Tensor
    rotation: torch.Tensor
    scaling: torch.Tensor
    shs: torch.Tensor

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    @property
    def active_sh_degree(self):
        return self.shs.shape[1]

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        for i in range(features_dc.shape[1] * features_dc.shape[2]):
            l.append("f_dc_{}".format(i))
        for i in range(features_rest.shape[1] * features_rest.shape[2]):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        for i in range(self.scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self.rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        xyz = self.xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        features_dc = self.shs[:, :1]
        features_rest = self.shs[:, 1:]
        f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = inverse_sigmoid(
            torch.clamp(self.opacity, 1e-3, 1 - 1e-3).detach().cpu().numpy()
        )
        scale = np.log(self.scaling.detach().cpu().numpy())
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, "f4") for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def __repr__(self) -> str:
        return f"GaussianModel(xyz={self.xyz.shape}, opacity={self.opacity.shape}, rotation={self.rotation.shape}, scaling={self.scaling.shape}, shs={self.shs.shape})"


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


class GaussianDecoder(nn.Module):
    # Implementation from TriplaneGaussian (03.2024):
    # https://github.com/VAST-AI-Research/TriplaneGaussian/blob/main/tgs/models/renderer.py#L128
    def __init__(self, in_channels, shs_degree=3, use_rgb=True, offset=True):
        super(GaussianDecoder, self).__init__()
        self.use_rgb = use_rgb
        self.offset = False

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(inplace=True),
        )

        self.feature_channels = {
            "scaling": 3,
            "rotation": 4,
            "opacity": 1,
            "shs": shs_degree,
            "xyz": 3,
        }

        self.decoders = torch.nn.ModuleList()

        for key, channels in self.feature_channels.items():
            layer = nn.Linear(128, channels)

            # if not (key == "shs" and self.use_rgb):
            #     nn.init.constant_(layer.weight, 0)
            #     nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, inverse_sigmoid(0.1))

            self.decoders.append(layer)

    def forward(self, x, pc=None):
        x = self.mlp(x)

        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.decoders):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = torch.nn.functional.softplus(v)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                # if self.offset:
                max_step = 1.2 / 32
                v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pc
                # else:
                #     v = pc
            ret[k] = v

        # guass = torch.load("/mnt/d/Tomasz/Pulpit/GaussianGAN/car1.pth")
        return GaussianModel(**ret)


class PointGNNConv(gnn.MessagePassing):
    r"""The PointGNN operator from the `"Point-GNN: Graph Neural Network for
    3D Object Detection in a Point Cloud" <https://arxiv.org/abs/2003.01251>`_
    paper.
    """

    def __init__(
        self,
        mlp_h: torch.nn.Module,
        mlp_f: torch.nn.Module,
        mlp_g: torch.nn.Module,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "mean")
        super().__init__(**kwargs)

        self.mlp_h = mlp_h
        self.mlp_f = mlp_f
        self.mlp_g = mlp_g

        # self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.mlp_h)
        reset(self.mlp_f)
        reset(self.mlp_g)

    def forward(self, x: Tensor, pos: Tensor, edge_index: Adj) -> Tensor:
        # Calculate delta in the forward function
        delta = self.mlp_h(x)

        # Pass delta to the message function
        out = self.propagate(edge_index, x=x, pos=pos, delta=delta)
        out = self.mlp_g(out)
        return x + out

    def message(
        self, pos_j: Tensor, pos_i: Tensor, x_i: Tensor, x_j: Tensor, delta_i: Tensor
    ) -> Tensor:
        # Use the passed delta_i directly, no need to calculate it here
        return torch.cat([pos_j - pos_i + delta_i, x_j], dim=-1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  mlp_h={self.mlp_h},\n"
            f"  mlp_f={self.mlp_f},\n"
            f"  mlp_g={self.mlp_g},\n"
            f")"
        )


class GNNConv(nn.Module):
    def __init__(self, din_in, dim_out):
        super().__init__()

        self.mlp_h = nn.Sequential(
            nn.Linear(din_in, din_in),
            nn.ReLU(inplace=True),
            nn.Linear(din_in, 3),
            nn.Tanh(),
        )

        self.mlp_f = nn.Sequential(
            nn.Linear(din_in + 3, dim_out),
            nn.ReLU(inplace=True),
        )

        self.mlp_g = nn.Sequential(
            nn.Linear(dim_out + 3, dim_out),
            nn.ReLU(inplace=True),
            nn.Linear(dim_out, din_in),
            nn.ReLU(inplace=True),
        )

        self.network = PointGNNConv(self.mlp_h, self.mlp_f, self.mlp_g)

    def forward(self, x, pos, edge_index):
        return self.network(x, pos, edge_index)


class PointGenerator(nn.Module):
    def __init__(self):
        super(PointGenerator, self).__init__()

        self.encoder = rff.layers.GaussianEncoding(sigma=10.0, input_size=3, encoded_size=64)

        self.global_conv = nn.Sequential(
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.tail = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 3),
            nn.Tanh(),
        )

        self.conv1 = GNNConv(128, 128)
        self.conv2 = GNNConv(128, 128)

    def forward(self, pos, edge_index, batch):
        x = self.encoder(pos)
        x = self.conv1(x, pos, edge_index)
        x = self.conv2(x, pos, edge_index)

        h = gnn.global_max_pool(x, batch)
        h = self.global_conv(h)
        h = h.repeat(x.size(0), 1)

        x = torch.cat([x, h], dim=1)

        return x, self.tail(x)


class GaussiansGenerator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = LINKX(8192*2, 256, 256, 256, 2)
        self.conv2 = LINKX(8192*2, 256, 256, 256, 2)

    def forward(self, x, pos, edge_index, batch):
        x_ = x
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)

        return torch.cat([x, x_], dim=1)


class Generator(nn.Module):
    def __init__(self, shs_degree=3, use_rgb=True, offset=True):
        super().__init__()
        self.points = PointGenerator()
        self.gaussians = GaussiansGenerator()
        self.decoder = GaussianDecoder(512, shs_degree, use_rgb, offset)

    def forward(self, sphere: Data):
        edge_index = sphere.edge_index
        gaussians, point_cloud = self.points(sphere.pos, edge_index, sphere.batch)
        # edge_index = gnn.knn_graph(point_cloud, k=6)
        gaussians = self.gaussians(gaussians, point_cloud, edge_index, sphere.batch)
        model = self.decoder(gaussians, point_cloud)
        return model
