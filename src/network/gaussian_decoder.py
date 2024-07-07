import torch
from torch import nn
from src.utils.render import GaussianModel
import numpy as np


class GaussianDecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        shs_degree=3,
        use_rgb=False,
        xyz_offset=True,
        restrict_offset=True,
    ):
        super(GaussianDecoder, self).__init__()
        self.use_rgb = use_rgb
        self.xyz_offset = xyz_offset
        self.restrict_offset = restrict_offset

        # self.mlp = nn.Sequential(
        #     nn.Linear(in_channels, 128),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(128, 128),
        #     nn.LeakyReLU(inplace=True),
        # )

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

            if not (key == "shs" and self.use_rgb):
                nn.init.constant_(layer.weight, 0)
                nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, np.log(0.1 / (1 - 0.1)))

            self.decoders.append(nn.Sequential(
                nn.Linear(in_channels, 128),
                nn.SiLU(inplace=True),
                nn.Linear(128, 128),
                nn.SiLU(inplace=True),
                layer,
            ))

    def forward(self, x, pts=None):
        # x = self.mlp(x)

        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.decoders):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                v = torch.clamp(v, min=0, max=0.02)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                if self.use_rgb:
                    v = torch.sigmoid(v)
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                if self.restrict_offset:
                    max_step = 1.2 / 32
                    v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pts if self.xyz_offset else pts
            ret[k] = v

        return GaussianModel(**ret)


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
