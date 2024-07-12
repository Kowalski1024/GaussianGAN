import torch
import torch.nn as nn
from typing import NamedTuple
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from .camera import Camera
import numpy as np

class GaussianModel(NamedTuple):
    xyz: torch.Tensor = None
    opacity: torch.Tensor = None
    rotation: torch.Tensor = None
    scaling: torch.Tensor = None
    shs: torch.Tensor = None

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        return f"GaussianModel(xyz={self.xyz.shape}, opacity={self.opacity.shape}, rotation={self.rotation.shape}, scaling={self.scaling.shape}, shs={self.shs.shape})"
    

class GaussianDecoder(nn.Module):
    feature_channels = {"scaling": 3, "rotation": 4, "opacity": 1, "shs": 3, "xyz": 3}

    def __init__(self, in_channels=640, use_rgb=True, use_pc=True):
        super(GaussianDecoder, self).__init__()
        self.use_rgb = use_rgb
        self.use_pc = use_pc

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(inplace=True),
        )

        self.decoders = torch.nn.ModuleList()

        for key, channels in self.feature_channels.items():
            layer = nn.Linear(512, channels)

            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, np.log(0.1 / (1 - 0.1)))

            self.decoders.append(layer)

    def forward(self, x, pc=None):
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
                max_step = 1.2 / 32
                v = (torch.sigmoid(v) - 0.5) * max_step
                v = v + pc
                # if pc is not None:
                #     v = v + pc
                # v = torch.tanh(v) * 0.35
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


def render(viewpoint_camera: Camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, use_rgb=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.xyz
    means2D = screenspace_points
    opacity = pc.opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.scaling
    rotations = pc.rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if use_rgb:
        colors_precomp = pc.shs.squeeze(1)
    else:
        shs = pc.shs

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, _ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return rendered_image * 2 - 1