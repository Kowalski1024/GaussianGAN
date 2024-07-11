from typing import NamedTuple, List
import math
import numpy as np

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from . import networks_stylegan2 as stylegan2


class GaussianModel(NamedTuple):
    xyz: torch.Tensor
    opacity: torch.Tensor
    rotation: torch.Tensor
    scaling: torch.Tensor
    shs: torch.Tensor

    def __repr__(self) -> str:
        return f"GaussianModel(xyz={self.xyz.shape}, opacity={self.opacity.shape}, rotation={self.rotation.shape}, scaling={self.scaling.shape}, shs={self.shs.shape})"


class Camera(NamedTuple):
    FoVx: float
    FoVy: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    image_height: int = 128
    image_width: int = 128


def get_projection_matrices(znear, zfar, fovX, fovY):
    bath_size = fovX.shape[0]
    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros((bath_size, 4, 4), device=fovX.device)

    z_sign = 1.0

    P[:, 0, 0] = 2.0 * znear / (right - left)
    P[:, 1, 1] = 2.0 * znear / (top - bottom)
    P[:, 0, 2] = (right + left) / (right - left)
    P[:, 1, 2] = (top + bottom) / (top - bottom)
    P[:, 3, 2] = z_sign
    P[:, 2, 2] = z_sign * zfar / (zfar - znear)
    P[:, 2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def extract_cameras(camera_to_world, intrinsics) -> List[Camera]:
    w2c = torch.inverse(camera_to_world)

    # Extract FoVx and FoVy from intrinsics matrix
    FoVx = 2 * torch.atan(intrinsics[:, 0, 2] / intrinsics[:, 0, 0])
    FoVy = 2 * torch.atan(intrinsics[:, 1, 2] / intrinsics[:, 1, 1])

    world_view_transform = w2c.transpose(-2, -1)
    zfar = 100.0
    znear = 0.01

    # Calculate projection matrix
    projection_matrix = get_projection_matrices(znear, zfar, FoVx, FoVy).transpose(-2, -1)

    full_proj_transform = torch.bmm(world_view_transform, projection_matrix)
    camera_center = torch.inverse(world_view_transform)[:, 3, :3]
    
    cameras = []
    for i in range(camera_to_world.shape[0]):
        cameras.append(
            Camera(
                FoVx[i].item(), 
                FoVy[i].item(), 
                world_view_transform[i], 
                full_proj_transform[i], 
                camera_center[i]
                )
            )
        
    return cameras


def uniform_circle(low: float, high: float):
    """Sample a point uniformly from half of a circle.

    The returned point is described by an angle from `low` to `high`. The samples
    are drawn so that the points on the circle would follow a uniform distribution.
    Angles don't follow a uniform distribution.
    """
    if low < 0 or high > 180 or low > high:
        raise ValueError("Angles must be in range [0, 180] and `low` must be smaller than `high`!")

    low = low / 180 * np.pi
    high = high / 180 * np.pi
    sample = np.random.uniform(low=0, high=1)
    sample_radians = np.arccos(np.cos(low) - sample * (np.cos(low) - np.cos(high)))
    return sample_radians / np.pi * 180


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()


rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()


rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


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

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return rendered_image


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
inverse_sigmoid = lambda x: np.log(x / (1 - x))


class MLP(torch.nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        n_neurons: int = 128,
        n_hidden_layers: int = 2,
        activation: str = "silu",
        output_activation = "silu",
        bias: bool = True,
    ):
        super().__init__()
        layers = [
            self.make_linear(
                dim_in, n_neurons, is_first=True, is_last=False, bias=bias
            ),
            self.make_activation(activation),
        ]
        for i in range(n_hidden_layers - 1):
            layers += [
                self.make_linear(
                    n_neurons, n_neurons, is_first=False, is_last=False, bias=bias
                ),
                self.make_activation(activation),
            ]
        layers += [
            self.make_linear(
                n_neurons, dim_out, is_first=False, is_last=True, bias=bias
            )
        ]
        if output_activation is not None:
            layers.append(self.make_activation(output_activation))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def make_linear(self, dim_in, dim_out, is_first, is_last, bias=True):
        layer = torch.nn.Linear(dim_in, dim_out, bias=bias)
        return layer

    def make_activation(self, activation):
        if activation == "relu":
            return torch.nn.ReLU(inplace=True)
        elif activation == "silu":
            return torch.nn.SiLU(inplace=True)
        else:
            raise NotImplementedError


class GaussianDecoder(torch.nn.Module):
    feature_channels = {
        "scaling": 3,
        "rotation": 4,
        "opacity": 1,
        "shs": 3,
        "xyz": 3
    }
    use_rgb = True

    def __init__(self, channels = None) -> None:
        super().__init__()
        self.mlp = MLP(channels, 128) if channels else None
        self.decoders = torch.nn.ModuleList()

        for key, channels in self.feature_channels.items():
            layer = torch.nn.Linear(128, channels)

            if key == "scaling":
                torch.nn.init.constant_(layer.bias, -5.0)
            elif key == "rotation":
                torch.nn.init.constant_(layer.bias, 0)
                torch.nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                torch.nn.init.constant_(layer.bias, inverse_sigmoid(0.1))

            self.decoders.append(layer)

    def forward(self, x) -> GaussianModel:
        if self.mlp is not None:
            x = self.mlp(x)

        ret = {}
        for k, layer in zip(self.feature_channels.keys(), self.decoders):
            v = layer(x)
            if k == "rotation":
                v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                v = torch.clamp(v, min=0, max=0.2)
            elif k == "opacity":
                v = torch.sigmoid(v)
            elif k == "shs":
                if self.use_rgb:
                    v = torch.sigmoid(v)
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                v = torch.tanh(v)
            ret[k] = v

        return GaussianModel(**ret)


class SynthesisNetwork(torch.nn.Module):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 4,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0
        super().__init__()
        self.background = torch.nn.parameter.Parameter(torch.tensor([1, 1, 1], dtype=torch.float32), requires_grad=False)

        self.backbone = stylegan2.SynthesisNetwork(
            w_dim=w_dim, 
            img_resolution=img_resolution, 
            img_channels=img_channels, 
            channel_base=channel_base, 
            channel_max=channel_max, 
            num_fp16_res=num_fp16_res, 
            **block_kwargs
        )
        self.gaussian_decoder = GaussianDecoder(img_channels)

        self.num_ws = self.backbone.num_ws
        self.img_resolution = img_resolution

    def forward(self, ws, c, **block_kwargs):
        # check if c is empty tensor
        # if c.numel() == 0:
        #     poses = torch.stack([
        #             pose_spherical(
        #                 theta=np.random.uniform(low=-180, high=180),
        #                 phi=90 - uniform_circle(low=90, high=180),
        #                 radius=1.25
        #                 )
        #             for _ in range(ws.shape[0])
        #         ], dim=0).to(c.device)
        #     poses[:, :3, 1:3] *= -1
        #     intrinsics = torch.tensor([[512.0, 0.0, 256.0], [0.0, 512.0, 256.0], [0.0, 0.0, 1.0]], device=c.device).unsqueeze(0).repeat(ws.shape[0], 1, 1)
        # else:
        poses = c[:, :16].view(-1, 4, 4).detach()
        intrinsics = c[:, 16:25].view(-1, 3, 3).detach() * 512
        cameras = extract_cameras(poses, intrinsics)

        hidden_features = self.backbone(ws, **block_kwargs)

        N = hidden_features.shape[0]
        images = []
        for i in range(N):
            camera = cameras[i]
            features = hidden_features[i].view(-1, self.img_resolution ** 2).transpose(0, 1)
            gaussian = self.gaussian_decoder(features)
            images.append(render(camera, gaussian, self.background, use_rgb=self.gaussian_decoder.use_rgb))

        return torch.stack(images, dim=0).contiguous()

        
class Generator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.synthesis = SynthesisNetwork(w_dim=w_dim, img_resolution=64, img_channels=128, **synthesis_kwargs)
        self.num_ws = self.synthesis.num_ws
        self.mapping = stylegan2.MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.num_ws, **mapping_kwargs)

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False, **synthesis_kwargs):
        ws = self.mapping(z, torch.zeros([], device=c.device), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        img = self.synthesis(ws, c, update_emas=update_emas, **synthesis_kwargs)
        return img