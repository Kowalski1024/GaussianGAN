from dataclasses import dataclass
import math

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
import torch

from src.utils.gaussian_splatting import Camera


@dataclass
class GaussianModel:
    xyz: torch.Tensor = None
    opacity: torch.Tensor = None
    rotation: torch.Tensor = None
    scaling: torch.Tensor = None
    shs: torch.Tensor = None

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            setattr(self, k, v)

    def to(self, device):
        return GaussianModel(
            xyz=self.xyz.to(device),
            opacity=self.opacity.to(device),
            rotation=self.rotation.to(device),
            scaling=self.scaling.to(device),
            shs=self.shs.to(device),
        )

    def __repr__(self) -> str:
        return (
            f"GaussianModel(xyz={self.xyz.shape}, "
            f"opacity={self.opacity.shape}, "
            f"rotation={self.rotation.shape}, "
            f"scaling={self.scaling.shape}, "
            f"shs={self.shs.shape})"
        )


def render(
    viewpoint_camera: Camera,
    pc: GaussianModel,
    bg_color: torch.Tensor,
    scaling_modifier=1.0,
    use_rgb=False,
):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = (
        torch.zeros_like(pc.xyz, dtype=pc.xyz.dtype, requires_grad=True, device="cuda")
        + 0
    )
    try:
        screenspace_points.retain_grad()
    except Exception:
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
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.xyz
    means2D = screenspace_points
    opacity = pc.opacity
    scales = pc.scaling
    rotations = pc.rotation
    cov3D_precomp = None

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
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    rendered_image = torch.clamp(rendered_image, 0.0, 1.0)
    rendered_image = rendered_image * 2.0 - 1.0
    return rendered_image
