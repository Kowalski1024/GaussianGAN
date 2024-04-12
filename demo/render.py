import torch
import torch.nn as nn
from gaussian_model import Generator, GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
import math
from camera import Camera, extract_cameras


class ImageGenerator(nn.Module):
    def __init__(
        self,
        background,
        image_resolution,
        shs_degree=3,
        use_rgb=False,
        offset=True,
    ):
        super().__init__()
        self.gaussians = Generator(shs_degree, use_rgb, offset)
        self.image_resolution = image_resolution

        self.background: torch.Tensor
        self.register_buffer("background", background)

    def forward(self, sphere, camera):
        with torch.no_grad():
            poses = camera[:, :16].view(-1, 4, 4)
            fovx = camera[:, 16]
            fovy = camera[:, 17]
            cameras = extract_cameras(poses, fovx, fovy, self.image_resolution)

        gaussian_model = self.gaussians(sphere)

        images = []
        for camera in cameras:
            image = render(camera, gaussian_model, self.background, use_rgb=True)
            images.append(image)

        return torch.stack(images, dim=0).contiguous()


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
        debug=False,
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
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return rendered_image * 2 - 1
