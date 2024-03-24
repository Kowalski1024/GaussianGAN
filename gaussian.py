import torch
from typing import NamedTuple
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


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


def extract_cameras(camera_to_world, intrinsics) -> list[Camera]:
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