from typing import NamedTuple
import torch


class Camera(NamedTuple):
    FoVx: float
    FoVy: float
    world_view_transform: torch.Tensor
    full_proj_transform: torch.Tensor
    camera_center: torch.Tensor
    image_height: int = 800
    image_width: int = 800


def extract_cameras(camera_to_world, FoVx, FoVy, image_size=128) -> list[Camera]:
    # camera_to_world[:, :3, 1:3] *= -1
    w2c = torch.inverse(camera_to_world)

    world_view_transform = w2c.transpose(-2, -1)
    zfar = 100.0
    znear = 0.01

    # Calculate projection matrix
    projection_matrix = _get_projection_matrices(znear, zfar, FoVx, FoVy).transpose(
        -2, -1
    )

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
                camera_center[i],
                image_height=image_size,
                image_width=image_size
            )
        )

    return cameras


def _get_projection_matrices(znear, zfar, fovX, fovY):
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
