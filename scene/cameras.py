#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
class Camera:
    def __init__(self, FoVx, FoVy, world_view_transform, full_proj_transform, camera_center):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = camera_center
        self.image_height = 128
        self.image_width = 128

        
