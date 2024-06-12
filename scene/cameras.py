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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2Torch

from utils.general_utils import PILtoTorch, NP_resize
from PIL import Image

class Camera(nn.Module):
    def __init__(self, colmap_id, R_gt, T_gt, resolution, FoVx, FoVy, image_path, depth_path,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id

        self.R_gt = R_gt
        self.T_gt = T_gt

        T_est = torch.eye(4, device=device)
        T_est[:3, :3] = torch.Tensor(R_gt)
        T_est[:3, 3] = torch.Tensor(T_gt)

        self.R_est = T_est[:3, :3]
        self.T_est = T_est[:3, 3]

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.depth_path = depth_path
        self.resolution_scale = 1.0


        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        try:
            self.device = torch.device(device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {device} failed, fallback to default cuda device" )
            self.device = torch.device("cuda")

        self.image_width = resolution[0]
        self.image_height = resolution[1]

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.R = torch.tensor(R_gt).to(device)
        self.T = torch.tensor(T_gt).to(device)

        # self.world_view_transform = torch.tensor(getWorld2View2(R_gt, T_gt, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]


    @property
    def world_view_transform(self):

        # return torch.tensor(getWorld2View2(self.R_gt, self.T_gt, self.trans, self.scale)).transpose(0, 1).cuda()
        return getWorld2View2Torch(self.R, self.T).transpose(0, 1)


    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]


    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)


        print("GT R:", self.R_gt)
        print("EST R:", self.R)
        print("GT T:", self.T_gt)
        print("EST T:", self.T)


    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def original_image(self):
        image_pil = Image.open(self.image_path)
        image_torch = PILtoTorch(image_pil, (self.image_width, self.image_height))
    
        image = image_torch[:3, ...]

        gt_alpha_mask = None
        if image_torch.shape[1] == 4:
            gt_alpha_mask = image_torch[3:4, ...]

        image = image.clamp(0.0, 1.0).to(self.device)

        if gt_alpha_mask is not None:
            image *= gt_alpha_mask.to(self.device)
        else:
            image *= torch.ones((1, self.image_height, self.image_width), device=self.device)

        return image


    @property
    def depth(self):

        depth = np.load(self.depth_path)
        resized_depth = NP_resize(depth, (self.image_width, self.image_height))
        resized_depth = torch.Tensor((resized_depth - resized_depth.min())/(resized_depth.max() - resized_depth.min())).cuda()

        return resized_depth


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

