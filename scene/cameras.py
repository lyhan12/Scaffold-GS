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
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2Torch, fov2focal

from utils.general_utils import PILtoTorch, NP_resize
from PIL import Image

from utils.pose_utils import update_pose

class Camera(nn.Module):
    def __init__(self, colmap_id, R_gt, T_gt, resolution, FoVx, FoVy, image_path, depth_path, normal_path,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id

        self.R_gt = torch.tensor(R_gt.copy(), device=device)
        self.T_gt = torch.tensor(T_gt.copy(), device=device)

        self.R_init = torch.tensor(R_gt.copy(), device=device)
        self.T_init = torch.tensor(T_gt.copy(), device=device)

        # self.T_init += torch.tensor([0.1, -0.1, 0.1]).cuda()


        # # Generate a random vector
        # rand_vec = np.random.randn(3)

        # # Normalize the random vector to have unit norm
        # rand_dir = rand_vec / np.linalg.norm(rand_vec)
        # norm = 0.3

        # Add the random unit vector to T_gt
        # T = T + norm * rand_dir
        # T = T + np.array([-0.05, 0.05, 0.05])


        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.image_path = image_path
        self.depth_path = depth_path
        self.normal_path = normal_path
        self.resolution_scale = 1.0


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

        self.R = self.R_init.clone()
        self.T = self.T_init.clone()

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

    def K(self, HW = None):
        if HW == None:
            W = self.image_width
            H = self.image_height
        else:
            assert len(HW) == 2
            H = HW[0]
            W = HW[1]

        fx = fov2focal(self.FoVx, W)
        fy = fov2focal(self.FoVy, H)
        cx = .5 * W
        cy = .5 * H

        K = torch.tensor([ [fx, 0, cx], [0, fy, cy], [0, 0, 1] ], dtype=torch.float32).to(self.device)

        return K



    def set_delta_RT(self, cam_rot_delta, cam_trans_delta):
        R, T = update_pose(self.R_init, self.T_init, cam_rot_delta, cam_trans_delta)
        self.R = R
        self.T = T


    # camera.cam_rot_delta.data.fill_(0)
    # camera.cam_trans_delta.data.fill_(0)


        print("GT R:", self.R_gt)
        print("INIT R:", self.R_init)
        print("CUR R:", self.R)
        print("GT T:", self.T_gt)
        print("INIT T:", self.T_init)
        print("CUR T:", self.T)


    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def image(self):
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
        # resized_depth = NP_resize(depth, (self.image_width, self.image_height))
        # resized_depth = torch.Tensor((resized_depth - resized_depth.min())/(resized_depth.max() - resized_depth.min())).cuda()

        return torch.tensor(depth).cuda() #resized_depth

    @property
    def normal(self):
        normal_pil = Image.open(self.normal_path)
        normal = torch.from_numpy(np.array(normal_pil)).permute(2, 0, 1)

        normal = normal.float() / 255.0    

        normal = 2.0 * normal - 1.0
        normal = normal.permute(1,2,0)

        normal = normal @ torch.tensor(
                [[0, 0, 1],
                [0, 1, 0],
                [1, 0, 0]], device=normal.device, dtype=normal.dtype)
        normal = normal.permute(2,0,1)
        normal = torch.nn.functional.normalize(normal, dim=0).cuda()


        return normal


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

