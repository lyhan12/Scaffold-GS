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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, NP_resize
from utils.graphics_utils import fov2focal

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):

    assert args.resolution == 1 or args.resolution == -1
    assert resolution_scale == 1.0

    # resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    # gt_image = resized_image_rgb[:3, ...]
    # loaded_mask = None

    # # print(f'gt_image: {gt_image.shape}')
    # if resized_image_rgb.shape[1] == 4:
    #     loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R_gt=cam_info.R, T_gt=cam_info.T, resolution=(cam_info.width, cam_info.height),
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, image_path=cam_info.image_path, depth_path=cam_info.depth_path, normal_path=cam_info.normal_path,
                  image_name=cam_info.image_name, uid=id, device=args.data_device)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry



# opengl to opencv transformation matrix
OPENGL_TO_OPENCV = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


# ndc space is x to the right y up. uv space is x to the right, y down.
def pix2ndc_x(x, W):
    x = x.float()
    return (2 * x) / W - 1


def pix2ndc_y(y, H):
    y = y.float()
    return 1 - (2 * y) / H


# ndc is y up and x right. uv is y down and x right
def ndc2pix_x(x, W):
    return (x + 1) * 0.5 * W


def ndc2pix_y(y, H):
    return (1 - y) * 0.5 * H


def euclidean_to_z_depth(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    device: torch.device,
) -> Tensor:
    """Convert euclidean depths to z_depths given camera intrinsics"""
    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
    image_coords = get_camera_coords(img_size=img_size)
    image_coords = image_coords.to(device)

    z_depth = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    z_depth[:, 0] = (image_coords[:, 0] - cx) / fx  # x
    z_depth[:, 1] = (image_coords[:, 1] - cy) / fy  # y
    z_depth[:, 2] = 1  # z

    z_depth = z_depth / torch.norm(z_depth, dim=-1, keepdim=True)
    z_depth = (z_depth * depths)[:, 2]  # pick only z component

    z_depth = z_depth[..., None]
    z_depth = z_depth.view(img_size[1], img_size[0], 1)

    return z_depth


def get_camera_coords(img_size: tuple, pixel_offset: float = 0.5) -> Tensor:
    """Generates camera pixel coordinates [W,H]

    Returns:
        stacked coords [H*W,2] where [:,0] corresponds to W and [:,1] corresponds to H
    """

    # img size is (w,h)
    image_coords = torch.meshgrid(
        torch.arange(img_size[0]),
        torch.arange(img_size[1]),
        indexing="xy",  # W = u by H = v
    )
    image_coords = (
        torch.stack(image_coords, dim=-1) + pixel_offset
    )  # stored as (x, y) coordinates
    image_coords = image_coords.view(-1, 2)
    image_coords = image_coords.float()

    return image_coords

def get_means3d_backproj(
    depths: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    device: torch.device,
    c2w: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, List]:
    """Backprojection using camera intrinsics and extrinsics

    image_coords -> (x,y,depth) -> (X, Y, depth)

    Returns:
        Tuple of (means: Tensor, image_coords: Tensor)
    """

    if depths.dim() == 3:
        depths = depths.view(-1, 1)
    elif depths.shape[-1] != 1:
        depths = depths.unsqueeze(-1).contiguous()
        depths = depths.view(-1, 1)
    if depths.dtype != torch.float:
        depths = depths.float()
        c2w = c2w.float()
    if c2w is not None and c2w.device != device:
        c2w = c2w.to(device)

    image_coords = get_camera_coords(img_size)
    image_coords = image_coords.to(device)  # note image_coords is (H,W)

    # TODO: account for skew / radial distortion
    means3d = torch.empty(
        size=(img_size[0], img_size[1], 3), dtype=torch.float32, device=device
    ).view(-1, 3)
    means3d[:, 0] = (image_coords[:, 0] - cx) * depths[:, 0] / fx  # x
    means3d[:, 1] = (image_coords[:, 1] - cy) * depths[:, 0] / fy  # y
    means3d[:, 2] = depths[:, 0]  # z

    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        means3d = means3d[mask]
        image_coords = image_coords[mask]

    if c2w is None:
        c2w = torch.eye(4, 4, device=device)

    # to world coords
    means3d = means3d @ torch.linalg.inv(c2w[..., :3, :3]) + c2w[..., :3, 3]
    return means3d, image_coords


def project_pix(
    p: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    c2w: Tensor,
    device: torch.device,
    return_z_depths: bool = False,
) -> Tensor:
    """Projects a world 3D point to uv coordinates using intrinsics/extrinsics

    Returns:
        uv coords
    """
    if c2w is None:
        c2w = torch.eye((p.shape[0], 4, 4), device=device)  # type: ignore
    if c2w.device != device:
        c2w = c2w.to(device)

    points_cam = (p.to(device) - c2w[..., :3, 3]) @ c2w[..., :3, :3]
    u = points_cam[:, 0] * fx / points_cam[:, 2] + cx  # x
    v = points_cam[:, 1] * fy / points_cam[:, 2] + cy  # y
    if return_z_depths:
        return torch.stack([u, v, points_cam[:, 2]], dim=-1)
    return torch.stack([u, v], dim=-1)


def get_colored_points_from_depth(
    depths: Tensor,
    rgbs: Tensor,
    c2w: Tensor,
    fx: float,
    fy: float,
    cx: int,
    cy: int,
    img_size: tuple,
    mask: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Return colored pointclouds from depth and rgb frame and c2w. Optional masking.

    Returns:
        Tuple of (points, colors)
    """
    points, _ = get_means3d_backproj(
        depths=depths.float(),
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        img_size=img_size,
        c2w=c2w.float(),
        device=depths.device,
    )
    points = points.squeeze(0)
    if mask is not None:
        if not torch.is_tensor(mask):
            mask = torch.tensor(mask, device=depths.device)
        colors = rgbs.view(-1, 3)[mask]
        points = points[mask]
    else:
        colors = rgbs.view(-1, 3)
        points = points
    return (points, colors)


def get_rays_x_y_1(H, W, focal, c2w):
    """Get ray origins and directions in world coordinates.

    Convention here is (x,y,-1) such that depth*rays_d give real z depth values in world coordinates.
    """
    assert c2w.shape == torch.Size([3, 4])
    image_coords = torch.meshgrid(
        torch.arange(W, dtype=torch.float32),
        torch.arange(H, dtype=torch.float32),
        indexing="ij",
    )
    i, j = image_coords
    # dirs = torch.stack([(i-W*0.5)/focal, -(j-H*0.5)/focal, -torch.ones_like(i)], dim = -1)
    dirs = torch.stack(
        [(pix2ndc_x(i, W)) / focal, pix2ndc_y(j, H) / focal, -torch.ones_like(i)],
        dim=-1,
    )
    dirs = dirs.view(-1, 3)
    rays_d = dirs[..., :] @ c2w[:3, :3]
    rays_o = c2w[:3, -1].expand_as(rays_d)

    # return world coordinate rays_o and rays_d
    return rays_o, rays_d


def get_projection_matrix(znear=0.001, zfar=1000, fovx=None, fovy=None, **kwargs):
    """Opengl projection matrix

    Returns:
        projmat: Tensor
    """

    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    return torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, (f + n) / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        **kwargs,
    )
