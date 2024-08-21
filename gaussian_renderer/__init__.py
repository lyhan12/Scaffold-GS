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
from einops import repeat
import numpy as np

from torchvision import transforms

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera


from utils.general_utils import build_rotation
import torch.nn.functional as F

import open3d as o3d

from utils.pcd_utils import get_o3d_pcd_from_images, get_o3d_pcd_from_points


from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftSilhouetteShader, HardPhongShader, BlendParams
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from matplotlib import pyplot as plt

    
def visualize_from_viewpoint(pc : GaussianModel, cam : Camera, visible_mask=None):

    depth_scales = pc.get_depth_scale

    if depth_scales is None:
        depth_scale = torch.tensor(1.0, device="cpu")
    else:
        depth_scale = depth_scales[cam.uid].cpu()

    gs_xyz, gs_color, _, gs_scaling, gs_rot = generate_neural_gaussians(cam, pc, visible_mask)

    rotations_mat = build_rotation(gs_rot)
    min_scales = torch.argmin(gs_scaling, dim=1)
    min_indices = torch.arange(min_scales.shape[0])
    gs_normal = rotations_mat[min_indices, :, min_scales]


    depth = cam.depth.cpu()
    normal = cam.normal.cpu()
    color = transforms.Resize(depth.shape)(cam.image.cpu())

    K_depth = cam.K(depth.shape).cpu()
    T_wc = torch.inverse(cam.world_view_transform.transpose(0,1)).cpu()



    if True:
        pcd = get_o3d_pcd_from_images(K_depth, depth, color, normal, depth_scale, T_wc) 
        pcd.colors = o3d.utility.Vector3dVector((np.array(pcd.colors)[:] + np.array([[0.0, 1.0, 0.0]])) / 2.0)

        pcd_raw = get_o3d_pcd_from_images(K_depth, depth, color, normal, 1.0, T_wc) 
        pcd_raw.colors = o3d.utility.Vector3dVector((np.array(pcd_raw.colors)[:] + np.array([[1.0, 0.0, 0.0]])) / 2.0)

        gs_pcd = get_o3d_pcd_from_points(gs_xyz, gs_color, gs_normal)

        # with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        #     gs_poi_mesh, gs_poi_density = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(gs_pcd, depth=10)

        # gs_poi_mask = gs_poi_density < np.quantile(gs_poi_density, 0.5)
        # gs_poi_mesh.remove_vertices_by_mask(gs_poi_mask)
        # mesh = o3d_to_pytorch3d_mesh(gs_poi_mesh)


        # mask_np = generate_mask_image(mesh, K_depth, cam.world_view_transform, depth.shape[1], depth.shape[0]).squeeze()
        # mask = torch.from_numpy(mask_np.copy())
        # mask_empty = ~mask


        # pcd_mask = get_o3d_pcd_from_images(K_depth, depth, color, normal, depth_scale, T_wc, mask_empty) 

        import ipdb
        ipdb.set_trace()

        pcd_raw_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd_raw, voxel_size=0.05)
        gs_pcd_voxel = o3d.geometry.VoxelGrid.create_from_point_cloud(gs_pcd, voxel_size=0.05)


        # plt.imshow(mask_np, cmap='gray')
        # plt.axis('off')
        # plt.show()


        o3d.visualization.draw_geometries([pcd, pcd_raw, gs_pcd])


    return




def generate_neural_gaussians_for_visualization(pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    # dist
    ob_dist = 2.0 * torch.ones([anchor.shape[0], 1], dtype=torch.float, device = anchor.device)

    cat_local_view = torch.cat([feat, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = feat

    if pc.appearance_dim > 0:
        camera_indices = torch.zeros_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device)
        appearance = pc.get_appearance(camera_indices)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets


    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot



def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    # cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3+1]
    # cat_local_view_wodist = torch.cat([feat, ob_view], dim=1) # [N, c+3]

    cat_local_view = torch.cat([feat, ob_dist], dim=1) # [N, c+3+1]
    cat_local_view_wodist = feat

    if pc.appearance_dim > 0:
        camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
        # camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * 10
        appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets


    if False:

        import ipdb
        ipdb.set_trace()

        points_np = xyz.clone().detach().cpu().numpy()
        colors_np = color.clone().detach().cpu().numpy()

        import open3d as o3d

        # Create an Open3D point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

        # Visualize the point cloud
        o3d.visualization.draw_geometries([pcd])


    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, render_normal=True, render_edge=False, render_density=False):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        xyz, color, opacity, scaling, rot = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)

    # print("XYZ Shape:", xyz.shape)

    render_anchor = True
    if render_anchor:
        # import ipdb
        # ipdb.set_trace()
        pass

    

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
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
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        kernel_size=0.0,
        require_depth=True,
        require_coord=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if pc.use_pose_update:
        cam_rot_deltas, cam_trans_deltas = pc.get_pose_parameter
        cam_rot_delta = cam_rot_deltas[viewpoint_camera.uid]
        cam_trans_delta = cam_trans_deltas[viewpoint_camera.uid]
    else:
        cam_rot_delta = torch.zeros(3, device="cuda")
        cam_trans_delta = torch.zeros(3, device="cuda")
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 


    color_image, radii, coord, mcoord, depth_image, middepth_image, opacity_image, normal_image = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)

    result = {"render": color_image,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "scaling": scaling,
                "opacity": opacity_image,
                "depth": depth_image,
                "normal": normal_image,
                }

    if is_training:
        result["neural_opacity"] = neural_opacity
        result["selection_mask"] = mask

    return result



def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
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
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        kernel_size=0.0,
        require_depth=True,
        require_coord=False
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
