import open3d as o3d


import torch
import numpy as np

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PerspectiveCameras, RasterizationSettings, MeshRenderer, MeshRasterizer, SoftSilhouetteShader, HardPhongShader, BlendParams
)
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex


def draw_gaussians_open3d(points, colors):
    points_np = points.clone().detach().cpu().numpy()
    colors_np = colors.clone().detach().cpu().numpy()

    # Create an Open3D point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.colors = o3d.utility.Vector3dVector(colors_np)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

    return


def generate_mask_image(mesh, K, transform, image_width, image_height):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh = mesh.to(device)
    image_size = (image_height, image_width)
    # image_size = torch.tensor([[image_height, image_width]], device=device)

    # Convert the Open3D mesh to PyTorch3D mesh

    # Define the intrinsic parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    R = transform[:3, :3].clone().to(device).unsqueeze(0)
    T = transform[3, :3].clone().to(device).unsqueeze(0)

    # Set up the camera
    cameras = PerspectiveCameras(
        device=device,
        focal_length=((fx, fy),),
        principal_point=((cx, cy),),
        R=R,
        T=T,
        image_size=((image_size),),
        in_ndc=False
    )

    # Set up the renderer
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    
    blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0, 0))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=blend_params)
    )

    # Render the silhouette
    silhouette = renderer(mesh)

    
    # Convert the rendered image to numpy array and generate the mask
    silhouette_np = silhouette[..., 3].cpu().numpy()  # Get the alpha channel

    mask = np.where(silhouette_np > 0, 1, 0).astype(np.bool)  # Binary mask

    return mask

def o3d_to_pytorch3d_mesh(poisson_mesh):
    """
    Convert Open4D mesh to PyTorch3D mesh.
    """
    vertices = torch.tensor(np.asarray(poisson_mesh.vertices), dtype=torch.float32)
    faces = torch.tensor(np.asarray(poisson_mesh.triangles), dtype=torch.int64)
    
    # Create a texture with the same color for each vertex
    textures = TexturesVertex(verts_features=0.5*torch.ones_like(vertices)[None]) 

    return Meshes(verts=[vertices], faces=[faces], textures=textures)

def get_o3d_pcd_from_points(points, colors, normals):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy())
    pcd.normals = o3d.utility.Vector3dVector(normals.detach().cpu().numpy())

    return pcd

def get_o3d_pcd_from_images(K, depth, color=None, normal=None, depth_scale=None, T_wc=None, mask=None):

    H = depth.shape[0]
    W = depth.shape[1]

    xs = torch.arange(W)
    ys = torch.arange(H)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    grid_z = torch.ones_like(grid_x)
    grid_w = torch.ones_like(grid_x).float()

    if mask is not None:
        grid_x = grid_x[mask]
        grid_y = grid_y[mask]
        grid_z = grid_z[mask]
        grid_w = grid_w[mask]

        depth = depth[mask]

        if color is not None:
            color = color[:, mask]
        if normal is not None:
            normal = normal[:, mask]

    if T_wc is None:
        R = torch.eye(3)
        T = torch.zeros(3)
    else:
        R = T_wc[:3, :3]
        T = T_wc[:3, 3]

    if depth_scale is None:
        depth_scale = 1.0

    transform = torch.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = T

    rays = torch.cat([grid_x.unsqueeze(0), grid_y.unsqueeze(0), grid_z.unsqueeze(0), grid_w.unsqueeze(0)], dim=0).float()
    rays[:3] = rays[:3] * depth * depth_scale
    rays = rays.reshape(4, -1)

    K_hom = torch.eye(4)
    K_hom[:3,:3] = K

    pts_c_hom = torch.inverse(K_hom) @ rays
    pts_w_hom = transform @ pts_c_hom

    pts_w = pts_w_hom[:3, :].transpose(1,0)


    pcd = o3d.geometry.PointCloud()

    points = pts_w.numpy()
    pcd.points = o3d.utility.Vector3dVector(points)

    if normal is not None:
        normals_c = normal.reshape(3, -1)
        normals = (R @ normals_c).transpose(1,0).detach().clone().numpy()

        pcd.normals = o3d.utility.Vector3dVector(normals)

    if color is not None:
        colors = color.reshape(3, -1).transpose(1,0).detach().clone().numpy()

        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


