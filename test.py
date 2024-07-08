import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftSilhouetteShader,
    TexturesUV,
    TexturesVertex,
    BlendParams
)


# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

from plot_image_grid import image_grid

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# Set paths
DATA_DIR = "./data"
obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

# Load obj file
mesh = load_objs_as_meshes([obj_filename], device=device)


R, T = look_at_view_transform(2.7, 0, 180) 
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)


blend_params = BlendParams(sigma=1e-4, gamma=1e-4, background_color=(0, 0, 0, 0))

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftSilhouetteShader(
        blend_params=blend_params
        # lights=lights
    )
)
images = renderer(mesh)


# Convert the rendered image to numpy array and extract the alpha channel
alpha_channel = images[0, ..., 3].cpu().numpy()  # Get the alpha channel

# Display the alpha channel
plt.figure(figsize=(10, 10))
plt.imshow(alpha_channel, cmap='gray')
plt.axis("off")
plt.title("Alpha Channel (Silhouette)")
plt.show()

import ipdb
ipdb.set_trace()
