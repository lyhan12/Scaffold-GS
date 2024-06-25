import torch
import numpy as np

def init_image_coord(height, width):
    x_row = np.arange(0, width)
    x = np.tile(x_row, (height, 1))
    x = x[np.newaxis, :, :]
    x = x.astype(np.float32)
    x = torch.from_numpy(x.copy()).cuda()
    u_u0 = x - width/2.0

    y_col = np.arange(0, height)  # y_col = np.arange(0, height)
    y = np.tile(y_col, (width, 1)).T
    y = y[np.newaxis, :, :]
    y = y.astype(np.float32)
    y = torch.from_numpy(y.copy()).cuda()
    v_v0 = y - height/2.0
    return u_u0, v_v0

def depth_to_xyz(depth, intrinsic):
    b, c, h, w = depth.shape
    u_u0, v_v0 = init_image_coord(h, w)
    x = (u_u0 - intrinsic[0][2]) * depth / intrinsic[0][0]
    y = (v_v0 - intrinsic[1][2]) * depth / intrinsic[1][1]
    z = depth
    pw = torch.cat([x, y, z], 1).permute(0, 2, 3, 1) # [b, h, w, c]
    return pw

def get_surface_normalv2(xyz, patch_size=5):
    """
    xyz: xyz coordinates
    patch: [p1, p2, p3,
            p4, p5, p6,
            p7, p8, p9]
    surface_normal = [(p9-p1) x (p3-p7)] + [(p6-p4) - (p8-p2)]
    return: normal [h, w, 3, b]
    """
    b, h, w, c = xyz.shape
    half_patch = patch_size // 2
    xyz_pad = torch.zeros((b, h + patch_size - 1, w + patch_size - 1, c), dtype=xyz.dtype, device=xyz.device)
    xyz_pad[:, half_patch:-half_patch, half_patch:-half_patch, :] = xyz

    # xyz_left_top = xyz_pad[:, :h, :w, :]  # p1
    # xyz_right_bottom = xyz_pad[:, -h:, -w:, :]# p9
    # xyz_left_bottom = xyz_pad[:, -h:, :w, :]   # p7
    # xyz_right_top = xyz_pad[:, :h, -w:, :]  # p3
    # xyz_cross1 = xyz_left_top - xyz_right_bottom  # p1p9
    # xyz_cross2 = xyz_left_bottom - xyz_right_top  # p7p3

    xyz_left = xyz_pad[:, half_patch:half_patch + h, :w, :]  # p4
    xyz_right = xyz_pad[:, half_patch:half_patch + h, -w:, :]  # p6
    xyz_top = xyz_pad[:, :h, half_patch:half_patch + w, :]  # p2
    xyz_bottom = xyz_pad[:, -h:, half_patch:half_patch + w, :]  # p8
    xyz_horizon = xyz_left - xyz_right  # p4p6
    xyz_vertical = xyz_top - xyz_bottom  # p2p8

    xyz_left_in = xyz_pad[:, half_patch:half_patch + h, 1:w+1, :]  # p4
    xyz_right_in = xyz_pad[:, half_patch:half_patch + h, patch_size-1:patch_size-1+w, :]  # p6
    xyz_top_in = xyz_pad[:, 1:h+1, half_patch:half_patch + w, :]  # p2
    xyz_bottom_in = xyz_pad[:, patch_size-1:patch_size-1+h, half_patch:half_patch + w, :]  # p8
    xyz_horizon_in = xyz_left_in - xyz_right_in  # p4p6
    xyz_vertical_in = xyz_top_in - xyz_bottom_in  # p2p8

    n_img_1 = torch.cross(xyz_horizon_in, xyz_vertical_in, dim=3)
    n_img_2 = torch.cross(xyz_horizon, xyz_vertical, dim=3)

    # re-orient normals consistently
    orient_mask = torch.sum(n_img_1 * xyz, dim=3) > 0
    n_img_1[orient_mask] *= -1
    orient_mask = torch.sum(n_img_2 * xyz, dim=3) > 0
    n_img_2[orient_mask] *= -1

    n_img1_L2 = torch.sqrt(torch.sum(n_img_1 ** 2, dim=3, keepdim=True))
    n_img1_norm = n_img_1 / (n_img1_L2 + 1e-8)

    n_img2_L2 = torch.sqrt(torch.sum(n_img_2 ** 2, dim=3, keepdim=True))
    n_img2_norm = n_img_2 / (n_img2_L2 + 1e-8)

    # average 2 norms
    n_img_aver = n_img1_norm + n_img2_norm
    n_img_aver_L2 = torch.sqrt(torch.sum(n_img_aver ** 2, dim=3, keepdim=True))
    n_img_aver_norm = n_img_aver / (n_img_aver_L2 + 1e-8)
    # re-orient normals consistently
    orient_mask = torch.sum(n_img_aver_norm * xyz, dim=3) > 0
    n_img_aver_norm[orient_mask] *= -1
    n_img_aver_norm_out = n_img_aver_norm.permute((1, 2, 3, 0))  # [h, w, c, b]

    # a = torch.sum(n_img1_norm_out*n_img2_norm_out, dim=2).cpu().numpy().squeeze()
    # plt.imshow(np.abs(a), cmap='rainbow')
    # plt.show()
    return n_img_aver_norm_out#n_img1_norm.permute((1, 2, 3, 0))

def surface_normal_from_depth(depth, intrinsic, valid_mask=None):
    # para depth: depth map, [b, c, h, w]
    b, c, h, w = depth.shape
    # focal_length = focal_length[:, None, None, None]
    depth_filter = torch.nn.functional.avg_pool2d(depth, kernel_size=3, stride=1, padding=1)
    depth_filter = torch.nn.functional.avg_pool2d(depth_filter, kernel_size=3, stride=1, padding=1)
    xyz = depth_to_xyz(depth_filter, intrinsic)
    sn_batch = []
    for i in range(b):
        xyz_i = xyz[i, :][None, :, :, :]
        normal = get_surface_normalv2(xyz_i)
        sn_batch.append(normal)
    sn_batch = torch.cat(sn_batch, dim=3).permute((3, 2, 0, 1))  # [b, c, h, w]
    if valid_mask is not None:
        mask_invalid = (~valid_mask).repeat(1, 3, 1, 1)
        sn_batch[mask_invalid] = 0.0

    return sn_batch


def mean_angular_error(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Compute the mean angular error between predicted and reference normals

    Args:
        predicted_normals: [B, C, H, W] tensor of predicted normals
        reference_normals : [B, C, H, W] tensor of gt normals

    Returns:
        mae: [B, H, W] mean angular error
    """
    dot_products = torch.sum(gt * pred, dim=1)  # over the C dimension
    # Clamp the dot product to ensure valid cosine values (to avoid nans)
    dot_products = torch.clamp(dot_products, -1.0, 1.0)
    # Calculate the angle between the vectors (in radians)
    mae = torch.acos(dot_products)
    return mae

