import torch.nn as nn
import torch
from kornia.utils import create_meshgrid
import torch.nn.functional as F
from lib.config import cfg
import math
import cv2
import sys
import torchvision.transforms as T
from PIL import Image
import numpy as np

def depth2point(depth, extrinsic, intrinsic):
    rot = extrinsic[:, :3, :3]
    trans = extrinsic[:, :3, 3:]
    B, H, W = depth.shape
    y, x = torch.meshgrid(torch.linspace(0.5, H-0.5, H, device=depth.device), torch.linspace(0.5, W-0.5, W, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)  
    pts_2d[..., 2] = depth
    pts_2d[:, :, :, 0] -= intrinsic[:, None, None, 0, 2]
    pts_2d[:, :, :, 1] -= intrinsic[:, None, None, 1, 2]
    pts_2d_xy = pts_2d[:, :, :, :2] * pts_2d[:, :, :, 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= (intrinsic[:, 0, 0][:, None, None]+1e-8)
    pts_2d[..., 1] /= (intrinsic[:, 1, 1][:, None, None]+1e-8)
    
    pts_2d = pts_2d.view(B, -1, 3).permute(0, 2, 1)
    rot_t = rot.permute(0, 2, 1)
    pts = torch.bmm(rot_t, pts_2d) - torch.bmm(rot_t, trans)
    return pts.permute(0, 2, 1)


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm2d):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=nn.BatchNorm3d):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

def get_proj_mats(batch, src_scale, tar_scale):
    B, S_V, C, H, W = batch['src_inps'].shape
    src_ext = batch['src_exts']
    src_ixt = batch['src_ixts'].clone()
    src_ixt[:, :, :2] *= src_scale
    src_projs = src_ixt @ src_ext[:, :, :3]
    tar_ext = batch['tar_ext']
    tar_ixt = batch['tar_ixt'].clone()
    tar_ixt[:, :2] *= tar_scale
    tar_projs = tar_ixt @ tar_ext[:, :3]
    tar_ones = torch.zeros((B, 1, 4)).to(tar_projs.device)
    tar_ones[:, :, 3] = 1
    tar_projs = torch.cat((tar_projs, tar_ones), dim=1)
    tar_projs_inv = torch.inverse(tar_projs)

    src_projs = src_projs.view(B, S_V, 3, 4)
    tar_projs_inv = tar_projs_inv.view(B, 1, 4, 4)

    proj_mats = src_projs @ tar_projs_inv
    return proj_mats

def homo_warp(src_feat, proj_mat, depth_values):
    B, D, H_T, W_T = depth_values.shape
    C, H_S, W_S = src_feat.shape[1:]
    device = src_feat.device

    R = proj_mat[:, :, :3] # (B, 3, 3)
    T = proj_mat[:, :, 3:] # (B, 3, 1)
    # create grid from the ref frame
    ref_grid = create_meshgrid(H_T, W_T, normalized_coordinates=False,
                               device=device) # (1, H, W, 2)
    ref_grid = ref_grid.permute(0, 3, 1, 2) # (1, 2, H, W)
    ref_grid = ref_grid.reshape(1, 2, H_T*W_T) # (1, 2, H*W)
    ref_grid = ref_grid.expand(B, -1, -1) # (B, 2, H*W)
    ref_grid = torch.cat((ref_grid, torch.ones_like(ref_grid[:,:1])), 1) # (B, 3, H*W)
    ref_grid_d = ref_grid.repeat(1, 1, D) # (B, 3, D*H*W)
    src_grid_d = R @ ref_grid_d + T/depth_values.view(B, 1, D*H_T*W_T)
    del ref_grid_d, ref_grid, proj_mat, R, T, depth_values # release (GPU) memory

    src_grid = src_grid_d[:, :2] / torch.clamp_min(src_grid_d[:, 2:], 1e-6) # divide by depth (B, 2, D*H*W)
    src_grid[:, 0] = (src_grid[:, 0])/((W_S - 1) / 2) - 1 # scale to -1~1
    src_grid[:, 1] = (src_grid[:, 1])/((H_S - 1) / 2) - 1 # scale to -1~1
    src_grid = src_grid.permute(0, 2, 1) # (B, D*H*W, 2)
    src_grid = src_grid.view(B, D, H_T*W_T, 2)

    warped_src_feat = F.grid_sample(src_feat, src_grid,
                                    mode='bilinear', padding_mode='zeros',
                                    align_corners=True) # (B, C, D, H*W)
    warped_src_feat = warped_src_feat.view(B, C, D, H_T, W_T)
    src_grid = src_grid.view(B, D, H_T, W_T, 2)
    if warped_src_feat.isnan().any():
        __import__('ipdb').set_trace()
    return warped_src_feat, src_grid


def get_depth_values(batch, D, level, device, depth, std, near_far):
    B = len(batch['src_inps'])
    H, W = batch['src_inps'].shape[-2:]
    v_s = cfg.mvsgs.cas_config.volume_scale[level]
    H, W = int(H * v_s), int(W * v_s)
    if depth is None:
        if cfg.mvsgs.cas_config.depth_inv[level]:
            disp_values = 1./batch['near_far'][:, :1] + (torch.linspace(0., 1., steps=D, device=device, dtype=torch.float32).view(1, -1).repeat(B, 1)) * \
                (1./batch['near_far'][:, 1:] - 1./batch['near_far'][:, :1])
            depth_values = 1./disp_values
        else:
            depth_values = batch['near_far'][:, :1] + (batch['near_far'][:, 1:] - batch['near_far'][:, :1]) * \
            torch.linspace(0., 1., steps=D, device=device, dtype=torch.float32).view(1, -1).repeat(B, 1)
        depth_values = depth_values.view(B, D, 1, 1).repeat(1, 1, H, W)
    else:
        up_scale = cfg.mvsgs.cas_config.volume_scale[level] / cfg.mvsgs.cas_config.volume_scale[level-1]
        if up_scale != 1.:
            depth = F.interpolate(depth[:, None], None, scale_factor=up_scale, recompute_scale_factor=True, align_corners=True, mode='bilinear')[:, 0]
            std = F.interpolate(std[:, None], None, scale_factor=up_scale, recompute_scale_factor=True, align_corners=True, mode='bilinear')[:, 0]
            near_far = F.interpolate(near_far, None, scale_factor=up_scale, recompute_scale_factor=True, align_corners=True, mode='bilinear')
        if cfg.mvsgs.cas_config.depth_inv[level-1]:
            near_far_ = torch.stack([depth + std, depth-std], dim=-1)
            mask = near_far_[..., 0] > near_far[:, 0]
            near_far_[..., 0][mask] = near_far[:, 0][mask]
            mask = near_far_[..., 1] < near_far[:, 1]
            near_far_[..., 1][mask] = near_far[:, 1][mask]
            near_far = 1. / near_far_
        else:
            __import__('ipdb').set_trace()
            near_far_ = torch.stack([depth-std, depth+std])
            mask = near_far_[..., 0] < near_far[:, 0]
            near_far_[..., 0][mask] = near_far[:, 0][mask]
            mask = near_far_[..., 1] > near_far[:, 1]
            near_far_[..., 1][mask] = near_far[:, 1][mask]

        if cfg.mvsgs.cas_config.depth_inv[level]:
            linspace = torch.linspace(0., 1., steps=D, device=device, dtype=torch.float32)
            linspace = linspace.view(1, 1, 1, -1).repeat(B, 1, 1, 1)
            disp = 1./near_far[..., :1] + linspace * (1./near_far[..., 1:] - 1./near_far[..., :1])
            depth_values = 1./disp
            depth_values = depth_values.permute((0, 3, 1, 2))
        else:
            depth_values = near_far[..., :1] + torch.linspace(0., 1., steps=D, device=device, dtype=torch.float32).view(1, 1, 1, -1).repeat(B, 1, 1, 1) * (near_far[..., 1:] - near_far[..., :1])
            # B H W D
            depth_values = depth_values.permute((0, 3, 1, 2))
            # B D H W
    near_far = depth_values[:, [0, -1], :, :].detach()
    if cfg.mvsgs.cas_config.depth_inv[level]:
        near_far = 1 / torch.clamp_min(near_far, 1e-6)
    return depth_values.contiguous() , near_far

def get_depth_values_composite(batch, D, level, device, layer_inter, layer_idx):
    B = len(batch['src_inps'])
    H, W = batch['src_inps'].shape[-2:]
    v_s = cfg.mvsgs.cas_config.volume_scale[level]
    H, W = int(H * v_s), int(W * v_s)

    if f'depth_{level-1}_{layer_idx}' in layer_inter:
        depth = layer_inter[f'depth_{level-1}_{layer_idx}']
        std = layer_inter[f'std_{level-1}_{layer_idx}']
        near_far = layer_inter[f'near_far_{level-1}_{layer_idx}']
    else:
        depth = None
        std = None
        near_far = None

    batch_near_far = batch['near_far'][:, layer_idx]

    if depth is None:
        if cfg.mvsgs.cas_config.depth_inv[level]:
            disp_values = 1./batch_near_far[:, :1] + (torch.linspace(0., 1., steps=D, device=device, dtype=torch.float32).view(1, -1).repeat(B, 1)) * \
                (1./batch_near_far[:, 1:] - 1./batch_near_far[:, :1])
            depth_values = 1./disp_values
        else:
            depth_values = batch_near_far[:, :1] + (batch_near_far[:, 1:] - batch_near_far_[:, :1]) * \
            torch.linspace(0., 1., steps=D, device=device, dtype=torch.float32).view(1, -1).repeat(B, 1)
        depth_values = depth_values.view(B, D, 1, 1).repeat(1, 1, H, W)
    else:
        up_scale = cfg.mvsgs.cas_config.volume_scale[level] / cfg.mvsgs.cas_config.volume_scale[level-1]
        if up_scale != 1.:
            depth = F.interpolate(depth[:, None], None, scale_factor=up_scale, recompute_scale_factor=True, align_corners=True, mode='bilinear')[:, 0]
            std = F.interpolate(std[:, None], None, scale_factor=up_scale, recompute_scale_factor=True, align_corners=True, mode='bilinear')[:, 0]
            near_far = F.interpolate(near_far, None, scale_factor=up_scale, recompute_scale_factor=True, align_corners=True, mode='bilinear')

        if cfg.mvsgs.cas_config.depth_inv[level-1]:
            near_far_ = torch.stack([depth + std, depth-std], dim=-1)
            mask = near_far_[..., 0] > near_far[:, 0]
            near_far_[..., 0][mask] = near_far[:, 0][mask]
            mask = near_far_[..., 1] < near_far[:, 1]
            near_far_[..., 1][mask] = near_far[:, 1][mask]
            near_far = 1. / near_far_
        else:
            __import__('ipdb').set_trace()
            near_far_ = torch.stack([depth-std, depth+std])
            mask = near_far_[..., 0] < near_far[:, 0]
            near_far_[..., 0][mask] = near_far[:, 0][mask]
            mask = near_far_[..., 1] > near_far[:, 1]
            near_far_[..., 1][mask] = near_far[:, 1][mask]

        if cfg.mvsgs.cas_config.depth_inv[level]:
            linspace = torch.linspace(0., 1., steps=D, device=device, dtype=torch.float32)
            linspace = linspace.view(1, 1, 1, -1).repeat(B, 1, 1, 1)
            disp = 1./near_far[..., :1] + linspace * (1./near_far[..., 1:] - 1./near_far[..., :1])
            depth_values = 1./disp
            depth_values = depth_values.permute((0, 3, 1, 2))
        else:
            depth_values = near_far[..., :1] + torch.linspace(0., 1., steps=D, device=device, dtype=torch.float32).view(1, 1, 1, -1).repeat(B, 1, 1, 1) * (near_far[..., 1:] - near_far[..., :1])
            # B H W D
            depth_values = depth_values.permute((0, 3, 1, 2))
            # B D H W
    near_far = depth_values[:, [0, -1], :, :].detach()
    if cfg.mvsgs.cas_config.depth_inv[level]:
        near_far = 1 / torch.clamp_min(near_far, 1e-6)
    return depth_values.contiguous() , near_far

def build_rays_composite(depth, std, batch, training, near_far, level, up_scale=2., xywh=None):
    device = depth.device
    up_scale = cfg.mvsgs.cas_config.render_scale[level] / cfg.mvsgs.cas_config.volume_scale[level]
    if up_scale != 1.:
        depth = F.interpolate(depth[:, None], scale_factor=up_scale, mode='bilinear', align_corners=True)[:, 0]
        std = F.interpolate(std[:, None], scale_factor=up_scale, mode='bilinear', align_corners=True)[:, 0]
        near_far = F.interpolate(near_far, scale_factor=up_scale, mode='bilinear', align_corners=True)

    rays = batch[f'rays_{level}']
    if cfg.mvsgs.cas_config.depth_inv[level]:
        rays_near_far = torch.stack([depth+std, depth-std], dim=-1)
        mask = rays_near_far[..., 0] > near_far[:, 0]
        rays_near_far[..., 0][mask] = near_far[:, 0][mask]
        mask = rays_near_far[..., 1] < near_far[:, 1]
        rays_near_far[..., 1][mask] = near_far[:, 1][mask]
    else:
        rays_near_far = torch.stack([depth-std, depth+std], dim=-1)
        mask = rays_near_far[..., 0] < near_far[:, 0]
        rays_near_far[..., 0][mask] = near_far[:, 0][mask]
        mask = rays_near_far[..., 1] > near_far[:, 1]
        rays_near_far[..., 1][mask] = near_far[:, 1][mask]
    near_far = near_far.permute(0, 2, 3, 1)
    rays = batch[f'rays_{level}']
    uv = rays[:, :, 6:].long()
    rays_near_far = torch.stack([rays_near_far[i][uv[i][:, 1], uv[i][:, 0]] for i in range(len(rays_near_far))])
    near_far = torch.stack([near_far[i][uv[i][:, 1], uv[i][:, 0]] for i in range(len(near_far))])
    rays = torch.cat([rays, rays_near_far, near_far], dim=-1)
    B, H, W = depth.shape
    rays = rays.reshape(B, H, W, rays.shape[-1])
    x, y, w, h = (xywh * cfg.mvsgs.cas_config.render_scale[level]).int()
    x, y, w, h = x.item(), y.item(), w.item(), h.item()
    rays = rays[:, y:y+h, x:x+w]
    return rays.reshape(B, -1, rays.shape[-1]), [x, y, w, h], [B,H,W]


def build_feature_volume(feature, batch, D, depth, std, near_far, level):
    B, S, C, H, W = feature.shape
    depth_values, near_far = get_depth_values(batch, D, level, feature.device, depth, std, near_far)
    proj_mats = get_proj_mats(batch, src_scale=cfg.mvsgs.cas_config.im_feat_scale[level], tar_scale=cfg.mvsgs.cas_config.volume_scale[level])
    volume_sum = 0
    volume_sq_sum = 0
    for s in range(S):
        feature_s = feature[:, s]
        proj_mat = proj_mats[:, s]
        warped_volume, _ = homo_warp(feature_s, proj_mat, depth_values)
        volume_sum = volume_sum + warped_volume
        volume_sq_sum = volume_sq_sum + warped_volume ** 2
    volume_variance = volume_sq_sum.div_(S).sub_(volume_sum.div_(S).pow_(2))
    volume_context = volume_sum.div_(S)
    feature_volume = torch.cat((volume_variance, volume_context),dim=1)
    return feature_volume, depth_values, near_far

def build_rays_st(depth, std, batch, training, near_far, level, up_scale=2.):
    device = depth.device
    up_scale = cfg.mvsgs.cas_config.render_scale[level] / cfg.mvsgs.cas_config.volume_scale[level]
    if up_scale != 1.:
        depth = F.interpolate(depth[:, None], scale_factor=up_scale, mode='bilinear', align_corners=True)[:, 0]
        std = F.interpolate(std[:, None], scale_factor=up_scale, mode='bilinear', align_corners=True)[:, 0]
        near_far = F.interpolate(near_far, scale_factor=up_scale, mode='bilinear', align_corners=True)

    rays = batch[f'rays_{level}']
    if cfg.mvsgs.cas_config.depth_inv[level]:
        rays_near_far = torch.stack([depth+std, depth-std], dim=-1)
        mask = rays_near_far[..., 0] > near_far[:, 0]
        rays_near_far[..., 0][mask] = near_far[:, 0][mask]
        mask = rays_near_far[..., 1] < near_far[:, 1]
        rays_near_far[..., 1][mask] = near_far[:, 1][mask]
    else:
        rays_near_far = torch.stack([depth-std, depth+std], dim=-1)
        mask = rays_near_far[..., 0] < near_far[:, 0]
        rays_near_far[..., 0][mask] = near_far[:, 0][mask]
        mask = rays_near_far[..., 1] > near_far[:, 1]
        rays_near_far[..., 1][mask] = near_far[:, 1][mask]
    near_far = near_far.permute(0, 2, 3, 1)
    rays = batch[f'rays_{level}']
    uv = rays[:, :, 6:].long()
    rays_near_far = torch.stack([rays_near_far[i][uv[i][:, 1], uv[i][:, 0]] for i in range(len(rays_near_far))])
    near_far = torch.stack([near_far[i][uv[i][:, 1], uv[i][:, 0]] for i in range(len(near_far))])
    rays = torch.cat([rays, rays_near_far, near_far], dim=-1)

    B, H, W = depth.shape
    rays = rays.reshape(B, H, W, rays.shape[-1])
    x, y, w, h = batch['bbox'][0]
    x, y, w, h = x.item(), y.item(), w.item(), h.item()
    rays = rays[:, y:y+h, x:x+w]
    return rays.reshape(B, -1, rays.shape[-1]), [x, y, w, h], [B,H,W]


def build_rays(depth, std, batch, training, near_far, level, up_scale=2.):
    device = depth.device
    up_scale = cfg.mvsgs.cas_config.render_scale[level] / cfg.mvsgs.cas_config.volume_scale[level]
    if up_scale != 1.:
        depth = F.interpolate(depth[:, None], scale_factor=up_scale, mode='bilinear', align_corners=True)[:, 0]
        std = F.interpolate(std[:, None], scale_factor=up_scale, mode='bilinear', align_corners=True)[:, 0]
        near_far = F.interpolate(near_far, scale_factor=up_scale, mode='bilinear', align_corners=True)

    rays = batch[f'rays_{level}']
    if cfg.mvsgs.cas_config.depth_inv[level]:
        rays_near_far = torch.stack([depth+std, depth-std], dim=-1)
        mask = rays_near_far[..., 0] > near_far[:, 0]
        rays_near_far[..., 0][mask] = near_far[:, 0][mask]
        mask = rays_near_far[..., 1] < near_far[:, 1]
        rays_near_far[..., 1][mask] = near_far[:, 1][mask]
    else:
        rays_near_far = torch.stack([depth-std, depth+std], dim=-1)
        mask = rays_near_far[..., 0] < near_far[:, 0]
        rays_near_far[..., 0][mask] = near_far[:, 0][mask]
        mask = rays_near_far[..., 1] > near_far[:, 1]
        rays_near_far[..., 1][mask] = near_far[:, 1][mask]
    near_far = near_far.permute(0, 2, 3, 1)
    rays = batch[f'rays_{level}']
    uv = rays[:, :, 6:].long()
    rays_near_far = torch.stack([rays_near_far[i][uv[i][:, 1], uv[i][:, 0]] for i in range(len(rays_near_far))])
    near_far = torch.stack([near_far[i][uv[i][:, 1], uv[i][:, 0]] for i in range(len(near_far))])
    rays = torch.cat([rays, rays_near_far, near_far], dim=-1)
    if rays_near_far.isnan().any():
        __import__('ipdb').set_trace()
    return rays

def sample_along_depth(rays, N_samples, level):
    ray_o, ray_d, uv = rays[..., :3], rays[..., 3:6], rays[..., 6:8]
    ray_near, ray_far, near, far = rays[..., 8:9], rays[..., 9:10], rays[..., 10:11], rays[..., 11:12]
    if N_samples == 1:
        z_vals =  ray_near + (ray_far - ray_near) * 0.5
    else:
        z_vals =  ray_near + (ray_far - ray_near) * torch.linspace(0., 1., N_samples, device=rays.device)[None, None]
    if cfg.mvsgs.cas_config.depth_inv[level]:
        world_xyz = ray_o[..., None, :] + ray_d[..., None, :] * (1/torch.clamp_min(z_vals[..., None], 1e-6))
    else:
        world_xyz = ray_o[..., None, :] + ray_d[..., None, :] * (z_vals[..., None])

    if cfg.mvsgs.cas_config.depth_inv[level]:
        d = (near - z_vals) / torch.clamp_min(near - far, 1e-6)
    else:
        d = (z_vals - near) / torch.clamp_min(far - near, 1e-6)
    uvd = torch.cat([uv[..., None, :].repeat(1, 1, N_samples, 1), d[..., None]], dim=-1)
    if z_vals.max() > 1e5:
        __import__('ipdb').set_trace()
    return world_xyz, uvd, z_vals

def get_norm_space(xyd, feature_volume, batch, N_samples, rays, rgbs, training):
    B, C, D, H, W = feature_volume.shape
    H, W = 2 * H, 2 * W
    xyd[..., 0] = xyd[..., 0]  / (W - 1)
    xyd[..., 1] = xyd[..., 1]  / (H - 1)
    xyd[..., 2] = (xyd[..., 2] - rays[..., 4:5]) / torch.clamp_min((rays[..., 5:6] - rays[..., 4:5]), 1e-6)
    return xyd


def get_vox_feat(ndc_xyz, feature_volume):
    feature = F.grid_sample(feature_volume, ndc_xyz[:, None, None]*2. - 1., align_corners=True)[:, :, 0, 0].permute((0, 2, 1))
    return feature
  
def raw2outputs_ngp(raw, z_vals, white_bkgd=False, raydir=None, N_samples=32):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    dists_z = z_vals * raydir.norm(dim=-1, keepdim=True)
    dists = dists_z[...,1:] - dists_z[...,:-1]
    dists = torch.cat([dists, dists[..., -1:]], dim=-1)
    dists[..., :-N_samples] = 1.

    raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
    alpha = raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
    rgb = raw[..., :3]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    T = torch.cumprod(1.-alpha+1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
    weights = alpha * T

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # weights = F.softmax(weights, dim=-1)
    if z_vals is not None:
        depth_map = torch.sum(weights*z_vals.detach(), -1)
    else:
        depth_map = None

    if white_bkgd:
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return {'rgb': rgb_map, 'depth': depth_map, 'weights': weights}

def raw2outputs_layer(layers, batch, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    net_output = layers[0]['net_output']
    z_vals = layers[0]['z_vals']
    for i in range(1, len(layers)):
        net_output = torch.cat([net_output, layers[i]['net_output']], dim=-2)
        z_vals = torch.cat([z_vals, layers[i]['z_vals']], dim=-1)
    z_vals_ori = z_vals
    z_vals, idx = torch.sort(z_vals, dim=-1)
    raw = net_output.gather(dim=2, index=idx[..., None].repeat(1, 1, 1, 4))

    raw2alpha = lambda raw: 1.-torch.exp(-raw)
    alpha = raw2alpha(raw[...,3])  # [N_rays, N_samples]
    rgb = raw[..., :3]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    T = torch.cumprod(1.-alpha+1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
    weights = alpha * T

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    # weights = F.softmax(weights, dim=-1)
    if z_vals is not None:
        depth_map = torch.sum(weights*z_vals.detach(), -1)
    else:
        depth_map = None

    if white_bkgd:
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return {'rgb': rgb_map, 'depth': depth_map, 'weights': weights, 'idx': idx, 'net_output': net_output, 'z_vals': z_vals_ori}


def raw2outputs(raw, z_vals, white_bkgd=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    raw2alpha = lambda raw: 1.-torch.exp(-raw)
    alpha = raw2alpha(raw[...,3])  # [N_rays, N_samples]
    rgb = raw[..., :3]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    T = torch.cumprod(1.-alpha+1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
    weights = alpha * T

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    if z_vals is not None:
        weights = F.softmax(weights, dim=-1)
        depth_map = torch.sum(weights*z_vals.detach(), -1)
    else:
        depth_map = None

    if white_bkgd:
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return {'rgb': rgb_map, 'depth': depth_map, 'weights': weights}

def unpreprocess(data, shape=(1, 1, 3, 1, 1), render_scale=1.):
    device = data.device
    # mean = torch.tensor([-0.485/0.229, -0.456/0.224, -0.406 / 0.225]).view(*shape).to(device)
    # std = torch.tensor([1 / 0.229, 1 / 0.224, 1 / 0.225]).view(*shape).to(device)
    img = data * 0.5 + 0.5
    B, S, C, H, W = img.shape
    img = F.interpolate(img.reshape(B*S, C, H, W), scale_factor=render_scale, align_corners=True, mode='bilinear', recompute_scale_factor=True).reshape(B, S, C, int(H*render_scale), int(W*render_scale))
    return img

def depth_regression(depth_prob, depth_values, level, batch):
    H, W = depth_values.shape[2:]

    if level == -1:
        inter = 9
        D = depth_values.shape[1]
        argsort = depth_prob.argsort(dim=1)
        idx = argsort[:, D-1:]

        mask = torch.zeros_like(depth_prob)
        for offset in range(-inter, inter+1):
            mask.scatter_(1, torch.clamp(idx+offset, 0, mask.shape[1]-1), 1)
        depth_prob[mask!=1] = -10

    prob_volume = F.softmax(depth_prob, 1)
    if cfg.mvsgs.cas_config.depth_inv[level]:
        depth_values = 1./torch.clamp_min(depth_values, 1e-6) # to disp
    depth = torch.sum(prob_volume * depth_values, 1)
    var =  (prob_volume * (depth_values - depth.unsqueeze(1))**2).sum(1)
    std = torch.clamp_min(var, 1e-10).sqrt()
    # std = var .sqrt()
    # vis_prob(std, depth, prob_volume, depth_values)
    return depth, std

def vis_prob(std, depth, prob, depth_v):
    import matplotlib.pyplot as plt
    def vis(u, v):
        y = prob[0, :, u, v].cpu().numpy()
        x = depth_v[0, :, u, v].cpu().numpy()

        plt.subplot(131)
        plt.imshow(depth[0].detach().cpu().numpy())
        plt.plot([v], [u], '.')

        plt.subplot(132)
        plt.imshow(std[0].detach().cpu().numpy())
        plt.plot([v], [u], '.')

        plt.subplot(133)
        plt.plot(x, y, '.-')
        plt.plot([(depth[0, u, v]-std[0, u, v]).item(), (depth[0, u, v]+std[0, u, v]).item()], [0.1, 0.1], '-')
        plt.show()
    __import__('ipdb').set_trace()

def get_img_feat(xyz, img_feat_rgb, batch, training, level):# B * N * S * (11+4)
    # img_feat_rgb_dir = get_img_feat(xyd, img_feat_rgb, batch) # B * N * S * (11+4)
    B, S, C, H, W = img_feat_rgb.shape
    B, N_rays, N_samples = xyz.shape[:3]
    xyz = xyz.view(B, N_rays * N_samples, xyz.shape[-1])
    xyz = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
    render_scale = cfg.mvsgs.cas_config.render_scale[level]

    ret_feat = []
    for i in range(S):
        xyz_img = (xyz @ batch['src_exts'][:, i].transpose(-1, -2))[..., :3]
        ixt_img = batch['src_ixts'][:, i].clone()
        ixt_img[:, :2] *= render_scale
        xyz_img = xyz_img[..., :3] @ ixt_img.transpose(-1, -2)
        grid = xyz_img[..., :2] / torch.clamp_min(xyz_img[..., 2:], 1e-6)
        grid[..., 0], grid[..., 1] = (grid[..., 0]) / (W - 1), (grid[..., 1]) / (H - 1)
        grid = grid * 2. - 1.
        feat = F.grid_sample(img_feat_rgb[:, i], grid[:, None], align_corners=True, mode='bilinear', padding_mode='border').permute(0, 2, 3, 1)[:, 0]
        tar_cam_xyz = batch['tar_ext'].inverse()[:, :3, 3]
        src_cam_xyz = batch['src_exts'][:, i].inverse()[:, :3, 3]
        tar_diff = xyz[..., :3] - tar_cam_xyz[:, None]
        src_diff = xyz[..., :3] - src_cam_xyz[:, None]

        tar_diff = tar_diff / (torch.norm(tar_diff, dim=-1, keepdim=True) + 1e-6)
        src_diff = src_diff / (torch.norm(src_diff, dim=-1, keepdim=True) + 1e-6)

        ray_diff = tar_diff - src_diff
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)

        ray_diff_dot = torch.sum(tar_diff * src_diff, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ret_feat.append(torch.cat([feat, ray_diff], dim=-1))
    return torch.stack(ret_feat, -2)

def sample_points_on_plane(rays, batch_near_far, N_samples=32):
    bounds = torch.tensor([[-4., -4., -0.2],
                           [4., 4., 0.2]]).to(rays.device)[None]

    ray_o_z = rays[..., 2]
    ray_d_z = rays[..., 5]
    ray_d_z[(ray_d_z>-1e-5) & (ray_d_z<1e-10) ] = -1e-5
    ray_d_z[(ray_d_z<1e-5) & (ray_d_z>-1e-10) ] = 1e-5

    near = (bounds[:, 1, 2] - ray_o_z) / ray_d_z
    far = (bounds[:, 0, 2] - ray_o_z) / ray_d_z
    z_vals = near[..., None] + (far - near)[..., None] * torch.linspace(0., 1., N_samples, device=near.device)[None, None, :]
    points = rays[..., :3][..., None, :] + rays[..., 3:6][..., None, :] * z_vals[..., None]

    points = (points - bounds[:, :1, None]) / (bounds[:, 1:2, None] - bounds[:, :1, None])
    return points, z_vals

def sample_points_along_sphere(rays, batch_near_far, N_samples=32):
    r1, r2 = 3., 4.
    o = rays[..., :2]
    d = rays[..., 3:5]
    d_n = d / d.norm(dim=-1, keepdim=True)
    x1 = (-o*d_n).sum(dim=-1)
    x2_2 = torch.clamp_min((o**2).sum(dim=-1) - (x1**2), 1e-6)
    if (r1**2 < x2_2).any():
        x2_2[r1**2 < x2_2] = r1**2 - 1e-4
    x3 = torch.sqrt(r1**2 - x2_2)
    x4 = torch.sqrt(r2**2 - x2_2)
    near = (x1 + x3) / d.norm(dim=-1)
    far = (x1 + x4) / d.norm(dim=-1)
    z_vals = near[..., None] + (far - near)[..., None] * torch.linspace(0., 1., N_samples, device=near.device)[None, None, :]
    points = rays[..., :3][..., None, :] + rays[..., 3:6][..., None, :] * z_vals[..., None]

    points_xyz = points[..., :2]
    points_xyz = points_xyz / points_xyz.norm(dim=-1, keepdim=True)
    theta = torch.zeros_like(points_xyz[..., 0])
    theta[points_xyz[..., 1] >= 0] = torch.arcsin(points_xyz[..., 0][points_xyz[..., 1] >= 0])
    theta[points_xyz[..., 1] < 0] =  math.pi-torch.arcsin(points_xyz[..., 0][points_xyz[..., 1] < 0])
    theta = (theta + math.pi/2.) / (2 * math.pi)
    y = (points[..., :2].norm(dim=-1) - r1 + 0.1) / (r2 - r1 + 0.2)
    z = ((points[..., 2] + 0.2) / 2.4)
    return torch.stack([theta, y, z], dim=-1), z_vals


SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = [
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
]
SH_C3 = [
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
]
SH_C4 = [
    2.5033429417967046,
    -1.7701307697799304,
    0.9461746957575601,
    -0.6690465435572892,
    0.10578554691520431,
    -0.6690465435572892,
    0.47308734787878004,
    -1.7701307697799304,
    0.6258357354491761,
]

MAX_SH_BASIS = 10
def eval_sh_bases(basis_dim : int, dirs : torch.Tensor):
    """
    Evaluate spherical harmonics bases at unit directions,
    without taking linear combination.
    At each point, the final result may the be
    obtained through simple multiplication.
    :param basis_dim: int SH basis dim. Currently, 1-25 square numbers supported
    :param dirs: torch.Tensor (..., 3) unit directions
    :return: torch.Tensor (..., basis_dim)
    """
    result = torch.empty((*dirs.shape[:-1], basis_dim), dtype=dirs.dtype, device=dirs.device)
    result[..., 0] = SH_C0
    if basis_dim > 1:
        x, y, z = dirs.unbind(-1)
        result[..., 1] = -SH_C1 * y;
        result[..., 2] = SH_C1 * z;
        result[..., 3] = -SH_C1 * x;
        if basis_dim > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z
            result[..., 4] = SH_C2[0] * xy;
            result[..., 5] = SH_C2[1] * yz;
            result[..., 6] = SH_C2[2] * (2.0 * zz - xx - yy);
            result[..., 7] = SH_C2[3] * xz;
            result[..., 8] = SH_C2[4] * (xx - yy);

            if basis_dim > 9:
                result[..., 9] = SH_C3[0] * y * (3 * xx - yy);
                result[..., 10] = SH_C3[1] * xy * z;
                result[..., 11] = SH_C3[2] * y * (4 * zz - xx - yy);
                result[..., 12] = SH_C3[3] * z * (2 * zz - 3 * xx - 3 * yy);
                result[..., 13] = SH_C3[4] * x * (4 * zz - xx - yy);
                result[..., 14] = SH_C3[5] * z * (xx - yy);
                result[..., 15] = SH_C3[6] * x * (xx - 3 * yy);

                if basis_dim > 16:
                    result[..., 16] = SH_C4[0] * xy * (xx - yy);
                    result[..., 17] = SH_C4[1] * yz * (3 * xx - yy);
                    result[..., 18] = SH_C4[2] * xy * (7 * zz - 1);
                    result[..., 19] = SH_C4[3] * yz * (7 * zz - 3);
                    result[..., 20] = SH_C4[4] * (zz * (35 * zz - 30) + 3);
                    result[..., 21] = SH_C4[5] * xz * (7 * zz - 3);
                    result[..., 22] = SH_C4[6] * (xx - yy) * (7 * zz - 1);
                    result[..., 23] = SH_C4[7] * xz * (xx - 3 * yy);
                    result[..., 24] = SH_C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy));
    return result

def parse_layer(layer, batch, hw, level, layer_idx):
    B, N_rays, N_samples, _ = layer['net_output'].shape
    H, W = hw
    H, W = int(H*cfg.mvsgs.cas_config.render_scale[level]), int(W*cfg.mvsgs.cas_config.render_scale[level])
    x, y, w, h = (batch['bbox'][0][layer_idx] * cfg.mvsgs.cas_config.render_scale[level]).int()
    x, y, w, h = x.item(), y.item(), w.item(), h.item()
    net_output = torch.zeros((B, H, W, N_samples, 4)).to(layer['net_output'].device)
    z_vals = torch.zeros((B, H, W, N_samples)).to(net_output.device)
    net_output[:, y:y+h, x:x+w] = layer['net_output'].reshape(B, h, w, N_samples, 4)
    z_vals[:, y:y+h, x:x+w] = layer['z_vals'].reshape(B, h, w, N_samples)
    net_output = net_output.reshape(B, -1, N_samples, 4)
    z_vals = z_vals.reshape(B, -1, N_samples)
    return net_output, z_vals

def raw2outputs_composite(layers, batch, hw=(512, 512), level=0, white_bkgd=False, num_fg_layers=1):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    net_output, z_vals = parse_layer(layers[0], batch, hw, level, 0)
    layers[0]['net_output'], layers[0]['z_vals'] = net_output, z_vals
    for i in range(1, num_fg_layers):
        net_output_, z_vals_ = parse_layer(layers[i], batch, hw, level, i)
        layers[i]['net_output'], layers[i]['z_vals'] = net_output_, z_vals_
        net_output = torch.cat([net_output, net_output_], dim=-2)
        z_vals = torch.cat([z_vals, z_vals_], dim=-1)
    z_vals_ori = z_vals
    if num_fg_layers > 1:
        z_vals, idx = torch.sort(z_vals, dim=-1)
        net_output = net_output.gather(dim=2, index=idx[..., None].repeat(1, 1, 1, 4))
    else:
        idx = None
        pass

    net_output = torch.cat([net_output, layers[-1]['net_output']], dim=-2)
    z_vals = torch.cat([z_vals, layers[-1]['z_vals']], dim=-1)

    raw = net_output

    raw2alpha = lambda raw: 1.-torch.exp(-raw)
    alpha = raw2alpha(raw[...,3])  # [N_rays, N_samples]
    rgb = raw[..., :3]
    T = torch.cumprod(1.-alpha+1e-10, dim=-1)[..., :-1]
    T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
    weights = alpha * T

    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
    if z_vals is not None:
        depth_map = torch.sum(weights*z_vals.detach(), -1)
    else:
        depth_map = None

    if white_bkgd:
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None])

    return {'rgb': rgb_map, 'depth': depth_map, 'weights': weights, 'net_output': net_output, 'idx': idx, 'z_vals': z_vals_ori}



def write_cam(file, ixt, ext):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(ext[i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(ixt[i][j]) + ' ')
        f.write('\n')

    f.close()


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
    x = (255 * x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]

