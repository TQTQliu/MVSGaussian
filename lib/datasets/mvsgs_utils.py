from lib.config import cfg
import cv2
import numpy as np

def sample_patch(num_patch, patch_size, H, W, msk_sample):
    half_patch_size = patch_size // 2
    if msk_sample.sum() > 0:
        num_fg_patch = num_patch
        non_zero = msk_sample.nonzero()
        permutation = np.random.permutation(msk_sample.sum())[:num_fg_patch].astype(np.int32)
        X_, Y_ = non_zero[1][permutation], non_zero[0][permutation]
        X_ = np.clip(X_, half_patch_size, W-half_patch_size)
        Y_ = np.clip(Y_, half_patch_size, H-half_patch_size)
    else:
        num_fg_patch = 0
    num_patch = num_patch - num_fg_patch
    X = np.random.randint(low=half_patch_size, high=W-half_patch_size, size=num_patch)
    Y = np.random.randint(low=half_patch_size, high=H-half_patch_size, size=num_patch)
    if num_fg_patch > 0:
        X = np.concatenate([X, X_]).astype(np.int32)
        Y = np.concatenate([Y, Y_]).astype(np.int32)
    grid = np.meshgrid(np.arange(patch_size)-half_patch_size, np.arange(patch_size)-half_patch_size)
    return np.concatenate([grid[0].reshape(-1) + x for x in X]), np.concatenate([grid[1].reshape(-1) + y for y in Y])

def build_rays(tar_img, tar_ext, tar_ixt, tar_msk, level, split):
    scale = cfg.mvsgs.cas_config.render_scale[level]
    if scale != 1.:
        tar_img = cv2.resize(tar_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        tar_msk = cv2.resize(tar_msk, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        tar_ixt = tar_ixt.copy()
        tar_ixt[:2] *= scale
    H, W = tar_img.shape[:2]
    c2w = np.linalg.inv(tar_ext)
    if split == 'train' and not cfg.mvsgs.cas_config.train_img[level]:
        if cfg.mvsgs.sample_on_mask: # 313
            msk_sample = tar_msk
            num_fg_rays = int(min(cfg.mvsgs.cas_config.num_rays[level]*0.75, tar_msk.sum()*0.95))
            non_zero = msk_sample.nonzero()
            permutation = np.random.permutation(tar_msk.sum())[:num_fg_rays].astype(np.int32)
            X_, Y_ = non_zero[1][permutation], non_zero[0][permutation]
        else:
            num_fg_rays = 0
            msk_sample = np.zeros_like(tar_msk)
        num_rays = cfg.mvsgs.cas_config.num_rays[level] - num_fg_rays
        X = np.random.randint(low=0, high=W, size=num_rays)
        Y = np.random.randint(low=0, high=H, size=num_rays)
        if num_fg_rays > 0:
            X = np.concatenate([X, X_]).astype(np.int32)
            Y = np.concatenate([Y, Y_]).astype(np.int32)
        if cfg.mvsgs.cas_config.num_patchs[level] > 0:
            X_, Y_ = sample_patch(cfg.mvsgs.cas_config.num_patchs[level], cfg.mvsgs.cas_config.patch_size[level], H, W, msk_sample)
            X = np.concatenate([X, X_]).astype(np.int32)
            Y = np.concatenate([Y, Y_]).astype(np.int32)
        num_rays = len(X)
        rays_o = c2w[:3, 3][None].repeat(num_rays, 0)
        XYZ = np.concatenate((X[:, None], Y[:, None], np.ones_like(X[:, None])), axis=-1)
        XYZ = XYZ @ (np.linalg.inv(tar_ixt).T @ c2w[:3, :3].T)
        rays = np.concatenate((rays_o, XYZ, X[..., None], Y[..., None]), axis=-1)
        rgb = tar_img[Y, X]
        msk = tar_msk[Y, X]
    else:
        rays_o = c2w[:3, 3][None, None]
        X, Y = np.meshgrid(np.arange(W), np.arange(H))
        XYZ = np.concatenate((X[:, :, None], Y[:, :, None], np.ones_like(X[:, :, None])), axis=-1)
        XYZ = XYZ @ (np.linalg.inv(tar_ixt).T @ c2w[:3, :3].T)
        rays_o = rays_o.repeat(H, axis=0)
        rays_o = rays_o.repeat(W, axis=1)
        rays = np.concatenate((rays_o, XYZ, X[..., None], Y[..., None]), axis=-1)
        rgb = tar_img
        msk = tar_msk
    return rays.astype(np.float32).reshape(-1, 8), rgb.reshape(-1, 3), msk.reshape(-1)


