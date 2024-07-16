import numpy as np
import os
from lib.config import cfg
import imageio
import cv2
import random
from lib.utils import data_utils
import torch
import json
from lib.datasets import mvsgs_utils
from lib.utils.video_utils import *

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_h_w = kwargs['input_h_w']
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.scale_factor = cfg.mvsgs.scale_factor
        self.build_metas()
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = [0.0, 0.0, 0.0]
        self.scale = 1.0

    def build_metas(self):
        if len(self.scenes) == 0:
            scenes = ['chair', 'drums', 'ficus', 'hotdog', 'lego', 'materials', 'mic', 'ship']
        else:
            scenes = self.scenes
        self.scene_infos = {}
        self.metas = []
        pairs = torch.load('data/mvsgs/pairs.th')
        for scene in scenes:
            json_info = json.load(open(os.path.join(self.data_root, scene,'transforms_train.json')))
            b2c = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            scene_info = {'ixts': [], 'exts': [], 'img_paths': []}
            for idx in range(len(json_info['frames'])):
                c2w = np.array(json_info['frames'][idx]['transform_matrix'])
                c2w = c2w @ b2c
                ext = np.linalg.inv(c2w)
                ixt = np.eye(3)
                ixt[0][2], ixt[1][2] = 400., 400.
                focal = .5 * 800 / np.tan(.5 * json_info['camera_angle_x'])
                ixt[0][0], ixt[1][1] = focal, focal
                scene_info['ixts'].append(ixt.astype(np.float32))
                scene_info['exts'].append(ext.astype(np.float32))
                img_path = os.path.join(self.data_root, scene, 'train/r_{}.png'.format(idx))
                scene_info['img_paths'].append(img_path)
            self.scene_infos[scene] = scene_info
            train_ids, render_ids = pairs[f'{scene}_train'], pairs[f'{scene}_val']
            if self.split == 'train':
                render_ids = train_ids
            c2ws = np.stack([np.linalg.inv(scene_info['exts'][idx]) for idx in train_ids])
            for idx in render_ids:
                c2w = np.linalg.inv(scene_info['exts'][idx])
                distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)

                argsorts = distance.argsort()
                argsorts = argsorts[1:] if idx in train_ids else argsorts

                input_views_num = cfg.mvsgs.train_input_views[1] + 1 if self.split == 'train' else cfg.mvsgs.test_input_views
                src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                self.metas += [(scene, idx, src_views)]

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if np.random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views, input_views_num)
        scene_info = self.scene_infos[scene]
        scene_info['scene_name'] = scene
        tar_img, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)

        ret = {'src_inps': src_inps.transpose(0, 3, 1, 2),
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        tar_mask = np.ones_like(tar_img[..., 0]).astype(np.uint8)
        H, W = tar_img.shape[:2]
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_mask': tar_mask})
        near_far = np.array([2.5*self.scale_factor , 5.5*self.scale_factor]).astype(np.float32)
        ret.update({'near_far': np.array(near_far).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})

        for i in range(cfg.mvsgs.cas_config.num):
            rays, rgb, msk = mvsgs_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            s = cfg.mvsgs.cas_config.volume_scale[i]
            ret['meta'].update({f'h_{i}': int(H*s), f'w_{i}': int(W*s)})
        
        R = np.array(tar_ext[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(tar_ext[:3, 3], np.float32)
        for i in range(cfg.mvsgs.cas_config.num):
            h, w = H*cfg.mvsgs.cas_config.render_scale[i], W*cfg.mvsgs.cas_config.render_scale[i]
            tar_ixt_ = tar_ixt.copy()
            tar_ixt_[:2,:] *= cfg.mvsgs.cas_config.render_scale[i]
            FovX = data_utils.focal2fov(tar_ixt_[0, 0], w)
            FovY = data_utils.focal2fov(tar_ixt_[1, 1], h)
            projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt_, h=h, w=w).transpose(0, 1)
            world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]
            novel_view_data = {
                'FovX':  torch.FloatTensor([FovX]),
                'FovY':  torch.FloatTensor([FovY]),
                'width': w,
                'height': h,
                'world_view_transform': world_view_transform,
                'full_proj_transform': full_proj_transform,
                'camera_center': camera_center
            }
            ret[f'novel_view{i}'] = novel_view_data    
        
        if cfg.save_video:
            rendering_video_meta = []
            render_path_mode = 'interpolate'            
            poses_paths = self.get_video_rendering_path(ref_poses=src_exts, mode=render_path_mode, near_far=None, train_c2w_all=None, n_frames=60)
            for pose in poses_paths[0]:
                R = np.array(pose[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
                T = np.array(pose[:3, 3], np.float32)
                FovX = data_utils.focal2fov(tar_ixt[0, 0], W)
                FovY = data_utils.focal2fov(tar_ixt[1, 1], H)
                projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt, h=H, w=W).transpose(0, 1)
                world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                rendering_meta = {
                    'FovX':  torch.FloatTensor([FovX]),
                    'FovY':  torch.FloatTensor([FovY]),
                    'width': W,
                    'height': H,
                    'world_view_transform': world_view_transform,
                    'full_proj_transform': full_proj_transform,
                    'camera_center': camera_center,
                    'tar_ext': pose
                }
                for i in range(cfg.mvsgs.cas_config.num):
                    tar_ext[:3] = pose
                    rays, _, _ = mvsgs_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
                    rendering_meta.update({f'rays_{i}': rays})
                rendering_video_meta.append(rendering_meta)
            ret['rendering_video_meta'] = rendering_video_meta
        return ret
        
    def get_video_rendering_path(self, ref_poses, mode, near_far, train_c2w_all, n_frames=60, batch=None):
        # loop over batch
        poses_paths = []
        ref_poses = ref_poses[None]
        for batch_idx, cur_src_poses in enumerate(ref_poses):
            if mode == 'interpolate':
                # convert to c2ws
                pose_square = torch.eye(4).unsqueeze(0).repeat(cur_src_poses.shape[0], 1, 1)
                cur_src_poses = torch.from_numpy(cur_src_poses)
                pose_square[:, :3, :] = cur_src_poses[:,:3]
                cur_c2ws = pose_square.double().inverse()[:, :3, :].to(torch.float32).cpu().detach().numpy()
                cur_path = get_interpolate_render_path(cur_c2ws, n_frames)
            elif mode == 'spiral':
                cur_c2ws_all = train_c2w_all
                cur_near_far = near_far.tolist()
                rads_scale = 0.3
                cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=rads_scale, N_views=n_frames)
            else:
                raise Exception(f'Unknown video rendering path mode {mode}')

            # convert back to extrinsics tensor
            cur_w2cs = torch.tensor(cur_path).inverse()[:, :3].to(torch.float32)
            poses_paths.append(cur_w2cs)

        poses_paths = torch.stack(poses_paths, dim=0)
        return poses_paths

    def read_src(self, scene, src_views):
        src_ids = src_views
        ixts, exts, imgs = [], [], []
        for idx in src_ids:
            img, orig_size = self.read_image(scene, idx)
            imgs.append((img*2-1).astype(np.float32))
            ixt, ext = self.read_cam(scene, idx, orig_size)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        img, orig_size = self.read_image(scene, view_idx)
        ixt, ext = self.read_cam(scene, view_idx, orig_size)
        return img, ext, ixt

    def read_cam(self, scene, view_idx, orig_size):
        ext = scene['exts'][view_idx].astype(np.float32)
        ext[:3,3] *= self.scale_factor 
        ixt = scene['ixts'][view_idx]
        ixt[0] *= self.input_h_w[1] / orig_size[0]
        ixt[1] *= self.input_h_w[0] / orig_size[1]
        # ext[:3,3] *= self.scale_factor 
        
        # ext_ = ext.copy()
        # ext_[:3,3] *= self.scale_factor 
        return ixt, ext

    def read_image(self, scene, view_idx):
        img_path = scene['img_paths'][view_idx]
        img = (np.array(imageio.imread(img_path)) / 255.).astype(np.float32)
        img = (img[..., :3] * img[..., -1:] + (1 - img[..., -1:])).astype(np.float32)
        orig_size = img.shape[:2][::-1]
        img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
        return img, orig_size

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K

