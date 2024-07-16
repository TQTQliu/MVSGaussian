import numpy as np
import os
from lib.config import cfg
import imageio
import cv2
import random
from lib.utils import data_utils
import torch
from lib.datasets import mvsgs_utils
from lib.utils.video_utils import *

class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        self.split = kwargs['split']
        self.input_h_w = kwargs['input_h_w']
        self.scale_factor = cfg.mvsgs.scale_factor
        self.build_metas()
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = [0.0, 0.0, 0.0]
        self.scale = 1.0

    def build_metas(self):
        self.scene_infos = {}
        self.metas = []
        scene = os.path.basename(self.data_root)
        img_dir = os.path.join(self.data_root, 'images')
        img_len = len(os.listdir(img_dir))
        #
        render_ids = [j for j in range(img_len//8, img_len, img_len//4)] 
        train_ids = [j for j in range(img_len) if j not in render_ids]
        #
        pose_bounds = np.load(os.path.join(self.data_root, 'poses_bounds.npy')) # c2w, -u, r, -t
        poses = pose_bounds[:, :15].reshape((-1, 3, 5))
        c2ws = np.eye(4)[None].repeat(len(poses), 0)
        c2ws[:, :3, 0], c2ws[:, :3, 1], c2ws[:, :3, 2], c2ws[:, :3, 3] = poses[:, :3, 1], poses[:, :3, 0], -poses[:, :3, 2], poses[:, :3, 3]
        ixts = np.eye(3)[None].repeat(len(poses), 0)
        ixts[:, 0, 0], ixts[:, 1, 1] = poses[:, 2, 4], poses[:, 2, 4]
        ixts[:, 0, 2], ixts[:, 1, 2] = poses[:, 1, 4]/2., poses[:, 0, 4]/2.

        img_paths = sorted([item for item in os.listdir(os.path.join(self.data_root, 'images')) if '.png' in item])
        depth_ranges = pose_bounds[:, -2:]
        scene_info = {'ixts': ixts.astype(np.float32), 'c2ws': c2ws.astype(np.float32), 'image_names': img_paths, 'depth_ranges': depth_ranges.astype(np.float32)}
        scene_info['scene_name'] = scene
        self.scene_infos[scene] = scene_info

        c2ws = c2ws[train_ids]
        for i in render_ids:
            c2w = scene_info['c2ws'][i]
            distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)
            argsorts = distance.argsort()
            argsorts = argsorts[1:] if i in train_ids else argsorts
            if self.split == 'train':
                src_views = [train_ids[i] for i in argsorts[:cfg.mvsgs.train_input_views[1]+1]]
            else:
                src_views = [train_ids[i] for i in argsorts[:cfg.mvsgs.test_input_views]]
            self.metas += [(scene, i, src_views)]
    
    def get_video_rendering_path(self, ref_poses, mode, near_far, train_c2w_all, n_frames=60, rads_scale=1.25):
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
                cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=rads_scale, N_views=n_frames)
            else:
                raise Exception(f'Unknown video rendering path mode {mode}')

            # convert back to extrinsics tensor
            cur_w2cs = torch.tensor(cur_path).inverse()[:, :3].to(torch.float32)
            poses_paths.append(cur_w2cs)

        poses_paths = torch.stack(poses_paths, dim=0)
        return poses_paths

    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]
        if self.split == 'train':
            if np.random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views, input_views_num)
        scene_info = self.scene_infos[scene]
        tar_img, tar_mask, tar_ext, tar_ixt = self.read_tar(scene_info, tar_view)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)
        ret = {'src_inps': src_inps.transpose(0, 3, 1, 2),
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        if self.split != 'train':
            ret.update({'tar_img': tar_img,
                        'tar_mask': tar_mask})

        H, W = tar_img.shape[:2]
        depth_ranges = np.array(scene_info['depth_ranges'])
        near_far = np.array([depth_ranges[:, 0].min().item()*self.scale_factor, depth_ranges[:, 1].max().item()*self.scale_factor]).astype(np.float32)
        # near_far = scene_info['depth_ranges'][tar_view]
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
            render_path_mode = 'spiral'
            train_c2w_all = np.linalg.inv(src_exts)
            poses_paths = self.get_video_rendering_path(src_exts, render_path_mode, near_far, train_c2w_all, n_frames=60)
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

    def read_src(self, scene, src_views):
        src_ids = src_views
        ixts, exts, imgs = [], [], []
        for idx in src_ids:
            img, orig_size = self.read_image(scene, idx)
            imgs.append(((img/255.)*2-1).astype(np.float32))
            ixt, ext, _ = self.read_cam(scene, idx, orig_size)
            ixts.append(ixt)
            exts.append(ext)
        return np.stack(imgs), np.stack(exts), np.stack(ixts)

    def read_tar(self, scene, view_idx):
        img, orig_size = self.read_image(scene, view_idx)
        img = (img/255.).astype(np.float32)
        ixt, ext, _ = self.read_cam(scene, view_idx, orig_size)
        mask = np.ones_like(img[..., 0]).astype(np.uint8)
        return img, mask, ext, ixt

    def read_cam(self, scene, view_idx, orig_size):
        ext = scene['c2ws'][view_idx].astype(np.float32)
        ixt = scene['ixts'][view_idx].copy()
        ixt[0] *= self.input_h_w[1] / orig_size[0]
        ixt[1] *= self.input_h_w[0] / orig_size[1]
        w2c = np.linalg.inv(ext)
        w2c[:3,3] *= self.scale_factor
        return ixt, w2c, 1

    def read_image(self, scene, view_idx):
        image_path = os.path.join(self.data_root, 'images', scene['image_names'][view_idx])
        img = (np.array(imageio.imread(image_path))).astype(np.float32)
        orig_size = img.shape[:2][::-1]
        img = cv2.resize(img, self.input_h_w[::-1], interpolation=cv2.INTER_AREA)
        return np.array(img), orig_size

    def __len__(self):
        return len(self.metas)

def get_K_from_params(params):
    K = np.zeros((3, 3)).astype(np.float32)
    K[0][0], K[0][2], K[1][2] = params[:3]
    K[1][1] = K[0][0]
    K[2][2] = 1.
    return K

