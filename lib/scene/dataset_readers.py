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

import os
import sys
import glob
import cv2
import torch
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from utils.video_utils import *
from utils.camera_utils import *
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras_LLFF(cam_extrinsics, cam_intrinsics, path, rgb_mapping, size=(640, 960)):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        h_o, w_o = intr.height, intr.width 
        height = size[0]
        width = size[1]
        
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_x = focal_length_x * width / w_o
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            focal_length_x = focal_length_x * width / w_o
            focal_length_y = focal_length_y * height / h_o
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0] * width / w_o
            focal_length_y = intr.params[0] * height / h_o
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        image_path = rgb_mapping[idx]
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = (np.array(image)).astype(np.float32)
        image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image.astype(np.uint8))
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo_LLFF(path, init_ply=None, video=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = 'images_4'
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_infos_unsorted = readColmapCameras_LLFF(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                            path=path, rgb_mapping=rgb_mapping)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    
    pairs = torch.load('data/mvsgs/pairs.th')
    scene = path.split('/')[-1]
    train_ids = pairs[f'{scene}_train']
    render_ids = pairs[f'{scene}_val']
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_ids]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in render_ids]
    if video:
        cam_infos_render_vd = []
        cur_c2ws_all = []
        for train_cam in train_cam_infos:
            R, T = train_cam.R, train_cam.T
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3] = T
            c2w = np.linalg.inv(w2c)
            cur_c2ws_all.append(c2w)
        cur_c2ws_all = np.stack(cur_c2ws_all)
        pose_bounds = np.load(os.path.join(path, 'poses_bounds.npy')) # c2w, -u, r, -t
        depth_ranges = pose_bounds[:, -2:]
        near_far = [depth_ranges.min(),depth_ranges.max()]
        cur_near_far = near_far
        cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=1.5, N_views=60)
        # convert back to extrinsics tensor
        cur_w2cs = np.linalg.inv(cur_path)[:, :3].astype(np.float32)
        for cur_w2c in cur_w2cs:
            R = cur_w2c[:,:3]
            T = cur_w2c[:,3]
            cam_info = CameraInfo(uid=0, R=R, T=T, FovY=cam_infos[0].FovY, FovX=cam_infos[0].FovX, image=cam_infos[0].image,
                                image_path=cam_infos[0].image_path, image_name=cam_infos[0].image_name, width=cam_infos[0].width, height=cam_infos[0].height)
            cam_infos_render_vd.append(cam_info)
        test_cam_infos = cam_infos_render_vd
        
    nerf_normalization = getNerfppNorm(train_cam_infos)
    if init_ply!='None':
        ply_path = f'{init_ply}/{scene}/{scene}.ply'
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    print(ply_path)
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png", init_ply=None, video=False, center_idx=0):
    cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    pairs = torch.load('data/mvsgs/pairs.th')
    scene = path.split('/')[-1]
    train_ids, render_ids = pairs[f'{scene}_train'], pairs[f'{scene}_val']
    train_cam_infos = [cam_infos[idx] for idx in train_ids]
    test_cam_infos = [cam_infos[idx] for idx in render_ids]
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)
    
    cur_c2ws_all = []
    for train_cam in train_cam_infos:
        R, T = train_cam.R, train_cam.T
        w2c = np.eye(4)
        w2c[:3,:3] = R
        w2c[:3,3] = T
        c2w = np.linalg.inv(w2c)
        cur_c2ws_all.append(c2w)
    cur_c2ws_all = np.stack(cur_c2ws_all)
    if video:
        cam_infos_render_vd = []
        json_info = json.load(open(os.path.join(path,'transforms_train.json')))
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
            img_path = os.path.join(path, 'train/r_{}.png'.format(idx))
            scene_info['img_paths'].append(img_path)
        train_ids, render_ids = pairs[f'{scene}_train'], pairs[f'{scene}_val']
        c2ws = np.stack([np.linalg.inv(scene_info['exts'][idx]) for idx in train_ids])
        test_view = render_ids[center_idx]
        c2w = np.linalg.inv(scene_info['exts'][test_view])
        distance = np.linalg.norm((c2w[:3, 3][None] - c2ws[:, :3, 3]), axis=-1)
        argsorts = distance.argsort()
        input_views_num = 3
        src_c2ws= [cur_c2ws_all[i] for i in argsorts[:input_views_num]]
        src_c2ws = np.stack(src_c2ws)[:, :3, :]
        cur_path = get_interpolate_render_path(src_c2ws, N_views=60)
        cur_w2cs = np.linalg.inv(cur_path)[:, :3].astype(np.float32)
        for cur_w2c in cur_w2cs:
            R = cur_w2c[:3,:3]
            T = cur_w2c[:3,3]
            cam_info = CameraInfo(uid=0, R=R, T=T, FovY=cam_infos[0].FovY, FovX=cam_infos[0].FovX, image=cam_infos[0].image,
                                image_path=cam_infos[0].image_path, image_name=cam_infos[0].image_name, width=cam_infos[0].width, height=cam_infos[0].height)
            cam_infos_render_vd.append(cam_info)
        test_cam_infos = cam_infos_render_vd
    if init_ply!='None':
        ply_path = f'{init_ply}/{scene}/{scene}.ply'
    else:
        ply_path = os.path.join(path, "points3d.ply")
    print(ply_path)
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3 
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapSceneInfo_TNT(path, init_ply=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = 'images'
    rgb_mapping = [f for f in sorted(glob.glob(os.path.join(path, reading_dir, '*')))
                   if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    cam_infos_unsorted = readColmapCameras_TNT(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                            path=path, rgb_mapping=rgb_mapping, init_ply=init_ply)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        
    pairs = torch.load('data/mvsgs/pairs.th')
    scene = path.split('/')[-1]
    train_ids = pairs[f'{scene}_train']
    render_ids = pairs[f'{scene}_test']
    
    train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in train_ids]
    test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in render_ids]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    if init_ply!='None':
        ply_path = f'{init_ply}/{scene}/{scene}.ply'
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    print(ply_path)
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readColmapCameras_TNT(cam_extrinsics, cam_intrinsics, path, rgb_mapping, size=(640, 960), init_ply=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        h_o, w_o = intr.height, intr.width 
        height = size[0]
        width = size[1]
        uid = intr.id
        
        if init_ply!='None':
            data_root = os.path.dirname(path)
            scene = path.split('/')[-1]
            proj_mat_filename = os.path.join(data_root, scene, f'cams_1/{uid-1:08d}_cam.txt')
            intrinsics, extrinsics, _, _ = read_cam_file(proj_mat_filename)
            intrinsics[0] = intrinsics[0] * width / w_o
            intrinsics[1] = intrinsics[1] * height / h_o
            focal_length_x = intrinsics[0,0]
            focal_length_y = intrinsics[1,1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            R = np.array(extrinsics[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(extrinsics[:3, 3], np.float32)
        else:
            R = np.transpose(qvec2rotmat(extr.qvec))
            T = np.array(extr.tvec)
            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_x = focal_length_x * width / w_o
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                focal_length_x = focal_length_x * width / w_o
                focal_length_y = focal_length_y * height / h_o
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="SIMPLE_RADIAL":
                focal_length_x = intr.params[0] * width / w_o
                focal_length_y = intr.params[0] * height / h_o
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
                
        image_path = rgb_mapping[idx]
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = (np.array(image)).astype(np.float32)
        image = cv2.resize(image, size[::-1], interpolation=cv2.INTER_AREA)
        image = Image.fromarray(image.astype(np.uint8))
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    
    return cam_infos



def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, size=(640, 960)):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        h_o, w_o = intr.height, intr.width 
        height = size[0]
        width = size[1]
        
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_x = focal_length_x * width / w_o
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            focal_length_x = focal_length_x * width / w_o
            focal_length_y = focal_length_y * height / h_o
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="SIMPLE_RADIAL":
            focal_length_x = intr.params[0] * width / w_o
            focal_length_y = intr.params[0] * height / h_o
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapSceneInfo(path, images, eval, llffhold=8, init_ply=None, video=False):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
  
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        # train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        # test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
        img_len = len(cam_infos)
        render_ids = [j for j in range(img_len//8, img_len, img_len//4)] 
        train_ids = [j for j in range(img_len) if j not in render_ids]
        train_cam_infos = [cam_infos[idx] for idx in train_ids]
        test_cam_infos = [cam_infos[idx] for idx in render_ids]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    
    if video:
        cam_infos_render_vd = []
        cur_c2ws_all = []
        for train_cam in train_cam_infos:
            R, T = train_cam.R, train_cam.T
            w2c = np.eye(4)
            w2c[:3,:3] = R
            w2c[:3,3] = T
            c2w = np.linalg.inv(w2c)
            cur_c2ws_all.append(c2w)
        cur_c2ws_all = np.stack(cur_c2ws_all)
        pose_bounds = np.load(os.path.join(path, 'poses_bounds.npy')) # c2w, -u, r, -t
        depth_ranges = pose_bounds[:, -2:]
        near_far = [depth_ranges.min(),depth_ranges.max()]
        cur_near_far = near_far
        cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=1.5, N_views=60)
        # convert back to extrinsics tensor
        cur_w2cs = np.linalg.inv(cur_path)[:, :3].astype(np.float32)
        for cur_w2c in cur_w2cs:
            R = cur_w2c[:,:3]
            T = cur_w2c[:,3]
            cam_info = CameraInfo(uid=0, R=R, T=T, FovY=cam_infos[0].FovY, FovX=cam_infos[0].FovX, image=cam_infos[0].image,
                                image_path=cam_infos[0].image_path, image_name=cam_infos[0].image_name, width=cam_infos[0].width, height=cam_infos[0].height)
            cam_infos_render_vd.append(cam_info)
        test_cam_infos = cam_infos_render_vd

    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene = path.split('/')[-1]
    if init_ply!='None':
        ply_path = f'{init_ply}/{scene}/{scene}.ply'
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
    print(ply_path)
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Blender" : readNerfSyntheticInfo,
    "LLFF": readColmapSceneInfo_LLFF,
    "TNT": readColmapSceneInfo_TNT,
    "Colmap": readColmapSceneInfo
}