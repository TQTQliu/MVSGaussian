import os
import re
import cv2
import glob
import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

# Filter hyperparameter Settings  
########################
s = 1
dist_base = 1/8
rel_diff_base = 1/10
########################

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

C0 = 0.28209479177387814
def RGB2SH(rgb):
    return (rgb - 0.5) / C0

# save a binary mask
def save_mask(filename, mask):
    # assert mask.dtype == np.bool
    assert mask.dtype == bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)
    
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img

def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    return intrinsics, extrinsics


# project the reference point cloud into the source view, then project back
def reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    ## step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(np.linalg.inv(intrinsics_ref),
                        np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1]))
    # source 3D space
    xyz_src = np.matmul(np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)),
                        np.vstack((xyz_ref, np.ones_like(x_ref))))[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    ## step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)
    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(np.linalg.inv(intrinsics_src),
                        np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1]))
    # reference 3D space
    xyz_reprojected = np.matmul(np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)),
                                np.vstack((xyz_src, np.ones_like(x_ref))))[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    K_xyz_reprojected[2:3][K_xyz_reprojected[2:3]==0] += 0.00001
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src

def inverse_sigmoid(x):
    return np.log(x/((1-x)+1e-10))

def check_geometric_consistency(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src):
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(depth_ref, intrinsics_ref, extrinsics_ref,
                                                                                                 depth_src, intrinsics_src, extrinsics_src)
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref 

    mask = None
    masks = []
    for i in range(s, 11):
        mask = np.logical_and(dist < i * dist_base, relative_depth_diff < i * rel_diff_base)
        masks.append(mask)
    depth_reprojected[~mask] = 0

    return masks, mask, depth_reprojected, x2d_src, y2d_src


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
    
def fusion(dir_ply, scene):
    data_dir = f'{dir_ply}/{scene}'
    img_dir = os.path.join(data_dir, 'images')
    depth_dir = os.path.join(data_dir, 'depth')
    cam_dir = os.path.join(data_dir, 'cam')

    img_ls = glob.glob(os.path.join(img_dir, '*.png'))
    img_name = [os.path.basename(im).split('.')[0] for im in img_ls]

    # for the final point cloud
    vertexs = []
    vertex_colors = []
    
    for name in img_name:
        ref_name = name
        src_names = [im for im in img_name if im!=name]
        
        ref_intrinsics, ref_extrinsics = read_camera_parameters(os.path.join(cam_dir, ref_name+'.txt'))
        ref_img = read_img(os.path.join(img_dir, ref_name+'.png'))
        ref_depth_est = read_pfm(os.path.join(depth_dir, ref_name+'.pfm'))[0]
        
        all_srcview_depth_ests = []
        all_srcview_x = []
        all_srcview_y = []
        all_srcview_geomask = []

        # compute the geometric mask
        geo_mask_sum = 0
        dy_range = len(src_names) + 1
        geo_mask_sums = [0] * (dy_range - s)
        for src_name in src_names:
            # camera parameters of the source view
            src_intrinsics, src_extrinsics = read_camera_parameters(os.path.join(cam_dir, src_name+'.txt'))
            # the estimated depth of the source view
            src_depth_est = read_pfm(os.path.join(depth_dir, src_name+'.pfm'))[0]
            masks, geo_mask, depth_reprojected, x2d_src, y2d_src = check_geometric_consistency(ref_depth_est, ref_intrinsics,
                                                                                            ref_extrinsics, src_depth_est,
                                                                                            src_intrinsics, src_extrinsics)
            geo_mask_sum += geo_mask.astype(np.int32)
            for i in range(s, dy_range):
                geo_mask_sums[i - s] += masks[i - s].astype(np.int32)

            all_srcview_depth_ests.append(depth_reprojected)
            all_srcview_x.append(x2d_src)
            all_srcview_y.append(y2d_src)
            all_srcview_geomask.append(geo_mask)

        depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
        
        geo_mask = geo_mask_sum >= dy_range
        for i in range(s, dy_range):
            geo_mask = np.logical_or(geo_mask, geo_mask_sums[i - s] >= i)

        final_mask = geo_mask

        os.makedirs(os.path.join(data_dir, "mask"), exist_ok=True)
        save_mask(os.path.join(data_dir, f"mask/{ref_name}_final.png"), final_mask)

        height, width = depth_est_averaged.shape[:2]
        x, y = np.meshgrid(np.arange(0, width), np.arange(0, height))
        valid_points = final_mask
        print("valid_points", valid_points.mean())
        x, y, depth = x[valid_points], y[valid_points], depth_est_averaged[valid_points]
        color = ref_img[valid_points]
        xyz_ref = np.matmul(np.linalg.inv(ref_intrinsics),
                            np.vstack((x, y, np.ones_like(x))) * depth)
        xyz_world = np.matmul(np.linalg.inv(ref_extrinsics),
                              np.vstack((xyz_ref, np.ones_like(x))))[:3]
        vertexs.append(xyz_world.transpose((1, 0)))
        vertex_colors.append((color * 255).astype(np.uint8))
        
    vertexs = np.concatenate(vertexs, axis=0)
    vertex_colors = np.concatenate(vertex_colors, axis=0)
    
    scene = os.path.basename(data_dir)
    ply_path = os.path.join(data_dir, f'{scene}.ply')
    storePly(ply_path, vertexs, vertex_colors)