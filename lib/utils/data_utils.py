import numpy as np
import cv2
import random
from torch import nn
import torch
from imgaug import augmenters as iaa
from lib.config import cfg
from plyfile import PlyData
from lib.utils import data_config
import re
import os
import math


def add_batch(batch):
    if isinstance(batch, tuple) or isinstance(batch, list):
        batch = [add_batch(b) for b in batch]
    elif isinstance(batch, dict):
        batch_ = {}
        for key in batch:
            batch_[key] = add_batch(batch[key])
        batch = batch_
    elif isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        batch = batch[None]
    else:
        batch = torch.tensor(batch)[None]
    return batch

def get_bound_2d_mask(bounds, K, H, W):
    points = bounds[:, :3] @ K.T
    corners_2d = points[..., :2] / points[..., 2:]
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    return intrinsics, extrinsics, depth_min

def read_pmn_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[1])
    return intrinsics, extrinsics, depth_min, depth_max

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

def to_tensor(rgb):
    rgb = rgb.astype(np.float32) / 255.
    rgb = (rgb-data_config.mean_rgb) / data_config.std_rgb
    return rgb.transpose(2, 0, 1)

def to_img(tensor):
    tensor = tensor.detach().cpu().clone().numpy()
    tensor = tensor.transpose(1, 2, 0)
    tensor = tensor * data_config.std_rgb + data_config.mean_rgb
    tensor *= 255.
    return tensor.astype(np.uint8)


def resize_images(imgs, masks, ixt, input_size):
    ori_h, ori_w = imgs[0].shape[0], imgs[0].shape[1]
    tar_h, tar_w = input_size
    imgs = [cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR) for img in imgs]
    masks = [cv2.resize(mask.astype(np.uint8), input_size, interpolation=cv2.INTER_NEAREST) for mask in masks]
    ixt[0][0] *= (tar_h/ori_h)
    ixt[0][2] *= (tar_h/ori_h)
    ixt[1][1] *= (tar_w/ori_w)
    ixt[1][2] *= (tar_w/ori_w)
    return imgs, masks, ixt

def resize_image(img, mask, ixt, input_size):
    ori_h, ori_w = img.shape[0], img.shape[1]
    tar_h, tar_w = input_size
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask.astype(np.uint8), input_size, interpolation=cv2.INTER_NEAREST)
    ixt[0][0] *= (tar_h/ori_h)
    ixt[0][2] *= (tar_h/ori_h)
    ixt[1][1] *= (tar_w/ori_w)
    ixt[1][2] *= (tar_w/ori_w)
    return img, mask, ixt

def load_matrix(path):
    lines = [[float(w) for w in line.strip().split()] for line in open(path)]
    if len(lines[0]) == 2:
        lines = lines[1:]
    if len(lines[-1]) == 2:
        lines = lines[:-1]
    return np.array(lines).astype(np.float32)

def load_nsvf_intrinsics(filepath, resized_width=None, invert_y=False):
    try:
        intrinsics = load_matrix(filepath)
        if intrinsics.shape[0] == 3 and intrinsics.shape[1] == 3:
            _intrinsics = np.zeros((4, 4), np.float32)
            _intrinsics[:3, :3] = intrinsics
            _intrinsics[3, 3] = 1
            intrinsics = _intrinsics
        if intrinsics.shape[0] == 1 and intrinsics.shape[1] == 16:
            intrinsics = intrinsics.reshape(4, 4)
        return intrinsics
    except ValueError:
        pass

    # Get camera intrinsics
    with open(filepath, 'r') as file:

        f, cx, cy, _ = map(float, file.readline().split())
    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])
    return full_intrinsic

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if b3 ** 2 - 4 * a3 * c3 < 0:
        r3 = min(r1, r2)
    else:
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_distribution(heatmap, center, sigma_x, sigma_y, rho, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), (sigma_x/3, sigma_y/3), rho)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def draw_heatmap_np(hm, point, box_size):
    """point: [x, y]"""
    # radius = gaussian_radius(box_size)
    radius = box_size[0]
    radius = max(0, int(radius))
    ct_int = np.array(point, dtype=np.int32)
    draw_umich_gaussian(hm, ct_int, radius)
    return hm


def get_edge(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return mask - cv2.erode(mask, kernel)


def compute_gaussian_1d(dmap, sigma=1):
    """dmap: each entry means a distance"""
    prob = np.exp(-dmap / (2 * sigma * sigma))
    prob[prob < np.finfo(prob.dtype).eps * prob.max()] = 0
    return prob


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt


def homography_transform(pt, H):
    """pt: [n, 2]"""
    pt = np.concatenate([pt, np.ones([len(pt), 1])], axis=1)
    pt = np.dot(pt, H.T)
    pt = pt[..., :2] / pt[..., 2:]
    return pt


def get_border(border, size):
    i = 1
    while np.any(size - border // i <= border // i):
        i *= 2
    return border // i


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)


def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2


def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])


def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha


def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)


def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)

    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)


def blur_aug(inp):
    if np.random.random() < 0.1:
        if np.random.random() < 0.8:
            inp = iaa.blur_gaussian_(inp, abs(np.clip(np.random.normal(0, 1.5), -3, 3)))
        else:
            inp = iaa.MotionBlur((3, 15), (-45, 45))(images=[inp])[0]


def gaussian_blur(image, sigma):
    from scipy import ndimage
    if image.ndim == 2:
        image[:, :] = ndimage.gaussian_filter(image[:, :], sigma, mode="mirror")
    else:
        nb_channels = image.shape[2]
        for channel in range(nb_channels):
            image[:, :, channel] = ndimage.gaussian_filter(image[:, :, channel], sigma, mode="mirror")


def inter_from_mask(pred, gt):
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)
    intersection = np.logical_and(gt, pred).sum()
    return intersection


def draw_poly(mask, poly):
    cv2.fillPoly(mask, [poly], 255)
    return mask


def inter_from_poly(poly, gt, width, height):
    mask_small = np.zeros((1, height, width), dtype=np.uint8)
    mask_small = draw_poly(mask_small, poly)
    mask_gt = gt[..., 0]

    return inter_from_mask(mask_small, mask_gt)


def inter_from_polys(poly, w, h, gt_mask):
    inter = inter_from_poly(poly, gt_mask, w, h)
    if inter > 0:
        return False
    return True


def select_point(shape, poly, gt_mask):
    for i in range(cfg.max_iter):
        y = np.random.randint(shape[0] - poly['bbox'][3])
        x = np.random.randint(shape[1] - poly['bbox'][2])
        delta = np.array([poly['bbox'][0] - x, poly['bbox'][1] - y])
        poly_move = np.array(poly['poly']) - delta
        inter = inter_from_polys(poly_move, shape[1], shape[0], gt_mask)
        if inter:
            return x, y
    x, y = -1, -1
    return x, y


def transform_small_gt(poly, box, x, y):
    delta = np.array([poly['bbox'][0] - x, poly['bbox'][1] - y])
    poly['poly'] -= delta
    box[:2] -= delta
    box[2:] -= delta
    return poly, box


def get_mask_img(img, poly):
    mask = np.zeros(img.shape[:2])[..., np.newaxis]
    cv2.fillPoly(mask, [np.round(poly['poly']).astype(int)], 1)
    poly_img = img * mask
    mask = mask[..., 0]
    return poly_img, mask


def add_small_obj(img, gt_mask, poly, box, polys_gt):
    poly_img, mask = get_mask_img(img, poly)
    x, y = select_point(img.shape, poly.copy(), gt_mask)
    if x == -1:
        box = []
        return img, poly, box
    poly, box = transform_small_gt(poly, box, x, y)
    _, mask_ori = get_mask_img(img, poly)
    gt_mask += mask_ori[..., np.newaxis]
    img[mask_ori == 1] = poly_img[mask == 1]
    return img, poly, box[np.newaxis, :], gt_mask


def get_gt_mask(img, poly):
    mask = np.zeros(img.shape[:2])[..., np.newaxis]
    for i in range(len(poly)):
        for j in range(len(poly[i])):
            cv2.fillPoly(mask, [np.round(poly[i][j]['poly']).astype(int)], 1)
    return mask


def small_aug(img, poly, box, label, num):
    N = len(poly)
    gt_mask = get_gt_mask(img, poly)
    for i in range(N):
        if len(poly[i]) > 1:
            continue
        if poly[i][0]['area'] < 32*32:
            for k in range(num):
                img, poly_s, box_s, gt_mask = add_small_obj(img, gt_mask, poly[i][0].copy(), box[i].copy(), poly)
                if len(box_s) == 0:
                    continue
                poly.append([poly_s])
                box = np.concatenate((box, box_s))
                label.append(label[i])
    return img, poly, box, label


def truncated_normal(mean, sigma, low, high, data_rng=None):
    if data_rng is None:
        data_rng = np.random.RandomState()
    value = data_rng.normal(mean, sigma)
    return np.clip(value, low, high)


def _nms(heat, kernel=3):
    """heat: [b, c, h, w]"""
    pad = (kernel - 1) // 2

    # find the local minimum of heat within the neighborhood kernel x kernel
    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feat(
        topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def clip_to_image(bbox, h, w):
    bbox[..., :2] = torch.clamp(bbox[..., :2], min=0)
    bbox[..., 2] = torch.clamp(bbox[..., 2], max=w-1)
    bbox[..., 3] = torch.clamp(bbox[..., 3], max=h-1)
    return bbox


def load_ply(path):
    ply = PlyData.read(path)
    data = ply.elements[0].data
    x, y, z = data['x'], data['y'], data['z']
    model = np.stack([x, y, z], axis=-1)
    return model

def to_cuda(batch, device=torch.device('cuda:0')):
    if isinstance(batch, tuple) or isinstance(batch, list):
        #batch[k] = [b.cuda() for b in batch[k]]
        #batch[k] = [b.to(self.device) for b in batch[k]]
        batch = [to_cuda(b, device) for b in batch]
    elif isinstance(batch, dict):
        #batch[k] = {key: self.to_cuda(batch[k][key]) for key in batch[k]}
        batch_ = {}
        for key in batch:
            if key == 'meta':
                batch_[key] = batch[key]
            else:
                batch_[key] = to_cuda(batch[key], device)
        batch = batch_
    else:
        # batch[k] = batch[k].cuda()
        batch = batch.to(device)
    return batch

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose

def get_near_far(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o[:1]) / viewdir
    tmax = (bounds[1:2] - ray_o[:1]) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box

def read_camera(intri_name, extri_name, cam_names=[]):
    assert os.path.exists(intri_name), intri_name
    assert os.path.exists(extri_name), extri_name

    intri = FileStorage(intri_name)
    extri = FileStorage(extri_name)
    cams, P = {}, {}
    cam_names = intri.read('names', dt='list')
    for cam in cam_names:
        # 内参只读子码流的
        cams[cam] = {}
        cams[cam]['K'] = intri.read('K_{}'.format( cam))
        cams[cam]['invK'] = np.linalg.inv(cams[cam]['K'])
        Rvec = extri.read('R_{}'.format(cam))
        Tvec = extri.read('T_{}'.format(cam))
        R = cv2.Rodrigues(Rvec)[0]
        RT = np.hstack((R, Tvec))

        cams[cam]['RT'] = RT
        cams[cam]['R'] = R
        cams[cam]['T'] = Tvec
        P[cam] = cams[cam]['K'] @ cams[cam]['RT']
        cams[cam]['P'] = P[cam]

        cams[cam]['dist'] = intri.read('dist_{}'.format(cam))
    cams['basenames'] = cam_names
    return cams

class FileStorage(object):
    def __init__(self, filename, isWrite=False):
        version = cv2.__version__
        self.major_version = int(version.split('.')[0])
        self.second_version = int(version.split('.')[1])

        if isWrite:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        else:
            self.fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    def __del__(self):
        cv2.FileStorage.release(self.fs)

    def write(self, key, value, dt='mat'):
        if dt == 'mat':
            cv2.FileStorage.write(self.fs, key, value)
        elif dt == 'list':
            if self.major_version == 4: # 4.4
                self.fs.startWriteStruct(key, cv2.FileNode_SEQ)
                for elem in value:
                    self.fs.write('', elem)
                self.fs.endWriteStruct()
            else: # 3.4
                self.fs.write(key, '[')
                for elem in value:
                    self.fs.write('none', elem)
                self.fs.write('none', ']')

    def read(self, key, dt='mat'):
        if dt == 'mat':
            output = self.fs.getNode(key).mat()
        elif dt == 'list':
            results = []
            n = self.fs.getNode(key)
            for i in range(n.size()):
                val = n.at(i).string()
                if val == '':
                    val = str(int(n.at(i).real()))
                if val != 'none':
                    results.append(val)
            output = results
        else:
            raise NotImplementedError
        return output

    def close(self):
        self.__del__(self)


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, K, h, w):
    near_fx = znear / K[0, 0]
    near_fy = znear / K[1, 1]
    left = - (w - K[0, 2]) * near_fx
    right = K[0, 2] * near_fx
    bottom = (K[1, 2] - h) * near_fy
    top = K[1, 2] * near_fy

    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))