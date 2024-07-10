import torch
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import cv2
from lib.utils import data_config


def unnormalize_img(img, mean, std):
    """
    img: [3, h, w]
    """
    img = img.detach().cpu().clone()
    # img = img / 255.
    img *= torch.tensor(std).view(3, 1, 1)
    img += torch.tensor(mean).view(3, 1, 1)
    min_v = torch.min(img)
    img = (img - min_v) / (torch.max(img) - min_v)
    return img


def bgr_to_rgb(img):
    return img[:, :, [2, 1, 0]]


def horizon_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((max(h0, h1), w0 + w1, 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[:h1, w0:(w0 + w1), :] = inp1
    else:
        inp = np.zeros((max(h0, h1), w0 + w1), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[:h1, w0:(w0 + w1)] = inp1
    return inp


def vertical_concate(inp0, inp1):
    h0, w0 = inp0.shape[:2]
    h1, w1 = inp1.shape[:2]
    if inp0.ndim == 3:
        inp = np.zeros((h0 + h1, max(w0, w1), 3), dtype=inp0.dtype)
        inp[:h0, :w0, :] = inp0
        inp[h0:(h0 + h1), :w1, :] = inp1
    else:
        inp = np.zeros((h0 + h1, max(w0, w1)), dtype=inp0.dtype)
        inp[:h0, :w0] = inp0
        inp[h0:(h0 + h1), :w1] = inp1
    return inp


def transparent_cmap(cmap):
    """Copy colormap and set alpha values"""
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = 0.3
    return mycmap

cmap = transparent_cmap(plt.get_cmap('jet'))


def set_grid(ax, h, w, interval=8):
    ax.set_xticks(np.arange(0, w, interval))
    ax.set_yticks(np.arange(0, h, interval))
    ax.grid()
    ax.set_yticklabels([])
    ax.set_xticklabels([])


color_list = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.167, 0.000, 0.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        1.000, 1.000, 1.000,
        0.50, 0.5, 0
    ]
).astype(np.float32)
colors = color_list.reshape((-1, 3)) * 255
colors = np.array(colors, dtype=np.uint8).reshape(len(colors), 1, 1, 3)

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.min(x[x>0]) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax

    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]
