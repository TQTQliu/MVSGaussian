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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer_ft import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer_ft import GaussianModel
import cv2
import imageio
import numpy as np
import time


def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if name == 'test':
            st = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        if name == 'test':
            print(1./(time.time()-st))
        gt = view.original_image[0:3, :, :]
        if args.crop:
            h,w = gt.shape[1:]
            H_crop, W_crop = int(h*0.1), int(w*0.1)
            rendering = rendering[:,H_crop:-H_crop, W_crop:-W_crop]
            gt = gt[:,H_crop:-H_crop, W_crop:-W_crop]
            
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, init_ply: str, video: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, init_ply=init_ply, video=video)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)
        
        if video:
            folder = f'{args.model_path}/test/'
            saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
            path = os.path.join(folder, f'ours_{max(saved_iters)}/renders')
            scene_name = os.path.basename(args.model_path)
            filelist = os.listdir(path)
            filelist = sorted(filelist)
            imgs = []
            for item in filelist:
                if item.endswith('.png'):
                    item = os.path.join(path, item)
                    img = cv2.imread(item)[:,:,::-1]
                    imgs.append(img)
            save_dir = f'{args.video_path}/{scene_name}'
            os.makedirs(save_dir,exist_ok=True)
            save_path = os.path.join(save_dir,f'{scene_name}.mp4')
            imageio.mimwrite(save_path, np.stack(imgs), fps=10, quality=10)
            print(f'The video is saved in {save_path}.')
            

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--crop', '-c', action="store_true")
    parser.add_argument("--init_ply", '-p', type=str, default='None')
    parser.add_argument("--video", '-v', action="store_true")
    parser.add_argument("--video_path", default='video_save')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, init_ply=args.init_ply, video=args.video)