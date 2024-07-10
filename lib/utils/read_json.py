import json
import os
import numpy as np
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--root', type=str, default="output")
    parser.add_argument("--scenes", nargs="+", type=str)
    parser.add_argument('--iter', type=int, default=30000)
    args = parser.parse_args()
    scenes_dict = {}
    for scene in args.scenes:
        path = os.path.join(args.root, scene, 'results.json')
        f = open(path, 'r')
        content = f.read()
        a = json.loads(content)
        scenes_dict[f'{scene}'] = a
        f.close()

    psnr_ls, ssim_ls, lpips_ls = [], [], []
    for scene in args.scenes:
        psnr = scenes_dict[scene][f'ours_{args.iter}']['PSNR']
        ssim = scenes_dict[scene][f'ours_{args.iter}']['SSIM']
        lpips = scenes_dict[scene][f'ours_{args.iter}']['LPIPS']
        psnr_ls.append(psnr)
        ssim_ls.append(ssim)
        lpips_ls.append(lpips)
    psnr_ls = np.array(psnr_ls).mean()
    ssim_ls = np.array(ssim_ls).mean()
    lpips_ls = np.array(lpips_ls).mean()
    print(f'iter:{args.iter}: PSNR:{psnr_ls}, SSIM:{ssim_ls}, LPIPS:{lpips_ls}')
    
        
        