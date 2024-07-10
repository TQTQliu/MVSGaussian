import torch
import torch.nn as nn
from lib.config import cfg
from lib.train.losses.vgg_perceptual_loss import VGGPerceptualLoss
from lib.train.losses.ssim_loss import SSIM


class NetworkWrapper(nn.Module):
    def __init__(self, net, train_loader):
        super(NetworkWrapper, self).__init__()
        self.device = torch.device('cuda:{}'.format(cfg.local_rank))
        self.net = net
        self.color_crit = nn.MSELoss(reduction='mean')
        self.mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
        self.perceptual_loss = VGGPerceptualLoss().to(self.device)

    def forward(self, batch):
        output = self.net(batch)

        scalar_stats = {}
        loss = 0
        for i in range(cfg.mvsgs.cas_config.num):
            if not cfg.mvsgs.cas_config.render_if[i]:
                continue
            color_loss = self.color_crit(batch[f'rgb_{i}'], output[f'rgb_level{i}'])
            scalar_stats.update({f'color_mse_{i}': color_loss})
            loss += cfg.mvsgs.cas_config.loss_weight[i] * color_loss
           
            psnr = -10. * torch.log(color_loss) / torch.log(torch.Tensor([10.]).to(color_loss.device))
            scalar_stats.update({f'psnr_{i}': psnr})
          
            num_patchs = cfg.mvsgs.cas_config.num_patchs[i]
            if cfg.mvsgs.cas_config.train_img[i]:
                render_scale = cfg.mvsgs.cas_config.render_scale[i]
                B, S, C, H, W = batch['src_inps'].shape
                H, W = int(H * render_scale), int(W * render_scale)
                inp = output[f'rgb_level{i}'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
                tar = batch[f'rgb_{i}'].reshape(B, H, W, 3).permute(0, 3, 1, 2)
                perceptual_loss = self.perceptual_loss(inp, tar)
                loss += 0.05 * perceptual_loss * cfg.mvsgs.cas_config.loss_weight[i]
                scalar_stats.update({f'perceptual_loss_{i}': perceptual_loss.detach()})
                ssim = SSIM(window_size = 7)
                ssim_loss = 1-ssim(inp, tar)
                loss += 0.1 * ssim_loss * cfg.mvsgs.cas_config.loss_weight[i]
                scalar_stats.update({f'ssim_loss_{i}': ssim_loss.detach()})
                
            elif num_patchs > 0:
                patch_size = cfg.mvsgs.cas_config.patch_size[i]
                num_rays = cfg.mvsgs.cas_config.num_rays[i]
                patch_rays = int(patch_size ** 2)
                inp = torch.empty((0, 3, patch_size, patch_size)).to(self.device)
                tar = torch.empty((0, 3, patch_size, patch_size)).to(self.device)
                for j in range(num_patchs):
                    inp = torch.cat([inp, output[f'rgb_level{i}'][:, num_rays+j*patch_rays:num_rays+(j+1)*patch_rays, :].reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)])
                    tar = torch.cat([tar, batch[f'rgb_{i}'][:, num_rays+j*patch_rays:num_rays+(j+1)*patch_rays, :].reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)])
                perceptual_loss = self.perceptual_loss(inp, tar)

                loss += 0.01 * perceptual_loss * cfg.mvsgs.cas_config.loss_weight[i]
                scalar_stats.update({f'perceptual_loss_{i}': perceptual_loss.detach()})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

