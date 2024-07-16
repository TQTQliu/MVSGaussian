import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from .utils import *
from .feature_net import Unet


class GS(nn.Module):
    def __init__(self, hid_n=64, feat_ch=16+3):
        """
        """
        super(GS, self).__init__()
        self.hid_n = hid_n
        self.agg = Agg(feat_ch)
        self.head_dim = 24
        self.Unet = Unet(self.head_dim, 16)
        self.opacity_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.Sigmoid()
        )
        # self.rotation_head = nn.Sequential(
        #     nn.Linear(self.head_dim, self.head_dim),
        #     nn.ReLU(),
        #     nn.Linear(self.head_dim, 4),
        # )
        self.scale_head = nn.Sequential(
            nn.Linear(self.head_dim, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Softplus()
        )
        self.color = nn.Sequential(
            nn.Linear(feat_ch+self.head_dim+4, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 1),
            nn.ReLU())
        self.sigma = nn.Sequential(nn.Linear(self.head_dim, 1), nn.Softplus())
        self.color_gs = nn.Sequential(
            nn.Linear(self.head_dim+3, self.head_dim),
            nn.ReLU(),
            nn.Linear(self.head_dim, 3),
            nn.Sigmoid()
        )
        
    def forward(self, vox_feat, img_feat_rgb_dir, z_vals, batch, size, level):
        
        H,W = size
        B, N_points, N_views = img_feat_rgb_dir.shape[:-1]
        S = img_feat_rgb_dir.shape[2]
        img_feat = self.agg(img_feat_rgb_dir)
        x = torch.cat((vox_feat, img_feat), dim=-1)
        
        # depth 
        d = z_vals.shape[-1]
        z_vals = z_vals.reshape(B,H,W,d)
        if cfg.mvsgs.cas_config.depth_inv[level]:
            z_vals = 1./torch.clamp_min(z_vals, 1e-6) # to disp
        depth = z_vals.permute(0,3,1,2)
        
        # sigma head
        sigma = self.sigma(x)
        
        # radiance head
        x0 = x.unsqueeze(2).repeat(1,1,S,1)
        x0 = torch.cat((x0, img_feat_rgb_dir), dim=-1)
        color_weight = F.softmax(self.color(x0), dim=-2)
        radiance = torch.sum((img_feat_rgb_dir[..., -7:-4] * color_weight), dim=-2)
        
        # volume rendering branch
        sigma = sigma.reshape(B,H*W,d)
        raw2alpha = lambda raw: 1.-torch.exp(-raw)
        alpha = raw2alpha(sigma)  # [N_rays, N_samples]
        T = torch.cumprod(1.-alpha+1e-10, dim=-1)[..., :-1]
        T = torch.cat([torch.ones_like(alpha[..., 0:1]), T], dim=-1)
        weights = alpha * T
        radiance = radiance.reshape(B,H*W,d,3)
        rgb_vr = torch.sum(weights[...,None] * radiance, -2) 
        rgb_vr = rgb_vr.reshape(B,H,W,3).permute(0,3,1,2)
        
        # enhance features using a UNet
        x = x.reshape(B,H*W,d,x.shape[-1])
        x = torch.sum(weights[...,None].unsqueeze(0) * x, -2) 
        x = x.reshape(B,H,W,x.shape[-1]).permute(0,3,1,2)
        x = self.Unet(x)
        x = x.flatten(-2).permute(0,2,1)
        
        # gs branch
        # rot head
        rot_out = torch.ones((B,x.shape[1],4)).to(x.device)
        # rot_out = self.rotation_head(x)
        rot_out = torch.nn.functional.normalize(rot_out, dim=-1)
      
        # scale head
        scale_out = self.scale_head(x)
        
        # opacity head
        opacity_out = self.opacity_head(x)
        
        # color head
        x0 = torch.cat((x,rgb_vr.flatten(2).permute(0,2,1)),dim=-1)
        color_out = self.color_gs(x0)
        
        # world_xyz
        weights = weights.reshape(B,H,W,d).permute(0,3,1,2)
        depth = torch.sum(weights * depth, 1) # B H W
        ext = batch['tar_ext']
        ixt = batch['tar_ixt'].clone()
        ixt[:,:2] *= cfg.mvsgs.cas_config.render_scale[level]
        world_xyz = depth2point(depth, ext, ixt)

        return world_xyz, rot_out, scale_out, opacity_out, color_out, rgb_vr



class Agg(nn.Module):
    def __init__(self, feat_ch):
        """
        """
        super(Agg, self).__init__()
        self.feat_ch = feat_ch
        if cfg.mvsgs.viewdir_agg:
            self.view_fc = nn.Sequential(
                    nn.Linear(4, feat_ch),
                    nn.ReLU(),
                    )
            self.view_fc.apply(weights_init)
        self.global_fc = nn.Sequential(
                nn.Linear(feat_ch*3, 32),
                nn.ReLU(),
                )

        self.agg_w_fc = nn.Sequential(
                nn.Linear(32, 1),
                nn.ReLU(),
                )
        self.fc = nn.Sequential(
                nn.Linear(32, 16),
                nn.ReLU(),
                )
        self.global_fc.apply(weights_init)
        self.agg_w_fc.apply(weights_init)
        self.fc.apply(weights_init)

    def forward(self, img_feat_rgb_dir):
        B, S = len(img_feat_rgb_dir), img_feat_rgb_dir.shape[-2]
        if cfg.mvsgs.viewdir_agg:
            view_feat = self.view_fc(img_feat_rgb_dir[..., -4:])
            img_feat_rgb =  img_feat_rgb_dir[..., :-4] + view_feat
        else:
            img_feat_rgb =  img_feat_rgb_dir[..., :-4]

        var_feat = torch.var(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)
        avg_feat = torch.mean(img_feat_rgb, dim=-2).view(B, -1, 1, self.feat_ch).repeat(1, 1, S, 1)

        feat = torch.cat([img_feat_rgb, var_feat, avg_feat], dim=-1)
        global_feat = self.global_fc(feat)
        agg_w = F.softmax(self.agg_w_fc(global_feat), dim=-2)
        im_feat = (global_feat * agg_w).sum(dim=-2)
        return self.fc(im_feat)



def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)

