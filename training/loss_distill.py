# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d

from torch.nn import functional as F
import lpips

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2DistillLoss(Loss):
    def __init__(self, device, G, D, G_T, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, cagan=0, blur_fade_kimg=0, rank=0):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.G_T                = G_T
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.cagan              = cagan
        self.percept_loss = lpips.PerceptualLoss(model='net-lin', net='vgg', \
            use_gpu=True, gpu_ids=[rank])

    def run_G(self, z, c, dir_vectors=None, update_emas=False, teacher=False):
        if teacher is True:
            if dir_vectors is not None:
                batch = z.shape[0]
                num_direction = dir_vectors.size(0)
                index = np.random.choice(np.arange(num_direction), size=(batch,)).astype(np.int64)
                offsets = dir_vectors[index].to(z.device)
                norm = torch.norm(offsets, dim=1, keepdim=True)
                offsets = offsets / norm
                weight = torch.randn(batch, 1, device=z.device) * 5.0
                offsets = offsets * weight
                offsets = offsets[:,None,:]

            ws = self.G.mapping(z, c, update_emas=update_emas)
            ws_t = self.G_T.mapping(z, c, update_emas=update_emas)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    mix_rand = torch.rand([], device=ws.device)
                    # student stage
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(mix_rand < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    new_rand = torch.randn_like(z)
                    ws[:, cutoff:] = self.G.mapping(new_rand, c, update_emas=False)[:, cutoff:]

                    # teacher stage
                    ws_t[:, cutoff:] = self.G_T.mapping(new_rand, c, update_emas=False)[:, cutoff:]
            ws = torch.cat([ws, ws+offsets], dim=0)
            ws_t = torch.cat([ws_t, ws_t+offsets], dim=0)
            img, feat_s = self.G.synthesis(ws, update_emas=update_emas, return_feat=True)
            img_t, feat_t = self.G_T.synthesis(ws_t, update_emas=update_emas, return_feat=True)
            img = [img, img_t]
            ws = [ws, ws_t]
            feat = [feat_s, feat_t]
        else:
            ws = self.G.mapping(z, c, update_emas=update_emas)
            if self.style_mixing_prob > 0:
                with torch.autograd.profiler.record_function('style_mixing'):
                    cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                    cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                    ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
            img = self.G.synthesis(ws, update_emas=update_emas)
        if dir_vectors is not None:
            return img, ws, feat
        else:
            return img, ws

    def run_D(self, img, c, blur_sigma=0, update_emas=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img.device).div(blur_sigma).square().neg().exp2()
                img = upfirdn2d.filter2d(img, f / f.sum())
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def Downsample_Image(self, im_tensor, size):
        im_tensor = F.interpolate(im_tensor, size=(size, size), mode='bilinear', align_corners=False)
        return im_tensor

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, dir_vectors, cur_nimg):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.pl_weight == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                gen_img, _gen_ws, feats = self.run_G(gen_z, gen_c, dir_vectors=dir_vectors, teacher=True)
                gen_img_st, gen_img_te = gen_img[0], gen_img[1]
                batch = gen_img_st.shape[0] // 2
                feat_s, feat_t = feats[0], feats[1]
                gen_logits = self.run_D(gen_img_st, gen_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits) # -log(sigmoid(gen_logits))
                training_stats.report('Loss/G/loss', loss_Gmain)

                # distillation
                lambda_l1 = 3.
                lambda_pe = 3.
                    
                kd_l1_loss = torch.mean(torch.abs(gen_img_st - gen_img_te))

                # perceptual loss
                img_size = gen_img_st.shape[-1]
                lpips_image_size = 256
                if img_size > lpips_image_size: # pooled the image for LPIPS for memory saving
                    gen_img_st = self.Downsample_Image(gen_img_st, lpips_image_size)
                    gen_img_te = self.Downsample_Image(gen_img_te, lpips_image_size)
                    kd_lpips_loss = torch.mean(self.percept_loss(gen_img_st, gen_img_te))
                else:
                    kd_lpips_loss = torch.mean(self.percept_loss(gen_img_st, gen_img_te))
                
                kd_simi_loss = torch.zeros_like(kd_l1_loss)
                # ld loss
                lambda_simi_loss = 30.
                # layer_names = ['L1_36_512', 'L2_36_512', 'L3_52_512', 'L4_52_512']
                layer_names = ['L1_36_512', 'L2_36_512']
                for layer_name in layer_names:
                    f1 = torch.flatten(feat_s[layer_name], start_dim=1)[:batch]
                    f2 = torch.flatten(feat_s[layer_name], start_dim=1)[batch:]
                    s_simi = F.cosine_similarity(f1[:,None,:], f2[None,:,:], dim=2)

                    f1 = torch.flatten(feat_t[layer_name], start_dim=1)[:batch]
                    f2 = torch.flatten(feat_t[layer_name], start_dim=1)[batch:]
                    t_simi = F.cosine_similarity(f1[:,None,:], f2[None,:,:], dim=2)

                    # kl
                    s_simi = F.log_softmax(s_simi, dim=1)
                    t_simi = F.softmax(t_simi, dim=1)
                    kd_simi_loss += F.kl_div(s_simi, t_simi, reduction='batchmean')

                training_stats.report('Loss/distill/l1', kd_l1_loss)
                training_stats.report('Loss/distill/lpip', kd_lpips_loss)
                training_stats.report('Loss/distill/simi_ld', kd_simi_loss)
                loss_Gmain = loss_Gmain + lambda_l1 * kd_l1_loss + lambda_pe * kd_lpips_loss + lambda_simi_loss * kd_simi_loss
            with torch.autograd.profiler.record_function('Gmain_backward'):
                loss_Gmain.mean().mul(gain).backward()

        # Gpl: Apply path length regularization.
        if phase in ['Greg', 'Gboth']:
            with torch.autograd.profiler.record_function('Gpl_forward'):
                batch_size = gen_z.shape[0] // self.pl_batch_shrink
                gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c[:batch_size])
                pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
                with torch.autograd.profiler.record_function('pl_grads'), conv2d_gradfix.no_weight_gradients(self.pl_no_weight_grad):
                    pl_grads = torch.autograd.grad(outputs=[(gen_img * pl_noise).sum()], inputs=[gen_ws], create_graph=True, only_inputs=True)[0]
                pl_lengths = pl_grads.square().sum(2).mean(1).sqrt()
                pl_mean = self.pl_mean.lerp(pl_lengths.mean(), self.pl_decay)
                self.pl_mean.copy_(pl_mean.detach())
                pl_penalty = (pl_lengths - pl_mean).square()
                training_stats.report('Loss/pl_penalty', pl_penalty)
                loss_Gpl = pl_penalty * self.pl_weight
                training_stats.report('Loss/G/reg', loss_Gpl)
            with torch.autograd.profiler.record_function('Gpl_backward'):
                loss_Gpl.mean().mul(gain).backward()

        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, update_emas=True)
                gen_logits = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Dgen = torch.nn.functional.softplus(gen_logits) # -log(1 - sigmoid(gen_logits))
            with torch.autograd.profiler.record_function('Dgen_backward'):
                loss_Dgen.mean().mul(gain).backward()

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp = real_img.detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_logits = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits) # -log(sigmoid(real_logits))
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                        r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
                    r1_penalty = r1_grads.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (self.r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1).mean().mul(gain).backward()

#----------------------------------------------------------------------------
