# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import gc

import fvdb
import fvdb.nn as fvnn
import numpy as np
import torch
import torch.nn.functional as F

import time
import webdataset as wds

from torch.utils.data import DataLoader
from fvdb import GridBatch, JaggedTensor
from loguru import logger
from pycg import exp
from icecream import ic
from einops import rearrange

from scube.data import build_dataset
from scube.data.base import DatasetSpec as DS
from scube.data.base import list_collate
from scube.models.base_model import BaseModel

from scube.modules.gsm_modules.encoder.unified_encoder import UnifiedEncoder
from scube.modules.gsm_modules.backbone import Pure3DUnet, Pure2DUnet
from scube.modules.gsm_modules.renderer import FeatureRenderer, RGBRenderer
from scube.modules.gsm_modules.loss.unified_loss import UnifiedLoss
from scube.modules.sky import SkyboxPanoramaFull, SkyboxNull, convert_to_camel_case

from scube.modules.gsm_modules.hparams import hparams_handler
from scube.modules.render.gsplat_renderer import IsoTransform, PinholeCamera
from scube.utils.voxel_util import generate_grid_mask_for_batch_data, keep_surface_voxels, prepare_semantic_jagged_tensor, get_distance_from_voxel
from scube.utils.voxel_util import clip_batch_grid, coarsen_batch_grid 
from scube.utils.depth_util import vis_depth
from scube.utils.common_util import mask_image_patches
from scube.utils.render_util import create_rays_from_intrinsic_torch_batch

def lambda_lr_wrapper(it, lr_config, batch_size, accumulate_grad_batches=1):
    return max(
        lr_config['decay_mult'] ** (int(it * batch_size * accumulate_grad_batches / lr_config['decay_step'])),
        lr_config['clip'] / lr_config['init'])

class Model(BaseModel):
    def __init__(self, hparams):
        hparams = hparams_handler(hparams) # ! put it here
        super().__init__(hparams)
        self.img_encoder = UnifiedEncoder(self.hparams)
        if self.hparams.use_skybox:
            self.skybox = eval("Skybox" + convert_to_camel_case(self.hparams.skybox_target))(self.hparams)
        else:
            self.skybox = SkyboxNull(hparams)
        self.backbone = eval(self.hparams.backbone.target)(**self.hparams.backbone.params) 
        self.renderer = eval(self.hparams.renderer.target)(self.hparams)
        self.loss = UnifiedLoss(self.hparams)

    def forward(self, batch, update_grid_mask=True, update_grid_distance=True):
        self.voxel_preprocess(batch, update_grid_mask=update_grid_mask, update_grid_distance=update_grid_distance)
        imgenc_output = self.img_encoder(batch)
        skyenc_output = self.skybox.encode_sky_feature(batch, imgenc_output)

        images = torch.stack(batch[DS.IMAGES_INPUT])
        depth = torch.stack(batch[DS.IMAGES_INPUT_DEPTH])

        depth = torch.clamp(depth - self.hparams.znear, min=0, max=self.hparams.zfar - self.hparams.znear) / (self.hparams.zfar - self.hparams.znear) 

        masked_depth = mask_image_patches(depth, P=16, p_mask=0.5)
        target_h, target_w = images.shape[2], images.shape[3]

        dav2_features = rearrange(imgenc_output['dav2'], 'b n c h w -> (b n) c h w')

        images = rearrange(images, 'b n h w c -> (b n) c h w')
        masked_depth = rearrange(masked_depth, 'b n h w c -> (b n) c h w')
        dav2_features = F.interpolate(dav2_features, (target_h, target_w), 
                                        mode='bilinear', align_corners=False, antialias=False)

        features = torch.cat(
            [dav2_features, images, masked_depth],
            dim=1
        )

        network_output = self.backbone(features, batch)
        network_output = self.skybox(skyenc_output, network_output)

        if not self.training:
            print("In eval mode, free the memory")
            torch.cuda.empty_cache()

        renderer_output = self.renderer(batch, network_output, self.skybox)
        return renderer_output, network_output


    def train_val_step(self, batch, batch_idx, is_val):
        renderer_output, network_output = self(batch)
        loss_dict, metric_dict, latent_dict, render_imgs_dict = self.loss(batch, renderer_output, network_output,
                                                                          compute_metric=is_val,
                                                                          global_step=self.global_step,
                                                                          current_epoch=self.current_epoch)

        if not is_val:
            self.log_dict_prefix('train_loss', loss_dict, prog_bar=True)
            self.log_dict_prefix('train_loss', latent_dict)
            self.log_dict_prefix('train_metric', metric_dict)

            visualize_global_step = getattr(self.hparams, 'visualize_global_step', 500)
            if self.trainer.global_step % visualize_global_step == 0:
                gt_alphas_close_sample = render_imgs_dict['gt_alphas_close'][0].permute(0, 3, 1, 2)
                gt_alphas_midground_sample = render_imgs_dict['gt_alphas_midground'][0].permute(0, 3, 1, 2)

                self.log_image("TRAIN pd_images <==> pd_images_foreground <==> gt_images", 
                                [torch.clamp(render_imgs_dict['pd_images'][0].permute(0, 3, 1, 2), min=0, max=1),
                                torch.clamp(render_imgs_dict['pd_images_fg'][0].permute(0, 3, 1, 2), min=0, max=1),
                                torch.clamp(render_imgs_dict['gt_images'][0].permute(0, 3, 1, 2), min=0, max=1)])
                
                if self.hparams.render_alpha:
                    self.log_image("TRAIN pd_alphas <==> gt_alphas <==> gt_alphas_close <==> gt_alphas_midground", 
                                    [torch.clamp(render_imgs_dict['pd_alphas'][0].permute(0, 3, 1, 2), min=0, max=1),
                                    torch.clamp(render_imgs_dict['gt_alphas'][0].permute(0, 3, 1, 2), min=0, max=1),
                                    gt_alphas_close_sample,
                                    gt_alphas_midground_sample])
                    
                if self.hparams.use_skybox and (self.hparams.skybox_net == 'identity' or self.hparams.skybox_net.endswith('decode-3')):
                    if self.skybox.visualize(network_output).shape[-1] == 3: 
                        self.log_image("TRAIN pd_skybox", [torch.clamp(self.skybox.visualize(network_output).permute(0, 3, 1, 2), min=0, max=1)])

                if self.hparams.use_sup_depth:
                    N = render_imgs_dict['pd_depths'].shape[1]
                    pd_depths_sample = torch.stack([vis_depth(render_imgs_dict['pd_depths'][0, i].squeeze(-1)) for i in range(N)]).permute(0, 3, 1, 2)
                    gt_depths_sample = torch.stack([vis_depth(render_imgs_dict['gt_depths'][0, i].squeeze(-1)) for i in range(N)]).permute(0, 3, 1, 2)
                    self.log_image("TRAIN pd_depth <==> pd_depth_close <==> pd_depth_midground <==> gt_depth <==> depth_loss_mask",
                                    [pd_depths_sample,
                                        pd_depths_sample * gt_alphas_close_sample.to(pd_depths_sample),
                                        pd_depths_sample * gt_alphas_midground_sample.to(pd_depths_sample),
                                        gt_depths_sample,
                                        render_imgs_dict['depth_loss_mask'][0].float().permute(0, 3, 1, 2)])

        else:
            self.log_dict_prefix('val_metric', metric_dict)
            self.log_dict_prefix('val_loss', loss_dict)
            self.log_dict_prefix('val_loss', latent_dict)

            if self.trainer.global_rank == 0: # only log the image on rank 0
                if batch_idx == 0 or batch_idx % self.val_sample_interval == 0: 
                    logger.info("running visualisation on rank 0...")

                    gt_alphas_close_sample = render_imgs_dict['gt_alphas_close'][0].permute(0, 3, 1, 2)
                    gt_alphas_midground_sample = render_imgs_dict['gt_alphas_midground'][0].permute(0, 3, 1, 2)

                    self.log_image("VAL pd_images <==> pd_images_foreground <==> gt_images", 
                                   [torch.clamp(render_imgs_dict['pd_images'][0].permute(0, 3, 1, 2), min=0, max=1),
                                    torch.clamp(render_imgs_dict['pd_images_fg'][0].permute(0, 3, 1, 2), min=0, max=1),
                                    torch.clamp(render_imgs_dict['gt_images'][0].permute(0, 3, 1, 2), min=0, max=1)])
                    
                    if self.hparams.render_alpha:
                        self.log_image("VAL pd_alphas <==> gt_alphas <==> gt_alphas_close <==> gt_alphas_midground", 
                                       [torch.clamp(render_imgs_dict['pd_alphas'][0].permute(0, 3, 1, 2), min=0, max=1),
                                        torch.clamp(render_imgs_dict['gt_alphas'][0].permute(0, 3, 1, 2), min=0, max=1),
                                        gt_alphas_close_sample,
                                        gt_alphas_midground_sample])
                        
                    if self.hparams.use_skybox and (self.hparams.skybox_net == 'identity' or self.hparams.skybox_net.endswith('decode-3')):
                        if self.skybox.visualize(network_output).shape[-1] == 3: 
                            self.log_image("VAL pd_skybox", [torch.clamp(self.skybox.visualize(network_output).permute(0, 3, 1, 2), min=0, max=1)])

                    if self.hparams.use_sup_depth:
                        N = render_imgs_dict['pd_depths'].shape[1]
                        pd_depths_sample = torch.stack([vis_depth(render_imgs_dict['pd_depths'][0, i].squeeze(-1)) for i in range(N)]).permute(0, 3, 1, 2)
                        gt_depths_sample = torch.stack([vis_depth(render_imgs_dict['gt_depths'][0, i].squeeze(-1)) for i in range(N)]).permute(0, 3, 1, 2)
                        self.log_image("VAL pd_depth <==> pd_depth_close <==> pd_depth_midground <==> gt_depth <==> depth_loss_mask",
                                        [pd_depths_sample,
                                         pd_depths_sample * gt_alphas_close_sample.to(pd_depths_sample),
                                         pd_depths_sample * gt_alphas_midground_sample.to(pd_depths_sample),
                                         gt_depths_sample,
                                         render_imgs_dict['depth_loss_mask'][0].float().permute(0, 3, 1, 2)])
            
        loss_sum = loss_dict.get_sum()
        self.log('val_loss' if is_val else 'train_loss/sum', loss_sum)
        self.log('val_step', self.global_step)

        # torch.cuda.empty_cache()

        return loss_sum

    def get_dataset_spec(self):
        all_specs = [DS.SHAPE_NAME, DS.INPUT_PC, DS.GT_SEMANTIC]
 
        all_specs.append(DS.IMAGES_INPUT)
        all_specs.append(DS.IMAGES_INPUT_MASK)
        all_specs.append(DS.IMAGES_INPUT_POSE)
        all_specs.append(DS.IMAGES_INPUT_INTRINSIC)

        all_specs.append(DS.IMAGES)
        all_specs.append(DS.IMAGES_MASK)
        all_specs.append(DS.IMAGES_POSE)
        all_specs.append(DS.IMAGES_INTRINSIC)
        
        if self.hparams.use_sup_depth and self.hparams.sup_depth_type == 'rectified_metric3d_depth':
            all_specs.append(DS.IMAGES_DEPTH_MONO_EST_RECTIFIED)
        if self.hparams.use_sup_depth and self.hparams.sup_depth_type == 'lidar_depth':
            all_specs.append(DS.IMAGES_DEPTH_LIDAR_PROJECT)
        if self.hparams.use_sup_depth and self.hparams.sup_depth_type == 'depth_anything_v2_depth_inv':
            all_specs.append(DS.IMAGES_DEPTH_ANYTHING_V2_DEPTH_INV)
        if self.hparams.use_sup_depth and self.hparams.sup_depth_type == 'voxel_depth':
            pass # voxel depth is generated on the fly

        return all_specs
    
    def get_collate_fn(self):
        return list_collate

    def get_hparams_metrics(self):
        return [('val_loss', True)]

    def configure_optimizers(self):
        # overwrite this from base model to fix pretrained vae layer
        lr_config = self.hparams.learning_rate
        # parameters = list(self.parameters())
        parameters = list(self.img_encoder.parameters())
        parameters += list(self.backbone.parameters())
        parameters += list(self.renderer.parameters())
        parameters += list(self.loss.parameters())

        if self.hparams.use_skybox:
            parameters += list(self.skybox.parameters())

        if self.hparams.optimizer == 'SGD':
            optimizer = torch.optim.SGD(parameters, lr=lr_config['init'], momentum=0.9,
                                        weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer == 'Adam':
            # AdamW corrects the bad weight dacay implementation in Adam.
            # AMSGrad also do some corrections to the original Adam.
            # The learning rate here is the maximum rate we can reach for each parameter.
            optimizer = torch.optim.AdamW(parameters, lr=lr_config['init'],
                                          weight_decay=self.hparams.weight_decay, amsgrad=True)        
        else:
            raise NotImplementedError

        # build scheduler
        import functools
        from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
        scheduler = LambdaLR(optimizer,
                             lr_lambda=functools.partial(
                                 lambda_lr_wrapper, lr_config=lr_config, batch_size=self.hparams.batch_size, accumulate_grad_batches=self.trainer.accumulate_grad_batches))

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]

    # update on 2023-05-15: set up the batchsize to avoid using world_size
    def train_dataset(self):
        return build_dataset(
            self.hparams.train_dataset, self.get_dataset_spec(), self.hparams, self.hparams.train_kwargs, duplicate_num=self.hparams.duplicate_num)

    def train_dataloader(self):
        train_set = self.train_dataset()
        return DataLoader(train_set, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.train_val_num_workers, collate_fn=self.get_collate_fn())

    def val_dataset(self):
        return build_dataset(
            self.hparams.val_dataset, self.get_dataset_spec(), self.hparams, self.hparams.val_kwargs)

    def val_dataloader(self):
        val_set = self.val_dataset()
        return DataLoader(val_set, batch_size=self.hparams.batch_size,
                          num_workers=0, collate_fn=self.get_collate_fn())
    
    def test_dataset(self):
        return build_dataset(
            self.hparams.test_dataset, self.get_dataset_spec(), self.hparams, self.hparams.test_kwargs)

    def test_dataloader(self):
        test_set = self.test_dataset()
        if self.hparams.test_set_shuffle:
            torch.manual_seed(0)
        if not hasattr(self.hparams, 'batch_len'):
            self.hparams.batch_len = 1
        print("===> batch_len: %d" % self.hparams.batch_len)
        return DataLoader(test_set, batch_size=self.hparams.batch_len, 
                          num_workers=0, collate_fn=self.get_collate_fn())


    def voxel_preprocess(self, batch, update_grid_mask=True, update_grid_distance=False):
        self.generate_fvdb_grid_on_the_fly(batch)
        prepare_semantic_jagged_tensor(batch)
        if self.hparams.clip_input_grid:
            clip_batch_grid(batch, self.hparams.ijk_min, self.hparams.ijk_max)
        if self.hparams.coarsen_input_grid:
            coarsen_batch_grid(batch, self.hparams.coarsen_factor)
        if self.hparams.keep_surface_voxels:
            keep_surface_voxels(batch)

        if update_grid_mask:
            generate_grid_mask_for_batch_data(batch, self.hparams.use_high_res_grid_for_alpha_mask)
        if update_grid_distance:
            if DS.IMAGES_INPUT_POSE in batch:
                B = len(batch[DS.IMAGES_INPUT_POSE])
                batch[DS.IMAGES_INPUT_DEPTH] = []
                for i in range(B):
                    poses = batch[DS.IMAGES_INPUT_POSE][i]
                    intrinsics = batch[DS.IMAGES_INPUT_INTRINSIC][i]
                    grid = batch[DS.INPUT_PC][i]
                    distance, depth = get_distance_from_voxel(poses, intrinsics, grid, return_depth=True)
                    batch[DS.IMAGES_INPUT_DEPTH].append(distance)

            if DS.IMAGES_POSE in batch:
                B = len(batch[DS.IMAGES_POSE])
                batch[DS.IMAGES_DEPTH_VOXEL] = []
                for i in range(B):
                    poses = batch[DS.IMAGES_POSE][i]
                    intrinsics = batch[DS.IMAGES_INTRINSIC][i]
                    grid = batch[DS.INPUT_PC][i]
                    distance, depth = get_distance_from_voxel(poses, intrinsics, grid, return_depth=True)
                    batch[DS.IMAGES_DEPTH_VOXEL].append(distance)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'loss_fn_alex' in k or 'perceptual_loss' in k:
                del state_dict[k]
        return state_dict
    
    # ! override the load_state_dict to avoid loading the lpips_loss
    def load_state_dict(self, state_dict, strict: bool = False):
        return super().load_state_dict(state_dict, strict)


    def on_validation_epoch_start(self):
        # random a int between 0 and 10
        self.val_sample_interval = 500
        logger.info(f"val_sample_interval: {self.val_sample_interval}")