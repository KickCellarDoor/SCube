# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import gc
import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import OmegaConf
from pathlib import Path
from loguru import logger
from scube.modules.render.gsplat_renderer import PinholeCamera, render_gsplat_api
from scube.data.base import DatasetSpec as DS
from scube.modules.gsm_modules.renderer.heads import NaiveConvHead, ResConvHead

class FeatureRenderer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.decoder_head = eval(self.hparams.renderer.head.target)(self.hparams.renderer.head.params)
        self.decoder_for_sky_only = getattr(self.hparams.renderer, 'decoder_for_sky_only', False)
        logger.info(f"FeatureRenderer decoder_for_sky_only: {self.decoder_for_sky_only}")

    def save_decoder(self, gaussian_saving_path):
        """
        dump decoder configuration to yaml and save the model weight
        """

        decoder_config = OmegaConf.create({
            'target': self.hparams.renderer.head.target,
            'params': self.hparams.renderer.head.params
        })

        gs_stem = Path(gaussian_saving_path).with_suffix('')

        # save yaml
        with open(f'{gs_stem}_decoder.yaml', 'w') as f:
            OmegaConf.save(config=decoder_config, f=f)

        # save model weight
        torch.save(self.decoder_head.state_dict(), f'{gs_stem}_decoder.pt')


    def prepare_rasterizing_params(self, batch):
        target_intrinsics = torch.stack(batch[DS.IMAGES_INTRINSIC])
        rasterizing_target_intrinsics = target_intrinsics / self.hparams.rasterizing_downsample
    
        target_poses = torch.stack(batch[DS.IMAGES_POSE])

        rasterizing_params = {
            'target_poses': target_poses,
            'rasterizing_target_intrinsics': rasterizing_target_intrinsics
        }
        return rasterizing_params
    

    def gsplat_render(self, rasterizing_params: dict, network_output: dict, skybox):
        """
        rasterizing_params:
            target_poses:
                [B, N, 4, 4]
            rasterizing_target_intrinsics:
                [B, N, 6]
        """
        target_poses = rasterizing_params['target_poses']
        rasterizing_target_intrinsics = rasterizing_params['rasterizing_target_intrinsics']

        assert self.hparams.with_render_branch and 'decoded_gaussians' in network_output.keys()

        batch_rendered_images = []
        batch_rendered_images_fg = []
        batch_rendered_alphas = []
        batch_rendered_depth = []

        decoded_gaussians = network_output['decoded_gaussians']
        renderer_output = {}

        for batch_idx in range(len(decoded_gaussians)):
            gaussians: torch.Tensor = decoded_gaussians[batch_idx]
            target_poses_one_batch = target_poses[batch_idx]
            target_intrinsic = rasterizing_target_intrinsics[batch_idx]

            one_sample_cameras = []
            for i, camera_pose in enumerate(target_poses_one_batch):
                intrinsic = target_intrinsic[i]
                camera = PinholeCamera('cuda', int(intrinsic[5]), int(intrinsic[4]), intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3])
                camera.pose.set_from_torch(camera_pose)
                one_sample_cameras.append(camera)

            assert self.hparams.use_skybox == True, "we want use_skybox to be enabled"
            # render_features shape [N_views, H, W, C] (C can be feature dim), depth and alpha shape [N_views, H, W, 1]
            render_features, render_depths, render_alphas = render_gsplat_api(
                                                            one_sample_cameras, None,
                                                            gaussians[:, :3], gaussians[:, 3:6], 
                                                            gaussians[:, 6:10], gaussians[:, 10:11],
                                                            gaussians[:, 11:],
                                                            bg=None,
                                                            **self.hparams.gsplat_params)

            sky_featrues = skybox.sample_batch(target_poses_one_batch, target_intrinsic,
                                            network_output, batch_idx)

            if self.decoder_for_sky_only:
                assert render_features.shape[-1] == 3, "3D gaussians should store RGB values"
                rendered_image = render_features
                sky_image = self.decoder_head(sky_featrues)
                decoded_render_image_full = rendered_image + (1 - render_alphas) * sky_image
                batch_rendered_images_fg.append(rendered_image)

            else:
                render_features_full = render_features + (1 - render_alphas) * sky_featrues
                decoded_render_image_full = self.decoder_head(render_features_full)
                batch_rendered_images_fg.append(torch.ones_like(decoded_render_image_full))

            batch_rendered_images.append(decoded_render_image_full)
            batch_rendered_alphas.append(render_alphas)
            batch_rendered_depth.append(render_depths)

        
        # torch.cuda.empty_cache()
        # stack for batch
        renderer_output.update({'pd_images': torch.stack(batch_rendered_images, dim=0)})
        renderer_output.update({'pd_images_fg': torch.stack(batch_rendered_images_fg, dim=0)})
        renderer_output.update({'pd_depths': torch.stack(batch_rendered_depth, dim=0)})
        renderer_output.update({'pd_alphas': torch.stack(batch_rendered_alphas, dim=0)})

        return renderer_output

    def forward(self, batch: dict, network_output: dict, skybox) -> dict:
        rasterizing_params = self.prepare_rasterizing_params(batch)
        renderer_output = self.gsplat_render(rasterizing_params, network_output, skybox)
        return renderer_output