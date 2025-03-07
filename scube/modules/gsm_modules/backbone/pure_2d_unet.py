import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scube.utils.depth_util import project_2D_to_3D
from scube.data.base import DatasetSpec as DS


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class Pure2DUnet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, gsplat_upsample=2, znear=0.5, zfar=300, scale_factor=None):
        super(Pure2DUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.gsplat_upsample = gsplat_upsample
        self.znear = znear
        self.zfar = zfar
        self.scale_factor = scale_factor

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x, batch):
        if self.scale_factor is not None:
            x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False, antialias=False)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        gaussians = self.decode(logits, batch)

        return gaussians

    def decode(self, logits, batch):
        # logits N, C, H, W
        decoded_gaussians = []
        B = len(batch[DS.IMAGES_INPUT_INTRINSIC])
        logits = rearrange(logits, '(b n) c h w -> b n c h w', b=B)

        for batch_idx in range(B):
            cur_decoded_gaussians = []
            for i in range(self.gsplat_upsample):
                start_idx = i * 12
                colors = logits[batch_idx, :, start_idx:start_idx+3, :, :]
                rotations = logits[batch_idx, :, start_idx+3:start_idx+7, :, :]
                scales = logits[batch_idx, :, start_idx+7:start_idx+10, :, :]
                opacities = logits[batch_idx, :, start_idx+10:start_idx+11, :, :]
                depths = logits[batch_idx, :, start_idx+11:start_idx+12, :, :]
                intrinsics = batch[DS.IMAGES_INPUT_INTRINSIC][batch_idx]
                poses = batch[DS.IMAGES_INPUT_POSE][batch_idx]

                depths = torch.sigmoid(depths)
                depths = (1 - depths) * self.znear + depths * self.zfar

                resize_h, resize_w = colors.shape[2], colors.shape[3]

                downsample_ratio_h = intrinsics[0, 5].int().item() / resize_h
                downsample_ratio_w = intrinsics[0, 4].int().item() / resize_w

                intrinsics_image = intrinsics.clone()
                intrinsics_image[:, 0] /= downsample_ratio_w
                intrinsics_image[:, 1] /= downsample_ratio_h
                intrinsics_image[:, 2] /= downsample_ratio_w
                intrinsics_image[:, 3] /= downsample_ratio_h 
                intrinsics_image[:, 4] = resize_w
                intrinsics_image[:, 5] = resize_h

                locations = project_2D_to_3D(
                    depths, 
                    intrinsics_image, 
                    poses
                )
                colors = rearrange(colors, 'b c h w ->(b h w) c')
                rotations = rearrange(rotations, 'b c h w ->(b h w) c')
                scales = rearrange(scales, 'b c h w ->(b h w) c')
                opacities = rearrange(opacities, 'b c h w ->(b h w) c')
                locations = rearrange(locations, 'b n c ->(b n) c')
                opacities = torch.sigmoid(opacities - 2.0)
                rotations = F.normalize(rotations, dim=1)
                scales = torch.clamp_max(torch.exp(scales - 2.3), 0.3)
                cur_decoded_gaussians.append(
                    torch.concat(
                        [locations, scales, rotations, opacities, colors], dim=1
                    )
                )
            decoded_gaussians.append(torch.concat(cur_decoded_gaussians, dim=0))

        network_output = {'decoded_gaussians': decoded_gaussians}

        return network_output