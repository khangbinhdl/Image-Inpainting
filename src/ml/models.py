from __future__ import annotations

import torch
import torch.nn as nn
from diffusers import UNet2DModel

IMAGE_SIZE = 64
UNET_BLOCK_OUT_CHANNELS = (128, 128, 256, 256)
UNET_DOWN_BLOCK_TYPES = ("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D")
UNET_UP_BLOCK_TYPES = ("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D")
LAYERS_PER_BLOCK = 2
VAE_CHANNELS = (128, 128, 256, 256)


def compose_inpaint(pred_image: torch.Tensor, masked_image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return masked_image * (1 - mask) + pred_image * mask


def build_unet_7ch() -> UNet2DModel:
    return UNet2DModel(
        sample_size=IMAGE_SIZE,
        in_channels=7,
        out_channels=3,
        layers_per_block=LAYERS_PER_BLOCK,
        block_out_channels=UNET_BLOCK_OUT_CHANNELS,
        down_block_types=UNET_DOWN_BLOCK_TYPES,
        up_block_types=UNET_UP_BLOCK_TYPES,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_norm: bool = True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_norm)]
        if use_norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.SiLU(inplace=True))
        self.block = nn.Sequential(*layers)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, layers: int = 2):
        super().__init__()
        blocks = [ConvBlock(in_ch, out_ch)] + [ConvBlock(out_ch, out_ch) for _ in range(layers - 1)]
        self.res = nn.Sequential(*blocks)
        self.down = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        skip = self.res(x)
        out = self.down(skip)
        return out, skip


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, layers: int = 2, upscale_factor: int = 2):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * upscale_factor ** 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(upscale_factor),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )
        blocks = [ConvBlock(out_ch + skip_ch, out_ch)] + [ConvBlock(out_ch, out_ch) for _ in range(layers - 1)]
        self.res = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class ConditionalUNetVAE(nn.Module):
    def __init__(self, image_size: int = IMAGE_SIZE, z_dim: int = 256, channels=VAE_CHANNELS, layers_per_block: int = 2):
        super().__init__()
        self.image_size = image_size
        self.z_dim = z_dim
        self.channels = channels
        c1, c2, c3, c4 = channels

        self.cond_in = nn.Sequential(nn.Conv2d(4, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(inplace=True))
        self.cond_d1 = DownBlock(c1, c1, layers_per_block)
        self.cond_d2 = DownBlock(c1, c2, layers_per_block)
        self.cond_d3 = DownBlock(c2, c3, layers_per_block)
        self.cond_d4 = DownBlock(c3, c4, layers_per_block)
        self.cond_mid = ConvBlock(c4, c4)

        self.post_in = nn.Sequential(nn.Conv2d(7, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1), nn.SiLU(inplace=True))
        self.post_d1 = DownBlock(c1, c1, layers_per_block)
        self.post_d2 = DownBlock(c1, c2, layers_per_block)
        self.post_d3 = DownBlock(c2, c3, layers_per_block)
        self.post_d4 = DownBlock(c3, c4, layers_per_block)
        self.post_mid = ConvBlock(c4, c4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(c4 * 2, z_dim)
        self.fc_logvar = nn.Linear(c4 * 2, z_dim)

        self.latent_hw = image_size // 32
        self.fc_z = nn.Linear(z_dim, c4 * self.latent_hw * self.latent_hw)
        self.z_up = nn.Sequential(
            nn.Conv2d(c4, c4 * 4, 3, padding=1, bias=False), nn.PixelShuffle(2), nn.BatchNorm2d(c4), nn.SiLU(inplace=True)
        )
        self.dec_mid = ConvBlock(c4 * 2, c4)
        self.up4 = UpBlock(c4, c4, c3, layers_per_block)
        self.up3 = UpBlock(c3, c3, c2, layers_per_block)
        self.up2 = UpBlock(c2, c2, c1, layers_per_block)
        self.up1 = UpBlock(c1, c1, c1, layers_per_block)
        self.out = nn.Sequential(nn.BatchNorm2d(c1), nn.SiLU(inplace=True), nn.Conv2d(c1, 3, 3, padding=1), nn.Tanh())

    def encode_condition(self, cond_image: torch.Tensor, mask: torch.Tensor):
        x = torch.cat([cond_image, mask], dim=1)
        x = self.cond_in(x)
        x, s1 = self.cond_d1(x)
        x, s2 = self.cond_d2(x)
        x, s3 = self.cond_d3(x)
        x, s4 = self.cond_d4(x)
        return self.cond_mid(x), (s1, s2, s3, s4)

    def decode(self, z: torch.Tensor, cond_image: torch.Tensor, mask: torch.Tensor, cond_feats):
        cond_deep, skips = cond_feats
        s1, s2, s3, s4 = skips
        h = self.fc_z(z).view(z.size(0), self.channels[-1], self.latent_hw, self.latent_hw)
        h = self.z_up(h)
        h = self.dec_mid(torch.cat([h, cond_deep], dim=1))
        h = self.up4(h, s4)
        h = self.up3(h, s3)
        h = self.up2(h, s2)
        h = self.up1(h, s1)
        pred_image = self.out(h)
        return compose_inpaint(pred_image, cond_image, mask), pred_image

    @torch.no_grad()
    def sample(self, cond_image: torch.Tensor, mask: torch.Tensor):
        cond_feats = self.encode_condition(cond_image, mask)
        z = torch.randn(cond_image.size(0), self.z_dim, device=cond_image.device, dtype=cond_image.dtype)
        return self.decode(z, cond_image, mask, cond_feats)
