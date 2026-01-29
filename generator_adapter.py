"""
生成器适配模块

提供统一的接口，用于将预测的latent解码成图像。
默认优先使用VAE Decoder（如diffusers AutoencoderKL），若不可用则回退到可运行的Stub。
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GeneratorAdapter(nn.Module):
    """生成器适配基类"""

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface only
        raise NotImplementedError

    def load_pretrained(self, path: Optional[str]) -> None:  # pragma: no cover - interface only
        raise NotImplementedError


class StubDecoder(nn.Module):
    """最小可运行的Decoder Stub，用于缺省权重或依赖不可用时。"""

    def __init__(self, z_channels: int = 4, latent_down: int = 8) -> None:
        super().__init__()
        if latent_down <= 0:
            raise ValueError("latent_down must be positive")
        self.z_channels = z_channels
        self.latent_down = latent_down
        out_channels = 3 * (latent_down ** 2)
        self.conv = nn.Conv2d(z_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(latent_down)
        self.act = nn.Tanh()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        x = self.conv(latent)
        x = self.pixel_shuffle(x)
        x = self.act(x)
        return x


class VAEGeneratorAdapter(GeneratorAdapter):
    """VAE Decoder适配器（优先diffusers AutoencoderKL）"""

    def __init__(
        self,
        z_channels: int = 4,
        latent_down: int = 8,
        generator_type: str = "vae",
        pretrained_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.z_channels = z_channels
        self.latent_down = latent_down
        self.generator_type = generator_type
        self.pretrained_path = pretrained_path
        self._use_diffusers = False
        self._init_generator(pretrained_path)

    def _init_generator(self, pretrained_path: Optional[str]) -> None:
        has_diffusers = importlib.util.find_spec("diffusers") is not None
        if pretrained_path and has_diffusers:
            from diffusers import AutoencoderKL

            self.vae = AutoencoderKL.from_pretrained(pretrained_path)
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False
            self._use_diffusers = True
        else:
            if pretrained_path and not has_diffusers:
                warnings.warn(
                    "diffusers不可用，无法加载VAE权重，使用StubDecoder。",
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    "未提供VAE权重路径，使用StubDecoder。",
                    RuntimeWarning,
                )
            self.decoder = StubDecoder(z_channels=self.z_channels, latent_down=self.latent_down)
            self.decoder.eval()
            for param in self.decoder.parameters():
                param.requires_grad = False
            self._use_diffusers = False

    def load_pretrained(self, path: Optional[str]) -> None:
        self.pretrained_path = path
        self._init_generator(path)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if self._use_diffusers:
            decoded = self.vae.decode(latent).sample
        else:
            decoded = self.decoder(latent)
        decoded = (decoded + 1.0) / 2.0
        return torch.clamp(decoded, 0.0, 1.0)
