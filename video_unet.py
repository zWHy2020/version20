"""
视频U-Net解码器

支持输入 [B, T, C, H, W]，通过展平为2D特征再还原回视频序列。
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """轻量级残差块，使用GroupNorm确保小batch稳定。"""

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=min(groups, in_channels), num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=min(groups, out_channels), num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=False)
        self.skip = None
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        if self.skip is not None:
            identity = self.skip(identity)
        return x + identity


class VideoUNetDecoder(nn.Module):
    """
    视频U-Net解码器。

    Args:
        in_channels: 输入通道（来自视频编码器的特征维度）。
        out_channels: 输出通道（通常为3）。
        base_channels: U-Net最浅层通道数。
        num_down: 下采样层数（包含最浅层）。
        num_res_blocks: 每层残差块数量。
        use_tanh: 是否在输出端使用Tanh，输出范围在[-1, 1]。
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,
        base_channels: int = 64,
        num_down: int = 4,
        num_res_blocks: int = 2,
        use_tanh: bool = True,
        normalize_output: bool = False,
        decode_chunk_size: Optional[int] = None,
    ):
        super().__init__()
        if num_down < 1:
            raise ValueError(f"num_down must be >= 1, got {num_down}")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.num_down = num_down
        self.num_res_blocks = num_res_blocks
        self.use_tanh = use_tanh
        self.normalize_output = normalize_output
        self.decode_chunk_size = decode_chunk_size

        self.in_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        channels = [base_channels * (2 ** i) for i in range(num_down)]
        for i, ch in enumerate(channels):
            res_layers = [ResBlock(ch, ch) for _ in range(num_res_blocks)]
            self.down_blocks.append(nn.Sequential(*res_layers))
            if i < num_down - 1:
                self.downsamples.append(
                    nn.Conv2d(ch, channels[i + 1], kernel_size=4, stride=2, padding=1)
                )

        for i in range(num_down - 2, -1, -1):
            up_in = channels[i + 1]
            up_out = channels[i]
            self.upsamples.append(
                nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(up_in, up_out, kernel_size=3, padding=1),
                )
            )
            res_layers = [ResBlock(up_out * 2, up_out)]
            res_layers.extend(ResBlock(up_out, up_out) for _ in range(num_res_blocks - 1))
            self.up_blocks.append(nn.Sequential(*res_layers))

        self.out_conv = nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1)
        self.out_act = nn.Tanh() if use_tanh else nn.Identity()
        self.register_buffer("output_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("output_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(
        self,
        noisy_features: torch.Tensor,
        guide_vectors: Optional[torch.Tensor] = None,
        reset_state: bool = False,
        semantic_context: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_features: [B, T, C, H, W]
            guide_vectors: 未使用，保持接口兼容。
            reset_state: 未使用，保持接口兼容。
            semantic_context: 未使用，保持接口兼容。
            output_size: 可选输出裁剪尺寸 (H, W)。
        Returns:
            重建视频 [B, T, C_out, H, W]
        """
        if noisy_features.dim() != 5:
            raise ValueError(
                f"VideoUNetDecoder expects [B, T, C, H, W], got {tuple(noisy_features.shape)}"
            )
        b, t, c, h, w = noisy_features.shape
        chunk_size = self.decode_chunk_size
        if chunk_size is None or chunk_size <= 0 or chunk_size >= t:
            x = noisy_features.reshape(b * t, c, h, w)
            x = self._forward_features(x, h, w, output_size)
            return x.view(b, t, self.out_channels, *x.shape[-2:])

        decoded_chunks = []
        for chunk in torch.split(noisy_features, chunk_size, dim=1):
            chunk_t = chunk.size(1)
            x = chunk.reshape(b * chunk_t, c, h, w)
            x = self._forward_features(x, h, w, output_size)
            x = x.view(b, chunk_t, self.out_channels, *x.shape[-2:])
            decoded_chunks.append(x)
        return torch.cat(decoded_chunks, dim=1)

    def _forward_features(
        self,
        x: torch.Tensor,
        h: int,
        w: int,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        x = self.in_conv(x)

        skips = []
        for i, down in enumerate(self.down_blocks):
            x = down(x)
            skips.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        for i, upsample in enumerate(self.upsamples):
            x = upsample(x)
            skip = skips[-(i + 2)]
            if x.shape[-2:] != skip.shape[-2:]:
                x = torch.nn.functional.interpolate(x, size=skip.shape[-2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = self.up_blocks[i](x)

        x = self.out_act(self.out_conv(x))
        x = (x + 1.0) / 2.0
        if self.normalize_output:
            x = (x - self.output_mean) / self.output_std
        h_out, w_out = h, w
        if output_size is not None:
            h_out = int(output_size[0])
            w_out = int(output_size[1])
            if x.shape[-2:] != (h_out, w_out):
                x = F.interpolate(x, size=(h_out, w_out), mode="bilinear", align_corners=False)
        return x
