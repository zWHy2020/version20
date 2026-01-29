"""
判别器模块（Phase 3: 对抗训练）

实现PatchGAN风格的判别器，用于对抗训练以提升重建质量。
参考SGDJSCC和图像修复/压缩领域的常见做法。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NLayerDiscriminator(nn.Module):
    """
    N层PatchGAN判别器
    
    使用PatchGAN架构,对图像的局部patch进行真假判断。
    这种设计比全图判别器更轻量，且能更好地处理纹理细节。
    
    Args:
        input_nc (int): 输入通道数(RGB为3)
        ndf (int): 判别器的基础通道数
        n_layers (int): 判别器层数（不包括最后一层）
        norm_layer: 归一化层类型
    """
    
    def __init__(
        self,
        input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        norm_layer: nn.Module = nn.BatchNorm2d
    ):
        super().__init__()
        self.input_nc = input_nc
        self.ndf = ndf
        self.n_layers = n_layers
        
        # 第一层（不使用归一化）
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 中间层
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # 限制最大通道数为8倍
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # 最后一层
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # 输出层（输出单个patch的真假概率）
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            input (torch.Tensor): 输入图像 [B, C, H, W]，范围[0, 1]
            
        Returns:
            torch.Tensor: 判别器输出 [B, 1, H', W']，每个位置表示该patch为真的概率
        """
        return self.model(input)


class MultimodalDiscriminator(nn.Module):
    """
    多模态判别器
    
    为图像和视频分别创建判别器，支持联合对抗训练。
    """
    
    def __init__(
        self,
        image_input_nc: int = 3,
        video_input_nc: int = 3,
        ndf: int = 64,
        n_layers: int = 3
    ):
        super().__init__()
        
        # 图像判别器
        self.image_discriminator = NLayerDiscriminator(
            input_nc=image_input_nc,
            ndf=ndf,
            n_layers=n_layers
        )
        
        # 视频判别器（对每一帧分别判别，然后聚合）
        self.video_discriminator = NLayerDiscriminator(
            input_nc=video_input_nc,
            ndf=ndf,
            n_layers=n_layers
        )
    
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        video: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            image (torch.Tensor, optional): 输入图像 [B, C, H, W]
            video (torch.Tensor, optional): 输入视频 [B, T, C, H, W]
            
        Returns:
            Tuple: (图像判别结果, 视频判别结果)
        """
        image_pred = None
        video_pred = None
        
        if image is not None:
            image_pred = self.image_discriminator(image)
        
        if video is not None:
            B, T, C, H, W = video.shape
            # 对每一帧分别判别
            video_reshaped = video.reshape(B * T, C, H, W)
            video_pred_frames = self.video_discriminator(video_reshaped)
            # 重塑回 [B, T, 1, H', W'] 然后平均
            _, _, H_pred, W_pred = video_pred_frames.shape
            video_pred_frames = video_pred_frames.view(B, T, 1, H_pred, W_pred)
            video_pred = video_pred_frames.mean(dim=1)  # [B, 1, H', W']
        
        return image_pred, video_pred

