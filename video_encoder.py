"""
视频JSCC编码器和解码器

基于MDVSC的视频语义通信架构：GOP级别潜空间变换、共性/个性特征提取、
熵模型重要性筛选、轻量JSCC编解码与潜空间逆变换。
保持接口兼容（类名/方法签名/guide_vectors），便于与多模态框架协作。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint_sequential
from typing import Optional, Tuple, List, Dict, Any
import math
import logging
from image_encoder import SNRModulator
from video_unet import ResBlock

logger = logging.getLogger(__name__)


def _log_nonfinite(name: str, tensor: torch.Tensor) -> bool:
    if not torch.is_tensor(tensor):
        return False
    finite_mask = torch.isfinite(tensor)
    if finite_mask.all():
        return False
    with torch.no_grad():
        finite_values = tensor[finite_mask]
        if finite_values.numel() > 0:
            t_min = finite_values.min().item()
            t_max = finite_values.max().item()
            t_mean = finite_values.mean().item()
        else:
            t_min = float("nan")
            t_max = float("nan")
            t_mean = float("nan")
    logger.warning(
        "%s has NaN/Inf values (finite stats min=%.6e max=%.6e mean=%.6e)",
        name,
        t_min,
        t_max,
        t_mean,
    )
    return True


class LatentTransformer(nn.Module):
    """
    GOP级别潜空间变换模块

    将输入视频帧压缩到潜空间以便后续JSCC编码。
    使用轻量卷积+残差块进行空间降采样和特征压缩。
    """

    def __init__(
        self,
        in_channels: int,
        latent_dim: int,
        num_res_blocks: int = 2,
        downsample_factor: int = 2,
        downsample_stride: Optional[int] = None,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if downsample_stride is not None:
            downsample_factor = downsample_stride
        if downsample_factor < 1 or downsample_factor & (downsample_factor - 1) != 0:
            raise ValueError(
                f"downsample_factor must be a power of 2, got {downsample_factor}"
            )
        self.downsample_factor = downsample_factor
        self.use_gradient_checkpointing = use_gradient_checkpointing
        downsample_layers = []
        num_downsample = int(math.log2(downsample_factor)) if downsample_factor > 1 else 0
        if num_downsample == 0:
            downsample_layers.append(
                nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1)
            )
            downsample_layers.append(nn.ReLU(inplace=False))
        else:
            downsample_layers.append(
                nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=2, padding=1)
            )
            downsample_layers.append(nn.ReLU(inplace=False))
            for _ in range(num_downsample - 1):
                downsample_layers.append(
                    nn.Conv2d(latent_dim, latent_dim, kernel_size=3, stride=2, padding=1)
                )
                downsample_layers.append(nn.ReLU(inplace=False))
        self.stem = nn.Sequential(*downsample_layers)
        self.res_blocks = nn.Sequential(
            *[ResBlock(latent_dim, latent_dim) for _ in range(num_res_blocks)]
        )

    def forward(self, video_frames: torch.Tensor) -> torch.Tensor:
        """
        将视频帧映射到潜空间

        Args:
            video_frames (torch.Tensor): 输入视频 [B, T, C, H, W]

        Returns:
            torch.Tensor: 潜空间特征 [B, T, C_latent, H', W']
        """
        B, T, C, H, W = video_frames.shape
        x = video_frames.reshape(B * T, C, H, W)
        x = self.stem(x)
        if self.use_gradient_checkpointing and self.training:
             x = checkpoint_sequential(self.res_blocks, len(self.res_blocks), x, use_reentrant=False)
        else:
            x = self.res_blocks(x)
        _, C_latent, H_latent, W_latent = x.shape
        x = x.view(B, T, C_latent, H_latent, W_latent)
        return x


class JSCCEncoder(nn.Module):
    """
    深度JSCC编码器（MDVSC版本）

    使用卷积+残差块提取鲁棒特征，输出用于信道传输的特征图。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_res_blocks: int = 3,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.body = nn.Sequential(
            *[ResBlock(hidden_dim, hidden_dim) for _ in range(num_res_blocks)]
        )
        self.tail = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint_sequential(self.body, len(self.body), x, use_reentrant=False)
        else:
            x = self.body(x)
        x = self.tail(x)
        return x


class JSCCDecoder(nn.Module):
    """
    深度JSCC解码器（MDVSC版本）

    从信道特征恢复潜空间特征，结构为轻量卷积+残差。
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        out_channels: int,
        num_res_blocks: int = 3,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.body = nn.Sequential(
            *[ResBlock(hidden_dim, hidden_dim) for _ in range(num_res_blocks)]
        )
        self.tail = nn.Conv2d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint_sequential(self.body, len(self.body), x, use_reentrant=False)
        else:
            x = self.body(x)
        x = self.tail(x)
        return x


class CommonFeatureExtractor(nn.Module):
    """
    共性/个性特征提取器（MDVSC）

    将T帧特征拼接后提取共性特征W_c，再得到每帧个性特征W_i = Y_i - W_c。
    """

    def __init__(self, channels: int, num_frames: int):
        super().__init__()
        self.num_frames = num_frames
        self.common_conv = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.residual_proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        提取共性与个性特征

        Args:
            features (torch.Tensor): 输入特征 [B, T, C, H, W]

        Returns:
            Tuple: (W_c, W_i) -> [B, C, H, W], [B, T, C, H, W]
        """
        B, T, C, H, W = features.shape
        concat_features = features.view(B, T * C, H, W)
        residual = self.residual_proj(features.mean(dim=1))
        common = self.common_conv(concat_features) + residual
        individual = features - common.unsqueeze(1)
        return common, individual


class EntropyModel(nn.Module):
    """
    熵模型：估计特征重要性并生成掩码

    输入共性特征W_c和个性特征W_i，输出熵/重要性评分。
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 128,
        temperature: float = 0.2,
        max_exact_quantile_elems: int = 2_000_000,
        quantile_sample_size: int = 262_144,
    ):
        super().__init__()
        self.temperature = temperature
        self.max_exact_quantile_elems = max_exact_quantile_elems
        self.quantile_sample_size = quantile_sample_size
        self.scorer = nn.Sequential(
            nn.Conv2d(channels * 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, channels, kernel_size=1),
        )

    def forward(self, common: torch.Tensor, individual: torch.Tensor) -> torch.Tensor:
        """
        估计熵分数

        Args:
            common (torch.Tensor): 共性特征 [B, C, H, W]
            individual (torch.Tensor): 个性特征 [B, T, C, H, W] 或 [B, C, H, W]

        Returns:
            torch.Tensor: 熵分数 [B, T, C, H, W] 或 [B, C, H, W]
        """
        if individual.dim() == 4:
            concat = torch.cat([common, individual], dim=1)
            scores = self.scorer(concat)
            return scores

        B, T, C, H, W = individual.shape
        common_rep = common.unsqueeze(1).expand(-1, T, -1, -1, -1)
        concat = torch.cat([common_rep, individual], dim=2)
        concat = concat.view(B * T, 2 * C, H, W)
        scores = self.scorer(concat)
        scores = scores.view(B, T, C, H, W)
        return scores

    def build_mask(self, scores: torch.Tensor, target_cbr: float, training: bool) -> torch.Tensor:
        """
        根据熵分数构建掩码

        Args:
            scores (torch.Tensor): 熵分数
            target_cbr (float): 目标码率比例（保留比例）
            training (bool): 是否训练模式

        Returns:
            torch.Tensor: 掩码（与scores同形状）
        """
        if target_cbr >= 1.0:
            return torch.ones_like(scores)
        if target_cbr <= 0.0:
            return torch.zeros_like(scores)

        B = scores.shape[0]
        flat = scores.view(B, -1)
        q = 1.0 - target_cbr
        use_sampled_quantile = (
            flat.is_cuda and flat.numel() > self.max_exact_quantile_elems
        )
        if use_sampled_quantile:
            numel_per_batch = flat.size(1)
            sample_size = min(self.quantile_sample_size, numel_per_batch)
            idx = torch.randint(
                0,
                numel_per_batch,
                (B, sample_size),
                device=flat.device,
                dtype=torch.long,
            )
            sampled = flat.gather(1, idx)
            quantile = torch.quantile(sampled.float(), q, dim=1)
        else:
            quantile = torch.quantile(flat.float(), q, dim=1)
        threshold_shape = [B] + [1] * (scores.dim() - 1)
        threshold = quantile.view(*threshold_shape).to(scores.dtype)
        hard_mask = (scores >= threshold).float()

        if not training:
            return hard_mask

        soft_mask = torch.sigmoid((scores - threshold) / self.temperature)
        # Straight-through估计：前向使用硬掩码，反向使用软掩码梯度
        mask = hard_mask + soft_mask - soft_mask.detach()
        return mask


class LatentInversion(nn.Module):
    """
    潜空间逆变换模块

    将潜空间特征恢复到像素域。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        upsample_factor: int = 2,
        upsample_stride: Optional[int] = None,
        normalize_output: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if upsample_stride is not None:
            upsample_factor = upsample_stride
        if upsample_factor < 1 or upsample_factor & (upsample_factor - 1) != 0:
            raise ValueError(f"upsample_factor must be a power of 2, got {upsample_factor}")
        self.upsample_factor = upsample_factor
        self.normalize_output = normalize_output
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
        )
        self.body = nn.Sequential(
            *[ResBlock(hidden_dim, hidden_dim) for _ in range(num_res_blocks)]
        )
        num_upsample = int(math.log2(upsample_factor)) if upsample_factor > 1 else 0
        if num_upsample == 0:
            self.upsample = nn.Identity()
        else:
            upsample_layers = []
            for _ in range(num_upsample):
                upsample_layers.append(
                    nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=2, stride=2)
                )
                upsample_layers.append(nn.ReLU(inplace=False))
            self.upsample = nn.Sequential(*upsample_layers)
        self.tail = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )
        self.register_buffer("output_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("output_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.head(x)
        if self.use_gradient_checkpointing and self.training:
            x = checkpoint_sequential(self.body, len(self.body), x, use_reentrant=False)
        else:
            x = self.body(x)
        x = self.upsample(x)
        x = self.tail(x)
        x = (x + 1.0) / 2.0
        if self.normalize_output:
            x = (x - self.output_mean) / self.output_std
        return x

class LightweightTemporalConv(nn.Module):
    """
    轻量级时序卷积模块（替代ConvLSTM）
    
    使用简单的时序卷积进行时序建模，显存占用远小于ConvLSTM。
    通过残差连接和门控机制保持时序建模能力。
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        kernel_size: int = 3,
        num_layers: int = 1,
        batch_first: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # 轻量级时序卷积层
        layers = []
        for i in range(num_layers):
            in_ch = input_dim if i == 0 else hidden_dim
            layers.append(nn.Sequential(
                nn.Conv2d(in_ch, hidden_dim, kernel_size=kernel_size, padding=kernel_size//2),
                nn.GroupNorm(8, hidden_dim),  # 使用GroupNorm替代BN，更节省显存
                nn.ReLU(inplace=False)
            ))
        self.layers = nn.ModuleList(layers)
        
        # 门控机制（轻量级）
        self.gate = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        input_tensor: torch.Tensor,
        hidden_state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            input_tensor (torch.Tensor): 输入张量 [B, T, C, H, W] 或 [B, 1, C, H, W]
            hidden_state (torch.Tensor, optional): 前一时刻的隐藏状态 [B, C, H, W]
            
        Returns:
            Tuple: (输出张量, 新的隐藏状态)
        """
        if not self.batch_first and input_tensor.dim() == 5:
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)
        
        if input_tensor.dim() == 5:
            # [B, T, C, H, W] -> 取最后一帧
            input_tensor = input_tensor[:, -1]  # [B, C, H, W]
        
        # 如果提供了隐藏状态，进行融合
        if hidden_state is not None:
            # 门控融合
            combined = torch.cat([input_tensor, hidden_state], dim=1)
            gate_weight = self.gate(combined)
            input_tensor = input_tensor * gate_weight + hidden_state * (1 - gate_weight)
        
        # 通过时序卷积层
        x = input_tensor
        for layer in self.layers:
            x = layer(x)
        
        # 残差连接
        if self.input_dim == self.hidden_dim:
            x = x + input_tensor
        
        # 返回输出和新的隐藏状态
        return x, x


class ConvLSTMCell(nn.Module):
    """ConvLSTM 单元"""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        padding = kernel_size // 2
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

    def forward(
        self,
        input_tensor: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_cur, c_cur = state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        conv_output = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.chunk(conv_output, 4, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    """单层 ConvLSTM"""

    def __init__(self, input_dim: int, hidden_dim: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias)

    def forward(
        self,
        input_tensor: torch.Tensor,
        state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if input_tensor.dim() != 4:
            raise ValueError(
                f"ConvLSTM expects [B, C, H, W] input, got {tuple(input_tensor.shape)}"
            )
        batch_size, _, height, width = input_tensor.shape
        if state is None:
            h = torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device)
            c = torch.zeros(batch_size, self.hidden_dim, height, width, device=input_tensor.device)
            state = (h, c)
        h_next, c_next = self.cell(input_tensor, state)
        return h_next, (h_next, c_next)







class LightweightOpticalFlow(nn.Module):
    """
    轻量级光流估计模块（简化版）
    
    使用更小的网络和简化的架构，大幅减少显存占用。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 32,  # 减小隐藏维度
        num_layers: int = 2  # 减少层数
    ):
        super().__init__()
        self.in_channels = in_channels
        
        # 简化的光流估计网络（2层）
        self.flow_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, kernel_size=3, padding=1, stride=2),  # 下采样减少计算
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 上采样回原尺寸
        )
    
    def forward(
        self,
        frame1: torch.Tensor,
        frame2: torch.Tensor
    ) -> torch.Tensor:
        """
        估计光流（轻量级版本）
        
        Args:
            frame1 (torch.Tensor): 参考帧 [B, C, H, W]
            frame2 (torch.Tensor): 目标帧 [B, C, H, W]
            
        Returns:
            torch.Tensor: 光流 [B, 2, H, W]
        """
        # 拼接两帧
        input_tensor = torch.cat([frame1, frame2], dim=1)
        
        # 估计光流
        flow = self.flow_net(input_tensor)
        
        return flow


# 保持兼容性
class OpticalFlow(LightweightOpticalFlow):
    """兼容性包装"""
    pass


class MotionCompensation(nn.Module):
    """
    运动补偿模块
    
    基于光流进行运动补偿，生成预测帧。
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        reference_frame: torch.Tensor,
        flow: torch.Tensor
    ) -> torch.Tensor:
        """
        运动补偿
        
        Args:
            reference_frame (torch.Tensor): 参考帧 [B, C, H, W]
            flow (torch.Tensor): 光流 [B, 2, H, W]
            
        Returns:
            torch.Tensor: 补偿后的帧 [B, C, H, W]
        """
        B, C, H, W = reference_frame.shape
        
        # 创建网格
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=reference_frame.device),
            torch.arange(W, device=reference_frame.device),
            indexing='ij'
        )
        grid = torch.stack([grid_x, grid_y], dim=0).float()
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 应用光流
        flow_grid = grid + flow
        
        # 归一化到[-1, 1]
        flow_grid[:, 0, :, :] = 2.0 * flow_grid[:, 0, :, :] / max(W - 1, 1) - 1.0
        flow_grid[:, 1, :, :] = 2.0 * flow_grid[:, 1, :, :] / max(H - 1, 1) - 1.0
        
        # 重排列为grid_sample格式
        flow_grid = flow_grid.permute(0, 2, 3, 1)
        
        # 双线性插值
        compensated_frame = F.grid_sample(
            reference_frame, flow_grid, mode='bilinear', padding_mode='border', align_corners=True
        )
        
        return compensated_frame


class ContextualEncoder(nn.Module):
    """
    上下文编码器
    
    基于DCVC的上下文编码器，移除量化模块，输出连续值特征。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        
        # 特征提取层
        layers = []
        in_ch = in_channels
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else hidden_dim
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False)
            ])
            if i < num_layers - 1:
                layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            in_ch = out_ch
        
        self.encoder = nn.Sequential(*layers)
        
        # 输出投影
        self.output_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        上下文编码
        
        Args:
            x (torch.Tensor): 输入帧 [B, C, H, W]
            
        Returns:
            torch.Tensor: 编码特征 [B, hidden_dim, H', W']
        """
        features = self.encoder(x)
        features = self.output_proj(features)
        return features


class ContextualDecoder(nn.Module):
    """
    上下文解码器
    
    基于DCVC的上下文解码器，从连续值特征重建帧。
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,
        hidden_dim: int = 256,
        num_layers: int = 4
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.input_proj = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        # 解码器层
        layers = []
        in_ch = hidden_dim
        for i in range(num_layers):
            out_ch = hidden_dim if i < num_layers - 1 else out_channels
            layers.extend([
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU(inplace=False) if i < num_layers - 1 else nn.Sigmoid()
            ])
            if i < num_layers - 1:
                layers.append(nn.ConvTranspose2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1))
            in_ch = out_ch
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        上下文解码
        
        Args:
            x (torch.Tensor): 输入特征 [B, in_channels, H, W]
            
        Returns:
            torch.Tensor: 重建帧 [B, out_channels, H', W']
        """
        x = self.input_proj(x)
        x = self.decoder(x)
        return x


class VideoJSCCEncoder(nn.Module):
    """
    视频JSCC编码器
    
    基于MDVSC架构：先做GOP潜空间变换，再进行JSCC编码，
    之后提取共性/个性特征并用熵模型进行重要性筛选。
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 256,
        num_frames: int = 5,
        use_optical_flow: bool = True,
        use_convlstm: bool = True,
        output_dim: int = 256,
        gop_size: Optional[int] = None,
        latent_downsample_factor: int = 2,
        latent_downsample_stride: Optional[int] = None,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        patch_embed: Optional[nn.Module] = None,
        swin_layers: Optional[nn.ModuleList] = None,
        swin_norm: Optional[nn.Module] = None,
        mlp_ratio: float = 4.0,
        use_gradient_checkpointing: bool = True,
        target_cbr: float = 0.5,
        entropy_temperature: float = 0.2,
        entropy_max_exact_quantile_elems: int = 2_000_000,
        entropy_quantile_sample_size: int = 262_144,
    ):
        super().__init__()
        if latent_downsample_stride is not None:
            latent_downsample_factor = latent_downsample_stride
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        # 兼容参数保留（MDVSC路径不再使用光流/ConvLSTM）
        self.use_optical_flow = use_optical_flow
        self.use_convlstm = use_convlstm
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.target_cbr = target_cbr
        self.gop_size = gop_size or num_frames
        self.latent_downsample_factor = latent_downsample_factor

        # GOP级别潜空间变换（替代Swin）
        self.latent_transformer = LatentTransformer(
            in_channels=in_channels,
            latent_dim=hidden_dim,
            num_res_blocks=2,
            downsample_factor=latent_downsample_factor,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # JSCC深度编码器
        self.jscc_encoder = JSCCEncoder(
            in_channels=hidden_dim,
            hidden_dim=hidden_dim,
            out_channels=output_dim,
            num_res_blocks=3,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        self.snr_modulator = SNRModulator(output_dim)

        # 共性/个性特征提取（按GOP长度构建）
        self.common_extractor = CommonFeatureExtractor(output_dim, self.gop_size)

        # 熵模型估计特征重要性
        self.entropy_model = EntropyModel(
            channels=output_dim,
            hidden_dim=max(64, output_dim // 2),
            temperature=entropy_temperature,
            max_exact_quantile_elems=entropy_max_exact_quantile_elems,
            quantile_sample_size=entropy_quantile_sample_size,
        )

        # 引导向量提取器（使用个性特征W_i）
        self.guide_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(output_dim, output_dim // 4),
        )

        # 记录码率信息
        self.last_rate_stats: Dict[str, torch.Tensor] = {}
    
    def _encode_gop(
        self, video_frames: torch.Tensor, snr_db: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """编码单个GOP，返回共性/个性特征与掩码均值。"""
        B, T, C, H, W = video_frames.shape
        # 1) GOP潜空间变换
        latent_features = self.latent_transformer(video_frames)  # [B, T, C_latent, H', W']

        # 2) JSCC编码（逐帧）
        latent_flat = latent_features.view(
            B * T,
            latent_features.size(2),
            latent_features.size(3),
            latent_features.size(4),
        )
        jscc_features = self.jscc_encoder(latent_flat)
        jscc_features = self.snr_modulator(jscc_features, snr_db)
        jscc_features = jscc_features.view(
            B,
            T,
            jscc_features.size(1),
            jscc_features.size(2),
            jscc_features.size(3),
        )

        # 3) 提取共性/个性特征
        common_feature, individual_features = self.common_extractor(jscc_features)

        # 4) 熵模型筛选特征
        entropy_scores = self.entropy_model(common_feature, individual_features)
        entropy_mask = self.entropy_model.build_mask(
            entropy_scores, target_cbr=self.target_cbr, training=self.training
        )
        common_feature = common_feature * entropy_mask.mean(dim=1)
        individual_features = individual_features * entropy_mask

        return common_feature, individual_features, entropy_mask.mean()

    def forward(
        self,
        video_frames: torch.Tensor,
        reset_state: bool = False,
        snr_db: float = 10.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        视频编码器前向传播（MDVSC）
        
        Args:
            video_frames (torch.Tensor): 视频帧 [B, T, C, H, W]
            reset_state (bool): 是否重置隐藏状态
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (编码特征, 引导向量)
        """
        B, T, C, H, W = video_frames.shape
        self.last_input_size = (H, W)

        if T % self.gop_size != 0:
            raise ValueError(
                f"video_frames长度T={T}不能被gop_size={self.gop_size}整除。"
                "请调整video_clip_len或video_gop_size以对齐。"
            )

        if T == self.gop_size:
            common_feature, individual_features, mask_mean = self._encode_gop(video_frames, snr_db)
        else:
            common_list = []
            individual_list = []
            mask_means = []
            for start in range(0, T, self.gop_size):
                end = start + self.gop_size
                gop_frames = video_frames[:, start:end]
                common_feature, individual_features, mask_mean = self._encode_gop(gop_frames, snr_db)
                common_list.append(common_feature)
                individual_list.append(individual_features)
                mask_means.append(mask_mean)
            common_feature = torch.stack(common_list, dim=1).mean(dim=1)
            individual_features = torch.cat(individual_list, dim=1)
            mask_mean = torch.stack(mask_means).mean()

        # 5) 引导向量来自个性特征
        guide_vectors = self.guide_extractor(
            individual_features.view(B * T, individual_features.size(2), individual_features.size(3), individual_features.size(4))
        ).view(B, T, -1)

        # 6) 输出：将W_c作为额外帧拼接在前
        encoded_features = torch.cat([common_feature.unsqueeze(1), individual_features], dim=1)

        # 码率统计（用于loss）
        self.last_rate_stats = {
            "video_mask_mean": mask_mean,
        }

        _log_nonfinite("VideoJSCCEncoder.encoded_features", encoded_features)
        _log_nonfinite("VideoJSCCEncoder.guide_vectors", guide_vectors)

        return encoded_features, guide_vectors
    
    def reset_hidden_state(self):
        """重置隐藏状态"""
        return None


class VideoJSCCDecoder(nn.Module):
    """
    视频JSCC解码器
    
    MDVSC接收端：组合共性/个性特征恢复Y_i，
    通过轻量JSCC解码与潜空间逆变换重建视频帧。
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 3,
        hidden_dim: int = 256,
        num_frames: int = 5,
        use_optical_flow: bool = True,
        use_convlstm: bool = True,
        input_dim: int = 256,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        semantic_context_dim: int = 256,
        mlp_ratio: float = 4.0,  # 添加语义上下文维度参数
        latent_upsample_factor: int = 2,
        normalize_output: bool = False,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_dim = hidden_dim
        self.num_frames = num_frames
        # 兼容参数保留（MDVSC路径不再使用光流/ConvLSTM）
        self.use_optical_flow = use_optical_flow
        self.use_convlstm = use_convlstm
        self.img_size = img_size
        self.patch_size = patch_size
        self.semantic_context_dim = semantic_context_dim
        self.normalize_output = normalize_output
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.register_buffer("output_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("output_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        # JSCC解码器（MDVSC）
        self.jscc_decoder = JSCCDecoder(
            in_channels=input_dim,
            hidden_dim=hidden_dim,
            out_channels=hidden_dim,
            num_res_blocks=3,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )

        # 潜空间逆变换（恢复像素）
        self.latent_inversion = LatentInversion(
            in_channels=hidden_dim,
            out_channels=out_channels,
            hidden_dim=hidden_dim // 2,
            num_res_blocks=2,
            upsample_factor=latent_upsample_factor,
            normalize_output=normalize_output,
            use_gradient_checkpointing=use_gradient_checkpointing,
        )
        
        # 引导向量处理器
        self.guide_processor = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # 语义对齐层：在 __init__ 中预定义，而不是在 forward 中动态创建
        # 用于将语义上下文（文本编码）的维度对齐到视频特征的维度
        # 文本编码维度：semantic_context_dim (例如 256)
        # 视频特征维度：hidden_dim (例如 256)
        self.semantic_aligner = nn.Linear(semantic_context_dim, hidden_dim)
        
        # 【修复】使用标准CrossAttention模块替代简化的自定义注意力实现
        # 导入并使用标准CrossAttention模块（与ImageJSCCDecoder保持一致）
        from cross_attention import CrossAttention
        # CrossAttention 需要 embed_dim 参数，这里使用 hidden_dim（对齐后的维度）
        self.cross_attention = CrossAttention(
            embed_dim=hidden_dim,
            num_heads=max(1, hidden_dim // 64),  # 自适应头数：256//64=4
            dropout=0.0  # 可以使用dropout，这里设为0保持与原有实现一致
        )
        
        # 记录最近的语义注意力统计
        self.last_semantic_gate_stats: Dict[str, Optional[float]] = {"mean": None, "std": None}
    
    def forward(
        self,
        noisy_features: torch.Tensor,
        guide_vectors: torch.Tensor,
        reset_state: bool = False,
        semantic_context: Optional[torch.Tensor] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        视频解码器前向传播（MDVSC接收端）
        
        Args:
            noisy_features (torch.Tensor): 带噪特征 [B, T(+1), C, H, W]
            guide_vectors (torch.Tensor): 引导向量 [B, T, guide_dim]
            reset_state (bool): 是否重置隐藏状态
            semantic_context (torch.Tensor, optional): 语义上下文 [B, seq_len, D_text]
            
        Returns:
            torch.Tensor: 重建视频 [B, T, C, H, W]
        """
        B, T, C, H, W = noisy_features.shape

        # 1) 分离共性/个性特征
        if T == self.num_frames + 1:
            common_feature = noisy_features[:, 0]
            individual_features = noisy_features[:, 1:]
        else:
            # 兼容旧接口：若没有W_c，视为全个性特征
            common_feature = torch.zeros_like(noisy_features[:, 0])
            individual_features = noisy_features

        T_individual = individual_features.size(1)

        # 2) 还原每帧特征 Y_i = W_c + W_i
        reconstructed_features = individual_features + common_feature.unsqueeze(1)

        # 3) 融合引导向量与语义上下文
        enhanced_features = []
        for t in range(T_individual):
            current_feature = reconstructed_features[:, t]
            current_guide = guide_vectors[:, t]
            guide_processed = self.guide_processor(current_guide)
            guide_expanded = guide_processed.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W)
            current_feature = current_feature + guide_expanded
            if semantic_context is not None:
                current_feature = self._apply_semantic_guidance(current_feature, semantic_context)
            enhanced_features.append(current_feature)

        enhanced_features = torch.stack(enhanced_features, dim=1)
        enhanced_flat = enhanced_features.reshape(
            B * T_individual,
            enhanced_features.size(2),
            enhanced_features.size(3),
            enhanced_features.size(4),
        )

        # 4) JSCC解码 -> 潜空间
        decoded_latent = self.jscc_decoder(enhanced_flat)
        decoded_latent = decoded_latent.view(
            B,
            T_individual,
            decoded_latent.size(1),
            decoded_latent.size(2),
            decoded_latent.size(3),
        )

        # 5) 潜空间逆变换 -> 像素
        decoded_frames = self.latent_inversion(
            decoded_latent.view(
                B * T_individual,
                decoded_latent.size(2),
                decoded_latent.size(3),
                decoded_latent.size(4),
            )
        )
        decoded_frames = decoded_frames.view(
            B,
            T_individual,
            decoded_frames.size(1),
            decoded_frames.size(2),
            decoded_frames.size(3),
        )

        if output_size is not None:
            decoded_frames = decoded_frames[..., : output_size[0], : output_size[1]]

        return decoded_frames
    
    def _apply_semantic_guidance(
        self, 
        video_features: torch.Tensor, 
        semantic_context: torch.Tensor
    ) -> torch.Tensor:
        """
        【修复】应用语义引导的交叉注意力 - 使用标准CrossAttention模块
        
        Args:
            video_features (torch.Tensor): 视频特征 [B, C, H, W]
            semantic_context (torch.Tensor): 语义上下文 [B, seq_len, D_text]
            
        Returns:
            torch.Tensor: 增强后的视频特征
        """
        B, C, H, W = video_features.shape
        
        # 将视频特征重塑为序列格式
        video_seq = video_features.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        
        # 维度对齐
        # 注意：semantic_aligner 已在 __init__ 中定义，不应在 forward 中动态创建
        # 如果维度不匹配，使用预定义的语义对齐层
        if video_seq.shape[-1] != semantic_context.shape[-1]:
            # 验证维度是否与预定义的对齐层匹配
            if (semantic_context.shape[-1] != self.semantic_context_dim or 
                video_seq.shape[-1] != self.hidden_dim):
                raise RuntimeError(
                    f"语义对齐层维度不匹配："
                    f"semantic_context维度={semantic_context.shape[-1]}, 预期={self.semantic_context_dim}; "
                    f"video_seq维度={video_seq.shape[-1]}, 预期={self.hidden_dim}。"
                    f"请检查VideoJSCCDecoder的初始化参数。"
                )
            aligned_semantic = self.semantic_aligner(semantic_context)
        else:
            aligned_semantic = semantic_context
        
        if aligned_semantic.shape[0] != B:
            raise RuntimeError(
                f"semantic_context batch mismatch: semantic_batch={aligned_semantic.shape[0]}, expected={B}. "
                "请检查 DataLoader/Collate 的 batch 对齐。"
            )
        
        # 【修复】使用标准CrossAttention模块（与ImageJSCCDecoder保持一致）
        # CrossAttention 内部会处理 Query 和 Key/Value 序列长度不同的情况
        # 视频特征作为Query，语义上下文作为Key和Value
        enhanced_video_seq, attn_weights = self.cross_attention(
            query=video_seq,  # [B, H*W, hidden_dim]
            guide_vector=aligned_semantic,  # [B, seq_len, hidden_dim]
            return_attention=True
        )
        if attn_weights is not None:
            self.last_semantic_gate_stats["mean"] = float(attn_weights.mean().item())
            self.last_semantic_gate_stats["std"] = float(attn_weights.std().item())
        
        # 残差连接
        enhanced_video_seq = video_seq + enhanced_video_seq
        
        # 重塑回视频格式
        enhanced_video_features = enhanced_video_seq.transpose(1, 2).view(B, C, H, W)
        
        return enhanced_video_features
    
    def reset_hidden_state(self):
        """重置隐藏状态"""
        return None