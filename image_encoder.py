"""
图像JSCC编码器和解码器

基于SwinJSCC的图像编码器，采用Swin Transformer架构。
包含PatchEmbed、SwinTransformerBlock、PatchMerging等关键模块。

内存优化版本：
- 使用梯度检查点减少显存占用
- 优化窗口注意力计算
- 及时释放中间变量

【Phase 1优化】支持迁移学习：使用预训练的ImageNet权重（timm库）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List,Union
import math
from torch.utils.checkpoint import checkpoint
import timm
import logging
# 【Phase 1】尝试导入timm库用于预训练权重
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("警告: timm库未安装，将无法使用预训练权重。建议安装: pip install timm")

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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

class TextModulator(nn.Module):
    def __init__(self, img_dim: int, text_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, img_dim),
            nn.SiLU(),
            nn.Linear(img_dim, img_dim * 2) 
        )
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            self.mlp[-1].bias.zero_()
    def forward(self, x: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        params = self.mlp(text_emb)
        gamma, beta = params.chunk(2, dim=-1)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        elif x.dim() == 4:
            gamma = gamma.unsqueeze(-1).unsqueeze(-1)
            beta = beta.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + gamma) + beta
class SNRModulator(nn.Module):
    """
    SNR 自适应调制器 (FiLM: Feature-wise Linear Modulation)
    
    根据输入的 SNR 值，动态生成缩放(scale)和偏移(shift)系数，
    对特征图进行逐通道调制。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # 使用 MLP 将标量 SNR 映射为通道调制参数
        self.mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dim * 2)  # 输出 scale 和 shift
        )
        
        # 初始化：使初始状态下的调制接近恒等变换 (scale=0, shift=0)
        # 这样加载预训练权重时不会破坏原有特征
        with torch.no_grad():
            self.mlp[-1].weight.data.zero_()
            self.mlp[-1].bias.data.zero_()

    def forward(self, x: torch.Tensor, snr_db: Union[float, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            x: 输入特征 [B, L, C] 或 [B, C, H, W]
            snr_db: 信噪比 (dB)
        """
        B = x.shape[0]
        
        # 统一处理 SNR 输入格式
        if isinstance(snr_db, (float, int)):
            snr = torch.tensor([snr_db], dtype=x.dtype, device=x.device)
            snr = snr.expand(B, 1)
        elif isinstance(snr_db, torch.Tensor):
            if snr_db.dim() == 0:
                snr = snr_db.view(1, 1).expand(B, 1)
            elif snr_db.dim() == 1:
                snr = snr_db.view(B, 1)
            else:
                snr = snr_db
        else:
            raise ValueError("Unsupported snr_db type")

        # 为了数值稳定性，将 SNR 归一化到一个较小范围 (例如除以 10)
        # 这是一个常用的工程技巧
        params = self.mlp(snr / 10.0)  # [B, 2*dim]
        
        # 分离 scale 和 shift
        scale, shift = params.chunk(2, dim=-1)  # [B, dim], [B, dim]
        
        # 调整形状以支持广播
        if x.dim() == 3:  # [B, L, C] (Swin Block 输出格式)
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        elif x.dim() == 4:  # [B, C, H, W] (Conv 输出格式)
            scale = scale.unsqueeze(-1).unsqueeze(-1)
            shift = shift.unsqueeze(-1).unsqueeze(-1)
            
        # 应用 FiLM 调制: y = (1 + scale) * x + shift
        return x * (1.0 + scale) + shift
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    将特征图分割成窗口（内存优化版本）
    
    Args:
        x (torch.Tensor): 输入特征 [B, H, W, C]
        window_size (int): 窗口大小
        
    Returns:
        torch.Tensor: 分割后的窗口 [B*num_windows, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    # 优化：使用更节省内存的方式，避免创建大的中间张量
    # 先reshape，然后使用更高效的permute方式
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    x = x.view(B, num_windows_h, window_size, num_windows_w, window_size, C)
    # 使用更节省内存的permute：先permute再contiguous，减少中间内存占用
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """
    将窗口重新组合成特征图
    
    Args:
        windows (torch.Tensor): 窗口特征 [B*num_windows, window_size, window_size, C]
        window_size (int): 窗口大小
        H, W (int): 特征图高度和宽度
        
    Returns:
        torch.Tensor: 重组后的特征图 [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    图像块嵌入层
    
    将图像分割成patches并嵌入到特征空间。
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 96,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None
    
    def forward(
        self,
        x: torch.Tensor,
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像 [B, C, H, W]
            
        Returns:
            torch.Tensor: 嵌入特征 [B, num_patches, embed_dim]
        """
        B, C, H, W = x.shape
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        H_pad, W_pad = H + pad_h, W + pad_w
        self.last_input_size = (H, W)
        self.last_pad = (pad_h, pad_w)
        self.last_padded_size = (H_pad, W_pad)
        self.grid_size = (H_pad // self.patch_size, W_pad // self.patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x


class PatchMerging(nn.Module):
    """
    图像块合并层
    
    将相邻的patches合并，实现下采样。
    """
    
    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        out_dim: Optional[int] = None,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or 2 * dim
        
        self.reduction = nn.Linear(4 * dim, self.out_dim, bias=False)
        self.norm = norm_layer(4 * dim) if norm_layer else None
    
    def forward(
        self,
        x: torch.Tensor,
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [B, H*W, C]
            
        Returns:
            torch.Tensor: 合并后的特征 [B, H*W/4, out_dim]
        """
        H, W = input_resolution or self.input_resolution
        B, L, C = x.shape
        if L != H * W:
            raise RuntimeError("输入特征长度与分辨率不匹配")
        
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        H_pad, W_pad = H + pad_h, W + pad_w
        x = x.permute(0, 2, 3, 1)
        
        # 合并2x2的patches
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H*W/4, 4*C]
        
        if self.norm is not None:
            x = self.norm(x)
        
        x = self.reduction(x)
        return x, (H_pad // 2, W_pad // 2)


class PatchReverseMerging(nn.Module):
    """
    图像块反向合并层
    
    将合并的patches分离，实现上采样。
    """
    
    def __init__(
        self,
        input_resolution: Tuple[int, int],
        dim: int,
        out_dim: Optional[int] = None,
        norm_layer: Optional[nn.Module] = None
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.out_dim = out_dim or dim // 2
        self.norm = norm_layer(dim) if norm_layer else None
        self.up_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=dim,
                out_channels=self.out_dim * 4,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.PixelShuffle(2)
        )
        #self.expansion = nn.Linear(dim, 4 * self.out_dim, bias=False)
        #self.norm = norm_layer(dim) if norm_layer else None
    
    def forward(
        self,
        x: torch.Tensor,
        input_resolution: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [B, H*W, C]
            
        Returns:
            torch.Tensor: 分离后的特征 [B, H*W*4, out_dim]
        """
        H, W = input_resolution or self.input_resolution
        B, L, C = x.shape
        
        if self.norm is not None:
            x = self.norm(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.up_conv(x)
        x = x.permute(0, 2, 3, 1).view(B, -1, self.out_dim)
        #x = self.expansion(x)  # [B, H*W, 4*out_dim]
        #x = x.view(B, H, W, 2, 2, self.out_dim)
        #x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        
        # 分离成4个部分
        #x0 = x[:, :, :, 0, :]  # [B, H, W, out_dim]
        #x1 = x[:, :, :, 1, :]  # [B, H, W, out_dim]
        #x2 = x[:, :, :, 2, :]  # [B, H, W, out_dim]
        #x3 = x[:, :, :, 3, :]  # [B, H, W, out_dim]
        
        # 重新排列成2x2的网格
       # x = torch.zeros(B, H * 2, W * 2, self.out_dim, device=x.device, dtype=x.dtype)
        #x[:, 0::2, 0::2, :] = x0
        #x[:, 1::2, 0::2, :] = x1
        #x[:, 0::2, 1::2, :] = x2
        #x[:, 1::2, 1::2, :] = x3
        
        #x = x.view(B, -1, self.out_dim)
        return x, (H * 2, W * 2)


class WindowAttention(nn.Module):
    """
    窗口注意力机制
    
    在窗口内进行自注意力计算，提高计算效率。
    """
    
    def __init__(
        self,
        dim: int,
        window_size: Tuple[int, int],
        num_heads: int,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        
        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        
        # 计算相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        window_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        前向传播（内存优化版本）
        
        Args:
            x (torch.Tensor): 输入特征 [B*num_windows, window_size*window_size, C]
            mask (torch.Tensor, optional): 注意力掩码
            
        Returns:
            torch.Tensor: 注意力输出
        """
        B_, N, C = x.shape
        target_window = to_2tuple(window_size) if window_size is not None else self.window_size
        # 优化QKV计算，使用更节省内存的方式
        qkv = self.qkv(x)
        # 立即reshape和permute，避免大的中间张量
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, num_heads, N, head_dim]
        # 立即分离（使用unbind，比索引更节省内存）
        q, k, v = qkv.unbind(0)
        del qkv  # 立即释放大的qkv张量
        
        # 缩放Q（使用in-place操作节省内存）
        q = q * self.scale
        
        # 计算注意力分数（使用更节省显存的方式）
        attn = torch.matmul(q, k.transpose(-2, -1))
        
        # 添加相对位置偏置
        if target_window == self.window_size:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(N, N, -1)
        else:
            original_window = self.window_size
            relative_position_bias_table = self.relative_position_bias_table.view(
                2 * original_window[0] - 1,
                2 * original_window[1] - 1,
                -1,
            ).permute(2, 0, 1).unsqueeze(0)
            relative_position_bias_table = F.interpolate(
                relative_position_bias_table,
                size=(2 * target_window[0] - 1, 2 * target_window[1] - 1),
                mode="bicubic",
                align_corners=False,
            )
            relative_position_bias_table = (
                relative_position_bias_table.squeeze(0)
                .permute(1, 2, 0)
                .reshape(-1, self.num_heads)
            )
            coords_h = torch.arange(target_window[0], device=x.device)
            coords_w = torch.arange(target_window[1], device=x.device)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += target_window[0] - 1
            relative_coords[:, :, 1] += target_window[1] - 1
            relative_coords[:, :, 0] *= 2 * target_window[1] - 1
            relative_position_index = relative_coords.sum(-1)
            relative_position_bias = relative_position_bias_table[
                relative_position_index.view(-1)
            ].view(N, N, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        del relative_position_bias  # 及时释放
        
        if mask is not None:
            nW = mask.shape[0]
            # 优化：避免创建大的中间张量，直接对每个窗口应用掩码
            # 将attn重塑为 [B_//nW, nW, num_heads, N, N]
            attn_reshaped = attn.view(B_ // nW, nW, self.num_heads, N, N)
            # 将mask扩展为 [1, nW, 1, N, N] 以便广播
            mask_expanded = mask.unsqueeze(0).unsqueeze(2)  # [1, nW, 1, N, N]
            # 应用掩码（广播相加）
            attn_reshaped = attn_reshaped + mask_expanded
            # 重塑回原始形状
            attn = attn_reshaped.view(-1, self.num_heads, N, N)
            del attn_reshaped, mask_expanded  # 及时释放
        
        # Softmax和dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        
        # 应用注意力权重（内存优化：先释放q和k，只保留attn和v）
        del q, k  # 在计算attn*v之前释放q和k，减少内存占用
        # 使用einsum可能更节省内存，但matmul通常更快，这里使用matmul
        x = torch.matmul(attn, v)
        del attn, v  # 及时释放
        x = x.transpose(1, 2).reshape(B_, N, C)
        
        # 输出投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer块
    
    包含窗口注意力和前馈网络，支持移位窗口机制。
    """
    
    def __init__(
        self,
        dim: int,
        input_resolution: Tuple[int, int],
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,  # 修复OOM：从4.0减小到2.0以节省显存
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size必须在0到window_size之间"
        
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # 修复OOM：减小MLP比例从4.0到2.0，减少hidden_features以节省显存
        #mlp_ratio = min(mlp_ratio, 2.0)  # 限制最大mlp_ratio为2.0
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
        self.register_buffer("attn_mask", None, persistent=False)

    def _build_attn_mask(
        self,
        H: int,
        W: int,
        window_size: int,
        shift_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if shift_size <= 0:
            return None
        img_mask = torch.zeros((1, H, W, 1), device=device, dtype=dtype)
        h_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        w_slices = (
            slice(0, -window_size),
            slice(-window_size, -shift_size),
            slice(-shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        mask_windows = window_partition(img_mask, window_size)
        mask_windows = mask_windows.view(-1, window_size * window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(
        self,
        x: torch.Tensor,
        input_resolution: Optional[Tuple[int, int]] = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        """
        前向传播（内存优化版本）
        
        Args:
            x (torch.Tensor): 输入特征 [B, H*W, C]
            use_checkpoint (bool): 是否使用梯度检查点
            
        Returns:
            torch.Tensor: 输出特征
        """
        H, W = input_resolution or self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征长度与分辨率不匹配"

        window_size = min(self.window_size, H, W)
        shift_size = self.shift_size if min(H, W) > window_size else 0
        if shift_size >= window_size:
            shift_size = 0

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        pad_b = (window_size - H % window_size) % window_size
        pad_r = (window_size - W % window_size) % window_size
        if pad_b or pad_r:
            x = x.permute(0, 3, 1, 2)
            x = F.pad(x, (0, pad_r, 0, pad_b))
            x = x.permute(0, 2, 3, 1)
        H_pad, W_pad = H + pad_b, W + pad_r
        attn_mask = self._build_attn_mask(
            H_pad, W_pad, window_size, shift_size, x.device, x.dtype
        )
        
        # 循环移位
        if shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-shift_size, -shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 窗口分割（内存优化：及时释放中间变量）
        x_windows = window_partition(shifted_x, window_size)
        x_windows = x_windows.view(-1, window_size * window_size, C)
        del shifted_x  # 及时释放，减少内存占用
        
        # 窗口注意力（使用梯度检查点以降低显存峰值）
        if use_checkpoint and self.training:
            # 使用梯度检查点：在反向传播时重新计算，以节省显存
            # 创建包装函数以支持关键字参数
            def attn_with_mask(x):
                return self.attn(x, mask=attn_mask, window_size=(window_size, window_size))
            attn_windows = checkpoint(attn_with_mask, x_windows, use_reentrant=False)
        else:
            attn_windows = self.attn(x_windows, mask=attn_mask, window_size=(window_size, window_size))
        del x_windows  # 及时释放
        
        attn_windows = attn_windows.view(-1, window_size, window_size, C)
        
        # 窗口重组
        shifted_x = window_reverse(attn_windows, window_size, H_pad, W_pad)
        del attn_windows  # 及时释放
        
        # 反向循环移位
        if shift_size > 0:
            x = torch.roll(shifted_x, shifts=(shift_size, shift_size), dims=(1, 2))
        else:
            x = shifted_x
        del shifted_x  # 及时释放
        if pad_b or pad_r:
            x = x[:, :H, :W, :]
        x = x.reshape(B, H * W, C)
        
        # 前馈网络（使用梯度检查点以降低显存峰值）
        x = shortcut + self.drop_path(x)
        x_norm = self.norm2(x)
        if use_checkpoint and self.training:
            # 使用梯度检查点：在反向传播时重新计算，以节省显存
            mlp_out = checkpoint(self.mlp, x_norm, use_reentrant=False)
        else:
            mlp_out = self.mlp(x_norm)
        del x_norm  # 及时释放
        x = x + self.drop_path(mlp_out)
        del mlp_out  # 及时释放
        
        return x


class Mlp(nn.Module):
    """多层感知机"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播（极端内存优化版本）
        
        分步计算，及时释放中间变量，减少内存占用
        使用更激进的内存管理策略
        """
        # MLP前向传播
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        
        return x


class DropPath(nn.Module):
    """随机深度"""
    
    def __init__(self, drop_prob: float = None):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def to_2tuple(x):
    """转换为2元组"""
    if isinstance(x, (list, tuple)):
        return x
    return (x, x)


class BasicLayer(nn.Module):
    """
    基础层
    
    包含多个Swin Transformer块和可选的patch merging。
    """
    
    def __init__(
        self,
        dim: int,
        out_dim: Optional[int],
        input_resolution: Tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,  # 修复OOM：从4.0减小到2.0以节省显存
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        downsample: Optional[nn.Module] = None
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        
        # 构建Swin Transformer块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim, input_resolution=input_resolution,
                num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])
        
        # Patch merging层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
            # 计算下采样后的分辨率（PatchMerging 会将分辨率减半）
            self.output_resolution = (input_resolution[0] // 2, input_resolution[1] // 2)
        else:
            self.downsample = None
            self.output_resolution = input_resolution
    
    def forward(
        self,
        x: torch.Tensor,
        input_resolution: Optional[Tuple[int, int]] = None,
        use_checkpoint: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入特征 [B, H*W, C]
            use_checkpoint (bool): 是否使用梯度检查点（训练时启用可节省显存）
            
        Returns:
            torch.Tensor: 输出特征
        """
        resolution = input_resolution or self.input_resolution
        # 先经过本层的 Swin Transformer 块
        for i, blk in enumerate(self.blocks):
            x = blk(x, input_resolution=resolution, use_checkpoint=use_checkpoint)

        # 再进行下采样/上采样（若配置）
        if self.downsample is not None:
            x, resolution = self.downsample(x, input_resolution=resolution)
        
        return x, resolution


class ImageJSCCEncoder(nn.Module):
    """
    图像JSCC编码器（Phase 1优化：支持迁移学习）
    
    基于Swin Transformer的图像编码器，将图像编码为适合信道传输的连续值特征。
    
    【Phase 1优化】支持加载预训练的ImageNet权重（通过timm库），大幅加速训练。
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 4,
        mlp_ratio: float = 4.0,  # 修复OOM：从4.0减小到2.0以节省显存
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        output_dim: int = 256,
        patch_embed: Optional[nn.Module] = None,
        swin_layers: Optional[nn.ModuleList] = None,
        swin_norm: Optional[nn.Module] = None,
        pretrained: bool = False,  # 【Phase 1】是否使用预训练权重
        disable_timm_pretrained: bool = True,  # 策略A：默认禁用timm预训练路径
        freeze_encoder: bool = False,  # 【Phase 1】是否冻结编码器主干（仅训练适配器）
        pretrained_model_name: str = 'swin_tiny_patch4_window7_224',  # 【Phase 1】预训练模型名称
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.patches_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        if pretrained and disable_timm_pretrained:
            logger.warning(
                "已选择禁用timm预训练路径（策略A），将使用仓库内部动态Swin实现。"
            )
            pretrained = False
        self.pretrained = pretrained and TIMM_AVAILABLE
        self.freeze_encoder = freeze_encoder
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.snr_modulators = nn.ModuleList()
        if hasattr(self, 'layers'):
            for i in range(len(self.layers)):
                dim = embed_dims[i]
                self.snr_modulators.append(SNRModulator(dim))
        
        # 【Phase 1】如果使用预训练权重，尝试从timm加载
        if self.pretrained:
            try:
                # 使用timm创建预训练的Swin Transformer模型
                self.pretrained_backbone = timm.create_model(
                    pretrained_model_name,
                    pretrained=True,
                    img_size=img_size,
                    num_classes=0,  # 移除分类头
                    global_pool=''  # 不使用全局池化
                )
                
                # 提取特征提取器（patch_embed + layers + norm）
                if hasattr(self.pretrained_backbone, 'patch_embed'):
                    self.patch_embed = self.pretrained_backbone.patch_embed
                else:
                    # 如果没有patch_embed，创建新的
                    self.patch_embed = PatchEmbed(
                        img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                        embed_dim=embed_dims[0], norm_layer=norm_layer
                    )
                
                if hasattr(self.pretrained_backbone, 'layers'):
                    self.layers = self.pretrained_backbone.layers
                else:
                    self.layers = nn.ModuleList()
                    for i in range(self.num_layers):
                        layer = BasicLayer(
                            dim=embed_dims[i],
                            out_dim=embed_dims[i + 1] if (i < self.num_layers - 1) else None,
                            input_resolution=self.patches_resolution,
                            depth=depths[i],
                            num_heads=num_heads[i],
                            window_size=window_size,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias, qk_scale=qk_scale,
                            drop=drop_rate, attn_drop=attn_drop_rate,
                            drop_path=drop_path_rate * (i / (self.num_layers - 1)),
                            norm_layer=norm_layer,
                            downsample=PatchMerging if (i < self.num_layers - 1) else None
                        )
                        self.layers.append(layer)
                
                if hasattr(self.pretrained_backbone, 'norm'):
                    self.norm = self.pretrained_backbone.norm
                else:
                    self.norm = norm_layer(embed_dims[-1])
                
                # 获取最终维度
                if hasattr(self.norm, 'normalized_shape'):
                    final_dim = self.norm.normalized_shape[0] if isinstance(self.norm.normalized_shape, (list, tuple)) else self.norm.normalized_shape
                else:
                    final_dim = embed_dims[-1]
                
                # 【Phase 1】如果freeze_encoder=True，冻结主干网络
                if self.freeze_encoder:
                    for param in self.patch_embed.parameters():
                        param.requires_grad = False
                    for layer in self.layers:
                        for param in layer.parameters():
                            param.requires_grad = False
                    for param in self.norm.parameters():
                        param.requires_grad = False
                    print("【Phase 1】已冻结预训练编码器主干，仅训练适配器层")
                
                print(f"【Phase 1】成功加载预训练模型: {pretrained_model_name}")
                
            except Exception as e:
                #print(f"【Phase 1】警告: 加载预训练权重失败: {e}，将使用随机初始化")
                if pretrained:
                    raise RuntimeError(
                        f"【致命错误】无法加载预训练权重 '{pretrained_model_name}'。\n"
                        f"原因: {e}\n"
                        f"请检查网络连接、timm版本或模型名称。\n"
                        f"如果想从头训练，请在 config.py 中设置 pretrained=False。"
                    )
                else:
                    print(f"加载失败，使用随机初始化: {e}")
                    self.pretrained = False
                #self.pretrained = False
                # 回退到原始实现
                self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                    embed_dim=embed_dims[0], norm_layer=norm_layer
                )
                self.layers = nn.ModuleList()
                current_resolution = self.patches_resolution
                for i_layer in range(self.num_layers):
                    layer = BasicLayer(
                        dim=embed_dims[i_layer],
                        out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                        input_resolution=current_resolution,
                        depth=depths[i_layer],
                        num_heads=num_heads[i_layer],
                        window_size=window_size,
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                        drop=drop_rate, attn_drop=attn_drop_rate,
                        drop_path=drop_path_rate * (i_layer / (self.num_layers - 1)),
                        norm_layer=norm_layer,
                        downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
                    )

                    self.layers.append(layer)
                    if i_layer < self.num_layers - 1:
                        current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)
                self.norm = norm_layer(embed_dims[-1])
                final_dim = embed_dims[-1]
        elif (patch_embed is not None) and (swin_layers is not None) and (swin_norm is not None):
            # 使用共享主干
            self.patch_embed = patch_embed
            self.layers = swin_layers
            self.norm = swin_norm
            # 推断最终维度
            if hasattr(self.norm, 'normalized_shape'):
                final_dim = self.norm.normalized_shape[0] if isinstance(self.norm.normalized_shape, (list, tuple)) else self.norm.normalized_shape
            else:
                # 退化处理：从patch_embed推断
                final_dim = getattr(self.patch_embed, 'proj', nn.Conv2d(in_chans, embed_dims[0], kernel_size=patch_size, stride=patch_size)).out_channels
        else:
            # Patch embedding（非共享路径）
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dims[0], norm_layer=norm_layer
            )
            
            # 构建编码器层（非共享路径）
            self.layers = nn.ModuleList()
            # 跟踪当前分辨率（从 patch_embed 的输出分辨率开始）
            current_resolution = self.patches_resolution
            for i_layer in range(self.num_layers):
                layer = BasicLayer(
                    dim=embed_dims[i_layer],
                    out_dim=embed_dims[i_layer + 1] if (i_layer < self.num_layers - 1) else None,
                    input_resolution=current_resolution,  # 使用当前分辨率，而不是基于 i_layer 计算
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=drop_path_rate * (i_layer / (self.num_layers - 1)),
                    norm_layer=norm_layer,
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
                )
                self.layers.append(layer)
                # 更新当前分辨率为下一层的输入分辨率（如果当前层有下采样，则分辨率减半）
                if i_layer < self.num_layers - 1:
                    current_resolution = (current_resolution[0] // 2, current_resolution[1] // 2)
            # 非共享路径的最终维度
            self.norm = norm_layer(embed_dims[-1])
            final_dim = embed_dims[-1]

        # 输出投影（依据最终维度构建）
        self.output_proj = nn.Sequential(
            nn.Linear(final_dim, max(final_dim // 2, 1)),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(max(final_dim // 2, 1), output_dim)
        )
        
        # 引导向量提取器（依据最终维度构建）
        self.guide_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(final_dim, max(final_dim // 4, 1))
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
 
    
    def forward(self, x: torch.Tensor, snr_db: Union[float, torch.Tensor] = 10.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x (torch.Tensor): 输入图像 [B, C, H, W]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (编码特征, 引导向量)
        """
        # Patch embedding
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        if x.dim() == 4:
            B, H, W, C = x.shape
            L = H * W
        else:
            B, L, C = x.shape
        patch_resolution = getattr(self.patch_embed, "grid_size", self.patches_resolution)
        self.last_input_size = getattr(self.patch_embed, "last_input_size", None)
        self.last_pad = getattr(self.patch_embed, "last_pad", (0, 0))
        self.last_patch_resolution = patch_resolution
        
        # 通过编码器层
        # 注意：移除 try-except 以暴露真正的错误，便于调试
        resolution = patch_resolution
        for i, layer in enumerate(self.layers):
            if self.pretrained:
                x = layer(x)
            else:
                use_checkpoint = self.use_gradient_checkpointing and self.training
                x, resolution = layer(x, input_resolution=resolution, use_checkpoint=use_checkpoint)
            if i < len(self.snr_modulators):
                x = self.snr_modulators[i](x, snr_db)
        x = self.norm(x)
        if self.pretrained:
            resolution = (
                patch_resolution[0] // (2 ** (self.num_layers - 1)),
                patch_resolution[1] // (2 ** (self.num_layers - 1)),
            )
        self.last_latent_resolution = resolution
        if x.dim() == 4:
            x = x.flatten(1, 2)
        guide_vector = self.guide_extractor(x.transpose(1, 2)).squeeze(-1)
        encoded_features = self.output_proj(x)
        assert encoded_features.shape[-1] == self.output_proj[-1].out_features, (
            f"ImageJSCCEncoder.output_proj 维度不匹配: got {encoded_features.shape[-1]},"
            f" expected {self.output_proj[-1].out_features}"
        )
        _log_nonfinite("ImageJSCCEncoder.encoded_features", encoded_features)
        _log_nonfinite("ImageJSCCEncoder.guide_vector", guide_vector)
        return encoded_features, guide_vector

            # 验证输入维度是否与 layer 的 input_resolution 匹配
            #B, L, C = x.shape
            #H, W = layer.input_resolution
            #expected_L = H * W
            #if L != expected_L:
                #raise RuntimeError(
                    #f"ImageJSCCEncoder 第 {i} 层输入维度不匹配："
                    #f"实际序列长度 L={L}，期望 L={H}*{W}={expected_L} "
                    #f"(input_resolution={layer.input_resolution})。"
                    #f"当前 x.shape={tuple(x.shape)}"
                #)
            #if self.pretrained:
                #x = layer(x)
            #else:
                #x = layer(x, use_checkpoint=self.training)  # 训练时自动启用梯度检查点
        
        # 归一化
        #x = self.norm(x)
        
        # 提取引导向量（全局平均池化）
        #guide_vector = self.guide_extractor(x.transpose(1, 2)).squeeze(-1)
        
        # 输出投影
        #encoded_features = self.output_proj(x)
        # 断言：输出特征最后一维应等于配置的 output_dim
        #assert encoded_features.shape[-1] == self.output_proj[-1].out_features, (
            #f"ImageJSCCEncoder.output_proj 维度不匹配: got {encoded_features.shape[-1]},"
            #f" expected {self.output_proj[-1].out_features}"
        #)
        
        #return encoded_features, guide_vector


class ImageJSCCDecoder(nn.Module):
    """
    图像JSCC解码器
    
    从经过信道传输的带噪特征重建原始图像。
    """
    
    def __init__(
        self,
        img_size: Tuple[int, int] = (224, 224),
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [2, 2, 6, 2],
        num_heads: List[int] = [3, 6, 12, 24],
        window_size: int = 4,
        mlp_ratio: float = 4.0,  # 修复OOM：从4.0减小到2.0以节省显存
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        input_dim: int = 256,
        guide_dim: Optional[int] = None,
        semantic_context_dim: int = 256,  # 【修复】添加参数，与 VideoJSCCDecoder 保持一致
        normalize_output: bool = False,
        use_gradient_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.img_size = img_size  # 保存img_size用于output_proj计算
        self.patches_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.normalize_output = normalize_output
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.register_buffer("output_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("output_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dims[-1] // 2, embed_dims[-1])
        )
        
        # 构建解码器层
        self.layers = nn.ModuleList()
        self.snr_modulators = nn.ModuleList()
        self.text_modulators = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dims[-(i_layer + 1)],
                
                out_dim=embed_dims[-(i_layer + 2)] if (i_layer < self.num_layers - 1) else None,
                input_resolution=(
                    self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                    self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))
                ),
                depth=depths[-(i_layer + 1)],
                num_heads=num_heads[-(i_layer + 1)],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate * (i_layer / (self.num_layers - 1)),
                norm_layer=norm_layer,
                downsample=PatchReverseMerging if (i_layer < self.num_layers - 1) else None
            )
            current_dim = embed_dims[-(i_layer + 1)]
            self.snr_modulators.append(SNRModulator(current_dim))
            self.text_modulators.append(TextModulator(current_dim, semantic_context_dim))
            self.layers.append(layer)
        
        # 输出层
        self.norm = norm_layer(embed_dims[0])
        
        # 计算 output_padding 以确保精确输出尺寸
        # ConvTranspose2d 输出尺寸公式：
        # output_size = (input_size - 1) × stride + kernel_size - 2 × padding + output_padding
        def calculate_output_padding(input_size: int, output_size: int, kernel_size: int, stride: int, padding: int = 0) -> int:
            """计算 ConvTranspose2d 所需的 output_padding"""
            calculated = (input_size - 1) * stride + kernel_size - 2 * padding
            if calculated == output_size:
                return 0
            elif calculated < output_size:
                return output_size - calculated
            else:
                raise ValueError(
                    f"Cannot achieve output_size={output_size} with current parameters. "
                    f"Calculated size: {calculated}, input_size: {input_size}, "
                    f"kernel_size: {kernel_size}, stride: {stride}, padding: {padding}"
                )
        
        # 计算高度和宽度的 output_padding
        H_in = self.patches_resolution[0]  # 例如 56
        W_in = self.patches_resolution[1]  # 例如 56
        H_out = img_size[0]  # 例如 224
        W_out = img_size[1]  # 例如 224
        
        output_padding_h = calculate_output_padding(H_in, H_out, patch_size, patch_size)
        output_padding_w = calculate_output_padding(W_in, W_out, patch_size, patch_size)
        
        # 使用 ConvTranspose2d 进行上采样，而不是 Conv2d 的下采样
        # 【优化】改进输出层以提高对比度和重建质量
        self.output_proj = nn.Sequential(
            nn.Conv2d(
                embed_dims[0],  # 输入通道数：96
                in_chans * (patch_size ** 2),       # 输出通道数：3
                kernel_size=3,  # 4
                stride=1,       # 4
                padding=1,
            ),
            # 【优化】使用Tanh替代Sigmoid，然后缩放到[0,1]，提高对比度
            # Tanh输出范围[-1,1]，通过 (tanh + 1) / 2 映射到[0,1]，但对比度更好
            nn.PixelShuffle(patch_size),
            nn.Tanh()
        )
        
        # 引导向量处理器
        # guide_processor 的输入维度需要与编码器给出的 guide 向量维度一致
        in_guide_dim = guide_dim if guide_dim is not None else (embed_dims[-1] // 4)
        self.guide_processor = nn.Sequential(
            nn.Linear(in_guide_dim, embed_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(embed_dims[-1] // 2, embed_dims[-1])
        )
        
        # 【重构】统一语义对齐和融合层（一次性融合策略）
        # 语义上下文（文本编码）的维度通过参数传入，与 VideoJSCCDecoder 保持一致
        # 统一将语义上下文对齐到图像主干网的维度 (embed_dims[-1])
        self.semantic_context_dim = semantic_context_dim
        self.semantic_aligner = nn.Linear(semantic_context_dim, embed_dims[-1])
        
        # 导入 CrossAttention
        from cross_attention import CrossAttention
        
        # 【修复】添加统一的语义融合模块，传入text_embed_dim以确保正确投影
        self.semantic_fusion = CrossAttention(
            embed_dim=embed_dims[-1],
            num_heads=num_heads[-1],  # 使用最后一层的头数
            dropout=drop_rate,
            text_embed_dim=None  # 【关键修复】传入文本嵌入维度，确保正确投影
        )
        
        # 【优化】添加可学习的语义融合权重（门控机制），不增加显存
        # 使用轻量级门控网络控制语义引导的强度
        self.semantic_gate = nn.Sequential(
            nn.Linear(embed_dims[-1] * 2, embed_dims[-1] // 4),  # 轻量级
            nn.ReLU(),
            nn.Linear(embed_dims[-1] // 4, 1),
            nn.Sigmoid()  # 输出0-1之间的权重
        )
        
        # 【优化】添加可学习的引导向量融合权重
        self.guide_gate = nn.Parameter(torch.ones(1) * 0.5)  # 初始权重0.5
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        """权重初始化"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(
        self,
        noisy_features: torch.Tensor,
        guide_vector: torch.Tensor,
        semantic_context: Optional[torch.Tensor] = None,
        multiple_semantic_contexts: Optional[List[torch.Tensor]] = None,
        snr_db: Union[float, torch.Tensor] = 10.0,
        input_resolution: Optional[Tuple[int, int]] = None,
        output_resolution: Optional[Tuple[int, int]] = None,
        output_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            noisy_features (torch.Tensor): 带噪特征 [B, num_patches, input_dim]
            guide_vector (torch.Tensor): 引导向量 [B, guide_dim]
            semantic_context (torch.Tensor, optional): 语义上下文 [B, seq_len, D_text]
            multiple_semantic_contexts (List[torch.Tensor], optional): 多条语义上下文列表
            
        Returns:
            torch.Tensor: 重建图像 [B, C, H, W]
        """
        if semantic_context is not None:
            text_global = semantic_context.mean(dim=1)
            actual_dim = semantic_context.shape[-1]
            expected_dim = self.semantic_aligner.in_features
            assert actual_dim == expected_dim, (
                f"维度不匹配错误: 传入的语义上下文维度为 {actual_dim},"
                f"但解码器期望的维度 (semantic_context_dim) 为 {expected_dim}。"
                f"请检查 config.py 中的 text_output_dim 是否与模型训练时一致。"
            )
        # 1. 投影噪声特征
        x = self.input_proj(noisy_features)  # [B, L, C]
        # 断言：输入投影后最后一维应为 embed_dims[-1]
        assert x.shape[-1] == self.embed_dims[-1], (
            f"ImageJSCCDecoder.input_proj 维度不匹配: got {x.shape[-1]},"
            f" expected {self.embed_dims[-1]}"
        )
        
        # 2. 投影自身引导向量 (来自Image Encoder)
        # 断言：原始引导向量最后一维应为配置的 in_guide_dim
        in_guide_dim = self.guide_processor[0].in_features
        assert guide_vector.shape[-1] == in_guide_dim, (
            f"ImageJSCCDecoder.guide_vector 维度不匹配: got {guide_vector.shape[-1]},"
            f" expected {in_guide_dim}"
        )
        guide_processed = self.guide_processor(guide_vector)  # [B, C]
        # 断言：处理后的引导向量最后一维应为 embed_dims[-1]
        assert guide_processed.shape[-1] == self.embed_dims[-1], (
            f"ImageJSCCDecoder.guide_processor 输出维度不匹配: got {guide_processed.shape[-1]},"
            f" expected {self.embed_dims[-1]}"
        )
        batch_size, seq_len = x.size(0), x.size(1)
        guide_expanded = guide_processed.unsqueeze(1).expand(batch_size, seq_len, -1)  # [B, L, C]
        
        # 3. 【新逻辑】投影并融合语义上下文 (来自Text Encoder)
        # 【优化】支持多条语义上下文，通过平均或加权融合
        if semantic_context is not None:
            # 处理单条语义上下文
            semantic_contexts_to_process = [semantic_context]
        else:
            semantic_contexts_to_process = []
        
        # 【新增】支持多条语义上下文
        if multiple_semantic_contexts is not None and len(multiple_semantic_contexts) > 0:
            semantic_contexts_to_process.extend(multiple_semantic_contexts)
        
        if len(semantic_contexts_to_process) > 0:
            # 处理所有语义上下文
            semantic_fused_list = []
            for ctx in semantic_contexts_to_process:
                # 使用统一的语义对齐层
                aligned_semantic = self.semantic_aligner(ctx)  # [B_semantic, T_text, C]
                
                if aligned_semantic.shape[0] != batch_size:
                    raise RuntimeError(
                        f"semantic_context batch mismatch in ImageJSCCDecoder: semantic_batch={aligned_semantic.shape[0]}, expected={batch_size}"
                    )
                
                # 融合 (图像特征为Q, 文本为K/V)
                semantic_fused = self.semantic_fusion(query=x, guide_vector=aligned_semantic)
                semantic_fused_list.append(semantic_fused)
            
            # 【优化】多条语义时，使用平均融合（简单有效，不增加计算）
            if len(semantic_fused_list) > 1:
                semantic_fused = torch.stack(semantic_fused_list, dim=0).mean(dim=0)  # 平均融合
            else:
                semantic_fused = semantic_fused_list[0]
            
            # 【优化】使用门控机制动态控制语义引导强度
            gate_input = torch.cat([x, semantic_fused], dim=-1)  # [B, L, 2*C]
            semantic_weight = self.semantic_gate(gate_input)  # [B, L, 1]
            
            # 加权融合：特征 + 自身引导（可学习权重） + 语义引导（门控权重）
            guide_weight = torch.clamp(self.guide_gate, 0.0, 1.0)  # 限制在[0,1]
            x = x + guide_weight * guide_expanded + semantic_weight * semantic_fused
        else:
            # 仅融合自身引导
            x = x + guide_expanded
        if input_resolution is None:
            input_resolution = (
                self.patches_resolution[0] // (2 ** (self.num_layers - 1)),
                self.patches_resolution[1] // (2 ** (self.num_layers - 1)),
            )
        if output_resolution is None:
            output_resolution = self.patches_resolution
        resolution = input_resolution
        for i, layer in enumerate(self.layers):
            if i < len(self.snr_modulators):
                x = self.snr_modulators[i](x, snr_db)
            if semantic_context is not None and i < len(self.text_modulators):
                x = self.text_modulators[i](x, text_global)
            use_checkpoint = self.use_gradient_checkpointing and self.training
            x, resolution = layer(x, input_resolution=resolution, use_checkpoint=use_checkpoint)
        # 4. 将融合后的 x 送入Swin解码器 (无需再注入)
        #for layer in self.layers:
            #x = layer(x, use_checkpoint=self.training)
        
        # 归一化
        x = self.norm(x)
        
        # 重塑为图像格式
        # 注意：经过解码器层后，特征序列长度应该等于输出分辨率
        B, L, C = x.shape
        inferred_resolution = None
        if input_resolution is not None:
            inferred_resolution = (
                input_resolution[0] * (2 ** (self.num_layers - 1)),
                input_resolution[1] * (2 ** (self.num_layers - 1)),
            )
        if output_resolution is None:
            output_resolution = inferred_resolution or self.patches_resolution
        H, W = output_resolution

        if L != H * W:
            resolved = None
            side = int(math.sqrt(L))
            if side * side == L:
                resolved = (side, side)

            def _resolve_from_ratio(ratio: Optional[float]) -> Optional[Tuple[int, int]]:
                if ratio is None or ratio <= 0:
                    return None
                cand_h = int(round(math.sqrt(L * ratio)))
                if cand_h <= 0:
                    return None
                if L % cand_h != 0:
                    return None
                cand_w = L // cand_h
                if cand_w <= 0:
                    return None
                if abs((cand_h / cand_w) - ratio) > 0.01:
                    return None
                return cand_h, cand_w

            if resolved is None and inferred_resolution:
                if inferred_resolution[0] * inferred_resolution[1] == L:
                    resolved = inferred_resolution
                else:
                    resolved = _resolve_from_ratio(inferred_resolution[0] / inferred_resolution[1])
            if resolved is None:
                resolved = _resolve_from_ratio(H / W if W else None)
            if resolved is None:
                resolved = _resolve_from_ratio(self.patches_resolution[0] / self.patches_resolution[1])
            if resolved is None:
                raise RuntimeError(
                    f"ImageJSCCDecoder重塑失败：序列长度 L={L} 不等于预期的 H*W={H}*{W}={H*W}。"
                    f"请检查上采样配置是否正确，patches_resolution={self.patches_resolution}。"
                )
            H, W = resolved
        
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        #for i, layer in enumerate(self.layers):
            #if i < len(self.snr_modulators):
                #x = self.snr_modulators[i](x, snr_db)
            #x = layer(x, use_checkpoint=self.training)

        
        # 输出投影
        x = self.output_proj(x)
        # Tanh -> [0, 1]
        x = (x + 1.0) / 2.0
        if self.normalize_output:
            x = (x - self.output_mean) / self.output_std
        if output_size is not None:
            x = x[..., : output_size[0], : output_size[1]]
        
        # 【优化】将Tanh输出[-1,1]映射到[0,1]，同时增强对比度
        # 使用 (tanh + 1) / 2 进行基本映射，然后应用轻微的对比度增强
        #x = (x + 1.0) / 2.0  # 映射到[0,1]
        # 轻微的对比度增强：x = x^0.9，保持范围在[0,1]但增强对比度
        #x = torch.clamp(x, 0.0, 1.0)  # 确保在[0,1]范围内
        
        _log_nonfinite("ImageJSCCDecoder.output", x)
        return x
