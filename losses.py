"""
多模态JSCC损失函数 (优化版)

根据专业建议，将图像/视频损失从 MSE 优化为 L1 和 MS-SSIM 的组合，
以显著提升重建的感知质量。

【重构】添加LPIPS感知损失和文本对比损失（CLIP-based）以提升重建质量和文本引导效果。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math

# 【新增】尝试导入LPIPS，如果不可用则使用VGG特征损失
try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("警告: LPIPS库未安装，将使用VGG特征损失作为替代。建议安装: pip install lpips")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------
# MS-SSIM 损失函数的实现
# --------------------------------------------------------------------------

def _gaussian_window(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """创建一维高斯核"""
    coords = torch.arange(size, dtype=dtype, device=device)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0) # [1, 1, size]

def _ssim(
    img1: torch.Tensor, 
    img2: torch.Tensor, 
    window: torch.Tensor, 
    window_size: int, 
    channel: int, 
    data_range: float = 1.0, 
    size_average: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    计算 SSIM 的核心函数
    返回 (ssim_map, cs_map)
    """
    # 动态填充
    padding = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=padding, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padding, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padding, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padding, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padding, groups=channel) - mu1_mu2
    
    # SSIM 参数 (稳定值)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # 亮度 (L)
    L = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    # 对比度-结构 (CS)
    CS = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    
    ssim_map = L * CS
    
    if size_average:
        return ssim_map.mean(), CS.mean()
    else:
        return ssim_map.mean([1, 2, 3]), CS.mean([1, 2, 3])


class MSSSIM(nn.Module):
    """
    多尺度结构相似性 (MS-SSIM) 损失模块
    """
    def __init__(
        self, 
        window_size: int = 11, 
        channel: int = 3, 
        data_range: float = 1.0, 
        num_scales: int = 5
    ):
        super().__init__()
        self.window_size = window_size
        self.channel = channel
        self.data_range = data_range
        self.num_scales = num_scales
        
        # 权重来自原始 MS-SSIM 论文
        self.register_buffer(
            'weights',
            torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=torch.float32)
        )

        # 创建高斯核 (可复用)
        # 我们使用2D高斯核，通过1D核的外积创建
        gauss_1d = _gaussian_window(window_size, 1.5, torch.device('cpu'), torch.float32)
        # 将可能的 [1,1,size] / [1,size] / [size] 统一转换为一维向量 [size]
        if gauss_1d.ndim == 3:
            gauss_1d = gauss_1d.squeeze(0).squeeze(0)
        elif gauss_1d.ndim == 2:
            gauss_1d = gauss_1d.squeeze(0)
        # 外积得到二维高斯核 [size, size]
        window = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        
        # 扩展到 C 个通道
        # 使用 register_buffer 而不是 nn.Parameter，因为 window 不需要梯度
        # 并且 register_buffer 可以更好地处理设备/类型转换
        self.register_buffer(
            'window',
            window.expand(channel, 1, window_size, window_size).contiguous()
        )

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        计算 MS-SSIM 损失 (1 - MS-SSIM)
        
        Args:
            img1 (torch.Tensor): 预测图像 [B, C, H, W]
            img2 (torch.Tensor): 目标图像 [B, C, H, W]
            
        Returns:
            torch.Tensor: MS-SSIM 损失值 (标量)
        """
        # 获取与输入相同设备和类型的 window 和 weights
        # 使用临时变量而不是直接赋值，避免 Parameter 类型问题
        window = self.window.to(device=img1.device, dtype=img1.dtype)
        weights = self.weights.to(device=img1.device, dtype=img1.dtype)
        levels = self.num_scales
        msssim_vals = []
        cs_vals = []

        for i in range(levels):
            ssim_map, cs_map = _ssim(
                img1, img2, 
                window=window,  # 使用临时变量
                window_size=self.window_size, 
                channel=self.channel, 
                data_range=self.data_range,
                size_average=False
            )
            
            msssim_vals.append(ssim_map)
            cs_vals.append(cs_map)
            
            # 下采样
            if i < levels - 1:
                img1 = F.avg_pool2d(img1, kernel_size=2, stride=2)
                img2 = F.avg_pool2d(img2, kernel_size=2, stride=2)
        
        msssim_stack = torch.stack(msssim_vals)
        cs_stack = torch.stack(cs_vals)
        
        # 确保权重形状正确（使用临时变量）
        weights_expanded = weights.view(levels, 1).expand(-1, msssim_stack.shape[1])
        
        # 根据论文公式计算 MS-SSIM
        # (CS_1^w1 * CS_2^w2 * ... * L_N^wN)
        # 这里使用简化版，对所有SSIM值加权
        # 在PyTorch中，我们通常对log(ssim)加权求和，然后exp()
        # 或者直接对 ssim 值加权求和（作为近似）
        # 我们这里采用更标准的：(CS_1*...*CS_{N-1}) * L_N
        
        # (CS_1^w1 * ... * CS_{N-1}^w_{N-1}) * SSIM_N^w_N
        # 使用log-space计算以保持数值稳定性
        # 确保所有值在有效范围内
        cs_stack = torch.clamp(cs_stack, min=1e-6, max=1.0)
        msssim_stack = torch.clamp(msssim_stack, min=1e-6, max=1.0)
        
        # 使用 log-space 计算，避免数值下溢
        log_cs = torch.log(cs_stack[:levels-1] + 1e-8) * weights_expanded[:levels-1]
        log_ssim = torch.log(msssim_stack[levels-1:levels] + 1e-8) * weights_expanded[levels-1:levels]
        log_stack = torch.cat([log_cs, log_ssim], dim=0)
        
        # 在 log-space 中求和，然后 exp
        log_ms_ssim = torch.sum(log_stack, dim=0)
        ms_ssim = torch.exp(log_ms_ssim)
        
        # 确保结果在有效范围内
        ms_ssim = torch.clamp(ms_ssim, min=0.0, max=1.0)

        # 返回损失 (1 - MS-SSIM)
        # 我们对批次中的所有样本取平均值
        loss = 1.0 - ms_ssim.mean()
        
        # 最终检查，确保损失值有效
        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, device=img1.device)
        
        return loss


# --------------------------------------------------------------------------
# 模态损失函数
# --------------------------------------------------------------------------

class TextImageContrastiveLoss(nn.Module):
    """
    文本-图像对比损失
    
    计算重建图像嵌入与原始文本嵌入之间的余弦相似度损失。
    用于确保文本引导机制有效工作，修复Text Loss = 0的问题。
    
    【关键修复】确保计算的是重建图像的特征与文本嵌入的相似度，而不是原始图像。
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        # 使用简单的MLP将图像特征投影到文本嵌入空间
        # 注意：这里假设图像特征和文本嵌入的维度可能不同
        self.image_proj = None  # 将在forward中动态创建（如果需要）
    
    def forward(
        self,
        reconstructed_image_features: torch.Tensor,
        text_embeddings: torch.Tensor,
        original_text_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算文本-图像对比损失
        
        Args:
            reconstructed_image_features: 重建图像的特征 [B, C] 或 [B, L, C]
            text_embeddings: 文本嵌入 [B, T, D_text] 或 [B, D_text]
            original_text_embeddings: 原始文本嵌入（用于对比学习，可选）
            
        Returns:
            torch.Tensor: 对比损失值
        """
        # 处理图像特征：如果是序列，使用全局平均池化
        if reconstructed_image_features.dim() == 5:
            # [B, T, C, H, W] -> [B, C]
            img_features = reconstructed_image_features.mean(dim=(1, 3, 4))
        elif reconstructed_image_features.dim() == 4:
            # [B, C, H, W] -> [B, C]
            img_features = reconstructed_image_features.mean(dim=(2, 3))
        elif reconstructed_image_features.dim() == 3:
            # [B, L, C] -> [B, C]
            img_features = reconstructed_image_features.mean(dim=1)
        else:
            img_features = reconstructed_image_features  # [B, C]
        
        # 处理文本嵌入：如果是序列，使用全局平均池化
        if text_embeddings.dim() == 3:
            # [B, T, D_text] -> [B, D_text]
            text_features = text_embeddings.mean(dim=1)
        else:
            text_features = text_embeddings  # [B, D_text]
        
        # 确保维度匹配
        if img_features.shape[-1] != text_features.shape[-1]:
            # 动态创建投影层（如果尚未创建）
            if self.image_proj is None or self.image_proj.out_features != text_features.shape[-1]:
                self.image_proj = nn.Linear(
                    img_features.shape[-1], 
                    text_features.shape[-1]
                ).to(img_features.device)
            img_features = self.image_proj(img_features)
        
        # 归一化
        img_features = F.normalize(img_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        # 计算余弦相似度
        similarity = torch.matmul(img_features, text_features.t()) / self.temperature
        
        # 创建正样本标签（对角线元素）
        batch_size = img_features.shape[0]
        labels = torch.arange(batch_size, device=img_features.device)
        
        # 计算交叉熵损失（对比学习）
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class TextLoss(nn.Module):
    """
    文本损失（重构版）

    使用交叉熵损失 + 文本-图像对比损失，确保文本引导机制有效工作。
    """
    def __init__(self, ignore_index: int = 0, use_contrastive: bool = True, contrastive_weight: float = 0.1):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        
        if self.use_contrastive:
            self.contrastive_loss_fn = TextImageContrastiveLoss()
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reconstructed_image_features: Optional[torch.Tensor] = None,
        text_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        前向传播（重构版）
        
        Args:
            pred: 预测的文本logits [B, T, V]
            target: 目标文本 [B, T]
            mask: 注意力掩码（可选）
            reconstructed_image_features: 重建图像的特征（用于对比损失）
            text_embeddings: 文本嵌入（用于对比损失）
        """
        B, T, V = pred.shape
        pred_flat = pred.view(B * T, V)
        target_flat = target.view(B * T)
        mask_flat = mask.view(B * T) if mask is not None else None
        if mask_flat is not None:
            pad_positions = target_flat[mask_flat == 0]
            if pad_positions.numel() > 0:
                self.loss_fn.ignore_index = int(pad_positions[0].item())
        
        # 1. 交叉熵损失
        if mask_flat is not None:
            valid_positions = mask_flat == 1
            if valid_positions.any():
                loss_ce = self.loss_fn(pred_flat[valid_positions], target_flat[valid_positions])
            else:
                loss_ce = torch.tensor(0.0, device=pred.device)
        else:
            loss_ce = self.loss_fn(pred_flat, target_flat)
        
        # 2. 文本-图像对比损失（如果提供）
        loss_contrastive = torch.tensor(0.0, device=pred.device)
        if self.use_contrastive and reconstructed_image_features is not None and text_embeddings is not None:
            try:
                loss_contrastive = self.contrastive_loss_fn(
                    reconstructed_image_features,
                    text_embeddings
                )
                if torch.isnan(loss_contrastive) or torch.isinf(loss_contrastive):
                    loss_contrastive = torch.tensor(0.0, device=pred.device)
            except Exception as e:
                # 如果对比损失计算失败，跳过
                loss_contrastive = torch.tensor(0.0, device=pred.device)
        
        # 汇总总损失
        total_loss = loss_ce + self.contrastive_weight * loss_contrastive
        
        return total_loss, {
            'text_ce_loss': loss_ce.item(),
            'text_contrastive_loss': loss_contrastive.item() if isinstance(loss_contrastive, torch.Tensor) else 0.0
        }


class PerceptualLoss(nn.Module):
    """
    感知损失（LPIPS或VGG特征损失）
    
    使用LPIPS（如果可用）或VGG特征损失来提升重建的感知质量。
    """
    def __init__(self, use_lpips: bool = True):
        super().__init__()
        self.use_lpips = use_lpips and LPIPS_AVAILABLE
        
        if self.use_lpips:
            # 使用LPIPS（AlexNet backbone，轻量级）
            self.lpips_fn = lpips.LPIPS(net='alex', verbose=False)
            # 冻结LPIPS参数
            for param in self.lpips_fn.parameters():
                param.requires_grad = False
        else:
            # 使用VGG特征损失作为替代
            vgg = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
            )
            # 加载预训练权重（如果可用）
            try:
                # 尝试加载预训练VGG权重
                vgg16 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
                vgg[0].weight.data = vgg16.features[0].weight.data
                vgg[1].weight.data = vgg16.features[2].weight.data
                vgg[3].weight.data = vgg16.features[5].weight.data
                vgg[4].weight.data = vgg16.features[7].weight.data
            except:
                pass
            self.vgg = vgg
            # 冻结VGG参数
            for param in self.vgg.parameters():
                param.requires_grad = False
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算感知损失
        
        Args:
            pred (torch.Tensor): 预测图像 [B, C, H, W]，范围[0, 1]
            target (torch.Tensor): 目标图像 [B, C, H, W]，范围[0, 1]
            
        Returns:
            torch.Tensor: 感知损失值
        """
        if self.use_lpips:
            # LPIPS需要输入范围[-1, 1]
            pred_norm = pred * 2.0 - 1.0
            target_norm = target * 2.0 - 1.0
            loss = self.lpips_fn(pred_norm, target_norm).mean()
        else:
            # VGG特征损失
            pred_features = self.vgg(pred)
            target_features = self.vgg(target)
            loss = F.mse_loss(pred_features, target_features)
        
        return loss


class ImageLoss(nn.Module):
    """
    图像损失 (重构版)
    
    使用L1重建损失 + LPIPS感知损失，以提升重建的感知质量。
    """
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        msssim_weight: float = 0.5,
        perceptual_weight: float = 0.1,
        gradient_weight: float = 0.1,
        data_range: float = 1.0,
        normalize: bool = False,
    ):
        super().__init__()
        self.recon_weight = reconstruction_weight
        self.msssim_weight = msssim_weight
        self.percep_weight = perceptual_weight
        self.grad_weight = gradient_weight
        self.data_range = data_range
        self.normalize = normalize
        self.epsilon = 1e-6
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.msssim_loss_fn = MSSSIM(
            window_size=11,
            channel=3,
            data_range=data_range,
            num_scales=5
        )
        # 【新增】感知损失（LPIPS或VGG）
        if self.percep_weight > 0:
            self.percep_loss_fn = PerceptualLoss(use_lpips=LPIPS_AVAILABLE)
        else:
            self.percep_loss_fn = None
    
    #def _normalize_to_range(self, img: torch.Tensor) -> torch.Tensor:
        """
        将图像归一化到 [0, 1] 范围
        
        如果输入使用了 ImageNet 归一化（范围约在 [-2.5, 2.5]），
        需要先反归一化，然后归一化到 [0, 1]。
        """
        # 检测是否使用了 ImageNet 归一化（通过数据范围判断）
        #img_min = img.min().item()
        #img_max = img.max().item()
        
        # 如果数据范围明显超出 [0, 1]，可能是 ImageNet 归一化的结果
       # if img_min < -0.5 or img_max > 1.5:
            # ImageNet 归一化参数
           # imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype)
            #imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype)
            
            # 添加维度以便广播 [1, 3, 1, 1]
           # mean = imagenet_mean.view(1, 3, 1, 1)
            #std = imagenet_std.view(1, 3, 1, 1)
            
            # 反归一化: x = normalized * std + mean
            #img = img * std + mean
        
        # 确保在 [0, 1] 范围内
        #img = torch.clamp(img, 0.0, 1.0)
        
        #return img
        
    def _maybe_denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return tensor
        return tensor * self.imagenet_std + self.imagenet_mean

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        前向传播（重构版）
        
        计算L1重建损失 + 梯度损失 + LPIPS感知损失。
        """
        pred_f32 = self._maybe_denormalize(pred.float())
        target_f32 = self._maybe_denormalize(target.float())
        loss_dict = {}
        # 1. L1重建损失（使用smooth L1以提升稳定性）
        diff = pred_f32 - target_f32
        loss_recon = F.l1_loss(pred_f32, target_f32)
        loss_dict['image_recon_loss'] = loss_recon.item()
        if self.msssim_weight > 0:
            loss_msssim = self.msssim_loss_fn(pred_f32, target_f32)
        else:
            loss_msssim = torch.tensor(0.0, device=pred.device)
        loss_dict['image_msssim_loss'] = loss_msssim.item()
        # 2. 梯度损失（保留边缘信息）
        if self.grad_weight > 0:
            pred_h = pred_f32[..., :, 1:] - pred_f32[..., :, :-1]
            target_h = target_f32[..., :, 1:] - target_f32[..., :, :-1]
            loss_grad_h = torch.mean(torch.abs(pred_h - target_h))
            pred_v = pred_f32[..., 1:, :] - pred_f32[..., :-1, :]
            target_v = target_f32[..., 1:, :] - target_f32[..., :-1, :]
            loss_grad_v = torch.mean(torch.abs(pred_v - target_v))
            loss_grad = loss_grad_h + loss_grad_v
        else:
            loss_grad = torch.tensor(0.0, device=pred.device)
        loss_dict['image_grad_loss'] = loss_grad.item()  
        
        # 3. 感知损失（LPIPS或VGG）
        loss_percep = torch.tensor(0.0, device=pred.device)
        if self.percep_loss_fn is not None and self.percep_weight > 0:
            try:
                loss_percep = self.percep_loss_fn(pred_f32, target_f32)
                if torch.isnan(loss_percep) or torch.isinf(loss_percep):
                    loss_percep = torch.tensor(0.0, device=pred.device)
            except Exception as e:
                # 如果感知损失计算失败，跳过
                loss_percep = torch.tensor(0.0, device=pred.device)
        loss_dict['image_percep_loss'] = loss_percep.item()
        
        # 汇总总损失
        total_loss = (
            self.recon_weight * loss_recon +
            self.msssim_weight * loss_msssim +
            self.grad_weight * loss_grad +
            self.percep_weight * loss_percep
        )
        
        # 检查损失值是否有效
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            total_loss = torch.tensor(0.0, device=pred.device)
        
        return total_loss, loss_dict
          

class GenerativeImageLoss(nn.Module):
    """
    生成式图像损失

    使用 GenerativeJSCC 核心形式: gamma1 * MSE + gamma2 * LPIPS
    """

    def __init__(
        self,
        gamma1: float = 1.0,
        gamma2: float = 0.1,
        normalize: bool = False,
    ):
        super().__init__()
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.normalize = normalize
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.percep_loss_fn = PerceptualLoss(use_lpips=LPIPS_AVAILABLE) if gamma2 > 0 else None

    def _maybe_denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return tensor
        return tensor * self.imagenet_std + self.imagenet_mean

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred_f32 = self._maybe_denormalize(pred.float())
        target_f32 = self._maybe_denormalize(target.float())
        loss_mse = F.mse_loss(pred_f32, target_f32)
        loss_lpips = torch.tensor(0.0, device=pred.device)
        if self.percep_loss_fn is not None and self.gamma2 > 0:
            try:
                loss_lpips = self.percep_loss_fn(pred_f32, target_f32)
                if torch.isnan(loss_lpips) or torch.isinf(loss_lpips):
                    loss_lpips = torch.tensor(0.0, device=pred.device)
            except Exception:
                loss_lpips = torch.tensor(0.0, device=pred.device)
        total = self.gamma1 * loss_mse + self.gamma2 * loss_lpips
        if torch.isnan(total) or torch.isinf(total):
            total = torch.tensor(0.0, device=pred.device)
        return total, {
            "image_mse": loss_mse.item(),
            "image_lpips": loss_lpips.item(),
            "image_total": total.item(),
        }


class VideoLoss(nn.Module):
    """
    视频损失 (轻量级优化版)
    
    只使用L1重建损失和简单的时序损失，移除MS-SSIM以节省显存。
    """
    def __init__(
        self, 
        reconstruction_weight: float = 1.0, 
        perceptual_weight: float = 0.0,  # 默认禁用
        temporal_weight: float = 0.1,
        temporal_consistency_weight: float = 0.0,
        temporal_perceptual_weight: float = 0.0,
        color_consistency_weight: float = 0.0,
        data_range: float = 1.0,
        normalize: bool = False,

    ):
        super().__init__()
        self.recon_weight = reconstruction_weight
        self.percep_weight = perceptual_weight
        self.temp_weight = temporal_weight
        self.consistency_weight = temporal_consistency_weight
        self.temporal_perceptual_weight = temporal_perceptual_weight
        self.color_consistency_weight = color_consistency_weight
        self.data_range = data_range
        self.normalize = normalize
        #self.epsilon = 1e-6
        
        # 重建损失 (L1)
        self.recon_loss_fn = nn.L1Loss()
        
        # 感知损失（LPIPS或VGG）
        if self.percep_weight > 0:
            self.percep_loss_fn = PerceptualLoss(use_lpips=LPIPS_AVAILABLE)
        else:
            self.percep_loss_fn = None
        self.register_buffer("imagenet_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("imagenet_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def _maybe_denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self.normalize:
            return tensor
        return tensor * self.imagenet_std + self.imagenet_mean
    
    #def _normalize_to_range(self, img: torch.Tensor) -> torch.Tensor:
        """
        将图像归一化到 [0, 1] 范围
        
        如果输入使用了 ImageNet 归一化（范围约在 [-2.5, 2.5]），
        需要先反归一化，然后归一化到 [0, 1]。
        """
        # 检测是否使用了 ImageNet 归一化（通过数据范围判断）
        #img_min = img.min().item()
        #img_max = img.max().item()
        
        # 如果数据范围明显超出 [0, 1]，可能是 ImageNet 归一化的结果
        #if img_min < -0.5 or img_max > 1.5:
            # ImageNet 归一化参数
            #imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=img.device, dtype=img.dtype)
            #imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=img.device, dtype=img.dtype)
            
            # 添加维度以便广播 [1, 3, 1, 1]
            #mean = imagenet_mean.view(1, 3, 1, 1)
            #std = imagenet_std.view(1, 3, 1, 1)
            
            # 反归一化: x = normalized * std + mean
            #img = img * std + mean
        
        # 确保在 [0, 1] 范围内
        #img = torch.clamp(img, 0.0, 1.0)
        
        #return img
            
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        B, T, C, H, W = pred.shape
        
        pred_f32 = self._maybe_denormalize(pred.float())
        target_f32 = self._maybe_denormalize(target.float())
        loss_recon_avg = F.l1_loss(pred_f32, target_f32)
        loss_temp = torch.tensor(0.0, device=pred.device)
        loss_consistency = torch.tensor(0.0, device=pred.device)
        loss_percep = torch.tensor(0.0, device=pred.device)
        loss_temporal_percep = torch.tensor(0.0, device=pred.device)
        loss_color = torch.tensor(0.0, device=pred.device)
        if pred_f32.size(1) > 1 and self.temp_weight > 0:
            pred_diff = pred_f32[:, 1:] - pred_f32[:, :-1]
            target_diff = target_f32[:, 1:] - target_f32[:, :-1]
            loss_temp = F.l1_loss(pred_diff, target_diff)
        if pred_f32.size(1) > 1 and self.consistency_weight > 0:
            pred_diff = pred_f32[:, 1:] - pred_f32[:, :-1]
            loss_consistency = pred_diff.abs().mean()
        if self.percep_loss_fn is not None and self.percep_weight > 0:
            try:
                pred_frames = pred_f32.reshape(B * T, C, H, W)
                target_frames = target_f32.reshape(B * T, C, H, W)
                loss_percep = self.percep_loss_fn(pred_frames, target_frames)
            except Exception:
                loss_percep = torch.tensor(0.0, device=pred.device)
        if pred_f32.size(1) > 1 and self.temporal_perceptual_weight > 0 and self.percep_loss_fn is not None:
            try:
                pred_diff = pred_f32[:, 1:] - pred_f32[:, :-1]
                target_diff = target_f32[:, 1:] - target_f32[:, :-1]
                pred_frames = pred_diff.reshape(B * (T - 1), C, H, W)
                target_frames = target_diff.reshape(B * (T - 1), C, H, W)
                loss_temporal_percep = self.percep_loss_fn(pred_frames, target_frames)
            except Exception:
                loss_temporal_percep = torch.tensor(0.0, device=pred.device)
        if self.color_consistency_weight > 0:
            pred_mean = pred_f32.mean(dim=(-2, -1), keepdim=True)
            target_mean = target_f32.mean(dim=(-2, -1), keepdim=True)
            pred_std = pred_f32.std(dim=(-2, -1), keepdim=True)
            target_std = target_f32.std(dim=(-2, -1), keepdim=True)
            loss_color = F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)
        total_loss = (
            (self.recon_weight * loss_recon_avg) +
            (self.temp_weight * loss_temp) +
            (self.consistency_weight * loss_consistency) +
            (self.percep_weight * loss_percep) +
            (self.temporal_perceptual_weight * loss_temporal_percep) +
            (self.color_consistency_weight * loss_color)
        )
        return total_loss, {
            'video_recon_loss_l1': loss_recon_avg.item(),
            'video_temporal_loss_l1': loss_temp.item(),
            'video_temporal_consistency_loss': loss_consistency.item(),
            'video_percep_loss': loss_percep.item(),
            'video_temporal_percep_loss': loss_temporal_percep.item(),
            'video_color_consistency_loss': loss_color.item(),
        }
        #diff = pred -target
        #loss_recon_avg= torch.mean(torch.sqrt(diff * diff + self.epsilon))
        #loss_percep_avg = torch.tensor(0.0, device=pred.device)
        #loss_temp = torch.tensor(0.0, device=pred.device)
        #if T > 1 and self.temp_weight > 0:
            #pred_diff = pred[:, 1:] - pred[:, :-1]
            #target_diff = target[:, 1:] - target[:, :-1]
            #diff_motion = pred_diff - target_diff
            #loss_temp = torch.mean(torch.sqrt(diff_motion * diff_motion + self.epsilon))

        
        # 1. 计算逐帧的重建损失 (L1) - 已移除MS-SSIM以节省显存
        #for t in range(T):
            #frame_pred = pred_normalized[:, t]
            #frame_target = target_normalized[:, t]
            
            # L1 重建损失
            #loss_recon_total += self.recon_loss_fn(frame_pred, frame_target)
        
        #loss_recon_avg = loss_recon_total / T
        #loss_percep_avg = torch.tensor(0.0, device=pred.device)  # 感知损失已移除
        
        # 2. 计算时序损失 (帧间差异, L1)
        #loss_temp = torch.tensor(0.0, device=pred.device)
        #if T > 1 and self.temp_weight > 0:
            #pred_diff = pred_normalized[:, 1:] - pred_normalized[:, :-1]
            #target_diff = target_normalized[:, 1:] - target_normalized[:, :-1]
            #loss_temp = self.recon_loss_fn(pred_diff, target_diff) # 使用 L1 计算运动差异
        
        # 3. 汇总总损失（添加数值稳定性检查）
        # 检查损失值是否有效
        #if torch.isnan(loss_recon_avg) or torch.isinf(loss_recon_avg):
            #loss_recon_avg = torch.tensor(0.0, device=pred.device)
        #if torch.isnan(loss_percep_avg) or torch.isinf(loss_percep_avg):
            #loss_percep_avg = torch.tensor(0.0, device=pred.device)
        #if torch.isnan(loss_temp) or torch.isinf(loss_temp):
            #loss_temp = torch.tensor(0.0, device=pred.device)
        
        #total_loss = (
            #(self.recon_weight * loss_recon_avg) +
            #(self.percep_weight * loss_percep_avg) +
            #(self.temp_weight * loss_temp)
        #)
        
        # 最终检查
        #if torch.isnan(total_loss) or torch.isinf(total_loss):
            #total_loss = torch.tensor(0.0, device=pred.device)
        
       # return total_loss, {
            #'video_recon_loss_l1': loss_recon_avg.item() if not torch.isnan(loss_recon_avg) else 0.0,
            ##'video_percep_loss_msssim': 0.0,  # 已移除
            #'video_temporal_loss_l1': loss_temp.item() if not torch.isnan(loss_temp) else 0.0
        #}


# --------------------------------------------------------------------------
# 对抗损失（Phase 3: GAN Loss）
# --------------------------------------------------------------------------

class AdversarialLoss(nn.Module):
    """
    对抗损失（Phase 3: GAN Loss）
    
    使用最小二乘GAN损失（LSGAN），比标准GAN损失更稳定。
    """
    def __init__(self, target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super().__init__()
        self.target_real_label = target_real_label
        self.target_fake_label = target_fake_label
        self.loss_fn = nn.MSELoss()
    
    def forward(
        self,
        discriminator_output: torch.Tensor,
        target_is_real: bool
    ) -> torch.Tensor:
        """
        计算对抗损失
        
        Args:
            discriminator_output: 判别器输出 [B, 1, H, W] 或 [B, 1]
            target_is_real: 目标是否为真实图像
            
        Returns:
            torch.Tensor: 对抗损失值
        """
        if target_is_real:
            target = torch.full_like(discriminator_output, self.target_real_label)
        else:
            target = torch.full_like(discriminator_output, self.target_fake_label)
        
        loss = self.loss_fn(discriminator_output, target)
        return loss


# --------------------------------------------------------------------------
# 多模态损失协调器 (保持不变，仅更新调用的子损失)
# --------------------------------------------------------------------------

class MultimodalLoss(nn.Module):
    """
    多模态损失协调器
    
    根据 run_training.py 中的参数，动态计算所有可用模态的加权损失。
    """
    def __init__(
        self,
        text_weight: float = 1.0,
        image_weight: float = 1.0,
        video_weight: float = 1.0,
        image_decoder_type: str = "baseline",
        
        # 图像/视频损失的子权重
        reconstruction_weight: float = 1.0,
        perceptual_weight: float = 0.1,
        temporal_weight: float = 0.1,
        video_perceptual_weight: float = 0.0,
        temporal_perceptual_weight: float = 0.0,
        text_contrastive_weight: float = 0.1,  # 【新增】文本对比损失权重
        video_text_contrastive_weight: float = 0.05,  # 【新增】视频-文本对比损失权重
        rate_weight: float = 1e-4,  # 【新增】码率/能量约束权重
        temporal_consistency_weight: float = 0.0,  # 【新增】视频时序一致性正则权重
        color_consistency_weight: float = 0.0,
        discriminator_weight: float = 0.01,  # 【Phase 3】对抗损失权重（默认较小）
        gan_weight: Optional[float] = None,  # 【Phase 6】对抗损失权重（统一命名）
        condition_margin_weight: float = 0.0,
        condition_margin: float = 0.05,
        generative_gamma1: float = 1.0,
        generative_gamma2: float = 0.1,
        
        # 假设数据范围是 [0, 1]
        data_range: float = 1.0,
        normalize: bool = False,
        use_adversarial: bool = False  # 【Phase 3】是否使用对抗训练
    ):
        super().__init__()
        
        self.text_weight = text_weight
        self.image_weight = image_weight
        self.video_weight = video_weight
        self.image_decoder_type = image_decoder_type
        
        # 【新增】文本对比损失权重
        self.text_contrastive_weight = text_contrastive_weight
        self.video_text_contrastive_weight = video_text_contrastive_weight
        self.rate_weight = rate_weight
        self.condition_margin_weight = condition_margin_weight
        self.condition_margin = condition_margin
        
        # 【Phase 3】对抗训练相关
        self.use_adversarial = use_adversarial
        if gan_weight is not None:
            discriminator_weight = gan_weight
        self.discriminator_weight = discriminator_weight
        if self.use_adversarial:
            self.adversarial_loss_fn = AdversarialLoss()
        
        # 初始化各个模态的损失函数
        self.text_loss_fn = TextLoss(
            use_contrastive=self.text_contrastive_weight > 0,
            contrastive_weight=self.text_contrastive_weight
        )
        self.video_text_contrastive_loss_fn = TextImageContrastiveLoss()
        
        if image_decoder_type.lower() == "generative":
            self.image_loss_fn = GenerativeImageLoss(
                gamma1=generative_gamma1,
                gamma2=generative_gamma2,
                normalize=normalize,
            )
        else:
            self.image_loss_fn = ImageLoss(
                reconstruction_weight=reconstruction_weight,
                perceptual_weight=perceptual_weight,
                data_range=data_range,
                normalize=normalize,
            )
        
        self.video_loss_fn = VideoLoss(
            reconstruction_weight=reconstruction_weight,
            perceptual_weight=video_perceptual_weight,
            temporal_weight=temporal_weight,
            temporal_consistency_weight=temporal_consistency_weight,
            temporal_perceptual_weight=temporal_perceptual_weight,
            color_consistency_weight=color_consistency_weight,
            data_range=data_range,
            normalize=normalize,
        )
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        targets: Dict[str, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        discriminator_outputs: Optional[Dict[str, torch.Tensor]] = None  # 【Phase 3】判别器输出
    ) -> Dict[str, torch.Tensor]:
        
        # 动态获取设备
        # 优先从输入张量获取设备，如果没有则从模块参数获取
        device = None
        if predictions:
            # 从第一个预测张量获取设备
            for value in predictions.values():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
        if device is None and targets:
            # 从第一个目标张量获取设备
            for value in targets.values():
                if isinstance(value, torch.Tensor):
                    device = value.device
                    break
        if device is None:
            # 最后尝试从模块参数获取设备
            try:
                device = next(iter(self.parameters())).device
            except StopIteration:
                # 如果模块没有任何参数，尝试从子模块获取
                try:
                    device = next(iter(self.text_loss_fn.parameters())).device
                except StopIteration:
                    # 如果还是没有，使用默认设备
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 检查 targets 是否为空，如果为空则返回 0 损失
        if not targets:
            return {'total_loss': torch.tensor(0.0, device=device)}
            
        total_loss: Optional[torch.Tensor] = None
        loss_dict = {}
        
        # --- 文本损失 ---
        if 'text_decoded' in predictions and 'text' in targets:
            try:
                # 【修复】传递重建图像特征和文本嵌入用于对比损失
                reconstructed_image_features = predictions.get('image_decoded', None)
                if reconstructed_image_features is None:
                    reconstructed_image_features = predictions.get('image_encoded', None)
                text_embeddings = predictions.get('text_encoded', None)
                
                # 如果图像特征是4D [B, C, H, W]，需要提取特征
                if reconstructed_image_features is not None and reconstructed_image_features.dim() == 4:
                    # 使用简单的全局平均池化提取特征
                    # 注意：这里使用简单的池化，实际可以使用预训练的视觉编码器
                    B, C, H, W = reconstructed_image_features.shape
                    reconstructed_image_features = reconstructed_image_features.view(B, C, -1).mean(dim=-1)  # [B, C]
                
                text_loss, text_loss_comps = self.text_loss_fn(
                    predictions['text_decoded'], 
                    targets['text'],
                    attention_mask,
                    reconstructed_image_features=reconstructed_image_features,
                    text_embeddings=text_embeddings
                )
                
                weighted_text_loss = self.text_weight * text_loss
                total_loss = weighted_text_loss if total_loss is None else (total_loss + weighted_text_loss)
                
                loss_dict['text_loss'] = weighted_text_loss.item()
                loss_dict.update(text_loss_comps)
                
            except Exception as e:
                print(f"警告: 计算文本损失时出错: {e}")
                import traceback
                print(traceback.format_exc())
                loss_dict['text_loss'] = 0.0

        # --- 图像损失 ---
        if 'image_decoded' in predictions and 'image' in targets:
            try:
                # 解码器输出在训练配置下可能已标准化，损失内部会负责还原到[0,1]
                pred_img = predictions['image_decoded']
                
                image_loss, image_loss_comps = self.image_loss_fn(
                    pred_img, 
                    targets['image']
                )
                
                weighted_image_loss = self.image_weight * image_loss
                total_loss = weighted_image_loss if total_loss is None else (total_loss + weighted_image_loss)
                
                loss_dict['image_loss'] = weighted_image_loss.item()
                loss_dict.update(image_loss_comps)
                
            except Exception as e:
                print(f"警告: 计算图像损失时出错: {e}")
                loss_dict['image_loss'] = 0.0

        # --- 视频损失 ---
        if 'video_decoded' in predictions and 'video' in targets:
            try:
                # 解码器输出在训练配置下可能已标准化，损失内部会负责还原到[0,1]
                pred_vid = predictions['video_decoded']
                
                video_loss, video_loss_comps = self.video_loss_fn(
                    pred_vid, 
                    targets['video']
                )
                
                weighted_video_loss = self.video_weight * video_loss
                total_loss = weighted_video_loss if total_loss is None else (total_loss + weighted_video_loss)
                
                loss_dict['video_loss'] = weighted_video_loss.item()
                loss_dict.update(video_loss_comps)
                
            except Exception as e:
                import traceback
                print(f"警告: 计算视频损失时出错: {e}")
                print(traceback.format_exc())
                # 如果视频损失计算失败，不添加到总损失中
                loss_dict['video_loss'] = 0.0
                loss_dict['video_recon_loss_l1'] = 0.0
                loss_dict['video_percep_loss_msssim'] = 0.0
                loss_dict['video_temporal_loss_l1'] = 0.0
                loss_dict['video_temporal_consistency_loss'] = 0.0

        # --- 视频-文本对比损失 ---
        if self.video_text_contrastive_weight > 0 and 'video_encoded' in predictions and 'text_encoded' in predictions:
            try:
                video_text_loss = self.video_text_contrastive_loss_fn(
                    predictions['video_encoded'],
                    predictions['text_encoded']
                )
                weighted_video_text_loss = self.video_text_contrastive_weight * video_text_loss
                total_loss = weighted_video_text_loss if total_loss is None else (total_loss + weighted_video_text_loss)
                loss_dict['video_text_contrastive_loss'] = weighted_video_text_loss.item()
            except Exception as e:
                print(f"警告: 计算视频-文本对比损失时出错: {e}")
                loss_dict['video_text_contrastive_loss'] = 0.0

        # --- 条件边际损失 ---
        if (
            self.condition_margin_weight > 0
            and 'video_decoded' in predictions
            and 'video_decoded_shuffled' in predictions
            and 'video' in targets
        ):
            pred_correct = predictions['video_decoded']
            pred_shuffled = predictions['video_decoded_shuffled']
            target_video = targets['video']
            lc = torch.mean(torch.abs(pred_correct - target_video))
            ls = torch.mean(torch.abs(pred_shuffled - target_video))
            condition_penalty = torch.relu(self.condition_margin - (ls - lc))
            condition_loss = condition_penalty.mean()
            weighted_condition_loss = self.condition_margin_weight * condition_loss
            total_loss = weighted_condition_loss if total_loss is None else (total_loss + weighted_condition_loss)
            loss_dict['condition_margin_loss'] = weighted_condition_loss.item()

        # --- 码率/能量约束 ---
        if self.rate_weight > 0 and 'rate_stats' in predictions:
            rate_loss = None
            try:
                for value in predictions['rate_stats'].values():
                    if isinstance(value, torch.Tensor):
                        rate_loss = value if rate_loss is None else (rate_loss + value)
                if rate_loss is not None:
                    weighted_rate_loss = self.rate_weight * rate_loss
                    total_loss = weighted_rate_loss if total_loss is None else (total_loss + weighted_rate_loss)
                    loss_dict['rate_loss'] = weighted_rate_loss.item()
            except Exception as e:
                print(f"警告: 计算码率损失时出错: {e}")
                loss_dict['rate_loss'] = 0.0
        
        # 【Phase 3】对抗损失（如果启用）
        if self.use_adversarial and discriminator_outputs is not None:
            adversarial_loss = torch.tensor(0.0, device=device)
            
            # 图像对抗损失
            if 'image' in discriminator_outputs and 'image_decoded' in predictions:
                image_disc_pred = discriminator_outputs['image']
                adversarial_loss_image = self.adversarial_loss_fn(image_disc_pred, target_is_real=False)
                adversarial_loss = adversarial_loss + adversarial_loss_image
                loss_dict['image_adversarial_loss'] = adversarial_loss_image.item()
            
            # 视频对抗损失
            if 'video' in discriminator_outputs and 'video_decoded' in predictions:
                video_disc_pred = discriminator_outputs['video']
                adversarial_loss_video = self.adversarial_loss_fn(video_disc_pred, target_is_real=False)
                adversarial_loss = adversarial_loss + adversarial_loss_video
                loss_dict['video_adversarial_loss'] = adversarial_loss_video.item()
            
            # 加权对抗损失
            weighted_adversarial_loss = self.discriminator_weight * adversarial_loss
            total_loss = total_loss + weighted_adversarial_loss if total_loss is not None else weighted_adversarial_loss
            loss_dict['adversarial_loss'] = weighted_adversarial_loss.item()
        
        if total_loss is None:
            raise RuntimeError("未能计算任何有效的损失项，请检查前向输出与目标是否匹配以及设备/维度配置。")
        loss_dict['total_loss'] = total_loss
        
        return loss_dict
