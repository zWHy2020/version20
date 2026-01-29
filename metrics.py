"""
多模态JSCC评估指标

实现PSNR、SSIM、BLEU、ROUGE等评估指标。
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple
import math


def calculate_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0
) -> torch.Tensor:
    """
    计算峰值信噪比(PSNR)
    
    Args:
        pred (torch.Tensor): 预测图像 [B, C, H, W] 或 [B, T, C, H, W]
        target (torch.Tensor): 目标图像 [B, C, H, W] 或 [B, T, C, H, W]
        max_val (float): 像素值最大值
        
    Returns:
        torch.Tensor: PSNR值
    """
    # 修复：对输入进行范围限制，确保在 [0, max_val] 范围内
    # 这对于模型输出可能超出 [0, 1] 范围的情况特别重要
    pred = torch.clamp(pred, 0.0, max_val)
    target = torch.clamp(target, 0.0, max_val)
    
    # 修复类型不匹配问题：将输入转换为float32，避免HalfTensor和FloatTensor不匹配
    # 这在混合精度训练（AMP）时特别重要，因为模型输出可能是float16
    if pred.dtype != torch.float32:
        pred = pred.float()
    if target.dtype != torch.float32:
        target = target.float()
    
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return torch.tensor(float('inf'), device=pred.device)
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True,
    data_range: float = 1.0
) -> torch.Tensor:
    """
    计算结构相似性指数(SSIM)
    
    Args:
        pred (torch.Tensor): 预测图像 [B, C, H, W]
        target (torch.Tensor): 目标图像 [B, C, H, W]
        window_size (int): 窗口大小
        size_average (bool): 是否平均
        data_range (float): 数据范围，默认为 1.0 (假设输入在 [0, 1])
        
    Returns:
        torch.Tensor: SSIM值
    """
    # 修复：对输入进行范围限制，确保在 [0, data_range] 范围内
    # 这对于模型输出可能超出 [0, 1] 范围的情况特别重要
    pred = torch.clamp(pred, 0.0, data_range)
    target = torch.clamp(target, 0.0, data_range)
    
    # 修复类型不匹配问题：将输入转换为float32，避免HalfTensor和FloatTensor不匹配
    # 这在混合精度训练（AMP）时特别重要，因为模型输出可能是float16
    if pred.dtype != torch.float32:
        pred = pred.float()
    if target.dtype != torch.float32:
        target = target.float()
    
    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32, device=pred.device)
        coords = coords - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        return g.unsqueeze(0).unsqueeze(0)
    
    window = gaussian_window(window_size).to(pred.device)
    window = window.expand(pred.size(1), 1, window_size, window_size)
    
    # 计算均值
    mu1 = F.conv2d(pred, window, padding=window_size//2, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=window_size//2, groups=target.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # 计算方差和协方差
    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size//2, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size//2, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size//2, groups=target.size(1)) - mu1_mu2
    
    # SSIM参数（使用 data_range）
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_bleu(
    predictions: List[List[int]],
    references: List[List[int]],
    n_gram: int = 4,
    weights: Optional[List[float]] = None
) -> float:
    """
    计算BLEU分数
    
    Args:
        predictions: 预测序列列表
        references: 参考序列列表
        n_gram: N-gram大小
        weights: N-gram权重
        
    Returns:
        float: BLEU分数
    """
    if weights is None:
        weights = [1.0 / n_gram] * n_gram
    
    def get_ngrams(tokens: List[int], n: int) -> Dict[tuple, int]:
        ngrams = {}
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] = ngrams.get(ngram, 0) + 1
        return ngrams
    
    def get_clipped_counts(pred_ngrams: Dict[tuple, int], ref_ngrams: Dict[tuple, int]) -> int:
        clipped = 0
        for ngram, count in pred_ngrams.items():
            clipped += min(count, ref_ngrams.get(ngram, 0))
        return clipped
    
    # 计算各N-gram的精确度
    precisions = []
    for n in range(1, n_gram + 1):
        pred_ngrams = {}
        ref_ngrams = {}
        total_pred = 0
        total_clipped = 0
        
        for pred, ref in zip(predictions, references):
            pred_ngram = get_ngrams(pred, n)
            ref_ngram = get_ngrams(ref, n)
            
            for ngram, count in pred_ngram.items():
                pred_ngrams[ngram] = pred_ngrams.get(ngram, 0) + count
            for ngram, count in ref_ngram.items():
                ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + count
            
            total_pred += sum(pred_ngram.values())
            total_clipped += get_clipped_counts(pred_ngram, ref_ngram)
        
        if total_pred == 0:
            precision = 0.0
        else:
            precision = total_clipped / total_pred
        
        precisions.append(precision)
    
    # 计算几何平均
    if any(p == 0 for p in precisions):
        return 0.0
    
    geometric_mean = math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
    
    # 计算简洁惩罚
    pred_lengths = [len(pred) for pred in predictions]
    ref_lengths = [len(ref) for ref in references]
    
    total_pred_length = sum(pred_lengths)
    total_ref_length = sum(ref_lengths)
    
    if total_pred_length == 0:
        return 0.0
    
    bp = min(1.0, total_pred_length / total_ref_length) if total_ref_length > 0 else 0.0
    
    return bp * geometric_mean


def calculate_rouge_l(
    predictions: List[List[int]],
    references: List[List[int]]
) -> float:
    """
    计算ROUGE-L分数
    
    Args:
        predictions: 预测序列列表
        references: 参考序列列表
        
    Returns:
        float: ROUGE-L分数
    """
    def lcs_length(seq1: List[int], seq2: List[int]) -> int:
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    
    for pred, ref in zip(predictions, references):
        lcs_len = lcs_length(pred, ref)
        
        if len(pred) == 0:
            precision = 0.0
        else:
            precision = lcs_len / len(pred)
        
        if len(ref) == 0:
            recall = 0.0
        else:
            recall = lcs_len / len(ref)
        
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        
        total_precision += precision
        total_recall += recall
        total_f1 += f1
    
    n = len(predictions)
    return {
        'rouge_l_precision': total_precision / n,
        'rouge_l_recall': total_recall / n,
        'rouge_l_f1': total_f1 / n
    }


def _normalize_image_range(
    img: torch.Tensor,
    target_range: Tuple[float, float] = (0.0, 1.0),
    detect_range: bool = True,
    imagenet_normalized: bool = False,
    imagenet_mean: Optional[List[float]] = None,
    imagenet_std: Optional[List[float]] = None
) -> torch.Tensor:
    """
    归一化图像到指定范围
    
    Args:
        img: 输入图像张量 [B, C, H, W] 或 [B, T, C, H, W]
        target_range: 目标范围 (min, max)，默认为 (0.0, 1.0)
        detect_range: 是否自动检测输入范围
        imagenet_normalized: 是否使用了 ImageNet 归一化（mean/std）
        imagenet_mean: ImageNet 均值，如果 imagenet_normalized=True 则必须提供
        imagenet_std: ImageNet 标准差，如果 imagenet_normalized=True 则必须提供
        
    Returns:
        torch.Tensor: 归一化后的图像
    """
    # 如果使用了 ImageNet 归一化，先反归一化
    if imagenet_normalized:
        if imagenet_mean is None or imagenet_std is None:
            imagenet_mean = [0.485, 0.456, 0.406]
            imagenet_std = [0.229, 0.224, 0.225]

        mean = torch.tensor(imagenet_mean, device=img.device, dtype=img.dtype)
        std = torch.tensor(imagenet_std, device=img.device, dtype=img.dtype)

        if img.dim() == 4:  # [B, C, H, W]
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        elif img.dim() == 5:  # [B, T, C, H, W]
            mean = mean.view(1, 1, 3, 1, 1)
            std = std.view(1, 1, 3, 1, 1)

        img = img * std + mean
    
    # 自动检测数据范围并归一化到 [0, 1]
    if detect_range:
        img_min = img.min()
        img_max = img.max()
        
        # 如果数据范围不在 [0, 1]，进行归一化
        if img_min < 0.0 or img_max > 1.0:
            # 线性归一化到 [0, 1]
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                # 全零或常数图像
                img = torch.zeros_like(img)
        pass
    
    # 裁剪到目标范围
    img = torch.clamp(img, target_range[0], target_range[1])
    
    return img


def _validate_and_normalize_visual_data(
    pred: torch.Tensor,
    target: torch.Tensor,
    target_range: Tuple[float, float] = (0.0, 1.0),
    imagenet_normalized: bool = False,
    imagenet_mean: Optional[List[float]] = None,
    imagenet_std: Optional[List[float]] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    验证并归一化视觉数据（图像/视频）
    
    Args:
        pred: 预测张量 [B, C, H, W] 或 [B, T, C, H, W]
        target: 目标张量 [B, C, H, W] 或 [B, T, C, H, W]
        target_range: 目标范围 (min, max)，默认为 (0.0, 1.0)
        imagenet_normalized: 目标是否使用了 ImageNet 归一化
        imagenet_mean: ImageNet 均值（如果使用 ImageNet 归一化）
        imagenet_std: ImageNet 标准差（如果使用 ImageNet 归一化）
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (归一化后的预测, 归一化后的目标)
    """
    # 验证维度匹配
    if pred.shape != target.shape:
        raise ValueError(
            f"预测和目标形状不匹配: pred={pred.shape}, target={target.shape}"
        )
    
    # 验证维度正确性（应该是 4D 或 5D）
    if pred.dim() not in [4, 5]:
        raise ValueError(
            f"输入张量维度错误: 应为 4D [B, C, H, W] 或 5D [B, T, C, H, W], "
            f"实际为 {pred.dim()}D, shape={pred.shape}"
        )
    
    # 归一化预测（只进行裁剪，不进行动态范围拉伸，避免扭曲评估结果）
    pred_normalized = _normalize_image_range(
        pred,
        target_range=target_range,
        detect_range=False,  # 只进行裁剪，不进行动态拉伸
        imagenet_normalized=imagenet_normalized,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std,
    )
    
    # 归一化目标（只进行裁剪，不进行动态范围拉伸）
    # 如果目标使用了 ImageNet 归一化，需要先反归一化
    target_normalized = _normalize_image_range(
        target,
        target_range=target_range,
        detect_range=False,  # 只进行裁剪，不进行动态拉伸
        imagenet_normalized=imagenet_normalized,
        imagenet_mean=imagenet_mean,
        imagenet_std=imagenet_std
    )
    
    return pred_normalized, target_normalized


def calculate_multimodal_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    imagenet_normalized: bool = False,
    imagenet_mean: Optional[List[float]] = None,
    imagenet_std: Optional[List[float]] = None
) -> Dict[str, float]:
    """
    计算多模态评估指标
    
    Args:
        predictions: 预测结果字典，应包含 'image_decoded', 'video_decoded', 'text_decoded'
        targets: 目标结果字典，应包含 'image', 'video', 'text'
        attention_mask: 文本注意力掩码 [B, seq_len]
        imagenet_normalized: 目标图像/视频是否使用了 ImageNet 归一化（mean/std）
        imagenet_mean: ImageNet 均值，如果 imagenet_normalized=True 则使用
                      默认为 [0.485, 0.456, 0.406]
        imagenet_std: ImageNet 标准差，如果 imagenet_normalized=True 则使用
                     默认为 [0.229, 0.224, 0.225]
        
    Returns:
        Dict[str, float]: 评估指标字典，包含：
            - image_psnr: 图像 PSNR
            - image_ssim: 图像 SSIM
            - video_psnr_mean: 视频平均 PSNR
            - video_psnr_std: 视频 PSNR 标准差
            - video_ssim_mean: 视频平均 SSIM
            - video_ssim_std: 视频 SSIM 标准差
            - text_bleu: 文本 BLEU 分数
            - rouge_l_precision: ROUGE-L 精确度
            - rouge_l_recall: ROUGE-L 召回率
            - rouge_l_f1: ROUGE-L F1 分数
    """
    metrics = {}
    
    # 图像指标
    if 'image_decoded' in predictions and 'image' in targets:
        try:
            pred_img = predictions['image_decoded']
            target_img = targets['image']
            
            # 验证并归一化图像数据
            pred_img_norm, target_img_norm = _validate_and_normalize_visual_data(
                pred_img,
                target_img,
                target_range=(0.0, 1.0),
                imagenet_normalized=imagenet_normalized,
                imagenet_mean=imagenet_mean,
                imagenet_std=imagenet_std
            )
            
            # 计算 PSNR
            psnr = calculate_psnr(pred_img_norm, target_img_norm, max_val=1.0)
            metrics['image_psnr'] = psnr.item()
            
            # 计算 SSIM
            ssim = calculate_ssim(pred_img_norm, target_img_norm, data_range=1.0)
            metrics['image_ssim'] = ssim.item()
            
        except Exception as e:
            print(f"警告: 计算图像指标时出错: {e}")
            metrics['image_psnr'] = 0.0
            metrics['image_ssim'] = 0.0
    
    # 视频指标
    if 'video_decoded' in predictions and 'video' in targets:
        try:
            pred_vid = predictions['video_decoded']
            target_vid = targets['video']
            
            # 验证并归一化视频数据
            pred_vid_norm, target_vid_norm = _validate_and_normalize_visual_data(
                pred_vid,
                target_vid,
                target_range=(0.0, 1.0),
                imagenet_normalized=imagenet_normalized,
                imagenet_mean=imagenet_mean,
                imagenet_std=imagenet_std
            )
            
            # 计算每帧的 PSNR 和 SSIM
            B, T, C, H, W = pred_vid_norm.shape
            frame_psnrs = []
            frame_ssims = []
            
            for t in range(T):
                frame_psnr = calculate_psnr(
                    pred_vid_norm[:, t], 
                    target_vid_norm[:, t], 
                    max_val=1.0
                )
                frame_ssim = calculate_ssim(
                    pred_vid_norm[:, t], 
                    target_vid_norm[:, t], 
                    data_range=1.0
                )
                frame_psnrs.append(frame_psnr.item())
                frame_ssims.append(frame_ssim.item())
            
            # 计算统计量
            frame_psnrs_tensor = torch.tensor(frame_psnrs)
            frame_ssims_tensor = torch.tensor(frame_ssims)
            
            metrics['video_psnr_mean'] = frame_psnrs_tensor.mean().item()
            metrics['video_ssim_mean'] = frame_ssims_tensor.mean().item()
            metrics['video_psnr_std'] = frame_psnrs_tensor.std().item()
            metrics['video_ssim_std'] = frame_ssims_tensor.std().item()
            
        except Exception as e:
            print(f"警告: 计算视频指标时出错: {e}")
            metrics['video_psnr_mean'] = 0.0
            metrics['video_ssim_mean'] = 0.0
            metrics['video_psnr_std'] = 0.0
            metrics['video_ssim_std'] = 0.0
    
    # 文本指标（需要转换为token序列）
    if 'text_decoded' in predictions and 'text' in targets:
        pred_text = predictions['text_decoded']
        target_text = targets['text']
        
        # 获取预测的token序列
        pred_tokens = torch.argmax(pred_text, dim=-1)  # [B, seq_len]
        
        # 转换为列表格式
        pred_sequences = []
        target_sequences = []
        
        for b in range(pred_tokens.size(0)):
            if attention_mask is not None:
                # 使用注意力掩码过滤
                mask = attention_mask[b].bool()
                pred_seq = pred_tokens[b][mask].tolist()
                target_seq = target_text[b][mask].tolist()
            else:
                pred_seq = pred_tokens[b].tolist()
                target_seq = target_text[b].tolist()
            
            pred_sequences.append(pred_seq)
            target_sequences.append(target_seq)
        
        # 计算BLEU
        bleu_score = calculate_bleu(pred_sequences, target_sequences)
        metrics['text_bleu'] = bleu_score
        
        # 计算ROUGE-L
        rouge_scores = calculate_rouge_l(pred_sequences, target_sequences)
        metrics.update(rouge_scores)
    
    return metrics


def calculate_compression_ratio(
    original_size: Dict[str, int],
    compressed_size: Dict[str, int]
) -> Dict[str, float]:
    """
    计算压缩比
    
    Args:
        original_size: 原始数据大小（字节）
        compressed_size: 压缩后数据大小（字节）

    Returns:
        Dict[str, float]: 压缩比字典
    """
    ratios = {}
    
    for modality in original_size:
        if modality in compressed_size:
            if compressed_size[modality] > 0:
                ratios[f'{modality}_compression_ratio'] = (
                    original_size[modality] / compressed_size[modality]
                )
            else:
                ratios[f'{modality}_compression_ratio'] = float('inf')
    
    return ratios


def calculate_transmission_efficiency(
    data_size: Dict[str, int],
    transmission_time: Dict[str, float],
    bandwidth: float = 1.0  # Mbps
) -> Dict[str, float]:
    """
    计算传输效率
    
    Args:
        data_size: 数据大小（字节）
        transmission_time: 传输时间（秒）
        bandwidth: 带宽（Mbps）
        
    Returns:
        Dict[str, float]: 传输效率字典
    """
    efficiencies = {}
    
    for modality in data_size:
        if modality in transmission_time:
            # 计算理论传输时间
            data_size_mb = data_size[modality] / (1024 * 1024)  # 转换为MB
            theoretical_time = data_size_mb / bandwidth
            
            # 计算传输效率
            if theoretical_time > 0:
                efficiency = theoretical_time / transmission_time[modality]
                efficiencies[f'{modality}_transmission_efficiency'] = efficiency
            else:
                efficiencies[f'{modality}_transmission_efficiency'] = 0.0
    
    return efficiencies

