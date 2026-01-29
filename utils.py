"""
多模态JSCC工具函数模块

包含图像/视频patch处理、指标统计、日志配置等辅助函数。
"""

import os
import random
import numpy as np
import torch
import logging
from typing import Dict, Any, Tuple, List
from datetime import datetime
import json


class AverageMeter:
    """
    计算和存储平均值和当前值
    
    用于跟踪训练过程中的指标变化。
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置所有统计信息"""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        """
        更新统计信息
        
        Args:
            val: 新的值
            n: 样本数量（默认1）
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
    
    def clear(self):
        """清空当前值，保留平均值"""
        self.val = 0


def seed_torch(seed: int = 42):
    """
    设置随机种子以确保可重复性
    
    Args:
        seed: 随机种子值
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def makedirs(path: str):
    """
    创建目录（如果不存在）
    
    Args:
        path: 目录路径
    """
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def logger_configuration(log_config, save_log: bool = True):
    """
    配置日志记录器
    
    Args:
        log_config: 日志配置对象，应包含以下属性：
            - workdir: 工作目录
            - log: 日志文件路径
            - samples: 样本保存目录
            - models: 模型保存目录
        save_log: 是否保存日志到文件
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    # 创建必要的目录
    if hasattr(log_config, 'workdir'):
        makedirs(log_config.workdir)
    if hasattr(log_config, 'samples'):
        makedirs(log_config.samples)
    if hasattr(log_config, 'models'):
        makedirs(log_config.models)
    
    # 创建日志记录器
    logger = logging.getLogger('multimodal_jscc')
    logger.setLevel(logging.INFO)
    
    # 避免重复添加处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器（如果启用）
    if save_log and hasattr(log_config, 'log'):
        makedirs(os.path.dirname(log_config.log))
        file_handler = logging.FileHandler(log_config.log, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_manifest(manifest_path: str) -> list:
    """
    加载数据清单
    
    Args:
        manifest_path: 清单文件路径
    
    Returns:
        list: 数据清单列表，如果文件不存在则返回空列表
    """
    if manifest_path is None or not os.path.exists(manifest_path):
        return []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_image_v2(
    image: torch.Tensor,
    patch_size: int = 128,
    overlap: int = 32
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    将图像分割成重叠的patches
    
    Args:
        image: 输入图像 [C, H, W] 或 [B, C, H, W]
        patch_size: patch大小
        overlap: 重叠区域大小
    
    Returns:
        patches: [N, C, patch_size, patch_size]
        meta: 包含位置信息的字典，用于后续合并
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)  # [1, C, H, W]
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    B, C, H, W = image.shape
    stride = patch_size - overlap
    
    patches = []
    positions = []
    
    for b in range(B):
        img = image[b]  # [C, H, W]
        batch_patches = []
        batch_positions = []
        
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                # 计算patch的实际位置
                y_end = min(y + patch_size, H)
                x_end = min(x + patch_size, W)
                
                # 提取patch
                patch = img[:, y:y_end, x:x_end]  # [C, h, w]
                
                # 【修复】记录原始patch尺寸（在padding之前）
                original_h = patch.shape[1]  # y_end - y
                original_w = patch.shape[2]  # x_end - x
                
                # 如果patch尺寸不足，进行padding
                if patch.shape[1] < patch_size or patch.shape[2] < patch_size:
                    pad_h = patch_size - patch.shape[1]
                    pad_w = patch_size - patch.shape[2]
                    
                    # 选择padding模式：
                    # PyTorch的reflect模式要求：padding大小 < 输入维度大小
                    # 如果任一维度的padding >= 输入维度，则使用replicate模式
                    h, w = patch.shape[1], patch.shape[2]
                    
                    # 分别处理高度和宽度，使用最适合的padding模式
                    # 先处理高度
                    if pad_h > 0:
                        if pad_h < h:
                            # padding大小小于输入维度，可以使用reflect
                            patch = torch.nn.functional.pad(
                                patch, (0, 0, 0, pad_h), mode='reflect'
                            )
                        else:
                            # padding大小 >= 输入维度，使用replicate
                            patch = torch.nn.functional.pad(
                                patch, (0, 0, 0, pad_h), mode='replicate'
                            )
                    
                    # 再处理宽度
                    if pad_w > 0:
                        if pad_w < patch.shape[2]:  # 注意：此时patch的高度可能已经改变
                            # padding大小小于输入维度，可以使用reflect
                            patch = torch.nn.functional.pad(
                                patch, (0, pad_w, 0, 0), mode='reflect'
                            )
                        else:
                            # padding大小 >= 输入维度，使用replicate
                            patch = torch.nn.functional.pad(
                                patch, (0, pad_w, 0, 0), mode='replicate'
                            )
                
                batch_patches.append(patch)
                batch_positions.append({
                    'y': y, 'x': x,
                    'y_end': y_end, 'x_end': x_end,
                    'h': original_h, 'w': original_w  # 【修复】使用原始尺寸而不是padding后的尺寸
                })
        
        patches.extend(batch_patches)
        positions.extend(batch_positions)
    
    patches_tensor = torch.stack(patches, dim=0)  # [N, C, patch_size, patch_size]
    
    meta = {
        'original_shape': (H, W),
        'positions': positions,
        'patch_size': patch_size,
        'overlap': overlap,
        'squeeze_batch': squeeze_batch
    }
    
    return patches_tensor, meta


def merge_image_v2(
    patches: torch.Tensor,
    meta: Dict[str, Any]
) -> torch.Tensor:
    """
    将patches合并回原始图像
    
    Args:
        patches: [N, C, patch_size, patch_size]
        meta: 从split_image_v2返回的元信息
    
    Returns:
        image: [C, H, W] 或 [B, C, H, W]
    """
    H, W = meta['original_shape']
    patch_size = meta['patch_size']
    overlap = meta['overlap']
    positions = meta['positions']
    squeeze_batch = meta.get('squeeze_batch', False)
    
    C = patches.shape[1]
    
    # 创建输出图像和权重图（用于重叠区域的加权平均）
    if squeeze_batch:
        output = torch.zeros(C, H, W, device=patches.device, dtype=patches.dtype)
        weight_map = torch.zeros(1, H, W, device=patches.device, dtype=patches.dtype)
    else:
        # 假设只有一个batch
        output = torch.zeros(C, H, W, device=patches.device, dtype=patches.dtype)
        weight_map = torch.zeros(1, H, W, device=patches.device, dtype=patches.dtype)
    
    # 创建权重窗口（用于重叠区域的平滑融合）
    weight_window = torch.ones(patch_size, patch_size, device=patches.device, dtype=patches.dtype)
    # 在边界区域使用线性衰减权重
    decay_size = overlap // 2
    for i in range(patch_size):
        for j in range(patch_size):
            weight = 1.0
            if i < decay_size:
                weight *= (i / decay_size) if decay_size > 0 else 1.0
            elif i >= patch_size - decay_size:
                weight *= ((patch_size - i) / decay_size) if decay_size > 0 else 1.0
            if j < decay_size:
                weight *= (j / decay_size) if decay_size > 0 else 1.0
            elif j >= patch_size - decay_size:
                weight *= ((patch_size - j) / decay_size) if decay_size > 0 else 1.0
            weight_window[i, j] = weight
    
    # 合并patches
    for idx, pos in enumerate(positions):
        patch = patches[idx]  # [C, patch_size, patch_size]
        y, x = pos['y'], pos['x']
        y_end, x_end = pos['y_end'], pos['x_end']
        h, w = pos['h'], pos['w']
        
        # 裁剪patch到实际大小
        patch_crop = patch[:, :h, :w]
        weight_crop = weight_window[:h, :w]
        
        # 加权累加
        output[:, y:y_end, x:x_end] += patch_crop * weight_crop.unsqueeze(0)
        weight_map[:, y:y_end, x:x_end] += weight_crop
    
    # 归一化（避免除零）
    weight_map = torch.clamp(weight_map, min=1e-8)
    output = output / weight_map
    
    if output.dim() == 3:
        return output.unsqueeze(0)  # [1, C, H, W]
    return output


def split_video_v2(
    video: torch.Tensor,
    patch_size: int = 128,
    overlap: int = 32
) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    """
    将视频分割成重叠的patches（逐帧处理）
    
    Args:
        video: 输入视频 [T, C, H, W] 或 [B, T, C, H, W]
        patch_size: patch大小
        overlap: 重叠区域大小
    
    Returns:
        patches_list: 每帧的patches列表，每个元素是 [N, C, patch_size, patch_size]
        meta_list: 每帧的元信息列表
    """
    if video.dim() == 5:
        video = video[0]  # [B, T, C, H, W] -> [T, C, H, W]（batch_size=1）
        squeeze_batch = True
    else:
        squeeze_batch = False
    
    T, C, H, W = video.shape
    
    patches_list = []
    meta_list = []
    
    # 对每一帧独立进行patch分割
    for t in range(T):
        frame = video[t]  # [C, H, W]
        patches, meta = split_image_v2(frame, patch_size, overlap)
        patches_list.append(patches)
        
        # 更新meta，添加时间维度信息
        meta['frame_idx'] = t
        meta['total_frames'] = T
        meta_list.append(meta)
    
    return patches_list, meta_list


def merge_video_v2(
    patches_list: List[torch.Tensor],
    meta_list: List[Dict[str, Any]]
) -> torch.Tensor:
    """
    将patches合并回原始视频（逐帧处理）
    
    Args:
        patches_list: 每帧的patches列表，每个元素是 [N, C, patch_size, patch_size]
        meta_list: 每帧的元信息列表
    
    Returns:
        video: 重建的视频 [T, C, H, W] 或 [1, T, C, H, W]
    """
    if not patches_list or not meta_list:
        raise ValueError("patches_list 和 meta_list 不能为空")
    
    T = len(patches_list)
    reconstructed_frames = []
    
    # 对每一帧独立进行patch合并
    for t in range(T):
        patches = patches_list[t]
        meta = meta_list[t]
        frame = merge_image_v2(patches, meta)
        if frame.dim() == 4:
            frame = frame.squeeze(0)  # [C, H, W]
        reconstructed_frames.append(frame)
    
    # 堆叠所有帧
    video = torch.stack(reconstructed_frames, dim=0)  # [T, C, H, W]
    
    # 如果需要，添加batch维度
    if meta_list[0].get('squeeze_batch', False):
        video = video.unsqueeze(0)  # [1, T, C, H, W]
    
    return video

