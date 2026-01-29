"""
多模态JSCC评估脚本

在多个SNR和Rate下评估模型性能，支持任意尺寸图像的patch-based推理。
参考 inference_config.py 和 inference_one.py 的评估框架。
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional, List, Tuple, Any
import argparse
import logging
import inspect
from datetime import datetime
from prettytable import PrettyTable
import json

# 导入模型和工具
from multimodal_jscc import MultimodalJSCC
from metrics import calculate_multimodal_metrics
from data_loader import MultimodalDataLoader, MultimodalDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# 从本地模块导入配置和工具函数
from config import EvaluationConfig
from utils import (
    AverageMeter, seed_torch, logger_configuration, makedirs, load_manifest,
    split_image_v2, merge_image_v2, split_video_v2, merge_video_v2
)


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EvaluationDataset(Dataset):
    """
    专门用于评估的数据集类
    
    与 MultimodalDataset 的区别：
    - 不使用 transforms.Resize，保留原始图像尺寸
    - 只进行 ToTensor 和 Normalize 操作
    - 用于支持 patch-based 推理处理任意尺寸图像
    """
    
    def __init__(
        self,
        data_dir: str,
        data_list: list,
        text_tokenizer=None,
        max_text_length: int = 512,
        max_video_frames: int = 10,
        normalize: bool = False
    ):
        self.data_dir = data_dir
        self.data_list = data_list
        self.text_tokenizer = text_tokenizer
        self.max_text_length = max_text_length
        self.max_video_frames = max_video_frames
        self.normalize = normalize
        
        # 图像变换：不使用 Resize，只进行 ToTensor 和 Normalize
        # 注意：保持 ImageNet 归一化，用于与训练时一致
        image_steps = [transforms.ToTensor()]
        if self.normalize:
            image_steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.image_transform = transforms.Compose(image_steps)
        
        # 视频变换：同样不使用 Resize
        video_steps = [transforms.ToTensor()]
        if self.normalize:
            video_steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
        self.video_transform = transforms.Compose(video_steps)
    
    def __len__(self) -> int:
        return len(self.data_list)
    
    def __getitem__(self, idx: int) -> dict:
        """获取单个样本，保持原始图像尺寸"""
        item = self.data_list[idx]
        sample = {}
        
        # 文本数据（与 MultimodalDataset 相同）
        if 'text' in item:
            text_data = self._load_text(item['text'])
            if text_data is not None:
                sample['text'] = text_data
        
        # 图像数据（不使用 Resize）
        if 'image' in item:
            image_data = self._load_image(item['image'])
            if image_data is not None:
                sample['image'] = image_data
        
        # 视频数据（不使用 Resize）
        if 'video' in item:
            video_data = self._load_video(item['video'])
            if video_data is not None:
                sample['video'] = video_data
        
        return sample
    
    def _load_text(self, text_info: dict):
        """加载文本数据（与 MultimodalDataset 相同）"""
        try:
            if 'texts' in text_info:
                texts = text_info['texts']
                if not texts:
                    raise ValueError("text.texts 为空")
                text = texts[0]
            elif 'file' in text_info:
                text_path = os.path.join(self.data_dir, text_info['file'])
                with open(text_path, 'r', encoding='utf-8') as f:
                    text = f.read().strip()
            else:
                text = text_info['text']
            
            if self.text_tokenizer:
                tokens = self.text_tokenizer(
                    text,
                    max_length=self.max_text_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                input_ids = tokens["input_ids"].squeeze(0)
                attention_mask = tokens["attention_mask"].squeeze(0)
            else:
                encoded = [ord(c) for c in text[: self.max_text_length]]
                input_ids = torch.tensor(encoded, dtype=torch.long)
                pad_len = self.max_text_length - input_ids.shape[0]
                if pad_len > 0:
                    input_ids = torch.cat([input_ids, torch.zeros(pad_len, dtype=torch.long)], dim=0)
                attention_mask = torch.zeros(self.max_text_length, dtype=torch.long)
                attention_mask[: len(encoded)] = 1
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
        except Exception as e:
            print(f"加载文本数据时出错: {e}")
            return None
    
    def _load_image(self, image_info: dict):
        """加载图像数据（不使用 Resize，保持原始尺寸）"""
        try:
            if 'files' in image_info:
                image_path = image_info['files'][0]
            elif 'file' in image_info:
                image_path = image_info['file']
            else:
                image_array = image_info['array']
                image = Image.fromarray(image_array).convert('RGB')
                return self.image_transform(image)
            image_path_full = os.path.join(self.data_dir, image_path)
            image = Image.open(image_path_full).convert('RGB')
            
            # 只进行 ToTensor 和 Normalize，不进行 Resize
            image_tensor = self.image_transform(image)
            return image_tensor
        except Exception as e:
            print(f"加载图像数据时出错: {e}")
            return None
    
    def _load_video(self, video_info: dict):
        """加载视频数据（不使用 Resize），支持 v2/v1"""
        try:
            video_path = video_info.get("file")
            if not video_path:
                raise FileNotFoundError("video.file 缺失")
            full_path = os.path.join(self.data_dir, video_path)
            import cv2
            cap = cv2.VideoCapture(full_path)
            if not cap.isOpened():
                raise FileNotFoundError(f"无法打开视频: {full_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                total_frames = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    total_frames += 1
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            sample_count = min(total_frames, self.max_video_frames)
            indices = (
                np.linspace(0, total_frames - 1, num=sample_count, dtype=int).tolist()
                if total_frames > 0
                else [0] * self.max_video_frames
            )
            frames: List[Image.Image] = []
            current_idx = 0
            target_ptr = 0
            while target_ptr < len(indices):
                ret, frame = cap.read()
                if not ret:
                    break
                if current_idx == indices[target_ptr]:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(Image.fromarray(frame_rgb))
                    target_ptr += 1
                current_idx += 1
            cap.release()
            if not frames:
                raise RuntimeError(f"未能读取任何帧: {full_path}")
            while len(frames) < self.max_video_frames:
                frames.append(frames[-1].copy())
            video_tensor = torch.stack([self.video_transform(frame) for frame in frames[: self.max_video_frames]])
            frame_mask = torch.zeros(self.max_video_frames, dtype=torch.float32)
            frame_mask[: min(sample_count, self.max_video_frames)] = 1.0
            return {"video": video_tensor, "video_frame_mask": frame_mask}
        except Exception as e:
            print(f"加载视频数据时出错: {e}")
            return None
    
    def _extract_frames_from_video(self, video_path: str):
        """从视频文件提取帧（与 MultimodalDataset 相同）"""
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append(frame_pil)
        
        cap.release()
        return frames


def collate_evaluation_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    评估专用的批次整理函数
    
    与 collate_multimodal_batch 的区别：
    - 图像不进行 stack，而是返回列表（因为尺寸可能不同）
    - 主要用于 batch_size=1 的情况
    - 支持 patch-based 推理所需的任意尺寸图像
    """
    inputs = {}
    targets = {}
    
    # 收集所有模态的数据
    modalities = set()
    for sample in batch:
        modalities.update(sample.keys())
    
    # 处理每个模态
    for modality in modalities:
        modality_data = []
        valid_samples = []
        
        # 收集有效样本
        for i, sample in enumerate(batch):
            if modality in sample and sample[modality] is not None:
                modality_data.append(sample[modality])
                valid_samples.append(i)
        
        if not modality_data:
            continue
        
        if modality == 'text':
            # 文本数据：可以堆叠（填充到相同长度）
            input_ids = [item['input_ids'] for item in modality_data]
            attention_mask = [item['attention_mask'] for item in modality_data]
            
            max_len = max(len(ids) for ids in input_ids)
            padded_input_ids = []
            padded_attention_mask = []
            
            for ids, mask in zip(input_ids, attention_mask):
                pad_len = max_len - len(ids)
                padded_input_ids.append(torch.cat([ids, torch.zeros(pad_len, dtype=ids.dtype)]))
                padded_attention_mask.append(torch.cat([mask, torch.zeros(pad_len, dtype=mask.dtype)]))
            
            # 为缺失的样本创建零填充
            full_batch_input_ids = []
            full_batch_attention_mask = []
            
            for i in range(len(batch)):
                if i in valid_samples:
                    idx = valid_samples.index(i)
                    full_batch_input_ids.append(padded_input_ids[idx])
                    full_batch_attention_mask.append(padded_attention_mask[idx])
                else:
                    full_batch_input_ids.append(torch.zeros(max_len, dtype=torch.long))
                    full_batch_attention_mask.append(torch.zeros(max_len, dtype=torch.long))
            
            inputs['text_input'] = torch.stack(full_batch_input_ids)
            inputs['text_attention_mask'] = torch.stack(full_batch_attention_mask)
            targets['text'] = inputs['text_input']
        
        elif modality == 'image':
            # 图像数据：不进行 stack，返回列表（因为尺寸可能不同）
            # 对于 batch_size=1 的情况，直接使用第一个元素
            if len(modality_data) == 1 and len(batch) == 1:
                # 单个样本，直接返回
                inputs['image_input'] = modality_data[0].unsqueeze(0)  # [1, C, H, W]
                targets['image'] = inputs['image_input']
            else:
                # 多个样本：由于尺寸不同，不能直接 stack
                # 这里应该只处理 batch_size=1 的情况
                # 如果 batch_size > 1，需要特殊处理或报错
                if len(batch) > 1:
                    raise ValueError(
                        f"评估模式不支持 batch_size > 1（因为图像尺寸不同）。"
                        f"当前 batch_size={len(batch)}"
                    )
                inputs['image_input'] = modality_data[0].unsqueeze(0)
                targets['image'] = inputs['image_input']
        
        elif modality == 'video':
            # 视频数据：可以堆叠（假设帧数相同）
            videos = []
            masks = []
            for item in modality_data:
                if isinstance(item, dict):
                    videos.append(item["video"])
                    masks.append(item.get("video_frame_mask", torch.ones(item["video"].shape[0])))
                else:
                    videos.append(item)
                    masks.append(torch.ones(item.shape[0]))
            videos = torch.stack(videos)
            masks = torch.stack(masks)
            full_batch_videos = []
            full_batch_masks = []
            
            for i in range(len(batch)):
                if i in valid_samples:
                    idx = valid_samples.index(i)
                    full_batch_videos.append(videos[idx])
                    full_batch_masks.append(masks[idx])
                else:
                    zero_video = torch.zeros_like(videos[0])
                    full_batch_videos.append(zero_video)
                    full_batch_masks.append(torch.zeros_like(masks[0]))
            
            inputs['video_input'] = torch.stack(full_batch_videos)
            targets['video'] = inputs['video_input']
            inputs['video_frame_mask'] = torch.stack(full_batch_masks)
    
    # 构建最终批次
    batch_data = {
        'inputs': inputs,
        'targets': targets
    }
    
    if 'text_attention_mask' in inputs:
        batch_data['attention_mask'] = inputs['text_attention_mask']
    
    return batch_data




def evaluate_single_snr(
    model: MultimodalJSCC,
    test_loader: DataLoader,
    config: EvaluationConfig,
    snr_db: float,
    logger: logging.Logger,
    device: torch.device
) -> Dict[str, float]:
    """
    在单个SNR下进行评估
    
    Returns:
        Dict[str, float]: 评估指标字典
    """
    model.eval()
    
    # 设置模型SNR
    model.set_snr(snr_db)
    
    # 按样本累计指标，避免跨样本拼接不同尺寸的张量
    metrics_accumulator: Dict[str, List[float]] = {}
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # 检查是否达到测试样本数量限制
            if config.test_num is not None and sample_count >= config.test_num:
                break
            
            # 从batch中提取数据
            inputs = batch['inputs']
            targets = batch['targets']
            attention_mask = batch.get('attention_mask', None)
            
            # 提取各模态输入
            text_input = inputs.get('text_input', None)
            image_input = inputs.get('image_input', None)
            video_input = inputs.get('video_input', None)
            
            # 移动到设备
            if text_input is not None:
                text_input = text_input.to(device)
                attention_mask = attention_mask.to(device) if attention_mask is not None else None
            if image_input is not None:
                image_input = image_input.to(device)
            if video_input is not None:
                video_input = video_input.to(device)
            
            # 移动targets到设备
            device_targets = {}
            for key, value in targets.items():
                if value is not None:
                    device_targets[key] = value.to(device)
            
            # 初始化结果字典
            results = {}
            
            # 【修复】预先编码文本以获取语义上下文（semantic_context）
            # 这对于图像和视频的语义引导解码至关重要
            semantic_context = None
            if text_input is not None:
                # 只编码文本以获取语义上下文，不进行完整的前向传播
                text_encoded_results = model.encode(
                    text_input=text_input,
                    text_attention_mask=attention_mask
                )
                if 'text_encoded' in text_encoded_results:
                    semantic_context = text_encoded_results['text_encoded']
                    logger.debug(f"获取语义上下文: shape={semantic_context.shape}")
            else:
                logger.warning("警告：当前样本没有文本输入，图像和视频解码将无法使用语义引导")
            
            # 第一步：处理视频（如果需要patch推理）
            if video_input is not None and config.use_patch_inference:
                # 检查视频帧尺寸是否一致（固定尺寸不需要patch推理）
                B, T, C, H, W = video_input.shape
                
                # 获取模型期望的视频帧尺寸（通常与图像尺寸相同）
                expected_h, expected_w = (224, 224)  # 默认值
                if hasattr(model, 'video_encoder') and hasattr(model.video_encoder, 'patch_embed'):
                    if hasattr(model.video_encoder.patch_embed, 'img_size'):
                        expected_h, expected_w = model.video_encoder.patch_embed.img_size
                
                # 判断是否需要patch推理
                need_patch = False
                if B == 1:
                    # 单个视频，检查第一帧的尺寸
                    if H != expected_h or W != expected_w:
                        need_patch = True
                else:
                    # batch处理，检查是否有不同尺寸的视频帧
                    h_set = set()
                    w_set = set()
                    for b in range(B):
                        h_set.add(video_input[b, 0].shape[1])  # 第一帧的H
                        w_set.add(video_input[b, 0].shape[2])  # 第一帧的W
                    if len(h_set) > 1 or len(w_set) > 1 or (h_set.pop() != expected_h or w_set.pop() != expected_w):
                        need_patch = True
                
                if need_patch:
                    # 使用 patch-based 推理处理任意尺寸视频
                    # 注意：由于 batch_size=1，这里 B 总是 1
                    video = video_input[0]  # [T, C, H, W] - 去除 batch 维度
                    
                    logger.debug(f"检测到非标准尺寸视频: {T}帧, {H}x{W}, 使用 patch-based 推理")
                    
                    # 分割视频（逐帧处理）
                    # 与模型期望的输入尺寸对齐，使用 image_encoder/video_encoder 的 img_size 作为 patch_size
                    patch_size = expected_h
                    patches_list, meta_list = split_video_v2(video, patch_size, config.patch_overlap)
                    
                    # 收集所有帧的所有patches（用于批量推理）
                    all_patches = []
                    all_meta = []
                    frame_start_indices = []  # 记录每帧的patches起始索引
                    
                    current_idx = 0
                    for t, (frame_patches, frame_meta) in enumerate(zip(patches_list, meta_list)):
                        frame_start_indices.append(current_idx)
                        num_patches = len(frame_patches)
                        all_patches.append(frame_patches)
                        all_meta.append(frame_meta)
                        current_idx += num_patches
                    frame_start_indices.append(current_idx)  # 最后一个是总patches数
                    
                    # 将所有patches合并为一个batch（所有帧的所有patches）
                    all_patches_tensor = torch.cat(all_patches, dim=0).to(device)  # [Total_Patches, C, patch_size, patch_size]
                    
                    logger.debug(f"视频分割为 {len(all_patches_tensor)} 个 patches（来自 {T} 帧）")
                    
                    # 【修复】使用分阶段处理：编码 -> 信道 -> 解码（传入语义上下文）
                    # 对所有patches进行推理（批量处理以提高效率）
                    patch_results_list = []
                    patch_batch_size = 4  # 减小batch size以减少内存占用
                    
                    for i in range(0, len(all_patches_tensor), patch_batch_size):
                        patch_batch = all_patches_tensor[i:i+patch_batch_size]
                        
                        # 阶段1：编码
                        patch_encoded = model.encode(image_input=patch_batch)
                        patch_encoded_features = patch_encoded['image_encoded']
                        patch_guide_vectors = patch_encoded['image_guide']
                        
                        # 阶段2：功率归一化
                        if 'image' in model.power_normalizer:
                            patch_encoded_features = model.power_normalizer['image'](patch_encoded_features)
                        
                        # 阶段3：信道传输
                        patch_transmitted = model.channel(patch_encoded_features)
                        
                        # 阶段4：解码（传入语义上下文以启用语义引导）
                        patch_decoded = model.decode(
                            transmitted_features={'image': patch_transmitted},
                            guide_vectors={'image': patch_guide_vectors},
                            semantic_context=semantic_context  # 【关键修复】传入语义上下文
                        )
                        # 立即移动到CPU以释放GPU内存
                        patch_results_list.append(patch_decoded['image_decoded'].cpu())
                        
                        # 清理中间变量
                        del patch_encoded, patch_encoded_features, patch_guide_vectors
                        del patch_transmitted, patch_decoded, patch_batch
                        # 每处理几个batch后清理GPU缓存
                        if (i // patch_batch_size + 1) % 5 == 0:
                            torch.cuda.empty_cache()
                    
                    # 合并所有patch结果（数据已在CPU上）
                    all_reconstructed_patches = torch.cat(patch_results_list, dim=0)  # [Total_Patches, C, patch_size, patch_size]
                    # 清理中间变量
                    del patch_results_list, all_patches_tensor
                    torch.cuda.empty_cache()
                    
                    # 按帧重组patches
                    reconstructed_patches_by_frame = []
                    for t in range(T):
                        start_idx = frame_start_indices[t]
                        end_idx = frame_start_indices[t + 1]
                        frame_patches = all_reconstructed_patches[start_idx:end_idx]
                        reconstructed_patches_by_frame.append(frame_patches)
                    
                    # 逐帧合并，重建视频
                    reconstructed_video = merge_video_v2(reconstructed_patches_by_frame, meta_list)
                    # merge_video_v2 already restores the batch dimension when needed,
                    # so avoid an extra unsqueeze that would introduce a spurious dimension.
                    reconstructed_video = reconstructed_video.to(device)
                    
                    # 保存视频结果
                    results['video_decoded'] = reconstructed_video
                    # 标记视频已处理，后续不再处理
                    video_input = None
            
            # 第二步：处理图像（如果需要patch推理）
            if image_input is not None and config.use_patch_inference:
                # 检查图像尺寸是否一致（固定尺寸不需要patch推理）
                B, C, H, W = image_input.shape
                
                # 获取模型期望的图像尺寸
                expected_h, expected_w = (224, 224)  # 默认值
                if hasattr(model, 'image_encoder') and hasattr(model.image_encoder, 'patch_embed'):
                    if hasattr(model.image_encoder.patch_embed, 'img_size'):
                        expected_h, expected_w = model.image_encoder.patch_embed.img_size
                
                # 判断是否需要patch推理
                need_patch = False
                if B == 1:
                    # 单个图像，检查尺寸
                    if H != expected_h or W != expected_w:
                        need_patch = True
                else:
                    # batch处理，检查是否有不同尺寸的图像
                    h_set = set()
                    w_set = set()
                    for i in range(B):
                        h_set.add(image_input[i].shape[1])
                        w_set.add(image_input[i].shape[2])
                    if len(h_set) > 1 or len(w_set) > 1 or (h_set.pop() != expected_h or w_set.pop() != expected_w):
                        need_patch = True
                
                if need_patch:
                    # 使用 patch-based 推理处理任意尺寸图像
                    # 注意：由于 batch_size=1，这里 B 总是 1
                    img = image_input[0]  # [C, H, W] - 去除 batch 维度
                    
                    # img 维度为 [C, H, W]，索引应使用 1/2 分别表示高和宽
                    logger.debug(f"检测到非标准尺寸图像: {img.shape[1]}x{img.shape[2]}, 使用 patch-based 推理")
                    
                    # 分割图像
                    # 与模型期望的输入尺寸对齐，使用 image_encoder 的 img_size 作为 patch_size
                    patch_size = expected_h
                    patches, meta = split_image_v2(img, patch_size, config.patch_overlap)
                    patches = patches.to(device)
                    
                    logger.debug(f"图像分割为 {len(patches)} 个 patches")
                    
                    # 【修复】使用分阶段处理：编码 -> 信道 -> 解码（传入语义上下文）
                    # 对每个patch进行推理（使用较小的batch_size以提高效率）
                    patch_results_list = []
                    patch_batch_size = 4  # 减小batch size以减少内存占用
                    
                    for i in range(0, len(patches), patch_batch_size):
                        patch_batch = patches[i:i+patch_batch_size]
                        
                        # 阶段1：编码
                        patch_encoded = model.encode(image_input=patch_batch)
                        patch_encoded_features = patch_encoded['image_encoded']
                        patch_guide_vectors = patch_encoded['image_guide']
                        
                        # 阶段2：功率归一化
                        if 'image' in model.power_normalizer:
                            patch_encoded_features = model.power_normalizer['image'](patch_encoded_features)
                        
                        # 阶段3：信道传输
                        patch_transmitted = model.channel(patch_encoded_features)
                        
                        # 阶段4：解码（传入语义上下文以启用语义引导）
                        patch_decoded = model.decode(
                            transmitted_features={'image': patch_transmitted},
                            guide_vectors={'image': patch_guide_vectors},
                            semantic_context=semantic_context  # 【关键修复】传入语义上下文
                        )
                        # 立即移动到CPU以释放GPU内存
                        patch_results_list.append(patch_decoded['image_decoded'].cpu())
                        
                        # 清理中间变量
                        del patch_encoded, patch_encoded_features, patch_guide_vectors
                        del patch_transmitted, patch_decoded, patch_batch
                        # 每处理几个batch后清理GPU缓存
                        if (i // patch_batch_size + 1) % 5 == 0:
                            torch.cuda.empty_cache()
                    
                    # 合并所有patch结果（数据已在CPU上）
                    all_patches = torch.cat(patch_results_list, dim=0)
                    # 清理中间变量
                    del patch_results_list, patches
                    torch.cuda.empty_cache()
                    reconstructed_image = merge_image_v2(all_patches, meta)
                    reconstructed_image = reconstructed_image.unsqueeze(0).to(device)  # [1, C, H, W]
                    
                    # 保存图像结果
                    results['image_decoded'] = reconstructed_image
                    # 标记图像已处理，后续不再处理
                    image_input = None
            
            # 第三步：统一处理所有模态（包括标准尺寸的模态和未通过patch处理的模态）
            # 如果所有模态都已通过patch处理，这里只处理文本
            remaining_inputs = {}
            if text_input is not None:
                remaining_inputs['text_input'] = text_input
            if image_input is not None:
                remaining_inputs['image_input'] = image_input
            if video_input is not None:
                remaining_inputs['video_input'] = video_input
            
            # 如果有未处理的模态，进行一次统一推理
            if remaining_inputs:
                final_results = model(
                    text_input=remaining_inputs.get('text_input', None),
                    image_input=remaining_inputs.get('image_input', None),
                    video_input=remaining_inputs.get('video_input', None),
                    text_attention_mask=attention_mask,
                    snr_db=snr_db
                )
                # 合并结果（不覆盖已有的patch处理结果）
                for key, value in final_results.items():
                    if key.endswith('_decoded') and key not in results:
                        results[key] = value
            
            # 收集预测和目标
            current_metrics = {}
            try:
                current_metrics = calculate_multimodal_metrics(
                    predictions=results,
                    targets=device_targets,
                    attention_mask=attention_mask,
                    imagenet_normalized=getattr(config, "normalize", False),
                )
            except Exception as e:
                logger.error(f"计算当前批次评估指标时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            for metric_name, metric_value in current_metrics.items():
                metrics_accumulator.setdefault(metric_name, []).append(metric_value)
            
            # 更新样本计数
            # 注意：image_input和video_input可能在patch处理后被设置为None
            original_image_input = inputs.get('image_input', None)
            original_video_input = inputs.get('video_input', None)
            
            if original_image_input is not None:
                batch_size = original_image_input.shape[0]
            elif original_video_input is not None:
                batch_size = original_video_input.shape[0]
            elif text_input is not None:
                batch_size = text_input.shape[0]
            else:
                batch_size = 1
            
            sample_count += batch_size
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"处理进度: {batch_idx + 1}/{len(test_loader)} batches, {sample_count} 样本")
    
    # 计算所有样本的平均指标
    if not metrics_accumulator:
        logger.warning("未累积到任何评估指标，返回空结果。可能数据集中缺少可用样本。") 
        return {}
    
    averaged_metrics = {}
    for metric_name, values in metrics_accumulator.items():
        if len(values) == 0:
            continue
        averaged_metrics[metric_name] = float(torch.tensor(values).mean().item())
    
    return averaged_metrics


def print_results_table(
    results: List[Dict[str, float]],
    snr_list: List[float],
    logger: logging.Logger
):
    """
    使用PrettyTable打印结果表格
    """
    if not results:
        logger.warning("没有结果可展示")
        return
    
    # 获取所有指标名称
    all_metrics = set()
    for result in results:
        all_metrics.update(result.keys())
    
    # 移除snr_db
    all_metrics.discard('snr_db')
    all_metrics = sorted(all_metrics)
    
    # 为每个指标创建表格
    for metric_name in all_metrics:
        if metric_name in ['image_psnr', 'image_ssim', 'video_psnr_mean', 'video_ssim_mean', 
                          'text_bleu', 'rouge_l_f1']:
            table = PrettyTable()
            table.field_names = ['SNR (dB)'] + [metric_name]
            
            for snr, result in zip(snr_list, results):
                value = result.get(metric_name, 0.0)
                if isinstance(value, (int, float)):
                    table.add_row([f'{snr:.1f}', f'{value:.4f}'])
            
            logger.info(f"\n{metric_name} 性能:")
            logger.info(f"\n{table}")
    
    # 创建综合表格（包含主要指标）
    main_table = PrettyTable()
    main_metrics = ['image_psnr', 'image_ssim', 'video_psnr_mean', 'text_bleu']
    available_metrics = [m for m in main_metrics if any(m in r for r in results)]
    
    if available_metrics:
        main_table.field_names = ['SNR (dB)'] + available_metrics
        
        for snr, result in zip(snr_list, results):
            row = [f'{snr:.1f}']
            for metric in available_metrics:
                if metric in result:
                    row.append(f'{result[metric]:.4f}')
                else:
                    row.append('N/A')
            main_table.add_row(row)
        
        logger.info(f"\n综合性能表:")
        logger.info(f"\n{main_table}")


def main():
    parser = argparse.ArgumentParser(description='多模态JSCC评估脚本')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    parser.add_argument('--test-manifest', type=str, default=None, help='测试数据清单路径（相对于data-dir）')
    parser.add_argument('--snr-list', type=str, default=None, 
                       help='SNR列表，逗号分隔，例如：-5,0,5,10,15')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    parser.add_argument('--test-num', type=int, default=None, help='测试样本数量')
    parser.add_argument('--no-patch', action='store_true', help='禁用patch-based推理')
    parser.add_argument('--save-images', action='store_true', help='保存重建图像')
    parser.add_argument(
        '--image-decoder-type',
        type=str,
        choices=['baseline', 'generative'],
        default=None,
        help='图像解码器类型（覆盖checkpoint配置）',
    )
    parser.add_argument('--generator-type', type=str, default=None, help='生成器类型（默认vae）')
    parser.add_argument('--generator-ckpt', type=str, default=None, help='生成器权重路径')
    parser.add_argument('--z-channels', type=int, default=None, help='生成器latent通道数')
    parser.add_argument('--latent-down', type=int, default=None, help='生成器latent下采样倍率')
    args = parser.parse_args()
    
    # 设置随机种子
    seed_torch(42)
    
    # 创建配置
    config = EvaluationConfig()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.model_path = args.model_path
    config.data_dir = args.data_dir
    
    # 设置manifest路径
    if args.test_manifest:
        config.test_manifest = os.path.join(config.data_dir, args.test_manifest)
    else:
        default_test_v2 = os.path.join(config.data_dir, "test_manifest_v2.json")
        default_val_v2 = os.path.join(config.data_dir, "val_manifest_v2.json")
        if os.path.exists(default_test_v2):
            config.test_manifest = default_test_v2
        elif os.path.exists(default_val_v2):
            config.test_manifest = default_val_v2
        else:
            config.test_manifest = os.path.join(config.data_dir, 'test_manifest.json')
            # 如果没有test_manifest，尝试使用val_manifest
            if not os.path.exists(config.test_manifest):
                config.test_manifest = os.path.join(config.data_dir, 'val_manifest.json')
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.test_num:
        config.test_num = args.test_num
    
    if args.no_patch:
        config.use_patch_inference = False
    
    if args.save_images:
        config.save_images = True
        config.image_save_dir = os.path.join(config.result_dir, 'reconstructed_images')
    if args.image_decoder_type is not None:
        config.image_decoder_type = args.image_decoder_type
    if args.generator_type is not None:
        config.generator_type = args.generator_type
    if args.generator_ckpt is not None:
        config.generator_ckpt = args.generator_ckpt
    if args.z_channels is not None:
        config.z_channels = args.z_channels
    if args.latent_down is not None:
        config.latent_down = args.latent_down
    
    if args.snr_list:
        config.snr_list = [float(x.strip()) for x in args.snr_list.split(',')]
    
    # 创建目录
    makedirs(config.log_dir)
    makedirs(config.result_dir)
    if config.save_images:
        makedirs(config.image_save_dir)
    
    # 配置日志
    log_config = type('LogConfig', (), {
        'workdir': config.log_dir,
        'log': config.log_file,
        'samples': os.path.join(config.result_dir, 'samples'),
        'models': os.path.join(config.result_dir, 'models')
    })()
    logger = logger_configuration(log_config, save_log=True)
    
    logger.info("=" * 80)
    logger.info("多模态JSCC评估脚本")
    logger.info("=" * 80)
    logger.info(f"运行名称: {config.run_name}")
    logger.info(f"设备: {config.device}")
    logger.info(f"模型路径: {config.model_path}")
    logger.info(f"数据目录: {config.data_dir}")
    logger.info(f"测试清单: {config.test_manifest}")
    logger.info(f"SNR列表: {config.snr_list}")
    
    # 加载模型
    logger.info("加载模型...")
    try:
        # 加载检查点
        checkpoint = torch.load(config.model_path, map_location=config.device)
        
        # 尝试从检查点中恢复模型配置（如果存在）
        model_kwargs = {
            'vocab_size': config.vocab_size,
            'text_embed_dim': config.text_embed_dim,
            'text_num_heads': config.text_num_heads,
            'text_num_layers': config.text_num_layers,
            'text_output_dim': config.text_output_dim,
            'img_size': config.img_size,
            'patch_size': config.img_patch_size,
            'img_embed_dims': config.img_embed_dims,
            'img_depths': config.img_depths,
            'img_num_heads': config.img_num_heads,
            'img_output_dim': config.img_output_dim,
            'video_hidden_dim': config.video_hidden_dim,
            'video_num_frames': config.video_num_frames,
            'video_use_optical_flow': config.video_use_optical_flow,
            'video_use_convlstm': config.video_use_convlstm,
            'video_output_dim': config.video_output_dim,
            'video_decoder_type': config.video_decoder_type,
            'video_unet_base_channels': config.video_unet_base_channels,
            'video_unet_num_down': config.video_unet_num_down,
            'video_unet_num_res_blocks': config.video_unet_num_res_blocks,
            'video_gop_size': getattr(config, "video_gop_size", None),
            'channel_type': config.channel_type,
            'snr_db': config.snr_db,
            'normalize_inputs': getattr(config, "normalize", False),
            'image_decoder_type': getattr(config, "image_decoder_type", "baseline"),
            'generator_type': getattr(config, "generator_type", "vae"),
            'generator_ckpt': getattr(config, "generator_ckpt", None),
            'z_channels': getattr(config, "z_channels", 4),
            'latent_down': getattr(config, "latent_down", 8),
        }
        
        # 如果检查点中包含模型配置，使用检查点的配置
        if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
            model_kwargs.update(checkpoint['model_config'])
            logger.info("从检查点恢复模型配置")
        if args.image_decoder_type is not None:
            model_kwargs["image_decoder_type"] = config.image_decoder_type
        if args.generator_type is not None:
            model_kwargs["generator_type"] = config.generator_type
        if args.generator_ckpt is not None:
            model_kwargs["generator_ckpt"] = config.generator_ckpt
        if args.z_channels is not None:
            model_kwargs["z_channels"] = config.z_channels
        if args.latent_down is not None:
            model_kwargs["latent_down"] = config.latent_down
        
        # 创建模型（过滤掉当前构造函数不支持的参数）
        try:
            model_param_names = set(inspect.signature(MultimodalJSCC.__init__).parameters.keys())
            model_param_names.discard("self")
            extra_keys = [key for key in model_kwargs if key not in model_param_names]
            if extra_keys:
                logger.warning(
                    "⚠️ Checkpoint 配置中包含未在当前模型构造函数中出现的参数: %s; 将自动忽略。",
                    ", ".join(sorted(extra_keys)),
                )
                for key in extra_keys:
                    model_kwargs.pop(key, None)
        except Exception as e:
            logger.warning("构造参数过滤失败，将继续尝试初始化模型: %s", e)
        model = MultimodalJSCC(**model_kwargs)
        
        # 加载权重
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                logger.info("加载模型权重（从model_state_dict）")
            else:
                # 直接是state_dict
                model.load_state_dict(checkpoint, strict=False)
                logger.info("加载模型权重（直接state_dict）")
        else:
            model.load_state_dict(checkpoint, strict=False)
            logger.info("加载模型权重（检查点本身）")
        
        model = model.to(config.device)
        model.eval()
        
        logger.info("模型加载成功")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"模型总参数: {total_params:,}")
        
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error("请确保模型路径正确，且模型结构与代码匹配")
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 加载测试数据清单
    logger.info("加载测试数据清单...")
    test_data_list = load_manifest(config.test_manifest)
    
    if not test_data_list:
        logger.error(f"测试数据清单为空或文件不存在: {config.test_manifest}")
        return
    
    logger.info(f"测试样本数: {len(test_data_list)}")
    
    # 创建评估专用的数据加载器
    logger.info("创建评估数据加载器...")
    logger.info("注意: 使用 EvaluationDataset 保持原始图像尺寸，支持 patch-based 推理")
    
    # 创建评估数据集（不使用 Resize）
    test_dataset = EvaluationDataset(
        data_dir=config.data_dir,
        data_list=test_data_list,
        text_tokenizer=None,  # 可以根据需要传入 tokenizer
        max_text_length=config.max_text_length,
        max_video_frames=config.max_video_frames,
        normalize=getattr(config, "normalize", False),
    )
    
    # 评估时必须使用 batch_size=1，因为不同尺寸的图像无法堆叠
    eval_batch_size = 1
    if config.batch_size != 1:
        logger.warning(
            f"评估模式建议使用 batch_size=1（因为图像尺寸可能不同）。"
            f"当前配置为 {config.batch_size}，将强制设置为 1"
        )
        eval_batch_size = 1
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_evaluation_batch,  # 使用评估专用的 collate 函数
        pin_memory=config.pin_memory
    )
    
    logger.info(f"数据加载器创建完成: batch_size={eval_batch_size}, 样本数={len(test_dataset)}")
    
    # 多SNR评估循环
    logger.info("开始多SNR评估...")
    logger.info("=" * 80)
    
    all_results = []
    
    for snr_idx, snr_db in enumerate(config.snr_list):
        logger.info(f"\n评估SNR: {snr_db} dB ({snr_idx + 1}/{len(config.snr_list)})")
        
        try:
            metrics = evaluate_single_snr(
                model=model,
                test_loader=test_loader,
                config=config,
                snr_db=snr_db,
                logger=logger,
                device=config.device
            )
            
            # 添加SNR信息
            metrics['snr_db'] = snr_db
            all_results.append(metrics)
            
            # 打印当前SNR的结果
            logger.info(f"SNR={snr_db}dB 评估结果:")
            if 'image_psnr' in metrics:
                logger.info(f"  图像 PSNR: {metrics['image_psnr']:.2f} dB")
            if 'image_ssim' in metrics:
                logger.info(f"  图像 SSIM: {metrics['image_ssim']:.4f}")
            if 'video_psnr_mean' in metrics:
                logger.info(f"  视频平均 PSNR: {metrics['video_psnr_mean']:.2f} dB")
            if 'video_ssim_mean' in metrics:
                logger.info(f"  视频平均 SSIM: {metrics['video_ssim_mean']:.4f}")
            if 'text_bleu' in metrics:
                logger.info(f"  文本 BLEU: {metrics['text_bleu']:.4f}")
            if 'rouge_l_f1' in metrics:
                logger.info(f"  文本 ROUGE-L F1: {metrics['rouge_l_f1']:.4f}")
        
        except Exception as e:
            logger.error(f"评估SNR={snr_db}dB时出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue
    
    # 打印综合结果表格
    logger.info("\n" + "=" * 80)
    logger.info("评估结果汇总")
    logger.info("=" * 80)
    print_results_table(all_results, config.snr_list, logger)
    
    # 保存结果到文件
    results_file = os.path.join(config.result_dir, 'evaluation_results.npz')
    
    # 将结果转换为可序列化的格式
    results_dict = {}
    for i, result in enumerate(all_results):
        for key, value in result.items():
            if key not in results_dict:
                results_dict[key] = []
            results_dict[key].append(value if isinstance(value, (int, float)) else float(value))
    
    np.savez(results_file, 
             snr_list=np.array(config.snr_list),
             **results_dict)
    logger.info(f"\n结果已保存到: {results_file}")
    
    logger.info("=" * 80)
    logger.info("评估完成！")


if __name__ == '__main__':
    main()




