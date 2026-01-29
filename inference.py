"""
多模态JSCC推理脚本

用于对单个文件（图像、文本、视频）进行推理和演示。
支持 patch-based 推理处理任意尺寸的输入。
"""

import os
import sys
import torch
import argparse
import logging
import inspect
from pathlib import Path
from typing import Optional, Dict, Any, List,Tuple
from PIL import Image
import numpy as np
import math

# 导入模型和工具
from multimodal_jscc import MultimodalJSCC
from config import EvaluationConfig
from utils import (
    seed_torch, logger_configuration, makedirs,
    split_image_v2, merge_image_v2
)
from torchvision import transforms
from utils_check import print_model_structure_info, check_state_dict_compatibility
import cv2
import torch.nn.functional as F

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def pad_image(image: torch.Tensor, patch_size: int) -> Tuple[torch.Tensor, int, int]:
    if image.dim() == 4:
        _, _, H, W = image.shape
    else:
        _, H, W = image.shape
    #_, _, H, W = image.shape
    pad_h = (patch_size - (H % patch_size)) % patch_size
    pad_w = (patch_size - (W % patch_size)) % patch_size
    padded_image = F.pad(image, (0, pad_w, 0, pad_h), mode='replicate')
    return padded_image, pad_h, pad_w
def crop_image(image: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if image.dim() == 5:
        _, _, _, H, W = image.shape
        return image[:, :, :, :H - pad_h, :W - pad_w]
    if image.dim() == 4:
        _, _, H, W = image.shape
        return image[:, :, :H - pad_h, :W - pad_w]
    if image.dim() == 3:
        _, H, W = image.shape
        return image[:, :H - pad_h, :W - pad_w]
    raise ValueError(f"Unsupported image shape for cropping: {image.shape}")

def load_image(image_path: str, normalize: bool = False) -> torch.Tensor:
    """
    加载图像并转换为tensor
    
    Args:
        image_path: 图像文件路径
        normalize: 是否使用ImageNet归一化
    
    Returns:
        torch.Tensor: [C, H, W] 图像
    """
    image = Image.open(image_path).convert('RGB')
    
    transform_steps = [transforms.ToTensor()]
    if normalize:
        # ImageNet归一化
        transform_steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transform = transforms.Compose(transform_steps)
    
    image_tensor = transform(image)
    return image_tensor


def load_text(text_path: str) -> str:
    """
    加载文本文件
    
    Args:
        text_path: 文本文件路径
    
    Returns:
        str: 文本内容
    """
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_video(video_path: str, max_frames: Optional[int] = None, normalize: bool = False) -> torch.Tensor:
    """
    加载视频并转换为tensor
    
    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数，None 表示读取全帧
        normalize: 是否使用ImageNet归一化
    
    Returns:
        torch.Tensor: [T, C, H, W] 视频帧
    """
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
    
    # 限制帧数
    if max_frames is not None and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]
    
    transform_steps = [transforms.ToTensor()]
    if normalize:
        # ImageNet归一化
        transform_steps.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    transform = transforms.Compose(transform_steps)
    
    video_tensor = torch.stack([transform(frame) for frame in frames])
    return video_tensor


def denormalize_image(image_tensor: torch.Tensor, normalized: bool = True) -> np.ndarray:
    """
    将ImageNet归一化的图像tensor转换回可显示的图像
    
    Args:
        image_tensor: [C, H, W] 或 [B, C, H, W]
    
    Returns:
        np.ndarray: [H, W, C]，uint8格式
    """
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]  # [B, C, H, W] -> [C, H, W]
    if normalized:
        mean = torch.tensor(IMAGENET_MEAN, device=image_tensor.device).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=image_tensor.device).view(3, 1, 1)
        image_tensor = image_tensor * std + mean
    
    image_tensor = torch.clamp(image_tensor, 0, 1)
    image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return image_np


def denormalize_video_frame(frame_tensor: torch.Tensor, normalized: bool = True) -> np.ndarray:
    frame = denormalize_image(frame_tensor, normalized=normalized)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def get_video_fps(video_path: str, default_fps: float) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return default_fps
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps and fps > 0:
        return fps
    return default_fps


def get_video_meta(video_path: str) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "width": 0,
            "height": 0,
            "fps": 0.0,
            "frame_count": 0,
            "duration": 0.0,
        }
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frame_count / fps if fps and fps > 0 else 0.0
    return {
        "width": width,
        "height": height,
        "fps": fps if fps else 0.0,
        "frame_count": frame_count,
        "duration": duration,
    }


def resolve_infer_window(config: EvaluationConfig) -> Tuple[int, int]:
    window_len = (
        config.infer_window_len
        if getattr(config, "infer_window_len", None)
        else getattr(config, "video_clip_len", None)
    )
    if not window_len:
        window_len = getattr(config, "max_video_frames", 10)
    stride = (
        config.infer_window_stride
        if getattr(config, "infer_window_stride", None)
        else getattr(config, "video_stride", None)
    )
    if not stride:
        stride = window_len
    if window_len <= 0 or stride <= 0:
        raise ValueError(f"无效滑窗参数: window_len={window_len}, stride={stride}")
    return window_len, stride


def reconstruct_video_clip(
    model: MultimodalJSCC,
    video_clip: torch.Tensor,
    config: EvaluationConfig,
    device: torch.device,
    logger: logging.Logger,
    use_amp: bool,
) -> Tuple[torch.Tensor, str]:
    original_shape = video_clip.shape
    patch_size = getattr(config, "patch_size", None)
    if patch_size is None and hasattr(model, "image_encoder") and hasattr(model.image_encoder, "patch_embed"):
        patch_size = getattr(model.image_encoder.patch_embed, "patch_size", 128)
    patch_size = patch_size or 128
    pad_multiple = patch_size
    padded_video, pad_h, pad_w = pad_image(video_clip, pad_multiple)
    need_patch = config.use_patch_inference and (
        original_shape[2] > patch_size or original_shape[3] > patch_size
    )
    if need_patch:
        path_used = "clip-patch-video"
        patches, meta = split_video_clip_patches(padded_video, patch_size, config.patch_overlap)
        patch_results_list = []
        patch_batch_size = 2
        for i in range(0, len(patches), patch_batch_size):
            patch_batch = patches[i : i + patch_batch_size].to(device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                results = model(video_input=patch_batch, snr_db=config.snr_db)
            patch_decoded_video = results["video_decoded"]
            patch_results_list.append(patch_decoded_video.cpu())
            del patch_batch, results, patch_decoded_video
            if (i // patch_batch_size + 1) % 5 == 0:
                torch.cuda.empty_cache()
        reconstructed_patches = torch.cat(patch_results_list, dim=0)
        del patch_results_list, patches
        torch.cuda.empty_cache()
        reconstructed_video = merge_video_clip_patches(reconstructed_patches, meta)
        reconstructed_video = crop_image(reconstructed_video, pad_h, pad_w)
    else:
        path_used = "standard"
        video_input = padded_video.unsqueeze(0).to(device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            results = model(video_input=video_input, snr_db=config.snr_db)
        reconstructed_video = results["video_decoded"]
        if reconstructed_video.dim() == 5:
            reconstructed_video = reconstructed_video.squeeze(0)
        reconstructed_video = crop_image(reconstructed_video, pad_h, pad_w)
    if reconstructed_video.shape != original_shape:
        logger.warning(
            "重建视频尺寸不一致: original=%s reconstructed=%s",
            original_shape,
            reconstructed_video.shape,
        )
    return reconstructed_video, path_used


def sliding_window_reconstruct(
    model: MultimodalJSCC,
    video_full: torch.Tensor,
    config: EvaluationConfig,
    device: torch.device,
    logger: logging.Logger,
    use_amp: bool,
    window_len: int,
    stride: int,
) -> Tuple[torch.Tensor, str]:
    total_frames = video_full.shape[0]
    accum = torch.zeros_like(video_full, dtype=torch.float32, device=device)
    counts = torch.zeros((total_frames, 1, 1, 1), dtype=torch.float32, device=device)
    path_used = "sliding-window"
    blend_mode = getattr(config, "infer_window_blend", "uniform")
    window_weights = build_window_blend_weights(window_len, stride, device, blend_mode)
    for start in range(0, total_frames, stride):
        end = min(start + window_len, total_frames)
        window = video_full[start:end]
        valid_len = window.shape[0]
        if valid_len < window_len:
            pad_frames = window[-1:].repeat(window_len - valid_len, 1, 1, 1)
            window = torch.cat([window, pad_frames], dim=0)
        if hasattr(model, "reset_hidden_states"):
            model.reset_hidden_states()
        reconstructed_window, path_used = reconstruct_video_clip(
            model=model,
            video_clip=window,
            config=config,
            device=device,
            logger=logger,
            use_amp=use_amp,
        )
        reconstructed_window = reconstructed_window[:valid_len].to(device)
        weights = window_weights[:valid_len]
        accum[start:end] += reconstructed_window * weights
        counts[start:end] += weights
    counts = torch.clamp(counts, min=1.0)
    final_video = accum / counts
    return final_video, path_used


def build_window_blend_weights(
    window_len: int,
    stride: int,
    device: torch.device,
    mode: str,
) -> torch.Tensor:
    if window_len <= 0:
        raise ValueError(f"window_len 必须大于 0，当前为 {window_len}")
    if stride <= 0:
        raise ValueError(f"stride 必须大于 0，当前为 {stride}")
    if mode == "uniform" or stride >= window_len:
        weights = torch.ones(window_len, dtype=torch.float32, device=device)
    else:
        overlap = window_len - stride
        if overlap <= 0:
            weights = torch.ones(window_len, dtype=torch.float32, device=device)
        else:
            if mode == "cosine":
                ramp = torch.linspace(0, 1, overlap, device=device)
                ramp = 0.5 - 0.5 * torch.cos(torch.pi * ramp)
            elif mode == "linear":
                ramp = torch.linspace(0, 1, overlap, device=device)
            else:
                raise ValueError(f"未知的滑窗融合模式: {mode}")
            weights = torch.ones(window_len, dtype=torch.float32, device=device)
            weights[:overlap] = ramp
            weights[-overlap:] = ramp.flip(0)
    return weights.view(-1, 1, 1, 1)


def register_branch_hooks(model: MultimodalJSCC) -> Tuple[Dict[str, int], List[Any]]:
    counters = {
        "image_encoder": 0,
        "video_encoder": 0,
        "image_decoder": 0,
        "video_decoder": 0,
    }
    handles = []

    def make_hook(name: str):
        def hook(_module, _inputs, _outputs):
            counters[name] += 1
        return hook

    if hasattr(model, "image_encoder"):
        handles.append(model.image_encoder.register_forward_hook(make_hook("image_encoder")))
    if hasattr(model, "video_encoder"):
        handles.append(model.video_encoder.register_forward_hook(make_hook("video_encoder")))
    if hasattr(model, "image_decoder"):
        handles.append(model.image_decoder.register_forward_hook(make_hook("image_decoder")))
    if hasattr(model, "video_decoder"):
        handles.append(model.video_decoder.register_forward_hook(make_hook("video_decoder")))
    return counters, handles


def pad_video_patch(patch: torch.Tensor, patch_size: int) -> torch.Tensor:
    h, w = patch.shape[-2], patch.shape[-1]
    pad_h = patch_size - h
    pad_w = patch_size - w
    if pad_h > 0:
        if pad_h < h:
            patch = F.pad(patch, (0, 0, 0, pad_h), mode="reflect")
        else:
            patch = F.pad(patch, (0, 0, 0, pad_h), mode="replicate")
    if pad_w > 0:
        current_w = patch.shape[-1]
        if pad_w < current_w:
            patch = F.pad(patch, (0, pad_w, 0, 0), mode="reflect")
        else:
            patch = F.pad(patch, (0, pad_w, 0, 0), mode="replicate")
    return patch


def split_video_clip_patches(
    video: torch.Tensor,
    patch_size: int,
    overlap: int,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    first_frame = video[0]
    _, meta = split_image_v2(first_frame, patch_size, overlap)
    patches = []
    for pos in meta["positions"]:
        y, x = pos["y"], pos["x"]
        y_end, x_end = pos["y_end"], pos["x_end"]
        patch = video[:, :, y:y_end, x:x_end]
        patch = pad_video_patch(patch, patch_size)
        patches.append(patch)
    patches_tensor = torch.stack(patches, dim=0)
    return patches_tensor, meta


def merge_video_clip_patches(
    patches: torch.Tensor,
    meta: Dict[str, Any],
) -> torch.Tensor:
    frame_count = patches.shape[1]
    reconstructed_frames = []
    for t in range(frame_count):
        frame_patches = patches[:, t]
        frame = merge_image_v2(frame_patches, meta)
        if frame.dim() == 4:
            frame = frame.squeeze(0)
        reconstructed_frames.append(frame)
    return torch.stack(reconstructed_frames, dim=0)


def resolve_patch_decode_resolution(
    model: MultimodalJSCC,
    patch_size_hw: Tuple[int, int],
    logger: logging.Logger,
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    patch_embed = getattr(model.image_encoder, "patch_embed", None)
    patch_embed_size = getattr(patch_embed, "patch_size", None)
    if patch_embed_size is None:
        patch_embed_size = getattr(model.image_encoder, "patch_size", None)
    if patch_embed_size is None:
        logger.warning("未能从模型中解析 patch_embed 大小，将跳过 patch 分辨率传参。")
        return None, None

    if isinstance(patch_embed_size, (list, tuple)):
        patch_embed_h, patch_embed_w = patch_embed_size
    else:
        patch_embed_h = patch_embed_w = patch_embed_size

    if patch_embed_h <= 0 or patch_embed_w <= 0:
        logger.warning("patch_embed 大小非法，将跳过 patch 分辨率传参。")
        return None, None

    patch_h, patch_w = patch_size_hw
    if patch_h % patch_embed_h != 0 or patch_w % patch_embed_w != 0:
        logger.warning(
            "patch 尺寸无法被 patch_embed 大小整除，将跳过 patch 分辨率传参。"
        )
        return None, None

    patch_grid = (patch_h // patch_embed_h, patch_w // patch_embed_w)
    num_layers = getattr(model.image_decoder, "num_layers", None)
    if num_layers is None:
        logger.warning("未能读取解码器层数，将跳过 patch 分辨率传参。")
        return None, None

    scale = 2 ** (num_layers - 1)
    if scale <= 0:
        logger.warning("解码器层数非法，将跳过 patch 分辨率传参。")
        return None, None

    if patch_grid[0] % scale != 0 or patch_grid[1] % scale != 0:
        logger.warning(
            "patch 网格尺寸无法整除解码器缩放比例，将跳过 patch 分辨率传参。"
        )
        return None, None

    input_resolution = (patch_grid[0] // scale, patch_grid[1] // scale)
    if input_resolution[0] <= 0 or input_resolution[1] <= 0:
        logger.warning("推理得到的输入分辨率非法，将跳过 patch 分辨率传参。")
        return None, None

    return input_resolution, patch_grid
 
    #image_tensor = torch.clamp(image_tensor, 0, 1)
    
    #image_np = (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    #return image_np


def infer_image(
    model: MultimodalJSCC,
    image_path: str,
    config: EvaluationConfig,
    device: torch.device,
    logger: logging.Logger,
    text_input: Optional[str] = None,
    text_attention_mask: Optional[torch.Tensor] = None,
    multiple_text_inputs: Optional[List[str]] = None,
    normalize: bool = False,
) -> Dict[str, Any]:
    """
    对单张图像进行推理
    
    Args:
        model: 训练好的模型
        image_path: 图像路径
        config: 评估配置
        device: 计算设备
        logger: 日志记录器
    
    Returns:
        Dict[str, Any]: 推理结果
    """
    logger.info(f"加载图像: {image_path}")
    image = load_image(image_path, normalize=normalize)  # [C, H, W]
    original_shape = image.shape[1:]
    
    logger.info(f"图像尺寸: {original_shape[0]}x{original_shape[1]}")
    
    # 【优化】支持多条语义上下文（不增加显存和计算）
    semantic_context = None
    multiple_semantic_contexts = None
    
    # 收集所有需要编码的文本输入
    texts_to_encode = []
    if text_input is not None:
        texts_to_encode.append(text_input)
    if multiple_text_inputs is not None and len(multiple_text_inputs) > 0:
        texts_to_encode.extend(multiple_text_inputs)
    
    if len(texts_to_encode) > 0:
        # 编码所有文本输入
        semantic_contexts_list = []
        for idx, text in enumerate(texts_to_encode):
            # 简单文本编码（应与训练时一致）
            #text_tensor = torch.tensor([ord(c) for c in text[:config.max_text_length]], dtype=torch.long)
            vocab_limit = config.vocab_size - 1
            text_tensor = torch.tensor([min(ord(c), vocab_limit) for c in text[:config.max_text_length]], dtype=torch.long)
            if len(text_tensor) < config.max_text_length:
                pad_len = config.max_text_length - len(text_tensor)
                text_tensor = torch.cat([text_tensor, torch.zeros(pad_len, dtype=torch.long)])
            text_tensor = text_tensor.unsqueeze(0).to(device)  # [1, seq_len]
            
            # 处理attention mask
            mask = torch.ones_like(text_tensor)
            if idx == 0 and text_attention_mask is not None:
                # 只对第一个文本使用提供的attention mask
                if text_attention_mask.dim() == 1:
                    mask = text_attention_mask.unsqueeze(0).to(device)
                else:
                    mask = text_attention_mask.to(device)
            
            # 编码文本以获取语义上下文
            text_encoded_results = model.encode(
                text_input=text_tensor,
                text_attention_mask=mask
            )
            if 'text_encoded' in text_encoded_results:
                semantic_contexts_list.append(text_encoded_results['text_encoded'])
                # 及时释放中间变量
                del text_tensor, mask, text_encoded_results
        
        # 分配语义上下文：第一条作为主语义，其余作为多条语义
        if len(semantic_contexts_list) > 0:
            semantic_context = semantic_contexts_list[0]
            logger.debug(f"获取主语义上下文: shape={semantic_context.shape}")
            
            if len(semantic_contexts_list) > 1:
                multiple_semantic_contexts = semantic_contexts_list[1:]
                logger.info(f"使用多条语义上下文: {len(multiple_semantic_contexts)} 条")
                # 清理列表引用，但保留tensor（它们会被传递给解码器）
                del semantic_contexts_list
    
    model.eval()
    with torch.no_grad():
        # 检查是否需要patch推理
        patch_size = getattr(config, "patch_size", None)
        if patch_size is None and hasattr(model, "image_encoder") and hasattr(model.image_encoder, "patch_embed"):
            patch_size = getattr(model.image_encoder.patch_embed, "patch_size", 128)
        patch_size = patch_size or 128
        need_patch = config.use_patch_inference and (
            original_shape[0] > patch_size or original_shape[1] > patch_size
        )
        
        if need_patch:
            pad_multiple = patch_size
            padded_image, pad_h, pad_w = pad_image(image, pad_multiple)
            logger.info(f"使用 patch-based 推理（patch_size={config.patch_size}, overlap={config.patch_overlap}）")
            
            # 分割图像
            #patches, meta = split_image_v2(image, config.patch_size, config.patch_overlap)
            #patches, meta = split_image_v2(image, expected_h, config.patch_overlap)
            patches, meta = split_image_v2(padded_image, patch_size, config.patch_overlap)
            logger.info(f"使用 patch-based 推理（patch_size={patch_size}, overlap={config.patch_overlap}）")
            patches = patches.to(device)
            patch_decode_input_res, patch_decode_output_res = resolve_patch_decode_resolution(
                model,
                (patches.shape[-2], patches.shape[-1]),
                logger,
            )
            if max(original_shape) > patch_size:
                global_input = transforms.functional.resize(image, (patch_size, patch_size))
            else:
                global_input = image
            global_input = global_input.unsqueeze(0).to(device)
            
            logger.info(f"图像分割为 {len(patches)} 个 patches")
            with torch.no_grad():
                global_encoded_results = model.encode(image_input=global_input)
                global_guide_vector = global_encoded_results['image_guide']
                global_features = global_encoded_results['image_encoded']
                global_power = torch.mean(global_features ** 2).item()
                logger.info(f"全局功率统计量: {global_power:.6f}")

                logger.info("已计算全局语义引导向量，将应用于所有 Patch")
                del global_encoded_results, global_input, global_features
            original_norm_flag = model.channel.channel_model.power_normalization
            model.channel.channel_model.power_normalization = False
            # 【修复】使用分阶段处理：编码 -> 信道 -> 解码（传入语义上下文）
            # 批量处理patches（使用较小的batch size以避免OOM）
            patch_results_list = []
            patch_batch_size = 4  # 减小batch size以减少内存占用
            
            for i in range(0, len(patches), patch_batch_size):
                patch_batch = patches[i:i+patch_batch_size]
                current_batch_size = patch_batch.size(0)
                
                # 阶段1：编码
                patch_encoded = model.encode(image_input=patch_batch, snr_db=config.snr_db)
                patch_encoded_features = patch_encoded['image_encoded']
                #patch_guide_vectors = patch_encoded['image_guide']
                #patch_guide_vectors = global_guide_vector.expand(current_batch_size, -1)
                
                # 阶段2：功率归一化
                #if 'image' in model.power_normalizer:
                    #patch_encoded_features = model.power_normalizer['image'](patch_encoded_features)
                patch_normalized = patch_encoded_features / (math.sqrt(global_power + 1e-6))
                patch_guide_vectors = global_guide_vector.expand(current_batch_size, -1)

                # 阶段3：信道传输
                patch_transmitted = model.channel(patch_normalized)
                
                # 阶段4：解码（传入语义上下文以启用语义引导）
                # 【优化】直接调用image_decoder以支持多条语义上下文
                #patch_decoded_image = model.image_decoder(
                    #patch_transmitted,
                    #patch_guide_vectors,
                    #semantic_context=semantic_context,
                    #multiple_semantic_contexts=multiple_semantic_contexts,
                    #snr_db=config.snr_db
                #)
                transmitted_dict = {'image': patch_transmitted}
                guide_dict = {'image': patch_guide_vectors}
                decoded_results = model.decode(
                    transmitted_features=transmitted_dict,
                    guide_vectors=guide_dict,
                    semantic_context=semantic_context,
                    multiple_semantic_contexts=multiple_semantic_contexts,
                    image_input_resolution=patch_decode_input_res,
                    image_output_resolution=patch_decode_output_res,
                    snr_db=config.snr_db
                )
                patch_decoded_image = decoded_results['image_decoded']
                # 立即移动到CPU以释放GPU内存
                patch_results_list.append(patch_decoded_image.cpu())
                
                # 清理中间变量
                del patch_encoded, patch_encoded_features, patch_guide_vectors
                del patch_transmitted, patch_decoded_image, patch_batch
                # 每处理几个batch后清理GPU缓存
                if (i // patch_batch_size + 1) % 5 == 0:
                    torch.cuda.empty_cache()
            #model.channel.channel_model.power_normalization = original_norm_flag
            # 合并patches（数据已在CPU上）
            all_patches = torch.cat(patch_results_list, dim=0)
            reconstructed_image = merge_image_v2(all_patches, meta)
            # 清理patch列表以释放内存
            del patch_results_list, all_patches, patches
            torch.cuda.empty_cache()
            reconstructed_image = crop_image(reconstructed_image, pad_h, pad_w)
            model.channel.channel_model.power_normalization = original_norm_flag
            
        else:
            logger.info("使用标准推理")
            # 如果尺寸不匹配，需要resize（但这不是理想情况）
            #if original_shape[0] != expected_h or original_shape[1] != expected_w:
                #logger.warning(f"图像尺寸不匹配，将resize到 {expected_h}x{expected_w}")
                #image = transforms.functional.resize(image, (expected_h, expected_w))
            pad_multiple = 64
            padded_image, pad_h, pad_w = pad_image(image, pad_multiple)
            image = padded_image
            image = image.unsqueeze(0).to(device)  # [1, C, H, W]
            # 【优化】对于标准推理，也需要支持多条语义
            # 使用完整的前向传播，但需要手动处理语义上下文
            #reconstructed_image = crop_image(reconstructed_image, pad_h, pad_w)
            if semantic_context is not None or multiple_semantic_contexts is not None:
                # 分阶段处理以支持多条语义
                encoded = model.encode(image_input=image, snr_db=config.snr_db)
                image_encoded = encoded['image_encoded']
                image_guide = encoded['image_guide']
            
                # 功率归一化
                if 'image' in model.power_normalizer:
                    image_encoded = model.power_normalizer['image'](image_encoded)
                
                # 信道传输
                transmitted = model.channel(image_encoded)
                
                # 解码（支持多条语义）
                reconstructed_image = model.image_decoder(
                    transmitted,
                    image_guide,
                    semantic_context=semantic_context,
                    multiple_semantic_contexts=multiple_semantic_contexts,
                    snr_db=config.snr_db
                )
            else:
                results = model(image_input=image, snr_db=config.snr_db)
                reconstructed_image = results['image_decoded']
            reconstructed_image = crop_image(reconstructed_image, pad_h, pad_w)
    return {
        'original': image.cpu(),
        'reconstructed': reconstructed_image.cpu(),
        'shape': original_shape,
        'normalized': normalize,
    }


def infer_text(
    model: MultimodalJSCC,
    text_path: str,
    config: EvaluationConfig,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    对文本进行推理
    
    Args:
        model: 训练好的模型
        text_path: 文本路径
        config: 评估配置
        device: 计算设备
        logger: 日志记录器
    
    Returns:
        Dict[str, Any]: 推理结果
    """
    logger.info(f"加载文本: {text_path}")
    text = load_text(text_path)
    logger.info(f"文本长度: {len(text)} 字符")
    
    # 简单文本编码（实际应该使用与训练时相同的tokenizer）
    # 这里使用简单的字符编码作为示例
    #text_tensor = torch.tensor([ord(c) for c in text[:config.max_text_length]], dtype=torch.long)
    vocab_limit = config.vocab_size - 1
    text_tensor = torch.tensor([min(ord(c), vocab_limit) for c in text[:config.max_text_length]], dtype=torch.long)
    if len(text_tensor) < config.max_text_length:
        # 填充到固定长度
        pad_len = config.max_text_length - len(text_tensor)
        text_tensor = torch.cat([text_tensor, torch.zeros(pad_len, dtype=torch.long)])
    
    text_tensor = text_tensor.unsqueeze(0).to(device)  # [1, seq_len]
    attention_mask = torch.ones_like(text_tensor)
    
    model.eval()
    with torch.no_grad():
        results = model(
            text_input=text_tensor,
            text_attention_mask=attention_mask,
            snr_db=config.snr_db
        )
    
    return {
        'original': text,
        'reconstructed': results['text_decoded'].cpu(),
        'text_tensor': text_tensor.cpu()
    }


def infer_video(
    model: MultimodalJSCC,
    video_path: str,
    config: EvaluationConfig,
    device: torch.device,
    logger: logging.Logger,
    text_input: Optional[str] = None,
    text_attention_mask: Optional[torch.Tensor] = None,
    multiple_text_inputs: Optional[List[str]] = None,
    diagnose_branches: bool = False,
    default_fps: float = 10.0,
    normalize: bool = False,
) -> Dict[str, Any]:
    """
    对视频进行推理
    
    Args:
        model: 训练好的模型
        video_path: 视频路径
        config: 评估配置
        device: 计算设备
        logger: 日志记录器
    
    Returns:
        Dict[str, Any]: 推理结果
    """
    logger.info(f"加载视频: {video_path}")
    video = load_video(video_path, max_frames=None, normalize=normalize)  # [T, C, H, W]
    fps = get_video_fps(video_path, default_fps)
    original_shape = video.shape
    
    logger.info(f"视频尺寸: {original_shape[0]}帧, {original_shape[2]}x{original_shape[3]}")
    
    # 【优化】支持多条语义上下文（与infer_image保持一致）
    semantic_context = None
    multiple_semantic_contexts = None
    
    # 收集所有需要编码的文本输入
    texts_to_encode = []
    if text_input is not None:
        texts_to_encode.append(text_input)
    if multiple_text_inputs is not None and len(multiple_text_inputs) > 0:
        texts_to_encode.extend(multiple_text_inputs)
    
    if len(texts_to_encode) > 0:
        # 编码所有文本输入
        semantic_contexts_list = []
        for idx, text in enumerate(texts_to_encode):
            # 简单文本编码（应与训练时一致）
            #text_tensor = torch.tensor([ord(c) for c in text[:config.max_text_length]], dtype=torch.long)
            vocab_limit = config.vocab_size - 1
            text_tensor = torch.tensor([min(ord(c), vocab_limit) for c in text[:config.max_text_length]], dtype=torch.long)
            if len(text_tensor) < config.max_text_length:
                pad_len = config.max_text_length - len(text_tensor)
                text_tensor = torch.cat([text_tensor, torch.zeros(pad_len, dtype=torch.long)])
            text_tensor = text_tensor.unsqueeze(0).to(device)  # [1, seq_len]
            
            # 处理attention mask
            mask = torch.ones_like(text_tensor)
            if idx == 0 and text_attention_mask is not None:
                if text_attention_mask.dim() == 1:
                    mask = text_attention_mask.unsqueeze(0).to(device)
                else:
                    mask = text_attention_mask.to(device)
            
            # 编码文本以获取语义上下文
            text_encoded_results = model.encode(
                text_input=text_tensor,
                text_attention_mask=mask
            )
            if 'text_encoded' in text_encoded_results:
                semantic_contexts_list.append(text_encoded_results['text_encoded'])
                del text_tensor, mask, text_encoded_results
        
        # 分配语义上下文
        if len(semantic_contexts_list) > 0:
            semantic_context = semantic_contexts_list[0]
            logger.debug(f"获取主语义上下文: shape={semantic_context.shape}")
            
            if len(semantic_contexts_list) > 1:
                multiple_semantic_contexts = semantic_contexts_list[1:]
                logger.info(f"使用多条语义上下文: {len(multiple_semantic_contexts)} 条")
                del semantic_contexts_list
    
    branch_counters: Dict[str, int] = {}
    hook_handles: List[Any] = []
    if diagnose_branches:
        branch_counters, hook_handles = register_branch_hooks(model)
    path_used = "standard"

    model.eval()
    use_amp = getattr(config, "use_amp", False) and device.type == "cuda"
    window_len, stride = resolve_infer_window(config)
    logger.info(
        "滑窗推理设置: window_len=%d stride=%d (sampling_strategy=%s)",
        window_len,
        stride,
        getattr(config, "video_sampling_strategy", "unknown"),
    )
    with torch.no_grad():
        reconstructed_video, path_used = sliding_window_reconstruct(
            model=model,
            video_full=video,
            config=config,
            device=device,
            logger=logger,
            use_amp=use_amp,
            window_len=window_len,
            stride=stride,
        )

    reconstructed_video = reconstructed_video.cpu()
    if getattr(config, "max_output_frames", None):
        max_output_frames = config.max_output_frames
        if max_output_frames > 0 and reconstructed_video.shape[0] > max_output_frames:
            reconstructed_video = reconstructed_video[:max_output_frames]
            video = video[:max_output_frames]
            original_shape = video.shape
    
    if diagnose_branches:
        for handle in hook_handles:
            handle.remove()
        logger.info(
            "[BranchDiagnose] image_encoder_calls=%d, video_encoder_calls=%d, image_decoder_calls=%d, video_decoder_calls=%d",
            branch_counters.get("image_encoder", 0),
            branch_counters.get("video_encoder", 0),
            branch_counters.get("image_decoder", 0),
            branch_counters.get("video_decoder", 0),
        )
        if path_used == "clip-patch-video":
            logger.info("[BranchDiagnose] 推理路径: clip-level patch video (model(video_input=patches))")
        else:
            logger.info("[BranchDiagnose] 推理路径: standard (model(video_input=...))")

        if branch_counters.get("video_encoder", 0) == 0 and branch_counters.get("image_encoder", 0) > 0:
            logger.warning("[BranchDiagnose] WARNING：视频推理退化为逐帧图像路径（视频分支未参与）")
        elif branch_counters.get("video_encoder", 0) > 0 and branch_counters.get("video_decoder", 0) > 0:
            logger.info("[BranchDiagnose] OK：视频分支生效")

    return {
        'original': video.cpu(),
        'reconstructed': reconstructed_video.cpu(),
        'shape': original_shape,
        'fps': fps,
        'path_used': path_used,
        'normalized': normalize,
    }


def save_result(
    result: Dict[str, Any],
    output_path: str,
    modality: str,
    video_fps: float = 10.0,
    save_video_mp4: bool = True,
    save_video_frames: bool = False,
):
    """
    保存推理结果
    
    Args:
        result: 推理结果字典
        output_path: 输出路径（必须是完整的文件路径，对于图像和文本）
        modality: 模态类型（'image', 'text', 'video'）
    """
    normalized = result.get("normalized", True)
    if modality == 'image':
        # 保存图像
        # 【修复】若输出路径是目录，则写入默认文件名，避免 IsADirectoryError
        if output_path.endswith(os.sep) or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "reconstructed.png")

        # 【修复】确保输出路径有有效的扩展名
        if not os.path.splitext(output_path)[1]:
            output_path = output_path + '.png'

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        reconstructed_np = denormalize_image(result['reconstructed'], normalized=normalized)
        Image.fromarray(reconstructed_np).save(output_path)
        return output_path
    elif modality == 'text':
        if output_path.endswith(os.sep) or os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
            output_path = os.path.join(output_path, "reconstructed.txt")

        if not os.path.splitext(output_path)[1]:
            output_path = output_path + '.txt'

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 保存文本（这里只是简单示例，实际应该解码）
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"原始文本:\n{result['original']}\n\n")
            f.write(f"重建文本（tensor）:\n{result['reconstructed']}\n")
        return output_path
    elif modality == 'video':
        reconstructed = result['reconstructed']
        if reconstructed.dim() == 5:
            reconstructed = reconstructed[0]  # [B, T, C, H, W] -> [T, C, H, W]

        if output_path.lower().endswith(".mp4"):
            mp4_path = output_path
            frames_dir = None
        else:
            os.makedirs(output_path, exist_ok=True)
            mp4_path = os.path.join(output_path, "reconstructed.mp4")
            frames_dir = os.path.join(output_path, "frames")

        if save_video_mp4:
            first_frame = reconstructed[0]
            frame_np = denormalize_video_frame(first_frame, normalized=normalized)
            height, width = frame_np.shape[:2]
            writer = cv2.VideoWriter(
                mp4_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                video_fps,
                (width, height),
            )
            for t in range(reconstructed.shape[0]):
                frame = denormalize_video_frame(reconstructed[t], normalized=normalized)
                writer.write(frame)
            writer.release()

        if save_video_frames:
            if frames_dir is None:
                frames_dir = os.path.splitext(output_path)[0] + "_frames"
            os.makedirs(frames_dir, exist_ok=True)
            for t in range(reconstructed.shape[0]):
                frame = denormalize_image(reconstructed[t], normalized=normalized)
                frame_path = os.path.join(frames_dir, f"frame_{t:04d}.png")
                Image.fromarray(frame).save(frame_path)
        return mp4_path


def load_model(model_path: str, config: EvaluationConfig, device: torch.device, logger: logging.Logger) -> MultimodalJSCC:
    """
    加载模型
    
    Args:
        model_path: 模型权重路径
        config: 评估配置
        device: 计算设备
        logger: 日志记录器
    
    Returns:
        MultimodalJSCC: 加载的模型
    """
    #logger.info(f"加载模型: {model_path}")
    print(f"正在加载模型权重: {model_path}")
    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_config' in checkpoint:
        #model_config = checkpoint['model_config']
        logger.info("✅ 发现 Checkpoint 中保存的模型配置，将直接使用它来初始化模型。")
        model_config = checkpoint['model_config']
        model_config['pretrained'] = checkpoint['model_config'].get('pretrained', False)
        model_config['freeze_encoder'] = False
        if "video_decoder_type" not in model_config:
            fallback_decoder = getattr(config, "video_decoder_type", "swin")
            logger.warning(
                "⚠️ Checkpoint 中缺少 video_decoder_type，回退到推理配置: %s",
                fallback_decoder,
            )
            model_config["video_decoder_type"] = fallback_decoder
        if "video_latent_downsample_factor" not in model_config:
            fallback_factor = getattr(config, "video_latent_downsample_factor", None)
            if fallback_factor is None:
                fallback_factor = getattr(config, "video_latent_downsample_stride", None)
            if fallback_factor is None:
                fallback_factor = 2
            logger.warning(
                "⚠️ Checkpoint 中缺少 video_latent_downsample_factor，回退到推理配置: %s",
                fallback_factor,
            )
            model_config["video_latent_downsample_factor"] = fallback_factor
        if "video_latent_downsample_stride" not in model_config:
            fallback_stride = model_config.get("video_latent_downsample_factor")
            if fallback_stride is None:
                fallback_stride = getattr(config, "video_latent_downsample_stride", None)
            if fallback_stride is None:
                fallback_stride = getattr(config, "video_latent_downsample_factor", 2)
            model_config["video_latent_downsample_stride"] = fallback_stride
        if 'img_embed_dims' in model_config:
            config.img_embed_dims = model_config['img_embed_dims']
        if getattr(config, "image_decoder_type_override", None) is not None:
            model_config["image_decoder_type"] = config.image_decoder_type_override
        if getattr(config, "generator_type_override", None) is not None:
            model_config["generator_type"] = config.generator_type_override
        if getattr(config, "generator_ckpt_override", None) is not None:
            model_config["generator_ckpt"] = config.generator_ckpt_override
        if getattr(config, "z_channels_override", None) is not None:
            model_config["z_channels"] = config.z_channels_override
        if getattr(config, "latent_down_override", None) is not None:
            model_config["latent_down"] = config.latent_down_override
        if getattr(config, "pretrained_model_name", None):
            logger.info(
                "使用命令行指定的 pretrained_model_name: %s",
                config.pretrained_model_name,
            )
            model_config["pretrained_model_name"] = config.pretrained_model_name
        elif model_config.get("pretrained", False) and "pretrained_model_name" not in model_config:
            logger.warning(
                "⚠️ Checkpoint 中缺少 pretrained_model_name，可能导致主干结构不匹配。"
                "请在推理时使用 --pretrained-model-name 指定训练时的模型名称。"
            )
        config.video_clip_len = model_config.get("video_clip_len", config.video_clip_len)
        config.video_stride = model_config.get("video_stride", config.video_stride)
        config.video_sampling_strategy = model_config.get(
            "video_sampling_strategy", config.video_sampling_strategy
        )
        config.video_eval_sampling_strategy = model_config.get(
            "video_eval_sampling_strategy", config.video_eval_sampling_strategy
        )
        if "video_gop_size" in model_config:
            config.video_gop_size = model_config["video_gop_size"]
        elif getattr(config, "video_gop_size", None) is not None:
            model_config["video_gop_size"] = config.video_gop_size
        if getattr(config, "video_gop_size_override", None) is not None:
            model_config["video_gop_size"] = config.video_gop_size_override
            config.video_gop_size = config.video_gop_size_override
    else:
        logger.warning("⚠️ Checkpoint 中未找到配置信息！将使用 EvaluationConfig 的默认值（极高风险！）")
        model_config = {
            'vocab_size': config.vocab_size,
            'text_embed_dim': config.text_embed_dim,
            'text_num_heads': config.text_num_heads,
            'text_num_layers': config.text_num_layers,
            'text_output_dim': config.text_output_dim,
            'img_size': config.img_size,
            'patch_size': getattr(config, 'img_patch_size', 4),
            #'patch_size': config.img_patch_size,
            'img_embed_dims': config.img_embed_dims,
            'img_depths': config.img_depths,
            'img_num_heads': config.img_num_heads,
            'img_output_dim': config.img_output_dim,
            'mlp_ratio': getattr(config, 'mlp_ratio', 4.0),
            'video_hidden_dim': config.video_hidden_dim,
            'video_num_frames': config.video_num_frames,
            'video_use_optical_flow': config.video_use_optical_flow,
            'video_use_convlstm': config.video_use_convlstm,
            'video_output_dim': config.video_output_dim,
            'video_gop_size': getattr(config, "video_gop_size", None),
            'video_decoder_type': getattr(config, "video_decoder_type", "unet"),
            'video_unet_base_channels': getattr(config, "video_unet_base_channels", 64),
            'video_unet_num_down': getattr(config, "video_unet_num_down", 4),
            'video_unet_num_res_blocks': getattr(config, "video_unet_num_res_blocks", 2),
            'channel_type': config.channel_type,
            'snr_db': config.snr_db,
            'normalize_inputs': getattr(config, "normalize", False),
            'image_decoder_type': getattr(config, "image_decoder_type", "baseline"),
            'generator_type': getattr(config, "generator_type", "vae"),
            'generator_ckpt': getattr(config, "generator_ckpt", None),
            'z_channels': getattr(config, "z_channels", 4),
            'latent_down': getattr(config, "latent_down", 8),
            'pretrained': False, # 推理时强制关闭
        }
        if getattr(config, "pretrained_model_name", None):
            model_config["pretrained_model_name"] = config.pretrained_model_name
    model_config["snr_db"] = config.snr_db
    model_config["use_text_guidance_image"] = getattr(config, "use_text_guidance_image", False)
    model_config["use_text_guidance_video"] = getattr(config, "use_text_guidance_video", False)
    model_config["normalize_inputs"] = getattr(config, "normalize", False)
    model_kwargs = dict(model_config)
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
        model = MultimodalJSCC(**model_kwargs)
    except TypeError as e:
        logger.error(f"模型初始化失败，参数不匹配: {e}")
        logger.error("这通常是因为训练时的代码版本与当前代码版本的参数列表不一致。")
        raise e
    m = model.image_encoder
    print(
        "layer1 downsample reduction:",
        m.layers[0].downsample.reduction.weight.shape
        if hasattr(m.layers[0].downsample, "reduction")
        else None,
    )
    print(
        "layer2 downsample reduction:",
        m.layers[1].downsample.reduction.weight.shape
        if hasattr(m.layers[1].downsample, "reduction")
        else None,
    )
    print(
        "layer3 downsample reduction:",
        m.layers[2].downsample.reduction.weight.shape
        if hasattr(m.layers[2].downsample, "reduction")
        else None,
    )
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    logger.info("=== 模型结构验证 ===")
    print_model_structure_info(model, logger)
    if not check_state_dict_compatibility(model, state_dict, logger):
        raise RuntimeError("严重错误：尽管使用了保存的配置，权重维度依然不匹配！请检查代码是否被修改过。")
    model_state_keys = set(model.state_dict().keys())
    unexpected_keys = [key for key in state_dict.keys() if key not in model_state_keys]
    if unexpected_keys:
        preview = ", ".join(sorted(unexpected_keys)[:10])
        logger.warning(
            "⚠️ 检测到 %d 个 checkpoint 权重在当前模型中不存在，将自动忽略。示例: %s",
            len(unexpected_keys),
            preview if preview else "无",
        )
        state_dict = {key: value for key, value in state_dict.items() if key in model_state_keys}
    model.load_state_dict(state_dict, strict=True) # 建议改为 True，有问题直接报出来
    model = model.to(device)
    model.eval()
    logger.info("模型加载成功！")
    return model
        

        #config.img_embed_dims = saved_config.get('img_embed_dims', config.img_embed_dims)
        #config.img_depths = saved_config.get('img_depths', config.img_depths)
        #config.img_num_heads = saved_config.get('img_num_heads', config.img_num_heads)
        #config.pretrained = saved_config.get('pretrained', False) # 恢复预训练开关
        #print(f"配置已更新: Pretrained={config.pretrained}, Dims={config.img_embed_dims}")
    #else:
        #print("警告: Checkpoint 中未找到配置信息，将使用 config.py 的默认值（可能导致维度不匹配！）")
    
    
    # 尝试从检查点中恢复模型配置
   # model_kwargs = {
        #'vocab_size': config.vocab_size,
        #'text_embed_dim': config.text_embed_dim,
        #'text_num_heads': config.text_num_heads,
        #'text_num_layers': config.text_num_layers,
        #'text_output_dim': config.text_output_dim,
        #'img_size': config.img_size,
        #'patch_size': config.img_patch_size,
        #'img_embed_dims': config.img_embed_dims,
        #'img_depths': config.img_depths,
        #'img_num_heads': config.img_num_heads,
        #'img_output_dim': config.img_output_dim,
        #'mlp_ratio': getattr(config, 'mlp_ratio', 4.0),
        #'video_hidden_dim': config.video_hidden_dim,
        #'video_num_frames': config.video_num_frames,
        #'video_use_optical_flow': config.video_use_optical_flow,
        #'video_use_convlstm': config.video_use_convlstm,
        #'video_output_dim': config.video_output_dim,
        #'channel_type': config.channel_type,
        #'snr_db': config.snr_db,
        #'pretrained': config.pretrained,
        #'pretrained_model_name': 'swin_tiny_patch4_window7_224'
    #}
    
    # 如果检查点中包含模型配置，使用检查点的配置
    #if isinstance(checkpoint, dict) and 'model_config' in checkpoint:
        #logger.info("发现 Checkpoint 中保存的模型配置，正在覆盖默认配置...")
        #saved_config = checkpoint['model_config']
       # keys_to_overwrite = ['img_embed_dims', 'img_depths', 'img_num_heads', 'patch_size', 'img_size']
        #for k in keys_to_overwrite:
            #if k in saved_config:
                #model_kwargs[k] = saved_config[k]
                #logger.info(f"  - 覆盖 {k}: {saved_config[k]}")
        #model_kwargs.update(saved_config)
   
    
    # 创建模型
    #model = MultimodalJSCC(**model_kwargs)
    #logger.info("=== 推理模型结构验证 ===")
    #print_model_structure_info(model, logger)
    #if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        #tate_dict = checkpoint['model_state_dict']
    #else:
        #state_dict = checkpoint
    # 加载权重
    #if isinstance(checkpoint, dict):
        #if 'model_state_dict' in checkpoint:
            #model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            #logger.info("加载模型权重（从model_state_dict）")
        #else:
            #model.load_state_dict(checkpoint, strict=False)
            #logger.info("加载模型权重（直接state_dict）")
    #else:
        #model.load_state_dict(checkpoint, strict=False)
        #logger.info("加载模型权重（检查点本身）")
    #if not check_state_dict_compatibility(model, state_dict, logger):
        #raise RuntimeError("模型配置与权重文件维度不匹配！请检查 inference.py 中的默认配置或 checkpoint 的来源。")
    #model.load_state_dict(state_dict, strict=False)
    #model = model.to(device)
    #model.eval()
    
    #logger.info("模型加载成功")
    
    #return model


def main():
    parser = argparse.ArgumentParser(description='多模态JSCC推理脚本')
    parser.add_argument('--model-path', type=str, required=True, help='模型权重路径')
    parser.add_argument('--input', type=str, required=True, help='输入文件路径（图像、文本或视频）')
    parser.add_argument('--output', type=str, default=None, help='输出路径（默认：输入文件名_reconstructed）')
    parser.add_argument('--modality', type=str, choices=['image', 'text', 'video'], 
                       default=None, help='模态类型（如果不指定，将从文件扩展名推断）')
    parser.add_argument('--snr', type=float, default=10.0, help='信噪比 (dB)')
    parser.add_argument('--snr-random', action='store_true', help='推理时启用随机SNR')
    parser.add_argument('--snr-min', type=float, default=None, help='推理随机SNR最小值')
    parser.add_argument('--snr-max', type=float, default=None, help='推理随机SNR最大值')
    parser.add_argument('--patch', action='store_true', help='启用patch-based推理')
    parser.add_argument('--video_fps', type=float, default=10.0, help='视频保存帧率')
    parser.add_argument('--infer-window-len', type=int, default=None, help='滑窗推理window长度')
    parser.add_argument('--infer-window-stride', type=int, default=None, help='滑窗推理stride')
    parser.add_argument(
        '--infer-window-blend',
        type=str,
        default='uniform',
        choices=['uniform', 'linear', 'cosine'],
        help='滑窗重叠区域融合模式'
    )
    parser.add_argument('--max-output-frames', type=int, default=None, help='推理输出最大帧数（调试用）')
    parser.add_argument(
        '--video-gop-size',
        type=int,
        default=None,
        help='显式指定视频GOP长度（用于对齐训练时的gop_size）',
    )
    parser.add_argument(
        '--video-sampling-strategy',
        type=str,
        default=None,
        choices=["contiguous_clip", "uniform", "fixed_start"],
        help='推理侧记录的采样策略（用于日志显示）',
    )
    parser.add_argument(
        '--normalize',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='使用ImageNet归一化（与旧模型兼容）',
    )
    parser.add_argument(
        '--save_video_mp4',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='保存视频 mp4',
    )
    parser.add_argument('--save_video_frames', action='store_true', help='保存视频帧序列')
    parser.add_argument('--diagnose_branches', action='store_true', help='诊断视频分支调用情况')
    parser.add_argument('--use-text-guidance-image', action='store_true', help='启用文本语义引导图像重建')
    parser.add_argument('--use-text-guidance-video', action='store_true', help='启用文本语义引导视频重建')
    parser.add_argument('--text-input', type=str, default=None, 
                       help='文本输入路径（用于语义引导图像/视频解码）')
    parser.add_argument('--prompt', type=str, default=None,
                       help='文本提示（直接输入文本，用于语义引导图像/视频解码）')
    parser.add_argument('--prompts', type=str, nargs='+', default=None,
                       help='多条文本提示（用于多条语义引导，与--prompt互斥）')
    parser.add_argument(
        '--pretrained-model-name',
        type=str,
        default=None,
        help='手动指定训练时的预训练主干名称（例如 swin_small_patch4_window7_224）',
    )
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
    config.snr_db = args.snr
    config.snr_random = args.snr_random
    if args.snr_min is not None:
        config.snr_min = args.snr_min
    if args.snr_max is not None:
        config.snr_max = args.snr_max
    config.use_patch_inference = bool(args.patch)
    config.pretrained_model_name = args.pretrained_model_name
    config.image_decoder_type_override = args.image_decoder_type
    config.generator_type_override = args.generator_type
    config.generator_ckpt_override = args.generator_ckpt
    config.z_channels_override = args.z_channels
    config.latent_down_override = args.latent_down
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
    config.infer_window_len = args.infer_window_len
    config.infer_window_stride = args.infer_window_stride
    config.infer_window_blend = args.infer_window_blend
    config.max_output_frames = args.max_output_frames
    config.normalize = args.normalize
    config.video_gop_size_override = args.video_gop_size
    if args.use_text_guidance_image:
        config.use_text_guidance_image = True
    if args.use_text_guidance_video:
        config.use_text_guidance_video = True
    if args.video_sampling_strategy:
        config.video_sampling_strategy = args.video_sampling_strategy
    if args.video_gop_size is not None:
        config.video_gop_size = args.video_gop_size
    if config.snr_random:
        snr_generator = torch.Generator(device="cpu")
        snr_generator.manual_seed(config.seed)
        rand_value = torch.rand(1, generator=snr_generator).item()
        config.snr_db = config.snr_min + (config.snr_max - config.snr_min) * rand_value
    
    # 推断模态类型
    if args.modality is None:
        ext = os.path.splitext(args.input)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            modality = 'image'
        elif ext in ['.txt', '.text']:
            modality = 'text'
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            modality = 'video'
        else:
            raise ValueError(f"无法从文件扩展名推断模态类型: {ext}，请使用 --modality 参数指定")
    else:
        modality = args.modality
    
    # 设置输出路径
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        if modality == 'image':
            args.output = f'{base_name}_reconstructed.png'
        elif modality == 'text':
            args.output = f'{base_name}_reconstructed.txt'
        elif modality == 'video':
            args.output = f'{base_name}_reconstructed'
    else:
        # 【修复】处理用户提供的输出路径
        # 检查输出路径是否是目录或没有扩展名
        output_ext = os.path.splitext(args.output)[1]
        is_directory = os.path.isdir(args.output) or (not output_ext and not os.path.exists(args.output))
        
        if is_directory:
            # 输出路径是目录，需要从输入路径提取文件名并添加扩展名
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            input_ext = os.path.splitext(args.input)[1].lower()
            
            if modality == 'image':
                # 如果输入有图像扩展名，使用相同的扩展名；否则使用png
                if input_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    file_ext = input_ext if input_ext != '.jpeg' else '.jpg'
                else:
                    file_ext = '.png'
                args.output = os.path.join(args.output, f'{base_name}_reconstructed{file_ext}')
            elif modality == 'text':
                args.output = os.path.join(args.output, f'{base_name}_reconstructed.txt')
            elif modality == 'video':
                # 视频输出使用目录
                args.output = os.path.join(args.output, f'{base_name}_reconstructed')
        elif not output_ext:
            # 输出路径是文件路径但没有扩展名，需要添加扩展名
            input_ext = os.path.splitext(args.input)[1].lower()
            
            if modality == 'image':
                # 如果输入有图像扩展名，使用相同的扩展名；否则使用png
                if input_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    file_ext = input_ext if input_ext != '.jpeg' else '.jpg'
                else:
                    file_ext = '.png'
                args.output = args.output + file_ext
            elif modality == 'text':
                args.output = args.output + '.txt'
            # video模式不需要扩展名，因为它输出的是目录

    if modality in ['image', 'text'] and os.path.isdir(args.output):
        base_name = os.path.splitext(os.path.basename(args.input))[0]
        if modality == 'image':
            input_ext = os.path.splitext(args.input)[1].lower()
            if input_ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                file_ext = input_ext if input_ext != '.jpeg' else '.jpg'
            else:
                file_ext = '.png'
            desired_name = f'{base_name}_reconstructed{file_ext}'
        else:
            desired_name = f'{base_name}_reconstructed.txt'

        output_basename = os.path.basename(args.output)
        if output_basename == desired_name:
            args.output = os.path.join(os.path.dirname(args.output), desired_name)
        else:
            args.output = os.path.join(args.output, desired_name)
    
    # 创建输出目录
    if modality == 'video':
        if not args.output.lower().endswith(".mp4"):
            makedirs(args.output)
        else:
            output_dir = os.path.dirname(args.output)
            if output_dir:
                makedirs(output_dir)
    else:
        # 对于图像和文本，确保输出目录存在
        output_dir = os.path.dirname(args.output)
        if output_dir:
            makedirs(output_dir)
    
    # 配置日志
    log_config = type('LogConfig', (), {
        'workdir': './logs',
        'log': os.path.join('./logs', 'inference.log'),
        'samples': args.output if modality != 'video' else os.path.dirname(args.output),
        'models': os.path.dirname(args.model_path)
    })()
    logger = logger_configuration(log_config, save_log=True)
    
    logger.info("=" * 80)
    logger.info("多模态JSCC推理脚本")
    logger.info("=" * 80)
    logger.info(f"模型路径: {config.model_path}")
    logger.info(f"输入文件: {args.input}")
    logger.info(f"输出路径: {args.output}")
    logger.info(f"模态类型: {modality}")
    if config.snr_random:
        logger.info(
            f"SNR: {config.snr_db:.2f} dB (random in [{config.snr_min}, {config.snr_max}])"
        )
    else:
        logger.info(f"SNR: {config.snr_db} dB")
    logger.info(f"设备: {config.device}")
    logger.info(f"Patch推理: {'启用' if config.use_patch_inference else '禁用'}")
    logger.info(f"Normalize: {'启用' if args.normalize else '禁用'} (默认与训练一致)")
    
    # 加载模型
    model = load_model(config.model_path, config, config.device, logger)
    
    # 加载文本输入（如果提供，用于语义引导）
    text_input = None
    text_attention_mask = None
    multiple_text_inputs = None
    
    # 【优化】支持多条语义输入
    if args.prompts:
        # 使用多条提示
        if args.prompt or args.text_input:
            logger.warning("--prompts 与 --prompt/--text-input 同时指定，将忽略 --prompt/--text-input")
        text_input = args.prompts[0]  # 第一条作为主语义
        multiple_text_inputs = args.prompts[1:] if len(args.prompts) > 1 else None
        logger.info(f"使用多条文本提示: 主提示 {len(text_input)} 字符, 额外 {len(multiple_text_inputs) if multiple_text_inputs else 0} 条")
    elif args.text_input:
        text_input = load_text(args.text_input)
        logger.info(f"从文件加载文本输入: {len(text_input)} 字符")
    elif args.prompt:
        text_input = args.prompt
        logger.info(f"使用命令行提示文本: {len(text_input)} 字符")
    
    if text_input:
        # 准备文本tensor和attention mask（仅用于第一个文本）
        text_tensor = torch.tensor([ord(c) for c in text_input[:config.max_text_length]], dtype=torch.long)
        if len(text_tensor) < config.max_text_length:
            pad_len = config.max_text_length - len(text_tensor)
            text_tensor = torch.cat([text_tensor, torch.zeros(pad_len, dtype=torch.long)])
        text_attention_mask = torch.ones_like(text_tensor)
    
    # 执行推理
    logger.info("开始推理...")
    if modality == 'image':
        result = infer_image(
            model,
            args.input,
            config,
            config.device,
            logger,
            text_input=text_input,
            text_attention_mask=text_attention_mask,
            multiple_text_inputs=multiple_text_inputs,
            normalize=args.normalize,
        )
    elif modality == 'text':
        result = infer_text(model, args.input, config, config.device, logger)
    elif modality == 'video':
        input_meta = get_video_meta(args.input)
        logger.info(
            "[InputVideoMeta] width=%d, height=%d, fps=%.3f, frame_count=%d, duration=%.3f",
            input_meta["width"],
            input_meta["height"],
            input_meta["fps"],
            input_meta["frame_count"],
            input_meta["duration"],
        )
        result = infer_video(
            model,
            args.input,
            config,
            config.device,
            logger,
            text_input=text_input,
            text_attention_mask=text_attention_mask,
            multiple_text_inputs=multiple_text_inputs,
            diagnose_branches=args.diagnose_branches,
            default_fps=args.video_fps,
            normalize=args.normalize,
        )
    else:
        raise ValueError(f"不支持的模态类型: {modality}")
    
    # 保存结果
    logger.info(f"保存结果到: {args.output}")
    saved_path = save_result(
        result,
        args.output,
        modality,
        video_fps=result.get("fps", args.video_fps),
        save_video_mp4=args.save_video_mp4,
        save_video_frames=args.save_video_frames,
    )
    if modality == 'video' and saved_path:
        output_meta = get_video_meta(saved_path)
        logger.info(
            "[OutputVideoMeta] width=%d, height=%d, fps=%.3f, frame_count=%d, duration=%.3f",
            output_meta["width"],
            output_meta["height"],
            output_meta["fps"],
            output_meta["frame_count"],
            output_meta["duration"],
        )
    
    logger.info("=" * 80)
    logger.info("推理完成！")


if __name__ == '__main__':
    main()
