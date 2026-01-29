"""
多模态JSCC训练脚本

实现完整的训练-验证循环，支持文本、图像、视频三种模态的联合训练。
参考 main.py 的训练框架，适配多模态模型接口和数据加载器。
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import argparse
import logging
import json
import math
from functools import partial

# 导入模型和工具
from multimodal_jscc import MultimodalJSCC
from losses import MultimodalLoss
from metrics import calculate_multimodal_metrics
from data_loader import MultimodalDataLoader, MultimodalDataset, collate_multimodal_batch

# 从本地模块导入配置和工具函数
from config import TrainingConfig
from utils import AverageMeter, seed_torch, logger_configuration, makedirs, load_manifest
from utils_check import print_model_structure_info, check_state_dict_compatibility


def create_model(config: TrainingConfig) -> MultimodalJSCC:
    #if getattr(config, 'pretrained', False):
        #if 'swin_tiny' in config.pretrained_model_name:
            #standard_dims = [96, 192, 384, 768]
            #standard_depths = [2, 2, 6, 2]
            #standard_heads = [3, 6, 12, 24]
            #if config.img_embed_dims != standard_dims:
                #print(f"【自动修正】检测到预训练模型，将 embed_dims 从 {config.img_embed_dims} 修正为 {standard_dims}")
                #config.img_embed_dims = standard_dims    
            #if config.img_depths != standard_depths:
                #config.img_depths = standard_depths
            #if config.img_num_heads != standard_heads:
                #config.img_num_heads = standard_heads
            #if config.img_depths != standard_depths:
                #print(f"正在强制修正 config.img_depths 为 {standard_depths} 以匹配预训练模型。")
                #config.img_depths = standard_depths
    """创建多模态JSCC模型"""
    model = MultimodalJSCC(
        vocab_size=config.vocab_size,
        text_embed_dim=config.text_embed_dim,
        text_num_heads=config.text_num_heads,
        text_num_layers=config.text_num_layers,
        text_output_dim=config.text_output_dim,
        img_size=config.img_size,
        patch_size=config.patch_size,
        img_embed_dims=config.img_embed_dims,
        img_depths=config.img_depths,
        img_num_heads=config.img_num_heads,
        img_output_dim=config.img_output_dim,
        img_window_size=getattr(config, 'img_window_size', 7),
        pretrained=getattr(config, 'pretrained', False), # 【Phase 1】预训练权重
        freeze_encoder=getattr(config, 'freeze_encoder', False),  # 【Phase 1】冻结编码器
        pretrained_model_name=getattr(config, 'pretrained_model_name', 'swin_tiny_patch4_window7_224'),
        image_decoder_type=getattr(config, "image_decoder_type", "baseline"),
        generator_type=getattr(config, "generator_type", "vae"),
        generator_ckpt=getattr(config, "generator_ckpt", None),
        z_channels=getattr(config, "z_channels", 4),
        latent_down=getattr(config, "latent_down", 8),
        video_hidden_dim=config.video_hidden_dim,
        video_num_frames=config.video_num_frames,
        video_use_optical_flow=config.video_use_optical_flow,
        video_use_convlstm=config.video_use_convlstm,
        video_output_dim=config.video_output_dim,
        video_gop_size=getattr(config, "video_gop_size", None),
        video_latent_downsample_factor=getattr(
            config,
            "video_latent_downsample_factor",
            getattr(config, "video_latent_downsample_stride", 2),
        ),
        video_entropy_max_exact_quantile_elems=getattr(
            config, "video_entropy_max_exact_quantile_elems", 2_000_000
        ),
        video_entropy_quantile_sample_size=getattr(
            config, "video_entropy_quantile_sample_size", 262_144
        ),
        video_decoder_type=getattr(config, "video_decoder_type", "unet"),
        video_unet_base_channels=getattr(config, "video_unet_base_channels", 64),
        video_unet_num_down=getattr(config, "video_unet_num_down", 4),
        video_unet_num_res_blocks=getattr(config, "video_unet_num_res_blocks", 2),
        video_decode_chunk_size=getattr(config, "video_decode_chunk_size", None),
        channel_type=config.channel_type,
        snr_db=config.snr_db,
        use_quantization_noise=getattr(config, 'use_quantization_noise', False),
        quantization_noise_range=getattr(config, 'quantization_noise_range', 0.5),
        normalize_inputs=getattr(config, "normalize", False),
        use_text_guidance_image=getattr(config, "use_text_guidance_image", False),
        use_text_guidance_video=getattr(config, "use_text_guidance_video", False),
        enforce_text_condition=getattr(config, "enforce_text_condition", True),
        condition_margin_weight=getattr(config, "condition_margin_weight", 0.0),
        condition_margin=getattr(config, "condition_margin", 0.05),
        condition_prob=getattr(config, "condition_prob", 0.0),
        condition_only_low_snr=getattr(config, "condition_only_low_snr", False),
        condition_low_snr_threshold=getattr(config, "condition_low_snr_threshold", 5.0),
        use_gradient_checkpointing=getattr(config, "use_gradient_checkpointing", True),
    )
    return model


def create_loss_fn(config: TrainingConfig) -> MultimodalLoss:
    """创建损失函数"""
    gan_weight = getattr(config, 'gan_weight', getattr(config, 'discriminator_weight', 0.01))
    loss_fn = MultimodalLoss(
        text_weight=config.text_weight,
        image_weight=config.image_weight,
        video_weight=config.video_weight,
        image_decoder_type=getattr(config, "image_decoder_type", "baseline"),
        reconstruction_weight=config.reconstruction_weight,
        perceptual_weight=config.perceptual_weight,
        temporal_weight=config.temporal_weight,
        video_perceptual_weight=getattr(config, "video_perceptual_weight", 0.0),
        temporal_perceptual_weight=getattr(config, "temporal_perceptual_weight", 0.0),
        color_consistency_weight=getattr(config, "color_consistency_weight", 0.0),
        text_contrastive_weight=getattr(config, 'text_contrastive_weight', 0.1),  # 【新增】文本对比损失权重
        video_text_contrastive_weight=getattr(config, 'video_text_contrastive_weight', 0.05),  # 【新增】视频-文本对比损失权重
        rate_weight=getattr(config, 'rate_weight', 1e-4),  # 【新增】码率/能量约束权重
        temporal_consistency_weight=getattr(config, 'temporal_consistency_weight', 0.02),  # 【新增】视频时序一致性正则权重
        gan_weight=gan_weight,  # 【Phase 6】对抗损失权重
        use_adversarial=getattr(config, 'use_adversarial', False),  # 【Phase 3】是否使用对抗训练
        condition_margin_weight=getattr(config, "condition_margin_weight", 0.0),
        condition_margin=getattr(config, "condition_margin", 0.05),
        generative_gamma1=getattr(config, "generative_gamma1", 1.0),
        generative_gamma2=getattr(config, "generative_gamma2", 0.1),
        normalize=getattr(config, "normalize", False),
        data_range=1.0
    )
    return loss_fn


def create_optimizer(model: MultimodalJSCC, config: TrainingConfig):
    """创建优化器"""
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    return optimizer


def create_scheduler(optimizer, config: TrainingConfig):
    """创建学习率调度器"""
    if config.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.lr_min
        )
    elif config.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
    elif config.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=config.lr_min
        )
    else:
        scheduler = None
    return scheduler


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def compute_bandwidth_ratio(config: TrainingConfig, epoch: int) -> float:
    start = float(getattr(config, "bandwidth_ratio_start", 1.0))
    end = float(getattr(config, "bandwidth_ratio_end", start))
    warmup = max(0, int(getattr(config, "bandwidth_warmup_epochs", 0)))
    anneal = max(0, int(getattr(config, "bandwidth_anneal_epochs", 0)))

    if end == start:
        return start
    if epoch < warmup:
        return start
    if anneal <= 0:
        return end
    progress = (epoch - warmup + 1) / float(anneal)
    progress = max(0.0, min(1.0, progress))
    return start + (end - start) * progress


def _align_video_gop_size(config: TrainingConfig, logger: Optional[logging.Logger] = None) -> None:
    clip_len = int(getattr(config, "video_clip_len", 0) or 0)
    gop_size = int(getattr(config, "video_gop_size", 0) or 0)
    if clip_len <= 0:
        return
    if gop_size <= 0:
        config.video_gop_size = clip_len
        if logger is not None:
            logger.warning(
                "video_gop_size未设置，已自动对齐为video_clip_len=%d。",
                clip_len,
            )
        return
    if clip_len % gop_size == 0:
        return
    divisors = [d for d in range(1, clip_len + 1) if clip_len % d == 0]
    closest = min(divisors, key=lambda d: (abs(d - gop_size), d))
    if logger is not None:
        logger.warning(
            "video_gop_size=%d不能整除video_clip_len=%d，已自动调整为%d。",
            gop_size,
            clip_len,
            closest,
        )
    config.video_gop_size = closest


def _log_nonfinite_tensor(
    logger: logging.Logger,
    tensor: torch.Tensor,
    name: str,
    batch_idx: int,
    stage: str,
) -> bool:
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
        "Batch %d stage=%s tensor=%s contains NaN/Inf (finite stats min=%.6e max=%.6e mean=%.6e)",
        batch_idx + 1,
        stage,
        name,
        t_min,
        t_max,
        t_mean,
    )
    return True


def _compute_r1_penalty(d_out: torch.Tensor, real_input: torch.Tensor) -> torch.Tensor:
    grad = torch.autograd.grad(
        outputs=d_out.sum(),
        inputs=real_input,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    grad = grad.view(grad.shape[0], -1)
    return grad.pow(2).sum(1).mean()


def train_one_epoch(
    model: MultimodalJSCC,
    train_loader: DataLoader,
    loss_fn: MultimodalLoss,
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    logger: logging.Logger,
    device: torch.device,
    scaler: Optional[Any] = None,  # 混合精度训练的scaler
    discriminator: Optional[Any] = None,  # 【Phase 3】判别器
    optimizer_d: Optional[optim.Optimizer] = None  # 【Phase 3】判别器优化器
) -> Dict[str, float]:
    """
    训练一个epoch
    
    Returns:
        Dict[str, float]: 训练指标字典
    """
    model.train()
    
    # 指标记录器
    meters = {
        'loss': AverageMeter(),
        'text_loss': AverageMeter(),
        'image_loss': AverageMeter(),
        'video_loss': AverageMeter(),
        'image_recon_loss': AverageMeter(),
        'image_percep_loss': AverageMeter(),
        'video_recon_loss': AverageMeter(),
        'video_percep_loss': AverageMeter(),
        'video_temporal_loss': AverageMeter(),
        'condition_margin_loss': AverageMeter(),
        'adversarial_loss': AverageMeter(),
        'disc_loss_real': AverageMeter(),
        'disc_loss_fake': AverageMeter(),
        'video_semantic_gate_mean': AverageMeter(),
        'time': AverageMeter()
    }

    # 跳过统计（用于定位NaN/Inf导致的更新缺失）
    skipped_loss_nan = 0
    skipped_grad_nan = 0
    
    global_step = epoch * len(train_loader)
    accumulation_steps = config.gradient_accumulation_steps
    snr_generator = None
    snr_strategy = getattr(config, "train_snr_strategy", "fixed")
    if snr_strategy == "random":
        snr_generator = torch.Generator(device="cpu")
        snr_generator.manual_seed(config.seed + epoch)
    
    # 只在每个累积周期的第一步清零梯度
    #optimizer.zero_grad()
    #if optimizer_d is not None:
        #optimizer_d.zero_grad()
    adv_loss_fn = None
    gan_enable_epoch = int(getattr(config, "gan_enable_epoch", 0))
    d_updates_per_g = max(1, int(getattr(config, "d_updates_per_g", 1)))
    use_r1 = bool(getattr(config, "use_r1_regularization", False))
    r1_gamma = float(getattr(config, "r1_gamma", 10.0))
    gan_active = bool(getattr(config, "use_adversarial", False)) and epoch >= gan_enable_epoch
    if discriminator is not None and gan_active:
        from losses import AdversarialLoss
        adv_loss_fn = AdversarialLoss().to(device)

    for batch_idx, batch in enumerate(train_loader):
        start_time = time.time()
        batch_indices = batch.get('indices', None)
        batch_indices_str = f"{list(batch_indices)}" if batch_indices is not None else "N/A"
        
        # 移除torch.cuda.empty_cache()以提升训练速度
        
        # [修复1] 重置隐藏状态，确保批次间计算图独立
        # 优化：只在需要时重置（视频/序列模型）
        #if hasattr(model, 'reset_hidden_states'):
            #model.reset_hidden_states()
        real_model = unwrap_model(model)
        if hasattr(real_model, 'reset_hidden_states'):
            real_model.reset_hidden_states()
        
        # 从batch中提取数据
        # batch格式: {'inputs': {...}, 'targets': {...}, 'attention_mask': ...}
        inputs = batch['inputs']
        targets = batch['targets']
        attention_mask = batch.get('attention_mask', None)
        
        # 提取各模态输入
        text_input = inputs.get('text_input', None)
        image_input = inputs.get('image_input', None)
        video_input = inputs.get('video_input', None)
        
        # 移动到设备
        if text_input is not None:
            text_input = text_input.to(device, non_blocking=True)
            attention_mask = attention_mask.to(device, non_blocking=True) if attention_mask is not None else None
        if image_input is not None:
            image_input = image_input.to(device, non_blocking=True)
        if video_input is not None:
            video_input = video_input.to(device, non_blocking=True)
        
        # 移动targets到设备
        device_targets = {}
        for key, value in targets.items():
            if value is not None:
                device_targets[key] = value.to(device, non_blocking=True)
        
        # SNR策略：random / curriculum / fixed
        if snr_strategy == "random":
            rand_value = torch.rand(1, generator=snr_generator).item()
            snr_db = config.train_snr_min + (config.train_snr_max - config.train_snr_min) * rand_value
        elif snr_strategy == "curriculum":
            total_epochs = max(1, int(getattr(config, "num_epochs", 1)))
            progress = min(1.0, max(0.0, epoch / float(total_epochs - 1))) if total_epochs > 1 else 1.0
            snr_db = config.train_snr_max - (config.train_snr_max - config.train_snr_min) * progress
        else:
            snr_db = config.snr_db
        
        # 移除torch.cuda.empty_cache()以提升训练速度
        
        # 前向传播（使用混合精度）
        use_amp = config.use_amp and device.type == "cuda"
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            results = model(
                text_input=text_input,
                image_input=image_input,
                video_input=video_input,
                text_attention_mask=attention_mask,
                snr_db=snr_db
            )
            for key, value in results.items():
                _log_nonfinite_tensor(
                    logger=logger,
                    tensor=value,
                    name=f"results[{key}]",
                    batch_idx=batch_idx,
                    stage="forward_output",
                )
            
            # 【Phase 3】如果启用对抗训练，计算判别器输出
            discriminator_outputs = None
            if discriminator is not None and gan_active:
                # 判别器在生成器训练时不需要梯度
                for p in discriminator.parameters():
                    p.requires_grad = False

                image_disc_pred = None
                video_disc_pred = None
                if 'image_decoded' in results:
                    image_disc_pred, _ = discriminator(image=results['image_decoded'])
                if 'video_decoded' in results:
                    _, video_disc_pred = discriminator(video=results['video_decoded'])
                discriminator_outputs = {}
                if image_disc_pred is not None:
                    discriminator_outputs['image'] = image_disc_pred
                if video_disc_pred is not None:
                    discriminator_outputs['video'] = video_disc_pred
                for p in discriminator.parameters():
                    p.requires_grad = True
            
            # 条件边际约束：对同一传输特征使用打乱文本解码，强制文本条件有效
            condition_enabled = (
                getattr(config, "enforce_text_condition", False)
                and getattr(config, "condition_margin_weight", 0.0) > 0
                and results.get("video_transmitted") is not None
                and results.get("video_guide") is not None
                and results.get("text_encoded") is not None
                and results.get("video_decoded") is not None
                and results["text_encoded"].shape[0] > 1
                and (not getattr(config, "condition_only_low_snr", False) or snr_db <= getattr(config, "condition_low_snr_threshold", 5.0))
                and np.random.rand() < getattr(config, "condition_prob", 0.0)
            )
            if condition_enabled:
                with torch.no_grad():
                    shuffle_idx = torch.randperm(results["text_encoded"].shape[0], device=device)
                shuffled_context = results["text_encoded"][shuffle_idx]
                # 使用相同的传输特征+guide进行第二次解码
                shuffled_video = real_model.video_decoder(
                    results["video_transmitted"],
                    results["video_guide"],
                    semantic_context=shuffled_context,
                    reset_state=True,
                )
                results["video_decoded_shuffled"] = shuffled_video
                if getattr(real_model.video_decoder, "last_semantic_gate_stats", None):
                    stats = real_model.video_decoder.last_semantic_gate_stats
                    results["video_semantic_gate_mean_shuffled"] = stats.get("mean")
                    results["video_semantic_gate_std_shuffled"] = stats.get("std")

            # 计算损失
            loss_dict = loss_fn(
                predictions=results,
                targets=device_targets,
                attention_mask=attention_mask,
                discriminator_outputs=discriminator_outputs  # 【Phase 3】传递判别器输出
            )
            for key, value in loss_dict.items():
                _log_nonfinite_tensor(
                    logger=logger,
                    tensor=value,
                    name=f"loss_dict[{key}]",
                    batch_idx=batch_idx,
                    stage="loss_output",
                )
            
            total_loss = loss_dict['total_loss']
            # 检查损失值是否有效
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                logger.warning(f"批次 {batch_idx + 1}: 生成器 Loss NaN/Inf，跳过")
                # 清理中间变量
                del results, loss_dict, total_loss
                skipped_loss_nan += 1
                continue
            
            # 梯度累积：损失需要除以累积步数
            total_loss = total_loss / accumulation_steps
            if _log_nonfinite_tensor(
                logger=logger,
                tensor=total_loss,
                name="total_loss",
                batch_idx=batch_idx,
                stage="pre_backward",
            ):
                logger.warning(f"批次 {batch_idx + 1}: 反向传播前 total_loss NaN/Inf，跳过 | indices={batch_indices_str}")
                del results, loss_dict, total_loss
                skipped_loss_nan += 1
                continue
        
        # 反向传播（使用混合精度）
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()
        disc_loss_real_value = None
        disc_loss_fake_value = None

        # 梯度累积：只在累积步数达到时更新参数
        if (batch_idx + 1) % accumulation_steps == 0:
            # 梯度裁剪（更严格的裁剪，防止梯度爆炸）
            skip_update_manual = False
            if config.grad_clip_norm > 0:
                # 诊断：在裁剪前检查梯度是否存在NaN/Inf
                bad_grads = []
                for name, p in model.named_parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        bad_grads.append(name)
                if bad_grads:
                    logger.warning(
                        f"批次 {batch_idx + 1}: 检测到非有限梯度 | indices={batch_indices_str} | bad_params={bad_grads[:5]}"
                    )
                    skip_update_manual = True

                if scaler is not None:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    if _log_nonfinite_tensor(
                        logger=logger,
                        tensor=grad_norm,
                        name="grad_norm",
                        batch_idx=batch_idx,
                        stage="grad_clip",
                    ):
                        logger.warning(
                            f"批次 {batch_idx + 1}: 检测到无效梯度 (norm={grad_norm:.2f})，Scaler将自动处理跳过 | indices={batch_indices_str}"
                        )
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
                    if _log_nonfinite_tensor(
                        logger=logger,
                        tensor=grad_norm,
                        name="grad_norm",
                        batch_idx=batch_idx,
                        stage="grad_clip",
                    ):
                        logger.warning(f"批次 {batch_idx + 1}: 检测到无效梯度，手动跳过更新 | indices={batch_indices_str}")
                        skip_update_manual = True
            if skip_update_manual:
                skipped_grad_nan += 1
       
            if skip_update_manual:
                optimizer.zero_grad()
                if discriminator is not None and optimizer_d is not None:
                    optimizer_d.zero_grad()
            else:
                if scaler is not None:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
            optimizer.zero_grad()
 
            
            # 【Phase 3】训练判别器（如果启用对抗训练）
            if (
                not skip_update_manual
                and discriminator is not None
                and optimizer_d is not None
                and gan_active
            ):
                for _ in range(d_updates_per_g):
                    optimizer_d.zero_grad()
                    disc_loss_real = torch.tensor(0.0, device=device)
                    disc_loss_fake = torch.tensor(0.0, device=device)
                    r1_penalty = torch.tensor(0.0, device=device)
                    with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                        if 'image' in device_targets:
                            real_image = device_targets['image']
                            if use_r1:
                                real_image = real_image.requires_grad_(True)
                            real_img_pred, _ = discriminator(image=real_image)
                            disc_loss_real = disc_loss_real + adv_loss_fn(real_img_pred, target_is_real=True)
                            if use_r1:
                                r1_penalty = r1_penalty + _compute_r1_penalty(real_img_pred, real_image)
                        if 'video' in device_targets:
                            real_video = device_targets['video']
                            if use_r1:
                                real_video = real_video.requires_grad_(True)
                            _, video_real_pred = discriminator(video=real_video)
                            disc_loss_real = disc_loss_real + adv_loss_fn(video_real_pred, target_is_real=True)
                            if use_r1:
                                r1_penalty = r1_penalty + _compute_r1_penalty(video_real_pred, real_video)
                        if 'image_decoded' in results:
                            image_fake_pred, _ = discriminator(image=results['image_decoded'].detach())
                            disc_loss_fake = disc_loss_fake + adv_loss_fn(image_fake_pred, target_is_real=False)
                        if 'video_decoded' in results:
                            _, video_fake_pred = discriminator(video=results['video_decoded'].detach())
                            disc_loss_fake = disc_loss_fake + adv_loss_fn(video_fake_pred, target_is_real=False)
                        disc_loss = (disc_loss_real + disc_loss_fake) * 0.5
                        if use_r1:
                            disc_loss = disc_loss + 0.5 * r1_gamma * r1_penalty
                    if scaler is not None:
                        scaler.scale(disc_loss).backward()
                    else:
                        disc_loss.backward()
                    if config.grad_clip_norm > 0:
                        if scaler is not None:
                            scaler.unscale_(optimizer_d)
                        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), config.grad_clip_norm)
                    if scaler is not None:
                        scaler.step(optimizer_d)
                    else:
                        optimizer_d.step()
                    disc_loss_real_value = disc_loss_real.item()
                    disc_loss_fake_value = disc_loss_fake.item()

     
            if scaler is not None:
                scaler.update()
            
            # 清零梯度
            optimizer.zero_grad()
        
        # 更新指标
        elapsed_time = time.time() - start_time
        meters['time'].update(elapsed_time)
        # 恢复原始损失值用于记录（因为total_loss已经被除以accumulation_steps）
        #loss_value = loss_dict['total_loss'].item() * accumulation_steps
        loss_value = loss_dict['total_loss'].item()
        meters['loss'].update(loss_value)
        
        if 'text_loss' in loss_dict:
            meters['text_loss'].update(loss_dict.get('text_loss', 0.0))
        if 'image_loss' in loss_dict:
            meters['image_loss'].update(loss_dict.get('image_loss', 0.0))
        if 'video_loss' in loss_dict:
            meters['video_loss'].update(loss_dict.get('video_loss', 0.0))
        if 'image_recon_loss' in loss_dict:
            meters['image_recon_loss'].update(loss_dict.get('image_recon_loss', 0.0))
        if 'image_percep_loss' in loss_dict:
            meters['image_percep_loss'].update(loss_dict.get('image_percep_loss', 0.0))
        if 'video_recon_loss_l1' in loss_dict:
            meters['video_recon_loss'].update(loss_dict.get('video_recon_loss_l1', 0.0))
        if 'video_percep_loss' in loss_dict:
            meters['video_percep_loss'].update(loss_dict.get('video_percep_loss', 0.0))
        if 'video_temporal_loss_l1' in loss_dict:
            meters['video_temporal_loss'].update(loss_dict.get('video_temporal_loss_l1', 0.0))
        if 'condition_margin_loss' in loss_dict:
            meters['condition_margin_loss'].update(loss_dict.get('condition_margin_loss', 0.0))
        if 'adversarial_loss' in loss_dict:
            meters['adversarial_loss'].update(loss_dict.get('adversarial_loss', 0.0))
        if disc_loss_real_value is not None:
            meters['disc_loss_real'].update(disc_loss_real_value)
        if disc_loss_fake_value is not None:
            meters['disc_loss_fake'].update(disc_loss_fake_value)
        if results.get("video_semantic_gate_mean") is not None:
            meters['video_semantic_gate_mean'].update(results.get("video_semantic_gate_mean", 0.0))
        
        # 修复OOM：在更新指标后立即清理中间变量
        del results, loss_dict, total_loss
        
        global_step += 1
        
        # 打印日志
        if (batch_idx + 1) % config.print_freq == 0:
            current_lr = optimizer.param_groups[0]['lr']
            log_msg = (
                f'Epoch [{epoch}/{config.num_epochs}] | '
                f'Step [{batch_idx + 1}/{len(train_loader)}] | '
                f'Loss {meters["loss"].val:.4f} ({meters["loss"].avg:.4f}) | '
                f'Time {meters["time"].val:.3f}s | '
                f'LR {current_lr:.6f} | '
                f'SNR {snr_db:.2f} dB'
            )
            
            if text_input is not None and meters['text_loss'].count > 0:
                log_msg += f' | TextLoss {meters["text_loss"].avg:.4f}'
            if image_input is not None and meters['image_loss'].count > 0:
                log_msg += f' | ImageLoss {meters["image_loss"].avg:.4f}'
            if video_input is not None and meters['video_loss'].count > 0:
                log_msg += f' | VideoLoss {meters["video_loss"].avg:.4f}'
            if meters['adversarial_loss'].count > 0:
                log_msg += f' | AdvLoss {meters["adversarial_loss"].avg:.4f}'
            if meters['disc_loss_real'].count > 0:
                log_msg += f' | DiscReal {meters["disc_loss_real"].avg:.4f}'
            if meters['disc_loss_fake'].count > 0:
                log_msg += f' | DiscFake {meters["disc_loss_fake"].avg:.4f}'
            
            logger.info(log_msg)
            
            # 清空部分指标（保留平均值）
            for key in ['loss', 'text_loss', 'image_loss', 'video_loss']:
                if key in meters:
                    meters[key].clear()
        #if text_input is not None:
            #del text_input
        #if image_input is not None:
            #del image_input
        #if video_input is not None:
            #del video_input
        #del device_targets
        
        # 移除torch.cuda.empty_cache()以提升训练速度
    
    # 返回平均指标
    avg_metrics = {key: meter.avg for key, meter in meters.items()}
    avg_metrics['skipped_loss_nan'] = skipped_loss_nan
    avg_metrics['skipped_grad_nan'] = skipped_grad_nan
    return avg_metrics


def validate(
    model: MultimodalJSCC,
    val_loader: DataLoader,
    loss_fn: MultimodalLoss,
    config: TrainingConfig,
    epoch: int,
    logger: logging.Logger,
    device: torch.device,
    snr_db: Optional[float] = None
) -> Dict[str, float]:
    """
    验证函数
    
    Args:
        snr_db: 如果为None，使用config中的默认SNR
    
    Returns:
        Dict[str, float]: 验证指标字典
    """
    model.eval()
    
    if snr_db is None:
        snr_db = config.snr_db
    
    # 指标记录器
    meters = {
        'loss': AverageMeter(),
        'text_loss': AverageMeter(),
        'image_loss': AverageMeter(),
        'video_loss': AverageMeter(),
        'image_psnr': AverageMeter(),
        'time': AverageMeter()
    }
    

    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            #model.reset_hidden_states()
            real_model = unwrap_model(model)
            if hasattr(real_model, 'reset_hidden_states'):
                real_model.reset_hidden_states()
            start_time = time.time()
            inputs = batch['inputs']
            targets = batch['targets']
            attention_mask = batch.get('attention_mask', None)
            text_input = inputs.get('text_input', None)
            image_input = inputs.get('image_input', None)
            video_input = inputs.get('video_input', None)
            if text_input is not None: text_input = text_input.to(device, non_blocking=True)
            if attention_mask is not None: attention_mask = attention_mask.to(device, non_blocking=True)
            if image_input is not None: image_input = image_input.to(device, non_blocking=True)
            if video_input is not None: video_input = video_input.to(device, non_blocking=True)
            device_targets = {}
            for key, value in targets.items():
                if value is not None:
                    device_targets[key] = value.to(device, non_blocking=True)
            use_amp = config.use_amp and device.type == "cuda"
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                results = model(
                    text_input=text_input,
                    image_input=image_input,
                    video_input=video_input,
                    text_attention_mask=attention_mask,
                    snr_db=snr_db
                )
                loss_dict = loss_fn(results, device_targets, attention_mask)
            meters['loss'].update(loss_dict['total_loss'].item())
            if 'text_loss' in loss_dict: meters['text_loss'].update(loss_dict['text_loss'])
            if 'image_loss' in loss_dict: meters['image_loss'].update(loss_dict['image_loss'])
            if 'video_loss' in loss_dict: meters['video_loss'].update(loss_dict['video_loss'])
            if 'image_decoded' in results and 'image' in device_targets:
                batch_psnr = calculate_multimodal_metrics(
                    {'image_decoded': results['image_decoded']},
                    {'image': device_targets['image']},
                    imagenet_normalized=getattr(config, "normalize", False),
                ).get('image_psnr', 0.0)
                meters['image_psnr'].update(batch_psnr, n=inputs['image_input'].size(0))
            meters['time'].update(time.time() - start_time)
            del results, loss_dict, device_targets

        if dist.is_initialized():
            for meter in meters.values():
                stats = torch.tensor([meter.sum, meter.count], device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                meter.sum = stats[0].item()
                meter.count = int(stats[1].item())
                meter.avg = meter.sum / meter.count if meter.count > 0 else 0
                meter.val = meter.avg

        avg_metrics = {key: meter.avg for key, meter in meters.items()}
        return avg_metrics



def main():
    parser = argparse.ArgumentParser(description='多模态JSCC训练脚本')
    parser.add_argument('--data-dir', type=str, required=True, help='数据目录')
    parser.add_argument('--train-manifest', type=str, default=None, help='训练数据清单路径（相对于data-dir）')
    parser.add_argument('--val-manifest', type=str, default=None, help='验证数据清单路径（相对于data-dir）')
    parser.add_argument('--resume', type=str, default=None, help='恢复训练的检查点路径')
    parser.add_argument('--batch-size', type=int, default=None, help='批次大小')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数')
    parser.add_argument('--max-samples', type=int, default=None, help='最大训练样本数（用于快速训练，None表示使用全部数据）')
    parser.add_argument('--max-val-samples', type=int, default=None, help='最大验证样本数（None表示使用全部数据）')
    parser.add_argument('--video-clip-len', type=int, default=None, help='视频训练clip长度')
    parser.add_argument('--video-stride', type=int, default=None, help='视频滑窗stride')
    parser.add_argument('--train-snr-random', action='store_true', help='训练时启用随机SNR')
    parser.add_argument(
        '--train-snr-strategy',
        type=str,
        default=None,
        choices=["random", "curriculum", "fixed"],
        help='训练SNR策略（默认使用配置）',
    )
    parser.add_argument('--train-snr-min', type=float, default=None, help='训练随机SNR最小值')
    parser.add_argument('--train-snr-max', type=float, default=None, help='训练随机SNR最大值')
    parser.add_argument('--use-text-guidance-image', action='store_true', help='启用文本语义引导图像重建')
    parser.add_argument('--use-text-guidance-video', action='store_true', help='启用文本语义引导视频重建')
    parser.add_argument(
        '--video-sampling-strategy',
        type=str,
        default=None,
        choices=["contiguous_clip", "uniform", "fixed_start"],
        help='训练视频采样策略',
    )
    parser.add_argument(
        '--video-eval-sampling-strategy',
        type=str,
        default=None,
        choices=["contiguous_clip", "uniform", "fixed_start"],
        help='验证视频采样策略',
    )
    parser.add_argument(
        '--video-decode-chunk-size',
        type=int,
        default=None,
        help='视频解码分段大小（仅U-Net解码器生效）',
    )
    parser.add_argument(
        '--video-latent-downsample-factor',
        type=int,
        default=None,
        help='视频潜空间总下采样倍率（2的幂）',
    )
    parser.add_argument(
        '--video-entropy-quantile-sample-size',
        type=int,
        default=None,
        help='视频熵模型分位数采样大小',
    )
    parser.add_argument(
        '--video-entropy-max-exact-quantile-elems',
        type=int,
        default=None,
        help='视频熵模型精确分位数最大元素数',
    )
    parser.add_argument(
        '--use-amp',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='启用混合精度训练（AMP）',
    )
    parser.add_argument(
        '--use-gradient-checkpointing',
        action=argparse.BooleanOptionalAction,
        default=None,
        help='启用梯度检查点以节省显存',
    )
    parser.add_argument(
        '--image-decoder-type',
        type=str,
        choices=['baseline', 'generative'],
        default=None,
        help='图像解码器类型',
    )
    parser.add_argument(
        '--generator-type',
        type=str,
        default=None,
        help='生成器类型（默认vae）',
    )
    parser.add_argument(
        '--generator-ckpt',
        type=str,
        default=None,
        help='生成器权重路径（VAE等）',
    )
    parser.add_argument('--z-channels', type=int, default=None, help='生成器latent通道数')
    parser.add_argument('--latent-down', type=int, default=None, help='生成器latent下采样倍率')
    parser.add_argument('--generative-gamma1', type=float, default=None, help='生成式图像MSE权重')
    parser.add_argument('--generative-gamma2', type=float, default=None, help='生成式图像LPIPS权重')
    parser.add_argument('--local-rank', type=int, default=None, help='分布式训练的本地进程rank')
    parser.add_argument('--distributed', action='store_true', help='启用分布式训练')
    args = parser.parse_args()
    
    # 设置随机种子
    seed_torch(42)
    
    # 创建配置
    config = TrainingConfig()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_rank = args.local_rank
    if local_rank is None and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = args.distributed or world_size > 1 or local_rank is not None
    if is_distributed:
        if local_rank is None:
            local_rank = 0
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        config.device = torch.device("cuda", local_rank)
        is_main_process = dist.get_rank() == 0
    else:
        is_main_process = True
    
    # 更新配置
    config.data_dir = args.data_dir
    
    # 设置manifest路径
    if args.train_manifest:
        config.train_manifest = os.path.join(config.data_dir, args.train_manifest)
    else:
        default_train_v2 = os.path.join(config.data_dir, "train_manifest_v2.json")
        config.train_manifest = default_train_v2 if os.path.exists(default_train_v2) else os.path.join(config.data_dir, 'train_manifest.json')
    
    if args.val_manifest:
        config.val_manifest = os.path.join(config.data_dir, args.val_manifest)
    else:
        default_val_v2 = os.path.join(config.data_dir, "val_manifest_v2.json")
        config.val_manifest = default_val_v2 if os.path.exists(default_val_v2) else os.path.join(config.data_dir, 'val_manifest.json')
    
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.video_clip_len:
        config.video_clip_len = args.video_clip_len
        config.max_video_frames = args.video_clip_len
        config.video_num_frames = args.video_clip_len
    if args.video_stride:
        config.video_stride = args.video_stride
    if args.video_sampling_strategy:
        config.video_sampling_strategy = args.video_sampling_strategy
    if args.video_eval_sampling_strategy:
        config.video_eval_sampling_strategy = args.video_eval_sampling_strategy
    if args.video_decode_chunk_size is not None:
        config.video_decode_chunk_size = args.video_decode_chunk_size
    if args.video_latent_downsample_factor is not None:
        config.video_latent_downsample_factor = args.video_latent_downsample_factor
        config.video_latent_downsample_stride = args.video_latent_downsample_factor
    if args.video_entropy_quantile_sample_size is not None:
        config.video_entropy_quantile_sample_size = args.video_entropy_quantile_sample_size
    if args.video_entropy_max_exact_quantile_elems is not None:
        config.video_entropy_max_exact_quantile_elems = args.video_entropy_max_exact_quantile_elems
    if args.use_amp is not None:
        config.use_amp = args.use_amp
    if args.use_gradient_checkpointing is not None:
        config.use_gradient_checkpointing = args.use_gradient_checkpointing
    if args.image_decoder_type:
        config.image_decoder_type = args.image_decoder_type
    if args.generator_type:
        config.generator_type = args.generator_type
    if args.generator_ckpt:
        config.generator_ckpt = args.generator_ckpt
    if args.z_channels is not None:
        config.z_channels = args.z_channels
    if args.latent_down is not None:
        config.latent_down = args.latent_down
    if args.generative_gamma1 is not None:
        config.generative_gamma1 = args.generative_gamma1
    if args.generative_gamma2 is not None:
        config.generative_gamma2 = args.generative_gamma2
    if args.train_snr_random:
        config.train_snr_strategy = "random"
        config.train_snr_random = True
    if args.train_snr_strategy:
        config.train_snr_strategy = args.train_snr_strategy
    if args.train_snr_min is not None:
        config.train_snr_min = args.train_snr_min
    if args.train_snr_max is not None:
        config.train_snr_max = args.train_snr_max
    if args.use_text_guidance_image:
        config.use_text_guidance_image = True
    if args.use_text_guidance_video:
        config.use_text_guidance_video = True
    
    # 配置日志
    if is_main_process:
        makedirs(config.save_dir)
        makedirs(config.log_dir)
        log_config = type('LogConfig', (), {
            'workdir': config.log_dir,
            'log': config.log_file,
            'samples': os.path.join(config.save_dir, 'samples'),
            'models': os.path.join(config.save_dir, 'models')
        })()
        logger = logger_configuration(log_config, save_log=True)
    else:
        logger = logging.getLogger('multimodal_jscc')
        logger.addHandler(logging.NullHandler())
        logger.disabled = True
    
    _align_video_gop_size(config, logger if is_main_process else None)

    logger.info("=" * 80)
    logger.info("多模态JSCC训练脚本")
    logger.info("=" * 80)
    logger.info(f"运行名称: {config.run_name}")
    logger.info(f"设备: {config.device}")
    logger.info(f"数据目录: {config.data_dir}")
    logger.info(f"训练清单: {config.train_manifest}")
    logger.info(f"验证清单: {config.val_manifest}")
    logger.info(f"图像解码器类型: {getattr(config, 'image_decoder_type', 'baseline')}")
    logger.info(
        "视频采样设置: clip_len=%d stride=%d train_strategy=%s val_strategy=%s",
        config.video_clip_len,
        config.video_stride,
        config.video_sampling_strategy,
        config.video_eval_sampling_strategy,
    )
    
    # 加载数据清单
    logger.info("加载数据清单...")
    train_data_list = load_manifest(config.train_manifest)
    val_data_list = load_manifest(config.val_manifest)
    
    if not train_data_list:
        logger.error(f"训练数据清单为空或文件不存在: {config.train_manifest}")
        return
    if not val_data_list:
        logger.warning(f"验证数据清单为空或文件不存在: {config.val_manifest}")
    
    # 数据采样（用于快速训练）
    original_train_size = len(train_data_list)
    if args.max_samples and args.max_samples > 0 and args.max_samples < len(train_data_list):
        import random
        random.seed(42)
        train_data_list = random.sample(train_data_list, args.max_samples)
        logger.info(f"数据采样: 从 {original_train_size} 个样本中采样 {len(train_data_list)} 个用于训练")
    
    original_val_size = len(val_data_list) if val_data_list else 0
    if args.max_val_samples and args.max_val_samples > 0 and val_data_list and args.max_val_samples < len(val_data_list):
        import random
        random.seed(42)
        val_data_list = random.sample(val_data_list, args.max_val_samples)
        logger.info(f"验证数据采样: 从 {original_val_size} 个样本中采样 {len(val_data_list)} 个用于验证")
    
    logger.info(f"训练样本数: {len(train_data_list)}")
    logger.info(f"验证样本数: {len(val_data_list) if val_data_list else 0}")
    def _log_manifest_stats(name: str, data_list):
        if not data_list:
            return
        vids = set()
        captions = []
        keyframes = []
        for item in data_list:
            vid = item.get("meta", {}).get("video_id") or os.path.basename(item.get("video", {}).get("file", ""))
            if vid:
                vids.add(vid)
            text_info = item.get("text", {})
            captions.append(len(text_info.get("texts", [])) if "texts" in text_info else (1 if "text" in text_info else 0))
            image_info = item.get("image", {})
            keyframes.append(len(image_info.get("files", [])) if "files" in image_info else (1 if "file" in image_info else 0))
        logger.info(f"{name} unique videos: {len(vids)}, captions/vid mean={np.mean(captions):.2f}, keyframes/vid mean={np.mean(keyframes):.2f}")
    _log_manifest_stats("训练", train_data_list)
    _log_manifest_stats("验证", val_data_list)
    
    # 计算每个epoch的迭代数
    iterations_per_epoch = (len(train_data_list) + config.batch_size - 1) // config.batch_size
    estimated_time_per_epoch = iterations_per_epoch * 0.5  # 假设每个batch约0.5秒
    logger.info(f"每个epoch迭代数: {iterations_per_epoch}")
    logger.info(f"预计每个epoch时间: {estimated_time_per_epoch / 60:.1f} 分钟")
    
    # 创建模型
    logger.info("创建模型...")
    model = create_model(config)
    model = model.to(config.device)
    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=getattr(config, 'ddp_find_unused_parameters', True)
        )
    config.print_config(logger)
    print_model_structure_info(unwrap_model(model), logger)
    #if args.resume:
        #logger.info(f"从检查点恢复: {args.resume}")
        #checkpoint = torch.load(args.resume, map_location=config.device)
        #state_dict = checkpoint['model_state_dict']
        #if not check_state_dict_compatibility(model, state_dict, logger):
            #logger.error("检测到维度不匹配，终止训练以防止错误传播！")
            #logger.error("请检查 config.py 中的 img_embed_dims 是否与 checkpoint 中的一致。")
            #sys.exit(1)
        #model.load_state_dict(state_dict)
    
    # 【Phase 1】如果启用预训练，记录信息
    if getattr(config, 'pretrained', False):
        real_model = unwrap_model(model)
        image_encoder = getattr(real_model, "image_encoder", None)
        image_pretrained_active = bool(getattr(image_encoder, "pretrained", False))
        if image_pretrained_active:
            logger.info(
                "【Phase 1】使用预训练权重: "
                f"{getattr(config, 'pretrained_model_name', 'swin_tiny_patch4_window7_224')}"
            )
            if getattr(config, 'freeze_encoder', False):
                logger.info("【Phase 1】编码器主干已冻结，仅训练适配器层")
        else:
            logger.info(
                "【Phase 1】预训练权重在配置中启用，但当前图像编码器未加载预训练权重，"
                "将使用随机初始化。"
            )
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"模型总参数: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")
    logger.info(f"模型大小: {total_params * 4 / (1024**2):.2f} MB")
    
    # 创建损失函数
    logger.info("创建损失函数...")
    loss_fn = create_loss_fn(config)
    loss_fn = loss_fn.to(config.device)
    
    # 【Phase 3】创建判别器（如果启用对抗训练）
    discriminator = None
    optimizer_d = None
    if getattr(config, 'use_adversarial', False):
        logger.info("创建判别器...")
        from discriminator import MultimodalDiscriminator
        discriminator = MultimodalDiscriminator(
            image_input_nc=3,
            video_input_nc=3,
            ndf=64,
            n_layers=3
        )
        discriminator = discriminator.to(config.device)
        if is_distributed:
            discriminator = DDP(
                discriminator,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=getattr(config, 'ddp_find_unused_parameters', True)
            )
        optimizer_d = optim.Adam(
            discriminator.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        logger.info("启用对抗训练（GAN Loss）")
    
    # 创建优化器
    logger.info("创建优化器...")
    optimizer = create_optimizer(model, config)
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, config)
    
    # 创建混合精度训练的scaler（如果启用）
    scaler = None
    if config.use_amp and config.device.type == "cuda":
        scaler = torch.amp.GradScaler("cuda")
        logger.info("启用混合精度训练（AMP），GradScaler 已启用。")
    elif config.use_amp:
        logger.warning("检测到非CUDA设备，已禁用混合精度训练（AMP）。")
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    data_loader_manager = MultimodalDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        shuffle=True,
        image_size=config.img_size,
        max_text_length=config.max_text_length,
        max_video_frames=config.max_video_frames,
        video_clip_len=config.video_clip_len,
        video_stride=config.video_stride,
        video_sampling_strategy=config.video_sampling_strategy,
        video_eval_sampling_strategy=config.video_eval_sampling_strategy,
        prefetch_factor=getattr(config, 'prefetch_factor', 2),
        allow_missing_modalities=getattr(config, "allow_missing_modalities", False),
        strict_mode=getattr(config, "strict_data_loading", True),
        required_modalities=("video", "text"),
        normalize=getattr(config, "normalize", False),
        seed=config.seed,
    )
    
    # 创建训练数据集和加载器
    train_dataset = data_loader_manager.create_dataset(train_data_list, is_train=True)
    train_sampler = None
    collate_fn = partial(
        collate_multimodal_batch,
        allow_missing_modalities=data_loader_manager.allow_missing_modalities,
        required_modalities=("video", "text"),
    )
    if is_distributed:
        train_sampler = DistributedSampler(train_dataset)
        actual_prefetch_factor = getattr(data_loader_manager, 'prefetch_factor', 2)
        if data_loader_manager.num_workers == 0:
            actual_prefetch_factor = None
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            sampler=train_sampler,
            shuffle=False,
            num_workers=data_loader_manager.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            prefetch_factor=actual_prefetch_factor,
            persistent_workers=True if data_loader_manager.num_workers > 0 else False
        )
    else:
        train_loader = data_loader_manager.create_dataloader(train_dataset, shuffle=True)
    
    # 创建验证数据集和加载器
    val_loader = None
    val_sampler = None
    if val_data_list:
        val_dataset = data_loader_manager.create_dataset(val_data_list, is_train=False)
        if is_distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False)
            actual_prefetch_factor = getattr(data_loader_manager, 'prefetch_factor', 2)
            if data_loader_manager.num_workers == 0:
                actual_prefetch_factor = None
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=data_loader_manager.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
                prefetch_factor=actual_prefetch_factor,
                persistent_workers=True if data_loader_manager.num_workers > 0 else False
            )
        else:
            val_loader = data_loader_manager.create_dataloader(val_dataset, shuffle=False)
    
    # 恢复训练（如果指定）
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        logger.info(f"从检查点恢复: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=config.device)
        state_dict = checkpoint['model_state_dict']
        model_to_load = unwrap_model(model)
        if not check_state_dict_compatibility(model_to_load, state_dict, logger):
            logger.error("!!! 维度不匹配，拒绝加载 Checkpoint !!!")
            sys.exit(1)
        model_to_load.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        logger.info(f"从epoch {start_epoch}恢复训练")
        #model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #if scheduler and 'scheduler_state_dict' in checkpoint:
            #scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #start_epoch = checkpoint['epoch'] + 1
        #best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        #logger.info(f"从epoch {start_epoch}恢复训练")
    
    # 保存模型配置到检查点
    model_config = {
        'vocab_size': config.vocab_size,
        'text_embed_dim': config.text_embed_dim,
        'text_num_heads': config.text_num_heads,
        'text_num_layers': config.text_num_layers,
        'text_output_dim': config.text_output_dim,
        'img_size': config.img_size,
        'patch_size': config.patch_size,
        'img_embed_dims': config.img_embed_dims,
        'img_depths': config.img_depths,
        'img_num_heads': config.img_num_heads,
        'img_output_dim': config.img_output_dim,
        'video_hidden_dim': config.video_hidden_dim,
        'video_num_frames': config.video_num_frames,
        'video_use_optical_flow': config.video_use_optical_flow,
        'video_use_convlstm': config.video_use_convlstm,
        'video_output_dim': config.video_output_dim,
        'video_gop_size': getattr(config, "video_gop_size", None),
        'channel_type': config.channel_type,
        'normalize_inputs': getattr(config, "normalize", False),
        'pretrained': getattr(config, 'pretrained', False),
        'freeze_encoder': getattr(config, 'freeze_encoder', False),
        'pretrained_model_name': getattr(config, 'pretrained_model_name', None),
        'image_decoder_type': getattr(config, 'image_decoder_type', 'baseline'),
        'generator_type': getattr(config, 'generator_type', 'vae'),
        'generator_ckpt': getattr(config, 'generator_ckpt', None),
        'z_channels': getattr(config, 'z_channels', 4),
        'latent_down': getattr(config, 'latent_down', 8),
        'video_clip_len': config.video_clip_len,
        'video_stride': config.video_stride,
        'video_sampling_strategy': config.video_sampling_strategy,
        'video_eval_sampling_strategy': config.video_eval_sampling_strategy,
        'snr_db': config.snr_db,
        'train_snr_random': config.train_snr_random,
        'train_snr_strategy': getattr(config, "train_snr_strategy", "fixed"),
        'train_snr_min': config.train_snr_min,
        'train_snr_max': config.train_snr_max,
        'bandwidth_ratio_start': getattr(config, "bandwidth_ratio_start", 1.0),
        'bandwidth_ratio_end': getattr(config, "bandwidth_ratio_end", 1.0),
        'bandwidth_warmup_epochs': getattr(config, "bandwidth_warmup_epochs", 0),
        'bandwidth_anneal_epochs': getattr(config, "bandwidth_anneal_epochs", 0),

    }
    
    # 训练循环
    logger.info("开始训练...")
    logger.info("=" * 80)
    
    for epoch in range(start_epoch, config.num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{config.num_epochs}")
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model_to_save = unwrap_model(model)
        current_bandwidth_ratio = compute_bandwidth_ratio(config, epoch)
        if hasattr(model_to_save, "set_bandwidth_ratio"):
            model_to_save.set_bandwidth_ratio(current_bandwidth_ratio)
        logger.info(
            "带宽门控 ratio=%.4f (start=%.4f end=%.4f warmup=%d anneal=%d)",
            current_bandwidth_ratio,
            getattr(config, "bandwidth_ratio_start", 1.0),
            getattr(config, "bandwidth_ratio_end", 1.0),
            getattr(config, "bandwidth_warmup_epochs", 0),
            getattr(config, "bandwidth_anneal_epochs", 0),
        )
        
        # 训练
        train_metrics = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            config=config,
            epoch=epoch,
            logger=logger,
            device=config.device,
            scaler=scaler,
            discriminator=discriminator,  # 【Phase 3】传递判别器
            optimizer_d=optimizer_d  # 【Phase 3】传递判别器优化器
        )
        
        logger.info(
            f"训练完成 | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"Time: {train_metrics['time']:.2f}s"
        )
        if train_metrics.get('skipped_loss_nan', 0) or train_metrics.get('skipped_grad_nan', 0):
            logger.info(
                "跳过统计 | "
                f"Loss NaN/Inf: {train_metrics.get('skipped_loss_nan', 0)} | "
                f"Grad NaN/Inf: {train_metrics.get('skipped_grad_nan', 0)}"
            )
        
        # 更新学习率
        if scheduler:
            if config.lr_scheduler == 'plateau':
                scheduler.step(train_metrics.get('loss', 0.0))
            else:
                scheduler.step()
        
        # 验证
        if val_loader is not None and (epoch + 1) % config.val_freq == 0:
            if val_sampler is not None:
                val_sampler.set_epoch(epoch)
            logger.info("开始验证...")
            val_metrics = validate(
                model=model,
                val_loader=val_loader,
                loss_fn=loss_fn,
                config=config,
                epoch=epoch,
                logger=logger,
                device=config.device
            )
            
            logger.info(
                f"验证完成 | "
                f"Loss: {val_metrics['loss']:.4f} | "
                f"Time: {val_metrics['time']:.2f}s"
            )
            
            # 打印详细指标
            if 'image_psnr' in val_metrics:
                logger.info(f"图像 PSNR: {val_metrics['image_psnr']:.2f} dB")
            if 'image_ssim' in val_metrics:
                logger.info(f"图像 SSIM: {val_metrics['image_ssim']:.4f}")
            if 'video_psnr_mean' in val_metrics:
                logger.info(f"视频平均 PSNR: {val_metrics['video_psnr_mean']:.2f} dB")
            if 'text_bleu' in val_metrics:
                logger.info(f"文本 BLEU: {val_metrics['text_bleu']:.4f}")
            # 保存最佳模型
            if is_main_process and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_path = os.path.join(config.save_dir, 'best_model.pth')
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model_to_save.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'val_metrics': val_metrics,
                    'model_config': model_config
                }
                if scheduler:
                    checkpoint['scheduler_state_dict'] = scheduler.state_dict()
                torch.save(checkpoint, best_model_path)
                logger.info(f"保存最佳模型到: {best_model_path}")
        
        # 定期保存检查点
        if is_main_process and (epoch + 1) % config.save_freq == 0:
            checkpoint_path = os.path.join(config.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'model_config': model_config
            }
            if scheduler:
                checkpoint['scheduler_state_dict'] = scheduler.state_dict()
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"保存检查点到: {checkpoint_path}")
    
    logger.info("=" * 80)
    logger.info("训练完成！")
    logger.info(f"最佳验证损失: {best_val_loss:.4f}")
    logger.info(f"模型保存在: {config.save_dir}")
    if is_distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

