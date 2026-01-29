"""
多模态JSCC配置模块

包含训练和评估的配置类，便于统一管理和共享参数。
"""

import os
from datetime import datetime
from typing import Tuple, List, Optional
import logging


class TrainingConfig:
    """训练配置类"""
    
    def __init__(self):
        # 基础设置
        self.seed = 42
        self.device = None  # 将在运行时设置
        self.num_workers = 0  # 优化：增加数据加载workers以充分利用CPU资源，确保GPU流水线饱和
        self.pin_memory = True
        
        # 日志设置
        self.workdir = './checkpoints'
        self.log_dir = './logs'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f'multimodal_jscc_{timestamp}'
        self.save_dir = os.path.join(self.workdir, self.run_name)
        self.log_file = os.path.join(self.log_dir, f'{self.run_name}.log')
        
        # 训练参数
        self.num_epochs = 100
        # 优化：通过梯度检查点释放显存，提高物理batch_size以加速训练
        # 等效批次大小 = batch_size * gradient_accumulation_steps = 8 * 8 = 64（保持不变）
        self.batch_size = 32  # 优化：从1提升到8，大幅减少迭代次数
        self.learning_rate = 5e-5  # 修复：将初始学习率从1e-6提高到1e-4，以便学习率调度器能够正常工作
        self.weight_decay = 1e-4
        self.grad_clip_norm = 1.0
        self.gradient_accumulation_steps = 2  # 优化：按比例降低，保持等效批次大小=8*8=64不变
        self.use_amp = True  # 是否使用混合精度训练（自动混合精度）
        self.use_gradient_checkpointing = True  # 是否启用梯度检查点节省显存
        
        # 学习率调度
        self.lr_scheduler = 'cosine'  # 'cosine', 'step', 'plateau'
        self.lr_step_size = 30
        self.lr_gamma = 0.1
        self.lr_min = 1e-6  # 最小学习率，cosine scheduler会从learning_rate逐渐降到lr_min
        
        # 验证设置
        self.val_freq = 10  # 每N个epoch验证一次（减少验证频率以加速训练）
        self.save_freq = 20  # 每N个epoch保存一次模型（减少保存频率）
        self.print_freq = 200  # 每N个step打印一次（减少打印频率以提升速度）
        
        # 损失权重（调整以平衡各模态损失）
        self.text_weight = 0.1  # 文本损失通常较大，降低权重
        self.image_weight = 1.0
        self.video_weight = 1.0
        self.reconstruction_weight = 1.0
        self.perceptual_weight = 0.01  # 【重构】启用LPIPS感知损失以提升重建质量
        self.temporal_weight = 0.05  # 降低时序损失权重
        self.text_contrastive_weight = 0.0  # 路线1默认关闭文本-图像对比
        self.video_text_contrastive_weight = 0.1  # 【新增】视频-文本对比损失权重
        self.rate_weight = 1e-4  # 【新增】码率/能量约束权重
        self.temporal_consistency_weight = 0.02  # 【新增】视频时序一致性正则权重
        # 文本引导与条件约束（路线1默认）
        self.use_text_guidance_image = True
        self.use_text_guidance_video = False
        self.enforce_text_condition = True
        self.condition_margin_weight = 0.1
        self.condition_margin = 0.05
        self.condition_prob = 0.5
        self.condition_only_low_snr = True
        self.condition_low_snr_threshold = 5.0
        
        # SNR设置
        self.train_snr_min = -5.0  # 训练时SNR范围
        self.train_snr_max = 15.0
        self.train_snr_random = False # 是否随机SNR
        self.train_snr_strategy = "curriculum"  # random | curriculum | fixed
        self.snr_db = 10.0

        # 带宽/码率调度（通道门控）
        self.bandwidth_ratio_start = 1.0
        self.bandwidth_ratio_end = 0.75
        self.bandwidth_warmup_epochs = 15
        self.bandwidth_anneal_epochs = 50
        # 模型参数
        self.vocab_size = 65536
        self.text_embed_dim = 512
        self.text_num_heads = 8
        self.text_num_layers = 6
        self.text_output_dim = 256
        self.pretrained_model_name = 'swin_base_patch4_window7_224'
        self.img_size = (256, 512)
        self.patch_size = 4
        self.img_window_size = 7
        # 重构：减小Swin Transformer的深度和宽度以节省显存
        self.img_embed_dims = [96, 192, 384, 768]  # 从 [96, 192, 384, 768] 减小
        self.img_depths = [2, 2, 18, 2]  # 从 [2, 2, 6, 2] 减小
        self.img_num_heads = [3, 6, 12, 24]  # 从 [3, 6, 12, 24] 减小
        self.img_output_dim = 256
        self.mlp_ratio = 4.0
        self.image_decoder_type = "generative"
        self.generator_type = "vae"
        self.generator_ckpt = "stabilityai/sd-vae-ft-mse"
        self.z_channels = 4
        self.latent_down = 8
        self.generative_gamma1 = 1.0
        self.generative_gamma2 = 0.1
        
        self.video_hidden_dim = 192
        self.video_num_frames = 10
        self.video_use_optical_flow = True  # 默认启用光流用于时序对齐
        self.video_use_convlstm = True  # 默认启用ConvLSTM建模时序
        self.video_output_dim = 192
        self.video_decoder_type = "swin"
        self.video_unet_base_channels = 64
        self.video_unet_num_down = 4
        self.video_unet_num_res_blocks = 3
        self.video_decode_chunk_size = None  # 视频解码分段大小（None表示不分段）
        self.video_gop_size = 5  # GOP长度（用于分组处理，降低显存，需能整除video_clip_len）
        self.video_latent_downsample_factor = 8  # 视频潜空间总下采样倍率
        self.video_latent_downsample_stride = self.video_latent_downsample_factor  # 兼容旧字段
        self.video_entropy_max_exact_quantile_elems = 2_000_000
        self.video_entropy_quantile_sample_size = 262_144
        
        self.channel_type = "awgn"
      
        
        # 数据路径和参数
        self.data_dir = None
        self.train_manifest = None
        self.val_manifest = None
        self.max_text_length = 512
        self.max_video_frames = 10
        self.video_clip_len = self.max_video_frames
        self.video_stride = 1
        self.video_sampling_strategy = "contiguous_clip"
        self.video_eval_sampling_strategy = "uniform"
        self.max_samples = 65536  # 【Phase 4】默认最大样本数
        self.allow_missing_modalities = False
        self.strict_data_loading = True
        
        # 【Phase 1】迁移学习参数
        self.pretrained = True  # 【Phase 4】默认启用预训练权重
        self.freeze_encoder = False  # 是否冻结编码器主干（初始训练时可设为True）
        self.pretrained_model_name = 'swin_small_patch4_window7_224'  # 预训练模型名称
        
        # 【Phase 3】对抗训练参数
        self.use_adversarial = False  # 是否使用对抗训练（默认关闭，需要时可启用）
        self.gan_enable_epoch = 5  # 【Phase 6】GAN warmup: 前N个epoch只做重建
        self.d_updates_per_g = 1  # 【Phase 6】判别器更新频率（每次生成器更新对应的D更新次数）
        self.gan_weight = 0.01  # 【Phase 6】对抗损失权重（默认较小）
        self.discriminator_weight = self.gan_weight  # 兼容旧字段
        self.use_r1_regularization = False  # 【Phase 6】可选R1正则
        self.r1_gamma = 10.0  # 【Phase 6】R1正则强度
        self.ddp_find_unused_parameters = True  # DDP下允许未使用参数（用于缺失模态场景）
        self.use_quantization_noise = True  # 【新增】是否启用量化噪声模拟
        self.quantization_noise_range = 0.5  # 【新增】量化噪声范围（均匀分布 [-r, r]）
        # 输入归一化（ImageNet mean/std）
        self.normalize = True
        # 视频感知/一致性附加损失权重
        self.video_perceptual_weight = 0.01
        self.temporal_perceptual_weight = 0.02
        self.color_consistency_weight = 0.02
    def print_config(self, logger=None):
        log_func = logger.info if logger else print
        log_func("\n=== 当前生效的配置 (TrainingConfig) ===")
        log_func(f"Image Embed Dims: {self.img_embed_dims}")
        log_func(f"Image Depths: {self.img_depths}")
        log_func(f"Use Adversarial: {getattr(self, 'use_adversarial', False)}")
        log_func(
            "Video sampling: clip_len="
            f"{getattr(self, 'video_clip_len', None)} stride="
            f"{getattr(self, 'video_stride', None)} train_strategy="
            f"{getattr(self, 'video_sampling_strategy', None)} eval_strategy="
            f"{getattr(self, 'video_eval_sampling_strategy', None)}"
        )
        log_func(
            "Video GOP/latent: gop_size="
            f"{getattr(self, 'video_gop_size', None)} latent_factor="
            f"{getattr(self, 'video_latent_downsample_factor', getattr(self, 'video_latent_downsample_stride', None))}"
        )
        log_func("=========================================\n")


class EvaluationConfig:
    """评估配置类"""
    
    def __init__(self):
        # 基础设置
        self.seed = 42
        self.device = None  # 将在运行时设置
        self.num_workers = 4
        self.pin_memory = True
        
        # 日志设置
        self.log_dir = './logs'
        self.save_dir = './evaluation_results'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_name = f'evaluation_{timestamp}'
        self.log_file = os.path.join(self.log_dir, f'{self.run_name}.log')
        self.result_dir = os.path.join(self.save_dir, self.run_name)
        
        # 模型路径
        self.model_path = None  # 必须指定
        
        # 评估参数
        self.batch_size = 8
        self.test_num = None  # 测试样本数量，None表示全部
        
        # SNR和Rate列表
        self.snr_list = [-5, -10, -5, 0, 5, 10, 15, 20]
        self.rate_list = None  # Rate列表（如果模型支持），None表示使用默认rate
        self.snr_db = 10.0
        self.snr_random = False
        self.snr_min = -5.0
        self.snr_max = 15.0
        
        # Patch-based推理设置（用于处理任意尺寸图像）
        self.use_patch_inference = False
        self.patch_size = 128  # patch大小
        self.patch_overlap = 32  # patch重叠区域
        
        # 数据路径
        self.data_dir = None
        self.test_manifest = None
        
        # 图像保存设置
        self.save_images = False
        self.image_save_dir = None
        
        # 数据加载参数
        self.image_size = (256, 512)
        self.max_text_length = 512
        self.max_video_frames = 10
        self.video_clip_len = self.max_video_frames
        self.video_stride = 1
        self.video_sampling_strategy = "contiguous_clip"
        self.video_eval_sampling_strategy = "uniform"
        self.infer_window_len = None
        self.infer_window_stride = None
        self.infer_window_blend = "uniform"
        self.max_output_frames = None
        
        # 模型参数（用于加载模型时，通常从检查点恢复）
        self.vocab_size = 10000
        self.text_embed_dim = 512
        self.text_num_heads = 8
        self.text_num_layers = 6
        self.text_output_dim = 256
        
        self.img_size = (256, 512)
        self.img_patch_size = 4 
        self.mlp_ratio = 4.0 # 注意：这是模型内部的patch_size，不同于推理时的patch_size
        # 重构：减小Swin Transformer的深度和宽度以节省显存
        self.img_embed_dims = [96, 192, 384, 768]  # 从 [96, 192, 384, 768] 减小
        self.img_depths = [2, 2, 18, 2]  # 从 [2, 2, 6, 2] 减小
        self.img_num_heads = [3, 6, 12, 24]  # 从 [3, 6, 12, 24] 减小
        self.img_output_dim = 256
        self.pretrained_model_name = None  # 推理时可手动指定，与训练保持一致
        self.image_decoder_type = "generative"
        self.generator_type = "vae"
        self.generator_ckpt = "stabilityai/sd-vae-ft-mse"
        self.z_channels = 4
        self.latent_down = 8
        self.generative_gamma1 = 1.0
        self.generative_gamma2 = 0.1
        
        self.video_hidden_dim = 256
        self.video_num_frames = 10
        self.video_use_optical_flow = True  # 推理侧默认启用光流
        self.video_use_convlstm = True  # 推理侧默认启用ConvLSTM
        self.video_output_dim = 256
        self.video_decoder_type = "swin"
        self.video_unet_base_channels = 64
        self.video_unet_num_down = 4
        self.video_unet_num_res_blocks = 3
        self.video_decode_chunk_size = None
        self.use_amp = False  # 推理是否启用混合精度
        self.use_gradient_checkpointing = False  # 推理默认关闭梯度检查点
        
        self.channel_type = "awgn"
        # 文本引导与条件约束（评估侧保持与训练一致的开关）
        self.use_text_guidance_image = True
        self.use_text_guidance_video = False
        self.enforce_text_condition = True
        self.condition_prob = 0.0  # 评估默认不触发额外 condition-margin 计算
        self.condition_margin = 0.05
        self.condition_margin_weight = 0.1
        self.condition_only_low_snr = True
        self.condition_low_snr_threshold = 5.0
        # 输入归一化（ImageNet mean/std）
        self.normalize = True
        self.video_latent_downsample_factor = 8
        self.video_latent_downsample_stride = self.video_latent_downsample_factor
        self.video_entropy_max_exact_quantile_elems = 2_000_000
        self.video_entropy_quantile_sample_size = 262_144

