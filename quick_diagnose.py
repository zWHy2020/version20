"""
快速 NaN 诊断脚本

快速检查训练过程中损失值变为 NaN 的常见原因。
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import sys

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multimodal_jscc import MultimodalJSCC
from losses import MultimodalLoss, GenerativeImageLoss
from data_loader import MultimodalDataLoader
from config import TrainingConfig
from utils import load_manifest, makedirs


def quick_check_tensor(tensor, name):
    """快速检查张量"""
    if tensor is None:
        return True, []
    
    issues = []
    if torch.isnan(tensor).any():
        issues.append(f"{name}: 包含 NaN")
    if torch.isinf(tensor).any():
        issues.append(f"{name}: 包含 Inf")
    
    if tensor.numel() > 0:
        max_val = tensor.abs().max().item()
        if max_val > 1e6:
            issues.append(f"{name}: 数值过大 (max={max_val:.2e})")
    
    return len(issues) == 0, issues


def run_self_test(device: torch.device) -> None:
    print("\n[SelfTest] 1) 构造随机batch并跑前向传播...")
    model = MultimodalJSCC(
        img_size=(224, 224),
        patch_size=4,
        image_decoder_type="generative",
        generator_ckpt=None,
        normalize_inputs=False,
    ).to(device)
    model.eval()
    images = torch.rand(2, 3, 224, 224, device=device)
    with torch.no_grad():
        outputs = model(image_input=images, snr_db=10.0)
    pred = outputs["image_decoded"]
    print(f"  输出shape: {tuple(pred.shape)}")
    if pred.shape != images.shape:
        raise RuntimeError("SelfTest失败：输出shape与输入不一致")
    pred_min = pred.min().item()
    pred_max = pred.max().item()
    print(f"  输出范围: min={pred_min:.4f}, max={pred_max:.4f}")
    if pred_min < -1e-3 or pred_max > 1.0 + 1e-3:
        raise RuntimeError("SelfTest失败：输出范围不在[0,1]")

    print("[SelfTest] 2) GenerativeImageLoss 反向传播...")
    model.train()
    pred = model(image_input=images, snr_db=10.0)["image_decoded"]
    loss_fn = GenerativeImageLoss(gamma1=1.0, gamma2=0.1, normalize=False).to(device)
    loss = loss_fn(pred, images)[0]
    loss.backward()
    print("  反向传播成功")

    print("[SelfTest] 3) 确认generator requires_grad=False...")
    generator_params = list(model.image_decoder.generator.parameters())
    if any(param.requires_grad for param in generator_params):
        raise RuntimeError("SelfTest失败：generator参数未冻结")
    print("  generator参数已冻结")


def main():
    parser = argparse.ArgumentParser(description='快速 NaN 诊断')
    parser.add_argument('--data-dir', type=str, default=None, help='数据目录')
    parser.add_argument('--batch-size', type=int, default=4, help='批次大小')
    parser.add_argument('--self-test', action='store_true', help='运行最小自测（不依赖数据）')
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if args.self_test:
        run_self_test(device)
        return
    if not args.data_dir:
        raise ValueError("--data-dir 在非 --self-test 模式下是必需的")
    
    # 创建配置
    config = TrainingConfig()
    config.data_dir = args.data_dir
    config.batch_size = args.batch_size
    config.device = device
    config.train_manifest = os.path.join(config.data_dir, 'train_manifest.json')
    
    # 加载数据
    print("\n1. 加载数据...")
    train_data_list = load_manifest(config.train_manifest)
    if not train_data_list:
        print(f"错误: 无法加载数据清单: {config.train_manifest}")
        return
    print(f"  样本数: {len(train_data_list)}")
    
    # 创建数据加载器
    data_loader_manager = MultimodalDataLoader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=False,
        image_size=config.image_size,
        max_text_length=config.max_text_length,
        max_video_frames=config.max_video_frames
    )
    train_dataset = data_loader_manager.create_dataset(train_data_list)
    train_loader = data_loader_manager.create_dataloader(train_dataset, shuffle=False)
    
    # 获取批次
    print("\n2. 检查输入数据...")
    batch = next(iter(train_loader))
    inputs = batch['inputs']
    targets = batch['targets']
    
    all_ok = True
    issues = []
    
    # 检查输入
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            ok, batch_issues = quick_check_tensor(value, f"input[{key}]")
            if not ok:
                all_ok = False
                issues.extend(batch_issues)
                print(f"  ❌ {key}: 有问题")
            else:
                print(f"  ✓ {key}: 正常")
    
    # 检查目标
    for key, value in targets.items():
        if isinstance(value, torch.Tensor):
            ok, batch_issues = quick_check_tensor(value, f"target[{key}]")
            if not ok:
                all_ok = False
                issues.extend(batch_issues)
                print(f"  ❌ {key}: 有问题")
            else:
                print(f"  ✓ {key}: 正常")
    
    if not all_ok:
        print("\n发现输入数据问题:")
        for issue in issues:
            print(f"  - {issue}")
        return
    
    # 创建模型
    print("\n3. 创建模型...")
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
        video_hidden_dim=config.video_hidden_dim,
        video_num_frames=config.video_num_frames,
        video_use_optical_flow=config.video_use_optical_flow,
        video_use_convlstm=config.video_use_convlstm,
        video_output_dim=config.video_output_dim,
        channel_type=config.channel_type,
        snr_db=config.snr_db
    )
    model = model.to(device)
    
    # 检查模型权重
    print("\n4. 检查模型权重...")
    weight_issues = []
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            weight_issues.append(f"{name}: 包含 NaN")
        if torch.isinf(param).any():
            weight_issues.append(f"{name}: 包含 Inf")
        if param.abs().max() > 1e6:
            weight_issues.append(f"{name}: 数值过大")
    
    if weight_issues:
        print("  发现权重问题:")
        for issue in weight_issues[:10]:
            print(f"    - {issue}")
        if len(weight_issues) > 10:
            print(f"    ... 还有 {len(weight_issues)-10} 个")
    else:
        print("  ✓ 模型权重正常")
    
    # 前向传播
    print("\n5. 测试前向传播...")
    model.eval()
    text_input = inputs.get('text_input', None)
    image_input = inputs.get('image_input', None)
    video_input = inputs.get('video_input', None)
    attention_mask = batch.get('attention_mask', None)
    
    if text_input is not None:
        text_input = text_input.to(device)
        attention_mask = attention_mask.to(device) if attention_mask is not None else None
    if image_input is not None:
        image_input = image_input.to(device)
    if video_input is not None:
        video_input = video_input.to(device)
    
    try:
        with torch.no_grad():
            results = model(
                text_input=text_input,
                image_input=image_input,
                video_input=video_input,
                text_attention_mask=attention_mask,
                snr_db=10.0
            )
        
        print("  前向传播成功")
        
        # 检查输出
        output_issues = []
        for key, value in results.items():
            if isinstance(value, torch.Tensor):
                ok, batch_issues = quick_check_tensor(value, f"output[{key}]")
                if not ok:
                    output_issues.extend(batch_issues)
        
        if output_issues:
            print("  发现输出问题:")
            for issue in output_issues[:10]:
                print(f"    - {issue}")
        else:
            print("  ✓ 模型输出正常")
    
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查损失函数
    print("\n6. 测试损失函数...")
    loss_fn = MultimodalLoss(
        text_weight=config.text_weight,
        image_weight=config.image_weight,
        video_weight=config.video_weight,
        reconstruction_weight=config.reconstruction_weight,
        perceptual_weight=config.perceptual_weight,
        temporal_weight=config.temporal_weight,
        data_range=1.0
    )
    loss_fn = loss_fn.to(device)
    
    device_targets = {}
    for key, value in targets.items():
        if value is not None:
            device_targets[key] = value.to(device)
    
    try:
        loss_dict = loss_fn(
            predictions=results,
            targets=device_targets,
            attention_mask=attention_mask
        )
        
        print("  损失计算成功")
        
        # 检查损失值
        loss_issues = []
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                ok, batch_issues = quick_check_tensor(value, f"loss[{key}]")
                if not ok:
                    loss_issues.extend(batch_issues)
                else:
                    print(f"    {key}: {value.item():.6f}")
        
        if loss_issues:
            print("  发现损失问题:")
            for issue in loss_issues:
                print(f"    - {issue}")
        else:
            print("  ✓ 损失值正常")
    
    except Exception as e:
        print(f"  ❌ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试反向传播
    print("\n7. 测试反向传播...")
    model.train()
    try:
        total_loss = loss_dict['total_loss']
        total_loss.backward()
        
        # 检查梯度
        grad_issues = []
        max_grad_norm = 0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                max_grad_norm = max(max_grad_norm, grad_norm)
                ok, batch_issues = quick_check_tensor(param.grad, f"grad[{name}]")
                if not ok:
                    grad_issues.extend(batch_issues)
        
        print(f"  最大梯度范数: {max_grad_norm:.4f}")
        
        if grad_issues:
            print("  发现梯度问题:")
            for issue in grad_issues[:10]:
                print(f"    - {issue}")
        elif max_grad_norm > 100:
            print(f"  ⚠ 警告: 梯度可能过大 (norm={max_grad_norm:.2f})")
        else:
            print("  ✓ 梯度正常")
    
    except Exception as e:
        print(f"  ❌ 反向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*60)
    print("快速诊断完成！")
    print("="*60)
    print("\n如果发现问题，请运行完整诊断脚本:")
    print("  python new_models/diagnose_nan.py --data-dir <data_dir> --batch-size <batch_size>")


if __name__ == '__main__':
    main()
