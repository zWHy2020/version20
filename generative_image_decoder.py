"""
生成式图像JSCC解码器

通过预测生成器所需的latent，再调用冻结的生成器重建图像。
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from image_encoder import (
    BasicLayer,
    PatchReverseMerging,
    SNRModulator,
    TextModulator,
    IMAGENET_MEAN,
    IMAGENET_STD,
    _log_nonfinite,
)
from generator_adapter import GeneratorAdapter


class GenerativeImageJSCCDecoder(nn.Module):
    """
    生成式图像JSCC解码器

    复用ImageJSCCDecoder主干，输出latent并由生成器重建图像。
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
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_layer: nn.Module = nn.LayerNorm,
        input_dim: int = 256,
        guide_dim: Optional[int] = None,
        semantic_context_dim: int = 256,
        normalize_output: bool = False,
        use_gradient_checkpointing: bool = True,
        z_channels: int = 4,
        latent_down: int = 8,
        generator_adapter: Optional[GeneratorAdapter] = None,
    ) -> None:
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dims = embed_dims
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.normalize_output = normalize_output
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.z_channels = z_channels
        self.latent_down = latent_down
        self.register_buffer("output_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("output_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embed_dims[-1] // 2),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dims[-1] // 2, embed_dims[-1]),
        )

        self.layers = nn.ModuleList()
        self.snr_modulators = nn.ModuleList()
        self.text_modulators = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dims[-(i_layer + 1)],
                out_dim=embed_dims[-(i_layer + 2)] if (i_layer < self.num_layers - 1) else None,
                input_resolution=(
                    self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                    self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer)),
                ),
                depth=depths[-(i_layer + 1)],
                num_heads=num_heads[-(i_layer + 1)],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_path_rate * (i_layer / (self.num_layers - 1)),
                norm_layer=norm_layer,
                downsample=PatchReverseMerging if (i_layer < self.num_layers - 1) else None,
            )
            current_dim = embed_dims[-(i_layer + 1)]
            self.snr_modulators.append(SNRModulator(current_dim))
            self.text_modulators.append(TextModulator(current_dim, semantic_context_dim))
            self.layers.append(layer)

        self.norm = norm_layer(embed_dims[0])

        self.latent_head = nn.Conv2d(embed_dims[0], z_channels, kernel_size=3, padding=1)

        in_guide_dim = guide_dim if guide_dim is not None else (embed_dims[-1] // 4)
        self.guide_processor = nn.Sequential(
            nn.Linear(in_guide_dim, embed_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(embed_dims[-1] // 2, embed_dims[-1]),
        )

        self.semantic_context_dim = semantic_context_dim
        self.semantic_aligner = nn.Linear(semantic_context_dim, embed_dims[-1])

        from cross_attention import CrossAttention

        self.semantic_fusion = CrossAttention(
            embed_dim=embed_dims[-1],
            num_heads=num_heads[-1],
            dropout=drop_rate,
            text_embed_dim=None,
        )

        self.semantic_gate = nn.Sequential(
            nn.Linear(embed_dims[-1] * 2, embed_dims[-1] // 4),
            nn.ReLU(),
            nn.Linear(embed_dims[-1] // 4, 1),
            nn.Sigmoid(),
        )
        self.guide_gate = nn.Parameter(torch.ones(1) * 0.5)

        if generator_adapter is None:
            raise ValueError("generator_adapter is required for GenerativeImageJSCCDecoder")
        self.generator = generator_adapter
        self.generator.eval()
        for param in self.generator.parameters():
            param.requires_grad = False

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
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
        if semantic_context is not None:
            text_global = semantic_context.mean(dim=1)
            actual_dim = semantic_context.shape[-1]
            expected_dim = self.semantic_aligner.in_features
            assert actual_dim == expected_dim, (
                f"维度不匹配错误: 传入的语义上下文维度为 {actual_dim},"
                f"但解码器期望的维度 (semantic_context_dim) 为 {expected_dim}。"
                f"请检查 config.py 中的 text_output_dim 是否与模型训练时一致。"
            )
        else:
            text_global = None

        x = self.input_proj(noisy_features)
        assert x.shape[-1] == self.embed_dims[-1], (
            f"GenerativeImageJSCCDecoder.input_proj 维度不匹配: got {x.shape[-1]},"
            f" expected {self.embed_dims[-1]}"
        )

        in_guide_dim = self.guide_processor[0].in_features
        assert guide_vector.shape[-1] == in_guide_dim, (
            f"GenerativeImageJSCCDecoder.guide_vector 维度不匹配: got {guide_vector.shape[-1]},"
            f" expected {in_guide_dim}"
        )
        guide_processed = self.guide_processor(guide_vector)
        assert guide_processed.shape[-1] == self.embed_dims[-1], (
            f"GenerativeImageJSCCDecoder.guide_processor 输出维度不匹配: got {guide_processed.shape[-1]},"
            f" expected {self.embed_dims[-1]}"
        )
        batch_size, seq_len = x.size(0), x.size(1)
        guide_expanded = guide_processed.unsqueeze(1).expand(batch_size, seq_len, -1)

        semantic_contexts_to_process: List[torch.Tensor] = []
        if semantic_context is not None:
            semantic_contexts_to_process.append(semantic_context)
        if multiple_semantic_contexts:
            semantic_contexts_to_process.extend(multiple_semantic_contexts)

        if semantic_contexts_to_process:
            semantic_fused_list = []
            for ctx in semantic_contexts_to_process:
                aligned_semantic = self.semantic_aligner(ctx)
                if aligned_semantic.shape[0] != batch_size:
                    raise RuntimeError(
                        "semantic_context batch mismatch in GenerativeImageJSCCDecoder: "
                        f"semantic_batch={aligned_semantic.shape[0]}, expected={batch_size}"
                    )
                semantic_fused = self.semantic_fusion(query=x, guide_vector=aligned_semantic)
                semantic_fused_list.append(semantic_fused)

            semantic_fused = (
                torch.stack(semantic_fused_list, dim=0).mean(dim=0)
                if len(semantic_fused_list) > 1
                else semantic_fused_list[0]
            )
            gate_input = torch.cat([x, semantic_fused], dim=-1)
            semantic_weight = self.semantic_gate(gate_input)
            guide_weight = torch.clamp(self.guide_gate, 0.0, 1.0)
            x = x + guide_weight * guide_expanded + semantic_weight * semantic_fused
        else:
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

        x = self.norm(x)

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
                    f"GenerativeImageJSCCDecoder重塑失败：序列长度 L={L} 不等于预期的 H*W={H}*{W}={H*W}。"
                    f"请检查上采样配置是否正确，patches_resolution={self.patches_resolution}。"
                )
            H, W = resolved

        x = x.view(B, H, W, C).permute(0, 3, 1, 2)

        latent = self.latent_head(x)
        if output_size is None:
            output_size = self.img_size
        target_latent_h = int(math.ceil(output_size[0] / self.latent_down))
        target_latent_w = int(math.ceil(output_size[1] / self.latent_down))
        if latent.shape[-2:] != (target_latent_h, target_latent_w):
            latent = F.interpolate(latent, size=(target_latent_h, target_latent_w), mode="bilinear", align_corners=False)
        _log_nonfinite("GenerativeImageJSCCDecoder.latent", latent)

        image = self.generator(latent)
        if self.normalize_output:
            image = (image - self.output_mean) / self.output_std
        if output_size is not None:
            image = image[..., : output_size[0], : output_size[1]]

        _log_nonfinite("GenerativeImageJSCCDecoder.output", image)
        return image
