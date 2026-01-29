"""
跨模态交叉注意力模块

实现跨模态交叉注意力机制，用于在编码和解码阶段进行模态间的信息交互。
参考SGD-JSCC去噪器中的实现，支持多模态间的引导向量交互。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class CrossAttention(nn.Module):
    """
    跨模态交叉注意力模块（修复版）
    
    该模块实现跨模态的交叉注意力机制，支持：
    1. 编码阶段：当前模态特征作为Query，其他模态的引导向量作为Key和Value
    2. 解码阶段：当前模态的带噪特征作为Query，其他模态重建的引导向量作为Key和Value
    
    【修复】确保文本嵌入正确投影到图像特征维度，并正确融合。
    
    Args:
        embed_dim (int): 特征维度（图像特征维度）
        num_heads (int): 注意力头数
        dropout (float): Dropout概率
        bias (bool): 是否使用偏置
        text_embed_dim (int, optional): 文本嵌入维度，如果提供则添加投影层
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True,
        text_embed_dim: Optional[int] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        # 【修复】如果提供了text_embed_dim，添加投影层将文本嵌入投影到图像特征维度
        if text_embed_dim is not None and text_embed_dim != embed_dim:
            self.text_proj = nn.Linear(text_embed_dim, embed_dim, bias=bias)
        else:
            self.text_proj = None
        
        # Query投影（当前模态特征，即图像特征）
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Key和Value投影（其他模态的引导向量，即文本嵌入）
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=bias)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self, 
        query: torch.Tensor, 
        guide_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播（修复版）
        
        Args:
            query (torch.Tensor): 查询张量（图像特征） [batch_size, seq_len, embed_dim]
            guide_vector (torch.Tensor): 引导向量（文本嵌入） [batch_size, guide_len, text_embed_dim] 或 [batch_size, guide_len, embed_dim]
            attention_mask (torch.Tensor, optional): 注意力掩码
            
        Returns:
            torch.Tensor: 交叉注意力输出 [batch_size, seq_len, embed_dim]
        """
        batch_size, seq_len, _ = query.shape
        
        # 【修复】如果文本嵌入维度与图像特征维度不同，先投影
        if self.text_proj is not None:
            guide_vector = self.text_proj(guide_vector)  # [batch_size, guide_len, embed_dim]
        
        guide_len = guide_vector.shape[1]
        
        # 确保维度匹配
        assert guide_vector.shape[-1] == self.embed_dim, \
            f"引导向量维度不匹配: got {guide_vector.shape[-1]}, expected {self.embed_dim}"
        
        # 投影到Q, K, V
        q = self.q_proj(query)  # [batch_size, seq_len, embed_dim]
        kv = self.kv_proj(guide_vector)  # [batch_size, guide_len, 2*embed_dim]
        k, v = kv.chunk(2, dim=-1)  # 每个都是 [batch_size, guide_len, embed_dim]
        
        # 重塑为多头注意力格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, guide_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, guide_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数（使用float32防止混合精度下的溢出）
        # 临时转换为float32进行计算，防止溢出
        q_f32 = q.float()
        k_f32 = k.float()
        attn_scores = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * self.scale
        attn_scores = attn_scores.to(q.dtype)  # 转换回原始dtype
        
        # 应用注意力掩码
        if attention_mask is not None:
            # 扩展掩码以匹配注意力分数形状 [batch_size, num_heads, seq_len, guide_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, guide_len]
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # [B, 1, seq_len, guide_len]
            
            # 使用与attn_scores相同dtype的最小值，兼容float16
            fill_value = torch.finfo(attn_scores.dtype).min if attn_scores.dtype.is_floating_point else -1e4
            attn_scores = attn_scores.masked_fill(attention_mask == 0, fill_value)
        
        # Softmax归一化
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        attn_output = torch.matmul(attn_weights, v)
        
        # 重塑回原始格式
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim
        )
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        if return_attention:
            return output, attn_weights
        return output


class CrossAttentionFusion(nn.Module):
    """
    跨模态交叉注意力融合模块
    
    将交叉注意力输出与原始特征进行融合，支持多种融合策略：
    1. 加法融合
    2. 门控融合
    3. 残差连接
    """
    
    def __init__(
        self,
        embed_dim: int,
        fusion_type: str = "gated",  # "add", "gated", "residual", "confidence"
        dropout: float = 0.1
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.embed_dim = embed_dim
        
        if fusion_type == "gated":
            # 门控融合
            self.gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid()
            )
            self.proj = nn.Linear(embed_dim * 2, embed_dim)
        elif fusion_type == "confidence":
            # 置信度融合（动态权重）
            self.confidence_head = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.ReLU(),
                nn.Linear(embed_dim, 2)
            )
        elif fusion_type == "residual":
            # 残差连接
            self.norm = nn.LayerNorm(embed_dim)
        elif fusion_type == "add":
            # 简单加法
            pass
        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        original_features: torch.Tensor, 
        cross_attn_output: torch.Tensor
    ) -> torch.Tensor:
        """
        融合原始特征和交叉注意力输出
        
        Args:
            original_features (torch.Tensor): 原始特征 [batch_size, seq_len, embed_dim]
            cross_attn_output (torch.Tensor): 交叉注意力输出 [batch_size, seq_len, embed_dim]
            
        Returns:
            torch.Tensor: 融合后的特征
        """
        if self.fusion_type == "add":
            # 简单加法融合
            output = original_features + cross_attn_output
            
        elif self.fusion_type == "gated":
            # 门控融合
            concat_features = torch.cat([original_features, cross_attn_output], dim=-1)
            gate = self.gate(concat_features)
            projected = self.proj(concat_features)
            output = gate * projected + (1 - gate) * original_features

        elif self.fusion_type == "confidence":
            # 置信度融合
            concat_features = torch.cat([original_features, cross_attn_output], dim=-1)
            logits = self.confidence_head(concat_features)
            weights = F.softmax(logits, dim=-1)
            original_weight = weights[..., 0:1]
            cross_weight = weights[..., 1:2]
            output = original_weight * original_features + cross_weight * cross_attn_output
            
        elif self.fusion_type == "residual":
            # 残差连接
            output = self.norm(original_features + cross_attn_output)
        
        return self.dropout(output)


class MultiModalCrossAttention(nn.Module):
    """
    多模态交叉注意力模块
    
    支持三个模态（文本、图像、视频）之间的交叉注意力交互。
    每个模态可以与其他两个模态进行交叉注意力计算。
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        fusion_type: str = "gated"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 为每个模态对创建交叉注意力模块
        # 文本 -> 图像
        self.text_to_image_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.text_to_image_fusion = CrossAttentionFusion(embed_dim, fusion_type, dropout)
        
        # 文本 -> 视频
        self.text_to_video_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.text_to_video_fusion = CrossAttentionFusion(embed_dim, fusion_type, dropout)
        
        # 图像 -> 文本
        self.image_to_text_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.image_to_text_fusion = CrossAttentionFusion(embed_dim, fusion_type, dropout)
        
        # 图像 -> 视频
        self.image_to_video_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.image_to_video_fusion = CrossAttentionFusion(embed_dim, fusion_type, dropout)
        
        # 视频 -> 文本
        self.video_to_text_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.video_to_text_fusion = CrossAttentionFusion(embed_dim, fusion_type, dropout)
        
        # 视频 -> 图像
        self.video_to_image_attn = CrossAttention(embed_dim, num_heads, dropout)
        self.video_to_image_fusion = CrossAttentionFusion(embed_dim, fusion_type, dropout)
        
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: torch.Tensor,
        video_features: torch.Tensor,
        text_guide: torch.Tensor,
        image_guide: torch.Tensor,
        video_guide: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        多模态交叉注意力前向传播
        
        Args:
            text_features: 文本特征 [batch_size, seq_len, embed_dim]
            image_features: 图像特征 [batch_size, seq_len, embed_dim]
            video_features: 视频特征 [batch_size, seq_len, embed_dim]
            text_guide: 文本引导向量 [batch_size, guide_len, embed_dim]
            image_guide: 图像引导向量 [batch_size, guide_len, embed_dim]
            video_guide: 视频引导向量 [batch_size, guide_len, embed_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 增强后的文本、图像、视频特征
        """
        # 文本模态的交叉注意力
        text_to_image_attn = self.text_to_image_attn(text_features, image_guide)
        text_to_video_attn = self.text_to_video_attn(text_features, video_guide)
        
        # 融合文本的交叉注意力
        text_enhanced = self.text_to_image_fusion(text_features, text_to_image_attn)
        text_enhanced = self.text_to_video_fusion(text_enhanced, text_to_video_attn)
        
        # 图像模态的交叉注意力
        image_to_text_attn = self.image_to_text_attn(image_features, text_guide)
        image_to_video_attn = self.image_to_video_attn(image_features, video_guide)
        
        # 融合图像的交叉注意力
        image_enhanced = self.image_to_text_fusion(image_features, image_to_text_attn)
        image_enhanced = self.image_to_video_fusion(image_enhanced, image_to_video_attn)
        
        # 视频模态的交叉注意力
        video_to_text_attn = self.video_to_text_attn(video_features, text_guide)
        video_to_image_attn = self.video_to_image_attn(video_features, image_guide)
        
        # 融合视频的交叉注意力
        video_enhanced = self.video_to_text_fusion(video_features, video_to_text_attn)
        video_enhanced = self.video_to_image_fusion(video_enhanced, video_to_image_attn)
        
        return text_enhanced, image_enhanced, video_enhanced

