"""
文本JSCC编码器和解码器

基于Transformer的文本语义编码器，支持文本数据的联合信源信道编码。
包含简化的BERT架构和自定义Transformer模块。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


def _log_nonfinite(name: str, tensor: torch.Tensor) -> bool:
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
        "%s has NaN/Inf values (finite stats min=%.6e max=%.6e mean=%.6e)",
        name,
        t_min,
        t_max,
        t_mean,
    )
    return True


class PositionalEncoding(nn.Module):
    """
    位置编码模块
    
    为序列数据添加位置信息，支持正弦和余弦位置编码。
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # 计算位置编码
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 注册为缓冲区
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        添加位置编码
        
        Args:
            x (torch.Tensor): 输入序列 [batch_size, seq_len, d_model]
            
        Returns:
            torch.Tensor: 添加位置编码后的序列
        """
        return x + self.pe[:, :x.size(1)]


class MultiHeadAttention(nn.Module):
    """
    多头自注意力机制
    
    实现Transformer中的多头注意力机制。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # 线性投影层
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        多头注意力前向传播
        
        Args:
            query, key, value: 查询、键、值张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码
            
        Returns:
            torch.Tensor: 注意力输出
        """
        batch_size, seq_len = query.size(0), query.size(1)
        
        # 线性投影（内存优化：分步计算）
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力分数（使用float32防止混合精度下的溢出）
        # 临时转换为float32进行计算，防止溢出
        Q_f32 = Q.float()
        K_f32 = K.float()
        scores = torch.matmul(Q_f32, K_f32.transpose(-2, -1)) / self.scale
        scores = scores.to(Q.dtype)  # 转换回原始dtype
        del Q, K, Q_f32, K_f32  # 及时释放Q和K
        
        # 应用掩码
        if mask is not None:
            # 允许 mask 形状为 [B, T] 或 [B, 1, 1, T] 或 [B, 1, T, T]
            if mask.dim() == 2:
                # [B, T] -> [B, 1, 1, T]
                mask_expanded = mask.unsqueeze(1).unsqueeze(1)
            elif mask.dim() == 3:
                # [B, 1, T] -> [B, 1, 1, T]
                mask_expanded = mask.unsqueeze(1)
            elif mask.dim() == 4:
                mask_expanded = mask
            else:
                raise ValueError(f"不支持的注意力掩码维度: {mask.shape}")
            # 广播到 [B, num_heads, T, T]
            mask_broadcast = mask_expanded
            if mask_broadcast.size(1) == 1:
                mask_broadcast = mask_broadcast.expand(-1, self.num_heads, -1, -1)
            # 使用与scores相同dtype的最小值，兼容float16
            fill_value = torch.finfo(scores.dtype).min if scores.dtype.is_floating_point else -1e4
            scores = scores.masked_fill(mask_broadcast == 0, fill_value)
            del mask_expanded, mask_broadcast  # 及时释放
        
        # Softmax归一化
        attn_weights = F.softmax(scores, dim=-1)
        del scores  # 及时释放
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        context = torch.matmul(attn_weights, V)
        del attn_weights, V  # 及时释放
        
        # 重塑并输出投影
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(context)


class FeedForward(nn.Module):
    """
    前馈神经网络
    
    Transformer中的前馈网络，包含两个线性层和激活函数。
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前馈网络前向传播"""
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Transformer编码器块
    
    包含多头自注意力和前馈网络，支持残差连接和层归一化。
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Transformer块前向传播
        
        Args:
            x (torch.Tensor): 输入序列 [batch_size, seq_len, d_model]
            mask (torch.Tensor, optional): 注意力掩码
            
        Returns:
            torch.Tensor: 输出序列
        """
        # 自注意力 + 残差连接
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TextJSCCEncoder(nn.Module):
    """
    文本JSCC编码器
    
    基于Transformer的文本语义编码器，将文本序列编码为适合信道传输的连续值特征。
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        output_dim: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.output_dim = output_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer编码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出投影层
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )
        
        # 引导向量提取器（用于跨模态注意力）
        self.guide_extractor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        文本编码器前向传播
        
        Args:
            input_ids (torch.Tensor): 输入文本ID [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): 注意力掩码
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (编码特征, 引导向量)
        """
        # 词嵌入
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer编码
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # 提取引导向量（全局平均池化）
        if attention_mask is not None:
            # 使用注意力掩码进行掩码平均池化
            mask_expanded = attention_mask.unsqueeze(-1).expand_as(x).to(x.dtype)
            x_masked = x * mask_expanded
            denom = mask_expanded.sum(dim=1)
            # 避免全零掩码导致除零/Inf
            zero_mask = (denom == 0)
            denom = denom + (~zero_mask).to(x.dtype) * 0.0 + (zero_mask).to(x.dtype) * 1.0
            guide_vector = x_masked.sum(dim=1) / denom
            # 对于全零掩码的样本，退化为无掩码的均值
            if zero_mask.any():
                guide_fallback = x.mean(dim=1)
                guide_vector = torch.where(
                    zero_mask.unsqueeze(-1), guide_fallback, guide_vector
                )
        else:
            guide_vector = x.mean(dim=1)
        
        # 输出投影
        encoded_features = self.output_proj(x)
        
        # 引导向量处理
        guide_vector = self.guide_extractor(guide_vector)
        _log_nonfinite("TextJSCCEncoder.encoded_features", encoded_features)
        _log_nonfinite("TextJSCCEncoder.guide_vector", guide_vector)
        return encoded_features, guide_vector


class TextJSCCDecoder(nn.Module):
    """
    文本JSCC解码器
    
    从经过信道传输的带噪特征重建原始文本序列。
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_len: int = 512,
        dropout: float = 0.1,
        input_dim: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 输入投影层
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Transformer解码器层
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 引导向量处理
        self.guide_processor = nn.Sequential(
            nn.Linear(d_model // 4, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        noisy_features: torch.Tensor,
        guide_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        文本解码器前向传播
        
        Args:
            noisy_features (torch.Tensor): 带噪特征 [batch_size, seq_len, input_dim]
            guide_vector (torch.Tensor): 引导向量 [batch_size, guide_dim]
            attention_mask (torch.Tensor, optional): 注意力掩码
            
        Returns:
            torch.Tensor: 重建的文本logits [batch_size, seq_len, vocab_size]
        """
        # 输入投影
        x = self.input_proj(noisy_features)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 处理引导向量
        guide_processed = self.guide_processor(guide_vector)
        
        # 将引导向量广播到序列长度
        batch_size, seq_len = x.size(0), x.size(1)
        guide_expanded = guide_processed.unsqueeze(1).expand(batch_size, seq_len, -1)
        
        # 融合引导信息
        x = x + guide_expanded
        
        # Transformer解码
        for layer in self.transformer_layers:
            x = layer(x, attention_mask)
        
        # 输出投影
        logits = self.output_layer(x)
        
        return logits
    
    def generate(
        self,
        noisy_features: torch.Tensor,
        guide_vector: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        生成文本序列
        
        Args:
            noisy_features (torch.Tensor): 带噪特征
            guide_vector (torch.Tensor): 引导向量
            max_length (int): 最大生成长度
            temperature (float): 采样温度
            
        Returns:
            torch.Tensor: 生成的文本序列
        """
        self.eval()
        with torch.no_grad():
            batch_size = noisy_features.size(0)
            device = noisy_features.device
            
            # 初始化生成序列
            generated = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
            
            for i in range(max_length):
                # 当前输入
                current_input = noisy_features[:, :i+1] if i < noisy_features.size(1) else noisy_features
                
                # 前向传播
                logits = self.forward(current_input, guide_vector)
                
                # 获取最后一个时间步的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # 采样下一个token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                if i < max_length:
                    generated[:, i] = next_token.squeeze(-1)
            
            return generated

