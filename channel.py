"""
信道建模模块

实现各种信道模型，包括AWGN信道、瑞利衰落信道等。
支持功率归一化和信道噪声模拟。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import math
import logging

logger = logging.getLogger(__name__)


class Channel(nn.Module):
    """
    通用信道模块
    
    支持多种信道类型的建模，包括：
    1. AWGN信道（加性白高斯噪声）
    2. 瑞利衰落信道
    3. 莱斯衰落信道
    4. 自定义信道模型
    
    Args:
        channel_type (str): 信道类型 ("awgn", "rayleigh", "rician")
        snr_db (float): 信噪比（dB）
        power_normalization (bool): 是否进行功率归一化
    """
    
    def __init__(
        self,
        channel_type: str = "awgn",
        snr_db: float = 10.0,
        power_normalization: bool = True,
        **kwargs
    ):
        super().__init__()
        self.channel_type = channel_type
        self.snr_db = snr_db
        self.power_normalization = power_normalization
        
        # 根据信道类型初始化参数
        if channel_type == "awgn":
            self.channel_model = AWGNChannel(snr_db, power_normalization)
        elif channel_type == "rayleigh":
            self.channel_model = RayleighChannel(snr_db, power_normalization, **kwargs)
        elif channel_type == "rician":
            self.channel_model = RicianChannel(snr_db, power_normalization, **kwargs)
        else:
            raise ValueError(f"不支持的信道类型: {channel_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        信道传输
        
        Args:
            x (torch.Tensor): 输入信号 [batch_size, ...]
            
        Returns:
            torch.Tensor: 经过信道传输后的信号
        """
        return self.channel_model(x)
    
    def set_snr(self, snr_db: float):
        """设置信噪比"""
        self.snr_db = snr_db
        self.channel_model.set_snr(snr_db)


class AWGNChannel(nn.Module):
    """
    AWGN信道（加性白高斯噪声）
    
    最基础的信道模型，添加高斯白噪声。
    """
    
    def __init__(self, snr_db: float = 10.0, power_normalization: bool = True):
        super().__init__()
        self.snr_db = snr_db
        self.power_normalization = power_normalization
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        AWGN信道传输
        
        Args:
            x (torch.Tensor): 输入信号
            
        Returns:
            torch.Tensor: 加噪后的信号
        """
        if self.power_normalization:
            # 功率归一化
            x = self._power_normalize(x)
        
        # 计算噪声功率
        #signal_power = torch.mean(x ** 2)
        #snr_linear = 10 ** (self.snr_db / 10)
        noise_power = 1.0 / (10 ** (self.snr_db / 10))
        
        # 生成高斯噪声
        noise = torch.randn_like(x) * torch.sqrt(torch.tensor(noise_power, device=x.device))
        
        return x + noise
    
    def _power_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """功率归一化(提高数值稳定性,适配float16)"""
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            reduction_dims = list(range(1, x_float.dim()))
            power = torch.mean(x_float ** 2, dim=reduction_dims, keepdim=True)
            power_min = power.min().item()
            power_max = power.max().item()
            power_mean = power.mean().item()
            logger.debug(
                "AWGNChannel power stats: min=%.6e max=%.6e mean=%.6e",
                power_min,
                power_max,
                power_mean,
            )
            if not torch.isfinite(power).all() or power_min < 1e-12:
                logger.warning(
                    "AWGNChannel power stats show non-finite or tiny values: "
                    "min=%.6e max=%.6e mean=%.6e",
                    power_min,
                    power_max,
                    power_mean,
                )
            k = torch.rsqrt(power + 1e-4)
            x_norm = x_float * k
        
        return x_norm.to(orig_dtype)
    
    def set_snr(self, snr_db: float):
        """设置信噪比"""
        self.snr_db = snr_db


class RayleighChannel(nn.Module):
    """
    瑞利衰落信道
    
    模拟多径传播环境下的瑞利衰落信道。
    """
    
    def __init__(
        self, 
        snr_db: float = 10.0, 
        power_normalization: bool = True,
        num_taps: int = 1
    ):
        super().__init__()
        self.snr_db = snr_db
        self.power_normalization = power_normalization
        self.num_taps = num_taps
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        瑞利衰落信道传输
        
        Args:
            x (torch.Tensor): 输入信号
            
        Returns:
            torch.Tensor: 经过瑞利衰落的信号
        """
        if self.power_normalization:
            x = self._power_normalize(x)
        
        # 生成瑞利衰落系数
        batch_size = x.shape[0]
        device = x.device
        
        # 瑞利分布：sqrt(X^2 + Y^2)，其中X, Y是独立的高斯分布
        h_real = torch.randn(batch_size, *x.shape[1:], device=device)
        h_imag = torch.randn(batch_size, *x.shape[1:], device=device)
        h = torch.sqrt(h_real ** 2 + h_imag ** 2)
        
        # 应用衰落
        faded_signal = x * h
        
        # 添加AWGN噪声
        signal_power = torch.mean(faded_signal ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(faded_signal) * torch.sqrt(noise_power)
        
        return faded_signal + noise
    
    def _power_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """功率归一化（提高数值稳定性，适配float16）"""
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            reduction_dims = list(range(1, x_float.dim()))
            power = torch.mean(x_float ** 2, dim=reduction_dims, keepdim=True)
            power_min = power.min().item()
            power_max = power.max().item()
            power_mean = power.mean().item()
            logger.debug(
                "RayleighChannel power stats: min=%.6e max=%.6e mean=%.6e",
                power_min,
                power_max,
                power_mean,
            )
            if not torch.isfinite(power).all() or power_min < 1e-12:
                logger.warning(
                    "RayleighChannel power stats show non-finite or tiny values: "
                    "min=%.6e max=%.6e mean=%.6e",
                    power_min,
                    power_max,
                    power_mean,
                )
            k = torch.rsqrt(power + 1e-4)
            x_norm = x_float * k
       
        return x_norm.to(orig_dtype)
      
    
    def set_snr(self, snr_db: float):
        """设置信噪比"""
        self.snr_db = snr_db


class RicianChannel(nn.Module):
    """
    莱斯衰落信道
    
    模拟存在直射路径的莱斯衰落信道。
    """
    
    def __init__(
        self,
        snr_db: float = 10.0,
        power_normalization: bool = True,
        k_factor: float = 3.0  # 莱斯K因子
    ):
        super().__init__()
        self.snr_db = snr_db
        self.power_normalization = power_normalization
        self.k_factor = k_factor
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        莱斯衰落信道传输
        
        Args:
            x (torch.Tensor): 输入信号
            
        Returns:
            torch.Tensor: 经过莱斯衰落的信号
        """
        if self.power_normalization:
            x = self._power_normalize(x)
        
        batch_size = x.shape[0]
        device = x.device
        
        # 莱斯衰落：直射路径 + 散射路径
        # 直射路径幅度
        los_amplitude = torch.sqrt(self.k_factor / (self.k_factor + 1))
        # 散射路径幅度
        nlos_amplitude = torch.sqrt(1 / (self.k_factor + 1))
        
        # 直射路径（确定性的）
        h_los = los_amplitude * torch.ones_like(x)
        
        # 散射路径（瑞利分布）
        h_nlos_real = torch.randn_like(x) * nlos_amplitude
        h_nlos_imag = torch.randn_like(x) * nlos_amplitude
        h_nlos = h_nlos_real + 1j * h_nlos_imag
        
        # 总信道响应
        h = h_los + h_nlos.real  # 只取实部
        
        # 应用衰落
        faded_signal = x * h
        
        # 添加AWGN噪声
        signal_power = torch.mean(faded_signal ** 2)
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        noise = torch.randn_like(faded_signal) * torch.sqrt(noise_power)
        
        return faded_signal + noise
    
    def _power_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """功率归一化（提高数值稳定性，适配float16）"""
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_float = x.float()
            reduction_dims = list(range(1, x_float.dim()))
            power = torch.mean(x_float ** 2, dim=reduction_dims, keepdim=True)
            power_min = power.min().item()
            power_max = power.max().item()
            power_mean = power.mean().item()
            logger.debug(
                "RicianChannel power stats: min=%.6e max=%.6e mean=%.6e",
                power_min,
                power_max,
                power_mean,
            )
            if not torch.isfinite(power).all() or power_min < 1e-12:
                logger.warning(
                    "RicianChannel power stats show non-finite or tiny values: "
                    "min=%.6e max=%.6e mean=%.6e",
                    power_min,
                    power_max,
                    power_mean,
                )
            k = torch.rsqrt(power + 1e-4)
            x_norm = x_float * k
           
    
        
        return x_norm.to(orig_dtype)
    
    def set_snr(self, snr_db: float):
        """设置信噪比"""
        self.snr_db = snr_db


class AdaptiveChannel(nn.Module):
    """
    自适应信道模块
    
    根据输入信号特性自动调整信道参数。
    """
    
    def __init__(
        self,
        base_channel_type: str = "awgn",
        snr_range: tuple = (0, 20),
        power_normalization: bool = True
    ):
        super().__init__()
        self.base_channel_type = base_channel_type
        self.snr_range = snr_range
        self.power_normalization = power_normalization
        
        # 创建基础信道
        self.base_channel = Channel(
            channel_type=base_channel_type,
            snr_db=snr_range[0],
            power_normalization=power_normalization
        )
        
    def forward(self, x: torch.Tensor, snr_db: Optional[float] = None) -> torch.Tensor:
        """
        自适应信道传输
        
        Args:
            x (torch.Tensor): 输入信号
            snr_db (float, optional): 指定信噪比，如果为None则随机选择
            
        Returns:
            torch.Tensor: 经过信道传输的信号
        """
        if snr_db is None:
            # 随机选择信噪比
            snr_db = torch.uniform(
                self.snr_range[0], 
                self.snr_range[1]
            ).item()
        
        # 设置信噪比
        self.base_channel.set_snr(snr_db)
        
        return self.base_channel(x)


class PowerNormalizer(nn.Module):
    """
    功率归一化模块
    
    确保传输信号的功率符合要求。
    """
    
    def __init__(self, target_power: float = 1.0):
        super().__init__()
        self.target_power = target_power
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        功率归一化
        
        Args:
            x (torch.Tensor): 输入信号
            
        Returns:
            torch.Tensor: 归一化后的信号
        """
        # 计算当前功率
        reduction_dims = list(range(1, x.dim()))
        
        # 归一化到目标功率（提高数值稳定性，适配float16）
        power = torch.mean(x ** 2, dim=reduction_dims, keepdim=True)
        x_norm = x / torch.sqrt(power + 1e-6)
        
        return x_norm

