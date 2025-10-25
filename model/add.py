import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba  # 假设使用Mamba官方实现

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossModalFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 线性变换 + 深度可分离卷积
        self.linear_conv_v = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)  # Depthwise Conv
        )
        self.linear_conv_t = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, groups=out_channels)  # Depthwise Conv
        )

    def forward(self, I_e, F_T):
        # I_e: 视觉特征 [B, C, H, W]
        # F_T: 文本特征 [B, L, D]

        # 视觉分支投影
        P_V = self.linear_conv_v(I_e.permute(0, 3, 1, 2))  # [B, D, H, W]
        P_V = P_V.permute(0, 2, 3, 1)  # [B, H, W, D]

        # 文本分支投影
        P_T = self.linear_conv_t(F_T.permute(0, 2, 1))  # [B, D, L]
        P_T = P_T.permute(0, 2, 1)  # [B, L, D]

        # 逐元素相乘
        M = P_V.unsqueeze(2) * P_T.unsqueeze(1).unsqueeze(1)  # [B, H, W, L, D]
        M = M.mean(dim=3)  # 压缩序列维度 [B, H, W, D]

        # 残差连接
        H_bar = M + I_e + F_T.mean(dim=1).unsqueeze(1).unsqueeze(1)  # [B, H, W, D]
        return H_bar

class FinalFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.es2d = ES2D()  # 假设ES2D为自定义的高效空间扫描模块
        self.ln = nn.LayerNorm(channels)
        self.eca = ECALayer(channels)  # 自定义ECA模块

    def forward(self, H_bar, I_e, F_T):
        # ES2D处理
        H = self.es2d(H_bar)  # [B, H, W, D]
        # 层归一化
        H_norm = self.ln(H)
        # ECA通道注意力
        X_f = self.eca(H_norm) + I_e + F_T.mean(dim=1).unsqueeze(1).unsqueeze(1)
        return X_f

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        k = k if k % 2 else k + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, H, W, C]
        y = self.avg_pool(x.permute(0, 3, 1, 2))  # [B, C, 1, 1]
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # [B, C, 1] -> [B, 1, C]
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * y.expand_as(x)

class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y_ = y.squeeze(-1).transpose(-1, -2)

        # Two different branches of ECA module
        y = self.conv(y_)
        y = y.transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

if __name__ == '__main__':
    # 实例化模型
    fusion_module = CrossModalFusion(in_channels=64, out_channels=64)
    final_fusion = FinalFusion(channels=64)

    # 生成测试数据
    # img = torch.randn(2, 196, 768)  # 视觉特征序列 [B, 196, 768]
    # txt = torch.randn(2, 32, 768)  # 文本特征序列 [B, 32, 768]
    # 模拟视觉特征 [B, H, W, C]
    img = torch.randn(2, 32, 32, 64)  # Batch=2, H=32, W=32, C=64
    # 模拟文本特征 [B, L, D]
    txt = torch.randn(2, 16, 64)  # Batch=2, 序列长度=16, 特征维度=64
    # Step 1: 跨模态融合
    H_bar = fusion_module(img, txt)
    print("H_bar shape:", H_bar.shape)  # 应输出 [2, 32, 32, 64]

    # Step 2: 最终融合
    X_f = final_fusion(H_bar, img, txt)
    print("X_f shape:", X_f.shape)  # 应输出 [2, 32, 32, 64]
    # 前向传播

