# -*- coding: utf-8 -*-
"""自定义网络模块，供改进模型 yaml 引用。

ultralytics 的 parse_model 通过 `globals()[模块名]` 解析 yaml 里的模块，
因此自定义类必须注入到 `ultralytics.nn.tasks` 的全局命名空间后才能被 yaml 引用。

用法（在构建 YOLO(yaml) 之前调用）：
    from training.models.custom_modules import register_modules
    register_modules()
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力：全局平均池化 + 全局最大池化，经共享 MLP 后相加得通道权重。"""

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.fc(self.avg_pool(x))
        mx = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """空间注意力：通道方向平均/最大池化拼接后经 7x7 卷积得空间权重。"""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """卷积块注意力模块（CBAM，Woo et al., ECCV 2018）。

    通道注意力与空间注意力串联，对特征做重加权；不改变通道数与空间尺寸，
    因此可作为残差旁路插入颈部特征之后。
    """

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


def register_modules() -> None:
    """把自定义模块注入 ultralytics 的模块解析命名空间。"""
    import ultralytics.nn.tasks as _tasks
    import ultralytics.nn.modules as _modules

    for cls in (CBAM, ChannelAttention, SpatialAttention):
        setattr(_tasks, cls.__name__, cls)
        setattr(_modules, cls.__name__, cls)
