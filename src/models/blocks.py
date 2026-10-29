"""Reusable building blocks shared by the CNN architectures.

Everything here is built from plain PyTorch layers (no torchvision models).
"""

from __future__ import annotations

import torch
from torch import nn


def check_num_classes(num_classes: int) -> None:
    """Raise a ValueError unless ``num_classes`` is a positive integer."""
    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes < 1:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes!r}")


def initialize_weights(model: nn.Module) -> None:
    """Randomly initialize a model from scratch.

    Conv: Kaiming normal (fan_out, ReLU). BatchNorm: weight 1, bias 0.
    Linear: N(0, 0.01), bias 0.
    """
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
            nn.init.zeros_(module.bias)


class ConvBlock(nn.Sequential):
    """Conv2d -> (optional BatchNorm2d) -> ReLU.

    When batch norm is used the convolution has no bias, since BN's shift
    makes it redundant.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        batch_norm: bool = False,
    ) -> None:
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not batch_norm)
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


# --------------------------------------------------------------------------- #
# GoogLeNet / Inception v1
# --------------------------------------------------------------------------- #
class InceptionModule(nn.Module):
    """Inception v1 module: four parallel branches concatenated on channels.

    Branches (lecture slide "GoogLeNet: Inception Module"):
        1. 1x1 conv
        2. 1x1 bottleneck -> 3x3 conv
        3. 1x1 bottleneck -> 5x5 conv
        4. 3x3 max pool (stride 1) -> 1x1 projection

    All branches preserve H and W, so the output is
    ``[N, ch1x1 + ch3x3 + ch5x5 + pool_proj, H, W]``.
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
        batch_norm: bool = False,
    ) -> None:
        super().__init__()
        self.branch1 = ConvBlock(in_channels, ch1x1, 1, batch_norm=batch_norm)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, ch3x3_reduce, 1, batch_norm=batch_norm),
            ConvBlock(ch3x3_reduce, ch3x3, 3, padding=1, batch_norm=batch_norm),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, ch5x5_reduce, 1, batch_norm=batch_norm),
            ConvBlock(ch5x5_reduce, ch5x5, 5, padding=2, batch_norm=batch_norm),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBlock(in_channels, pool_proj, 1, batch_norm=batch_norm),
        )
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = (self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x))
        return torch.cat(branches, dim=1)


class InceptionAux(nn.Module):
    """Auxiliary classifier attached to intermediate GoogLeNet layers.

    Layout follows Szegedy et al. (2015), since the lecture only describes the
    idea: 5x5 avg pool (stride 3) -> 1x1 conv (128) -> FC 1024 -> dropout 0.7
    -> FC num_classes. An adaptive pool to 4x4 follows the 5x5 pool; it is a
    no-op for 224x224 inputs and keeps the FC input fixed for larger inputs.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        batch_norm: bool = False,
        dropout: float = 0.7,
    ) -> None:
        super().__init__()
        self.pool = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=3), nn.AdaptiveAvgPool2d((4, 4)))
        self.conv = ConvBlock(in_channels, 128, 1, batch_norm=batch_norm)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.conv(self.pool(x)))


# --------------------------------------------------------------------------- #
# ResNet
# --------------------------------------------------------------------------- #
def _projection_shortcut(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    """1x1 conv + BN shortcut used when the residual changes shape (He et al., option B)."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class BasicBlock(nn.Module):
    """ResNet basic block: two 3x3 convs, used by ResNet-18/34.

    Original (post-activation) ordering from the lecture:
    conv-BN-ReLU-conv-BN, add shortcut, ReLU.
    """

    expansion = 1

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = _projection_shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class Bottleneck(nn.Module):
    """ResNet bottleneck block, used by ResNet-50/101/152.

    1x1 (reduce to C) -> 3x3 (C) -> 1x1 (expand to 4C), each followed by BN,
    with ReLU after the first two and after the residual addition. When a
    stage downsamples, the stride is applied in the 3x3 conv (see the
    architecture README for this choice).
    """

    expansion = 4

    def __init__(self, in_channels: int, channels: int, stride: int = 1) -> None:
        super().__init__()
        out_channels = channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.shortcut: nn.Module = _projection_shortcut(in_channels, out_channels, stride)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.shortcut(x))
