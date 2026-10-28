"""ResNet-18/34/50/101/152 (He et al., CVPR 2016), as presented in the lecture.

    stem:    conv 7x7/2 (64) + BN + ReLU -> maxpool 3x3/2   (224 -> 56)
    stage 1: C=64,  stride 1
    stage 2: C=128, first block stride 2
    stage 3: C=256, first block stride 2
    stage 4: C=512, first block stride 2
    head:    global average pool -> linear

Basic blocks (ResNet-18/34) output C channels; bottleneck blocks
(ResNet-50/101/152) output 4C channels. Block counts per stage are taken from
the lecture's ResNet table.
"""

from __future__ import annotations

from typing import Sequence, Type, Union

import torch
from torch import nn

from .blocks import BasicBlock, Bottleneck, check_num_classes, initialize_weights

ResidualBlock = Type[Union[BasicBlock, Bottleneck]]

RESNET_STAGE_CHANNELS = (64, 128, 256, 512)

RESNET_SPECS: dict[str, tuple[ResidualBlock, tuple[int, int, int, int]]] = {
    "resnet18": (BasicBlock, (2, 2, 2, 2)),
    "resnet34": (BasicBlock, (3, 4, 6, 3)),
    "resnet50": (Bottleneck, (3, 4, 6, 3)),
    "resnet101": (Bottleneck, (3, 4, 23, 3)),
    "resnet152": (Bottleneck, (3, 8, 36, 3)),
}


class ResNet(nn.Module):
    """Configurable residual network.

    Args:
        block: ``BasicBlock`` or ``Bottleneck``.
        blocks_per_stage: Number of residual blocks in each of the 4 stages.
        num_classes: Number of output logits.
        in_channels: Input image channels.
    """

    def __init__(
        self,
        block: ResidualBlock,
        blocks_per_stage: Sequence[int],
        num_classes: int,
        in_channels: int = 3,
    ) -> None:
        super().__init__()
        check_num_classes(num_classes)
        if len(blocks_per_stage) != len(RESNET_STAGE_CHANNELS) or any(n < 1 for n in blocks_per_stage):
            raise ValueError(f"blocks_per_stage must be 4 positive integers, got {tuple(blocks_per_stage)}")

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        channels = 64
        stages = []
        for index, (width, num_blocks) in enumerate(zip(RESNET_STAGE_CHANNELS, blocks_per_stage)):
            stride = 1 if index == 0 else 2
            blocks = [block(channels, width, stride)]
            channels = width * block.expansion
            blocks += [block(channels, width, 1) for _ in range(num_blocks - 1)]
            stages.append(nn.Sequential(*blocks))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages

        self.feature_channels = channels
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, num_classes)
        initialize_weights(self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the stage-4 feature map, e.g. ``[N, 512 or 2048, 7, 7]`` for 224 inputs."""
        x = self.stem(x)
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.avgpool(self.forward_features(x)), 1))


def _resnet(name: str, num_classes: int, **kwargs) -> ResNet:
    block, blocks_per_stage = RESNET_SPECS[name]
    return ResNet(block, blocks_per_stage, num_classes, **kwargs)


def resnet18(num_classes: int, **kwargs) -> ResNet:
    """ResNet-18: basic blocks (2, 2, 2, 2)."""
    return _resnet("resnet18", num_classes, **kwargs)


def resnet34(num_classes: int, **kwargs) -> ResNet:
    """ResNet-34: basic blocks (3, 4, 6, 3)."""
    return _resnet("resnet34", num_classes, **kwargs)


def resnet50(num_classes: int, **kwargs) -> ResNet:
    """ResNet-50: bottleneck blocks (3, 4, 6, 3)."""
    return _resnet("resnet50", num_classes, **kwargs)


def resnet101(num_classes: int, **kwargs) -> ResNet:
    """ResNet-101: bottleneck blocks (3, 4, 23, 3)."""
    return _resnet("resnet101", num_classes, **kwargs)


def resnet152(num_classes: int, **kwargs) -> ResNet:
    """ResNet-152: bottleneck blocks (3, 8, 36, 3)."""
    return _resnet("resnet152", num_classes, **kwargs)
