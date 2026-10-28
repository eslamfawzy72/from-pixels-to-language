"""VGG-16 and VGG-19, built from the lecture's design rules.

Design rules (lecture, "VGG: Deeper Networks, Regular Design"):
    * all convs are 3x3, stride 1, pad 1
    * all max pools are 2x2, stride 2
    * after each pool the number of channels doubles (capped at 512)
    * 5 stages; VGG-19 has 4 convs in stages 3, 4 and 5 instead of 3
    * classifier: FC 4096 -> FC 4096 -> FC num_classes

Dropout (0.5) in the first two FC layers comes from the original paper.
BatchNorm is off by default (the lecture/paper network has none) and can be
enabled with ``batch_norm=True``.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from .blocks import ConvBlock, check_num_classes, initialize_weights

VGG_STAGE_CHANNELS = (64, 128, 256, 512, 512)

# Number of 3x3 conv layers in each of the 5 stages.
VGG_STAGE_DEPTHS: dict[str, tuple[int, ...]] = {
    "vgg16": (2, 2, 3, 3, 3),
    "vgg19": (2, 2, 4, 4, 4),
}


class VGG(nn.Module):
    """Configurable VGG network.

    Args:
        stage_depths: Number of 3x3 convs in each of the five stages.
        num_classes: Number of output logits.
        in_channels: Input image channels.
        batch_norm: Insert BatchNorm after every conv (not in the lecture).
        dropout: Dropout probability after FC6 and FC7.

    An adaptive average pool to 7x7 precedes the classifier; it is the
    identity for 224x224 inputs (keeping the 25088-d flatten) and allows other
    input sizes.
    """

    feature_channels = 512

    def __init__(
        self,
        stage_depths: Sequence[int],
        num_classes: int,
        in_channels: int = 3,
        batch_norm: bool = False,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        check_num_classes(num_classes)
        if len(stage_depths) != len(VGG_STAGE_CHANNELS) or any(d < 1 for d in stage_depths):
            raise ValueError(f"stage_depths must be 5 positive integers, got {tuple(stage_depths)}")

        layers: list[nn.Module] = []
        channels = in_channels
        for depth, width in zip(stage_depths, VGG_STAGE_CHANNELS):
            for _ in range(depth):
                layers.append(ConvBlock(channels, width, kernel_size=3, padding=1, batch_norm=batch_norm))
                channels = width
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        initialize_weights(self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the stage-5 feature map, ``[N, 512, 7, 7]`` for 224 inputs."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.forward_features(x)))


def vgg16(num_classes: int, **kwargs) -> VGG:
    """VGG-16: 13 conv + 3 FC layers."""
    return VGG(VGG_STAGE_DEPTHS["vgg16"], num_classes, **kwargs)


def vgg19(num_classes: int, **kwargs) -> VGG:
    """VGG-19: 16 conv + 3 FC layers."""
    return VGG(VGG_STAGE_DEPTHS["vgg19"], num_classes, **kwargs)
