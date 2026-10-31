"""GoogLeNet (Inception v1) encoder.

From the lecture:
    * aggressive stem: conv 7x7/2 (64) -> maxpool 3x3/2 -> conv 1x1 (64)
      -> conv 3x3 (192) -> maxpool 3x3/2, taking 224 -> 28 spatial resolution
    * Inception modules with 1x1 bottlenecks before the 3x3 and 5x5 convs
    * global average pooling + a single linear layer (no large FC layers)
    * auxiliary classifiers at intermediate points (training-time only)

The classifier (global average pool -> dropout -> linear) and the two auxiliary
classifiers are ``GoogLeNetHead`` in classifier.py. The auxiliary classifiers read
the inception-4a and 4d feature maps, which this encoder returns on request.

The per-module channel widths are not in the lecture; they follow Table 1 of
Szegedy et al., "Going deeper with convolutions" (CVPR 2015). LRN is omitted,
matching the lecture's stem table.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import torch
from torch import nn

from .blocks import ConvBlock, InceptionModule

# (in_channels, #1x1, #3x3 reduce, #3x3, #5x5 reduce, #5x5, pool proj)
INCEPTION_SPECS: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "3a": (192, 64, 96, 128, 16, 32, 32),
    "3b": (256, 128, 128, 192, 32, 96, 64),
    "4a": (480, 192, 96, 208, 16, 48, 64),
    "4b": (512, 160, 112, 224, 24, 64, 64),
    "4c": (512, 128, 128, 256, 24, 64, 64),
    "4d": (512, 112, 144, 288, 32, 64, 64),
    "4e": (528, 256, 160, 320, 32, 128, 128),
    "5a": (832, 256, 160, 320, 32, 128, 128),
    "5b": (832, 384, 192, 384, 48, 128, 128),
}


class GoogLeNetOutputs(NamedTuple):
    """Training-mode output of the GoogLeNet classifier when auxiliary classifiers are enabled."""

    logits: torch.Tensor
    aux1: torch.Tensor
    aux2: torch.Tensor


class GoogLeNetEncoder(nn.Module):
    """Stem + Inception stages: image -> ``[N, 1024, 7, 7]`` for 224 inputs.

    Args:
        in_channels: Input image channels.
        batch_norm: Add BatchNorm after every conv (not in the original network).
    """

    out_channels = 1024
    aux_channels = (512, 528)  # channels of the inception-4a and 4d outputs (auxiliary classifier inputs)

    def __init__(self, in_channels: int = 3, batch_norm: bool = False) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3, batch_norm=batch_norm),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBlock(64, 64, kernel_size=1, batch_norm=batch_norm),
            ConvBlock(64, 192, kernel_size=3, padding=1, batch_norm=batch_norm),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        def inception(key: str) -> InceptionModule:
            return InceptionModule(*INCEPTION_SPECS[key], batch_norm=batch_norm)

        self.inception3a = inception("3a")
        self.inception3b = inception("3b")
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception4a = inception("4a")
        self.inception4b = inception("4b")
        self.inception4c = inception("4c")
        self.inception4d = inception("4d")
        self.inception4e = inception("4e")
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = inception("5a")
        self.inception5b = inception("5b")

    def forward(
        self, x: torch.Tensor, return_aux_features: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]]:
        """Return the inception-5b feature map; with ``return_aux_features`` also the 4a and 4d maps."""
        x = self.stem(x)
        x = self.inception3b(self.inception3a(x))
        after_4a = self.inception4a(self.pool3(x))
        after_4d = self.inception4d(self.inception4c(self.inception4b(after_4a)))
        x = self.inception4e(after_4d)
        x = self.inception5b(self.inception5a(self.pool4(x)))
        return (x, (after_4a, after_4d)) if return_aux_features else x
