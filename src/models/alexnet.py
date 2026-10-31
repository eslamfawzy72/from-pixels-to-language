"""AlexNet encoder, following the layer table in the lecture (CNN Architectures, AlexNet slides).

    conv1  64 x 11x11, stride 4, pad 2   -> 64 x 56 x 56   (227 input)
    pool1  3x3, stride 2                 -> 64 x 27 x 27
    conv2  192 x 5x5, stride 1, pad 2    -> 192 x 27 x 27
    pool2  3x3, stride 2                 -> 192 x 13 x 13
    conv3  384 x 3x3, pad 1              -> 384 x 13 x 13
    conv4  256 x 3x3, pad 1              -> 256 x 13 x 13
    conv5  256 x 3x3, pad 1              -> 256 x 13 x 13
    pool5  3x3, stride 2                 -> 256 x 6 x 6     <- encoder output

The classifier (flatten 9216 -> fc6 4096 -> fc7 4096 -> fc8 num_classes) is
``FCHead`` in classifier.py. Local response normalization is omitted ("not used
anymore" per the lecture).
"""

from __future__ import annotations

import torch
from torch import nn


class AlexNetEncoder(nn.Module):
    """AlexNet conv layers (single-GPU variant used in the lecture): image -> ``[N, 256, 6, 6]``."""

    out_channels = 256

    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)
