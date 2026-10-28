"""AlexNet, following the layer table in the lecture (CNN Architectures, AlexNet slides).

    conv1  64 x 11x11, stride 4, pad 2   -> 64 x 56 x 56   (227 input)
    pool1  3x3, stride 2                 -> 64 x 27 x 27
    conv2  192 x 5x5, stride 1, pad 2    -> 192 x 27 x 27
    pool2  3x3, stride 2                 -> 192 x 13 x 13
    conv3  384 x 3x3, pad 1              -> 384 x 13 x 13
    conv4  256 x 3x3, pad 1              -> 256 x 13 x 13
    conv5  256 x 3x3, pad 1              -> 256 x 13 x 13
    pool5  3x3, stride 2                 -> 256 x 6 x 6
    flatten                              -> 9216
    fc6 4096, fc7 4096, fc8 num_classes

Local response normalization is omitted ("not used anymore" per the lecture).
Dropout on fc6/fc7 comes from the original paper, not the lecture.
"""

from __future__ import annotations

import torch
from torch import nn

from .blocks import check_num_classes, initialize_weights


class AlexNet(nn.Module):
    """AlexNet classifier (single-GPU variant used in the lecture).

    Args:
        num_classes: Number of output logits.
        in_channels: Input image channels (3 for RGB, 1 for grayscale).
        dropout: Dropout probability applied after fc6 and fc7.

    An adaptive average pool to 6x6 sits before the classifier. It is the
    identity for 224/227 inputs (pool5 is already 6x6), so the lecture's
    9216-d flatten is preserved, and it lets other input sizes work.
    """

    feature_channels = 256

    def __init__(self, num_classes: int, in_channels: int = 3, dropout: float = 0.5) -> None:
        super().__init__()
        check_num_classes(num_classes)
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
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )
        initialize_weights(self)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the pool5 feature map, ``[N, 256, 6, 6]`` for 224/227 inputs."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(self.forward_features(x)))
