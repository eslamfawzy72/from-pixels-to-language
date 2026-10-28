"""GoogLeNet (Inception v1).

From the lecture:
    * aggressive stem: conv 7x7/2 (64) -> maxpool 3x3/2 -> conv 1x1 (64)
      -> conv 3x3 (192) -> maxpool 3x3/2, taking 224 -> 28 spatial resolution
    * Inception modules with 1x1 bottlenecks before the 3x3 and 5x5 convs
    * global average pooling + a single linear layer (no large FC layers)
    * auxiliary classifiers at intermediate points (training-time only)

The per-module channel widths, dropout rates and aux head layout are not in
the lecture; they follow Table 1 of Szegedy et al., "Going deeper with
convolutions" (CVPR 2015). LRN is omitted, matching the lecture's stem table.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Union

import torch
from torch import nn

from .blocks import ConvBlock, InceptionAux, InceptionModule, check_num_classes, initialize_weights

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
    """Training-mode output when auxiliary classifiers are enabled."""

    logits: torch.Tensor
    aux1: torch.Tensor
    aux2: torch.Tensor


class GoogLeNet(nn.Module):
    """GoogLeNet / Inception v1 classifier.

    Args:
        num_classes: Number of output logits.
        in_channels: Input image channels.
        aux_logits: Build the two auxiliary classifiers (after 4a and 4d).
        batch_norm: Add BatchNorm after every conv (not in the original
            network; the lecture notes BN makes aux classifiers unnecessary).
        dropout: Dropout before the final linear layer (0.4 in the paper).
        aux_dropout: Dropout inside the auxiliary heads (0.7 in the paper).

    Output contract (exception to the common interface):
        * ``model.train()`` with ``aux_logits=True``: returns
          ``GoogLeNetOutputs(logits, aux1, aux2)``.
        * otherwise: returns the main ``logits`` tensor only.
      Use ``src.training.losses.classification_loss`` to handle both.

    The auxiliary heads use a 5x5/3 average pool, so with ``aux_logits=True``
    training inputs must be at least 72x72 (4a's feature map must be >= 5x5).
    """

    feature_channels = 1024

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 3,
        aux_logits: bool = True,
        batch_norm: bool = False,
        dropout: float = 0.4,
        aux_dropout: float = 0.7,
    ) -> None:
        super().__init__()
        check_num_classes(num_classes)
        self.aux_logits = aux_logits

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

        self.aux1: Optional[InceptionAux] = None
        self.aux2: Optional[InceptionAux] = None
        if aux_logits:
            self.aux1 = InceptionAux(self.inception4a.out_channels, num_classes, batch_norm, aux_dropout)
            self.aux2 = InceptionAux(self.inception4d.out_channels, num_classes, batch_norm, aux_dropout)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.inception5b.out_channels, num_classes)
        initialize_weights(self)

    def _trunk(self, x: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run stem + Inception stages; aux outputs are computed only in training mode."""
        use_aux = self.training and self.aux_logits
        x = self.stem(x)
        x = self.inception3b(self.inception3a(x))
        x = self.inception4a(self.pool3(x))
        aux1 = self.aux1(x) if use_aux else None
        x = self.inception4d(self.inception4c(self.inception4b(x)))
        aux2 = self.aux2(x) if use_aux else None
        x = self.inception4e(x)
        x = self.inception5b(self.inception5a(self.pool4(x)))
        return x, aux1, aux2

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the inception-5b feature map, ``[N, 1024, 7, 7]`` for 224 inputs."""
        features, _, _ = self._trunk(x)
        return features

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, GoogLeNetOutputs]:
        features, aux1, aux2 = self._trunk(x)
        logits = self.fc(self.dropout(torch.flatten(self.avgpool(features), 1)))
        if aux1 is not None and aux2 is not None:
            return GoogLeNetOutputs(logits, aux1, aux2)
        return logits
