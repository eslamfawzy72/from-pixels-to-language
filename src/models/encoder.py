"""Image encoders: image ``[N, C, H, W]`` -> feature map ``[N, out_channels, h, w]``.

The encoder is the part of a CNN that turns pixels into features. The same encoder
is the backbone of all three tasks: classification adds a classifier head
(classifier.py); colorization and captioning (Milestone 2) add their own decoders.

    from src.models import build_encoder

    encoder = build_encoder("resnet18")                       # from scratch
    encoder = build_encoder("resnet50_pretrained")            # ImageNet weights
    features = encoder(images)                                # [N, encoder.out_channels, 7, 7] at 224
"""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50

from .alexnet import AlexNetEncoder
from .googlenet import GoogLeNetEncoder
from .resnet import RESNET_SPECS, resnet_encoder
from .vgg import vgg16_encoder, vgg19_encoder


class PretrainedResNetEncoder(nn.Module):
    """torchvision ResNet with ImageNet weights, without its average pool and 1000-class layer.

    Our from-scratch ResNets name their layers differently, so the ImageNet weights
    cannot be loaded into them. The layer names here (conv1, bn1, layer1-4) are
    torchvision's, so the downloaded weights load as they are.
    """

    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.conv1, self.bn1, self.relu, self.maxpool = network.conv1, network.bn1, network.relu, network.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = network.layer1, network.layer2, network.layer3, network.layer4
        self.out_channels = network.fc.in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))


def _pretrained_resnet(network: Callable[..., nn.Module], weights: Any) -> Callable[..., PretrainedResNetEncoder]:
    def build(pretrained: bool = True, freeze: bool = False) -> PretrainedResNetEncoder:
        """``pretrained=False`` gives random weights (used by the tests); ``freeze`` stops training them."""
        encoder = PretrainedResNetEncoder(network(weights=weights if pretrained else None))
        encoder.requires_grad_(not freeze)
        return encoder
    return build


# IMAGENET1K_V1 is the original release, the one most published CUB results use.
_FROM_SCRATCH: dict[str, Callable[..., nn.Module]] = {
    "alexnet": AlexNetEncoder,
    "vgg16": vgg16_encoder,
    "vgg19": vgg19_encoder,
    "googlenet": GoogLeNetEncoder,
    **{name: (lambda name=name, **kwargs: resnet_encoder(name, **kwargs)) for name in RESNET_SPECS},
}
_PRETRAINED: dict[str, Callable[..., nn.Module]] = {
    "resnet18_pretrained": _pretrained_resnet(resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet50_pretrained": _pretrained_resnet(resnet50, ResNet50_Weights.IMAGENET1K_V1),
}


def available_encoders() -> list[str]:
    """Names accepted by :func:`build_encoder`: from-scratch first, then pretrained."""
    return list(_FROM_SCRATCH) + list(_PRETRAINED)


def build_encoder(name: str, **kwargs: Any) -> nn.Module:
    """Build an encoder by name. Every encoder has an ``out_channels`` attribute.

    Options: ``in_channels`` (from-scratch encoders, e.g. 1 for grayscale), ``batch_norm``
    (VGG, GoogLeNet), ``pretrained`` and ``freeze`` (pretrained encoders).
    """
    builders = {**_FROM_SCRATCH, **_PRETRAINED}
    if name not in builders:
        raise ValueError(f"Unknown encoder {name!r}. Available encoders: {', '.join(builders)}")
    return builders[name](**kwargs)
