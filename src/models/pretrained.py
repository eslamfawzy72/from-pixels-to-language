"""Pretrained models for transfer learning (torchvision ImageNet weights).

Our from-scratch ResNets (resnet.py) name their layers differently from torchvision's,
so the ImageNet weights cannot be loaded into them. We use torchvision's ResNets instead
and replace their last layer (1000 ImageNet classes) with a new one for our classes.

Two models, so the comparison with the from-scratch ResNet-18 is fair:
    resnet18_pretrained: same architecture as the from-scratch baseline, only the starting weights differ
    resnet50_pretrained: bigger model, the one most published CUB results use (benchmarking)
And VGG-16 (torchvision's, same architecture as our from-scratch vgg16 without batch norm), whose
last layer is ``classifier[6]`` instead of ``fc``:
    vgg16_pretrained: compared with the from-scratch VGG-16 (the effect of pretraining on VGG)
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, ResNet50_Weights, VGG16_Weights, resnet18, resnet50, vgg16

from .blocks import check_num_classes


def _freeze(model: nn.Module, freeze_backbone: bool) -> None:
    """Freeze every layer; called before the new last layer is added, so that one stays trainable."""
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False


def _new_last_layer(model: nn.Module, num_classes: int, freeze_backbone: bool) -> nn.Module:
    """Optionally freeze every layer, then replace the 1000-class last layer (always trainable)."""
    _freeze(model, freeze_backbone)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def resnet18_pretrained(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """ResNet-18 with ImageNet weights (IMAGENET1K_V1) and a new last layer (512 -> num_classes).

    Args: see :func:`resnet50_pretrained`.
    """
    check_num_classes(num_classes)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    return _new_last_layer(model, num_classes, freeze_backbone)


def resnet50_pretrained(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """ResNet-50 with ImageNet weights (IMAGENET1K_V1) and a new last layer (2048 -> num_classes).

    Args:
        num_classes: Number of output classes (200 for CUB).
        pretrained: Load the ImageNet weights (IMAGENET1K_V1, the original release that most
            published CUB results use). False gives random weights (used by the tests).
        freeze_backbone: Train only the new last layer (feature extraction). False fine-tunes
            every layer.
    """
    check_num_classes(num_classes)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    return _new_last_layer(model, num_classes, freeze_backbone)


def vgg16_pretrained(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """VGG-16 with ImageNet weights (IMAGENET1K_V1) and a new last layer (4096 -> num_classes).

    The two hidden 4096-unit layers (with dropout) keep their ImageNet weights; only
    ``classifier[6]`` is replaced. With ``freeze_backbone`` only that layer is trained.

    Args: see :func:`resnet50_pretrained`.
    """
    check_num_classes(num_classes)
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
    _freeze(model, freeze_backbone)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model
