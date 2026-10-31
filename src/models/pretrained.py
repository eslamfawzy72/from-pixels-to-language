"""Pretrained models for transfer learning (torchvision ImageNet weights).

Our from-scratch ResNet-50 (resnet.py) names its layers differently from torchvision's,
so the ImageNet weights cannot be loaded into it. We use torchvision's ResNet-50 instead
and replace its last layer (1000 ImageNet classes) with a new one for our classes.
"""

from __future__ import annotations

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from .blocks import check_num_classes


def resnet50_pretrained(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    """ResNet-50 with ImageNet weights and a new, randomly initialized last layer.

    Args:
        num_classes: Number of output classes (200 for CUB).
        pretrained: Load the ImageNet weights (IMAGENET1K_V1, the original release that most
            published CUB results use). False gives random weights (used by the tests).
        freeze_backbone: Train only the new last layer (feature extraction). False fine-tunes
            every layer.
    """
    check_num_classes(num_classes)
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    if freeze_backbone:
        for parameter in model.parameters():
            parameter.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 2048 -> num_classes, always trainable
    return model
