"""ResNet-50 from the article "Bird by Bird using Deep Learning" (Sofya Lipnitskaya, 2021).

Two variants, both starting from the ImageNet weights:
    use_attention=False: the same network as ``resnet50_pretrained``. The article's 80.77% comes
        from this model, so any gain over ours comes from its data and training settings.
    use_attention=True: adds one attention block per stage (82.4% in the article, which also
        predicted bounding boxes; our dataset has none, so that task is left out).

Differences from the article's code: its attention channels (64, 128, 256, 512) fit ResNet-18/34
and fail on ResNet-50, whose stages output 256, 512, 1024 and 2048 channels, so these are used here.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .pretrained import resnet50_pretrained


class Attention(nn.Module):
    """Compute a mask in [0, 2] from a stage's input, to multiply with that stage's output.

    The last batch norm starts at zero, so the mask starts as all ones: the block changes
    nothing at first and the model begins exactly as the pretrained network.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int],
                 padding: tuple[int, int]) -> None:
        super().__init__()
        self.conv_depth = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=in_channels)
        self.conv_point = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.Tanh()
        nn.init.zeros_(self.bn.weight)
        nn.init.zeros_(self.bn.bias)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = F.adaptive_max_pool2d(x, output_size)
        x = self.bn(self.conv_point(self.conv_depth(x)))
        return self.activation(x) + 1.0


class AttentionResNet50(nn.Module):
    """A torchvision ResNet-50 with an attention block after each of its four stages."""

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.base = base
        self.att1 = Attention(64, 256, kernel_size=(3, 5), padding=(1, 2))
        self.att2 = Attention(256, 512, kernel_size=(5, 3), padding=(2, 1))
        self.att3 = Attention(512, 1024, kernel_size=(3, 5), padding=(1, 2))
        self.att4 = Attention(1024, 2048, kernel_size=(5, 3), padding=(2, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base
        x = base.maxpool(base.relu(base.bn1(base.conv1(x))))
        stages = ((base.layer1, self.att1), (base.layer2, self.att2), (base.layer3, self.att3), (base.layer4, self.att4))
        for stage, attention in stages:
            stage_input = x
            x = stage(stage_input)
            x = x * attention(stage_input, x.shape[-2:])
        return base.fc(torch.flatten(base.avgpool(x), 1))


def resnet50_article(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False,
                     use_attention: bool = False) -> nn.Module:
    """The article's ResNet-50, with a new last layer (2048 -> num_classes).

    Args:
        num_classes: Number of output classes (200 for CUB).
        pretrained: Load the ImageNet weights. False gives random weights (used by the tests).
        freeze_backbone: Train only the new last layer (and the attention blocks, if used).
        use_attention: Add the attention blocks.
    """
    model = resnet50_pretrained(num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone)
    return AttentionResNet50(model) if use_attention else model
