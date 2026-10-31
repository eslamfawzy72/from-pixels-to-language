"""Image classifiers: an encoder (encoder.py) followed by a classifier head.

    image -> encoder -> feature map [N, C, h, w] -> head -> logits [N, num_classes]

    from src.models import create_model

    model = create_model("resnet18", num_classes=200)        # encoder + head, randomly initialized
    model.encoder                                            # reusable for colorization / captioning
    model.head                                               # pool + linear layer(s)

Heads:
    LinearHead     global average pool -> linear                  (ResNet, pretrained ResNets)
    FCHead         pool to a fixed size -> FC 4096 -> FC 4096 -> FC  (AlexNet, VGG; dropout 0.5 from the papers)
    GoogLeNetHead  global average pool -> dropout -> linear, plus two auxiliary classifiers
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Union

import torch
from torch import nn

from .blocks import InceptionAux, check_num_classes, initialize_weights
from .encoder import _PRETRAINED, build_encoder
from .googlenet import GoogLeNetOutputs
from .resnet import RESNET_SPECS

# --------------------------------------------------------------------------- #
# Heads
# --------------------------------------------------------------------------- #


class LinearHead(nn.Module):
    """Global average pool -> linear layer."""

    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.avgpool(features), 1))


class FCHead(nn.Module):
    """AlexNet / VGG classifier: pool to ``pool_size`` x ``pool_size`` -> FC 4096 -> FC 4096 -> FC num_classes.

    The adaptive pool is the identity for 224 inputs (AlexNet pool5 is 6x6, VGG's is 7x7),
    so the lecture's 9216 / 25088-d flatten is kept, and it lets other input sizes work.
    """

    def __init__(self, in_channels: int, pool_size: int, num_classes: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * pool_size * pool_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(4096, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.avgpool(features))


class GoogLeNetHead(nn.Module):
    """Global average pool -> dropout (0.4) -> linear, plus the two auxiliary classifiers (paper layout)."""

    def __init__(self, in_channels: int, aux_channels: tuple[int, int], num_classes: int, aux_logits: bool = True,
                 batch_norm: bool = False, dropout: float = 0.4, aux_dropout: float = 0.7) -> None:
        super().__init__()
        self.aux1: Optional[InceptionAux] = None
        self.aux2: Optional[InceptionAux] = None
        if aux_logits:
            self.aux1 = InceptionAux(aux_channels[0], num_classes, batch_norm, aux_dropout)
            self.aux2 = InceptionAux(aux_channels[1], num_classes, batch_norm, aux_dropout)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(torch.flatten(self.avgpool(features), 1)))


# --------------------------------------------------------------------------- #
# Classifiers
# --------------------------------------------------------------------------- #


class ImageClassifier(nn.Module):
    """``head(encoder(x))``. Every model created by :func:`create_model` is one of these."""

    def __init__(self, encoder: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """The encoder's feature map, e.g. ``[N, 512, 7, 7]`` for ResNet-18 at 224."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x))


class GoogLeNetClassifier(ImageClassifier):
    """GoogLeNet: in training mode with auxiliary classifiers, returns ``GoogLeNetOutputs(logits, aux1, aux2)``.

    Otherwise (eval mode, or ``aux_logits=False``) it returns the main logits only.
    ``src.training.losses.classification_loss`` handles both.
    """

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, GoogLeNetOutputs]:
        if not (self.training and self.head.aux1 is not None):
            return super().forward(x)
        features, (after_4a, after_4d) = self.encoder(x, return_aux_features=True)
        return GoogLeNetOutputs(self.head(features), self.head.aux1(after_4a), self.head.aux2(after_4d))


# --------------------------------------------------------------------------- #
# Builders: one per model name; keyword arguments are that model's options
# --------------------------------------------------------------------------- #


def _alexnet(num_classes: int, in_channels: int = 3, dropout: float = 0.5) -> ImageClassifier:
    encoder = build_encoder("alexnet", in_channels=in_channels)
    return initialize_weights(ImageClassifier(encoder, FCHead(encoder.out_channels, 6, num_classes, dropout)))


def _vgg(name: str) -> Callable[..., ImageClassifier]:
    def build(num_classes: int, in_channels: int = 3, batch_norm: bool = False, dropout: float = 0.5) -> ImageClassifier:
        encoder = build_encoder(name, in_channels=in_channels, batch_norm=batch_norm)
        return initialize_weights(ImageClassifier(encoder, FCHead(encoder.out_channels, 7, num_classes, dropout)))
    return build


def _googlenet(num_classes: int, in_channels: int = 3, aux_logits: bool = True, batch_norm: bool = False,
               dropout: float = 0.4, aux_dropout: float = 0.7) -> GoogLeNetClassifier:
    """Auxiliary heads use a 5x5/3 average pool, so with ``aux_logits=True`` training inputs must be >= 72x72."""
    encoder = build_encoder("googlenet", in_channels=in_channels, batch_norm=batch_norm)
    head = GoogLeNetHead(encoder.out_channels, encoder.aux_channels, num_classes, aux_logits, batch_norm,
                         dropout, aux_dropout)
    return initialize_weights(GoogLeNetClassifier(encoder, head))


def _resnet(name: str) -> Callable[..., ImageClassifier]:
    def build(num_classes: int, in_channels: int = 3) -> ImageClassifier:
        encoder = build_encoder(name, in_channels=in_channels)
        return initialize_weights(ImageClassifier(encoder, LinearHead(encoder.out_channels, num_classes)))
    return build


def _pretrained(name: str) -> Callable[..., ImageClassifier]:
    def build(num_classes: int, pretrained: bool = True, freeze_backbone: bool = False) -> ImageClassifier:
        """ImageNet encoder + a new linear head. ``freeze_backbone`` trains only the head (feature extraction)."""
        encoder = build_encoder(name, pretrained=pretrained, freeze=freeze_backbone)
        return ImageClassifier(encoder, LinearHead(encoder.out_channels, num_classes))
    return build


_FROM_SCRATCH: dict[str, Callable[..., ImageClassifier]] = {
    "alexnet": _alexnet,
    "vgg16": _vgg("vgg16"),
    "vgg19": _vgg("vgg19"),
    "googlenet": _googlenet,
    **{name: _resnet(name) for name in RESNET_SPECS},
}
_TRANSFER: dict[str, Callable[..., ImageClassifier]] = {name: _pretrained(name) for name in _PRETRAINED}
_ALIASES: dict[str, str] = {"inception_v1": "googlenet"}


def available_models() -> list[str]:
    """Names of the from-scratch architectures accepted by :func:`create_model`."""
    return list(_FROM_SCRATCH)


def pretrained_models() -> list[str]:
    """Names of the transfer-learning models accepted by :func:`create_model`."""
    return list(_TRANSFER)


def create_model(name: str, num_classes: int, **kwargs: Any) -> ImageClassifier:
    """Build a classifier by name: randomly initialized, or with an ImageNet encoder for ``*_pretrained``.

    Args:
        name: Architecture name (case-insensitive), e.g. ``"resnet18"`` or ``"resnet50_pretrained"``.
        num_classes: Number of output classes.
        **kwargs: Model options, e.g. ``in_channels=1``, ``batch_norm=True`` (VGG, GoogLeNet),
            ``aux_logits=False`` (GoogLeNet), ``freeze_backbone=True`` (pretrained).

    Raises:
        ValueError: If the name is unknown or ``num_classes`` is invalid.
    """
    check_num_classes(num_classes)
    key = name.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    builders = {**_FROM_SCRATCH, **_TRANSFER}
    if key not in builders:
        raise ValueError(f"Unknown model {name!r}. Available models: {', '.join(builders)}")
    return builders[key](num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)`` of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# --------------------------------------------------------------------------- #
# Checkpoints saved before the encoder / head split
# --------------------------------------------------------------------------- #

# Top-level module names that belonged to the classifier part in the old single-block models.
_OLD_HEAD_MODULES = {"fc", "classifier", "avgpool", "dropout", "aux1", "aux2"}


def upgrade_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename weights saved before the split (``layer4.0.conv1.weight`` -> ``encoder.layer4.0.conv1.weight``,
    ``fc.weight`` -> ``head.fc.weight``). New-style state dicts are returned unchanged.

    The layer names inside the encoders and heads were kept, so adding the prefix is enough.
    """
    if any(key.startswith(("encoder.", "head.")) for key in state):
        return state
    return {("head." if key.split(".", 1)[0] in _OLD_HEAD_MODULES else "encoder.") + key: value
            for key, value in state.items()}
