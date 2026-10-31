"""Create any supported architecture by name.

Example:
    >>> model = create_model("resnet18", num_classes=200)
"""

from __future__ import annotations

from typing import Any, Callable

from torch import nn

from .alexnet import AlexNet
from .googlenet import GoogLeNet
from .pretrained import resnet50_pretrained
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vgg import vgg16, vgg19

_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "alexnet": AlexNet,
    "vgg16": vgg16,
    "vgg19": vgg19,
    "googlenet": GoogLeNet,
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnet152": resnet152,
}

_ALIASES: dict[str, str] = {"inception_v1": "googlenet"}

# Transfer learning (ImageNet weights). Kept apart from the from-scratch registry above, which the
# architecture tests and the benchmark loop over.
_PRETRAINED: dict[str, Callable[..., nn.Module]] = {
    "resnet50_pretrained": resnet50_pretrained,
}


def available_models() -> list[str]:
    """Return the names of the from-scratch architectures accepted by :func:`create_model`."""
    return list(_REGISTRY)


def pretrained_models() -> list[str]:
    """Return the names of the pretrained models accepted by :func:`create_model`."""
    return list(_PRETRAINED)


def create_model(name: str, num_classes: int, **kwargs: Any) -> nn.Module:
    """Build a model: randomly initialized, or with ImageNet weights for ``*_pretrained`` names.

    Args:
        name: Architecture name (case-insensitive), e.g. ``"vgg16"``.
        num_classes: Number of output classes.
        **kwargs: Architecture-specific options, e.g. ``in_channels=1``,
            ``batch_norm=True`` (VGG, GoogLeNet) or ``aux_logits=False``
            (GoogLeNet).

    Raises:
        ValueError: If the name is unknown or ``num_classes`` is invalid.
    """
    key = name.strip().lower().replace("-", "_")
    key = _ALIASES.get(key, key)
    builders = {**_REGISTRY, **_PRETRAINED}
    if key not in builders:
        raise ValueError(f"Unknown model {name!r}. Available models: {', '.join(builders)}")
    return builders[key](num_classes=num_classes, **kwargs)
