"""Small model inspection helpers."""

from __future__ import annotations

from torch import nn


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total_parameters, trainable_parameters)`` of a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
