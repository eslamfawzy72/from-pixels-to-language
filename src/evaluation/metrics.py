"""Classification metrics."""

from __future__ import annotations

import torch


def correct_predictions(logits: torch.Tensor, targets: torch.Tensor) -> int:
    """Number of samples whose arg-max prediction equals the target."""
    return int((logits.argmax(dim=1) == targets).sum().item())


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy in [0, 1] for logits ``[N, C]`` and integer targets ``[N]``."""
    if targets.numel() == 0:
        raise ValueError("accuracy is undefined for an empty batch")
    return correct_predictions(logits, targets) / targets.numel()
