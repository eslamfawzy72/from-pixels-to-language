"""Validation loop (no gradient updates)."""

from __future__ import annotations

from typing import Callable, Iterable

import torch
from torch import nn

from src.evaluation.metrics import correct_predictions

from .losses import split_outputs


def validate(
    model: nn.Module,
    loader: Iterable[tuple[torch.Tensor, torch.Tensor]],
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate ``model`` on ``loader`` and return ``{"loss", "accuracy"}``.

    The loader must yield ``(images [N, C, H, W], labels [N])`` batches.
    The model is put in eval mode, so GoogLeNet returns main logits only and
    the loss is the main-classifier loss.
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits, _ = split_outputs(model(images))
            batch_size = labels.size(0)
            total_loss += criterion(logits, labels).item() * batch_size
            total_correct += correct_predictions(logits, labels)
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("validation loader produced no samples")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}
