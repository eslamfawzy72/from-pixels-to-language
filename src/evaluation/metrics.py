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


def confusion_matrix(logits: torch.Tensor, targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Counts ``[C, C]`` for logits ``[N, C]`` and integer targets ``[N]``: rows are actual, columns predicted."""
    cells = targets * num_classes + logits.argmax(dim=1)
    return torch.bincount(cells, minlength=num_classes ** 2).reshape(num_classes, num_classes)


def per_class_metrics(confusion: torch.Tensor) -> dict[str, torch.Tensor]:
    """Precision, recall, F1 and support of each class, from a confusion matrix.

    A value that would divide by zero (a class never predicted or never present) is 0.
    """
    correct = confusion.diag().double()
    predicted, support = confusion.sum(dim=0), confusion.sum(dim=1)
    return {"precision": correct / predicted.clamp(min=1), "recall": correct / support.clamp(min=1),
            "f1": 2 * correct / (predicted + support).clamp(min=1), "support": support}


def macro_metrics(confusion: torch.Tensor) -> dict[str, float]:
    """Macro-averaged precision, recall and F1: every class counts equally, however many images it has.

    The average covers the classes that were present or predicted, like scikit-learn's
    default, so a class missing from the split does not pull the scores down.
    """
    if confusion.sum() == 0:
        raise ValueError("metrics are undefined for an empty confusion matrix")
    per_class = per_class_metrics(confusion)
    seen = per_class["support"] + confusion.sum(dim=0) > 0
    return {f"macro_{name}": per_class[name][seen].mean().item() for name in ("precision", "recall", "f1")}
