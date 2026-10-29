"""Reusable training loop for image classification.

Data loading is deliberately outside this module: every function takes data
loaders that yield ``(images [N, C, H, W] float, labels [N] long)`` batches.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Union

import numpy as np
import torch
from torch import nn

from src.evaluation.metrics import correct_predictions

from .checkpoint import save_checkpoint
from .config import MONITORS, TrainConfig
from .losses import classification_loss
from .validate import validate

Batches = Iterable[tuple[torch.Tensor, torch.Tensor]]
EpochCallback = Callable[[int, dict[str, float]], None]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU and CUDA) RNGs.

    ``deterministic=True`` also forces deterministic cuDNN kernels, which is
    slower but makes GPU runs repeatable.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def resolve_device(device: str = "auto") -> torch.device:
    """Map ``"auto"`` to CUDA when available, else CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    """Create the optimizer named in ``config``."""
    params = [p for p in model.parameters() if p.requires_grad]
    if config.optimizer == "sgd":
        return torch.optim.SGD(params, lr=config.learning_rate, momentum=config.momentum,
                               weight_decay=config.weight_decay)
    if config.optimizer == "adam":
        return torch.optim.Adam(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    return torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)


def train_one_epoch(
    model: nn.Module,
    loader: Batches,
    criterion: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    aux_weight: float = 0.3,
) -> dict[str, float]:
    """Train for one pass over ``loader`` and return ``{"loss", "accuracy"}``.

    For GoogLeNet with auxiliary classifiers the optimized (and reported) loss
    includes ``aux_weight`` times the auxiliary losses; accuracy always uses
    the main classifier.
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss, logits = classification_loss(model(images), labels, criterion, aux_weight)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += correct_predictions(logits.detach(), labels)
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("training loader produced no samples")
    return {"loss": total_loss / total_samples, "accuracy": total_correct / total_samples}


def _is_improvement(value: float, best: Optional[float], mode: str) -> bool:
    if best is None:
        return True
    return value > best if mode == "max" else value < best


def fit(
    model: nn.Module,
    train_loader: Batches,
    val_loader: Batches,
    config: TrainConfig,
    run_dir: Union[str, Path],
    criterion: Optional[nn.Module] = None,
    on_epoch_end: Optional[EpochCallback] = None,
) -> dict[str, Any]:
    """Train ``model`` for ``config.epochs`` epochs with validation after each one.

    Writes to ``run_dir``:
        * ``last.pt``: checkpoint after the latest epoch
        * ``best.pt``: checkpoint with the best ``config.monitor`` value
        * ``history.json``: config + per-epoch metrics

    ``on_epoch_end(epoch, metrics)`` is called after every epoch and is the
    hook for an experiment tracker (MLflow, W&B, ...).

    Call :func:`set_seed` *before* creating the model if weight initialization
    must be reproducible; ``fit`` re-seeds only for the training loop.

    Returns a dict with ``history``, ``best_epoch`` and ``best_metric``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed, config.deterministic)
    device = resolve_device(config.device)
    model.to(device)
    criterion = criterion or nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    mode = MONITORS[config.monitor]

    history: list[dict[str, float]] = []
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, config.aux_weight)
        val_metrics = validate(model, val_loader, criterion, device)
        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(metrics)

        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, metrics, config.to_dict())
        if _is_improvement(metrics[config.monitor], best_metric, mode):
            best_metric, best_epoch = metrics[config.monitor], epoch
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, metrics, config.to_dict())

        summary = {"config": config.to_dict(), "best_epoch": best_epoch,
                   "best_metric": best_metric, "history": history}
        (run_dir / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)

    return {"history": history, "best_epoch": best_epoch, "best_metric": best_metric}
