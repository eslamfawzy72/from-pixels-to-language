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


Scheduler = Union[torch.optim.lr_scheduler.LRScheduler, torch.optim.lr_scheduler.ReduceLROnPlateau]


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainConfig) -> Optional[Scheduler]:
    """Create the per-epoch learning-rate scheduler named in ``config`` (``None`` for ``"none"``).

    With ``warmup_epochs`` > 0 the learning rate rises linearly from
    ``learning_rate / (warmup_epochs + 1)`` to ``learning_rate`` before the
    main schedule starts.
    """
    lr_scheduler = torch.optim.lr_scheduler
    if config.scheduler == "plateau":
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode=MONITORS[config.monitor], factor=config.gamma,
                                              patience=config.plateau_patience)
    main_epochs = config.epochs - config.warmup_epochs
    if config.scheduler == "cosine":
        main: Optional[lr_scheduler.LRScheduler] = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_epochs, eta_min=config.min_lr)
    elif config.scheduler == "step":
        main = lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    else:
        main = None

    if config.warmup_epochs == 0:
        return main
    warmup = lr_scheduler.LinearLR(optimizer, start_factor=1.0 / (config.warmup_epochs + 1),
                                   total_iters=config.warmup_epochs)
    if main is None:
        return warmup
    return lr_scheduler.SequentialLR(optimizer, [warmup, main], milestones=[config.warmup_epochs])


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
    checkpoint_extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train ``model`` for up to ``config.epochs`` epochs with validation after each one.

    Writes to ``run_dir``:
        * ``last.pt``: checkpoint after the latest epoch
        * ``best.pt``: checkpoint with the best ``config.monitor`` value
        * ``history.json``: config + per-epoch metrics

    The learning rate follows ``config.scheduler`` and training stops early
    after ``config.early_stopping_patience`` epochs without improvement.

    By default the training loss is cross-entropy with
    ``config.label_smoothing`` and the validation loss is plain cross-entropy,
    so validation losses stay comparable across runs. Passing ``criterion``
    uses it for both.

    ``on_epoch_end(epoch, metrics)`` is called after every epoch and is the
    hook for an experiment tracker (see ``src.training.tracking``).
    ``checkpoint_extra`` (e.g. class names) is stored in every checkpoint.

    Call :func:`set_seed` *before* creating the model if weight initialization
    must be reproducible; ``fit`` re-seeds only for the training loop.

    Returns a dict with ``history``, ``best_epoch``, ``best_metric``,
    ``stopped_early`` and ``run_dir``.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.seed, config.deterministic)
    device = resolve_device(config.device)
    model.to(device)
    train_criterion = criterion or nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    val_criterion = criterion or nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    mode = MONITORS[config.monitor]

    history: list[dict[str, float]] = []
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None
    stopped_early = False
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(model, train_loader, train_criterion, optimizer, device, config.aux_weight)
        val_metrics = validate(model, val_loader, val_criterion, device)
        metrics = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "lr": lr,
            "epoch_seconds": time.perf_counter() - start,
        }
        history.append(metrics)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics[config.monitor])
        elif scheduler is not None:
            scheduler.step()

        save_checkpoint(run_dir / "last.pt", model, optimizer, epoch, metrics, config.to_dict(), checkpoint_extra)
        if _is_improvement(metrics[config.monitor], best_metric, mode):
            best_metric, best_epoch = metrics[config.monitor], epoch
            save_checkpoint(run_dir / "best.pt", model, optimizer, epoch, metrics, config.to_dict(),
                            checkpoint_extra)

        patience = config.early_stopping_patience
        stopped_early = patience is not None and epoch - best_epoch >= patience and epoch < config.epochs
        summary = {"config": config.to_dict(), "best_epoch": best_epoch, "best_metric": best_metric,
                   "stopped_early": stopped_early, "history": history}
        (run_dir / "history.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if on_epoch_end is not None:
            on_epoch_end(epoch, metrics)
        if stopped_early:
            break

    return {"history": history, "best_epoch": best_epoch, "best_metric": best_metric,
            "stopped_early": stopped_early, "run_dir": str(run_dir)}
