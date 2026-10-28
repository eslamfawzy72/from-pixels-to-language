"""Saving and loading training checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

import torch
from torch import nn

PathLike = Union[str, Path]


def save_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[dict[str, float]] = None,
    config: Optional[dict[str, Any]] = None,
) -> Path:
    """Save model (and optionally optimizer) state plus run metadata.

    ``config`` should contain plain Python values (e.g. ``TrainConfig.to_dict()``)
    so the checkpoint can be loaded with ``weights_only=True``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "epoch": epoch,
            "metrics": metrics or {},
            "config": config or {},
        },
        path,
    )
    return path


def load_checkpoint(
    path: PathLike,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> dict[str, Any]:
    """Load weights into ``model`` (and ``optimizer`` if given); return the checkpoint dict."""
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    if optimizer is not None:
        if checkpoint["optimizer_state"] is None:
            raise ValueError(f"{path} contains no optimizer state")
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    return checkpoint
