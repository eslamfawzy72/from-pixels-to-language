"""Classification task: train, evaluate and predict.

This module wires together the data pipeline (``src.data``), the models
(``src.models``), the training loop (``src.training``) and optional MLflow
tracking. ``src.cli`` only parses arguments and calls these functions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import torch
from torch import nn

from src.models import count_parameters, create_model
from src.training import TrainConfig, fit, resolve_device, set_seed, validate
from src.training.tracking import MLflowTracker

PathLike = Union[str, Path]


def train_classification(
    config: TrainConfig,
    run_dir: PathLike,
    num_workers: int = 2,
    image_size: int = 224,
    tracker: Optional[MLflowTracker] = None,
) -> dict[str, Any]:
    """Train ``config.model_name`` on CUB and return the result of :func:`fit`.

    The class names and input size are stored in every checkpoint so that
    :func:`predict_classification` works without downloading the dataset.
    """
    from src.data.dataset import get_class_names, get_dataloaders

    set_seed(config.seed, config.deterministic)
    model = create_model(config.model_name, num_classes=config.num_classes, **config.model_kwargs)
    total_params, trainable_params = count_parameters(model)
    train_loader, val_loader, _ = get_dataloaders(batch_size=config.batch_size, num_workers=num_workers,
                                                  image_size=image_size, seed=config.seed)
    class_names = list(get_class_names())
    if len(class_names) != config.num_classes:
        raise ValueError(f"num_classes={config.num_classes} but the dataset has {len(class_names)} classes")

    extra = {"class_names": class_names, "image_size": image_size,
             "total_params": total_params, "trainable_params": trainable_params}
    if tracker is not None:
        tracker.log_params({**config.to_dict(), "image_size": image_size, "num_workers": num_workers,
                            "total_params": total_params, "trainable_params": trainable_params,
                            "train_images": len(train_loader.dataset), "val_images": len(val_loader.dataset)})
        extra["mlflow_run_id"] = tracker.run_id

    print(f"Training {config.model_name}: {total_params:,} parameters, "
          f"{len(train_loader.dataset)} train / {len(val_loader.dataset)} val images -> {run_dir}")

    def report(epoch: int, metrics: dict[str, float]) -> None:
        print(f"epoch {epoch:3d}/{config.epochs}  lr={metrics['lr']:.5f}  "
              f"train_loss={metrics['train_loss']:.3f} train_acc={metrics['train_accuracy']:.3f}  "
              f"val_loss={metrics['val_loss']:.3f} val_acc={metrics['val_accuracy']:.3f}  "
              f"({metrics['epoch_seconds']:.0f}s)", flush=True)
        if tracker is not None:
            tracker.log_epoch(epoch, metrics)

    result = fit(model, train_loader, val_loader, config, run_dir, on_epoch_end=report, checkpoint_extra=extra)
    if tracker is not None:
        tracker.log_run_outputs(result)
    stop = " (stopped early)" if result["stopped_early"] else ""
    print(f"Best {config.monitor}: {result['best_metric']:.4f} at epoch {result['best_epoch']}{stop}")
    return result


def load_model_from_checkpoint(checkpoint_path: PathLike, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild the model recorded in a checkpoint and load its weights (eval mode)."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    config = checkpoint.get("config") or {}
    if "model_name" not in config:
        raise ValueError(f"{checkpoint_path} has no model config; was it saved by src.training.fit?")
    model = create_model(config["model_name"], num_classes=config["num_classes"], **config.get("model_kwargs", {}))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def evaluate_classification(
    checkpoint_path: PathLike,
    split: str = "test",
    batch_size: int = 64,
    num_workers: int = 2,
    device: str = "auto",
) -> dict[str, float]:
    """Compute loss and accuracy of a checkpoint on the ``"val"`` or ``"test"`` split.

    Results are also written next to the checkpoint as ``<split>_metrics.json``.
    """
    from src.data.dataset import get_dataloaders

    if split not in ("val", "test"):
        raise ValueError(f"split must be 'val' or 'test', got {split!r}")
    torch_device = resolve_device(device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, torch_device)
    image_size = checkpoint.get("extra", {}).get("image_size", 224)
    _, val_loader, test_loader = get_dataloaders(batch_size=batch_size, num_workers=num_workers,
                                                 image_size=image_size)
    loader = test_loader if split == "test" else val_loader
    metrics = validate(model, loader, nn.CrossEntropyLoss(), torch_device)
    metrics = {**metrics, "images": len(loader.dataset), "epoch": checkpoint.get("epoch")}

    output = Path(checkpoint_path).with_name(f"{split}_metrics.json")
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"{split}: loss={metrics['loss']:.4f} accuracy={metrics['accuracy']:.4f} "
          f"on {metrics['images']} images -> {output}")
    return metrics


def predict_classification(
    checkpoint_path: PathLike,
    image_paths: Sequence[PathLike],
    top_k: int = 5,
    device: str = "auto",
) -> list[list[tuple[str, float]]]:
    """Return the ``top_k`` (species, probability) pairs for each image file."""
    from PIL import Image

    from src.data.transforms import build_eval_transform

    torch_device = resolve_device(device)
    model, checkpoint = load_model_from_checkpoint(checkpoint_path, torch_device)
    extra = checkpoint.get("extra", {})
    num_classes = checkpoint["config"]["num_classes"]
    class_names = extra.get("class_names") or [f"class_{i}" for i in range(num_classes)]
    transform = build_eval_transform(extra.get("image_size", 224))

    predictions = []
    with torch.inference_mode():
        for path in image_paths:
            with Image.open(path) as image:
                batch = transform(image).unsqueeze(0).to(torch_device)
            probabilities = torch.softmax(model(batch), dim=1)[0]
            values, indices = probabilities.topk(min(top_k, num_classes))
            predictions.append([(class_names[i], float(p)) for p, i in zip(values.tolist(), indices.tolist())])
    return predictions
