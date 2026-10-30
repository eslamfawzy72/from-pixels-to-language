"""Command line interface for training, evaluation and inference.

Examples:

    python -m src.cli train --task classification --config configs/classification.yaml
    python -m src.cli train --task classification --model googlenet --epochs 80 \
        --set train.model_kwargs.batch_norm=true --set train.model_kwargs.aux_logits=false
    python -m src.cli evaluate --task classification --checkpoint outputs/runs/<run>/best.pt --split test
    python -m src.cli predict --task classification --checkpoint outputs/runs/<run>/best.pt --image bird.jpg

Settings come from a YAML file (see ``configs/classification.yaml``).
Shortcut flags (``--model``, ``--epochs``, ``--lr``, ...) and generic
``--set section.key=value`` overrides are applied on top of it.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import yaml

TASKS = ("classification",)
DEFAULT_CONFIG = Path("configs/classification.yaml")


def load_settings(path: Path) -> dict[str, Any]:
    """Read a YAML settings file into a dict."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open(encoding="utf-8") as handle:
        settings = yaml.safe_load(handle) or {}
    for section in ("train", "data", "output", "tracking"):
        settings.setdefault(section, {})
    return settings


def apply_override(settings: dict[str, Any], assignment: str) -> None:
    """Apply one ``"section.key=value"`` override in place; the value is parsed as YAML."""
    if "=" not in assignment:
        raise ValueError(f"Override must look like section.key=value, got {assignment!r}")
    dotted_key, raw_value = assignment.split("=", 1)
    keys = dotted_key.strip().split(".")
    if len(keys) < 2:
        raise ValueError(f"Override key needs a section, e.g. train.epochs=10, got {dotted_key!r}")
    target = settings
    for key in keys[:-1]:
        node = target.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot set {dotted_key!r}: {key!r} is not a section")
        target = node
    target[keys[-1]] = yaml.safe_load(raw_value)


def resolve_train_settings(args: argparse.Namespace) -> dict[str, Any]:
    """Load the config file and apply shortcut flags and ``--set`` overrides (later wins)."""
    settings = copy.deepcopy(load_settings(args.config))
    shortcuts = {
        "train.model_name": args.model,
        "train.epochs": args.epochs,
        "train.learning_rate": args.lr,
        "train.batch_size": args.batch_size,
        "train.optimizer": args.optimizer,
        "train.scheduler": args.scheduler,
        "train.seed": args.seed,
        "output.dir": args.output_dir,
        "output.run_name": args.run_name,
        "tracking.tracking_uri": args.tracking_uri,
        "tracking.experiment": args.experiment,
    }
    for key, value in shortcuts.items():
        if value is not None:
            section, name = key.split(".")
            settings[section][name] = value
    if args.no_tracking:
        settings["tracking"]["enabled"] = False
    for assignment in args.set or []:
        apply_override(settings, assignment)
    return settings


def default_run_name(model_name: str) -> str:
    return f"{model_name}_{time.strftime('%Y%m%d-%H%M%S')}"


def cmd_train(args: argparse.Namespace) -> None:
    from src.tasks.classification import train_classification
    from src.training import TrainConfig
    from src.training.tracking import MLflowTracker

    settings = resolve_train_settings(args)
    config = TrainConfig.from_dict(settings["train"])
    run_name = settings["output"].get("run_name") or default_run_name(config.model_name)
    run_dir = Path(settings["output"].get("dir") or "outputs/runs") / run_name
    data = settings["data"]
    tracking = settings["tracking"]

    def run(tracker: Optional[MLflowTracker]) -> None:
        train_classification(config, run_dir, num_workers=data.get("num_workers", 2),
                             image_size=data.get("image_size", 224), tracker=tracker)

    if tracking.get("enabled", True):
        with MLflowTracker(tracking.get("experiment", "classification"), run_name=run_name,
                           tracking_uri=tracking.get("tracking_uri"),
                           log_checkpoints=tracking.get("log_checkpoints", False),
                           tags={"task": args.task, "model": config.model_name}) as tracker:
            run(tracker)
    else:
        run(None)


def cmd_evaluate(args: argparse.Namespace) -> None:
    from src.tasks.classification import evaluate_classification

    evaluate_classification(args.checkpoint, split=args.split, batch_size=args.batch_size,
                            num_workers=args.num_workers, device=args.device)


def cmd_predict(args: argparse.Namespace) -> None:
    from src.tasks.classification import predict_classification

    predictions = predict_classification(args.checkpoint, args.image, top_k=args.top_k, device=args.device)
    for path, top in zip(args.image, predictions):
        print(path)
        for rank, (name, probability) in enumerate(top, start=1):
            print(f"  {rank}. {name:<35} {probability:6.2%}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a model")
    train.add_argument("--task", choices=TASKS, required=True)
    train.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"YAML settings (default: {DEFAULT_CONFIG})")
    train.add_argument("--model", help="architecture name, e.g. resnet18")
    train.add_argument("--epochs", type=int)
    train.add_argument("--lr", type=float, help="initial learning rate")
    train.add_argument("--batch-size", type=int)
    train.add_argument("--optimizer", choices=("sgd", "adam", "adamw"))
    train.add_argument("--scheduler", choices=("none", "cosine", "step", "plateau"))
    train.add_argument("--seed", type=int)
    train.add_argument("--output-dir", help="parent folder for run directories")
    train.add_argument("--run-name", help="run folder / MLflow run name (default: <model>_<timestamp>)")
    train.add_argument("--experiment", help="MLflow experiment name")
    train.add_argument("--tracking-uri", help="MLflow tracking URI, e.g. file:/content/drive/MyDrive/.../mlruns")
    train.add_argument("--no-tracking", action="store_true", help="disable MLflow tracking")
    train.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                       help="override any config value, e.g. --set train.model_kwargs.batch_norm=true")
    train.set_defaults(func=cmd_train)

    evaluate = commands.add_parser("evaluate", help="evaluate a checkpoint on the val or test split")
    evaluate.add_argument("--task", choices=TASKS, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--split", choices=("val", "test"), default="test")
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--num-workers", type=int, default=2)
    evaluate.add_argument("--device", default="auto")
    evaluate.set_defaults(func=cmd_evaluate)

    predict = commands.add_parser("predict", help="predict the species of image files")
    predict.add_argument("--task", choices=TASKS, required=True)
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--image", type=Path, nargs="+", required=True)
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--device", default="auto")
    predict.set_defaults(func=cmd_predict)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
