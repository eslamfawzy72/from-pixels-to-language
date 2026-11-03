"""Command line interface for training, evaluation and inference.

Examples:

    python -m src.cli train --task classification --config configs/classification.yaml
    python -m src.cli train --task classification --model googlenet --epochs 80 \
        --set train.model_kwargs.batch_norm=true --set train.model_kwargs.aux_logits=false
    python -m src.cli tune --task classification --method grid      # search space: configs/tuning.yaml
    python -m src.cli compare-tuning outputs/runs/tune_resnet18_grid outputs/runs/tune_resnet18_random
    python -m src.cli evaluate --task classification --checkpoint outputs/runs/<run>/best.pt --split test
    python -m src.cli predict --task classification --checkpoint outputs/runs/<run>/best.pt --image bird.jpg
    python -m src.cli gradcam --task classification --checkpoint outputs/runs/<run>/best.pt

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
DEFAULT_TUNING = Path("configs/tuning.yaml")


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


def resolve_train_settings(args: argparse.Namespace, preset: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load the config file and apply ``preset`` train values, shortcut flags and ``--set`` overrides (later wins)."""
    settings = copy.deepcopy(load_settings(args.config))
    settings["train"].update(preset or {})
    shortcuts = {
        "train.model_name": args.model,
        "train.epochs": args.epochs,
        "train.learning_rate": args.lr,
        "train.batch_size": args.batch_size,
        "train.optimizer": args.optimizer,
        "train.scheduler": args.scheduler,
        "train.seed": args.seed,
        "output.dir": args.output_dir,
        "output.run_name": getattr(args, "run_name", None),
        "tracking.dir": args.tracking_dir,
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
                             image_size=data.get("image_size", 224), tracker=tracker,
                             random_crop=data.get("random_crop", False))

    if tracking.get("enabled", True):
        with MLflowTracker(tracking.get("experiment", "classification"), run_name=run_name,
                           tracking_dir=tracking.get("dir", "outputs/mlflow"),
                           tracking_uri=tracking.get("tracking_uri"),
                           log_checkpoints=tracking.get("log_checkpoints", False),
                           tags={"task": args.task, "model": config.model_name}) as tracker:
            run(tracker)
    else:
        run(None)


def yaml_scalar(value: Any) -> str:
    """Write ``value`` so ``--set`` parses it back unchanged (YAML reads ``1e-05`` as a string, ``1.0e-05`` as a float)."""
    text = repr(value)
    if isinstance(value, float) and "e" in text and "." not in text:
        text = text.replace("e", ".0e")
    return text


def build_trials(method: str, tuning: dict[str, Any], num_trials: Optional[int] = None,
                 search_seed: Optional[int] = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the trial values and the search space for ``method`` from the tuning settings."""
    from src.tuning import grid_trials, random_trials

    if method == "grid":
        grid, crossed = tuning.get("grid") or {}, tuning.get("grid_crossed")
        return grid_trials(grid, crossed), {"values": grid, "crossed": crossed or list(grid)}
    options = tuning.get("random") or {}
    space = options.get("space") or {}
    trials = random_trials(space, num_trials or options.get("num_trials", 6),
                           seed=options.get("seed", 0) if search_seed is None else search_seed)
    return trials, space


def cmd_tune(args: argparse.Namespace) -> None:
    from src.tasks.classification import tune_classification

    tuning = load_settings(args.space)
    args.config = args.config or Path(tuning.get("config", DEFAULT_CONFIG))  # the space file can name its base config
    settings = resolve_train_settings(args, preset=tuning.get("trial"))
    trials, space = build_trials(args.method, tuning, args.num_trials, args.search_seed)
    base = settings["train"]
    study_name = args.study_name or f"tune_{base['model_name']}_{args.method}"
    study_dir = Path(settings["output"].get("dir") or "outputs/runs") / study_name
    tracking = settings["tracking"]
    if tracking.get("enabled", True):
        tracking = {**tracking, "experiment": args.experiment or tuning.get("experiment", "tuning")}
    else:
        tracking = None
    data = settings["data"]

    print(f"{args.method} search: {len(trials)} trials of {base['epochs']} epochs -> {study_dir}")
    summary = tune_classification(base, trials, study_dir, args.method, space=space,
                                  num_workers=data.get("num_workers", 2), image_size=data.get("image_size", 224),
                                  tracking=tracking, random_crop=data.get("random_crop", False))
    overrides = " ".join(f"--set train.{key}={yaml_scalar(value)}" for key, value in summary["best"]["params"].items())
    print(f"Full training run with the best setting:\n  python -m src.cli train --task {args.task} "
          f"--config {args.config.as_posix()} --model {base['model_name']} --run-name {base['model_name']}_tuned "
          f"{overrides}")


def cmd_compare_tuning(args: argparse.Namespace) -> None:
    from src.tuning import comparison_rows, load_study, markdown_table

    table = markdown_table(comparison_rows([load_study(path) for path in args.studies]))
    print(table)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(table + "\n", encoding="utf-8")
        print(f"Saved {args.output}")


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


def cmd_gradcam(args: argparse.Namespace) -> None:
    from src.tasks.classification import gradcam_classification

    gradcam_classification(args.checkpoint, split=args.split, num_examples=args.num_examples,
                           output_dir=args.output_dir, target_class=args.target_class, batch_size=args.batch_size,
                           num_workers=args.num_workers, seed=args.seed, device=args.device)


def add_settings_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``train`` and ``tune`` (see :func:`resolve_train_settings`)."""
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help=f"YAML settings (default: {DEFAULT_CONFIG})")
    parser.add_argument("--model", help="architecture name, e.g. resnet18")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float, help="initial learning rate")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--optimizer", choices=("sgd", "adam", "adamw"))
    parser.add_argument("--scheduler", choices=("none", "cosine", "step", "plateau"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output-dir", help="parent folder for run directories")
    parser.add_argument("--experiment", help="MLflow experiment name")
    parser.add_argument("--tracking-dir", help="folder for the MLflow database and artifacts "
                        "(default: outputs/mlflow; on Colab use a Google Drive folder)")
    parser.add_argument("--tracking-uri", help="MLflow tracking URI (e.g. a remote server); overrides --tracking-dir")
    parser.add_argument("--no-tracking", action="store_true", help="disable MLflow tracking")
    parser.add_argument("--set", action="append", metavar="SECTION.KEY=VALUE",
                        help="override any config value, e.g. --set train.model_kwargs.batch_norm=true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a model")
    add_settings_arguments(train)
    train.add_argument("--run-name", help="run folder / MLflow run name (default: <model>_<timestamp>)")
    train.set_defaults(func=cmd_train)

    tune = commands.add_parser("tune", help="grid or random hyperparameter search (short trials, validation split)")
    add_settings_arguments(tune)
    tune.add_argument("--method", choices=("grid", "random"), required=True)
    tune.add_argument("--space", type=Path, default=DEFAULT_TUNING,
                      help=f"YAML search space and trial settings (default: {DEFAULT_TUNING})")
    tune.add_argument("--num-trials", type=int, help="random search: number of trials (overrides the YAML)")
    tune.add_argument("--search-seed", type=int, help="random search: sampling seed (overrides the YAML)")
    tune.add_argument("--study-name", help="study folder / MLflow parent run name (default: tune_<model>_<method>); "
                      "re-running the same study skips finished trials")
    tune.set_defaults(func=cmd_tune, config=None)  # None: the space file's `config`, else DEFAULT_CONFIG

    compare = commands.add_parser("compare-tuning", help="compare finished tuning studies in a table")
    compare.add_argument("studies", type=Path, nargs="+",
                         help="study folders containing results.json, or study summary JSON files")
    compare.add_argument("--output", type=Path, help="also save the Markdown table to this file")
    compare.set_defaults(func=cmd_compare_tuning)

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

    gradcam = commands.add_parser("gradcam", help="save Grad-CAM figures of correctly and incorrectly classified images")
    gradcam.add_argument("--task", choices=TASKS, required=True)
    gradcam.add_argument("--checkpoint", type=Path, required=True)
    gradcam.add_argument("--split", choices=("val", "test"), default="test")
    gradcam.add_argument("--num-examples", type=int, default=5, help="images per folder (correct, incorrect)")
    gradcam.add_argument("--target-class", choices=("predicted", "actual"), default="predicted",
                         help="the class each heatmap explains")
    gradcam.add_argument("--output-dir", type=Path, default=Path("outputs/GradCAM"))
    gradcam.add_argument("--batch-size", type=int, default=64)
    gradcam.add_argument("--num-workers", type=int, default=2)
    gradcam.add_argument("--seed", type=int, default=42, help="seed of the random image order")
    gradcam.add_argument("--device", default="auto")
    gradcam.set_defaults(func=cmd_gradcam)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
