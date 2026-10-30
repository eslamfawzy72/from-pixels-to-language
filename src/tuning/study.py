"""Run a hyperparameter search ("study") as a sequence of short training runs.

A study lives in one folder::

    <study_dir>/
        trial_00/            best.pt, last.pt, history.json, trial_result.json
        trial_01/
        ...
        results.json         study summary + one record per finished trial
        results.csv          the same records as a table

``trial_result.json`` is written only when a trial finishes, so re-running the
same study command skips finished trials and restarts an interrupted one from
scratch (useful when a Colab session disconnects).

Training itself is delegated to ``train_fn(config, run_dir, run_name)``, which
must return the result dict of :func:`src.training.fit`. This keeps the study
logic independent of the dataset and of MLflow.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

from src.training.config import MONITORS, TrainConfig

PathLike = Union[str, Path]
TrainFn = Callable[[TrainConfig, Path, str], dict[str, Any]]
METHODS = ("grid", "random")
RESULT_FILE = "trial_result.json"
# Recorded for every trial (taken from its TrainConfig), whether or not the search changes them.
RECORDED_HYPERPARAMETERS = ("learning_rate", "weight_decay", "label_smoothing")


def trial_name(index: int) -> str:
    return f"trial_{index:02d}"


def build_trial_configs(base: dict[str, Any], trials: Sequence[dict[str, Any]]) -> list[TrainConfig]:
    """Merge each trial's values into the base ``train`` settings and validate them all up front."""
    configs = []
    for index, params in enumerate(trials):
        try:
            configs.append(TrainConfig.from_dict({**base, **params}))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{trial_name(index)} {params}: {error}") from error
    return configs


def trial_record(index: int, params: dict[str, Any], config: TrainConfig, result: dict[str, Any],
                 seconds: float) -> dict[str, Any]:
    """Summarize one finished trial (metrics are taken at its best epoch on the validation split)."""
    history = result["history"]
    best = next(m for m in history if m["epoch"] == result["best_epoch"])
    return {
        "trial": trial_name(index),
        "params": params,
        "hyperparameters": {key: getattr(config, key) for key in RECORDED_HYPERPARAMETERS},
        "monitor": config.monitor,
        "best_metric": result["best_metric"],
        "best_epoch": result["best_epoch"],
        "epochs_run": len(history),
        "val_accuracy": best["val_accuracy"],
        "val_loss": best["val_loss"],
        "train_accuracy": best["train_accuracy"],
        "final_val_accuracy": history[-1]["val_accuracy"],
        "stopped_early": result["stopped_early"],
        "seconds": seconds,
        "run_dir": str(result["run_dir"]),
    }


def best_trial(records: Sequence[dict[str, Any]], monitor: str) -> Optional[dict[str, Any]]:
    if not records:
        return None
    pick = max if MONITORS[monitor] == "max" else min
    return pick(records, key=lambda r: r["best_metric"])


def write_results(study_dir: Path, summary: dict[str, Any]) -> None:
    (study_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    other_keys = sorted({key for record in summary["trials"] for key in record["params"]}
                        - set(RECORDED_HYPERPARAMETERS))
    columns = ["trial", *RECORDED_HYPERPARAMETERS, *other_keys, "val_accuracy", "best_epoch", "epochs_run",
               "train_accuracy", "val_loss", "final_val_accuracy", "seconds"]
    with (study_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in summary["trials"]:
            writer.writerow({**record, **record.get("hyperparameters", {}), **record["params"]})


def run_study(
    base: dict[str, Any],
    trials: Sequence[dict[str, Any]],
    study_dir: PathLike,
    method: str,
    train_fn: TrainFn,
    space: Optional[dict[str, Any]] = None,
    on_trial_end: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Train one model per entry of ``trials`` and return the study summary.

    ``base`` holds the ``train`` settings shared by every trial (including the
    seed, so trials differ only in the tuned values). ``space`` is stored in
    ``results.json`` for documentation. ``on_trial_end(record)`` is called
    after every trial, including ones reused from an earlier run.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not trials:
        raise ValueError("no trials to run")
    study_dir = Path(study_dir)
    study_dir.mkdir(parents=True, exist_ok=True)
    configs = build_trial_configs(base, trials)
    monitor = configs[0].monitor

    summary: dict[str, Any] = {"method": method, "study": study_dir.name, "monitor": monitor,
                               "num_trials": len(trials), "base_config": base, "space": space,
                               "trials": [], "best": None, "total_seconds": 0.0}
    for index, (params, config) in enumerate(zip(trials, configs)):
        name = trial_name(index)
        run_dir = study_dir / name
        result_path = run_dir / RESULT_FILE
        if result_path.is_file():
            record = json.loads(result_path.read_text(encoding="utf-8"))
            if record["params"] != params:
                raise ValueError(f"{result_path} was produced with {record['params']}, not {params}; "
                                 f"use a new study name for a different search space")
            print(f"[{name}] already finished, reusing it: {params}")
        else:
            print(f"[{name}] {index + 1}/{len(trials)} {params}", flush=True)
            start = time.perf_counter()
            result = train_fn(config, run_dir, f"{study_dir.name}_{name}")
            record = trial_record(index, params, config, result, time.perf_counter() - start)
            result_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

        summary["trials"].append(record)
        summary["total_seconds"] = sum(r["seconds"] for r in summary["trials"])
        summary["best"] = best_trial(summary["trials"], monitor)
        write_results(study_dir, summary)
        if on_trial_end is not None:
            on_trial_end(record)

    best = summary["best"]
    print(f"{method} search finished: best {monitor}={best['best_metric']:.4f} ({best['trial']}, "
          f"{best['params']}); {len(trials)} trials in {summary['total_seconds'] / 60:.1f} min -> {study_dir}")
    return summary


def load_study(study_dir: PathLike) -> dict[str, Any]:
    path = Path(study_dir) / "results.json"
    if not path.is_file():
        raise FileNotFoundError(f"No results.json in {study_dir}; run the study first")
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_rows(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per study: method, trials finished, best value and params, time."""
    rows = []
    for summary in summaries:
        best, finished = summary["best"], len(summary["trials"])
        rows.append({
            "study": summary["study"],
            "method": summary["method"],
            "trials": f"{finished}/{summary['num_trials']}",
            "monitor": summary["monitor"],
            "best_value": best["best_metric"] if best else None,
            "best_trial": best["trial"] if best else None,
            "best_params": best["params"] if best else None,
            "total_minutes": summary["total_seconds"] / 60,
            "minutes_per_trial": summary["total_seconds"] / 60 / finished if finished else None,
        })
    return rows


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return ", ".join(f"{k}={v}" for k, v in value.items())
    return "-" if value is None else str(value)


def markdown_table(rows: Sequence[dict[str, Any]]) -> str:
    """Render rows (dicts with the same keys) as a Markdown table."""
    if not rows:
        return ""
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(_format_cell(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines)
