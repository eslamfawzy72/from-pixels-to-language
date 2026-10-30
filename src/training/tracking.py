"""MLflow experiment tracking for training runs.

Usage:

    with MLflowTracker("classification", run_name="resnet18_lr0.05") as tracker:
        tracker.log_params(config.to_dict())
        result = fit(..., on_epoch_end=tracker.log_epoch)
        tracker.log_run_outputs(result)

Each run records its hyperparameters, per-epoch metrics (so MLflow draws the
training/validation curves), ``history.json`` and a reference to the
checkpoints. Checkpoint files are only uploaded when ``log_checkpoints=True``,
because VGG checkpoints exceed 1 GB.

``tracking_uri`` chooses where runs are stored, e.g.
``file:/content/drive/MyDrive/from-pixels-to-language/mlruns`` on Colab so they
survive the session. ``None`` uses MLflow's default (``./mlruns`` or the
``MLFLOW_TRACKING_URI`` environment variable).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def _flatten(params: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts (e.g. ``model_kwargs``) into ``"model_kwargs.batch_norm"`` keys."""
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        else:
            flat[name] = value
    return flat


class MLflowTracker:
    """Context manager that wraps one MLflow run."""

    def __init__(
        self,
        experiment: str,
        run_name: Optional[str] = None,
        tracking_uri: Optional[str] = None,
        log_checkpoints: bool = False,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        try:
            import mlflow
        except ImportError as error:
            raise ImportError("MLflow is not installed. Run: pip install mlflow "
                              "(or disable tracking with --no-tracking)") from error
        self._mlflow = mlflow
        self.experiment = experiment
        self.run_name = run_name
        self.tracking_uri = tracking_uri
        self.log_checkpoints = log_checkpoints
        self.tags = tags or {}
        self.run_id: Optional[str] = None

    def __enter__(self) -> "MLflowTracker":
        if self.tracking_uri:
            self._mlflow.set_tracking_uri(self.tracking_uri)
        self._mlflow.set_experiment(self.experiment)
        run = self._mlflow.start_run(run_name=self.run_name, tags=self.tags)
        self.run_id = run.info.run_id
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._mlflow.end_run(status="FAILED" if exc_type is not None else "FINISHED")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log hyperparameters (nested dicts are flattened)."""
        flat = {key: ("None" if value is None else value) for key, value in _flatten(params).items()}
        self._mlflow.log_params(flat)

    def log_epoch(self, epoch: int, metrics: dict[str, float]) -> None:
        """``fit(on_epoch_end=...)`` callback: log every numeric metric at step ``epoch``."""
        values = {k: float(v) for k, v in metrics.items() if k != "epoch" and isinstance(v, (int, float))}
        self._mlflow.log_metrics(values, step=epoch)

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        """Log one-off metrics, e.g. final test results with ``prefix="test_"``."""
        self._mlflow.log_metrics({f"{prefix}{k}": float(v) for k, v in metrics.items()})

    def log_run_outputs(self, result: dict[str, Any]) -> None:
        """Log the outcome of :func:`src.training.fit` and its files."""
        run_dir = Path(result["run_dir"])
        if result["best_metric"] is not None:
            self._mlflow.log_metrics({"best_metric": float(result["best_metric"]),
                                      "best_epoch": float(result["best_epoch"])})
        self._mlflow.set_tags({"run_dir": str(run_dir.resolve()),
                               "best_checkpoint": str((run_dir / "best.pt").resolve()),
                               "stopped_early": str(result["stopped_early"])})
        history = run_dir / "history.json"
        if history.is_file():
            self._mlflow.log_artifact(str(history))
        if self.log_checkpoints:
            for name in ("best.pt", "last.pt"):
                if (run_dir / name).is_file():
                    self._mlflow.log_artifact(str(run_dir / name), artifact_path="checkpoints")
