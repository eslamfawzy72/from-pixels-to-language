"""Training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional

OPTIMIZERS = ("sgd", "adam", "adamw")
SCHEDULERS = ("none", "cosine", "step", "plateau")
MONITORS = {"val_accuracy": "max", "val_loss": "min"}


@dataclass
class TrainConfig:
    """Settings for one classification training run.

    ``batch_size`` is not used by the training loop itself (the data loader
    owns batching); it is stored so every run records it.

    Learning-rate schedule (stepped once per epoch):
        * ``"none"``: constant learning rate
        * ``"cosine"``: cosine decay from ``learning_rate`` to ``min_lr``
        * ``"step"``: multiply by ``gamma`` every ``step_size`` epochs
        * ``"plateau"``: multiply by ``gamma`` when ``monitor`` stops improving
          for ``plateau_patience`` epochs
    ``warmup_epochs`` > 0 linearly ramps the learning rate up first (not
    supported with ``"plateau"``).

    ``early_stopping_patience`` stops training after that many epochs without
    improvement in ``monitor``; ``None`` disables it.
    """

    model_name: str
    num_classes: int
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    epochs: int = 30
    batch_size: int = 32
    optimizer: str = "sgd"
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    scheduler: str = "cosine"
    warmup_epochs: int = 0
    min_lr: float = 0.0
    step_size: int = 30
    gamma: float = 0.1
    plateau_patience: int = 5
    label_smoothing: float = 0.0
    early_stopping_patience: Optional[int] = None
    aux_weight: float = 0.3
    monitor: str = "val_accuracy"
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.scheduler not in SCHEDULERS:
            raise ValueError(f"scheduler must be one of {SCHEDULERS}, got {self.scheduler!r}")
        if self.monitor not in MONITORS:
            raise ValueError(f"monitor must be one of {tuple(MONITORS)}, got {self.monitor!r}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if not 0 <= self.warmup_epochs < self.epochs:
            raise ValueError(f"warmup_epochs must be in [0, epochs), got {self.warmup_epochs}")
        if self.warmup_epochs and self.scheduler == "plateau":
            raise ValueError("warmup_epochs is not supported with the 'plateau' scheduler")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError(f"early_stopping_patience must be >= 1 or None, got {self.early_stopping_patience}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        """Build a config from a dict, rejecting unknown keys with a clear message."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"Unknown TrainConfig keys: {unknown}. Valid keys: {sorted(known)}")
        return cls(**values)
