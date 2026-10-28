"""Training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

OPTIMIZERS = ("sgd", "adam", "adamw")
MONITORS = {"val_accuracy": "max", "val_loss": "min"}


@dataclass
class TrainConfig:
    """Settings for one classification training run.

    ``batch_size`` is not used by the training loop itself (the data loader
    owns batching); it is stored so every run records it.
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
    aux_weight: float = 0.3
    monitor: str = "val_accuracy"
    seed: int = 42
    deterministic: bool = False
    device: str = "auto"

    def __post_init__(self) -> None:
        if self.optimizer not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {self.optimizer!r}")
        if self.monitor not in MONITORS:
            raise ValueError(f"monitor must be one of {tuple(MONITORS)}, got {self.monitor!r}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {self.epochs}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
