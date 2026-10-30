"""Hyperparameter search spaces: grid (Cartesian product) and random sampling.

Every key is a :class:`src.training.TrainConfig` field, e.g. ``learning_rate``.

Grid search takes a list of values per key::

    {"learning_rate": [0.02, 0.05, 0.1], "weight_decay": [5e-4, 2e-3]}   # 3 x 2 = 6 trials

Random search takes one distribution per key::

    {"learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.15},
     "label_smoothing": {"type": "choice", "values": [0.0, 0.1, 0.2]}}

Distribution types: ``loguniform`` and ``uniform`` (floats in [low, high]),
``int`` (integers in [low, high]) and ``choice`` (one of ``values``).
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Any

DISTRIBUTIONS = ("loguniform", "uniform", "int", "choice")
SIGNIFICANT_DIGITS = 3  # sampled floats are rounded so run names and tables stay readable


def grid_trials(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Return every combination of the values in ``grid`` (keys keep their order)."""
    if not grid:
        raise ValueError("grid search space is empty")
    for key, values in grid.items():
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"grid values for {key!r} must be a non-empty list, got {values!r}")
    keys = list(grid)
    return [dict(zip(keys, combination)) for combination in itertools.product(*(grid[k] for k in keys))]


def _round(value: float) -> float:
    return float(f"{value:.{SIGNIFICANT_DIGITS}g}")


def validate_distribution(key: str, spec: dict[str, Any]) -> None:
    """Raise ``ValueError`` if ``spec`` is not a valid distribution."""
    if not isinstance(spec, dict) or spec.get("type") not in DISTRIBUTIONS:
        raise ValueError(f"{key!r}: distribution needs 'type' in {DISTRIBUTIONS}, got {spec!r}")
    if spec["type"] == "choice":
        if not spec.get("values"):
            raise ValueError(f"{key!r}: 'choice' needs a non-empty 'values' list")
        return
    if not all(isinstance(spec.get(bound), (int, float)) for bound in ("low", "high")):
        raise ValueError(f"{key!r}: {spec['type']!r} needs numeric 'low' and 'high' "
                         f"(write 1.0e-4, not 1e-4, in YAML), got {spec!r}")
    if spec["low"] > spec["high"]:
        raise ValueError(f"{key!r}: low must be <= high, got {spec['low']} > {spec['high']}")
    if spec["type"] == "loguniform" and spec["low"] <= 0:
        raise ValueError(f"{key!r}: 'loguniform' needs low > 0, got {spec['low']}")


def sample_value(spec: dict[str, Any], rng: random.Random) -> Any:
    """Draw one value from a distribution spec (see the module docstring)."""
    kind = spec["type"]
    if kind == "choice":
        return rng.choice(list(spec["values"]))
    low, high = spec["low"], spec["high"]
    if kind == "int":
        return rng.randint(int(low), int(high))
    if kind == "uniform":
        return _round(rng.uniform(low, high))
    return _round(math.exp(rng.uniform(math.log(low), math.log(high))))


def random_trials(space: dict[str, dict[str, Any]], num_trials: int, seed: int = 0) -> list[dict[str, Any]]:
    """Sample ``num_trials`` configurations; the same ``seed`` gives the same trials."""
    if not space:
        raise ValueError("random search space is empty")
    if num_trials < 1:
        raise ValueError(f"num_trials must be >= 1, got {num_trials}")
    for key, spec in space.items():
        validate_distribution(key, spec)
    rng = random.Random(seed)
    return [{key: sample_value(spec, rng) for key, spec in space.items()} for _ in range(num_trials)]
