"""Heatmap of a confusion matrix."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

MAX_NAMED_CLASSES = 30  # with more classes the species names do not fit; the axes show class indices
INDEX_TICK_STEP = 20
TEXT = "#0b0b0b"
MUTED = "#52514e"


def plot_confusion_matrix(confusion: torch.Tensor, class_names: Sequence[str], title: str, output: Path) -> None:
    """Save ``confusion`` ``[C, C]`` (rows actual, columns predicted) as a heatmap.

    Each row is shown as a share of that class's images, so the diagonal is the
    class's recall. The color scale is a square root: most confusions are a few
    percent, which a linear scale would leave almost white.
    """
    from matplotlib.colors import PowerNorm
    from matplotlib.figure import Figure  # no pyplot: works without a display and keeps no global state

    shares = confusion.double() / confusion.sum(dim=1, keepdim=True).clamp(min=1)
    positions = list(range(len(class_names)))
    named = len(class_names) <= MAX_NAMED_CLASSES
    if not named:
        positions = positions[::INDEX_TICK_STEP]
    labels = [class_names[position] if named else str(position) for position in positions]

    figure = Figure(figsize=(11, 9.5), layout="constrained", facecolor="white")
    axis = figure.subplots()
    shown = axis.imshow(shares.numpy(), cmap="Blues", norm=PowerNorm(gamma=0.5, vmin=0, vmax=1),
                        interpolation="nearest")
    axis.set_xticks(positions, labels=labels, rotation=90 if named else 0)
    axis.set_yticks(positions, labels=labels)
    axis.tick_params(colors=MUTED, length=0, labelsize=8)
    axis.set_xlabel("Predicted class", color=MUTED)
    axis.set_ylabel("Actual class", color=MUTED)
    axis.set_title(title, loc="left", fontsize=13, color=TEXT)
    for spine in axis.spines.values():
        spine.set_color(MUTED)
    colorbar = figure.colorbar(shown, ax=axis, shrink=0.6, ticks=[0, 0.05, 0.2, 0.5, 1])
    colorbar.ax.set_yticklabels(["0%", "5%", "20%", "50%", "100%"])
    colorbar.ax.tick_params(colors=MUTED, length=0, labelsize=9)
    colorbar.set_label("Share of the actual class's images (square-root scale)", color=MUTED, fontsize=9)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
