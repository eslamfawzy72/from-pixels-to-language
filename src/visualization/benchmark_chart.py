"""Bar chart of our test accuracy against published CUB-200-2011 results.

    python -m src.visualization.benchmark_chart
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional, Sequence

GROUPS = {"ours": ("This project", "#2a78d6"), "published": ("Published", "#eb6834")}
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def plot_benchmark(rows: Sequence[dict[str, str]], output: Path) -> None:
    """Save a horizontal bar chart of ``rows``, highest accuracy at the top."""
    from matplotlib.figure import Figure  # no pyplot: works without a display and keeps no global state
    from matplotlib.patches import Patch

    rows = sorted(rows, key=lambda row: float(row["top1_accuracy"]))
    labels = [f"{row['model']}\n{row['source']}" for row in rows]
    values = [float(row["top1_accuracy"]) for row in rows]

    figure = Figure(figsize=(9.5, 6), layout="constrained", facecolor="white")
    axis = figure.subplots()
    axis.barh(labels, values, height=0.62, color=[GROUPS[row["group"]][1] for row in rows])
    for position, value in enumerate(values):
        axis.text(value + 1, position, f"{value:.1f}%", va="center", fontsize=10, color=TEXT)

    axis.set_xlim(0, 100)
    axis.set_xlabel("Top-1 test accuracy (%)", color=MUTED)
    axis.set_title("Top-1 accuracy on CUB-200-2011: this project vs published results", loc="left", fontsize=13, color=TEXT)
    axis.tick_params(axis="x", colors=MUTED, length=0)
    axis.tick_params(axis="y", colors=TEXT, length=0, labelsize=9)
    axis.grid(axis="x", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    for side in ("top", "right", "bottom"):
        axis.spines[side].set_visible(False)
    axis.spines["left"].set_color(MUTED)
    axis.legend(handles=[Patch(color=color, label=label) for label, color in GROUPS.values()],
                loc="lower right", frameon=False, labelcolor=TEXT)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Plot our accuracy against published results.")
    parser.add_argument("--input", type=Path, default=Path("results/benchmarking/benchmark.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmarking/benchmark_accuracy.png"))
    args = parser.parse_args(argv)

    with open(args.input, newline="", encoding="utf-8") as handle:
        plot_benchmark(list(csv.DictReader(handle)), args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
