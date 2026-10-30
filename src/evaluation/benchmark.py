"""Technical benchmark: parameter counts, forward latency and peak memory.

These numbers describe model cost only, not accuracy. Run:

    python -m src.evaluation.benchmark --models resnet18 vgg16 --image-size 224
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
from torch import nn

from src.models import available_models, count_parameters, create_model


@dataclass
class BenchmarkResult:
    """Cost measurements for one model in eval mode."""

    model: str
    device: str
    num_classes: int
    batch_size: int
    image_size: int
    total_params: int
    trainable_params: int
    latency_ms_mean: float
    latency_ms_median: float
    latency_ms_std: float
    peak_memory_mb: Optional[float]
    note: str = ""


def environment_info(device: torch.device) -> dict[str, Any]:
    """Describe the hardware and software the benchmark ran on."""
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_threads": torch.get_num_threads(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["cuda_version"] = torch.version.cuda
    return info


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark_model(
    model: nn.Module,
    name: str,
    num_classes: int,
    device: torch.device,
    batch_size: int = 1,
    image_size: int = 224,
    warmup: int = 3,
    iterations: int = 10,
) -> BenchmarkResult:
    """Time ``iterations`` eval-mode forward passes after ``warmup`` untimed ones.

    Peak memory is measured with the CUDA allocator and is only available on
    GPU; on CPU it is reported as ``None``.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    total, trainable = count_parameters(model)
    model = model.to(device).eval()
    inputs = torch.randn(batch_size, 3, image_size, image_size, device=device)

    with torch.inference_mode():
        for _ in range(warmup):
            model(inputs)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        timings = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(inputs)
            _synchronize(device)
            timings.append((time.perf_counter() - start) * 1000.0)

    peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    note = "" if peak_mb is not None else "peak memory not measured on CPU"
    return BenchmarkResult(
        model=name,
        device=str(device),
        num_classes=num_classes,
        batch_size=batch_size,
        image_size=image_size,
        total_params=total,
        trainable_params=trainable,
        latency_ms_mean=statistics.mean(timings),
        latency_ms_median=statistics.median(timings),
        latency_ms_std=statistics.stdev(timings) if len(timings) > 1 else 0.0,
        peak_memory_mb=peak_mb,
        note=note,
    )


def save_results(results: Sequence[BenchmarkResult], environment: dict[str, Any], output: Path) -> None:
    """Write ``<output>.json`` (results + environment) and ``<output>.csv``."""
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    payload = {"environment": environment, "results": rows}
    output.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[Sequence[str]] = None) -> list[BenchmarkResult]:
    parser = argparse.ArgumentParser(description="Benchmark model size and forward latency.")
    parser.add_argument("--models", nargs="+", default=available_models(), help="model names (default: all)")
    parser.add_argument("--num-classes", type=int, default=1000,
                        help="classifier size; 1000 matches the lecture's ImageNet numbers")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', 'cuda', 'cuda:0', ...")
    parser.add_argument("--output", type=Path, default=None,
                        help="output path without extension (default: results/benchmark_<device>)")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    output = args.output or Path("results") / f"benchmark_{device.type}"
    environment = environment_info(device)
    print(json.dumps(environment, indent=2))

    results = []
    for name in args.models:
        torch.manual_seed(0)
        model = create_model(name, num_classes=args.num_classes)
        result = benchmark_model(model, name, args.num_classes, device, args.batch_size,
                                 args.image_size, args.warmup, args.iterations)
        results.append(result)
        memory = f"{result.peak_memory_mb:.1f} MB" if result.peak_memory_mb is not None else "n/a"
        print(f"{name:<10} params={result.total_params:>12,}  "
              f"latency={result.latency_ms_mean:8.1f} ± {result.latency_ms_std:6.1f} ms  peak_mem={memory}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    save_results(results, environment, output)
    print(f"Saved {output.with_suffix('.json')} and {output.with_suffix('.csv')}")
    return results


if __name__ == "__main__":
    main()
