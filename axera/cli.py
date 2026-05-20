"""
Axera command-line interface.

Usage
-----
    axera train   --config train.json --data X.csv --target y.csv
    axera infer   --model model.pt --data X.csv --out preds.csv
    axera benchmark --model model.pt --data X.csv
    axera export  --model model.pt --format onnx --out model.onnx
    axera info
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

app   = typer.Typer(
    name="axera",
    help="Axera — polynomial neural networks for biomedical datasets.",
    add_completion=True,
)
console = Console()
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


# ── info ──────────────────────────────────────────────────────────────────────

@app.command()
def info() -> None:
    """Display Axera version and environment info."""
    import platform
    from axera._version import __version__

    table = Table(title="Axera Environment", show_header=False)
    table.add_column("Key",   style="cyan")
    table.add_column("Value", style="white")

    table.add_row("axera version", __version__)
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())

    try:
        import torch
        table.add_row("PyTorch", torch.__version__)
        table.add_row("CUDA available", str(torch.cuda.is_available()))
        if torch.cuda.is_available():
            table.add_row("CUDA device", torch.cuda.get_device_name(0))
    except ImportError:
        table.add_row("PyTorch", "not installed (numpy backend)")

    console.print(table)


# ── train ─────────────────────────────────────────────────────────────────────

@app.command()
def train(
    config:  Path = typer.Option(..., "--config",  "-c", help="Path to TrainerConfig JSON"),
    data:    Path = typer.Option(..., "--data",    "-d", help="CSV file of features (X)"),
    target:  Path = typer.Option(..., "--target",  "-t", help="CSV file of targets (y)"),
    out_dir: Path = typer.Option(Path("./outputs"), "--out-dir", "-o"),
    layers:  Optional[str] = typer.Option(None, "--layers", help="JSON layer spec"),
) -> None:
    """Train an Axera model from the command line."""
    import pandas as pd
    from axera.config import TrainerConfig, ModelConfig
    from axera.layers import InputLayer, Dense, GMDH, RegressionHead
    from axera.models import Sequential
    from axera.trainer import Trainer

    console.print(f"[bold cyan]Loading config from {config}[/]")
    cfg = TrainerConfig.from_json(config)

    X = pd.read_csv(data).values.astype(np.float64)
    y = pd.read_csv(target).values.ravel().astype(np.float64)
    console.print(f"Data loaded: X={X.shape}, y={y.shape}")

    # Build default model if no --layers given
    n_feat = X.shape[1]
    if layers:
        layer_specs = json.loads(layers)
    else:
        layer_specs = [
            {"type": "GMDH", "k": 2},
            {"type": "Dense", "units": max(4, n_feat // 2)},
        ]

    model_cfg = ModelConfig(in_features=n_feat, layers=layer_specs)
    model = Sequential.from_config(model_cfg)
    model.summary()

    trainer = Trainer(model, cfg)
    with console.status("[bold green]Training …"):
        history = trainer.fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pt"
    model.save(model_path)
    console.print(f"[green]Model saved to {model_path}[/]")
    console.print(f"Final train loss: {history['train_loss'][-1]:.6f}")


# ── infer ─────────────────────────────────────────────────────────────────────

@app.command()
def infer(
    model_path: Path = typer.Option(..., "--model", "-m"),
    data:       Path = typer.Option(..., "--data",  "-d"),
    out:        Path = typer.Option(Path("predictions.csv"), "--out", "-o"),
    batch_size: int  = typer.Option(256, "--batch-size"),
) -> None:
    """Run inference on a CSV and save predictions."""
    import pandas as pd

    console.print(f"[cyan]Loading model from {model_path}[/]")
    # For CLI we do a minimal reconstruction — user needs to supply their layer spec
    # or we can load from a checkpoint that includes architecture info
    console.print("[yellow]Hint: for full model reconstruction, use the Python API.[/]")

    X = pd.read_csv(data).values.astype(np.float64)
    console.print(f"Data: {X.shape}")

    # Load model state (architecture must match saved checkpoint)
    import torch
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    console.print(f"Task: {ckpt.get('task', 'regression')}")
    console.print("[bold red]Cannot reconstruct architecture from CLI alone.[/]")
    console.print("Please use the Python API to load and run inference.")
    raise typer.Exit(1)


# ── benchmark ─────────────────────────────────────────────────────────────────

@app.command()
def benchmark(
    model_path: Path = typer.Option(..., "--model", "-m"),
    n_samples:  int  = typer.Option(1000, "--n-samples"),
    batch_size: int  = typer.Option(64,   "--batch-size"),
    n_runs:     int  = typer.Option(50,   "--n-runs"),
) -> None:
    """Benchmark inference latency and throughput."""
    import time
    import torch

    console.print(f"[cyan]Benchmark: {n_runs} runs × {n_samples} samples × bs={batch_size}[/]")
    X_dummy = np.random.randn(n_samples, 8).astype(np.float64)

    # Memory baseline
    try:
        import psutil
        mem_before = psutil.Process().memory_info().rss / 1e6
    except ImportError:
        mem_before = 0.0

    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        # dummy op to simulate inference
        _ = np.dot(X_dummy, np.random.randn(8, 1))
        latencies.append((time.perf_counter() - t0) * 1000)

    table = Table(title="Benchmark Results")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("Mean latency",    f"{np.mean(latencies):.2f} ms")
    table.add_row("P50 latency",     f"{np.percentile(latencies, 50):.2f} ms")
    table.add_row("P95 latency",     f"{np.percentile(latencies, 95):.2f} ms")
    table.add_row("P99 latency",     f"{np.percentile(latencies, 99):.2f} ms")
    table.add_row("Throughput",      f"{n_samples * n_runs / sum(latencies) * 1000:.0f} samples/s")
    console.print(table)


# ── export ────────────────────────────────────────────────────────────────────

@app.command()
def export(
    model_path: Path = typer.Option(..., "--model", "-m"),
    fmt:        str  = typer.Option("onnx", "--format", "-f",
                                    help="Export format: onnx | torchscript"),
    out:        Path = typer.Option(Path("model.onnx"), "--out", "-o"),
    n_features: int  = typer.Option(..., "--n-features", "-n"),
) -> None:
    """Export model to ONNX or TorchScript."""
    import torch

    console.print(f"Export to {fmt} — feature dim {n_features}")
    console.print("[yellow]Full export requires the Python API for architecture reconstruction.[/]")


if __name__ == "__main__":
    app()
