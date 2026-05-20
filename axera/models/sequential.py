"""
Sequential model — the primary user-facing model class.

Supports both PyTorch autograd (gradient optimizers) and derivative-free
(MOPSO) training via a unified API.  Compatible with the NumPy backend.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from axera.config import ModelConfig
from axera.hooks import HookRegistry
from axera.layers.dense import GMDH, Dense
from axera.layers.input import InputLayer
from axera.layers.output import ClassificationHead, RegressionHead

logger = logging.getLogger(__name__)


class Sequential(nn.Module):
    """
    Axera sequential model with LIP/GMDH layers.

    Parameters
    ----------
    layers : list of nn.Module
        Ordered list of layers.  The first must be an ``InputLayer``.
    task : str
        ``'regression'``, ``'binary'``, or ``'multiclass'``.
    n_classes : int
        Output width for classification (ignored for regression).

    Examples
    --------
    >>> from axera import Sequential
    >>> from axera.layers import InputLayer, GMDH, Dense, RegressionHead
    >>> from axera.activations import LIP
    >>>
    >>> model = Sequential([
    ...     InputLayer(in_features=8),
    ...     GMDH(in_features=8, k=2),
    ...     Dense(out_features=4, in_features=28),  # C(8,2) = 28
    ...     RegressionHead(in_features=4),
    ... ])
    """

    def __init__(
        self,
        layers: list[nn.Module],
        task: Literal["regression", "binary", "multiclass"] = "regression",
        n_classes: int = 1,
        hooks: HookRegistry | None = None,
    ) -> None:
        super().__init__()
        if not layers:
            raise ValueError("layers must be non-empty.")
        self.layer_stack = nn.ModuleList(layers)
        self.task        = task
        self.n_classes   = n_classes
        self.hooks       = hooks or HookRegistry()
        self._weights_n: NDArray | None = None   # swarm optimizer state
        self._compiled   = False

    # ── Construction helpers ──────────────────────────────────────────────────

    @classmethod
    def from_config(cls, cfg: ModelConfig) -> Sequential:
        """Build a Sequential model from a ``ModelConfig`` object."""

        layers: list[nn.Module] = [
            InputLayer(cfg.in_features, normalize=cfg.normalize_input)
        ]

        prev_out = cfg.in_features
        for spec in cfg.layers:
            ltype = spec.get("type", "Dense").lower()
            if ltype == "gmdh":
                k    = spec.get("k", 2)
                layer = GMDH(
                    in_features=prev_out,
                    k=k,
                    activation=spec.get("activation", "lip"),
                    degree=spec.get("degree", 2),
                )
                prev_out = layer.out_features
            elif ltype == "dense":
                units = spec.get("units", 16)
                layer = Dense(
                    out_features=units,
                    in_features=prev_out,
                    activation=spec.get("activation", "lip"),
                    degree=spec.get("degree", 2),
                )
                prev_out = units
            else:
                raise ValueError(f"Unknown layer type: {ltype!r}")
            layers.append(layer)

        # Output head
        n_out = 1 if cfg.task == "regression" else cfg.n_classes
        if cfg.task == "regression":
            layers.append(RegressionHead(prev_out, 1))
        else:
            layers.append(ClassificationHead(prev_out, cfg.n_classes))

        return cls(layers, task=cfg.task, n_classes=n_out)

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layer_stack:
            x = layer(x)
        return x

    # ── Inference API ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, X: NDArray | torch.Tensor, batch_size: int = 256) -> NDArray:
        """
        Run prediction on X.

        Parameters
        ----------
        X : array or Tensor  shape (n, p)
        batch_size : int   chunks X to avoid OOM on large datasets

        Returns
        -------
        NDArray  shape (n,) for regression/binary, (n, n_classes) for multiclass
        """
        self.eval()
        device = next(self.parameters()).device

        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X).to(dtype=torch.float64, device=device)
        else:
            X_t = X.to(dtype=torch.float64, device=device)

        X_t = self.hooks.run("pre_predict", X_t)

        results = []
        for i in range(0, len(X_t), batch_size):
            batch = X_t[i : i + batch_size]
            out = self(batch)
            results.append(out.cpu())

        pred = torch.cat(results, dim=0).numpy()

        if self.task == "binary":
            pred = torch.sigmoid(torch.from_numpy(pred)).numpy()
        elif self.task == "multiclass":
            pred = torch.softmax(torch.from_numpy(pred), dim=-1).numpy()

        return self.hooks.run("post_predict", pred)

    async def apredict(self, X: NDArray | torch.Tensor, batch_size: int = 256) -> NDArray:
        """Asyncio-native prediction (runs predict in executor)."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.predict, X, batch_size)

    def predict_batch(
        self, prompts: list[NDArray], batch_size: int = 256
    ) -> list[NDArray]:
        """
        Batched prediction for throughput workloads.

        Parameters
        ----------
        prompts : list of NDArray — variable-length list of input arrays

        Returns
        -------
        list of NDArray — in the same order as ``prompts``
        """
        X = np.vstack(prompts)
        preds = self.predict(X, batch_size=batch_size)
        # split back into per-prompt chunks
        sizes = [len(p) for p in prompts]
        out = []
        i = 0
        for s in sizes:
            out.append(preds[i : i + s])
            i += s
        return out

    # ── swarm predict helper (flat weights → output) ──────────────────────────

    def predict_from_weights(self, weights: NDArray, X: NDArray) -> NDArray:
        """
        Used by the swarm optimizer's cost oracle.
        Temporarily load a weight vector, predict, restore originals.
        """
        orig = self._get_flat_weights()
        self._set_flat_weights(weights)
        out  = self.predict(X)
        self._set_flat_weights(orig)
        return out

    def _get_flat_weights(self) -> NDArray:
        return np.concatenate([
            p.detach().cpu().numpy().ravel() for p in self.parameters()
        ])

    def _set_flat_weights(self, weights: NDArray) -> None:
        ptr = 0
        for p in self.parameters():
            n = p.numel()
            p.data = torch.from_numpy(
                weights[ptr : ptr + n].reshape(p.shape)
            ).to(dtype=p.dtype, device=p.device)
            ptr += n

    # ── serialisation ─────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model weights and architecture description."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "task":       self.task,
            "n_classes":  self.n_classes,
        }, path)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str | Path, layers: list[nn.Module], **kwargs: Any) -> Sequential:
        """Load model weights from path."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(layers, task=ckpt["task"], n_classes=ckpt["n_classes"], **kwargs)
        model.load_state_dict(ckpt["state_dict"])
        logger.info("Model loaded from %s", path)
        return model

    # ── introspection ─────────────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """Print and return a layer-by-layer parameter summary."""
        rows = []
        total = 0
        for i, layer in enumerate(self.layer_stack):
            n = sum(p.numel() for p in layer.parameters())
            total += n
            rows.append({"layer": i, "type": type(layer).__name__, "params": n})
        print(f"{'Layer':<6} {'Type':<24} {'Params':>8}")
        print("─" * 42)
        for r in rows:
            print(f"{r['layer']:<6} {r['type']:<24} {r['params']:>8,}")
        print("─" * 42)
        print(f"{'Total':>30} {total:>8,}")
        return {"layers": rows, "total_params": total}


__all__ = ["Sequential"]
