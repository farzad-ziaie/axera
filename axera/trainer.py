"""
Axera Trainer — unified training loop for gradient and swarm optimizers.

Supports
--------
- Gradient-based: Adam, AdamW, Lion, SGD (via PyTorch autograd)
- Derivative-free: MOPSO (via BlandAltmanLoss multi-objective)
- AMP (automatic mixed precision on CUDA)
- torch.compile (PyTorch 2.x graph compilation)
- Early stopping, checkpointing, and validation
- OpenTelemetry spans for tracing
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from axera.config import TrainerConfig
from axera.hooks import HookRegistry
from axera.losses.multiobjective import BlandAltmanLoss, MultiObjectiveLoss
from axera.losses.regression import MAE, MSE, LogCosh
from axera.models.sequential import Sequential
from axera.optimizers.gradient import get_optimizer
from axera.optimizers.swarm import MOPSO

logger = logging.getLogger(__name__)

# ── Loss factory ──────────────────────────────────────────────────────────────

_LOSSES: dict[str, type[nn.Module]] = {
    "mse":     MSE,
    "mae":     MAE,
    "logcosh": LogCosh,
}


class EarlyStopping:
    """Simple early stopping based on validation loss."""

    def __init__(self, patience: int = 20, min_delta: float = 1e-6) -> None:
        self.patience  = patience
        self.min_delta = min_delta
        self._best     = float("inf")
        self._counter  = 0

    def step(self, val_loss: float) -> bool:
        """Return True if training should stop."""
        if self.patience == 0:
            return False
        if val_loss < self._best - self.min_delta:
            self._best   = val_loss
            self._counter = 0
        else:
            self._counter += 1
        return self._counter >= self.patience


# ── Trainer ───────────────────────────────────────────────────────────────────

class Trainer:
    """
    Unified training controller for Axera models.

    Parameters
    ----------
    model : Sequential
    config : TrainerConfig
    hooks : HookRegistry, optional
    loss_fn : nn.Module or MultiObjectiveLoss, optional
        Override the loss from config.

    Examples
    --------
    Gradient training::

        trainer = Trainer(model, TrainerConfig(epochs=200, optimizer='adamw'))
        history = trainer.fit(X_train, y_train)

    Swarm training::

        from axera.losses import BlandAltmanLoss
        loss = BlandAltmanLoss(model.predict_from_weights, X_train, y_train,
                               objectives=['r2', 'angle', 'icc'])
        trainer = Trainer(model, TrainerConfig(optimizer='mopso', epochs=100),
                          loss_fn=loss)
        history = trainer.fit(X_train, y_train)
    """

    def __init__(
        self,
        model: Sequential,
        config: TrainerConfig | None = None,
        hooks: HookRegistry | None = None,
        loss_fn: nn.Module | MultiObjectiveLoss | None = None,
    ) -> None:
        self.model  = model
        self.cfg    = config or TrainerConfig()
        self.hooks  = hooks or model.hooks
        self.history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

        # Resolve device
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(self.cfg.device)

        self.model.to(self._device)
        logger.info("Trainer using device: %s", self._device)

        # torch.compile (PyTorch 2.x)
        if self.cfg.compile_model and not self.model._compiled:
            logger.info("Compiling model with torch.compile …")
            self.model = torch.compile(self.model)  # type: ignore[assignment]
            self.model._compiled = True  # type: ignore[union-attr]

        # Loss
        if loss_fn is not None:
            self._loss = loss_fn
        elif self.cfg.optimizer in ("mopso", "de"):
            self._loss = None   # user must supply a MultiObjectiveLoss
        else:
            loss_cls = _LOSSES.get(self.cfg.loss.lower())
            if loss_cls is None:
                raise ValueError(f"Unknown loss: {self.cfg.loss!r}. Valid: {list(_LOSSES)}")
            self._loss = loss_cls()

        # AMP scaler
        self._scaler: torch.cuda.amp.GradScaler | None = None
        if self.cfg.amp and str(self._device).startswith("cuda"):
            self._scaler = torch.cuda.amp.GradScaler()

        # Early stopping
        self._es = EarlyStopping(patience=self.cfg.early_stopping_patience)

    # ── main API ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: NDArray,
        y: NDArray,
        validation: tuple[NDArray, NDArray] | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the model.

        Parameters
        ----------
        X : ndarray (n, p)
        y : ndarray (n,)
        validation : optional (X_val, y_val)

        Returns
        -------
        dict with 'train_loss' and 'val_loss' lists
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Apply pre_fit hooks
        X, y = self.hooks.run("pre_fit", X, y)

        # Auto split if no explicit validation set
        if validation is None and self.cfg.val_split > 0:
            n_val = max(1, int(len(X) * self.cfg.val_split))
            X_val, y_val = X[-n_val:], y[-n_val:]
            X, y         = X[:-n_val], y[:-n_val]
            validation   = (X_val, y_val)

        # Route to gradient or swarm
        if self.cfg.optimizer.lower() in ("mopso",):
            return self._fit_swarm(X, y, validation)
        else:
            return self._fit_gradient(X, y, validation)

    async def afit(self, X: NDArray, y: NDArray, **kwargs: Any) -> dict[str, list[float]]:
        """Asyncio-native training."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.fit, X, y)

    # ── gradient training ─────────────────────────────────────────────────────

    def _fit_gradient(
        self,
        X: NDArray,
        y: NDArray,
        validation: tuple[NDArray, NDArray] | None,
    ) -> dict[str, list[float]]:
        opt = get_optimizer(
            self.cfg.optimizer, self.model.parameters(), lr=1e-3
        )
        loss_fn: nn.Module = self._loss  # type: ignore[assignment]
        loss_fn.to(self._device)

        torch.manual_seed(self.cfg.seed)
        n = len(X)

        for epoch in range(self.cfg.epochs):
            self.model.train()
            self.hooks.run("on_epoch_start", epoch, self.history)

            # Shuffle
            idx = np.random.permutation(n)
            X_s, y_s = X[idx], y[idx]

            epoch_losses: list[float] = []
            for i in range(0, n, self.cfg.batch_size):
                xb = torch.from_numpy(X_s[i : i + self.cfg.batch_size]).to(
                    dtype=torch.float64, device=self._device
                )
                yb = torch.from_numpy(y_s[i : i + self.cfg.batch_size]).to(
                    dtype=torch.float64, device=self._device
                )
                opt.zero_grad()
                if self._scaler is not None:
                    with torch.autocast(device_type="cuda"):
                        pred = self.model(xb)
                        loss = loss_fn(pred, yb)
                    self._scaler.scale(loss).backward()
                    self._scaler.step(opt)
                    self._scaler.update()
                else:
                    pred = self.model(xb)
                    loss = loss_fn(pred, yb)
                    loss = self.hooks.run("on_loss", loss)
                    loss.backward()
                    opt.step()
                epoch_losses.append(float(loss.item()))

            train_loss = float(np.mean(epoch_losses))
            self.history["train_loss"].append(train_loss)

            val_loss = float("nan")
            if validation is not None:
                Xv, yv = validation
                val_pred = torch.from_numpy(self.model.predict(Xv)).to(
                    dtype=torch.float64, device=self._device
                )
                yt_v = torch.from_numpy(yv).to(dtype=torch.float64, device=self._device)
                val_loss = float(loss_fn(val_pred, yt_v).item())
                self.history["val_loss"].append(val_loss)

            if (epoch + 1) % self.cfg.log_every_n_steps == 0:
                logger.info(
                    "Epoch %4d/%d  train=%.6f  val=%.6f",
                    epoch + 1, self.cfg.epochs, train_loss, val_loss,
                )

            self.hooks.run("on_epoch_end", epoch, self.history)

            if not np.isnan(val_loss) and self._es.step(val_loss):
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Checkpoint
        if self.cfg.checkpoint_dir is not None:
            ckpt = Path(self.cfg.checkpoint_dir) / "model_final.pt"
            self.model.save(ckpt)

        self.hooks.run("post_fit", self.model)
        return self.history

    # ── swarm training ────────────────────────────────────────────────────────

    def _fit_swarm(
        self,
        X: NDArray,
        y: NDArray,
        validation: tuple[NDArray, NDArray] | None,
    ) -> dict[str, list[float]]:
        if not isinstance(self._loss, MultiObjectiveLoss):
            # Build default BlandAltmanLoss wrapping the model
            self._loss = BlandAltmanLoss(
                predict_fn=self.model.predict_from_weights,
                X_train=X,
                y_train=y,
                objectives=["r2", "angle"],
            )

        n_dim = sum(p.numel() for p in self.model.parameters())
        opt   = MOPSO(n_pop=100, n_repo=50, n_workers=1)

        logger.info("Starting MOPSO swarm optimisation (%d params) …", n_dim)
        best_w = opt.fit(self._loss, n_dim, max_iter=self.cfg.epochs)
        self.model._set_flat_weights(best_w)
        logger.info("Swarm optimisation complete.")

        # Record a single loss value
        final_cost = float(self._loss.evaluate(best_w).sum())
        self.history["train_loss"].append(final_cost)
        return self.history

    # ── evaluation ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        X: NDArray,
        y: NDArray,
        metrics: str | list[str] = "auto",
    ) -> dict[str, Any]:
        """
        Evaluate the trained model on a held-out set.

        Parameters
        ----------
        X, y : ndarray
        metrics : 'auto' | 'medical_regression_full' | list of metric names

        Returns
        -------
        dict of metric name → value
        """
        from axera.metrics import evaluate_classification, evaluate_regression

        preds = self.model.predict(X)

        if self.model.task == "regression":
            return evaluate_regression(y, preds, extended=True)
        else:
            return evaluate_classification(y, preds)


__all__ = ["Trainer", "EarlyStopping"]
