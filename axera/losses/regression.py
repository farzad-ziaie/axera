"""Differentiable regression losses (for gradient-based optimizers)."""

from __future__ import annotations

import torch
import torch.nn as nn


class MSE(nn.Module):
    """Mean Squared Error."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2).mean()


class MAE(nn.Module):
    """Mean Absolute Error."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.abs(pred - target).mean()


class LogCosh(nn.Module):
    """Log-cosh loss — smooth approximation to MAE, twice differentiable."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.cosh(pred - target)).mean()


class HuberLoss(nn.Module):
    """Huber (smooth L1) loss."""
    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.huber_loss(pred, target, delta=self.delta)


__all__ = ["MSE", "MAE", "LogCosh", "HuberLoss"]
