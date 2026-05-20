"""Standard activation wrappers as nn.Module subclasses."""

from __future__ import annotations

import torch
import torch.nn as nn


class Tanh(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)


class Sigmoid(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x)


class LeakyReLU(nn.Module):
    def __init__(self, alpha: float = 0.01) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x >= 0, x, self.alpha * x)


__all__ = ["Tanh", "Sigmoid", "ReLU", "LeakyReLU"]
