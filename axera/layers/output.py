"""Output head layers."""

from __future__ import annotations

import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Linear output head for regression (single or multi-target)."""

    def __init__(self, in_features: int, out_features: int = 1) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.to(self.linear.weight.dtype))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


class ClassificationHead(nn.Module):
    """
    Output head for classification.

    For binary tasks (n_classes=2) outputs a single logit.
    For multiclass (n_classes>2) outputs raw logits of shape (batch, n_classes).
    Apply ``torch.sigmoid`` or ``torch.softmax`` externally as needed.
    """

    def __init__(self, in_features: int, n_classes: int = 2) -> None:
        super().__init__()
        self.n_classes = n_classes
        out = 1 if n_classes == 2 else n_classes
        self.linear = nn.Linear(in_features, out).double()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x.to(self.linear.weight.dtype))
        return out.squeeze(-1) if out.shape[-1] == 1 else out
