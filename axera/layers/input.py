"""Input normalization layer."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn


class InputLayer(nn.Module):
    """
    Input layer with optional feature normalization.

    Parameters
    ----------
    in_features : int
        Expected number of input features.
    normalize : {'none', 'standard', 'minmax'}
        Runtime normalization mode.  Statistics are estimated from the
        first forward pass (fit-on-first-batch) and then fixed.
    """

    def __init__(
        self,
        in_features: int,
        normalize: Literal["none", "standard", "minmax"] = "standard",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.normalize = normalize
        self._fitted = False

        self.register_buffer("mean_", torch.zeros(in_features))
        self.register_buffer("std_", torch.ones(in_features))
        self.register_buffer("min_", torch.zeros(in_features))
        self.register_buffer("max_", torch.ones(in_features))

    def fit(self, x: torch.Tensor) -> InputLayer:
        """Compute normalization statistics from data."""
        with torch.no_grad():
            self.mean_.copy_(x.mean(0))
            self.std_.copy_(x.std(0).clamp_min(1e-8))
            self.min_.copy_(x.min(0).values)
            self.max_.copy_((x.max(0).values - x.min(0).values).clamp_min(1e-8))
        self._fitted = True
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._fitted and self.normalize != "none":
            self.fit(x)
        if self.normalize == "standard":
            return (x - self.mean_) / self.std_
        if self.normalize == "minmax":
            return (x - self.min_) / self.max_
        return x
