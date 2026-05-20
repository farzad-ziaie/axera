"""PyTorch backend — full autograd + CUDA/MPS support."""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray


def _resolve_device(device: Optional[str] = None) -> torch.device:
    """Auto-select best available device: CUDA → MPS → CPU."""
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchBackend:
    """PyTorch backend wrapping common ops with auto-device selection."""

    name = "torch"

    def __init__(self, device: Optional[str] = None) -> None:
        self._device = _resolve_device(device)

    # ── Array creation ────────────────────────────────────────────────────────

    def array(self, data: Any, dtype: Any = torch.float64) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(dtype=dtype, device=self._device)
        return torch.tensor(data, dtype=dtype, device=self._device)

    def zeros(self, shape: tuple[int, ...], dtype: Any = torch.float64) -> torch.Tensor:
        return torch.zeros(shape, dtype=dtype, device=self._device)

    def ones(self, shape: tuple[int, ...], dtype: Any = torch.float64) -> torch.Tensor:
        return torch.ones(shape, dtype=dtype, device=self._device)

    def randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, dtype=torch.float64, device=self._device)

    def rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, dtype=torch.float64, device=self._device)

    def arange(self, *args: Any) -> torch.Tensor:
        return torch.arange(*args, device=self._device)

    def linspace(self, start: float, stop: float, num: int) -> torch.Tensor:
        return torch.linspace(start, stop, num, device=self._device)

    # ── Math ops ──────────────────────────────────────────────────────────────

    def tanh(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x)

    def sigmoid(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)

    def relu(self, x: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
        return torch.where(x >= 0, x, alpha * x)

    def exp(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(x)

    def log(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x)

    def cosh(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cosh(x)

    def abs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.abs(x)

    def mean(self, x: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        return x.mean() if axis is None else x.mean(dim=axis)

    def sum(self, x: torch.Tensor, axis: Optional[int] = None) -> torch.Tensor:
        return x.sum() if axis is None else x.sum(dim=axis)

    def sqrt(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(x)

    def concatenate(self, arrays: list[torch.Tensor], axis: int = 0) -> torch.Tensor:
        return torch.cat(arrays, dim=axis)

    def stack(self, arrays: list[torch.Tensor], axis: int = 0) -> torch.Tensor:
        return torch.stack(arrays, dim=axis)

    def split(self, x: torch.Tensor, n: int, axis: int = 0) -> list[torch.Tensor]:
        return torch.tensor_split(x, n, dim=axis)

    # ── Device / dtype ────────────────────────────────────────────────────────

    def to_numpy(self, x: Union[torch.Tensor, NDArray, Any]) -> NDArray:
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def device(self) -> str:
        return str(self._device)

    def is_available_gpu(self) -> bool:
        return torch.cuda.is_available() or (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    # ── Grad ──────────────────────────────────────────────────────────────────

    def no_grad(self):  # type: ignore[return]
        return torch.no_grad()

    # ── Module base ───────────────────────────────────────────────────────────

    def module_class(self) -> type:
        return nn.Module
