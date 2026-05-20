"""NumPy-only backend — no PyTorch dependency."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


class NumpyBackend:
    """Thin numpy wrapper that mirrors the TorchBackend API surface."""

    name = "numpy"

    # ── Array creation ────────────────────────────────────────────────────────

    def array(self, data: ArrayLike, dtype: Any = np.float64) -> NDArray:
        return np.asarray(data, dtype=dtype)

    def zeros(self, shape: tuple[int, ...], dtype: Any = np.float64) -> NDArray:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: tuple[int, ...], dtype: Any = np.float64) -> NDArray:
        return np.ones(shape, dtype=dtype)

    def randn(self, *shape: int) -> NDArray:
        return np.random.randn(*shape)

    def rand(self, *shape: int) -> NDArray:
        return np.random.rand(*shape)

    def arange(self, *args: Any) -> NDArray:
        return np.arange(*args)

    def linspace(self, start: float, stop: float, num: int) -> NDArray:
        return np.linspace(start, stop, num)

    # ── Math ops ──────────────────────────────────────────────────────────────

    def tanh(self, x: NDArray) -> NDArray:
        return np.tanh(x)

    def sigmoid(self, x: NDArray) -> NDArray:
        return 1.0 / (1.0 + np.exp(-x))

    def relu(self, x: NDArray, alpha: float = 0.0) -> NDArray:
        return np.where(x >= 0, x, alpha * x)

    def exp(self, x: NDArray) -> NDArray:
        return np.exp(x)

    def log(self, x: NDArray) -> NDArray:
        return np.log(x)

    def cosh(self, x: NDArray) -> NDArray:
        return np.cosh(x)

    def abs(self, x: NDArray) -> NDArray:
        return np.abs(x)

    def mean(self, x: NDArray, axis: int | None = None) -> NDArray:
        return np.mean(x, axis=axis)

    def sum(self, x: NDArray, axis: int | None = None) -> NDArray:
        return np.sum(x, axis=axis)

    def sqrt(self, x: NDArray) -> NDArray:
        return np.sqrt(x)

    def concatenate(self, arrays: list[NDArray], axis: int = 0) -> NDArray:
        return np.concatenate(arrays, axis=axis)

    def stack(self, arrays: list[NDArray], axis: int = 0) -> NDArray:
        return np.stack(arrays, axis=axis)

    def split(self, x: NDArray, n: int, axis: int = 0) -> list[NDArray]:
        return np.array_split(x, n, axis=axis)

    # ── Device / dtype ────────────────────────────────────────────────────────

    def to_numpy(self, x: NDArray | Any) -> NDArray:
        return np.asarray(x)

    def device(self) -> str:
        return "cpu"

    def is_available_gpu(self) -> bool:
        return False

    # ── Grad (no-op for numpy) ────────────────────────────────────────────────

    def no_grad(self):  # noqa: ANN202
        """Context manager (no-op for numpy backend)."""
        import contextlib
        return contextlib.nullcontext()

    # ── Module base ───────────────────────────────────────────────────────────

    def module_class(self) -> type:
        """Return the base Module class for this backend."""
        return _NumpyModule


class _NumpyModule:
    """Minimal Module-like base class for NumPy layers."""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def parameters(self) -> list[NDArray]:
        return []

    def train(self, mode: bool = True) -> _NumpyModule:
        return self

    def eval(self) -> _NumpyModule:
        return self
