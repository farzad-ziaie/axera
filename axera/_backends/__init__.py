"""
Backend abstraction layer.

Axera supports two backends:
  - ``torch``  : PyTorch tensors + autograd + GPU/MPS support (preferred)
  - ``numpy``  : Pure NumPy arrays, no autograd, CPU-only (fallback)

The active backend is selected once at import time from the environment
variable ``AXERA_BACKEND`` or via :func:`set_backend`.

Usage::

    from axera._backends import backend as bk
    x = bk.array([1.0, 2.0, 3.0])
    y = bk.zeros((4, 4))
"""

from __future__ import annotations

import os
from typing import Any

_BACKEND_NAME: str = os.environ.get("AXERA_BACKEND", "auto").lower()

# ── try to import torch ───────────────────────────────────────────────────────
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

if _BACKEND_NAME == "auto":
    _BACKEND_NAME = "torch" if _TORCH_AVAILABLE else "numpy"

if _BACKEND_NAME == "torch" and not _TORCH_AVAILABLE:
    import warnings
    warnings.warn(
        "AXERA_BACKEND=torch requested but PyTorch is not installed. "
        "Falling back to numpy backend.",
        RuntimeWarning,
        stacklevel=2,
    )
    _BACKEND_NAME = "numpy"

if _BACKEND_NAME == "torch":
    from axera._backends._torch import TorchBackend as _Backend
elif _BACKEND_NAME == "numpy":
    from axera._backends._numpy import NumpyBackend as _Backend  # type: ignore[assignment]
else:
    raise ValueError(f"Unknown backend: {_BACKEND_NAME!r}. Choose 'torch' or 'numpy'.")

backend: Any = _Backend()


def get_backend_name() -> str:
    """Return the name of the active backend: ``'torch'`` or ``'numpy'``."""
    return _BACKEND_NAME


def set_backend(name: str) -> None:
    """Switch the global backend at runtime (affects subsequent calls only)."""
    global backend, _BACKEND_NAME
    _BACKEND_NAME = name
    if name == "torch":
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed.")
        from axera._backends._torch import TorchBackend
        backend = TorchBackend()
    elif name == "numpy":
        from axera._backends._numpy import NumpyBackend
        backend = NumpyBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}")


__all__ = ["backend", "get_backend_name", "set_backend"]
