"""
Gradient-based optimizers for Axera.

Adam is re-implemented from scratch, correcting three bugs in the original:
  1. V accumulation: was ``β₂² + (1−β₂)g²``, now ``β₂V + (1−β₂)g²``
  2. Bias correction used fixed ``**2`` exponent, now uses ``**t``
  3. Parent ``__init__`` was never called in SGD/RMSprop (fixed)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.optim import Optimizer

# ── Thin wrappers around torch.optim ─────────────────────────────────────────

def get_optimizer(
    name: str,
    params, # noqa: ANN001
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    **kwargs,
) -> Optimizer:
    """
    Factory for gradient-based optimizers.

    Parameters
    ----------
    name : str
        ``'adam'``, ``'adamw'``, ``'sgd'``, ``'rmsprop'``, ``'lion'``
    params : iterable
        Model parameters (from ``model.parameters()``).
    lr : float
        Learning rate.
    weight_decay : float
    **kwargs
        Passed to the underlying optimizer.

    Returns
    -------
    torch.optim.Optimizer
    """
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay, **kwargs)
    if name == "lion":
        return Lion(params, lr=lr, weight_decay=weight_decay, **kwargs)
    raise ValueError(f"Unknown optimizer: {name!r}. "
                     f"Valid: 'adam', 'adamw', 'sgd', 'rmsprop', 'lion'")


# ── Lion optimizer ────────────────────────────────────────────────────────────

class Lion(Optimizer):
    """
    Lion (EvoLved Sign Momentum) optimizer — Chen et al. (2023).
    https://arxiv.org/abs/2302.06675

    More memory-efficient than Adam (stores only momentum, not variance).
    Works well on small datasets with high regularisation.
    """

    def __init__(
        self,
        params, # noqa: ANN001
        lr: float = 1e-4,
        betas: tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr}")
        defaults = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> float | None:  # noqa: ANN001
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                m = state["exp_avg"]
                # update: p = p - lr * (sign(β₁m + (1−β₁)g) + wd*p)
                update = (beta1 * m + (1 - beta1) * g).sign()
                p.add_(update + wd * p, alpha=-lr)
                # momentum update
                m.mul_(beta2).add_(g, alpha=1 - beta2)

        return loss


# ── Numpy Adam (used by MOPSO-free swarm path, not needed by torch trainer) ──

class NumpyAdam:
    """
    NumPy-only Adam for weight-update in the derivative-free / numpy backend.

    Fixed relative to the original optimizers.py:
      - V update: ``self.V = β₂·V + (1−β₂)·g²``  (not ``β₂² + …``)
      - Bias correction: ``/(1 - β**t)`` with step counter
    """

    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        self.lr     = lr
        self.beta1  = beta1
        self.beta2  = beta2
        self.eps    = eps
        self.wd     = weight_decay
        self.t      = 0
        self.M: np.ndarray | None = None   # first moment
        self.V: np.ndarray | None = None   # second moment

    def init(self, n_params: int) -> None:
        self.M = np.zeros(n_params)
        self.V = np.zeros(n_params)
        self.t = 0

    def step(self, grad: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """Return updated weights given gradient ``grad``."""
        if self.M is None or self.V is None:
             raise ValueError("Call init() first to initialize optimizer state.")
        self.t += 1
        g = grad + self.wd * weights
        self.M = self.beta1 * self.M + (1 - self.beta1) * g
        self.V = self.beta2 * self.V + (1 - self.beta2) * (g ** 2)
        m_hat = self.M / (1 - self.beta1 ** self.t)
        v_hat = self.V / (1 - self.beta2 ** self.t)
        return weights - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


__all__ = ["get_optimizer", "Lion", "NumpyAdam"]
