"""
LIP (Locally Independent Polynomial) activation functions.

Each neuron receives a list of input signals and applies a full polynomial
expansion over all non-empty combinations of those inputs up to a chosen
degree, giving the network interpretable, low-parameter expressiveness
suited to underdetermined biomedical datasets.

Mathematical form for a 2-input neuron with degree 2
(x₁, x₂ are inputs; θ are learned weights; b is bias):

    y = b + θ₀x₁ + θ₁x₁² + θ₂x₂ + θ₃x₂² + θ₄x₁x₂ + θ₅(x₁x₂)²

References
----------
Ivakhnenko, A. G. (1971). Polynomial theory of complex systems.
  IEEE Trans. Syst. Man Cybern., 1(4), 364–378.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

# Rust extension or pure-Python fallback
try:
    from axera._core import combinations_all  # type: ignore[import]
except ImportError:
    from axera._core_fallback import combinations_all


class LIP(nn.Module):
    """
    Locally Independent Polynomial activation layer.

    Parameters
    ----------
    input_size : int
        Number of input signals fed to this neuron.
    degree : int
        Maximum polynomial degree applied to each combination product.
    bias : bool
        If True, include a learnable scalar bias term.

    Attributes
    ----------
    combinations : list[list[int]]
        All non-empty subset indices (computed once at construction time).
    weight : nn.Parameter  shape (n_combinations * degree,)
    bias_param : nn.Parameter | None

    Examples
    --------
    >>> lip = LIP(input_size=3, degree=2)
    >>> x = torch.randn(16, 3)          # batch × features
    >>> y = lip(x)                       # batch-wise scalar output
    """

    def __init__(
        self,
        input_size: int,
        degree: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if input_size < 1:
            raise ValueError("input_size must be ≥ 1")
        if degree < 1:
            raise ValueError("degree must be ≥ 1")

        self.input_size = input_size
        self.degree = degree
        self._use_bias = bias

        self.combinations: list[list[int]] = combinations_all(input_size)
        n_terms = len(self.combinations) * degree

        w = torch.empty(n_terms)
        # kaiming_uniform_ needs ≥ 2-D; unsqueeze temporarily but don't squeeze back
        nn.init.kaiming_uniform_(w.unsqueeze(0), a=0.01)
        self.weight = nn.Parameter(w)          # always shape (n_terms,), never 0-dim

        self.bias_param: Optional[nn.Parameter]
        if bias:
            self.bias_param = nn.Parameter(torch.zeros(1))
        else:
            self.bias_param = None

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (..., input_size)
        Returns
        -------
        Tensor  shape (...,)  — scalar output per sample
        """
        out = torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

        if self.bias_param is not None:
            out = out + self.bias_param.squeeze()

        for combo_idx, combo in enumerate(self.combinations):
            # product of selected inputs: (...,)
            prod = x[..., combo[0]]
            for idx in combo[1:]:
                prod = prod * x[..., idx]

            for d in range(self.degree):
                param_idx = combo_idx * self.degree + d
                out = out + self.weight[param_idx] * prod ** (d + 1)

        return out

    # ── helpers ───────────────────────────────────────────────────────────────

    def extra_repr(self) -> str:
        return (
            f"input_size={self.input_size}, degree={self.degree}, "
            f"bias={self._use_bias}, n_params={self.n_params}"
        )

    @property
    def n_params(self) -> int:
        return self.weight.numel() + (1 if self._use_bias else 0)


# ── Composed LIP activations ─────────────────────────────────────────────────

class LIPTanh(nn.Module):
    """tanh( LIP(x) ) — bounded polynomial activation."""

    def __init__(self, input_size: int, degree: int = 2, bias: bool = True) -> None:
        super().__init__()
        self.lip = LIP(input_size, degree, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.lip(x))


class LIPSigmoid(nn.Module):
    """σ( LIP(x) ) — sigmoid polynomial activation."""

    def __init__(self, input_size: int, degree: int = 2, bias: bool = True) -> None:
        super().__init__()
        self.lip = LIP(input_size, degree, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.lip(x))


class LIPReLU(nn.Module):
    """ReLU( LIP(x) ) — rectified polynomial activation."""

    def __init__(
        self,
        input_size: int,
        degree: int = 2,
        bias: bool = True,
        alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.lip = LIP(input_size, degree, bias)
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.lip(x)
        return torch.where(h >= 0, h, self.alpha * h)


__all__ = ["LIP", "LIPTanh", "LIPSigmoid", "LIPReLU"]
