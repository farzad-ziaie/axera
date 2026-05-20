"""
Axera neural network layers.

Dense   — fully-connected layer, each neuron can use any activation.
GMDH    — Group Method of Data Handling layer; neurons receive all k-wise
           combinations of the previous layer's outputs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from axera.activations.lip import LIP, LIPReLU, LIPSigmoid, LIPTanh

# Rust extension or fallback
try:
    from axera._core import combinations_k  # type: ignore[import]
except ImportError:
    from axera._core_fallback import combinations_k

_ActivationType = LIP | LIPTanh | LIPSigmoid | LIPReLU | nn.Module


# ── Dense ─────────────────────────────────────────────────────────────────────

class Dense(nn.Module):
    """
    Fully-connected layer where every neuron receives all inputs.

    Each neuron is an independent ``LIP`` (or any ``nn.Module``) applied
    to the full input vector.  Output is a 1-D vector of per-neuron scalars.

    Parameters
    ----------
    out_features : int
        Number of neurons (output width).
    in_features : int
        Number of input signals.  Set automatically by :class:`Sequential`
        if ``None``.
    activation : str or nn.Module
        Activation type: ``'lip'``, ``'tanh'``, ``'sigmoid'``, ``'relu'``,
        or any ``nn.Module``.
    degree : int
        Polynomial degree (used only when activation is LIP-based).
    bias : bool
        Include per-neuron bias.

    Examples
    --------
    >>> layer = Dense(out_features=8, in_features=4, activation='lip', degree=2)
    >>> x = torch.randn(32, 4)
    >>> y = layer(x)        # (32, 8)
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        activation: str | nn.Module = "lip",
        degree: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.out_features = out_features
        self.in_features = in_features
        self.degree = degree

        self.neurons = nn.ModuleList()
        for _ in range(out_features):
            self.neurons.append(
                _build_activation(activation, in_features, degree, bias)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (batch, in_features)
        Returns
        -------
        Tensor  shape (batch, out_features)
        """
        outs = [neuron(x) for neuron in self.neurons]
        return torch.stack(outs, dim=-1)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, degree={self.degree}"


# ── GMDH ─────────────────────────────────────────────────────────────────────

class GMDH(nn.Module):
    """
    Group Method of Data Handling (GMDH) layer.

    Each neuron receives exactly *k* inputs selected from all C(n, k)
    combinations of the previous layer's outputs, then applies the chosen
    activation to those *k* inputs.

    Parameters
    ----------
    in_features : int
        Width of the incoming layer (must be ≥ k).
    k : int
        Number of inputs per neuron (pair = 2 is classic GMDH).
    activation : str or nn.Module
        Activation applied to each k-tuple.
    degree : int
        Polynomial degree for LIP-based activations.
    bias : bool
        Include per-neuron bias.

    Examples
    --------
    >>> layer = GMDH(in_features=6, k=2, activation='lip', degree=2)
    >>> x = torch.randn(32, 6)
    >>> y = layer(x)        # (32, C(6,2)) = (32, 15)
    """

    def __init__(
        self,
        in_features: int,
        k: int = 2,
        activation: str | nn.Module = "lip",
        degree: int = 2,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_features < k:
            raise ValueError(
                f"in_features ({in_features}) must be ≥ k ({k}) for GMDH layer."
            )
        self.in_features = in_features
        self.k = k
        self.degree = degree

        self.index_pairs: list[list[int]] = combinations_k(in_features, k)
        self.out_features = len(self.index_pairs)

        self.neurons = nn.ModuleList()
        for _ in self.index_pairs:
            self.neurons.append(
                _build_activation(activation, k, degree, bias)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor  shape (batch, in_features)
        Returns
        -------
        Tensor  shape (batch, out_features)
        """
        outs = []
        for neuron, idxs in zip(self.neurons, self.index_pairs, strict=True):
            x_sub = x[:, idxs]          # (batch, k)
            outs.append(neuron(x_sub))  # (batch,)
        return torch.stack(outs, dim=-1)  # (batch, out_features)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, k={self.k}, "
            f"out={self.out_features}, degree={self.degree}"
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_activation(
    activation: str | nn.Module,
    in_features: int,
    degree: int,
    bias: bool,
) -> nn.Module:
    """Instantiate an activation module given a name or a pre-built module."""
    if isinstance(activation, str):
        key = activation.lower()
        if key == "lip":
            return LIP(in_features, degree=degree, bias=bias)
        if key in ("lip_tanh", "liptanh"):
            return LIPTanh(in_features, degree=degree, bias=bias)
        if key in ("lip_sigmoid", "lipsigmoid"):
            return LIPSigmoid(in_features, degree=degree, bias=bias)
        if key in ("lip_relu", "liprelu"):
            return LIPReLU(in_features, degree=degree, bias=bias)
        if key == "tanh":
            return nn.Sequential(nn.Linear(in_features, 1, bias=bias).double(), nn.Tanh())
        if key == "sigmoid":
            return nn.Sequential(nn.Linear(in_features, 1, bias=bias).double(), nn.Sigmoid())
        if key == "relu":
            return nn.Sequential(nn.Linear(in_features, 1, bias=bias).double(), nn.ReLU())
        if key == "linear":
            return nn.Linear(in_features, 1, bias=bias).double()
        raise ValueError(f"Unknown activation string: {activation!r}")
    # pre-built module passed directly
    return activation
