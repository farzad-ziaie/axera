# Architecture

## LIP activation

**LIP (Locally Independent Polynomial)** is the core building block.
Given `n` input signals `x₁…xₙ` and degree `d`, a single LIP neuron computes:

```
y = b + Σ_{S ⊆ {1…n}, S≠∅} Σ_{k=1}^{d} θ_{S,k} · (∏_{i∈S} xᵢ)^k
```

This gives `|non-empty subsets| × d = (2ⁿ − 1) × d` learnable weights plus one bias.
For `n=4, d=2`: 15 × 2 + 1 = **31 parameters per neuron**.

The Rust extension (`axera._core.combinations_all`) generates all subset indices in
sorted order, so the polynomial is fully reproducible across runs.

## GMDH layer

The GMDH layer restricts each neuron to exactly `k` inputs chosen from all `C(in, k)`
combinations of the previous layer's output:

```
width_out = C(in_features, k)
```

Classic GMDH uses `k=2` (pairwise); Axera supports arbitrary `k`.

## Sequential model

```
InputLayer  →  [GMDH | Dense]+  →  [RegressionHead | ClassificationHead]
```

Layers are stored in `nn.ModuleList` and executed sequentially in `forward()`.

## Backend dispatch

```python
AXERA_BACKEND = "auto"   # torch (if available) else numpy
AXERA_BACKEND = "torch"  # force torch
AXERA_BACKEND = "numpy"  # force numpy-only
```

The `axera._backends` module selects the backend at import time.

## Rust extension

`axera_core` (PyO3 crate) provides:

| Function | Purpose |
|---|---|
| `combinations_all(n)` | All 2ⁿ−1 non-empty subsets |
| `combinations_k(n, k)` | All C(n,k) k-combinations |
| `pareto_fast(costs)` | Non-dominated front (parallel via Rayon) |
| `grid_find_index(cost, ubs, n_grid)` | MOPSO grid cell index |

If the Rust extension is unavailable, `axera._core_fallback` provides identical
pure-Python implementations automatically.
