# Axera 🧬

> Polynomial neural networks and multi-objective optimisation for underdetermined biomedical datasets.

[![CI](https://github.com/farzad-ziaie/axera/actions/workflows/ci.yml/badge.svg)](https://github.com/farzad-ziaie/axera/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg)](https://pytorch.org/)
[![CUDA 12](https://img.shields.io/badge/CUDA-12-76B900.svg)](https://developer.nvidia.com/cuda-downloads)

---

This a refactored version of the original code i wrote years ago using Theano and its simbolic lib.

## Overview

Biomedical research routinely faces the *n << p* problem: small patient cohorts,
many clinical predictors, and strict reporting requirements.  Standard deep learning
architectures are poorly suited to this regime.

**Axera** implements:

| Component | Description |
|---|---|
| **LIP activation** | Locally Independent Polynomial — all non-empty polynomial combinations of input signals |
| **GMDH layer** | Group Method of Data Handling — k-wise neuron combinations (width = C(n, k)) |
| **MOPSO optimizer** | Vectorised multi-objective particle swarm, Pareto-front archive, Rust extension |
| **Bland-Altman loss** | 7-objective multi-criteria fitness function for method-comparison studies |
| **Medical metrics** | ICC (6 cases, bug-fixed), Bland-Altman, CCC, AUC (DeLong CI), Hosmer-Lemeshow |
| **Async API** | `await model.apredict(X)`, `await trainer.afit(X, y)` |
| **Plugin hooks** | 8 hook slots for pre/post processing |
| **CLI** | `axera train`, `axera infer`, `axera benchmark`, `axera info` |
| **OTel tracing** | Optional OpenTelemetry instrumentation for production deployments |

---

## Installation

```bash
# Standard (auto-selects GPU if available)
pip install axera

# GPU (CUDA 12)
pip install axera[gpu]

# NumPy-only (no PyTorch required)
AXERA_BACKEND=numpy pip install axera

# All extras (dev + docs + server + gpu)
pip install "axera[gpu,server,dev,docs]"
```

> **Rust extension**: `pip install axera` downloads a pre-compiled wheel.
> To build from source: `pip install maturin && maturin develop --release`

---

## Quick example

```python
import numpy as np
from axera import Sequential, Trainer, TrainerConfig
from axera.layers import InputLayer, GMDH, Dense, RegressionHead
from axera.medical import bland_altman, icc

# --- Data (n=60, p=6 — classic underdetermined biomedical setting) ---
rng = np.random.default_rng(42)
X   = rng.standard_normal((60, 6))
y   = 2.5*X[:,0] - X[:,1]**2 + 0.5*X[:,2]*X[:,3] + rng.normal(0, 0.2, 60)

# --- Model ---
model = Sequential([
    InputLayer(in_features=6),
    GMDH(in_features=6, k=2),              # C(6,2)=15 neurons, 2-input polynomial each
    Dense(out_features=6, in_features=15), # 6 LIP neurons on all 15 inputs
    RegressionHead(in_features=6),
])
model.summary()

# --- Train (Adam + log-cosh loss, auto early stopping) ---
trainer = Trainer(model, TrainerConfig(epochs=300, optimizer="adamw", loss="logcosh"))
trainer.fit(X[:45], y[:45])

# --- Evaluate agreement (Bland-Altman) ---
preds = model.predict(X[45:])
ba    = bland_altman(preds, y[45:])
icc_r = icc(np.column_stack([preds, y[45:]]), icc_type="C-1")

print(f"Bias:  {ba.bias:+.3f}  [{ba.bias_lower:+.3f}, {ba.bias_upper:+.3f}]")
print(f"LoA:   [{ba.loa_lower:.3f}, {ba.loa_upper:.3f}]")
print(f"ICC:   {icc_r['r']:.3f}")
```

---

## Architecture

```
InputLayer (standardisation)
    ↓
GMDH layer  →  C(p, k) neurons, each a polynomial of k inputs
    ↓
Dense layer →  out_features neurons, each a full LIP polynomial
    ↓
RegressionHead / ClassificationHead
```

### LIP activation

```
y = b + Σ_{S ⊆ {1…n}, S≠∅} Σ_{k=1}^{d} θ_{S,k} · (∏_{i∈S} xᵢ)^k
```

For `n=4, d=2`: 31 parameters per neuron (vs 5 for a linear unit).

---

## Medical metrics

All metrics return dataclasses with bootstrap confidence intervals (BCa):

```python
from axera.medical import (
    bland_altman,          # bias, LoA, proportional-bias test
    concordance_correlation,  # Lin's CCC
    cohen_kappa,           # weighted / unweighted κ
    roc_auc,               # DeLong variance CI
    operating_point,       # sens, spec, PPV, NPV, LR+, LR−
    reclassification,      # NRI + IDI (Pencina 2008)
    brier_score,           # with skill score
    calibration_error,     # ECE + MCE
    hosmer_lemeshow,       # H-L goodness-of-fit
    icc,                   # all 6 McGraw-Wong cases (bugs fixed)
)
```

### ICC bug fix

The original codebase used Python's `^` (bitwise XOR) where `**` (exponentiation) was intended — a silent bug producing completely wrong values for ICC cases A-1 and A-k.  All six cases are now verified against McGraw & Wong (1996) Table 1.

---

## CLI

```bash
axera info                                # environment & version
axera train --config cfg.json --data X.csv --target y.csv
axera infer --model model.pt --data X.csv --out preds.csv
axera benchmark --model model.pt --n-samples 5000
axera export --model model.pt --format onnx --n-features 6 --out model.onnx
```

---

## Development

```bash
git clone https://github.com/farzad-ziaie/axera.git
cd axera

# Install with dev extras + Rust extension
pip install maturin
maturin develop --release
pip install -e ".[dev]"

# Pre-commit hooks
pre-commit install

# Tests
pytest tests/ -x -v --cov=axera

# Type check
mypy axera --strict
```

---

## Bugs fixed from original codebase

| File | Bug | Fix |
|---|---|---|
| `stats.py` (ICC A-1) | `^` used instead of `**` (XOR not power) | `**` |
| `stats.py` (ICC A-k) | Same XOR bug | `**` |
| `optimizers.py` (Adam) | `V = β₂² + (1−β₂)g²` | `V = β₂·V + (1−β₂)g²` |
| `optimizers.py` (Adam bias) | `1 − β**2` in correction | `1 − β**t` |
| `optimizers.py` (SGD/RMSprop) | `super().__init__()` never called | Fixed |
| `optimizers.py` (basinhopping) | `x0=bounds` (wrong type) | `x0=initial_vector` |
| `util.py` (Pareto) | `_dominates` reversed variable names | Fixed |
| `util.py` (imputation) | Drop-then-impute ordering bug | Impute-then-drop |
| `model.py` (compile) | Throwaway layer instantiated for typecheck | Removed |

---

## Citation

```bibtex
@software{ziaie_nezhad_axera_2025,
  author  = {Ziaie Nezhad, Farzad},
  title   = {{Axera}: Polynomial Neural Networks for Biomedical Datasets},
  year    = {2025},
  version = {0.1.0},
  url     = {https://github.com/farzad-ziaie/axera},
  license = {AGPL-3.0-or-later},
}
```

---

## License

AGPLv3 — see [LICENSE](LICENSE).  Commercial licensing available on request.

---

*Maintained by [Farzad Ziaie Nezhad](mailto:farzadziaien@gmail.com)*
