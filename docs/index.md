# Axera

**Polynomial neural networks and multi-objective optimisation for underdetermined biomedical datasets.**

[![CI](https://github.com/farzad-ziaie/axera/actions/workflows/ci.yml/badge.svg)](https://github.com/farzad-ziaie/axera/actions/workflows/ci.yml)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)

---

## What is Axera?

Axera is a research-grade Python package implementing **LIP (Locally Independent Polynomial)** neural networks and **GMDH (Group Method of Data Handling)** architectures for situations common in biomedicine:

- **Small n, large p** — dozens of observations, tens of features
- **Method-comparison studies** — do two measurement devices agree?
- **Clinical prediction models** — discrimination, calibration, and reporting
- **Multi-objective fitting** — optimise Bland-Altman + ICC + R² simultaneously

Axera runs on **PyTorch** (GPU/MPS/CPU) with a **NumPy-only fallback** for environments without a GPU or where PyTorch cannot be installed.  The core combinatorial routines are accelerated by a **Rust extension** (PyO3/maturin).

---

## Installation

```bash
# CPU (recommended for most users)
pip install axera

# GPU (CUDA 12)
pip install axera[gpu]

# No-GPU (NumPy-only backend)
AXERA_BACKEND=numpy pip install axera
```

---

## 30-second example

```python
import numpy as np
from axera import Sequential, Trainer, TrainerConfig
from axera.layers import InputLayer, GMDH, Dense, RegressionHead

# Data — classic underdetermined biomedical setting
X = np.random.randn(50, 6)
y = X[:, 0] ** 2 + 0.5 * X[:, 1] - X[:, 2] + np.random.normal(0, 0.1, 50)

# Model: GMDH layer → Dense LIP layer → regression head
model = Sequential([
    InputLayer(in_features=6),
    GMDH(in_features=6, k=2),          # C(6,2)=15 neurons, 2-input polynomial each
    Dense(out_features=4, in_features=15),
    RegressionHead(in_features=4),
])

# Train
trainer = Trainer(model, TrainerConfig(epochs=200, optimizer="adamw"))
history = trainer.fit(X, y)

# Predict
preds = model.predict(X)

# Medical metrics
from axera.medical import bland_altman, icc
ba = bland_altman(preds, y)
print(f"Bias: {ba.bias:.3f}  LoA: [{ba.loa_lower:.3f}, {ba.loa_upper:.3f}]")
```

---

## Key features

| Feature | Description |
|---|---|
| **LIP activation** | All non-empty polynomial combinations of input signals |
| **GMDH layer** | k-wise neuron combinations, width = C(n, k) |
| **MOPSO optimizer** | Vectorised multi-objective particle swarm, Rust Pareto |
| **Bland-Altman loss** | 7-objective multi-criteria fitness for MOPSO |
| **ICC (all 6 cases)** | McGraw & Wong fixed — no XOR bugs |
| **DeLong AUC CI** | Variance-based confidence intervals |
| **Hosmer-Lemeshow** | Calibration goodness-of-fit test |
| **Async API** | `await model.apredict(X)`, `await trainer.afit(X, y)` |
| **Plugin hooks** | 8 hook slots: pre/post fit, predict, per-epoch |
| **CLI** | `axera train`, `axera infer`, `axera benchmark` |
| **OTel tracing** | Optional OpenTelemetry instrumentation |

---

## Citation

If you use Axera in research, please cite:

```bibtex
@software{ziaie_nezhad_axera_2025,
  author  = {Ziaie Nezhad, Farzad},
  title   = {Axera: Polynomial Neural Networks for Biomedical Datasets},
  year    = {2025},
  url     = {https://github.com/farzad-ziaie/axera},
  license = {AGPL-3.0},
}
```
