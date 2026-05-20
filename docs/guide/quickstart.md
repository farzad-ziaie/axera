# Quick Start

This guide gets you from zero to a trained Bland-Altman-optimised model in minutes.

## Installation

```bash
pip install axera               # CPU / GPU auto-detect
pip install axera[gpu]          # explicit GPU support (CUDA 12)
AXERA_BACKEND=numpy pip install axera   # pure NumPy, no PyTorch
```

## Basic regression

```python
import numpy as np
from axera import Sequential, Trainer, TrainerConfig
from axera.layers import InputLayer, GMDH, Dense, RegressionHead

X = np.random.randn(60, 4)
y = 2 * X[:, 0] - X[:, 1] ** 2 + np.random.normal(0, 0.1, 60)

model = Sequential([
    InputLayer(in_features=4),
    GMDH(in_features=4, k=2),         # 6 neurons
    Dense(out_features=4, in_features=6),
    RegressionHead(in_features=4),
])

trainer = Trainer(model, TrainerConfig(epochs=100))
history = trainer.fit(X, y)
preds = model.predict(X)
```

## Bland-Altman method comparison

```python
from axera.medical import bland_altman, concordance_correlation, icc

ba  = bland_altman(preds, y)
ccc = concordance_correlation(preds, y)
icc_result = icc(np.column_stack([preds, y]), icc_type="C-1")

print(f"Bias:  {ba.bias:+.3f}  [{ba.bias_lower:+.3f}, {ba.bias_upper:+.3f}]")
print(f"LoA:   [{ba.loa_lower:.3f}, {ba.loa_upper:.3f}]")
print(f"CCC:   {ccc.ccc:.3f}  [{ccc.lower:.3f}, {ccc.upper:.3f}]")
print(f"ICC:   {icc_result['r']:.3f}")
```

## Multi-objective MOPSO training

```python
from axera.losses import BlandAltmanLoss
from axera.config import TrainerConfig

loss = BlandAltmanLoss(
    predict_fn=model.predict_from_weights,
    X_train=X, y_train=y,
    objectives=["r2", "angle", "icc"],
)
cfg = TrainerConfig(optimizer="mopso", epochs=100)
trainer = Trainer(model, cfg, loss_fn=loss)
trainer.fit(X, y)
```

## Clinical classification

```python
from axera.layers import ClassificationHead
from axera.medical import roc_auc, operating_point, brier_score

clf_model = Sequential([
    InputLayer(in_features=5),
    GMDH(in_features=5, k=2),
    Dense(out_features=4, in_features=10),
    ClassificationHead(in_features=4, n_classes=2),
], task="binary")

trainer = Trainer(clf_model, TrainerConfig(epochs=50))
trainer.fit(X_train, y_train)

probs = clf_model.predict(X_test)
auc   = roc_auc(y_test, probs)
op    = operating_point(y_test, probs, strategy="youden")

print(f"AUC:  {auc.auc:.3f}  [{auc.lower:.3f}, {auc.upper:.3f}]")
print(f"Sens: {op.sensitivity:.3f}  Spec: {op.specificity:.3f}")
```
