"""
Training benchmarks — run with:
  pytest tests/benchmarks/bench_training.py --benchmark-only -v
"""

from __future__ import annotations

import numpy as np
import pytest

from axera.config import TrainerConfig
from axera.layers import Dense, GMDH, InputLayer, RegressionHead
from axera.models import Sequential
from axera.trainer import Trainer


@pytest.fixture(scope="module")
def regression_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 6))
    y = 2 * X[:, 0] - X[:, 1] ** 2 + rng.normal(0, 0.1, 80)
    return X, y


def _make_model():
    return Sequential([
        InputLayer(in_features=6),
        GMDH(in_features=6, k=2),
        Dense(out_features=4, in_features=15),
        RegressionHead(in_features=4),
    ])


def test_bench_train_10_epochs(benchmark, regression_data):
    X, y = regression_data
    def _train():
        model = _make_model()
        cfg = TrainerConfig(epochs=10, batch_size=32, optimizer="adam", loss="mse",
                            val_split=0.0, early_stopping_patience=0)
        Trainer(model, cfg).fit(X, y)
    benchmark(_train)


def test_bench_train_logcosh(benchmark, regression_data):
    X, y = regression_data
    def _train():
        model = _make_model()
        cfg = TrainerConfig(epochs=10, batch_size=32, optimizer="adamw", loss="logcosh",
                            val_split=0.0, early_stopping_patience=0)
        Trainer(model, cfg).fit(X, y)
    benchmark(_train)
