"""Shared pytest fixtures for Axera test suite."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def small_regression_data(rng):
    """n=60, p=4 regression dataset — simulates an underdetermined biomedical study."""
    X = rng.standard_normal((60, 4))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] ** 2 + rng.normal(0, 0.1, 60)
    return X.astype(np.float64), y.astype(np.float64)


@pytest.fixture
def small_classification_data(rng):
    """n=80, p=5 binary classification dataset."""
    X = rng.standard_normal((80, 5))
    logit = X[:, 0] + 0.5 * X[:, 1] - X[:, 2]
    y = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
    return X.astype(np.float64), y


@pytest.fixture
def method_comparison_data(rng):
    """Paired method-comparison data (e.g. two measurement devices)."""
    y_ref = rng.uniform(50, 200, 80)           # reference method
    bias  = 3.0 + 0.02 * y_ref                 # proportional bias
    noise = rng.normal(0, 5, 80)
    y_new = y_ref + bias + noise
    return y_ref.astype(np.float64), y_new.astype(np.float64)


@pytest.fixture
def binary_prediction_data(rng):
    """Binary labels with predicted probabilities."""
    y_true = rng.integers(0, 2, 100)
    y_prob = np.clip(y_true.astype(float) + rng.normal(0, 0.3, 100), 0.01, 0.99)
    return y_true, y_prob.astype(np.float64)
