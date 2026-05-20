"""
Inference benchmarks — run with:
  pytest tests/benchmarks/bench_inference.py --benchmark-only -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from axera.layers import Dense, GMDH, InputLayer, RegressionHead
from axera.models import Sequential


@pytest.fixture(scope="module")
def model():
    m = Sequential([
        InputLayer(in_features=8),
        GMDH(in_features=8, k=2),       # C(8,2)=28 neurons
        Dense(out_features=8, in_features=28),
        RegressionHead(in_features=8),
    ])
    m.eval()
    return m


@pytest.fixture(scope="module")
def X_small():
    return np.random.randn(64, 8).astype(np.float64)


@pytest.fixture(scope="module")
def X_large():
    return np.random.randn(2048, 8).astype(np.float64)


def test_bench_inference_64(benchmark, model, X_small):
    benchmark(model.predict, X_small)


def test_bench_inference_2048(benchmark, model, X_large):
    benchmark(model.predict, X_large)


def test_bench_inference_batched(benchmark, model, X_large):
    benchmark(model.predict, X_large, batch_size=128)
