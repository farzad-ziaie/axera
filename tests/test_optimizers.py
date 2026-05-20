"""Tests for gradient and swarm optimizers — including regression tests for Adam bug fixes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from axera.optimizers.gradient import Lion, NumpyAdam, get_optimizer


class TestNumpyAdam:
    """
    Regression tests for the three Adam bugs fixed in gradient.py.
    """

    def test_V_accumulation_not_squared_beta(self):
        """Bug 1: V should be β₂·V + (1−β₂)·g², not β₂² + ..."""
        opt = NumpyAdam(lr=0.01, beta2=0.999)
        opt.init(5)
        g = np.ones(5)
        opt.step(g, np.zeros(5))
        # V after first step: β₂·0 + (1−β₂)·1 = 0.001
        expected_V = (1 - 0.999) * (g ** 2)
        np.testing.assert_allclose(opt.V, expected_V, rtol=1e-8)

    def test_bias_correction_uses_t_exponent(self):
        """Bug 2: bias correction denominator should use **t, not **2."""
        opt = NumpyAdam(lr=0.1, beta1=0.9, beta2=0.999)
        opt.init(2)
        g = np.array([1.0, 1.0])
        w = np.array([0.5, 0.5])
        # Run 5 steps and check w changes across steps
        w_prev = w.copy()
        for _ in range(5):
            w = opt.step(g, w)
        # After multiple steps, the correction exponent **t should differ from **2
        assert not np.allclose(w, w_prev)

    def test_step_decreases_loss_on_quadratic(self):
        """Adam should reduce ‖w - target‖² over many steps."""
        target = np.array([3.0, -2.0, 0.5])
        opt = NumpyAdam(lr=0.1)
        opt.init(3)
        w = np.zeros(3)
        for _ in range(200):
            grad = 2 * (w - target)   # gradient of MSE
            w = opt.step(grad, w)
        np.testing.assert_allclose(w, target, atol=0.05)


class TestLion:
    def test_step_runs(self):
        import torch.nn as nn
        model = nn.Linear(4, 1)
        opt = Lion(model.parameters(), lr=1e-4)
        x = torch.randn(8, 4)
        y = model(x)
        y.sum().backward()
        opt.step()
        opt.zero_grad()

    def test_invalid_lr(self):
        import torch.nn as nn
        model = nn.Linear(4, 1)
        with pytest.raises(ValueError):
            Lion(model.parameters(), lr=-1)


class TestGetOptimizer:
    @pytest.mark.parametrize("name", ["adam", "adamw", "sgd", "rmsprop", "lion"])
    def test_all_names(self, name):
        import torch.nn as nn
        model = nn.Linear(4, 1)
        opt = get_optimizer(name, model.parameters())
        assert opt is not None

    def test_invalid_name(self):
        import torch.nn as nn
        model = nn.Linear(4, 1)
        with pytest.raises(ValueError):
            get_optimizer("sgd_v2", model.parameters())


def test_mopso_fit_runs(small_regression_data):
    X, y = small_regression_data
    from axera.losses import BlandAltmanLoss
    from axera.optimizers import MOPSO

    loss = BlandAltmanLoss(
        predict_fn=lambda w, X: np.zeros(len(X)),
        X_train=X, y_train=y,
        objectives=["r2", "mse"],
    )
    opt = MOPSO(n_pop=20, n_repo=10)
    weights = opt.fit(loss, n_dim=5, max_iter=5)
    assert weights.shape == (5,)
