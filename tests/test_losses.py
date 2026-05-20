"""Tests for axera.losses."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from axera.losses import MSE, MAE, LogCosh, HuberLoss, BlandAltmanLoss


class TestRegressionLosses:
    @pytest.mark.parametrize("cls", [MSE, MAE, LogCosh])
    def test_perfect_prediction_is_zero(self, cls):
        loss = cls()
        y = torch.randn(16)
        assert loss(y, y).item() == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.parametrize("cls", [MSE, MAE, LogCosh])
    def test_positive_loss(self, cls):
        loss = cls()
        y = torch.zeros(16)
        p = torch.ones(16)
        assert loss(p, y).item() > 0

    def test_mse_known_value(self):
        mse = MSE()
        y = torch.zeros(4)
        p = torch.tensor([1.0, 2.0, 3.0, 4.0])
        # mean of (1,4,9,16) = 7.5
        assert mse(p, y).item() == pytest.approx(7.5, rel=1e-5)

    def test_huber_less_than_mse_on_outlier(self):
        mse = MSE()
        hub = HuberLoss(delta=1.0)
        y = torch.zeros(5)
        p = torch.tensor([0.1, 0.2, 0.3, 0.4, 100.0])  # outlier
        assert hub(p, y).item() < mse(p, y).item()

    def test_gradient_flows_through_logcosh(self):
        lc = LogCosh()
        y  = torch.zeros(8)
        p  = torch.randn(8, requires_grad=True)
        loss = lc(p, y)
        loss.backward()
        assert p.grad is not None


class TestBlandAltmanLoss:
    @pytest.fixture
    def simple_predict(self, small_regression_data):
        X, y = small_regression_data
        # trivial predict: return constant
        def _predict(weights, X_data):
            return np.full(len(X_data), weights[0])
        return _predict, X, y

    def test_evaluate_returns_array(self, simple_predict, small_regression_data):
        X, y = small_regression_data
        predict_fn, X, y = simple_predict

        ba = BlandAltmanLoss(
            predict_fn=predict_fn,
            X_train=X,
            y_train=y,
            objectives=["r2", "angle"],
        )
        out = ba.evaluate(np.array([y.mean()]))
        assert out.shape == (2,)
        assert np.isfinite(out).all()

    def test_n_objectives(self, small_regression_data):
        X, y = small_regression_data
        ba = BlandAltmanLoss(
            predict_fn=lambda w, X: np.zeros(len(X)),
            X_train=X, y_train=y,
            objectives=["r2", "icc", "mse"],
        )
        assert ba.n_objectives == 3

    def test_invalid_objective(self, small_regression_data):
        X, y = small_regression_data
        with pytest.raises(ValueError):
            BlandAltmanLoss(
                predict_fn=lambda w, X: np.zeros(len(X)),
                X_train=X, y_train=y,
                objectives=["nonexistent"],
            )
