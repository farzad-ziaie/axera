"""Tests for axera.models.Sequential."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from axera.config import ModelConfig
from axera.layers import GMDH, ClassificationHead, Dense, InputLayer, RegressionHead
from axera.models import Sequential


@pytest.fixture
def simple_model():
    return Sequential([
        InputLayer(in_features=4),
        GMDH(in_features=4, k=2),  # C(4,2) = 6 outputs
        Dense(out_features=4, in_features=6),
        RegressionHead(in_features=4),
    ])


class TestSequential:
    def test_forward_pass(self, simple_model, small_regression_data):
        X, y = small_regression_data
        X_t = torch.from_numpy(X)
        out = simple_model(X_t)
        assert out.shape == (len(X),)

    def test_predict_returns_numpy(self, simple_model, small_regression_data):
        X, y = small_regression_data
        preds = simple_model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert preds.shape == (len(X),)

    def test_from_config(self):
        cfg = ModelConfig(
            in_features=5,
            task="regression",
            layers=[
                {"type": "GMDH", "k": 2},
                {"type": "Dense", "units": 4},
            ],
        )
        model = Sequential.from_config(cfg)
        X = torch.randn(8, 5)
        out = model(X)
        assert out.shape == (8,)

    def test_summary(self, simple_model, capsys):
        summary = simple_model.summary()
        assert "total_params" in summary
        assert summary["total_params"] > 0
        captured = capsys.readouterr()
        assert "Total" in captured.out

    def test_save_load(self, simple_model, tmp_path, small_regression_data):
        X, y = small_regression_data
        preds_before = simple_model.predict(X)

        path = tmp_path / "model.pt"
        simple_model.save(path)
        assert path.exists()

        loaded = Sequential.load(path, [
            InputLayer(in_features=4),
            GMDH(in_features=4, k=2),
            Dense(out_features=4, in_features=6),
            RegressionHead(in_features=4),
        ])
        preds_after = loaded.predict(X)
        np.testing.assert_allclose(preds_before, preds_after, rtol=1e-5)

    def test_flat_weights_roundtrip(self, simple_model):
        w = simple_model._get_flat_weights()
        assert w.ndim == 1
        assert len(w) == simple_model.summary()["total_params"]
        simple_model._set_flat_weights(w)
        w2 = simple_model._get_flat_weights()
        np.testing.assert_allclose(w, w2)

    def test_classification_predict_probabilities(self):
        model = Sequential([
            InputLayer(in_features=3),
            Dense(out_features=4, in_features=3),
            ClassificationHead(in_features=4, n_classes=2),
        ], task="binary")
        X = np.random.randn(20, 3)
        preds = model.predict(X)
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0
