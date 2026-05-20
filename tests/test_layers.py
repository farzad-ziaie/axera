"""Tests for axera.layers."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from axera.layers import Dense, GMDH, InputLayer, RegressionHead, ClassificationHead


class TestDense:
    def test_output_shape(self):
        layer = Dense(out_features=8, in_features=4)
        x = torch.randn(32, 4)
        y = layer(x)
        assert y.shape == (32, 8)

    def test_parameters_exist(self):
        layer = Dense(out_features=4, in_features=3)
        params = list(layer.parameters())
        assert len(params) > 0

    def test_gradient(self):
        layer = Dense(out_features=2, in_features=3)
        x = torch.randn(8, 3)
        y = layer(x).sum()
        y.backward()
        for p in layer.parameters():
            assert p.grad is not None


class TestGMDH:
    def test_output_width_is_combinations(self):
        # C(5, 2) = 10
        layer = GMDH(in_features=5, k=2)
        x = torch.randn(16, 5)
        y = layer(x)
        assert y.shape == (16, 10)

    def test_k3_combinations(self):
        # C(5, 3) = 10 as well
        layer = GMDH(in_features=5, k=3)
        x = torch.randn(8, 5)
        y = layer(x)
        assert y.shape == (8, 10)

    def test_invalid_k_too_large(self):
        with pytest.raises(ValueError):
            GMDH(in_features=3, k=5)

    def test_gradients(self):
        layer = GMDH(in_features=4, k=2)
        x = torch.randn(10, 4, requires_grad=True)
        y = layer(x).sum()
        y.backward()
        assert x.grad is not None


class TestInputLayer:
    def test_standard_normalization(self):
        layer = InputLayer(in_features=3, normalize="standard")
        X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = layer(X)
        # After normalization, check approximately zero mean for batch
        assert out.shape == (2, 3)

    def test_minmax_normalization(self):
        layer = InputLayer(in_features=2, normalize="minmax")
        X = torch.tensor([[0.0, 10.0], [1.0, 20.0]])
        out = layer(X)
        assert out.shape == (2, 2)

    def test_no_normalization(self):
        layer = InputLayer(in_features=3, normalize="none")
        X = torch.randn(5, 3)
        out = layer(X)
        assert torch.allclose(out, X)


class TestOutputHeads:
    def test_regression_head(self):
        head = RegressionHead(in_features=8, out_features=1)
        x = torch.randn(16, 8)
        y = head(x)
        assert y.shape == (16,)   # squeezed

    def test_classification_binary(self):
        head = ClassificationHead(in_features=8, n_classes=2)
        x = torch.randn(16, 8)
        y = head(x)
        assert y.shape == (16,)   # single logit, squeezed

    def test_classification_multiclass(self):
        head = ClassificationHead(in_features=8, n_classes=5)
        x = torch.randn(16, 8)
        y = head(x)
        assert y.shape == (16, 5)
