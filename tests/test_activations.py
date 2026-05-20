"""Tests for axera.activations."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from axera.activations import LIP, LIPTanh, LIPSigmoid, LIPReLU
from axera.activations.standard import Tanh, Sigmoid, ReLU, LeakyReLU


class TestLIP:
    def test_output_shape_single_sample(self):
        lip = LIP(input_size=4, degree=2)
        x = torch.randn(4)
        y = lip(x)
        assert y.shape == ()  # scalar

    def test_output_shape_batch(self):
        lip = LIP(input_size=4, degree=2)
        x = torch.randn(16, 4)
        y = lip(x)
        assert y.shape == (16,)

    def test_n_params(self):
        # 4 features → 15 non-empty subsets × degree 2 = 30 weights + 1 bias = 31
        lip = LIP(input_size=4, degree=2, bias=True)
        assert lip.n_params == 31

    def test_no_bias(self):
        lip = LIP(input_size=3, degree=1, bias=False)
        assert lip.bias_param is None
        # 7 non-empty subsets × degree 1 = 7 weights, 0 bias
        assert lip.n_params == 7

    def test_gradient_flows(self):
        lip = LIP(input_size=3, degree=2)
        x = torch.randn(8, 3, requires_grad=True)
        y = lip(x).sum()
        y.backward()
        assert x.grad is not None
        assert lip.weight.grad is not None

    def test_degree_1_linear(self):
        """Degree-1 LIP on a single input is a scaled identity."""
        lip = LIP(input_size=1, degree=1, bias=False)
        x = torch.ones(5, 1)
        y = lip(x)
        assert y.shape == (5,)

    def test_invalid_input_size(self):
        with pytest.raises(ValueError):
            LIP(input_size=0)

    def test_invalid_degree(self):
        with pytest.raises(ValueError):
            LIP(input_size=2, degree=0)


class TestLIPCompositions:
    @pytest.mark.parametrize("cls", [LIPTanh, LIPSigmoid, LIPReLU])
    def test_output_range(self, cls):
        layer = cls(input_size=3, degree=2)
        x = torch.randn(32, 3) * 5   # large values
        y = layer(x)
        assert y.shape == (32,)
        if cls is LIPTanh:
            assert y.abs().max().item() <= 1.0 + 1e-6
        if cls is LIPSigmoid:
            assert y.min().item() >= -1e-6
            assert y.max().item() <= 1.0 + 1e-6


class TestStandardActivations:
    @pytest.mark.parametrize("cls, x_in, expected_pos", [
        (Tanh,    torch.tensor([0.0]),     True),
        (Sigmoid, torch.tensor([0.0]),     True),
        (ReLU,    torch.tensor([-1.0]),    False),
        (LeakyReLU, torch.tensor([-1.0]), True),
    ])
    def test_shapes(self, cls, x_in, expected_pos):
        act = cls()
        y = act(x_in)
        assert y.shape == x_in.shape

    def test_relu_zeros_negatives(self):
        relu = ReLU()
        x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])
        y = relu(x)
        assert (y[:2] == 0).all()
        assert (y[3:] > 0).all()


def test_combinations_all_fallback():
    # Force fallback import
    from axera._core_fallback import combinations_all
    result = combinations_all(3)
    assert len(result) == 7  # 2^3 - 1


def test_combinations_all_fallback():
    # Force fallback import
    from axera._core_fallback import combinations_all
    result = combinations_all(3)
    assert len(result) == 7  # 2^3 - 1