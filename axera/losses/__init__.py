"""Axera loss functions."""

from axera.losses.regression import LogCosh, MAE, MSE, HuberLoss
from axera.losses.multiobjective import BlandAltmanLoss, MultiObjectiveLoss

__all__ = [
    "MSE", "MAE", "LogCosh", "HuberLoss",
    "BlandAltmanLoss", "MultiObjectiveLoss",
]
