"""
Axera — polynomial neural networks and multi-objective optimisation
for underdetermined biomedical datasets.

Author : Farzad Ziaie Nezhad <farzadziaien@gmail.com>
License: AGPLv3
URL    : https://github.com/farzad-ziaie/axera
"""

from __future__ import annotations

from axera._version import __version__
from axera.activations import LIP, LIPReLU, LIPSigmoid, LIPTanh
from axera.config import (
    AdamConfig,
    DenseConfig,
    GMDHConfig,
    InferenceConfig,
    LIPConfig,
    MOPSOConfig,
    ModelConfig,
    TrainerConfig,
)
from axera.hooks import HookRegistry, get_global_registry, register
from axera.layers import GMDH, Dense, InputLayer
from axera.losses import BlandAltmanLoss, LogCosh, MAE, MSE, MultiObjectiveLoss
from axera.medical import (
    bland_altman,
    brier_score,
    calibration_error,
    cohen_kappa,
    concordance_correlation,
    hosmer_lemeshow,
    icc,
    operating_point,
    reclassification,
    roc_auc,
)
from axera.metrics import evaluate_classification, evaluate_regression
from axera.models import Sequential
from axera.optimizers import MOPSO, Lion, get_optimizer
from axera.trainer import Trainer

__all__ = [
    "__version__",
    # model
    "Sequential",
    # layers
    "InputLayer", "Dense", "GMDH",
    # activations
    "LIP", "LIPTanh", "LIPSigmoid", "LIPReLU",
    # losses
    "MSE", "MAE", "LogCosh", "BlandAltmanLoss", "MultiObjectiveLoss",
    # optimizers
    "get_optimizer", "Lion", "MOPSO",
    # trainer
    "Trainer",
    # configs
    "LIPConfig", "GMDHConfig", "DenseConfig", "ModelConfig",
    "MOPSOConfig", "AdamConfig", "TrainerConfig", "InferenceConfig",
    # hooks
    "HookRegistry", "register", "get_global_registry",
    # metrics
    "evaluate_regression", "evaluate_classification",
    # medical
    "bland_altman", "concordance_correlation", "cohen_kappa",
    "roc_auc", "operating_point", "reclassification",
    "brier_score", "calibration_error", "hosmer_lemeshow",
    "icc",
]
