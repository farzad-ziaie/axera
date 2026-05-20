"""Axera metrics — standard and medical-grade evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from axera.medical import bland_altman, concordance_correlation, icc, roc_auc


def evaluate_regression(
    y_true: NDArray,
    y_pred: NDArray,
    extended: bool = True,
) -> dict[str, Any]:
    """
    Comprehensive regression metrics including medical-grade agreement statistics.

    Parameters
    ----------
    y_true, y_pred : ndarray (n,)
    extended : bool
        If True, compute Bland-Altman, ICC, CCC, and bootstrap CIs.

    Returns
    -------
    dict  metric name → value
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    results: dict[str, Any] = {
        "mse":  float(mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae":  float(mean_absolute_error(y_true, y_pred)),
        "r2":   float(r2_score(y_true, y_pred)),
    }

    if extended:
        ba = bland_altman(y_pred, y_true)
        results["ba_bias"]          = ba.bias
        results["ba_bias_ci"]       = (ba.bias_lower, ba.bias_upper)
        results["ba_loa_upper"]     = ba.loa_upper
        results["ba_loa_lower"]     = ba.loa_lower
        results["ba_prop_bias_p"]   = ba.proportional_bias_p
        results["ba_angle"]         = ba.proportional_bias_angle

        ccc = concordance_correlation(y_pred, y_true)
        results["ccc"]              = ccc.ccc
        results["ccc_ci"]           = (ccc.lower, ccc.upper)

        M = np.column_stack([y_pred, y_true])
        icc_res = icc(M, icc_type="C-1")
        results["icc_C1"]           = icc_res["r"]
        results["icc_C1_ci"]        = (icc_res["lower"], icc_res["upper"])

    return results


def evaluate_classification(
    y_true: NDArray,
    y_score: NDArray,
) -> dict[str, Any]:
    """Standard + medical classification metrics."""
    y_true  = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    y_pred  = (y_score >= 0.5).astype(int)

    from axera.medical import brier_score, operating_point

    op  = operating_point(y_true, y_score)
    bs  = brier_score(y_true, y_score)
    auc = roc_auc(y_true, y_score)

    return {
        "auc":              auc.auc,
        "auc_ci":           (auc.lower, auc.upper),
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "sensitivity":      op.sensitivity,
        "specificity":      op.specificity,
        "ppv":              op.ppv,
        "npv":              op.npv,
        "lr_pos":           op.lr_pos,
        "lr_neg":           op.lr_neg,
        "youden":           op.youden,
        "brier_score":      bs.brier_score,
        "brier_skill":      bs.brier_skill,
        "optimal_threshold": op.threshold,
    }


__all__ = ["evaluate_regression", "evaluate_classification"]
