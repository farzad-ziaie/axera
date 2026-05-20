"""
Multi-objective loss functions for derivative-free (swarm) optimizers.

These losses are NOT differentiable in the autograd sense.  They are designed
for population-based optimizers (MOPSO, Differential Evolution) that query the
loss as a black box.

The flagship is ``BlandAltmanLoss``, which returns a *vector* of clinical
agreement objectives — each one corresponds to a Bland-Altman quality criterion
for the current weight vector.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

from axera.medical.icc import icc


# ── Protocol ──────────────────────────────────────────────────────────────────

class MultiObjectiveLoss:
    """
    Base class for multi-objective losses used by population-based optimisers.

    Subclasses must implement ``evaluate(weights) -> NDArray[float]`` where the
    returned array length equals the number of objectives.
    """

    def evaluate(self, weights: NDArray) -> NDArray:
        raise NotImplementedError

    @property
    def n_objectives(self) -> int:
        raise NotImplementedError


# ── Bland-Altman multi-objective loss ─────────────────────────────────────────

class BlandAltmanLoss(MultiObjectiveLoss):
    """
    Multi-objective Bland-Altman loss for method-comparison studies.

    Each call to ``evaluate(weights)`` runs the model on all training samples,
    then returns a vector of objective values to *minimise*.

    Available objectives (controlled by ``objectives`` parameter)
    -------------------------------------------------------------
    ``'r2'``           1 − R² (lower is better agreement)
    ``'angle'``        Proportional-bias angle (0 = no bias)
    ``'icc'``          1 − ICC(C-1) (0 = perfect consistency)
    ``'iqr'``          IQR of |differences|
    ``'mse'``          Mean squared error
    ``'rmse'``         Root MSE
    ``'paired_t'``     p-value from paired t-test (1 − p, so 0 = no sig. diff)

    Parameters
    ----------
    predict_fn : callable
        Function ``(weights, X) -> NDArray`` that returns predictions given a
        weight vector.
    X_train : NDArray  shape (n, p)
    y_train : NDArray  shape (n,)
    objectives : sequence of str
        Which objectives to compute and return.  Order defines the Pareto front.
    """

    def __init__(
        self,
        predict_fn: Callable[[NDArray, NDArray], NDArray],
        X_train: NDArray,
        y_train: NDArray,
        objectives: Sequence[str] = ("r2", "angle"),
    ) -> None:
        self._predict = predict_fn
        self._X = np.asarray(X_train, dtype=float)
        self._Y = np.asarray(y_train, dtype=float)
        self._objectives = list(objectives)
        self._validate_objectives()

    def _validate_objectives(self) -> None:
        valid = {"r2", "angle", "icc", "iqr", "mse", "rmse", "paired_t"}
        bad = set(self._objectives) - valid
        if bad:
            raise ValueError(f"Unknown objectives: {bad}. Valid: {valid}")

    # ── core objectives ───────────────────────────────────────────────────────

    def _get_predictions(self, weights: NDArray) -> NDArray:
        return np.asarray(self._predict(weights, self._X), dtype=float)

    def _r2(self, pred: NDArray) -> float:
        return float(1 - r2_score(y_pred=pred, y_true=self._Y))

    def _angle(self, pred: NDArray) -> float:
        diff = pred - self._Y
        mean_ = (pred + self._Y) / 2.0
        try:
            def line(x: NDArray, a: float, b: float) -> NDArray:
                return a * x + b
            popt, _ = curve_fit(line, self._Y, diff)
            return float(np.abs(2 * np.arctan(popt[0]) / np.pi))
        except Exception:
            return 0.0

    def _icc_loss(self, pred: NDArray) -> float:
        M = np.column_stack([pred, self._Y])
        try:
            result = icc(M, icc_type="C-1")
            return float(1 - result["r"])
        except Exception:
            return 1.0

    def _iqr(self, pred: NDArray) -> float:
        from scipy.stats import iqr
        return float(iqr(np.abs(pred - self._Y)))

    def _mse(self, pred: NDArray) -> float:
        return float(np.mean((pred - self._Y) ** 2))

    def _rmse(self, pred: NDArray) -> float:
        return float(np.sqrt(self._mse(pred)))

    def _paired_t(self, pred: NDArray) -> float:
        from scipy import stats
        _, pval = stats.ttest_rel(pred, self._Y)
        # Minimise: return 1 − p so that significant difference = high cost
        return float(1 - pval)

    # ── public API ────────────────────────────────────────────────────────────

    def evaluate(self, weights: NDArray) -> NDArray:
        """
        Evaluate all objectives for a weight vector.

        Parameters
        ----------
        weights : NDArray  (n_weights,)

        Returns
        -------
        NDArray  (n_objectives,)  — all values to *minimise*
        """
        pred = self._get_predictions(weights)
        _map = {
            "r2":       self._r2,
            "angle":    self._angle,
            "icc":      self._icc_loss,
            "iqr":      self._iqr,
            "mse":      self._mse,
            "rmse":     self._rmse,
            "paired_t": self._paired_t,
        }
        return np.array([_map[obj](pred) for obj in self._objectives])

    def evaluate_single(self, weights: NDArray) -> float:
        """Return the sum of objectives (for single-objective DE/basinhopping)."""
        return float(self.evaluate(weights).sum())

    @property
    def n_objectives(self) -> int:
        return len(self._objectives)


__all__ = ["MultiObjectiveLoss", "BlandAltmanLoss"]
