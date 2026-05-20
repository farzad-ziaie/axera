"""
Discrimination metrics for binary classifiers in clinical research.

Includes
--------
- ROC-AUC with DeLong confidence interval
- Sensitivity, specificity, PPV, NPV, LR+, LR−
- Youden-optimal operating point
- Net Reclassification Improvement (NRI)
- Integrated Discrimination Improvement (IDI)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

# ── DeLong AUC CI ─────────────────────────────────────────────────────────────

def _delong_var(y_true: NDArray, y_score: NDArray) -> float:
    """
    Variance of the AUC estimator using DeLong (1988).
    Returns variance, not SE.
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return 0.0

    # Placement values
    V10 = np.array([np.mean(p > neg) + 0.5 * np.mean(p == neg) for p in pos])
    V01 = np.array([np.mean(n < pos) + 0.5 * np.mean(n == pos) for n in neg])

    var = (
        np.var(V10, ddof=1) / n_pos
        + np.var(V01, ddof=1) / n_neg
    )
    return float(var)


@dataclass(frozen=True)
class AUCResult:
    auc: float
    lower: float       # DeLong CI lower
    upper: float       # DeLong CI upper
    se: float
    p: float           # H0: AUC = 0.5
    n_pos: int
    n_neg: int


def roc_auc(
    y_true: NDArray,
    y_score: NDArray,
    ci: float = 0.95,
) -> AUCResult:
    """
    ROC-AUC with DeLong confidence interval.

    Parameters
    ----------
    y_true  : binary array (0/1)
    y_score : predicted probability or score
    ci      : confidence level (default 0.95)

    Returns
    -------
    AUCResult
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    auc = float(roc_auc_score(y_true, y_score))
    var = _delong_var(y_true, y_score)
    se = float(np.sqrt(var))
    z = stats.norm.ppf((1 + ci) / 2)
    lo = float(np.clip(auc - z * se, 0, 1))
    hi = float(np.clip(auc + z * se, 0, 1))
    # H0: AUC = 0.5 (i.e. z-test)
    z_stat = (auc - 0.5) / se if se > 0 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z_stat))))
    return AUCResult(
        auc=auc, lower=lo, upper=hi, se=se, p=p,
        n_pos=int(y_true.sum()), n_neg=int((1 - y_true).sum()),
    )


# ── Operating-point metrics ────────────────────────────────────────────────────

@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    sensitivity: float     # recall / TPR
    specificity: float     # TNR
    ppv: float             # precision / positive predictive value
    npv: float             # negative predictive value
    lr_pos: float          # LR+
    lr_neg: float          # LR−
    f1: float
    balanced_accuracy: float
    youden: float          # Youden J = sens + spec - 1


def operating_point(
    y_true: NDArray,
    y_score: NDArray,
    threshold: float | None = None,
    strategy: str = "youden",
) -> OperatingPoint:
    """
    Compute clinical classification metrics at a threshold.

    Parameters
    ----------
    y_true    : binary labels
    y_score   : predicted scores / probabilities
    threshold : fixed cut-point.  If None, selected by ``strategy``.
    strategy  : ``'youden'`` (maximise J), ``'sens90'`` (specificity at
                sensitivity ≥ 0.90), ``'spec90'`` (sensitivity at spec ≥ 0.90).

    Returns
    -------
    OperatingPoint
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    tnr = 1 - fpr

    if threshold is None:
        if strategy == "youden":
            j = tpr + tnr - 1
            idx = int(np.argmax(j))
        elif strategy == "sens90":
            valid = np.where(tpr >= 0.90)[0]
            idx = int(valid[np.argmax(tnr[valid])]) if len(valid) else 0
        elif strategy == "spec90":
            valid = np.where(tnr >= 0.90)[0]
            idx = int(valid[np.argmax(tpr[valid])]) if len(valid) else 0
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        threshold = float(thresholds[idx])

    y_pred = (y_score >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    lr_pos = sens / (1 - spec) if (1 - spec) > 0 else float("inf")
    lr_neg = (1 - sens) / spec if spec > 0 else float("inf")
    f1   = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    ba   = (sens + spec) / 2
    j    = sens + spec - 1

    return OperatingPoint(
        threshold=threshold, sensitivity=sens, specificity=spec,
        ppv=ppv, npv=npv, lr_pos=lr_pos, lr_neg=lr_neg,
        f1=f1, balanced_accuracy=ba, youden=j,
    )


# ── NRI & IDI ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ReclassificationResult:
    nri: float
    nri_events: float
    nri_non_events: float
    idi: float
    idi_events: float
    idi_non_events: float
    p_nri: float
    p_idi: float


def reclassification(
    y_true: NDArray,
    p_old: NDArray,
    p_new: NDArray,
    n_boot: int = 2000,
    seed: int = 42,
) -> ReclassificationResult:
    """
    Net Reclassification Improvement (NRI) and Integrated Discrimination
    Improvement (IDI) — Pencina et al. (2008).

    Parameters
    ----------
    y_true : binary array
    p_old  : predicted probabilities from old model
    p_new  : predicted probabilities from new model
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    p_old  = np.asarray(p_old, dtype=float).ravel()
    p_new  = np.asarray(p_new, dtype=float).ravel()

    events = y_true == 1
    non_ev = ~events

    # IDI
    idi_ev  = float(p_new[events].mean() - p_old[events].mean())
    idi_nev = float(p_old[non_ev].mean() - p_new[non_ev].mean())
    idi = idi_ev + idi_nev

    # Continuous NRI
    def _nri_ev(po: NDArray, pn: NDArray, ev: NDArray) -> float:
        up = (pn[ev] > po[ev]).mean()
        dn = (pn[ev] < po[ev]).mean()
        return float(up - dn)

    def _nri_nev(po: NDArray, pn: NDArray, ev: NDArray) -> float:
        non = ~ev
        up = (pn[non] > po[non]).mean()
        dn = (pn[non] < po[non]).mean()
        return float(dn - up)

    nri_ev  = _nri_ev(p_old, p_new, events)
    nri_nev = _nri_nev(p_old, p_new, events)
    nri = nri_ev + nri_nev

    # Bootstrap p-values
    rng = np.random.default_rng(seed)
    boot_nri, boot_idi = [], []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        _, po, pn, ev = y_true[idx], p_old[idx], p_new[idx], events[idx]
        b_ev  = _nri_ev(po, pn, ev)
        b_nev = _nri_nev(po, pn, ev)
        boot_nri.append(b_ev + b_nev)
        boot_idi.append(
            (pn[ev].mean() - po[ev].mean()) +
            (po[~ev].mean() - pn[~ev].mean()) if ev.any() and (~ev).any() else 0.0
        )

    boot_nri_arr = np.array(boot_nri)
    boot_idi_arr = np.array(boot_idi)
    p_nri = float(np.mean(boot_nri_arr <= 0)) if nri > 0 else float(np.mean(boot_nri_arr >= 0))
    p_idi = float(np.mean(boot_idi_arr <= 0)) if idi > 0 else float(np.mean(boot_idi_arr >= 0))

    return ReclassificationResult(
        nri=nri, nri_events=nri_ev, nri_non_events=nri_nev,
        idi=idi, idi_events=idi_ev, idi_non_events=idi_nev,
        p_nri=p_nri, p_idi=p_idi,
    )


__all__ = [
    "roc_auc", "AUCResult",
    "operating_point", "OperatingPoint",
    "reclassification", "ReclassificationResult",
]
