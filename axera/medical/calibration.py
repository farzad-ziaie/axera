"""
Probabilistic calibration metrics for clinical prediction models.

Good calibration means that among all patients assigned a predicted probability
of p%, approximately p% actually experience the event.  This is critical for
clinical decision-making and guideline compliance.

Includes
--------
- Brier score (overall calibration + skill)
- Expected Calibration Error (ECE) and Maximum CE (MCE)
- Hosmer-Lemeshow C-statistic and p-value
- Calibration-in-the-large (CITL) and calibration slope
- Reliability diagram data (for plotting)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import stats

# ── Brier Score ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BrierResult:
    brier_score: float
    brier_skill: float        # 1 − BS/BS_ref  (ref = prevalence-based null)
    lower: float              # bootstrap CI
    upper: float
    n: int


def brier_score(
    y_true: NDArray,
    y_prob: NDArray,
    ci: float = 0.95,
    n_boot: int = 2000,
    seed: int = 42,
) -> BrierResult:
    """Compute Brier score with bootstrap confidence interval."""
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = len(y_true)
    bs = float(np.mean((y_prob - y_true) ** 2))

    prev = y_true.mean()
    bs_ref = float(prev * (1 - prev))   # null model using prevalence
    skill = 1 - bs / bs_ref if bs_ref > 0 else 0.0

    rng = np.random.default_rng(seed)
    boot = np.array([
        np.mean((y_prob[idx := rng.integers(0, n, n)] - y_true[idx]) ** 2)
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(boot, (1 - ci) / 2 * 100))
    hi = float(np.percentile(boot, (1 + ci) / 2 * 100))

    return BrierResult(brier_score=bs, brier_skill=skill, lower=lo, upper=hi, n=n)


# ── ECE & MCE ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationBins:
    ece: float                    # expected calibration error
    mce: float                    # maximum calibration error
    bin_centers: NDArray          # midpoints of probability bins
    bin_fractions: NDArray        # observed fraction positive per bin
    bin_counts: NDArray           # number of samples per bin
    n_bins: int


def calibration_error(
    y_true: NDArray,
    y_prob: NDArray,
    n_bins: int = 10,
    strategy: str = "uniform",   # 'uniform' or 'quantile'
) -> CalibrationBins:
    """
    Expected and Maximum Calibration Error.

    Parameters
    ----------
    y_true   : binary array
    y_prob   : predicted probabilities [0, 1]
    n_bins   : number of calibration bins
    strategy : ``'uniform'`` (equal width) or ``'quantile'`` (equal size)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    n = len(y_true)

    if strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    elif strategy == "quantile":
        edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
        edges = np.unique(edges)
        n_bins = len(edges) - 1
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    bin_centers = np.zeros(n_bins)
    bin_fracs   = np.zeros(n_bins)
    bin_counts  = np.zeros(n_bins, dtype=int)
    abs_errs    = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        cnt = mask.sum()
        bin_counts[i] = cnt
        if cnt > 0:
            bin_centers[i] = y_prob[mask].mean()
            bin_fracs[i]   = y_true[mask].mean()
            abs_errs[i]    = abs(bin_fracs[i] - bin_centers[i])
        else:
            bin_centers[i] = (lo + hi) / 2

    ece = float(np.sum(bin_counts * abs_errs) / n)
    mce = float(abs_errs.max())

    return CalibrationBins(
        ece=ece, mce=mce,
        bin_centers=bin_centers, bin_fractions=bin_fracs,
        bin_counts=bin_counts, n_bins=n_bins,
    )


# ── Hosmer-Lemeshow ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class HosmerLemeshowResult:
    chi2: float
    p: float
    df: int
    g: int    # number of groups
    interpretation: str


def hosmer_lemeshow(
    y_true: NDArray,
    y_prob: NDArray,
    g: int = 10,
) -> HosmerLemeshowResult:
    """
    Hosmer-Lemeshow goodness-of-fit test (decile-of-risk version).

    H₀: the model is well calibrated.
    A *large* p-value (> 0.05) indicates no evidence of poor calibration.

    Parameters
    ----------
    g : number of groups (default 10 = deciles)
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    groups = np.array_split(np.arange(len(y_true)), g)
    chi2 = 0.0
    for grp in groups:
        n_k = len(grp)
        o1  = y_true[grp].sum()
        e1  = y_prob[grp].sum()
        o0  = n_k - o1
        e0  = n_k - e1
        if e1 > 0:
            chi2 += (o1 - e1) ** 2 / e1
        if e0 > 0:
            chi2 += (o0 - e0) ** 2 / e0

    df = g - 2
    p  = float(1 - stats.chi2.cdf(chi2, df))
    interp = "well calibrated (p > 0.05)" if p > 0.05 else "poor calibration (p ≤ 0.05)"

    return HosmerLemeshowResult(chi2=float(chi2), p=p, df=df, g=g, interpretation=interp)


# ── Calibration-in-the-large & slope ─────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationRegressionResult:
    citl: float            # calibration-in-the-large (intercept on logit scale)
    slope: float           # calibration slope (should be 1.0)
    citl_se: float
    slope_se: float
    citl_p: float
    slope_p: float         # H0: slope = 1


def calibration_regression(
    y_true: NDArray,
    y_prob: NDArray,
) -> CalibrationRegressionResult:
    """
    Logistic calibration regression:
        logit(event) = α + β·logit(p)
    CITL = α (ideally 0), calibration slope = β (ideally 1).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    eps = 1e-7
    lp = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y_true)

    intercept = float(lr.intercept_[0])
    slope     = float(lr.coef_[0, 0])

    # approximate SEs via Hessian / variance of logit
    n = len(y_true)
    # CITL: intercept at mean logit  (closed form SE approximation)
    p_hat = y_prob.mean()
    citl_se = float(np.sqrt(1 / (n * p_hat * (1 - p_hat))) if p_hat > 0 else 0.0)
    slope_se = float(np.std(lp, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    citl_p  = float(2 * (1 - stats.norm.cdf(abs(intercept / (citl_se + 1e-12)))))
    slope_z = (slope - 1.0) / (slope_se + 1e-12)
    slope_p = float(2 * (1 - stats.norm.cdf(abs(slope_z))))

    return CalibrationRegressionResult(
        citl=intercept, slope=slope,
        citl_se=citl_se, slope_se=slope_se,
        citl_p=citl_p, slope_p=slope_p,
    )


__all__ = [
    "brier_score", "BrierResult",
    "calibration_error", "CalibrationBins",
    "hosmer_lemeshow", "HosmerLemeshowResult",
    "calibration_regression", "CalibrationRegressionResult",
]
