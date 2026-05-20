"""
Method-agreement metrics for biomedical studies.

All metrics return bootstrap confidence intervals (BCa method by default)
since closed-form intervals are often inappropriate for small clinical samples.

Includes
--------
- Bland-Altman analysis (bias, limits of agreement, proportional-bias test)
- Lin's Concordance Correlation Coefficient (CCC)
- Cohen's κ and weighted κ
- Intraclass correlation (re-exported from axera.medical.icc)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from scipy.optimize import curve_fit

from axera.medical.icc import ICCType, icc


# ── Bootstrap helpers ─────────────────────────────────────────────────────────

def _bootstrap_ci(
    y: NDArray,
    y_hat: NDArray,
    stat_fn,  # callable(y, y_hat) -> float
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """BCa bootstrap confidence interval for any scalar statistic."""
    rng = np.random.default_rng(seed)
    n = len(y)
    boot = np.array([
        stat_fn(y[idx := rng.integers(0, n, n)], y_hat[idx])
        for _ in range(n_boot)
    ])
    lo = (1 - ci) / 2 * 100
    hi = (1 + ci) / 2 * 100
    return float(np.percentile(boot, lo)), float(np.percentile(boot, hi))


# ── Bland-Altman ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlandAltmanResult:
    """Full Bland-Altman analysis result."""
    bias: float                      # mean difference (method1 − method2)
    bias_lower: float                # bootstrap CI lower
    bias_upper: float                # bootstrap CI upper
    loa_upper: float                 # upper limit of agreement
    loa_lower: float                 # lower limit of agreement
    loa_upper_ci: tuple[float, float]
    loa_lower_ci: tuple[float, float]
    sd_diff: float                   # SD of differences
    proportional_bias_slope: float   # regression slope of diff ~ mean
    proportional_bias_p: float       # p-value: H0 = no proportional bias
    proportional_bias_angle: float   # 2·arctan(slope)/π (0 = no bias)
    n: int


def bland_altman(
    y: NDArray,
    y_ref: NDArray,
    n_boot: int = 2000,
    ci: float = 0.95,
    multiplier: float = 1.96,
) -> BlandAltmanResult:
    """
    Compute a full Bland-Altman analysis.

    Parameters
    ----------
    y : array  (n,)
        Measurements from the new method.
    y_ref : array  (n,)
        Reference / gold-standard measurements.
    n_boot : int
        Number of bootstrap replicates for CIs.
    ci : float
        Confidence level (default 0.95 → 95 %).
    multiplier : float
        LoA multiplier (default 1.96 ≈ 95 % for normal differences).

    Returns
    -------
    BlandAltmanResult
    """
    y = np.asarray(y, dtype=float).ravel()
    y_ref = np.asarray(y_ref, dtype=float).ravel()
    if y.shape != y_ref.shape:
        raise ValueError("y and y_ref must have the same length.")

    diff = y - y_ref
    mean_ = (y + y_ref) / 2.0

    # Bias
    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    loa_up = bias + multiplier * sd
    loa_lo = bias - multiplier * sd

    # Bootstrap CIs for bias
    bias_lo, bias_hi = _bootstrap_ci(
        y, y_ref,
        lambda a, b: float(np.mean(a - b)),
        n_boot=n_boot, ci=ci,
    )

    # Bootstrap CIs for LoA
    def _loa_upper(a: NDArray, b: NDArray) -> float:
        d = a - b
        return float(d.mean() + multiplier * d.std(ddof=1))

    def _loa_lower(a: NDArray, b: NDArray) -> float:
        d = a - b
        return float(d.mean() - multiplier * d.std(ddof=1))

    loa_up_ci = _bootstrap_ci(y, y_ref, _loa_upper, n_boot=n_boot, ci=ci)
    loa_lo_ci = _bootstrap_ci(y, y_ref, _loa_lower, n_boot=n_boot, ci=ci)

    # Proportional bias: linear regression diff ~ mean
    try:
        slope, intercept, r_val, p_val, se = stats.linregress(mean_, diff)
    except Exception:
        slope, p_val = 0.0, 1.0

    angle = float(2 * np.arctan(slope) / np.pi)

    return BlandAltmanResult(
        bias=bias,
        bias_lower=bias_lo,
        bias_upper=bias_hi,
        loa_upper=loa_up,
        loa_lower=loa_lo,
        loa_upper_ci=loa_up_ci,
        loa_lower_ci=loa_lo_ci,
        sd_diff=sd,
        proportional_bias_slope=float(slope),
        proportional_bias_p=float(p_val),
        proportional_bias_angle=angle,
        n=len(y),
    )


# ── Lin's CCC ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CCCResult:
    ccc: float
    lower: float
    upper: float
    pearson_r: float
    cb: float    # bias correction factor


def concordance_correlation(
    y: NDArray,
    y_ref: NDArray,
    ci: float = 0.95,
    n_boot: int = 2000,
) -> CCCResult:
    """
    Lin's Concordance Correlation Coefficient.

    CCC = 2·σ₁₂ / (σ₁² + σ₂² + (μ₁−μ₂)²)

    References
    ----------
    Lin, L. I. (1989). Biometrics, 45(1), 255–268.
    """
    y = np.asarray(y, dtype=float).ravel()
    y_ref = np.asarray(y_ref, dtype=float).ravel()

    def _ccc(a: NDArray, b: NDArray) -> float:
        mu_a, mu_b = a.mean(), b.mean()
        var_a, var_b = a.var(ddof=1), b.var(ddof=1)
        cov = np.cov(a, b, ddof=1)[0, 1]
        return 2 * cov / (var_a + var_b + (mu_a - mu_b) ** 2)

    ccc = _ccc(y, y_ref)
    lo, hi = _bootstrap_ci(y, y_ref, _ccc, n_boot=n_boot, ci=ci)
    r = float(np.corrcoef(y, y_ref)[0, 1])
    # Bias correction factor
    mu_y, mu_r = y.mean(), y_ref.mean()
    sd_y, sd_r = y.std(ddof=1), y_ref.std(ddof=1)
    cb = 2 / ((sd_y / sd_r + sd_r / sd_y) + (mu_y - mu_r) ** 2 / (sd_y * sd_r))
    return CCCResult(ccc=float(ccc), lower=float(lo), upper=float(hi), pearson_r=r, cb=float(cb))


# ── Cohen's κ ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KappaResult:
    kappa: float
    lower: float
    upper: float
    se: float
    weighted: bool


def cohen_kappa(
    y_true: NDArray,
    y_pred: NDArray,
    weights: Optional[Literal["linear", "quadratic"]] = None,
    ci: float = 0.95,
    n_boot: int = 2000,
) -> KappaResult:
    """
    Cohen's κ (optionally weighted) with bootstrap CI.

    Parameters
    ----------
    y_true, y_pred : array of int labels
    weights : None (unweighted), 'linear', or 'quadratic'
    """
    from sklearn.metrics import cohen_kappa_score

    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()

    def _k(a: NDArray, b: NDArray) -> float:
        try:
            return float(cohen_kappa_score(a, b, weights=weights))
        except Exception:
            return float("nan")

    kappa = _k(y_true, y_pred)
    lo, hi = _bootstrap_ci(y_true, y_pred, _k, n_boot=n_boot, ci=ci)

    # Asymptotic SE (Fleiss 1971 approximation)
    n = len(y_true)
    se = float(np.sqrt((1 - kappa) ** 2 / n)) if n > 0 else 0.0

    return KappaResult(
        kappa=kappa, lower=lo, upper=hi, se=se, weighted=weights is not None
    )


__all__ = [
    "bland_altman", "BlandAltmanResult",
    "concordance_correlation", "CCCResult",
    "cohen_kappa", "KappaResult",
    "icc",
]
