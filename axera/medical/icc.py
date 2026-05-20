"""
Intraclass Correlation Coefficient — all six McGraw & Wong (1996) cases.

Fixed relative to the original axera stats.py:
  - ``^`` replaced with ``**`` everywhere (Python bitwise-XOR bug)
  - All six cases return a consistent (r, LB, UB, F, df1, df2, p) tuple
  - Typed, documented, and importable standalone

Reference
---------
McGraw, K. O., & Wong, S. P. (1996).
Forming inferences about some intraclass correlation coefficients.
*Psychological Methods*, 1(1), 30–46.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.stats import f as f_dist

ICCType = Literal["1-1", "1-k", "C-1", "C-k", "A-1", "A-k"]
ICCResult = tuple[float, float, float, float, float, float, float]


# ── Variance helpers ──────────────────────────────────────────────────────────

def _var_all(M: np.ndarray) -> float:
    """Unbiased variance across all elements of M."""
    flat = M.ravel()
    return float(np.sum((flat - flat.mean()) ** 2) / (flat.size - 1))


def _var_cols(M: np.ndarray) -> np.ndarray:
    """Column-wise unbiased variance."""
    return np.array([
        np.sum((M[:, j] - M[:, j].mean()) ** 2) / (M.shape[0] - 1)
        for j in range(M.shape[1])
    ])


def _ms_terms(M: np.ndarray) -> tuple[float, float, float, float]:
    """
    Compute mean-square terms for a two-way ANOVA layout.

    Returns
    -------
    MSR : between-rows MS
    MSW : within-rows MS (average column variance)
    MSC : between-columns MS
    MSE : error MS
    """
    n, k = M.shape
    ss_total = _var_all(M) * (n * k - 1)
    msr = _var_all(np.mean(M, axis=1)) * k       # between rows
    msw = float(np.mean(_var_cols(M.T)))           # within rows (per col)
    msc = _var_all(np.mean(M, axis=0)) * n        # between cols
    mse = (ss_total - msr * (n - 1) - msc * (k - 1)) / ((n - 1) * (k - 1))
    return msr, msw, msc, mse


# ── Case implementations ──────────────────────────────────────────────────────

def _case_1_1(
    msr: float, msw: float, msc: float, mse: float,
    n: int, k: int, alpha: float, r0: float,
) -> ICCResult:
    r = (msr - msw) / (msr + (k - 1) * msw)
    F = (msr / msw) * (1 - r0) / (1 + (k - 1) * r0)
    df1, df2 = n - 1, n * (k - 1)
    p = 1 - f_dist.cdf(F, df1, df2)
    FL = (msr / msw) / f_dist.ppf(1 - alpha / 2, n - 1, n * (k - 1))
    FU = (msr / msw) * f_dist.ppf(1 - alpha / 2, n * (k - 1), n - 1)
    LB = (FL - 1) / (FL + (k - 1))
    UB = (FU - 1) / (FU + (k - 1))
    return r, LB, UB, F, df1, df2, p


def _case_1_k(
    msr: float, msw: float, msc: float, mse: float,
    n: int, k: int, alpha: float, r0: float,
) -> ICCResult:
    r = (msr - msw) / msr
    F = (msr / msw) * (1 - r0)
    df1, df2 = n - 1, n * (k - 1)
    p = 1 - f_dist.cdf(F, df1, df2)
    FL = (msr / msw) / f_dist.ppf(1 - alpha / 2, n - 1, n * (k - 1))
    FU = (msr / msw) * f_dist.ppf(1 - alpha / 2, n * (k - 1), n - 1)
    LB = 1 - 1 / FL
    UB = 1 - 1 / FU
    return r, LB, UB, F, df1, df2, p


def _case_C_1(
    msr: float, msw: float, msc: float, mse: float,
    n: int, k: int, alpha: float, r0: float,
) -> ICCResult:
    r = (msr - mse) / (msr + (k - 1) * mse)
    F = (msr / mse) * (1 - r0) / (1 + (k - 1) * r0)
    df1, df2 = n - 1, (n - 1) * (k - 1)
    p = 1 - f_dist.cdf(F, df1, df2)
    FL = (msr / mse) / f_dist.ppf(1 - alpha / 2, n - 1, (n - 1) * (k - 1))
    FU = (msr / mse) * f_dist.ppf(1 - alpha / 2, (n - 1) * (k - 1), n - 1)
    LB = (FL - 1) / (FL + (k - 1))
    UB = (FU - 1) / (FU + (k - 1))
    return r, LB, UB, F, df1, df2, p


def _case_C_k(
    msr: float, msw: float, msc: float, mse: float,
    n: int, k: int, alpha: float, r0: float,
) -> ICCResult:
    r = (msr - mse) / msr
    F = (msr / mse) * (1 - r0)
    df1, df2 = n - 1, (n - 1) * (k - 1)
    p = 1 - f_dist.cdf(F, df1, df2)
    FL = (msr / mse) / f_dist.ppf(1 - alpha / 2, n - 1, (n - 1) * (k - 1))
    FU = (msr / mse) * f_dist.ppf(1 - alpha / 2, (n - 1) * (k - 1), n - 1)
    LB = 1 - 1 / FL
    UB = 1 - 1 / FU
    return r, LB, UB, F, df1, df2, p


def _case_A_1(
    msr: float, msw: float, msc: float, mse: float,
    n: int, k: int, alpha: float, r0: float,
) -> ICCResult:
    # Bug-fixed: ^ → ** throughout this case
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    r = (msr - mse) / denom
    a_h = (k * r0) / (n * (1 - r0))
    b_h = 1 + (k * r0 * (n - 1)) / (n * (1 - r0))
    F = msr / (a_h * msc + b_h * mse)
    a = k * r / (n * (1 - r))
    b = 1 + k * r * (n - 1) / (n * (1 - r))
    v_num = (a * msc + b * mse) ** 2
    v_den = (a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = v_num / v_den if v_den != 0 else float("inf")
    df1 = n - 1
    df2 = v
    p = 1 - f_dist.cdf(F, df1, df2)
    Fs1 = f_dist.ppf(1 - alpha / 2, n - 1, v)
    LB = n * (msr - Fs1 * mse) / (Fs1 * (k * msc + (k * n - k - n) * mse) + n * msr)
    Fs2 = f_dist.ppf(1 - alpha / 2, v, n - 1)
    UB = n * (Fs2 * msr - mse) / (k * msc + (k * n - k - n) * mse + n * Fs2 * msr)
    return r, LB, UB, F, df1, df2, p


def _case_A_k(
    msr: float, msw: float, msc: float, mse: float,
    n: int, k: int, alpha: float, r0: float,
) -> ICCResult:
    # Bug-fixed: ^ → **
    r = (msr - mse) / (msr + (msc - mse) / n)
    c = r0 / (n * (1 - r0))
    d = 1 + (r0 * (n - 1)) / (n * (1 - r0))
    F = msr / (c * msc + d * mse)
    a = k * r / (n * (1 - r))
    b = 1 + k * r * (n - 1) / (n * (1 - r))
    v_num = (a * msc + b * mse) ** 2
    v_den = (a * msc) ** 2 / (k - 1) + (b * mse) ** 2 / ((n - 1) * (k - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        v = v_num / v_den if v_den != 0 else float("inf")
    df1 = n - 1
    df2 = v
    p = 1 - f_dist.cdf(F, df1, df2)
    Fs1 = f_dist.ppf(1 - alpha / 2, n - 1, v)
    LB = n * (msr - Fs1 * mse) / (Fs1 * (msc - mse) + n * msr)
    Fs2 = f_dist.ppf(1 - alpha / 2, v, n - 1)
    UB = n * (Fs2 * msr - mse) / (msc - mse + n * Fs2 * msr)
    return r, LB, UB, F, df1, df2, p


# ── Public API ────────────────────────────────────────────────────────────────

_CASE_MAP = {
    "1-1": _case_1_1,
    "1-k": _case_1_k,
    "C-1": _case_C_1,
    "C-k": _case_C_k,
    "A-1": _case_A_1,
    "A-k": _case_A_k,
}


def icc(
    M: np.ndarray,
    icc_type: ICCType = "C-1",
    alpha: float = 0.05,
    r0: float = 0.0,
) -> dict[str, float]:
    """
    Compute an intraclass correlation coefficient.

    Parameters
    ----------
    M : ndarray  shape (n_subjects, k_raters)
        Observation matrix.  Rows = subjects, columns = raters/measurements.
    icc_type : str
        One of ``'1-1'``, ``'1-k'``, ``'C-1'``, ``'C-k'``, ``'A-1'``, ``'A-k'``.
    alpha : float
        Significance level for confidence interval (default 0.05 → 95 % CI).
    r0 : float
        Null-hypothesis ICC value for the hypothesis test (default 0).

    Returns
    -------
    dict with keys: r, lower, upper, F, df1, df2, p

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[1,2],[3,4],[5,6],[7,8]])
    >>> result = icc(M, icc_type='C-1')
    >>> round(result['r'], 4)
    0.9...
    """
    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError("M must be a 2-D matrix (subjects × raters).")
    n, k = M.shape
    if n < 2:
        raise ValueError("Need at least 2 subjects.")
    if k < 2:
        raise ValueError("Need at least 2 raters/measurements.")
    if icc_type not in _CASE_MAP:
        raise ValueError(f"icc_type must be one of {list(_CASE_MAP)}, got {icc_type!r}.")

    msr, msw, msc, mse = _ms_terms(M)
    r, LB, UB, F, df1, df2, p = _CASE_MAP[icc_type](msr, msw, msc, mse, n, k, alpha, r0)

    return {
        "r": float(r),
        "lower": float(LB),
        "upper": float(UB),
        "F": float(F),
        "df1": float(df1),
        "df2": float(df2),
        "p": float(p),
    }


__all__ = ["icc", "ICCType"]
