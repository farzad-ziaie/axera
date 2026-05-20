"""
Axera utility helpers.

Provides data loading, imputation, and validation functions.
Fixes the original util.py bug where data was dropped before imputation,
causing NaN leakage into the imputed columns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer

# ── Imputation (bug-fixed) ────────────────────────────────────────────────────

def impute_and_drop(
    df: pd.DataFrame,
    impute_cols: list[str] | None = None,
    drop_cols: list[str] | None = None,
    strategy: str = "mean",
) -> pd.DataFrame:
    """
    Impute missing values, then drop unwanted columns.

    Original bug: columns were dropped *before* imputation, causing NaN
    leakage when the imputer tried to compute column statistics.

    Parameters
    ----------
    df          : Input DataFrame
    impute_cols : Columns to impute.  If None, all numeric columns.
    drop_cols   : Columns to drop after imputation.
    strategy    : Imputation strategy ('mean', 'median', 'most_frequent', 'constant')

    Returns
    -------
    pd.DataFrame  with imputed values and dropped columns
    """
    df = df.copy()

    # Step 1: Impute (BEFORE dropping!)
    if impute_cols is None:
        impute_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if impute_cols:
        imp = SimpleImputer(strategy=strategy)
        df[impute_cols] = imp.fit_transform(df[impute_cols])

    # Step 2: Drop (after imputation)
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df


def check_dimensions(X: NDArray, y: NDArray) -> None:
    """Validate X and y have compatible shapes."""
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError(f"X ({len(X)}) and y ({len(y)}) must have the same number of samples.")
    if len(X) < 2:
        raise ValueError("Need at least 2 samples.")


def train_test_split_stratified(
    X: NDArray,
    y: NDArray,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Simple stratified split for binary y."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


__all__ = ["impute_and_drop", "check_dimensions", "train_test_split_stratified"]
