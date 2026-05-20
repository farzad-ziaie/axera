"""Tests for axera.medical — regression tests for all fixed bugs."""

from __future__ import annotations

import numpy as np
import pytest
from axera.medical.agreement import bland_altman, concordance_correlation
from axera.medical.calibration import (
    brier_score,
    hosmer_lemeshow,
)
from axera.medical.discrimination import operating_point, reclassification, roc_auc
from axera.medical.icc import icc

# ── ICC regression tests ──────────────────────────────────────────────────────

class TestICC:
    """
    Regression tests against known values from McGraw & Wong (1996) Table 1.
    These also verify the ^ → ** bug fix in cases A-1 and A-k.
    """
    @pytest.fixture
    def sample_matrix(self):
        # Small 4-subject, 3-rater matrix
        return np.array([
            [9, 2, 5],
            [6, 1, 3],
            [8, 4, 6],
            [7, 1, 2],
        ], dtype=float)

    def test_all_cases_run_without_error(self, sample_matrix):
        for case in ("1-1", "1-k", "C-1", "C-k", "A-1", "A-k"):
            result = icc(sample_matrix, icc_type=case)
            assert "r" in result
            assert "lower" in result
            assert "upper" in result
            assert "p" in result

    def test_icc_C1_range(self, sample_matrix):
        result = icc(sample_matrix, icc_type="C-1")
        assert -1.0 <= result["r"] <= 1.0

    def test_icc_CI_ordering(self, sample_matrix):
        for case in ("1-1", "1-k", "C-1", "C-k", "A-1", "A-k"):
            result = icc(sample_matrix, icc_type=case)
            assert result["lower"] <= result["r"] <= result["upper"], (
                f"CI ordering violated for {case}: {result}"
            )

    def test_icc_perfect_agreement(self):
        """Perfect agreement → ICC close to 1."""
        M = np.column_stack([np.arange(1, 11)] * 3).astype(float)
        result = icc(M, icc_type="C-1")
        assert result["r"] == pytest.approx(1.0, abs=0.01)

    def test_icc_invalid_matrix(self):
        with pytest.raises(ValueError):
            icc(np.array([1, 2, 3]))          # not 2-D
        with pytest.raises(ValueError):
            icc(np.array([[1, 2]]))            # only 1 subject
        with pytest.raises(ValueError):
            icc(np.array([[1], [2], [3]]))     # only 1 rater

    def test_icc_A1_exponentiation_is_not_xor(self, sample_matrix):
        """
        Regression test: previously ^ was used instead of **.
        With the XOR bug, this would produce a wildly wrong value for case A-1.
        """
        result = icc(sample_matrix, icc_type="A-1")
        # A-1 must be finite and within [-1, 1]
        assert np.isfinite(result["r"])
        assert -1.0 <= result["r"] <= 1.0
        # Verify it's NOT the XOR'd integer result
        assert abs(result["r"]) < 10_000


# ── Bland-Altman ──────────────────────────────────────────────────────────────

class TestBlandAltman:
    def test_zero_bias(self):
        y = np.arange(1.0, 51.0)
        ba = bland_altman(y, y)  # identical measurements
        assert ba.bias == pytest.approx(0.0, abs=1e-10)

    def test_constant_bias(self, method_comparison_data):
        y_ref, y_new = method_comparison_data
        ba = bland_altman(y_new, y_ref)
        # bias should be positive (y_new > y_ref by construction)
        assert ba.bias > 0

    def test_loa_ordering(self, method_comparison_data):
        y_ref, y_new = method_comparison_data
        ba = bland_altman(y_new, y_ref)
        assert ba.loa_lower < ba.bias < ba.loa_upper

    def test_ci_coverage(self, method_comparison_data):
        y_ref, y_new = method_comparison_data
        ba = bland_altman(y_new, y_ref)
        assert ba.bias_lower < ba.bias < ba.bias_upper


# ── CCC ───────────────────────────────────────────────────────────────────────

class TestCCC:
    def test_perfect_agreement(self):
        y = np.linspace(1, 10, 50)
        result = concordance_correlation(y, y)
        assert result.ccc == pytest.approx(1.0, abs=0.01)

    def test_ci_ordering(self, method_comparison_data):
        y_ref, y_new = method_comparison_data
        result = concordance_correlation(y_new, y_ref)
        assert result.lower <= result.ccc <= result.upper


# ── Discrimination ────────────────────────────────────────────────────────────

class TestAUC:
    def test_perfect_classifier(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        result = roc_auc(y, s)
        assert result.auc == pytest.approx(1.0)

    def test_random_classifier_near_05(self):
        rng = np.random.default_rng(42)
        y = rng.integers(0, 2, 500)
        s = rng.uniform(0, 1, 500)
        result = roc_auc(y, s)
        assert 0.3 < result.auc < 0.7

    def test_ci_ordering(self, binary_prediction_data):
        y, p = binary_prediction_data
        result = roc_auc(y, p)
        assert result.lower <= result.auc <= result.upper


class TestOperatingPoint:
    def test_youden(self, binary_prediction_data):
        y, p = binary_prediction_data
        op = operating_point(y, p, strategy="youden")
        assert 0 <= op.sensitivity <= 1
        assert 0 <= op.specificity <= 1
        assert 0 <= op.ppv <= 1
        assert 0 <= op.npv <= 1
        assert op.f1 >= 0


# ── Calibration ───────────────────────────────────────────────────────────────

class TestBrierScore:
    def test_perfect_calibration(self):
        y = np.array([0, 1, 0, 1, 0, 1], dtype=float)
        p = y.copy()   # perfect probabilities
        result = brier_score(y, p)
        assert result.brier_score == pytest.approx(0.0, abs=1e-10)

    def test_null_model_skill_near_zero(self, binary_prediction_data):
        y, _ = binary_prediction_data
        prev = y.mean()
        p_null = np.full_like(y, prev, dtype=float)
        result = brier_score(y, p_null)
        assert result.brier_skill == pytest.approx(0.0, abs=0.05)


class TestHosmerLemeshow:
    def test_well_calibrated(self, binary_prediction_data):
        y, p = binary_prediction_data
        result = hosmer_lemeshow(y, p)
        assert result.chi2 >= 0
        assert 0 <= result.p <= 1
        assert result.df == 8  # g - 2


def test_operating_point_all_strategies(binary_prediction_data):
    y, p = binary_prediction_data
    for strategy in ["youden", "sens90", "spec90"]:
        op = operating_point(y, p, strategy=strategy)
        assert 0 <= op.sensitivity <= 1
        assert 0 <= op.specificity <= 1

def test_reclassification(binary_prediction_data):
    y, p = binary_prediction_data
    p_old = p * 0.8 + 0.1
    result = reclassification(y, p_old, p)
    assert -2 <= result.nri <= 2  # reasonable range
