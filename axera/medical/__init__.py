"""
Axera Medical Metrics — high-standard clinical evaluation tools.

All metrics provide bootstrap confidence intervals and are aligned with
reporting guidelines for method-comparison studies (CLSI EP09, Passing-Bablok,
Bland-Altman) and clinical prediction models (TRIPOD, Transparent Reporting).

Submodules
----------
agreement       Bland-Altman, CCC, Cohen's κ
discrimination  ROC-AUC (DeLong CI), sensitivity/specificity, NRI, IDI
calibration     Brier score, ECE, Hosmer-Lemeshow, calibration slope
icc             Intraclass Correlation Coefficient (all 6 McGraw-Wong types)
"""

from axera.medical.agreement import (
    BlandAltmanResult,
    CCCResult,
    KappaResult,
    bland_altman,
    cohen_kappa,
    concordance_correlation,
)
from axera.medical.calibration import (
    BrierResult,
    CalibrationBins,
    CalibrationRegressionResult,
    HosmerLemeshowResult,
    brier_score,
    calibration_error,
    calibration_regression,
    hosmer_lemeshow,
)
from axera.medical.discrimination import (
    AUCResult,
    OperatingPoint,
    ReclassificationResult,
    operating_point,
    reclassification,
    roc_auc,
)
from axera.medical.icc import ICCType, icc

__all__ = [
    # agreement
    "bland_altman", "BlandAltmanResult",
    "concordance_correlation", "CCCResult",
    "cohen_kappa", "KappaResult",
    # discrimination
    "roc_auc", "AUCResult",
    "operating_point", "OperatingPoint",
    "reclassification", "ReclassificationResult",
    # calibration
    "brier_score", "BrierResult",
    "calibration_error", "CalibrationBins",
    "hosmer_lemeshow", "HosmerLemeshowResult",
    "calibration_regression", "CalibrationRegressionResult",
    # icc
    "icc", "ICCType",
]
