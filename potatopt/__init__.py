# ==============================================================================
# PotatOpt: Universal Potato-Optimized Industrial AI Engine (OOP Architecture)
# Designed for Low-Resource, CPU-Friendly Machine Learning & Smart Manufacturing
# ==============================================================================
"""
PotatOpt is a zero-code, production-grade Machine Learning engine designed 
specifically for Industrial Engineering (IE) and Smart Factory environments.

Key Architecture Principles:
----------------------------
1. LowSpecML (Potato Hardware First):
   Minimizes RAM and CPU overhead using lossless numerical downcasting and 
   ordinal encoding instead of memory-heavy one-hot matrices.

2. Production Guardrails:
   Automatically detects extreme class imbalance and seamlessly transitions 
   to unsupervised Anomaly Detection (Isolation Forest) when defects are too rare (< 5 samples).

3. Zero Data Leakage:
   Strictly learns statistical boundaries, imputations, and encodings during `fit()`, 
   and applies them idempotently during `transform()` inference.

4. Industrial Financial Layer:
   Integrates Cost of Quality (CoQ) equations directly into threshold tuning, 
   maximizing ROI by balancing scrap costs against false-alarm inspection labor.

5. ISO 9001 Traceability:
   Serializes each model alongside a SHA-256 integrity hash and a metadata
   sidecar, so a deployed file can be shown to be the file that was released.
   That is an integrity record, not a signature - it detects a corrupted or
   swapped file but cannot prove who produced one. See `PotatOptEngine.save()`.
"""

from __future__ import annotations

__version__ = "1.6.0"

from typing import Any

# The redundant `x as x` form marks a deliberate re-export rather than an unused
# import. These four are private, so they cannot be declared in __all__, but the
# test suite reaches for them through the package object and they have to resolve.
from ._lazy import _load_automl as _load_automl
from ._lazy import _load_shap as _load_shap
from ._lazy import _quiet_dependency_warnings as _quiet_dependency_warnings
from ._lazy import logger
from ._utils import (
    enable_audit_log,
    get_library_versions,
    to_jsonable,
)
from .analysis import (
    auto_analyze,
    run_seed_sweep,
)
from .calibration import (
    check_calibration,
)
from .constants import (
    AUTOCORRELATION_WARN,
    CALIBRATION_DEFAULT_BINS,
    CALIBRATION_ECE_LIMIT,
    CAPABILITY_OUTLIER_RATE_LIMIT,
    CAPABILITY_SIGMA_RATIO_LIMIT,
    CONTROL_RULE_DESCRIPTIONS,
    CONTROL_RULES_NELSON,
    CONTROL_RULES_WESTERN_ELECTRIC,
    CUSUM_DEFAULT_DECISION,
    CUSUM_DEFAULT_SLACK,
    DEFAULT_RANDOM_STATE,
    DQS_PRODUCTION_READY,
    DQS_USABLE,
    DQS_WEIGHTS,
    DRIFT_MIN_ROWS,
    DRIFT_NOISE_SIGMAS,
    EWMA_DEFAULT_LAMBDA,
    EWMA_DEFAULT_SIGMAS,
    MIN_TRAIN_ROWS,
    MISSING_SCHEMA_WARN_RATIO,
    MODIFIED_ZSCORE_THRESHOLD,
    MOVING_RANGE_D2,
    NUMERIC_SENTINELS,
    OEE_WORLD_CLASS,
    OUT_OF_BOUNDS_WARN_RATIO,
    PARETO_CUTOFF,
    PSI_DEFAULT_BINS,
    PSI_MAJOR_SHIFT,
    PSI_MAX_CATEGORIES,
    PSI_MODERATE_SHIFT,
    SEED_SWEEP_DEFAULT,
    SILENT_NULL_TOKENS,
)
from .data import (
    audit_data_quality,
    calculate_correlations,
    detect_outliers,
    detect_silent_nulls,
    inspect_data,
    split_data,
    split_data_three_way,
)
from .drift import (
    calculate_categorical_psi,
    calculate_psi,
    check_asset_drift,
    check_data_drift,
)
from .engine import (
    PotatOpt,
    PotatOptEngine,
)
from .quality import (
    calculate_capability,
    calculate_gauge_rr,
)
from .reliability import (
    calculate_availability,
    calculate_maintenance_savings,
    calculate_mtbf,
    calculate_mttr,
    calculate_oee,
    calculate_pareto,
    wilson_confidence_interval,
)
from .spc import _lag1_autocorrelation as _lag1_autocorrelation
from .spc import (
    calculate_control_rules,
    calculate_cusum_chart,
    calculate_ewma_chart,
    calculate_spc_limits,
)


def __getattr__(name: str) -> Any:
    """
    PEP 562 module-level attribute access.

    Keeps `potatopt.AutoML` and `potatopt.shap` resolvable for callers and tests
    that reach for them directly; they simply trigger the import on first touch
    instead of at module load.
    """
    if name == "AutoML":
        return _load_automl()
    if name == "shap":
        return _load_shap()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# The supported public surface. Everything not listed here is an implementation
# detail and may change without notice - including the third-party modules this
# file imports, which would otherwise leak out as `potatopt.np`, `potatopt.shap`
# and so on.
__all__ = [  # noqa: RUF022
    # Engine
    "PotatOptEngine",
    "PotatOpt",
    # Data inspection, calibration and splitting
    "inspect_data",
    "calculate_correlations",
    "split_data",
    "split_data_three_way",
    "auto_analyze",
    "check_calibration",
    "run_seed_sweep",
    # Data quality
    "audit_data_quality",
    "detect_silent_nulls",
    "detect_outliers",
    # Statistical process control and drift
    "calculate_spc_limits",
    "calculate_ewma_chart",
    "calculate_cusum_chart",
    "calculate_control_rules",
    "calculate_capability",
    "calculate_gauge_rr",
    "check_data_drift",
    "check_asset_drift",
    "calculate_psi",
    "calculate_categorical_psi",
    "wilson_confidence_interval",
    "calculate_maintenance_savings",
    # Reliability, availability and OEE
    "calculate_mtbf",
    "calculate_mttr",
    "calculate_availability",
    "calculate_oee",
    "calculate_pareto",
    # Interop and traceability
    "to_jsonable",
    "enable_audit_log",
    "get_library_versions",
    "logger",
    "__version__",
    # Tuning constants
    "DEFAULT_RANDOM_STATE",
    "SEED_SWEEP_DEFAULT",
    "CALIBRATION_DEFAULT_BINS",
    "CALIBRATION_ECE_LIMIT",
    "CAPABILITY_SIGMA_RATIO_LIMIT",
    "CAPABILITY_OUTLIER_RATE_LIMIT",
    "MIN_TRAIN_ROWS",
    "MISSING_SCHEMA_WARN_RATIO",
    "OUT_OF_BOUNDS_WARN_RATIO",
    "PSI_MODERATE_SHIFT",
    "PSI_MAJOR_SHIFT",
    "PSI_DEFAULT_BINS",
    "PSI_MAX_CATEGORIES",
    "DRIFT_MIN_ROWS",
    "DRIFT_NOISE_SIGMAS",
    "EWMA_DEFAULT_LAMBDA",
    "EWMA_DEFAULT_SIGMAS",
    "CUSUM_DEFAULT_SLACK",
    "CUSUM_DEFAULT_DECISION",
    "MOVING_RANGE_D2",
    "AUTOCORRELATION_WARN",
    "CONTROL_RULES_WESTERN_ELECTRIC",
    "CONTROL_RULES_NELSON",
    "CONTROL_RULE_DESCRIPTIONS",
    "DQS_WEIGHTS",
    "DQS_PRODUCTION_READY",
    "DQS_USABLE",
    "SILENT_NULL_TOKENS",
    "NUMERIC_SENTINELS",
    "MODIFIED_ZSCORE_THRESHOLD",
    "PARETO_CUTOFF",
    "OEE_WORLD_CLASS",
]
