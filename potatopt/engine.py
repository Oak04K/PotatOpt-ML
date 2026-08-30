from __future__ import annotations

# Standard library
import hashlib  # SHA-256 model-file hashing (save/load) + SHA-1 threshold-tuning fingerprint
import json  # metadata sidecar file (save/load) and NaN/Infinity-safe serialization (to_jsonable)
import os  # path handling in save()/load() (abspath, dirname, makedirs, splitext)
import re  # column-name sanitizing + ID-like column detection (drop_id_like_columns)
from datetime import datetime, timezone  # UTC timestamps written into save() metadata
from typing import Any  # loose return-type annotations for JSON-shaped dicts

# Core numerical / dataframe stack - the four packages that make up the "core install"
import joblib  # model serialization: joblib.dump/load in PotatOptEngine.save()/load()
import numpy as np  # arrays + math for SPC/EWMA/CUSUM limits, downcasting, anomaly scoring
import pandas as pd  # DataFrame/Series is the data contract for every public function
import sklearn  # only for sklearn.__version__, recorded in save() metadata
from sklearn.base import (
    BaseEstimator,  # lets PotatOptEngine plug into cross_val_score/GridSearchCV/Pipeline
)
from sklearn.ensemble import (
    IsolationForest,  # unsupervised fallback when a defect class has < 5 samples
)
from sklearn.metrics import (  # the metric functions behind evaluate() / calculate_cost_of_quality()
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import (  # the LowSpecML preprocessing layer fit() learns and transform() replays
    LabelEncoder,  # target encoding for classification
    MinMaxScaler,  # scaler="minmax" option
    OrdinalEncoder,  # categorical features - chosen over one-hot to keep RAM flat
    StandardScaler,  # scaler="standard" option (default)
)

from ._lazy import _load_automl, _load_shap, _quiet_dependency_warnings, logger
from ._utils import _is_numeric_series, get_library_versions, to_jsonable
from .calibration import check_calibration
from .constants import (
    CALIBRATION_DEFAULT_BINS,
    DEFAULT_RANDOM_STATE,
    MIN_TRAIN_ROWS,
    MISSING_SCHEMA_WARN_RATIO,
    OUT_OF_BOUNDS_WARN_RATIO,
    PSI_MAJOR_SHIFT,
    PSI_MODERATE_SHIFT,
    SILENT_NULL_TOKENS,
)
from .data import audit_data_quality
from .drift import _build_psi_bins, _psi_core
from .reliability import calculate_maintenance_savings, wilson_confidence_interval


def _package_version() -> str:
    """
    Read the one `__version__` the package declares, at call time.

    The literal lives in `potatopt/__init__.py` and must stay there: setuptools
    reads `attr = "potatopt.__version__"` straight out of the AST, and only for a
    literal assignment. Copying it here as a second literal is what the package
    split originally did, and the two silently disagreed the first time the
    version was bumped - the training report and the saved model metadata kept
    stamping the old number. The import is deferred to call time so that this
    module never depends on how far `__init__.py` has executed.
    """
    from potatopt import __version__

    return __version__


# ==============================================================================
# 2. PotatOpt Engine (Unified ML Pipeline Class)
# ==============================================================================

class PotatOptEngine(BaseEstimator):
    """
    The core Potato-Optimized OOP Pipeline that automates data preprocessing, 
    model training, evaluation, and deployment preparation with zero data leakage.
    """
    def __init__(self, task: str = "auto", time_budget: int = 30, scale_method: str = 'standard',
                 apply_smote: bool | None = None, cost_sensitive_weighting: bool = False,
                 collinear_threshold: float = 0.9, estimators: list[str] | None = None,
                 handle_silent_nulls: bool = True, audit_data: bool = True,
                 n_jobs: int = -1, verbose: int = 0,
                 random_state: int = DEFAULT_RANDOM_STATE) -> None:
        self.task = task
        self.time_budget = max(1, time_budget)
        self.scale_method = scale_method
        # Support both explicit cost_sensitive_weighting and legacy apply_smote alias (Default: False / Opt-In)
        self.cost_sensitive_weighting = cost_sensitive_weighting if apply_smote is None else apply_smote
        self.apply_smote = self.cost_sensitive_weighting
        self.collinear_threshold = collinear_threshold
        self.estimators = estimators if estimators is not None else ["lgbm", "xgboost", "rf"]

        # Convert placeholder strings ("N/A", "-", "null") to NaN so they are imputed
        # rather than learned as a real category. Disable only if a token in
        # SILENT_NULL_TOKENS is a legitimate value in your process data.
        self.handle_silent_nulls = handle_silent_nulls
        # Run a Data Quality Score audit on the training set during fit()
        self.audit_data = audit_data

        # Worker count handed to IsolationForest and FLAML. Follows the joblib
        # convention: -1 uses every core, -2 leaves one free, a positive integer
        # pins an exact worker count. Kept configurable because saturating every
        # core on a 2-core machine is slower than using one, and low-spec targets
        # still need the machine to stay usable while a model trains.
        if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)):
            raise ValueError(f"n_jobs must be an integer, got {n_jobs!r}.")  # noqa: TRY004
        if int(n_jobs) == 0:
            raise ValueError("n_jobs must not be 0; use -1 for all cores or a positive worker count.")
        self.n_jobs = int(n_jobs)
        # FLAML logs its entire search at INFO unless told otherwise - one line
        # per iteration and then the winning estimator's full repr, which is
        # roughly 200 lines for a 15-second budget. A library should be quiet
        # unless the caller asks it to speak, and an agent driving this through
        # auto_analyze() pays for every one of those lines. verbose=1 or higher
        # hands the search log back when you need to see why a model was chosen.
        self.verbose = max(0, int(verbose))

        # One seed for everything stochastic in this pipeline: FLAML's search and
        # the IsolationForest fallback. It used to be hard-coded 42 in three
        # places, which made every result a single-seed result that nobody could
        # vary. Held as a parameter so a report can say which seed it used and
        # run_seed_sweep() can show how much of the score was the seed.
        if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
            raise ValueError(f"random_state must be an integer, got {random_state!r}.")  # noqa: TRY004
        self.random_state = int(random_state)
        self.shap_additivity_relaxed = False

        # Initialize internal state machine
        self._reset_state()

    def _reset_state(self) -> None:
        """ Reset all learned parameters to allow safe, idempotent re-fitting. """
        self.impute_values = {}
        self.feature_bounds = {}
        self.raw_feature_stats = {}
        self.scaler = None
        self.ordinal_encoder = None
        self.label_encoder = None 
        
        # Schema tracking to guarantee column alignment during inference
        self.dropped_collinear = []
        self.zero_variance_cols = []  
        self.high_cardinality_cols = [] 
        self.datetime_cols = []       
        self.all_nan_cols = []        
        self.categorical_cols = []
        self.numeric_cols = []
        self.raw_feature_names = []
        self.feature_names = []
        self.cat_categories = {} 
        self.column_dtypes = {}
        
        self.model = None
        self.optimal_threshold = 0.5
        self.is_fitted = False

        # Fingerprint of the dataset the decision threshold was tuned on, so
        # evaluate() can detect that results are being reported on the same
        # rows used for tuning (an optimistically biased cost figure).
        self.threshold_tuning_fingerprint = None
        
        # Unsupervised anomaly fallback state for extreme imbalance
        self.is_anomaly_model = False 
        self.anomaly_majority_class = None
        self.anomaly_minority_class = None
        
        # Index of the positive/minority defect class to prevent alphabetical label traps
        self.pos_label_idx = 1
        self.last_predict_warnings = []

        # AutoML search diagnostics for over-fitting defence
        self.validation_loss = None
        self.best_config = None
        self.train_rows = 0
        self.automl_metric = None

        # Post-deployment drift and ISO 9001 provenance state
        self.train_profile = {}
        self.train_timestamp = None
        self.train_data_hash = None

        # Data quality and inference observability counters
        self.train_data_quality = None
        self.silent_nulls_converted = 0
        self.transform_calls = 0
        self.rows_transformed = 0
        self.warning_events = 0

    def __sklearn_is_fitted__(self) -> bool:
        """
        Tell scikit-learn's `check_is_fitted` that this engine is trained.

        The convention `check_is_fitted` looks for is an attribute ending in an
        underscore. This engine records its state as `is_fitted` instead, so
        without this hook `Pipeline.predict` raises NotFittedError on an engine
        that is demonstrably fitted.
        """
        return bool(getattr(self, "is_fitted", False))

    @property
    def classes_(self) -> np.ndarray:
        """
        The class labels seen during fit, in the encoder's own order.

        scikit-learn's classification scorers read this attribute; without it
        `cross_val_score(..., scoring="f1")` fails with an AttributeError. Raising
        AttributeError before fit is the documented scikit-learn idiom - it makes
        `hasattr(engine, "classes_")` correctly report False on an unfitted engine.
        """
        if getattr(self, "label_encoder", None) is not None:
            return self.label_encoder.classes_
        raise AttributeError("classes_ is only available after fitting a classification task.")

    def __sklearn_tags__(self) -> Any:
        """
        Declare whether this engine is acting as a classifier or a regressor.

        The task is a constructor argument rather than a fixed property of the
        class, so the tag is resolved per instance. This matters in practice: with
        no tag scikit-learn falls back to plain `KFold`, and on imbalanced defect
        data a fold can land with almost no positives - a verified 3-fold F1 run
        went from [0.000, 0.706, 0.737] to [0.571, 0.743, 0.800] once the tag let
        it use `StratifiedKFold`.

        `task="auto"` stays untagged on purpose: the task is only resolved during
        fit(), and scikit-learn asks for tags before then.
        """
        tags = super().__sklearn_tags__()
        try:
            from sklearn.utils import ClassifierTags, RegressorTags
        except ImportError:
            # scikit-learn < 1.6 has no tag dataclasses; the base tags still work.
            return tags

        if self.task == "classification":
            tags.estimator_type = "classifier"
            tags.classifier_tags = ClassifierTags()
        elif self.task in ("regression", "forecasting"):
            tags.estimator_type = "regressor"
            tags.regressor_tags = RegressorTags()
        return tags

    def _reduce_mem_usage(self, df: pd.DataFrame) -> pd.DataFrame:
        """ 
        [LowSpecML] Lossless Memory Downcaster.
        Safely compresses integer and float dtypes into smaller byte representations 
        to conserve server and edge-device RAM.
        """
        for col in df.columns:
            col_type = df[col].dtype
            if pd.api.types.is_numeric_dtype(col_type):
                c_min, c_max = df[col].min(), df[col].max()
                if pd.isna(c_min) or pd.isna(c_max):
                    continue
                type_str = str(col_type).lower()
                
                # Downcast integers safely (only if no nulls to avoid NA conversion errors)
                if 'int' in type_str:
                    if not df[col].isnull().any():
                        if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                            df[col] = df[col].astype(np.int8)
                        elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                            df[col] = df[col].astype(np.int16)
                        elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                            df[col] = df[col].astype(np.int32)
                
                # Downcast float64 to float32 if values fit within standard single-precision bounds
                elif 'float' in type_str and c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        return df

    def _convert_silent_nulls(self, X_proc: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """
        Replace placeholder strings that mean "missing" with real NaN.
        Left alone, a column of "N/A" is ordinal-encoded into a legitimate looking
        category and the model learns from an absence of data. The token list is
        fixed, not learned, so applying it at inference introduces no leakage.
        """
        if not getattr(self, "handle_silent_nulls", True):
            return (X_proc, 0)
        converted = 0
        try:
            if X_proc is None or not isinstance(X_proc, pd.DataFrame):
                return (X_proc, converted)
            for col in list(X_proc.columns):
                if X_proc[col].dtype == 'object' or str(X_proc[col].dtype) == 'string':
                    series = X_proc[col]
                    if isinstance(series, pd.DataFrame):
                        continue
                    normalised = series.astype(str).str.strip().str.lower()
                    mask = series.notna() & normalised.isin(SILENT_NULL_TOKENS)
                    n = int(mask.sum())
                    if n == 0:
                        continue
                    X_proc.loc[mask, col] = np.nan
                    converted += n
            return (X_proc, converted)
        except (ValueError, TypeError, KeyError, AttributeError):
            return (X_proc, converted)

    def _preprocess_fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        """ 
        Learn data structures and statistical parameters strictly from the training dataset.
        Learned boundaries and encoders are stored in `self` for production inference.
        """
        X_proc = X.copy()
        
        # 1. Drop nested structures (lists/dicts) that cannot be scaled or encoded
        for col in list(X_proc.columns):
            if X_proc[col].apply(lambda x: isinstance(x, (list, dict))).any():
                X_proc.drop(columns=[col], inplace=True)

        # A pandas Categorical is how a memory-conscious user hands over a string
        # column, and it is the one dtype numpy's own predicates refuse to read.
        # Unwrapping it to plain objects here means every later step - the silent-null
        # sweep, imputation, ordinal encoding, the LightGBM category cast - handles it
        # exactly as it handles any other text column, with no second code path.
        for col in list(X_proc.columns):
            if isinstance(X_proc[col].dtype, pd.CategoricalDtype):
                X_proc[col] = X_proc[col].astype(object)
                
        # 2. Sanitize column names: collision-proof loop preventing duplicate suffix clashes
        cleaned_cols = []
        seen = set()
        for col in X_proc.columns:
            clean_name = re.sub(r'[^a-zA-Z0-9_\u0E00-\u0E7F]+', '_', str(col)).strip('_')
            if not clean_name:
                clean_name = "feature"
            base_name = clean_name
            counter = 1
            while clean_name in seen:
                clean_name = f"{base_name}_{counter}"
                counter += 1
            seen.add(clean_name)
            cleaned_cols.append(clean_name)
            
        X_proc.columns = cleaned_cols
        X_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Convert placeholder strings to NaN before the all-NaN drop, so a column
        # that is entirely "N/A" is recognised as empty rather than constant.
        X_proc, self.silent_nulls_converted = self._convert_silent_nulls(X_proc)
        if self.silent_nulls_converted > 0:
            logger.warning(f"Converted {self.silent_nulls_converted} placeholder values (e.g. 'N/A', '-', 'null') to NaN before imputation.")
        
        # 3. Drop all-NaN columns
        self.all_nan_cols = [col for col in X_proc.columns if X_proc[col].isnull().all()]
        X_proc = X_proc.drop(columns=self.all_nan_cols)
        
        self.raw_feature_names = list(X_proc.columns)
        
        # 4. Extract temporal features from datetime columns and ISO datetime strings
        for col in list(X_proc.columns):
            is_dt = False
            if pd.api.types.is_datetime64_any_dtype(X_proc[col]):
                is_dt = True
            elif X_proc[col].dtype == 'object' or str(X_proc[col].dtype) == 'string':
                sample = X_proc[col].dropna().head(10)
                if not sample.empty and sample.astype(str).str.contains(r'^\d{4}[-/]\d{2}[-/]\d{2}').all():
                    try:
                        converted_dt = pd.to_datetime(X_proc[col], errors='coerce', utc=True)
                        if converted_dt.notnull().sum() > (0.8 * len(X_proc)):
                            X_proc[col] = converted_dt
                            is_dt = True
                    except (ValueError, TypeError, OverflowError):
                        is_dt = False
            
            if is_dt:
                self.datetime_cols.append(col)
                X_proc[f"{col}_year"] = X_proc[col].dt.year.fillna(0)
                X_proc[f"{col}_month"] = X_proc[col].dt.month.fillna(0)
                X_proc[f"{col}_day"] = X_proc[col].dt.day.fillna(0)
                X_proc[f"{col}_hour"] = X_proc[col].dt.hour.fillna(0)
                X_proc[f"{col}_dayofweek"] = X_proc[col].dt.dayofweek.fillna(0)
                X_proc = X_proc.drop(columns=[col])
                
        # 5. Automatically convert numeric strings to numeric dtype to preserve continuous sensor physics
        for col in list(X_proc.columns):
            if X_proc[col].dtype == 'object' or str(X_proc[col].dtype) == 'string':
                num_conv = pd.to_numeric(X_proc[col], errors='coerce')
                if num_conv.notnull().sum() > (0.8 * len(X_proc)) and num_conv.nunique() > 5:
                    X_proc[col] = num_conv

        # 6. Drop high-cardinality noise features (UUIDs, transaction IDs, monotonic serial numbers)
        n_rows = len(X_proc)
        id_pattern = re.compile(r'^(id|index|serial|uuid|guid|code|trans_id|row_id|run_id|seq)$|(_id|_no|_num|_idx)$', re.IGNORECASE)
        for col in list(X_proc.columns):
            col_str = str(col).lower()
            n_unique = X_proc[col].nunique()
            
            # String identifier check
            if X_proc[col].dtype == 'object' or str(X_proc[col].dtype) == 'string':
                if (n_unique > (0.5 * n_rows) and n_rows > 30) or n_unique > 1000:
                    self.high_cardinality_cols.append(col)
            # Monotonic integer sequential ID check
            elif (
                _is_numeric_series(X_proc[col])
                and n_rows > 30
                and n_unique == n_rows
                and bool(id_pattern.search(col_str))
            ):
                self.high_cardinality_cols.append(col)
                    
        X_proc = X_proc.drop(columns=self.high_cardinality_cols)
                
        # 7. Missing Value Imputation: Median for numeric features, Mode for categorical features
        for col in list(X_proc.columns):
            if _is_numeric_series(X_proc[col]):
                val = X_proc[col].median()
            else:
                val = X_proc[col].mode()[0] if not X_proc[col].mode().empty else "Unknown"
            
            if pd.isna(val): val = 0 
            
            self.impute_values[col] = val
            X_proc[col] = X_proc[col].fillna(val)
            
            if X_proc[col].dtype == 'object' or str(X_proc[col].dtype) == 'string':
                self.categorical_cols.append(col)
            else:
                self.numeric_cols.append(col)
                
        X_proc = self._reduce_mem_usage(X_proc)
                
        # 8. Identify and drop zero-variance constant columns
        for col in list(self.numeric_cols):
            if X_proc[col].nunique() <= 1:
                self.zero_variance_cols.append(col)
                self.numeric_cols.remove(col)
                
        for col in list(self.categorical_cols):
            if X_proc[col].nunique() <= 1:
                self.zero_variance_cols.append(col)
                self.categorical_cols.remove(col)
                
        X_proc = X_proc.drop(columns=self.zero_variance_cols)
                
        # Record training bounds for inference outlier clipping
        for col in self.numeric_cols:
            self.feature_bounds[col] = {'min': float(X_proc[col].min()), 'max': float(X_proc[col].max())}
            # Raw engineering-unit statistics, captured before scaling so drift can be
            # reported in the operator's units instead of scaler space.
            raw_std = X_proc[col].std()
            self.raw_feature_stats[col] = {
                'mean': float(X_proc[col].mean()),
                'std': 0.0 if pd.isna(raw_std) else float(raw_std)
            }
                
        # 9. LowSpecML Ordinal Encoding: Memory-efficient categorical mapping with unseen label safety (-1)
        if len(self.categorical_cols) > 0:
            self.ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            encoded_vals = self.ordinal_encoder.fit_transform(X_proc[self.categorical_cols].astype(str))
            X_proc[self.categorical_cols] = encoded_vals
            
            # Explicitly include -1 in categorical categories to prevent runtime NaNs during LightGBM inference
            self.cat_categories = {}
            for i, col in enumerate(self.categorical_cols):
                cats = list(np.arange(len(self.ordinal_encoder.categories_[i])))
                if -1 not in cats:
                    cats.append(-1)
                self.cat_categories[col] = cats
                X_proc[col] = pd.Categorical(X_proc[col], categories=cats)
            
        # 10. Target-Aware Multicollinearity Pruning: Keeps the feature with stronger correlation to target
        if self.collinear_threshold < 1.0 and 0 < len(self.numeric_cols) <= 1000:
            corr_matrix = X_proc[self.numeric_cols].corr().abs()
            self.dropped_collinear = []
            
            is_binary_or_reg = (self.task == "regression") or (self.task == "classification" and self.label_encoder and len(self.label_encoder.classes_) == 2)
            
            if y is not None and pd.api.types.is_numeric_dtype(y) and is_binary_or_reg:
                target_corr = X_proc[self.numeric_cols].apply(lambda col: col.corr(y)).abs().fillna(0)
                for i in range(len(corr_matrix.columns)):
                    for j in range(i + 1, len(corr_matrix.columns)):
                        col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                        if (
                            corr_matrix.iloc[i, j] > self.collinear_threshold
                            and col1 not in self.dropped_collinear
                            and col2 not in self.dropped_collinear
                        ):
                            if target_corr.get(col1, 0) >= target_corr.get(col2, 0):
                                self.dropped_collinear.append(col2)
                            else:
                                self.dropped_collinear.append(col1)
            else:
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                self.dropped_collinear = [column for column in upper.columns if any(upper[column] > self.collinear_threshold)]
                
            X_proc = X_proc.drop(columns=self.dropped_collinear)
            self.numeric_cols = [c for c in self.numeric_cols if c not in self.dropped_collinear]
            
        # 11. Feature Scaling (StandardScaler or MinMaxScaler)
        if self.scale_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.scale_method == 'standard':
            self.scaler = StandardScaler()
            
        if self.scaler and len(self.numeric_cols) > 0:
            X_proc[self.numeric_cols] = self.scaler.fit_transform(X_proc[self.numeric_cols])
            
        # Store exact column dtypes for deterministic inference downcasting
        self.column_dtypes = {col: X_proc[col].dtype for col in X_proc.columns}
        self.feature_names = list(X_proc.columns)
        return X_proc

    def _check_input_anomalies(self, X_proc_raw: pd.DataFrame) -> None:
        """
        Production-safety observer detecting schema mismatch and out-of-bounds sensor inputs.
        Appends warnings to self.last_predict_warnings and logs them without raising exceptions.
        """
        try:
            # 1. Schema mismatch check
            if hasattr(self, 'raw_feature_names') and self.raw_feature_names:
                total_raw = len(self.raw_feature_names)
                missing_cols = [c for c in self.raw_feature_names if c not in X_proc_raw.columns]
                missing_count = len(missing_cols)
                missing_ratio = missing_count / total_raw if total_raw > 0 else 0.0

                if missing_ratio > MISSING_SCHEMA_WARN_RATIO:
                    pct = missing_ratio * 100
                    msg = (
                        f"Schema mismatch detected: {missing_count} of {total_raw} raw feature columns "
                        f"({pct:.1f}%) are missing from input data."
                    )
                    self.last_predict_warnings.append(msg)
                    logger.warning(msg)

            # 2. Out-of-bounds sensor values check
            if hasattr(self, 'feature_bounds') and self.feature_bounds:
                total_numeric_cells = 0
                total_oob_cells = 0
                col_oob_counts = {}

                for col, bounds in self.feature_bounds.items():
                    if col in X_proc_raw.columns:
                        series = pd.to_numeric(X_proc_raw[col], errors='coerce').dropna()
                        n_cells = len(series)
                        if n_cells > 0:
                            min_val = bounds.get('min')
                            max_val = bounds.get('max')
                            oob_mask = pd.Series(False, index=series.index)
                            if min_val is not None:
                                oob_mask = oob_mask | (series < min_val)
                            if max_val is not None:
                                oob_mask = oob_mask | (series > max_val)
                            oob_count = int(oob_mask.sum())
                            total_numeric_cells += n_cells
                            total_oob_cells += oob_count
                            if oob_count > 0:
                                col_oob_counts[col] = oob_count

                if total_numeric_cells > 0:
                    overall_oob_ratio = total_oob_cells / total_numeric_cells
                    if overall_oob_ratio > OUT_OF_BOUNDS_WARN_RATIO:
                        pct = overall_oob_ratio * 100
                        top_cols = sorted(col_oob_counts.keys(), key=lambda c: col_oob_counts[c], reverse=True)[:3]
                        msg = (
                            f"Out-of-bounds sensor values detected: {pct:.1f}% of numeric values "
                            f"exceed training bounds. Top offending columns: {top_cols}."
                        )
                        self.last_predict_warnings.append(msg)
                        logger.warning(msg)
        except Exception:  # noqa: BLE001, S110
            pass

    def _preprocess_transform(self, X: pd.DataFrame, apply_bounds_clip: bool = True) -> pd.DataFrame:
        """ 
        Apply the exact schema and statistics learned during `fit()` to new production data.
        Guarantees zero data leakage and ensures robust inference across various input formats 
        (DataFrame, Series, Dict, List, NumPy array, SciPy sparse matrix).
        Set apply_bounds_clip=False for monitoring paths that must observe the
        un-clipped distribution, such as drift detection.
        """
        # Automatically unpack diverse input data structures
        if hasattr(X, "toarray"):
            X = X.toarray()
            
        if isinstance(X, dict):
            X = pd.json_normalize([X])
        elif isinstance(X, list):
            X = pd.json_normalize(X)
        elif isinstance(X, pd.Series):
            X = pd.DataFrame([X])
        elif isinstance(X, np.ndarray):
            cols = self.raw_feature_names if hasattr(self, 'raw_feature_names') and len(self.raw_feature_names) == X.shape[1] else None
            X = pd.DataFrame(X, columns=cols)
            
        X_proc = X.copy()
        
        # Purge nested JSON objects
        for col in list(X_proc.columns):
            if X_proc[col].apply(lambda x: isinstance(x, (list, dict))).any():
                X_proc.drop(columns=[col], inplace=True)

        # The inference path must unwrap categoricals identically or a model trained on unwrapped columns would meet a dtype it never saw.
        for col in list(X_proc.columns):
            if isinstance(X_proc[col].dtype, pd.CategoricalDtype):
                X_proc[col] = X_proc[col].astype(object)
                
        # Remap column names identically with collision-proof sanitizer
        new_cols = []
        seen = set()
        for col in X_proc.columns:
            clean_name = re.sub(r'[^a-zA-Z0-9_\u0E00-\u0E7F]+', '_', str(col)).strip('_')
            if not clean_name:
                clean_name = "feature"
            base_name = clean_name
            counter = 1
            while clean_name in seen:
                clean_name = f"{base_name}_{counter}"
                counter += 1
            seen.add(clean_name)
            new_cols.append(clean_name)
        X_proc.columns = new_cols
        
        self.last_predict_warnings = []
        self._check_input_anomalies(X_proc)
        self.transform_calls += 1
        self.rows_transformed += len(X_proc)
        if self.last_predict_warnings:
            self.warning_events += 1
        
        X_proc.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Ensure all expected raw features exist, filling missing ones with NaN
        if hasattr(self, 'raw_feature_names') and self.raw_feature_names:
            for col in self.raw_feature_names:
                if col not in X_proc.columns:
                    X_proc[col] = np.nan
            X_proc = X_proc[[c for c in self.raw_feature_names if c in X_proc.columns]]
            
        # Identical placeholder handling as training; the token list is fixed, not learned
        X_proc, _ = self._convert_silent_nulls(X_proc)

        # Parse datetime features
        for col in self.datetime_cols:
            if col in X_proc.columns:
                if not pd.api.types.is_datetime64_any_dtype(X_proc[col]):
                    X_proc[col] = pd.to_datetime(X_proc[col], errors='coerce', utc=True)
                X_proc[f"{col}_year"] = X_proc[col].dt.year.fillna(0)
                X_proc[f"{col}_month"] = X_proc[col].dt.month.fillna(0)
                X_proc[f"{col}_day"] = X_proc[col].dt.day.fillna(0)
                X_proc[f"{col}_hour"] = X_proc[col].dt.hour.fillna(0)
                X_proc[f"{col}_dayofweek"] = X_proc[col].dt.dayofweek.fillna(0)
                X_proc = X_proc.drop(columns=[col])
                
        # Auto-convert string numbers to numeric
        for col in self.numeric_cols:
            if col in X_proc.columns and (X_proc[col].dtype == 'object' or str(X_proc[col].dtype) == 'string'):
                X_proc[col] = pd.to_numeric(X_proc[col], errors='coerce')
                
        # Drop high-cardinality columns
        cols_to_drop_hc = [c for c in self.high_cardinality_cols if c in X_proc.columns]
        X_proc = X_proc.drop(columns=cols_to_drop_hc)
        
        # Apply learned median/mode imputation
        for col, val in self.impute_values.items():
            if col in X_proc.columns:
                X_proc[col] = X_proc[col].fillna(val)
                
        # Drop zero-variance columns identified during training
        cols_to_drop_var = [c for c in self.zero_variance_cols if c in X_proc.columns]
        X_proc = X_proc.drop(columns=cols_to_drop_var)
                
        # Clip extreme outliers using training bounds. Skipped for monitoring
        # callers, because clipping would hide the very shift they measure.
        if apply_bounds_clip:
            for col, bounds in self.feature_bounds.items():
                if col in X_proc.columns:
                    X_proc[col] = X_proc[col].clip(lower=bounds['min'], upper=bounds['max'])
                
        # Apply learned ordinal encoding
        if self.ordinal_encoder and len(self.categorical_cols) > 0:
            encoded = self.ordinal_encoder.transform(X_proc[self.categorical_cols].astype(str))
            X_proc[self.categorical_cols] = encoded
            for col in self.categorical_cols:
                X_proc[col] = pd.Categorical(X_proc[col], categories=self.cat_categories[col])
                
        # Drop collinear features identified during training
        cols_to_drop_coll = [c for c in self.dropped_collinear if c in X_proc.columns]
        X_proc = X_proc.drop(columns=cols_to_drop_coll)
        
        # Apply learned numerical scaler
        if self.scaler and len(self.numeric_cols) > 0:
            X_proc[self.numeric_cols] = self.scaler.transform(X_proc[self.numeric_cols])
            
        # Enforce exact learned column dtypes for deterministic execution
        for col, dt in self.column_dtypes.items():
            if col in X_proc.columns:
                try:
                    X_proc[col] = X_proc[col].astype(dt)
                except (ValueError, TypeError):
                    pass
        
        # Enforce exact column sequence to prevent tree-split displacement
        missing_cols = set(self.feature_names) - set(X_proc.columns)
        for col in missing_cols:
            X_proc[col] = 0
            
        return X_proc[self.feature_names]

    def _hash_training_data(self, X_train: pd.DataFrame, y_train: pd.Series) -> str | None:
        """
        Produce a deterministic SHA-256 fingerprint of the training dataset.
        ISO 9001 requires that a deployed model can be traced back to the exact
        data it was trained on; the hash proves identity without storing the data.
        """
        try:
            x_hash = pd.util.hash_pandas_object(pd.DataFrame(X_train), index=False).values
            y_hash = pd.util.hash_pandas_object(pd.Series(y_train).reset_index(drop=True), index=False).values
            digest = hashlib.sha256()
            digest.update(np.asarray(x_hash).tobytes())
            digest.update(np.asarray(y_hash).tobytes())
            return digest.hexdigest()
        except (ValueError, TypeError, AttributeError, MemoryError):
            return None

    def _validate_fit_inputs(self, X_train: Any, y_train: Any) -> None:
        """ Validate input data format, dimensions, and schema consistency prior to fitting. """
        if X_train is None or y_train is None:
            raise ValueError("Training data cannot be None.")

        if len(X_train) == 0:
            raise ValueError("Training dataset is empty. Please provide non-empty training data.")

        X_df = pd.DataFrame(X_train)
        if X_df.shape[1] == 0:
            raise ValueError("No feature columns were supplied in X_train.")

        if len(X_train) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"Insufficient training data: X_train has {len(X_train)} rows, "
                f"but a minimum of {MIN_TRAIN_ROWS} rows is required."
            )

        if hasattr(X_train, "columns"):
            cols = list(X_train.columns)
            seen = set()
            dup_cols = []
            for c in cols:
                if c in seen and c not in dup_cols:
                    dup_cols.append(c)
                seen.add(c)
            if dup_cols:
                logger.warning(f"Duplicate column names detected in training data: {dup_cols}")

        # Duplicate rows leak between cross-validation folds and inflate the score
        try:
            dup_rows = int(pd.DataFrame(X_train).duplicated().sum())
        except (ValueError, TypeError, MemoryError):
            dup_rows = 0
        if dup_rows > 0:
            dup_pct = 100.0 * dup_rows / max(1, len(X_train))
            logger.warning(f"{dup_rows} duplicate training rows ({dup_pct:.1f}%) detected. Identical records can land in both cross-validation folds and inflate the reported score.")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> PotatOptEngine:
        """ 
        Orchestrate the entire automated ML pipeline:
        Preprocessing -> Guardrails Validation -> AutoML Optimization.
        """
        if hasattr(X_train, "toarray"):
            X_train = X_train.toarray()

        self._validate_fit_inputs(X_train, y_train)

        if len(X_train) != len(y_train):
            raise ValueError(f"Length mismatch: X_train has {len(X_train)} rows, y_train has {len(y_train)} rows.")
            
        self._reset_state()
        
        X_train = pd.DataFrame(X_train).reset_index(drop=True)

        # Score the training data before any transformation touches it
        if self.audit_data:
            audit = audit_data_quality(X_train)
            if "error" not in audit:
                self.train_data_quality = {
                    "dqs": audit.get("dqs"),
                    "grade": audit.get("grade"),
                    "verdict": audit.get("verdict"),
                    "top_issues": audit.get("remediation", [])[:5]
                }
        
        if isinstance(y_train, pd.DataFrame):
            y_train = y_train.iloc[:, 0]
        y_train = pd.Series(y_train).reset_index(drop=True)
        
        # Auto-detect task type if set to 'auto'
        if self.task == "auto":
            y_num = pd.to_numeric(y_train, errors='coerce')
            if y_num.notnull().sum() > (0.9 * len(y_train)) and y_num.nunique() > 10:
                self.task = "regression"
                y_train = y_num
            elif y_num.notnull().sum() == len(y_train) and y_num.nunique() <= 2:
                self.task = "classification"
            else:
                self.task = "classification"
        elif self.task in ["regression", "forecasting"]:
            y_train = pd.to_numeric(y_train, errors='coerce')
            
        valid_idx = y_train.dropna().index
        X_train = X_train.loc[valid_idx]
        y_train = y_train.loc[valid_idx]
        
        if len(y_train) == 0:
            raise ValueError("Dataset is empty after dropping NaN targets.")
                
        if self.task == "classification" and y_train.nunique() < 2:
            raise ValueError("Target must have at least 2 distinct classes for classification.")
            
        y_proc = y_train.copy()
        
        # Encode classification labels and identify the minority defect class
        if self.task == "classification":
            self.label_encoder = LabelEncoder()
            y_proc = pd.Series(self.label_encoder.fit_transform(y_proc), index=y_proc.index)
            
            if len(self.label_encoder.classes_) == 2:
                minority_class_encoded = y_proc.value_counts().idxmin()
                self.pos_label_idx = int(minority_class_encoded)
            
        X_proc = self._preprocess_fit_transform(X_train, y=y_proc)

        if X_proc.shape[1] == 0:
            raise ValueError(
                f"All feature columns were removed during preprocessing. "
                f"Likely causes include all-NaN columns ({len(self.all_nan_cols)} dropped: {self.all_nan_cols}), "
                f"high-cardinality identifier columns ({len(self.high_cardinality_cols)} dropped: {self.high_cardinality_cols}), "
                f"zero-variance constant columns ({len(self.zero_variance_cols)} dropped: {self.zero_variance_cols}), "
                f"or nested list/dict columns."
            )

        self.train_rows = len(X_proc)

        # Freeze a compact statistical fingerprint of the training data so drift
        # can still be measured after deployment, when the training set is gone.
        self.train_timestamp = datetime.now(timezone.utc).isoformat()
        self.train_data_hash = self._hash_training_data(X_train, y_train)
        self.train_profile = {}
        for col in self.numeric_cols:
            if col not in X_proc.columns:
                continue
            series = X_proc[col]
            edges, freq = _build_psi_bins(series)
            self.train_profile[col] = {
                "mean": float(series.mean()),
                "std": float(series.std()) if not pd.isna(series.std()) else 0.0,
                "min": float(series.min()),
                "max": float(series.max()),
                "bin_edges": edges,
                "bin_freq": freq
            }
        
        # Imbalance and Anomaly Guardrails
        if self.task == "classification":
            counts = y_proc.value_counts()
            min_class_count = counts.min()
            
            # Multiclass scarcity: Check if we have at least 2 viable classes with >= 5 samples
            if len(counts) > 2:
                viable_classes = counts[counts >= 5].index
                rare_classes = counts[counts < 5].index
                
                if len(viable_classes) >= 2:
                    if len(rare_classes) > 0:
                        logger.warning(f"Dropping rare classes {list(rare_classes)} (< 5 samples) to prevent Cross-Validation crash.")
                        valid_idx = y_proc.isin(viable_classes)
                        X_proc = X_proc.loc[valid_idx]
                        y_proc = y_proc.loc[valid_idx]
                else:
                    # Critical Loophole Closed: When fewer than 2 classes have >= 5 samples across multiclass dataset
                    logger.warning("Extreme Multiclass Scarcity Detected (fewer than 2 classes have >= 5 samples). Auto-switching to Anomaly Detection (Isolation Forest).")
                    self.is_anomaly_model = True
                    self.anomaly_majority_class = int(counts.idxmax())
                    self.anomaly_minority_class = -1
                    
                    X_normal = X_proc[y_proc == self.anomaly_majority_class].copy()
                    for col in self.categorical_cols:
                        if col in X_normal.columns:
                            X_normal[col] = pd.to_numeric(X_normal[col], errors='coerce').fillna(-1).astype(np.int32)
                            
                    self.model = IsolationForest(contamination='auto', random_state=self.random_state, n_jobs=self.n_jobs)
                    self.model.fit(X_normal)
                    self.is_fitted = True
                    return self
            
            # Guardrail: Automatically switch to Unsupervised Anomaly Detection if minority defects < 5 in binary classification
            elif min_class_count < 5 and len(counts) == 2:
                logger.warning(f"Extreme Imbalance Detected ({min_class_count} minority samples). Auto-switching to Anomaly Detection (Isolation Forest).")
                self.is_anomaly_model = True
                self.anomaly_majority_class = int(counts.idxmax())
                self.anomaly_minority_class = int(counts.idxmin())
                
                X_normal = X_proc[y_proc == self.anomaly_majority_class].copy()
                for col in self.categorical_cols:
                    if col in X_normal.columns:
                        X_normal[col] = pd.to_numeric(X_normal[col], errors='coerce').fillna(-1).astype(np.int32)
                        
                self.model = IsolationForest(contamination='auto', random_state=self.random_state, n_jobs=self.n_jobs)
                self.model.fit(X_normal)
                self.is_fitted = True
                return self
        
        fit_kwargs = {}
        # Apply Zero-RAM Cost-Sensitive sample weighting
        if (self.cost_sensitive_weighting or self.apply_smote) and self.task == "classification":
            minority_ratio = y_proc.value_counts(normalize=True).min()
            if minority_ratio < 0.20:
                sample_weights = sklearn.utils.class_weight.compute_sample_weight(class_weight='balanced', y=y_proc)
                fit_kwargs["sample_weight"] = sample_weights

        flaml_task = "classification" if self.task == "classification" else "regression"

        # Mirror FLAML's metric="auto" resolution so validation_loss can be interpreted:
        # binary -> roc_auc (loss = 1 - roc_auc), multiclass -> log_loss (raw loss),
        # regression/forecasting -> r2 (loss = 1 - r2).
        if flaml_task == "classification":
            self.automl_metric = "roc_auc" if int(pd.Series(y_proc).nunique()) == 2 else "log_loss"
        else:
            self.automl_metric = "r2"
        
        # AutoML Hyperparameter Optimization via FLAML
        self.model = _load_automl()()
        settings = {
            "time_budget": self.time_budget,
            "metric": "auto",  
            "task": flaml_task,
            "estimator_list": self.estimators,
            "seed": self.random_state,
            "n_jobs": self.n_jobs,
            # Quiet unless the caller asked otherwise; see __init__.
            "verbose": self.verbose,
        }

        # Time-series rows must never be shuffled during model selection.
        # FLAML maps regression + split_type="auto" to "uniform" (a random
        # shuffle), which would let future rows validate past rows - the exact
        # temporal leakage that split_data(task="forecasting") avoids upstream.
        if self.task == "forecasting":
            settings["split_type"] = "time"
            settings["eval_method"] = "cv"
            
        # FLAML's search emits convergence and deprecation warnings from whichever
        # estimator it happens to try; scoped here so they do not reach the caller.
        with _quiet_dependency_warnings():
            self.model.fit(X_train=X_proc, y_train=y_proc, **settings, **fit_kwargs)
        
        if getattr(self.model, "best_estimator", None) is None:
            raise RuntimeError("AutoML failed to find a valid model. Please inspect input data.")

        # Record the AutoML search outcome so generalisation can be argued for
        self.validation_loss = getattr(self.model, "best_loss", None)
        self.best_config = getattr(self.model, "best_config", None)
            
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """ Generate predictions using learned schemas and optimized decision thresholds. """
        if not self.is_fitted: raise ValueError("Engine is not fitted.")
        X_proc = self._preprocess_transform(X)
        
        # Inference path for Anomaly Detection fallback
        if self.is_anomaly_model:
            X_proc_anom = X_proc.copy()
            for col in self.categorical_cols:
                if col in X_proc_anom.columns:
                    X_proc_anom[col] = pd.to_numeric(X_proc_anom[col], errors='coerce').fillna(-1).astype(np.int32)
                    
            preds = self.model.predict(X_proc_anom) 
            
            # Multiclass sentinel case: minority was set to -1 (bypass inverse_transform to prevent unseen label crash)
            if self.anomaly_minority_class == -1:
                majority_label = (self.label_encoder.inverse_transform([self.anomaly_majority_class])[0] 
                                  if self.label_encoder else self.anomaly_majority_class)
                return np.where(preds == 1, majority_label, "ANOMALY")
                
            y_pred_numeric = np.where(preds == 1, self.anomaly_majority_class, self.anomaly_minority_class)
            if self.label_encoder is not None:
                return self.label_encoder.inverse_transform(np.asarray(y_pred_numeric, dtype=np.int64))
            return y_pred_numeric
        
        y_pred_numeric = None
        # Apply custom probability threshold for binary classification if tuned
        if self.task == "classification" and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_proc)
            if proba.shape[1] == 2 and self.optimal_threshold != 0.5:
                prob_defect = proba[:, self.pos_label_idx]
                is_defect = (prob_defect >= self.optimal_threshold).astype(int)
                y_pred_numeric = np.where(is_defect == 1, self.pos_label_idx, 1 - self.pos_label_idx)
            else:
                y_pred_numeric = self.model.predict(X_proc)
        else:
            y_pred_numeric = self.model.predict(X_proc)
            
        if self.task == "classification" and self.label_encoder is not None:
            return self.label_encoder.inverse_transform(np.asarray(y_pred_numeric, dtype=np.int64))
        return y_pred_numeric
        
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        """ Return prediction probabilities or calibrated anomaly scores. """
        if not self.is_fitted: raise ValueError("Engine is not fitted.")
        if self.task != "classification": raise ValueError("predict_proba is only available for classification.")
        X_proc = self._preprocess_transform(X)
        
        if self.is_anomaly_model:
            X_proc_anom = X_proc.copy()
            for col in self.categorical_cols:
                if col in X_proc_anom.columns:
                    X_proc_anom[col] = pd.to_numeric(X_proc_anom[col], errors='coerce').fillna(-1).astype(np.int32)
                    
            scores = self.model.decision_function(X_proc_anom)
            # Deterministic Sigmoid Calibration:
            # IsolationForest decision_function: < 0 is anomaly, > 0 is normal.
            # P(anomaly) = 1 / (1 + exp(k * score))
            # Guaranteed to return identical, reproducible confidence whether evaluated as a single sample or in a large batch.
            probs_anomaly = 1.0 / (1.0 + np.exp(np.clip(scores * 5.0, -15.0, 15.0)))
            return np.vstack([1.0 - probs_anomaly, probs_anomaly]).T
            
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_proc)
        return None

    @staticmethod
    def _dataset_fingerprint(X: Any, y: Any) -> str | None:
        """
        Cheap, allocation-light identity hash for a (features, target) pair.

        Used only to notice that optimize_threshold and evaluate were handed the
        same rows. Returns None if the data cannot be hashed - a missing
        fingerprint simply disables the warning, it never blocks evaluation.
        """
        try:
            # str/bytes are sized but are not a column of labels; pandas would
            # silently wrap them into a 1-row Series and produce a meaningless hash.
            if isinstance(y, (str, bytes, bytearray)):
                return None
            if not hasattr(y, "__len__") and not hasattr(y, "__array__"):
                return None
            y_series = pd.Series(y).reset_index(drop=True)
            y_bytes = pd.util.hash_pandas_object(y_series, index=False).values.tobytes()
            n_rows, n_cols = (X.shape[0], X.shape[1]) if hasattr(X, "shape") and len(X.shape) == 2 else (len(y_series), 0)
            digest = hashlib.sha1(y_bytes, usedforsecurity=False).hexdigest()[:16]
            return f"{n_rows}x{n_cols}:{digest}"
        except (ValueError, TypeError, AttributeError, KeyError):
            return None

    def optimize_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, cost_scrap: float = 500, cost_fa: float = 150, cost_insp: float = 20) -> float:
        """
        [Financial Layer] Determines the optimal decision threshold that minimizes total Cost of Quality:
        Total Cost = (False Negatives * Scrap Cost) + (False Positives * False Alarm Cost) + (True Positives * Inspection Cost).

        WARNING - methodology: pass a dedicated VALIDATION partition here, not the
        Test partition you will report results on. The threshold is chosen to be
        the cheapest option ON THIS DATA, so reporting evaluate()/
        calculate_cost_of_quality() on the same rows overstates the saving.
        Use split_data_three_way() to obtain a separate validation set; evaluate()
        logs a warning and sets "threshold_leakage_warning" if the two sets match.

        WARNING - interpretation: the threshold this returns is the cheapest CUT on
        the model's score, which holds whether or not the score is a calibrated
        probability. It is not a statement that the machine has that chance of
        failing. Run `check_calibration` on the same validation rows before writing
        the threshold up as a probability, quoting an expected cost per call-out, or
        carrying the number across to a line with a different failure rate.
        """
        if self.task != "classification" or self.is_anomaly_model: 
            return 0.5 
            
        if self.label_encoder is not None and len(self.label_encoder.classes_) != 2:
            return 0.5
            
        if self.label_encoder is not None:
            try:
                y_val_num = self.label_encoder.transform(pd.Series(y_val).reset_index(drop=True))
            except (ValueError, TypeError, KeyError):
                y_val_num = pd.to_numeric(pd.Series(y_val).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values
        else:
            y_val_num = pd.to_numeric(pd.Series(y_val).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values
        
        X_proc = self._preprocess_transform(X_val)
        proba = self.model.predict_proba(X_proc)
        prob_defect = proba[:, self.pos_label_idx] if proba.shape[1] > 1 else proba[:, 0]
        
        min_cost = float('inf')
        for t in np.arange(0.05, 0.95, 0.05):
            is_defect = (prob_defect >= t).astype(int)
            y_pred = np.where(is_defect == 1, self.pos_label_idx, 1 - self.pos_label_idx)
            
            # Lock confusion matrix explicitly to 2x2 to prevent shape crash on test sets with zero defects
            cm = confusion_matrix(y_val_num, y_pred, labels=[0, 1])
            if cm.size == 4:
                if self.pos_label_idx == 1:
                    _tn, fp, fn, tp = cm.ravel()
                else:
                    tp, fn, fp, _tn = cm.ravel()
                    
                cost = (fn * cost_scrap) + (fp * cost_fa) + (tp * cost_insp)
                if cost < min_cost:
                    min_cost = cost
                    self.optimal_threshold = float(t)
        self.threshold_tuning_fingerprint = self._dataset_fingerprint(X_val, y_val)
        return self.optimal_threshold

    def optimize_maintenance_threshold(self, X_val: pd.DataFrame, y_val: pd.Series, cost_breakdown: float = 50000.0, cost_planned: float = 8000.0, cost_inspection: float = 1500.0) -> float:
        """
        Choose the decision threshold that minimises total maintenance cost.

        The same search as `optimize_threshold`, costed as maintenance rather
        than as quality: a missed failure is charged an unplanned breakdown, a
        caught failure an inspection plus a planned repair, and a false alarm an
        inspection. Because a breakdown normally costs several times a planned
        repair, the cheapest threshold here usually sits LOWER than an accuracy-
        driven 0.5 - it is worth several wasted call-outs to avoid one breakdown.

        WARNING - methodology: pass a dedicated VALIDATION partition, not the
        Test partition you will report on. The threshold is chosen to be cheapest
        ON THIS DATA, so reporting on the same rows overstates the saving. Use
        split_data_three_way(); evaluate() sets "threshold_leakage_warning" if
        the two sets turn out to match.
        """
        if self.task != "classification" or self.is_anomaly_model:
            return 0.5

        if self.label_encoder is not None and len(self.label_encoder.classes_) != 2:
            return 0.5

        if self.label_encoder is not None:
            try:
                y_val_num = self.label_encoder.transform(pd.Series(y_val).reset_index(drop=True))
            except (ValueError, TypeError, KeyError):
                y_val_num = pd.to_numeric(pd.Series(y_val).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values
        else:
            y_val_num = pd.to_numeric(pd.Series(y_val).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values

        X_proc = self._preprocess_transform(X_val)
        proba = self.model.predict_proba(X_proc)
        prob_defect = proba[:, self.pos_label_idx] if proba.shape[1] > 1 else proba[:, 0]

        min_cost = float('inf')
        for t in np.arange(0.05, 0.95, 0.05):
            is_defect = (prob_defect >= t).astype(int)
            y_pred = np.where(is_defect == 1, self.pos_label_idx, 1 - self.pos_label_idx)

            cm = confusion_matrix(y_val_num, y_pred, labels=[0, 1])
            if cm.size == 4:
                if self.pos_label_idx == 1:
                    _tn, fp, fn, tp = cm.ravel()
                else:
                    tp, fn, fp, _tn = cm.ravel()

                cost = (fn * cost_breakdown) + (tp * (cost_inspection + cost_planned)) + (fp * cost_inspection)
                if cost < min_cost:
                    min_cost = cost
                    self.optimal_threshold = float(t)
        self.threshold_tuning_fingerprint = self._dataset_fingerprint(X_val, y_val)
        return self.optimal_threshold

    def check_calibration(self, X: pd.DataFrame, y: pd.Series, n_bins: int = CALIBRATION_DEFAULT_BINS) -> dict[str, Any]:
        """
        Measure whether this model's probabilities mean what they say.

        A thin wrapper over the module-level `check_calibration`: it pulls the
        positive-class column out of `predict_proba`, maps `y` onto that same class,
        and hands both over. See that function for what the numbers mean.

        Run it on the VALIDATION partition alongside `optimize_threshold`, on the
        same rows the threshold was chosen from. That is the partition whose
        probabilities the cost argument rests on, and it keeps the Test partition
        untouched for the figures you report.

        An anomaly fallback model always comes back miscalibrated and that is not a
        defect: `predict_proba` there is a fixed sigmoid squashed onto
        IsolationForest's decision function, built to be deterministic and ordered,
        never to estimate a failure rate. The result carries `probability_source`
        so the reason is visible in the output rather than only in this docstring.
        """
        if not self.is_fitted:
            return {"error": "Engine is not fitted."}
        if self.task != "classification":
            return {"error": f"Calibration only applies to classification, task is '{self.task}'."}

        try:
            proba = self.predict_proba(X)
        except (ValueError, TypeError, KeyError) as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if proba is None or np.asarray(proba).ndim != 2 or np.asarray(proba).shape[1] < 2:
            return {"error": "The fitted model does not expose two-class probabilities."}

        proba = np.asarray(proba, dtype=float)
        if self.label_encoder is not None and len(self.label_encoder.classes_) != 2:
            return {"error": f"Calibration is measured for binary problems; this model has {len(self.label_encoder.classes_)} classes."}

        # Map the target onto the same class the probability column refers to, so
        # "positive" means one thing in both halves of the comparison.
        if self.label_encoder is not None:
            try:
                y_num = self.label_encoder.transform(pd.Series(y).reset_index(drop=True))
            except (ValueError, TypeError, KeyError):
                return {"error": "y contains labels the model was not trained on."}
            positive_class = str(self.label_encoder.inverse_transform([self.pos_label_idx])[0])
        else:
            y_num = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce").to_numpy()
            positive_class = str(self.pos_label_idx)

        outcome = (np.asarray(y_num) == self.pos_label_idx).astype(int)
        result = check_calibration(outcome, proba[:, self.pos_label_idx], n_bins=n_bins)
        if "error" in result:
            return result

        result["positive_class"] = positive_class
        result["probability_source"] = "isolation_forest_sigmoid" if self.is_anomaly_model else "model_predict_proba"
        if self.is_anomaly_model:
            result["is_well_calibrated"] = False
            result["interpretation"] = (
                "Anomaly fallback: these scores come from a fixed sigmoid over IsolationForest's "
                "decision function, not from a fitted probability model. Use them to rank machines, "
                "never as a failure rate."
            )
        return result

    def _binary_count_breakdown(self, y_true: Any, y_pred: Any, pos_label: Any) -> dict[str, int]:
        """
        Count TP / FN / FP for the positive (defect) class using string comparison.
        Comparing as strings keeps the count correct whether the labels arrived as
        integers, numpy integers, or original string class names.
        """
        try:
            y_true_s = pd.Series(y_true).reset_index(drop=True).astype(str)
            y_pred_s = pd.Series(y_pred).reset_index(drop=True).astype(str)
            pos_s = str(pos_label)
            tp = int(((y_true_s == pos_s) & (y_pred_s == pos_s)).sum())
            fn = int(((y_true_s == pos_s) & (y_pred_s != pos_s)).sum())
            fp = int(((y_true_s != pos_s) & (y_pred_s == pos_s)).sum())
            return {"tp": tp, "fn": fn, "fp": fp}
        except (ValueError, TypeError, KeyError):
            return {"tp": 0, "fn": 0, "fp": 0}

    def _compute_auc_metrics(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        """
        Compute threshold-independent ranking metrics for binary problems.
        PR-AUC (Average Precision) is the decisive metric for rare-defect
        detection: at a 1% defect rate both accuracy and ROC-AUC stay optimistic
        while PR-AUC collapses toward the defect base rate. Never raises; returns
        None values when the metrics are undefined.
        """
        result = {"roc_auc": None, "pr_auc": None, "defect_base_rate": None}
        try:
            proba = self.predict_proba(X_test)
            if proba is None or proba.ndim != 2 or proba.shape[1] != 2:
                return result

            if self.label_encoder is not None:
                try:
                    y_true = self.label_encoder.transform(pd.Series(y_test).reset_index(drop=True))
                except (ValueError, TypeError, KeyError):
                    y_true = pd.to_numeric(pd.Series(y_test).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values
            else:
                y_true = pd.to_numeric(pd.Series(y_test).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values

            y_true = np.asarray(y_true)

            if self.is_anomaly_model:
                if self.anomaly_minority_class == -1:
                    return result
                prob_defect = proba[:, 1]
                y_true_bin = (y_true == self.anomaly_minority_class).astype(int)
            else:
                prob_defect = proba[:, self.pos_label_idx]
                y_true_bin = (y_true == self.pos_label_idx).astype(int)

            if len(prob_defect) != len(y_true_bin):
                return result
            if len(np.unique(y_true_bin)) < 2:
                return result

            result["roc_auc"] = float(roc_auc_score(y_true_bin, prob_defect))
            result["pr_auc"] = float(average_precision_score(y_true_bin, prob_defect))
            result["defect_base_rate"] = float(np.mean(y_true_bin))
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            pass
        return result

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        """ Evaluate model performance and return clean, UI-ready metrics dictionary. """
        y_pred = self.predict(X_test)

        # Detect the classic methodology slip: tuning the decision threshold on
        # the very rows used to report results. The reported cost saving is then
        # biased in the model's favour.
        threshold_leakage = False
        if self.threshold_tuning_fingerprint is not None:
            current_fingerprint = self._dataset_fingerprint(X_test, y_test)
            if current_fingerprint is not None and current_fingerprint == self.threshold_tuning_fingerprint:
                threshold_leakage = True
                logger.warning(
                    "Threshold was tuned on this same dataset - reported cost/recall is optimistically "
                    "biased. Tune on a separate validation set (see split_data_three_way)."
                )
        best_algo = "IsolationForest" if self.is_anomaly_model else getattr(self.model, "best_estimator", "Custom")
        
        if self.task == "classification":
            is_binary = (self.label_encoder is not None and len(self.label_encoder.classes_) == 2) or (self.label_encoder is None and len(np.unique(y_test)) <= 2)
            labels_list = self.label_encoder.classes_ if self.label_encoder else None
            if labels_list is not None:
                class_labels = to_jsonable(labels_list)
            else:
                class_labels = to_jsonable(np.unique(np.concatenate([np.asarray(y_test).ravel(), np.asarray(y_pred).ravel()])))
            
            if self.is_anomaly_model and self.anomaly_minority_class == -1:
                try:
                    cm = confusion_matrix(y_test, y_pred)
                except (ValueError, TypeError):
                    cm = np.array([[len(y_test), 0], [0, 0]])
                return {
                    "task_type": "multi_class_classification",
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
                    "confusion_matrix": cm.tolist(),
                    "class_labels": class_labels,
                    "best_model_name": best_algo,
                    "note": "Evaluated via Unsupervised Anomaly Detection fallback due to extreme class scarcity.",
                    "n_test_rows": len(pd.Series(y_test))
                }
            
            if is_binary:
                try:
                    cm = confusion_matrix(y_test, y_pred, labels=labels_list)
                except (ValueError, TypeError):
                    cm = confusion_matrix(y_test, y_pred)
                pos_label = self.label_encoder.inverse_transform([self.pos_label_idx])[0] if self.label_encoder else self.pos_label_idx
                auc_metrics = self._compute_auc_metrics(X_test, y_test)
                counts = self._binary_count_breakdown(y_test, y_pred, pos_label)
                recall_ci = wilson_confidence_interval(counts["tp"], counts["tp"] + counts["fn"])
                precision_ci = wilson_confidence_interval(counts["tp"], counts["tp"] + counts["fp"])
                try:
                    mcc_value = float(matthews_corrcoef(pd.Series(y_test).reset_index(drop=True).astype(str), pd.Series(y_pred).reset_index(drop=True).astype(str)))
                except (ValueError, TypeError):
                    mcc_value = None
                return {
                    "task_type": "binary_classification",
                    "threshold_leakage_warning": threshold_leakage,
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0, pos_label=pos_label, labels=labels_list),
                    "recall": recall_score(y_test, y_pred, zero_division=0, pos_label=pos_label, labels=labels_list),
                    "f1": f1_score(y_test, y_pred, zero_division=0, pos_label=pos_label, labels=labels_list),
                    "confusion_matrix": cm.tolist(),
                    "class_labels": class_labels,
                    "best_model_name": best_algo,
                    "threshold_used": self.optimal_threshold,
                    "roc_auc": auc_metrics["roc_auc"],
                    "pr_auc": auc_metrics["pr_auc"],
                    "defect_base_rate": auc_metrics["defect_base_rate"],
                    "mcc": mcc_value,
                    "recall_ci_95": recall_ci,
                    "precision_ci_95": precision_ci,
                    "n_test_rows": len(pd.Series(y_test))
                }
            else:
                try:
                    cm = confusion_matrix(y_test, y_pred, labels=labels_list)
                except (ValueError, TypeError):
                    cm = confusion_matrix(y_test, y_pred)
                try:
                    mcc_value = float(matthews_corrcoef(pd.Series(y_test).reset_index(drop=True).astype(str), pd.Series(y_pred).reset_index(drop=True).astype(str)))
                except (ValueError, TypeError):
                    mcc_value = None
                return {
                    "task_type": "multi_class_classification",
                    "accuracy": accuracy_score(y_test, y_pred),
                    "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0, labels=labels_list),
                    "classification_report": classification_report(y_test, y_pred, output_dict=True, labels=labels_list, zero_division=0),
                    "confusion_matrix": cm.tolist(),
                    "class_labels": class_labels,
                    "best_model_name": best_algo,
                    "mcc": mcc_value,
                    "n_test_rows": len(pd.Series(y_test))
                }
        else:
            y_test_num = pd.to_numeric(pd.Series(y_test).reset_index(drop=True), errors='coerce')
            y_pred_num = pd.to_numeric(pd.Series(y_pred).reset_index(drop=True), errors='coerce')
            valid_mask = y_test_num.notnull() & y_pred_num.notnull()
            y_test_clean = y_test_num[valid_mask]
            y_pred_clean = y_pred_num[valid_mask]
            
            if len(y_test_clean) == 0:
                return {
                    "task_type": "regression_forecasting",
                    "r2": 0.0,
                    "rmse": 0.0,
                    "best_model_name": best_algo,
                    "mae": 0.0,
                    "mape": None,
                    "n_test_rows": 0
                }
                
            # Exclude rows where actual value is zero because MAPE is undefined there
            nonzero_mask = y_test_clean != 0
            if nonzero_mask.any():
                mape_value = float(np.mean(np.abs((y_test_clean[nonzero_mask] - y_pred_clean[nonzero_mask]) / y_test_clean[nonzero_mask])) * 100.0)
            else:
                mape_value = None

            return {
                "task_type": "regression_forecasting",
                "r2": float(r2_score(y_test_clean, y_pred_clean)),
                "rmse": float(np.sqrt(mean_squared_error(y_test_clean, y_pred_clean))),
                "best_model_name": best_algo,
                "mae": float(mean_absolute_error(y_test_clean, y_pred_clean)),
                "mape": mape_value,
                "n_test_rows": len(y_test_clean)
            }
            
    def _binary_confusion(self, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[int, int, int, int] | None:
        """
        Return (tp, fp, fn, tn) for the positive class, or None when the task is
        not a fitted binary classification or the matrix comes back degenerate.
        """
        is_binary = (self.label_encoder is not None and len(self.label_encoder.classes_) == 2) or (self.label_encoder is None and len(np.unique(y_test)) <= 2)
        if self.task != "classification" or not is_binary or self.is_anomaly_model:
            return None

        y_pred = self.predict(X_test)

        if self.label_encoder is not None:
            try:
                y_test_num = self.label_encoder.transform(pd.Series(y_test).reset_index(drop=True))
                y_pred_num = self.label_encoder.transform(pd.Series(y_pred).reset_index(drop=True))
            except (ValueError, TypeError, KeyError):
                y_test_num = pd.to_numeric(pd.Series(y_test).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values
                y_pred_num = pd.to_numeric(pd.Series(y_pred).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values
        else:
            y_test_num = pd.to_numeric(pd.Series(y_test).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values
            y_pred_num = pd.to_numeric(pd.Series(y_pred).reset_index(drop=True), errors='coerce').fillna(0).astype(np.int64).values

        cm = confusion_matrix(y_test_num, y_pred_num, labels=[0, 1])
        if cm.size != 4:
            return None
        if self.pos_label_idx == 1:
            tn, fp, fn, tp = cm.ravel()
        else:
            tp, fn, fp, tn = cm.ravel()
        return int(tp), int(fp), int(fn), int(tn)

    def calculate_cost_of_quality(self, X_test: pd.DataFrame, y_test: pd.Series, cost_scrap: float = 500.0, cost_fa: float = 150.0, cost_insp: float = 20.0, base_det_rate: float = 0.0, base_fp_rate: float = 0.0) -> dict[str, Any]:
        """ Calculate total financial savings compared to manual baseline inspection. """
        is_binary = (self.label_encoder is not None and len(self.label_encoder.classes_) == 2) or (self.label_encoder is None and len(np.unique(y_test)) <= 2)
        if self.task != "classification" or not is_binary or self.is_anomaly_model:
            return {"error": "Cost of Quality is only supported for standard Binary Classification."}
            
        counts = self._binary_confusion(X_test, y_test)
        if counts is None:
            return {"baseline_cost": 0.0, "model_cost": 0.0, "cost_savings": 0.0, "savings_percentage": "0.00%"}
        tp, fp, fn, tn = counts

        total_defects, total_normal = tp + fn, tn + fp
        base_tp = int(total_defects * base_det_rate)
        base_fn = total_defects - base_tp
        base_fp = int(total_normal * base_fp_rate)
        
        base_cost = (base_fn * cost_scrap) + (base_fp * cost_fa) + (base_tp * cost_insp)
        model_cost = (fn * cost_scrap) + (fp * cost_fa) + (tp * cost_insp)
        savings = base_cost - model_cost
        
        return {
            "baseline_cost": float(base_cost),
            "model_cost": float(model_cost),
            "cost_savings": float(savings),
            "savings_percentage": f"{(savings/base_cost*100) if base_cost>0 else 0:.2f}%"
        }

    def calculate_maintenance_cost(self, X_test: pd.DataFrame, y_test: pd.Series, cost_breakdown: float = 50000.0, cost_planned: float = 8000.0, cost_inspection: float = 1500.0) -> dict[str, Any]:
        """
        The predictive-maintenance counterpart of `calculate_cost_of_quality`.

        Same confusion matrix, different question. `calculate_cost_of_quality`
        asks what the defects cost; this asks what the BREAKDOWNS cost, and
        compares the model against running the machines to failure. See
        `calculate_maintenance_savings` for the arithmetic and the reasoning.
        """
        counts = self._binary_confusion(X_test, y_test)
        if counts is None:
            return {"error": "Maintenance cost is only supported for a fitted binary classification model."}
        tp, fp, fn, _tn = counts
        return calculate_maintenance_savings(tp, fp, fn, cost_breakdown=cost_breakdown, cost_planned=cost_planned, cost_inspection=cost_inspection)

    def get_feature_importance(self) -> pd.DataFrame | None:
        """ 
        Extract descending feature importances supporting both Tree-based models 
        (`feature_importances_`) and Linear/SGD models (`coef_`).
        """
        if self.is_anomaly_model or not self.is_fitted:
            return None 
        try:
            estimator = None
            if hasattr(self.model, "model"):
                estimator = getattr(self.model.model, "estimator", self.model.model)
            else:
                estimator = self.model
                
            if hasattr(estimator, "feature_importances_"):
                importances = estimator.feature_importances_
                if len(self.feature_names) == len(importances):
                    return pd.DataFrame({"feature": self.feature_names, "importance": importances}).sort_values(by="importance", ascending=False)
            elif hasattr(estimator, "coef_"):
                coef = estimator.coef_
                if coef.ndim > 1:
                    importances = np.mean(np.abs(coef), axis=0)
                else:
                    importances = np.abs(coef)
                if len(self.feature_names) == len(importances):
                    return pd.DataFrame({"feature": self.feature_names, "importance": importances}).sort_values(by="importance", ascending=False)
            return None
        except (AttributeError, ValueError, TypeError, KeyError):
            return None

    def get_training_report(self) -> dict[str, Any]:
        """
        Report how the champion model was selected during AutoML search.
        Exposes the internal cross-validation loss FLAML minimised, which is the
        evidence that the reported test scores are not the product of over-fitting.
        """
        if not self.is_fitted:
            return {"error": "Engine is not fitted."}

        try:
            # Only roc_auc and r2 are stored by FLAML as (1 - score); log_loss is a raw
            # loss and has no meaningful 1 - loss inversion, so the score stays None there.
            if (self.validation_loss is None
                    or self.is_anomaly_model
                    or self.automl_metric not in ("roc_auc", "r2")):
                validation_score = None
            else:
                validation_score = float(1.0 - self.validation_loss)

            return {
                "potatopt_version": _package_version(),
                "task": self.task,
                "is_anomaly_model": self.is_anomaly_model,
                # Reported because a score without its seed is not reproducible,
                # and a reader cannot tell a stable result from a lucky one.
                "random_state": self.random_state,
                "best_estimator": "IsolationForest" if self.is_anomaly_model else getattr(self.model, "best_estimator", "Unknown"),
                "best_config": self.best_config,
                "validation_loss": None if self.validation_loss is None else float(self.validation_loss),
                "validation_score": validation_score,
                "metric_optimized": self.automl_metric,
                "search_time_budget_sec": int(self.time_budget),
                "estimators_searched": list(self.estimators),
                "train_rows": int(self.train_rows),
                "n_features_used": len(self.feature_names),
                "cost_sensitive_weighting": bool(self.cost_sensitive_weighting),
                "optimal_threshold": float(self.optimal_threshold)
            }
        except (ValueError, TypeError, AttributeError, KeyError):
            return {"error": "Training report unavailable."}

    def detect_drift(self, X_batch: pd.DataFrame) -> dict[str, Any]:
        """
        Compare a live production batch against the frozen training profile.
        This is the DMAIC Control-phase check: it runs on a deployed model with no
        access to the original training set, using the bin frequencies captured
        during fit(). Reports PSI per feature plus the standard-deviation ratio,
        which exposes variance inflation that a mean comparison would miss.
        Never raises; returns an "error" key when it cannot run.
        """
        if not self.is_fitted:
            return {"error": "Engine is not fitted."}
        if not self.train_profile:
            return {"error": "No training profile available. Re-fit the model with this version to enable drift detection."}

        try:
            # Observe the un-clipped distribution so drift is not masked by the guardrail
            X_proc = self._preprocess_transform(X_batch, apply_bounds_clip=False)
            features = {}
            max_psi = None
            drift_detected = False

            for col, profile in self.train_profile.items():
                if col not in X_proc.columns:
                    continue
                batch_series = pd.to_numeric(X_proc[col], errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
                if batch_series.empty:
                    continue

                psi = None
                if profile.get("bin_edges") is not None and profile.get("bin_freq") is not None:
                    psi = _psi_core(profile["bin_freq"], batch_series, profile["bin_edges"])

                train_std = float(profile.get("std") or 0.0)
                batch_std = float(batch_series.std()) if not pd.isna(batch_series.std()) else 0.0
                std_ratio = float(batch_std / train_std) if train_std > 1e-6 else None
                mean_shift_sigma = float(abs(float(batch_series.mean()) - float(profile.get("mean", 0.0))) / train_std) if train_std > 1e-6 else None

                # Both supported scalers are linear, so one ratio converts scaler
                # space back into the operator's original units.
                raw_stats = self.raw_feature_stats.get(col) if self.raw_feature_stats else None
                train_mean_raw = None
                batch_mean_raw = None
                train_std_raw = None
                batch_std_raw = None
                if raw_stats is not None and train_std > 1e-6:
                    train_mean_raw = float(raw_stats.get("mean", 0.0))
                    train_std_raw = float(raw_stats.get("std", 0.0))
                    unit_scale = train_std_raw / train_std
                    batch_mean_raw = train_mean_raw + (float(batch_series.mean()) - float(profile.get("mean", 0.0))) * unit_scale
                    batch_std_raw = batch_std * unit_scale

                if psi is None:
                    severity = "unknown"
                elif psi > PSI_MAJOR_SHIFT:
                    severity = "major"
                elif psi > PSI_MODERATE_SHIFT:
                    severity = "moderate"
                else:
                    severity = "stable"

                if severity == "major":
                    drift_detected = True
                if psi is not None and (max_psi is None or psi > max_psi):
                    max_psi = psi

                features[col] = {
                    "psi": psi,
                    "severity": severity,
                    "train_mean_raw": train_mean_raw,
                    "batch_mean_raw": batch_mean_raw,
                    "train_std_raw": train_std_raw,
                    "batch_std_raw": batch_std_raw,
                    "mean_shift_sigma": mean_shift_sigma,
                    "std_ratio": std_ratio,
                    "train_mean_scaled": float(profile.get("mean", 0.0)),
                    "batch_mean_scaled": float(batch_series.mean())
                }

            if drift_detected:
                recommendation = "Major covariate shift detected. Retrain the model before trusting further predictions."
            elif any(f.get("severity") == "moderate" for f in features.values()):
                recommendation = "Moderate shift detected. Increase monitoring frequency and schedule a retrain review."
            else:
                recommendation = "Input distribution is stable relative to the training data."

            if drift_detected:
                logger.warning(recommendation)
            else:
                logger.info(recommendation)

            return {
                "drift_detected": drift_detected,
                "max_psi": max_psi,
                "n_features_checked": len(features),
                "psi_thresholds": {"moderate": PSI_MODERATE_SHIFT, "major": PSI_MAJOR_SHIFT},
                "units_note": "*_raw values are in original engineering units; *_scaled values are in scaler space.",
                "features": features,
                "recommendation": recommendation,
                "batch_rows": len(X_batch)
            }
        except (ValueError, TypeError, KeyError, AttributeError, IndexError):
            return {"error": "Drift detection failed on this batch."}

    def get_inference_health(self) -> dict[str, Any]:
        """
        Report what the engine has seen since it was fitted.
        Counts every transform performed for prediction, evaluation or monitoring,
        and how many of them raised an input-anomaly warning. A rising warning rate
        is the first sign that the production line has moved away from the data the
        model was trained on.
        """
        if not getattr(self, "is_fitted", False):
            return {"error": "Engine is not fitted."}
        try:
            warning_rate = self.warning_events / self.transform_calls if self.transform_calls else 0.0
            return {
                "transform_calls": int(self.transform_calls),
                "rows_transformed": int(self.rows_transformed),
                "warning_events": int(self.warning_events),
                "warning_rate": float(warning_rate),
                "last_predict_warnings": list(self.last_predict_warnings),
                "silent_nulls_converted_during_fit": int(self.silent_nulls_converted),
                "train_data_quality": self.train_data_quality
            }
        except (ValueError, TypeError, AttributeError, ZeroDivisionError):
            return {"error": "Inference health unavailable."}

    def get_shap_values(self, X_test: pd.DataFrame) -> tuple[Any, Any]:
        """ 
        [XAI Transparency] Tri-layer fallback strategy (TreeExplainer -> LinearExplainer -> Generic Explainer) 
        ensuring model explainability across any AutoML learner.
        """
        self.shap_additivity_relaxed = False
        if self.is_anomaly_model or not self.is_fitted:
            return None, None
            
        X_proc = self._preprocess_transform(X_test)
        
        # Convert categoricals to numeric codes for SHAP compatibility
        candidate_frames = [X_proc]
        if self.categorical_cols:
            X_numeric = X_proc.copy()
            for col in self.categorical_cols:
                if col in X_numeric.columns:
                    X_numeric[col] = pd.to_numeric(X_numeric[col], errors='coerce').fillna(-1)
            candidate_frames.append(X_numeric)
        
        # Safely extract underlying estimator across FLAML wrapper variants
        estimator = None
        if hasattr(self.model, "model"):
            estimator = getattr(self.model.model, "estimator", self.model.model)
        else:
            estimator = self.model
            
        shap_module = _load_shap()
        last_exception = None
        
        # The whole fallback ladder is third-party code, and every rung that fails
        # warns on its way out. Suppression is scoped to the ladder rather than to
        # the importing process; see _quiet_dependency_warnings().
        with _quiet_dependency_warnings():
            for frame in candidate_frames:
                try:
                    explainer = shap_module.TreeExplainer(estimator)
                    return explainer, explainer.shap_values(frame)
                except Exception as exc:  # noqa: BLE001
                    last_exception = exc

            for frame in candidate_frames:
                try:
                    explainer = shap_module.LinearExplainer(estimator, frame)
                    return explainer, explainer.shap_values(frame)
                except Exception as exc:  # noqa: BLE001
                    last_exception = exc

                try:
                    explainer = shap_module.Explainer(estimator, frame)
                    return explainer, explainer(frame)
                except Exception as exc:  # noqa: BLE001
                    last_exception = exc

            try:
                frame = candidate_frames[0]
                explainer = shap_module.TreeExplainer(estimator)
                values = explainer.shap_values(frame, check_additivity=False)
                self.shap_additivity_relaxed = True
                return explainer, values
            except Exception as exc:  # noqa: BLE001
                last_exception = exc


        if last_exception is not None:
            raise last_exception
        return None, None

    def explain_predictions(self, X: pd.DataFrame, top_k: int | None = None, max_rows: int = 1000) -> dict[str, Any]:
        """
        [XAI Transparency] JSON-ready global feature attribution.

        A serialisable view over `get_shap_values()`: the SHAP explainer object is
        dropped and the per-row matrix is reduced to mean absolute attribution per
        feature, ranked high to low. That is the shape a dashboard, an API
        response or a language model can actually consume - the raw matrix is one
        float per row per feature.

        Parameters:
        -----------
        top_k : int or None
            Keep only the N most influential features. None keeps all of them.
        max_rows : int
            Explain at most this many rows, taken from the head of X. SHAP is the
            most expensive step in the pipeline, so this keeps the call inside a
            low-spec RAM budget by default.

        Returns:
        --------
        dict:
            Always JSON-serialisable, and never raises. `available` is False with
            a human-readable `reason` when attribution cannot be produced - the
            anomaly fallback is in use, the engine is unfitted, or SHAP failed on
            this estimator. `additivity_check_relaxed` is True if SHAP's additivity check had to be disabled to produce the attributions, so treat the ranking as indicative rather than exact.
        """
        result = {
            "available": False,
            "reason": None,
            "n_rows_explained": 0,
            "top_k": to_jsonable(top_k),
            "feature_attributions": [],
            "additivity_check_relaxed": False,
        }

        if not self.is_fitted:
            result["reason"] = "Engine is not fitted."
            return result

        if self.is_anomaly_model:
            result["reason"] = "SHAP is unavailable for the IsolationForest anomaly fallback."
            return result

        try:
            X_head = X.head(max_rows) if hasattr(X, "head") else X[:max_rows]
            _explainer, shap_values = self.get_shap_values(X_head)
            if shap_values is None:
                result["reason"] = "SHAP returned no values for this estimator."
                return result

            # Normalise across SHAP's several return shapes: an Explanation
            # object, a list of one array per class, or a plain (rows, features)
            # or (rows, features, classes) array.
            values = getattr(shap_values, "values", shap_values)
            if isinstance(values, list):
                values = np.mean([np.abs(np.asarray(v)) for v in values], axis=0)
            values = np.abs(np.asarray(values, dtype=float))
            if values.ndim == 3:
                values = values.mean(axis=2)
            if values.ndim != 2:
                result["reason"] = f"Unexpected SHAP value shape {values.shape}."
                return result

            mean_abs = np.nanmean(values, axis=0)
            names = list(self.feature_names)
            if len(names) != len(mean_abs):
                names = [f"feature_{i}" for i in range(len(mean_abs))]

            ranked = sorted(
                ({"feature": n, "mean_abs_shap": float(v)} for n, v in zip(names, mean_abs)),
                key=lambda row: row["mean_abs_shap"],
                reverse=True,
            )
            if top_k is not None:
                ranked = ranked[:int(top_k)]

            result["available"] = True
            result["n_rows_explained"] = int(values.shape[0])
            result["feature_attributions"] = to_jsonable(ranked)
            result["additivity_check_relaxed"] = self.shap_additivity_relaxed
            return result
        except Exception as exc:  # noqa: BLE001
            result["reason"] = f"SHAP failed: {type(exc).__name__}: {exc}"
            return result

    def save(self, filepath: str = "potatopt_model.pkl") -> str:
        """
        [ISO 9001 Compliance] Serialize the model alongside a SHA-256 integrity hash
        and a metadata sidecar, so a deployed file can be shown to be the file that
        was released.

        The hash is written in the clear into `<name>_metadata.json`. It is an
        integrity record, not a signature: it detects a corrupted or swapped file,
        and it gives an audit trail a version to point at. It cannot prove who
        produced the file - see `load()` for what that means in practice.
        """
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
            
        joblib.dump(self, filepath)
        
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        model_hash = sha256_hash.hexdigest()
        
        metadata = {
            "model_file": filepath,
            "model_hash_sha256": model_hash, 
            "scikit_learn_version": sklearn.__version__,
            "task": self.task,
            "is_anomaly_model": self.is_anomaly_model,
            "random_state": self.random_state,
            "imputed_columns": list(self.impute_values.keys()),
            "dropped_collinear_columns": self.dropped_collinear,
            "zero_variance_columns": self.zero_variance_cols,
            "high_cardinality_columns": self.high_cardinality_cols,
            "optimal_threshold": self.optimal_threshold,
            "target_classes": [str(c) for c in self.label_encoder.classes_] if self.label_encoder else [],
            "best_estimator": "IsolationForest" if self.is_anomaly_model else getattr(self.model, "best_estimator", "Unknown"),
            "features_used": self.feature_names,
            "potatopt_version": _package_version(),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "trained_at_utc": self.train_timestamp,
            "train_data_sha256": self.train_data_hash,
            "n_train_rows": int(self.train_rows),
            "automl_metric": self.automl_metric,
            "validation_loss": None if self.validation_loss is None else float(self.validation_loss),
            "library_versions": get_library_versions(),
            "drift_profile_features": sorted(self.train_profile.keys()) if self.train_profile else [],
            "train_data_quality": self.train_data_quality
        }
        
        base_path = os.path.splitext(filepath)[0]
        meta_path = f"{base_path}_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        return filepath

    @classmethod
    def load(cls, filepath: str = "potatopt_model.pkl", enforce_security: bool = True) -> PotatOptEngine:
        """
        Load a saved model, checking its SHA-256 against the hash recorded at save time.

        What this check proves, and what it does not: it proves INTEGRITY - the bytes on
        disk are the bytes that were saved - and nothing more. It is not AUTHENTICITY.
        There is no key and no signature; the expected hash sits in plaintext next to the
        model, so anyone able to overwrite `model.pkl` can equally overwrite
        `model_metadata.json` and recompute a matching hash.

        This matters because `joblib` is pickle. Loading a model file executes code inside
        it, before any of this class's code runs, so a hostile file is dangerous no matter
        what this check says. The rule is therefore not "the hash passed, so it is safe" -
        it is: load only files that you or your own pipeline produced, over a channel you
        control. The check is here to catch corruption, a truncated copy, and the wrong
        file being deployed - all of which happen far more often than an attacker.
        """
        if enforce_security:
            base_path = os.path.splitext(filepath)[0]
            meta_path = f"{base_path}_metadata.json"
            if not os.path.exists(meta_path):
                raise FileNotFoundError(
                    f"[SECURITY ALERT] Integrity checking is enabled, but the metadata file '{meta_path}' "
                    f"was not found! Model loading aborted because integrity could not be verified."
                )
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            expected_hash = metadata.get("model_hash_sha256")
            if not expected_hash:
                raise ValueError(f"[SECURITY ALERT] Metadata file '{meta_path}' is missing the 'model_hash_sha256' integrity hash!")
                
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            if sha256_hash.hexdigest() != expected_hash:
                raise RuntimeError(f"[SECURITY ALERT] Model file '{filepath}' does not match the hash recorded when it was saved - it has been corrupted, truncated or replaced! (SHA-256 mismatch)")
        
        return joblib.load(filepath)


# Ergonomic Alias
PotatOpt = PotatOptEngine
