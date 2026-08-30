from __future__ import annotations

from typing import Any  # loose return-type annotations for JSON-shaped dicts

import numpy as np  # arrays + math for SPC/EWMA/CUSUM limits, downcasting, anomaly scoring
import pandas as pd  # DataFrame/Series is the data contract for every public function
from sklearn.model_selection import (
    train_test_split,  # backs split_data() / split_data_three_way()
)

from ._lazy import logger
from ._utils import _is_numeric_series, _is_text_series, _text_columns, to_jsonable
from .constants import (
    DEFAULT_RANDOM_STATE,
    DQS_PRODUCTION_READY,
    DQS_USABLE,
    DQS_WEIGHTS,
    MODIFIED_ZSCORE_THRESHOLD,
    NUMERIC_SENTINELS,
    SILENT_NULL_TOKENS,
)


def inspect_data(df: pd.DataFrame, target_col: str) -> dict[str, Any]:
    """
    Perform a rapid health check on the dataset and recommend the optimal 
    machine learning task and evaluation metric.

    Parameters:
    -----------
    df : pd.DataFrame
        The raw dataset to inspect.
    target_col : str
        The name of the target column.

    Returns:
    --------
    dict:
        A diagnostic summary containing row/column counts, missing values, 
        recommended task ('classification' or 'regression'), and default metric.
    """
    if df is None or df.empty:
        return {"error": "Dataset is empty."}
        
    if target_col not in df.columns:
        return {"error": f"Target column '{target_col}' not found in the dataset."}
        
    total_rows, total_cols = df.shape
    missing_count = int(df.isnull().sum().sum())
    
    # Handle duplicate target columns defensively
    target_data = df[target_col]
    if isinstance(target_data, pd.DataFrame):
        target_series = target_data.iloc[:, 0].dropna()
    else:
        target_series = target_data.dropna()
    
    if target_series.empty:
        return {"error": f"Target column '{target_col}' contains only NaN values."}
        
    audit = audit_data_quality(df, target_col=target_col)
    quality_summary = {
        "dqs": audit.get("dqs"),
        "grade": audit.get("grade"),
        "top_issues": audit.get("remediation", [])[:3]
    }

    unique_targets = target_series.nunique()
    
    # A valid supervised learning problem requires at least 2 distinct target classes/values
    if unique_targets <= 1:
        return {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "missing_values": missing_count,
            "recommended_task": "invalid",
            "recommended_metric": "none",
            "message": f"Target column has only {unique_targets} unique class. Minimum 2 classes required.",
            "data_quality": quality_summary
        }
        
    is_numeric = _is_numeric_series(target_series)
    
    # Check if target values are strings that represent numbers (e.g. "0", "1")
    if not is_numeric:
        num_attempt = pd.to_numeric(target_series, errors='coerce')
        if num_attempt.notnull().sum() == len(target_series):
            is_numeric = True
            unique_targets = num_attempt.nunique()

    # Determine task based on target cardinality and data type
    if unique_targets == 2:
        task = "classification"
        minority_ratio = float(target_series.value_counts(normalize=True).min())
        imbalance_flag = minority_ratio < 0.20
        recommended_metric = "f1"
        msg = f"Binary Classification. Imbalance: {'Yes' if imbalance_flag else 'No'}."
    elif not is_numeric or unique_targets <= 10:
        task = "classification"
        recommended_metric = "macro_f1"
        msg = "Multi-class Classification."
    else:
        task = "regression"
        recommended_metric = "r2"
        msg = "Regression."

    return {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "missing_values": missing_count,
        "recommended_task": task,
        "recommended_metric": recommended_metric,
        "message": msg,
        "data_quality": quality_summary
    }


# `float | int` is redundant to a type checker, but the two are not interchangeable
# here: a float is a proportion and an int is an absolute row count. Spelling both
# out is what tells a reader the int form is intended rather than an accident.
def split_data(df: pd.DataFrame, target_col: str, task: str = "classification", test_size: float | int = 0.2, random_state: int = DEFAULT_RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:  # noqa: PYI041
    """
    Split the dataset into Training and Testing partitions (default 80/20).

    - Classification: Uses stratified sampling to maintain class balance.
    - Forecasting / Time-Series: Disables shuffling to preserve temporal order.
    - test_size: float in (0, 1) for a proportion, or a positive int for an
      absolute row count. Defaults to 0.2 (an 80/20 split).
    - random_state: seed for the shuffle. Which rows land in Test moves the score,
      especially on the small, imbalanced datasets this library is aimed at, so
      change it deliberately and report which one you used. Ignored for
      forecasting, where nothing is shuffled.
    """
    if df is None or df.empty:
        raise ValueError("Dataset is empty.")

    # A float is a proportion; a plain int is an absolute row count. Booleans are
    # rejected explicitly because bool is a subclass of int in Python.
    if isinstance(test_size, bool):
        # ValueError (not TypeError) on purpose: every guard in split_data raises
        # ValueError, so callers can catch one exception type for all bad input.
        raise ValueError(f"test_size must be a float in (0, 1) or a positive int row count, got {test_size!r}.")  # noqa: TRY004
    if isinstance(test_size, (int, np.integer)) and not isinstance(test_size, float):
        if int(test_size) < 1:
            raise ValueError(f"test_size given as a row count must be at least 1, got {test_size!r}.")
        test_size = int(test_size)
    elif isinstance(test_size, (float, np.floating)):
        if not 0 < float(test_size) < 1:
            raise ValueError(f"test_size must be a float strictly between 0 and 1, got {test_size!r}.")
        test_size = float(test_size)
    else:
        raise ValueError(f"test_size must be a float in (0, 1) or a positive int row count, got {test_size!r}.")  # noqa: TRY004
        
    # bool is a subclass of int, so it would otherwise pass silently as seed 0/1.
    if isinstance(random_state, bool) or not isinstance(random_state, (int, np.integer)):
        raise ValueError(f"random_state must be an integer, got {random_state!r}.")  # noqa: TRY004
    random_state = int(random_state)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
        
    # Handle duplicate target columns defensively
    target_data = df[target_col]
    y_raw = target_data.iloc[:, 0] if isinstance(target_data, pd.DataFrame) else target_data
    
    # Drop rows where the target value is missing
    df_clean = df.copy()
    valid_idx = y_raw.dropna().index
    df_clean = df_clean.loc[valid_idx]
    
    if df_clean.empty:
        raise ValueError("Dataset is empty after dropping NaN targets.")
        
    # Separate features (X) and target (y)
    X = df_clean.drop(columns=[target_col])
    target_after_drop = df_clean[target_col]
    y = target_after_drop.iloc[:, 0] if isinstance(target_after_drop, pd.DataFrame) else target_after_drop
    
    # Time-series data must never be shuffled
    if task == "forecasting":
        return train_test_split(X, y, test_size=test_size, shuffle=False)
        
    # Stratify by default for classification unless a class has fewer than 2 samples
    stratify_option = y if task == "classification" else None
    if stratify_option is not None and (y.nunique() < 2 or y.value_counts().min() < 2):
        stratify_option = None
            
    try:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_option)
    except (ValueError, TypeError):
        # Fallback to unstratified split if stratification fails due to edge-case grouping
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=None)


def split_data_three_way(df: pd.DataFrame, target_col: str, task: str = "classification", val_size: float = 0.2, test_size: float = 0.2, random_state: int = DEFAULT_RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the dataset into Training / Validation / Testing partitions.

    The Validation partition exists so that decision thresholds can be tuned
    (see PotatOptEngine.optimize_threshold) on data the model did not train on,
    WITHOUT consuming the Test partition that is used to report final results.
    Tuning a threshold on the same rows that are later reported produces a
    systematically optimistic cost figure.

    - Classification: both splits are stratified to preserve class balance.
    - Forecasting / Time-Series: no shuffling; the partitions stay in
      chronological order (Train -> Validation -> Test).

    Parameters:
    -----------
    val_size : float
        Fraction of the FULL dataset reserved for validation. It is converted to
        an exact row count on the second cut, so the partition sizes land on
        whole rows without floating-point drift.
    test_size : float
        Fraction of the FULL dataset reserved for testing.
        val_size + test_size must stay strictly below 1.0.
    random_state : int
        Seed handed to both cuts. Passing one seed here keeps the three-way split
        reproducible as a unit, which is what a reported figure has to be.

    Returns:
    --------
    tuple:
        (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if not isinstance(val_size, (int, float)) or not 0 < float(val_size) < 1:
        raise ValueError(f"val_size must be a float strictly between 0 and 1, got {val_size!r}.")

    if float(val_size) + float(test_size) >= 1.0:
        raise ValueError(
            f"val_size + test_size must be strictly below 1.0 "
            f"(got {val_size!r} + {test_size!r})."
        )

    # First cut: hold out the Test partition. For forecasting this keeps the
    # most recent rows as Test, which is the only honest layout for time-series.
    X_pool, X_test, y_pool, y_test = split_data(df, target_col, task=task, test_size=test_size, random_state=random_state)

    # Second cut: carve the Validation partition out of what is left. An exact
    # integer row count is used rather than a rescaled fraction, because
    # val_size / (1 - test_size) is not always representable in binary floating
    # point and rounds a whole row across the boundary (e.g. 0.1 / 0.6 on 200
    # rows yielded 21 validation rows instead of 20).
    n_rows_kept = len(X_pool) + len(X_test)
    n_val_rows = round(float(val_size) * n_rows_kept)
    if n_val_rows < 1:
        raise ValueError(
            f"val_size={val_size!r} leaves no validation rows out of {n_rows_kept} usable rows; "
            f"raise val_size or supply more data."
        )
    if n_val_rows >= len(X_pool):
        raise ValueError(
            f"val_size={val_size!r} would consume the entire training pool "
            f"({n_val_rows} of {len(X_pool)} rows); lower val_size or test_size."
        )

    df_pool = X_pool.copy()
    df_pool[target_col] = y_pool
    X_train, X_val, y_train, y_val = split_data(df_pool, target_col, task=task, test_size=n_val_rows, random_state=random_state)

    return X_train, X_val, X_test, y_train, y_val, y_test


def detect_silent_nulls(df: pd.DataFrame) -> dict[str, Any]:
    """
    Find values that mean "missing" but are not stored as NaN.
    A column filled with "N/A" or "-" reports 100% completeness while carrying
    no information, so every downstream quality metric is optimistic until
    these are counted.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    try:
        report = {}
        str_cols = dict.fromkeys(_text_columns(df))
        for col in str_cols:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            normalised = series.astype(str).str.strip().str.lower()
            mask = series.notna() & normalised.isin(SILENT_NULL_TOKENS)
            count = int(mask.sum())
            if count == 0:
                continue
            tokens = {str(k): int(v) for k, v in normalised[mask].value_counts().to_dict().items()}
            report[col] = {
                "count": count,
                "ratio": float(count / len(df)),
                "tokens": tokens,
                "kind": "placeholder_string",
            }

        num_cols = dict.fromkeys(df.select_dtypes(include=[np.number]).columns)
        for col in num_cols:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            mask = series.isin(list(NUMERIC_SENTINELS))
            count = int(mask.sum())
            if count == 0:
                continue
            tokens = {str(k): int(v) for k, v in series[mask].value_counts().to_dict().items()}
            report[col] = {
                "count": count,
                "ratio": float(count / len(df)),
                "tokens": tokens,
                "kind": "numeric_sentinel",
            }

        return report
    except (ValueError, TypeError, KeyError, AttributeError):
        return {}


def detect_outliers(df: pd.DataFrame, method: str = "modified_zscore", threshold: float | None = None) -> dict[str, Any]:
    """
    Flag numeric outliers per column.
    "modified_zscore" is the Iglewicz-Hoaglin robust statistic
    0.6745 * (x - median) / MAD, which does not collapse when the outliers
    themselves inflate the standard deviation. "iqr" is the classic
    1.5 * IQR fence. Reports counts only; it never modifies the data, because
    a physically impossible reading and a legitimate rare extreme look
    identical without domain input.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {}

    if threshold is None:
        threshold = 1.5 if method == "iqr" else MODIFIED_ZSCORE_THRESHOLD

    try:
        report = {}
        num_cols = dict.fromkeys(df.select_dtypes(include=[np.number]).columns)
        for col in num_cols:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            clean = pd.to_numeric(series, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < 3:
                continue

            if method == "iqr":
                q1 = float(clean.quantile(0.25))
                q3 = float(clean.quantile(0.75))
                iqr = q3 - q1
                if iqr <= 0:
                    continue
                lower = float(q1 - threshold * iqr)
                upper = float(q3 + threshold * iqr)
                mask = (clean < lower) | (clean > upper)
                lower_value = lower
                upper_value = upper
            else:
                median = float(clean.median())
                mad = float((clean - median).abs().median())
                if mad > 0:
                    scores = 0.6745 * (clean - median) / mad
                else:
                    # MAD collapses when more than half the values are identical;
                    # fall back to the mean absolute deviation form of the statistic.
                    mean_ad = float((clean - median).abs().mean())
                    if mean_ad <= 0:
                        continue
                    scores = (clean - median) / (1.253314 * mean_ad)
                mask = scores.abs() > threshold
                flagged = clean[mask]
                lower_value = float(flagged.min()) if not flagged.empty else None
                upper_value = float(flagged.max()) if not flagged.empty else None

            count = int(mask.sum())
            if count == 0:
                continue

            report[col] = {
                "count": count,
                "ratio": float(count / len(clean)),
                "method": method,
                "threshold": float(threshold),
                "flagged_min": lower_value,
                "flagged_max": upper_value,
            }

        return report
    except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError):
        return {}


def audit_data_quality(df: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    """
    Score dataset health on five weighted dimensions and return a remediation plan.
    Completeness 30%, Consistency 25%, Validity 20%, Uniqueness 15%, Timeliness 10%.
    Run this before fit(): a model trained on a red-grade dataset produces
    confident numbers that cannot be defended. Grades follow the standard
    bands - 85+ production ready, 65-84 usable with documented caveats,
    below 65 remediation required.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {"error": "Dataset is empty or invalid."}

    try:
        n_rows, n_cols = df.shape
        issues = []

        # Completeness
        silent = detect_silent_nulls(df)
        explicit_nulls = int(df.isnull().sum().sum())
        silent_nulls = int(sum(v["count"] for v in silent.values()))
        total_cells = n_rows * n_cols
        null_ratio = (explicit_nulls + silent_nulls) / total_cells if total_cells else 0.0
        completeness = 100.0 * (1.0 - min(1.0, null_ratio))

        for col in dict.fromkeys(df.columns):
            col_data = df[col]
            col_series = col_data.iloc[:, 0] if isinstance(col_data, pd.DataFrame) else col_data
            col_nulls = int(col_series.isnull().sum())
            col_silent = int(silent.get(col, {}).get("count", 0))
            col_ratio = (col_nulls + col_silent) / n_rows if n_rows else 0.0
            if col_ratio > 0:
                if col_ratio < 0.01:
                    sev = "low"
                    act = "Drop the affected rows, or impute with median/mode."
                elif col_ratio < 0.10:
                    sev = "medium"
                    act = "Impute and add a binary <col>_was_null indicator column."
                elif col_ratio < 0.30:
                    sev = "high"
                    act = "Impute cautiously and investigate the upstream root cause."
                else:
                    sev = "critical"
                    act = "Do not impute blindly. Send for domain review or drop the column."
                issues.append({
                    "dimension": "completeness",
                    "severity": sev,
                    "column": col,
                    "detail": f"{col_ratio * 100:.1f}% missing",
                    "action": act,
                })

        for col, s_info in silent.items():
            token_list = ", ".join(f"'{k}'" for k in s_info.get("tokens", {}))
            issues.append({
                "dimension": "completeness",
                "severity": "high",
                "column": col,
                "detail": f"Placeholder tokens found: {token_list}",
                "action": "Convert these placeholder values to NaN before training.",
            })

        # Consistency
        mixed = 0
        str_cols = dict.fromkeys(_text_columns(df))
        for col in str_cols:
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            parsed = pd.to_numeric(series, errors='coerce')
            non_null_cnt = int(series.notnull().sum())
            if non_null_cnt > 0:
                parsed_cnt = int(parsed.notnull().sum())
                ratio = parsed_cnt / non_null_cnt
                if 0.0 < ratio < 1.0:
                    mixed += 1
                    issues.append({
                        "dimension": "consistency",
                        "severity": "medium",
                        "column": col,
                        "detail": f"{ratio * 100:.1f}% of non-null values parse as numeric (mixed types)",
                        "action": "Split or coerce the column to a single type before training.",
                    })
        consistency = 100.0 * (1.0 - mixed / n_cols) if n_cols else 100.0

        # Validity
        outliers = detect_outliers(df)
        numeric_cells = int(df.select_dtypes(include=[np.number]).notnull().sum().sum())
        flagged = int(sum(v["count"] for v in outliers.values()))
        outlier_ratio = flagged / numeric_cells if numeric_cells else 0.0
        validity = 100.0 * (1.0 - min(1.0, outlier_ratio))

        for col, v in outliers.items():
            if v["ratio"] > 0.05:
                issues.append({
                    "dimension": "validity",
                    "severity": "medium",
                    "column": col,
                    "detail": f"{v['ratio'] * 100:.1f}% of values flagged as outliers (modified z-score)",
                    "action": "Confirm with the process owner whether these are sensor faults or legitimate extremes.",
                })

        # Uniqueness
        dup_rows = int(df.duplicated().sum())
        dup_ratio = dup_rows / n_rows if n_rows else 0.0
        uniqueness = 100.0 * (1.0 - min(1.0, dup_ratio))

        if dup_rows > 0:
            issues.append({
                "dimension": "uniqueness",
                "severity": "high" if dup_ratio > 0.05 else "medium",
                "column": None,
                "detail": f"{dup_rows} duplicate rows ({dup_ratio * 100:.1f}%)",
                "action": "Confirm the uniqueness key with the data owner, then de-duplicate keeping the most recent record.",
            })

        col_counts = pd.Series(df.columns).value_counts()
        dup_cols = col_counts[col_counts > 1].index.tolist()
        if dup_cols:
            issues.append({
                "dimension": "uniqueness",
                "severity": "high",
                "column": None,
                "detail": f"Duplicate column names found: {', '.join(str(c) for c in dup_cols)}",
                "action": "Rename or drop the duplicated column names.",
            })

        # Timeliness
        time_cols = []
        for col in dict.fromkeys(df.columns):
            series = df[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                parsed_dt = pd.to_datetime(series, errors='coerce', utc=True)
                time_cols.append((col, parsed_dt))
            elif _is_text_series(series):
                col_lower = str(col).lower()
                if "date" in col_lower or "time" in col_lower:
                    non_null_cnt = int(series.notnull().sum())
                    if non_null_cnt > 0:
                        parsed_dt = pd.to_datetime(series, errors='coerce', utc=True)
                        valid_cnt = int(parsed_dt.notnull().sum())
                        if (valid_cnt / non_null_cnt) > 0.80:
                            time_cols.append((col, parsed_dt))

        if not time_cols:
            timeliness = None
        else:
            now_utc = pd.Timestamp.now(tz="UTC")
            min_utc = pd.Timestamp("1970-01-01", tz="UTC")
            total_time_invalid = 0
            total_time_count = 0
            for col, parsed_dt in time_cols:
                valid_timestamps = parsed_dt.dropna()
                col_total = len(valid_timestamps)
                total_time_count += col_total
                col_invalid = int(((valid_timestamps > now_utc) | (valid_timestamps < min_utc)).sum())
                total_time_invalid += col_invalid
                if col_invalid > 0:
                    issues.append({
                        "dimension": "timeliness",
                        "severity": "high",
                        "column": col,
                        "detail": f"{col_invalid} timestamps outside the plausible range",
                        "action": "Check the source system clock and timezone handling.",
                    })
            timeliness = 100.0 * (1.0 - total_time_invalid / total_time_count) if total_time_count else 100.0

        # Scores
        scores = {
            "completeness": completeness,
            "consistency": consistency,
            "validity": validity,
            "uniqueness": uniqueness,
            "timeliness": timeliness,
        }

        active_weights = {k: DQS_WEIGHTS[k] for k, v in scores.items() if v is not None}
        total_weight = sum(active_weights.values())
        normalised_weights = {k: w / total_weight for k, w in active_weights.items()} if total_weight > 0 else {}

        dqs = round(sum(scores[k] * normalised_weights[k] for k in normalised_weights), 1)

        if dqs >= DQS_PRODUCTION_READY:
            grade = "production_ready"
            verdict = f"DQS {dqs}/100 - production ready."
        elif dqs >= DQS_USABLE:
            grade = "usable_with_caveats"
            verdict = f"DQS {dqs}/100 - usable with documented caveats."
        else:
            grade = "remediation_required"
            verdict = f"DQS {dqs}/100 - remediation required before use."

        # Target check
        if target_col is not None and target_col in df.columns:
            target_series = df[target_col]
            if isinstance(target_series, pd.DataFrame):
                target_series = target_series.iloc[:, 0]
            n_missing = int(target_series.isnull().sum())
            if n_missing > 0:
                issues.append({
                    "dimension": "completeness",
                    "severity": "critical",
                    "column": target_col,
                    "detail": f"{n_missing} missing target values",
                    "action": "Drop these rows; they cannot be used for supervised training.",
                })

        severity_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        issues.sort(key=lambda x: severity_rank.get(x.get("severity", "low"), 4))

        remediation = []
        seen = set()
        for issue in issues:
            col_label = issue["column"] if issue["column"] is not None else "dataset"
            entry = f"[{issue['severity'].upper()}] {col_label}: {issue['action']}"
            if entry not in seen:
                seen.add(entry)
                remediation.append(entry)
                if len(remediation) == 10:
                    break

        if grade == "remediation_required":
            logger.warning(verdict)
        else:
            logger.info(verdict)

        return {
            "dqs": dqs,
            "grade": grade,
            "verdict": verdict,
            "total_rows": int(n_rows),
            "total_columns": int(n_cols),
            "dimensions": {
                name: {
                    "score": round(float(scores[name]), 1),
                    "weight": float(normalised_weights[name]),
                }
                for name in normalised_weights
            },
            "silent_nulls": silent,
            "outliers": outliers,
            "duplicate_rows": dup_rows,
            "issues": issues,
            "remediation": remediation,
        }
    except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError, IndexError):
        return {"error": "Data quality audit failed."}


def calculate_correlations(
    df: pd.DataFrame,
    method: str = "pearson",
    min_abs: float = 0.7,
    max_columns: int = 30,
) -> dict[str, Any]:
    """
    Compute pairwise feature correlations, flag collinear pairs, and audit excluded columns.

    Two sensors that move together carry one piece of information. When a model
    receives both collinear features, it splits the credit between them,
    making the importance ranking harder to interpret and inflating model
    variance. `PotatOptEngine` already prunes collinear features during `fit()`,
    and this function exposes what it will prune and why before training starts.

    When the number of eligible numeric columns exceeds `max_columns`, columns
    with the most non-null values are retained to avoid unreadable high-dimensional
    matrices.

    Parameters:
    -----------
    df : pd.DataFrame
        The dataset containing features to correlate.
    method : str, default="pearson"
        Correlation method: "pearson", "spearman", or "kendall".
    min_abs : float, default=0.7
        Minimum absolute correlation coefficient to flag a pair in `strong_pairs`.
    max_columns : int, default=30
        Maximum number of numeric columns to correlate.

    Returns:
    --------
    dict:
        JSON-ready diagnostic dictionary containing:
        - "method": the correlation method used ("pearson", "spearman", or "kendall").
        - "columns": list of correlated column names in matrix order.
        - "matrix": 2D list of correlation coefficients, with None for non-finite entries.
        - "strong_pairs": list of `{"a": ..., "b": ..., "correlation": ...}` dicts for
          unique pairs with |r| >= min_abs, sorted descending by absolute correlation.
        - "n_rows": number of rows in `df`.
        - "skipped_columns": mapping of excluded column names to the reason excluded
          (non-numeric, constant column, or fewer than two non-null values).
        - "note": present only when `max_columns` truncated the columns, detailing how
          many columns were kept.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "df must be a pandas DataFrame."}
    if df.empty:
        return {"error": "Dataset is empty."}

    if not isinstance(method, str) or method.lower() not in ("pearson", "spearman", "kendall"):
        return {"error": f"Invalid correlation method '{method}'. Valid methods are 'pearson', 'spearman', 'kendall'."}
    method_str = method.lower()

    if isinstance(min_abs, bool) or not isinstance(min_abs, (int, float, np.floating, np.integer)):
        min_abs_val = 0.7
    else:
        min_abs_val = float(min_abs)
        if not np.isfinite(min_abs_val) or min_abs_val < 0:
            min_abs_val = 0.7

    if isinstance(max_columns, bool) or not isinstance(max_columns, (int, np.integer)):
        max_cols = 30
    else:
        max_cols = int(max_columns)
        if max_cols < 1:
            max_cols = 30

    skipped_columns: dict[str, str] = {}
    candidate_columns: list[str] = []

    for col in dict.fromkeys(df.columns):
        series = df[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        if not _is_numeric_series(series) or pd.api.types.is_bool_dtype(series):
            skipped_columns[str(col)] = "Not numeric."
            continue

        clean = pd.to_numeric(series, errors="coerce").dropna()
        if len(clean) < 2:
            skipped_columns[str(col)] = "Fewer than two non-null values."
            continue

        if clean.nunique() <= 1:
            skipped_columns[str(col)] = "Constant column (variance is zero)."
            continue

        candidate_columns.append(col)

    note: str | None = None
    if len(candidate_columns) > max_cols:
        total_eligible = len(candidate_columns)
        sorted_by_count = sorted(
            candidate_columns,
            key=lambda c: int(df[c].iloc[:, 0].notna().sum() if isinstance(df[c], pd.DataFrame) else df[c].notna().sum()),
            reverse=True,
        )
        kept_set = set(sorted_by_count[:max_cols])
        kept_columns = [c for c in candidate_columns if c in kept_set]
        note = f"Matrix truncated from {total_eligible} to {max_cols} columns with the most non-null values."
    else:
        kept_columns = list(candidate_columns)

    matrix: list[list[float | None]] = []
    strong_pairs: list[dict[str, Any]] = []

    if kept_columns:
        corr_df = df[kept_columns].corr(method=method_str)
        for i in range(len(kept_columns)):
            row_vals: list[float | None] = []
            for j in range(len(kept_columns)):
                val = corr_df.iloc[i, j]
                if pd.isna(val) or not np.isfinite(val):
                    row_vals.append(None)
                else:
                    row_vals.append(float(val))
            matrix.append(row_vals)

        for i in range(len(kept_columns)):
            for j in range(i + 1, len(kept_columns)):
                val = corr_df.iloc[i, j]
                if not pd.isna(val) and np.isfinite(val):
                    c_val = float(val)
                    if abs(c_val) >= min_abs_val:
                        strong_pairs.append({
                            "a": str(kept_columns[i]),
                            "b": str(kept_columns[j]),
                            "correlation": c_val,
                        })

        strong_pairs.sort(key=lambda p: abs(p["correlation"]), reverse=True)

    result: dict[str, Any] = {
        "method": method_str,
        "columns": [str(c) for c in kept_columns],
        "matrix": matrix,
        "strong_pairs": strong_pairs,
        "n_rows": len(df),
        "skipped_columns": skipped_columns,
    }
    if note is not None:
        result["note"] = note

    return to_jsonable(result)
