from __future__ import annotations

from typing import Any  # loose return-type annotations for JSON-shaped dicts

import numpy as np  # arrays + math for PSI bins and histogram frequencies
import pandas as pd  # DataFrame/Series is the data contract for every public function

from .constants import (
    DRIFT_MIN_ROWS,
    DRIFT_NOISE_SIGMAS,
    PSI_DEFAULT_BINS,
    PSI_MAJOR_SHIFT,
    PSI_MAX_CATEGORIES,
)


def check_data_drift(train_df: pd.DataFrame, batch_df: pd.DataFrame, threshold_pct: float = 0.2, min_rows: int = 0, psi_bins: int = PSI_DEFAULT_BINS, include_categorical: bool = False) -> dict[str, Any]:
    """
    Detect statistical data drift by comparing an incoming production batch
    against the training distribution.

    `min_rows` counts NON-NULL readings per column, not rows in the frame. A
    column that is mostly NaN still has plenty of rows, and comparing what little
    is left produces a confident answer from almost no evidence: measured on 200
    rows holding 4 real readings, a batch drawn from the training distribution
    itself raised a false alarm 100% of the time. Columns below the gate are
    listed under "skipped_features" instead of being judged.

    Columns present in training but absent from the batch are also reported
    there. Silence about a sensor that has stopped arriving is the one answer
    this function must never give.

    Set `include_categorical` to also compare label columns through
    `calculate_categorical_psi`. It is off by default so existing callers keep
    their exact behaviour.
    """
    if train_df is None or batch_df is None or train_df.empty or batch_df.empty:
        return {"drift_detected": False, "drifted_features": {}, "skipped_features": {}, "error": "Empty or None dataset."}

    drift_report: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    is_drifted = False
    max_psi = None
    gate = max(0, int(min_rows))
    num_cols = train_df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if col not in batch_df.columns:
            skipped[str(col)] = "missing from the batch"
            continue
        t_data = train_df[col]
        b_data = batch_df[col]
        t_col = t_data.iloc[:, 0] if isinstance(t_data, pd.DataFrame) else t_data
        b_col = b_data.iloc[:, 0] if isinstance(b_data, pd.DataFrame) else b_data

        # Clean infinite values and NaNs
        t_series = pd.to_numeric(t_col, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        b_series = pd.to_numeric(b_col, errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()

        if t_series.empty or b_series.empty:
            skipped[str(col)] = "no usable numeric readings"
            continue
        if len(t_series) < gate or len(b_series) < gate:
            skipped[str(col)] = f"{len(b_series)} batch / {len(t_series)} train readings, below min_rows={gate}"
            continue

        t_mean, b_mean = t_series.mean(), b_series.mean()
        t_std = t_series.std()

        # Robust normalization scale: use std if available, else abs(mean) or 1.0
        scale = t_std if (not pd.isna(t_std) and t_std > 1e-6) else (abs(t_mean) if abs(t_mean) > 1e-6 else 1.0)
        diff = abs(b_mean - t_mean) / scale

        psi_value = calculate_psi(t_series, b_series, n_bins=psi_bins)
        b_std = b_series.std()
        # Variance inflation catches progressive tool wear, which leaves the mean intact
        std_ratio = float(b_std / t_std) if (not pd.isna(t_std) and t_std > 1e-6 and not pd.isna(b_std)) else None

        if psi_value is not None and (max_psi is None or psi_value > max_psi):
            max_psi = psi_value

        psi_triggered = psi_value is not None and psi_value > PSI_MAJOR_SHIFT
        if diff > threshold_pct or psi_triggered:
            is_drifted = True
            drift_report[str(col)] = {
                "kind": "numeric",
                "train_mean": float(t_mean),
                "batch_mean": float(b_mean),
                "drift_magnitude": float(diff),
                "drift_%": float(diff * 100),
                "psi": psi_value,
                "std_ratio": std_ratio,
                "trigger": "mean_shift" if diff > threshold_pct else "psi_shift"
            }

    if include_categorical:
        cat_cols = [c for c in train_df.columns if c not in set(num_cols)]
        for col in cat_cols:
            if col not in batch_df.columns:
                skipped[str(col)] = "missing from the batch"
                continue
            t_data = train_df[col]
            b_data = batch_df[col]
            t_col = t_data.iloc[:, 0] if isinstance(t_data, pd.DataFrame) else t_data
            b_col = b_data.iloc[:, 0] if isinstance(b_data, pd.DataFrame) else b_data
            t_clean = pd.Series(t_col).dropna()
            b_clean = pd.Series(b_col).dropna()
            if len(t_clean) < gate or len(b_clean) < gate:
                skipped[str(col)] = f"{len(b_clean)} batch / {len(t_clean)} train readings, below min_rows={gate}"
                continue
            psi_value = calculate_categorical_psi(t_clean, b_clean)
            if psi_value is None:
                skipped[str(col)] = "not usable as a category (single value, or an identifier)"
                continue
            if max_psi is None or psi_value > max_psi:
                max_psi = psi_value
            if psi_value > PSI_MAJOR_SHIFT:
                is_drifted = True
                drift_report[str(col)] = {
                    "kind": "categorical",
                    "psi": float(psi_value),
                    "n_categories": int(pd.Series(t_clean).astype(str).nunique()),
                    "trigger": "psi_shift",
                }

    return {"drift_detected": is_drifted, "drifted_features": drift_report, "skipped_features": skipped, "max_psi": max_psi}


def _build_psi_bins(train_values: Any, n_bins: int = PSI_DEFAULT_BINS) -> tuple[list[float] | None, list[float] | None]:
    """
    Build quantile bin edges and the training frequency vector used for PSI.
    """
    try:
        clean = pd.to_numeric(pd.Series(train_values), errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean) < n_bins or clean.nunique() < 3:
            return (None, None)
        edges = np.unique(np.quantile(clean.to_numpy(dtype=float), np.linspace(0.0, 1.0, n_bins + 1)))
        if len(edges) < 3:
            return (None, None)
        counts, _ = np.histogram(clean.to_numpy(dtype=float), bins=edges)
        total = counts.sum()
        if total <= 0:
            return (None, None)
        train_freq = counts / float(total)
        return (edges.tolist(), train_freq.tolist())
    except (ValueError, TypeError, IndexError, ZeroDivisionError):
        return (None, None)


def _psi_core(train_freq: Any, batch_values: Any, bin_edges: Any) -> float | None:
    """
    Shared PSI kernel comparing a live batch against stored training bin frequencies.
    """
    try:
        edges = np.asarray(bin_edges, dtype=float)
        t_freq = np.asarray(train_freq, dtype=float)
        if edges.size < 3 or t_freq.size != edges.size - 1:
            return None
        clean = pd.to_numeric(pd.Series(batch_values), errors='coerce').replace([np.inf, -np.inf], np.nan).dropna()
        if clean.empty:
            return None
        # Clip the batch into the training range so out-of-range values fall into the end bins instead of being discarded
        arr = np.clip(clean.to_numpy(dtype=float), edges[0], edges[-1])
        counts, _ = np.histogram(arr, bins=edges)
        total = counts.sum()
        if total <= 0:
            return None
        b_freq = counts / float(total)
        # Replace empty bins with a small floor so the logarithm stays finite
        eps = 1e-4
        t_safe = np.where(t_freq <= 0.0, eps, t_freq)
        b_safe = np.where(b_freq <= 0.0, eps, b_freq)
        psi = float(np.sum((b_safe - t_safe) * np.log(b_safe / t_safe)))
        if not np.isfinite(psi):
            return None
        return psi
    except (ValueError, TypeError, IndexError, ZeroDivisionError, FloatingPointError):
        return None


def calculate_psi(train_values: Any, batch_values: Any, n_bins: int = PSI_DEFAULT_BINS) -> float | None:
    """
    Population Stability Index between a training distribution and a live batch.
    PSI is the standard industrial early-warning statistic for covariate shift.
    It reacts to change anywhere in the distribution, whereas comparing means
    stays silent when the mean holds but the spread widens - the classic
    signature of progressive tool wear.
    Bands: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 major shift.
    Returns None when the training column is too small or too constant to bin.
    """
    edges, freq = _build_psi_bins(train_values, n_bins)
    if edges is None:
        return None
    return _psi_core(freq, batch_values, edges)


def calculate_categorical_psi(train_values: Any, batch_values: Any, max_categories: int = PSI_MAX_CATEGORIES) -> float | None:
    """
    Population Stability Index for a categorical column.

    `calculate_psi` bins numbers by quantile, which cannot work on labels such as
    operator, shift, recipe or lot code. This version compares category
    frequencies directly, so a change in the mix of who ran the machine, or which
    recipe it ran, becomes visible instead of silently dropping out of the report.

    A category that appears in the batch but never in training is a real event -
    a new operator, a new part number - so it is kept and floored at a small
    epsilon rather than discarded.

    Returns None when the column cannot support the statistic: fewer than two
    training categories, or more than `max_categories` distinct values, which
    means the column is an identifier rather than a category.
    Bands match `calculate_psi`: < 0.10 stable, 0.10-0.25 moderate, > 0.25 major.
    """
    try:
        t_clean = pd.Series(train_values).dropna().astype(str)
        b_clean = pd.Series(batch_values).dropna().astype(str)
        if t_clean.empty or b_clean.empty:
            return None
        t_counts = t_clean.value_counts()
        b_counts = b_clean.value_counts()
        if len(t_counts) < 2 or len(t_counts) > max_categories:
            return None
        categories = sorted(set(t_counts.index) | set(b_counts.index))
        if len(categories) > max_categories:
            return None
        t_freq = np.array([t_counts.get(c, 0) for c in categories], dtype=float) / float(len(t_clean))
        b_freq = np.array([b_counts.get(c, 0) for c in categories], dtype=float) / float(len(b_clean))
        # An empty bin is floored so the logarithm stays finite, exactly as in _psi_core
        eps = 1e-4
        t_safe = np.where(t_freq <= 0.0, eps, t_freq)
        b_safe = np.where(b_freq <= 0.0, eps, b_freq)
        psi = float(np.sum((b_safe - t_safe) * np.log(b_safe / t_safe)))
        return psi if np.isfinite(psi) else None
    except (ValueError, TypeError, ZeroDivisionError, AttributeError, FloatingPointError):
        return None


def check_asset_drift(train_df: pd.DataFrame, batch_df: pd.DataFrame, asset_col: str, threshold_pct: float = 0.2, min_rows: int = DRIFT_MIN_ROWS, n_sigma_floor: float = DRIFT_NOISE_SIGMAS, include_categorical: bool = True) -> dict[str, Any]:
    """
    Check drift separately for every machine, instead of pooling them together.

    Machines of the same type still differ from one another: one runs hotter, one
    sits nearer a door. Pooled into a single profile, that between-machine spread
    becomes the yardstick, and two things go wrong at once. A change in the MIX of
    machines reporting - one taken down for maintenance - moves the pooled mean
    and is read as drift although no machine changed. And a real fault on one
    machine is divided by a standard deviation that contains the spread between
    all of them: measured on three machines, a +3 C fault showed as 3.08 per asset
    but only 0.244 pooled, a 12.5x dilution, and the pooled report cannot say
    which machine to send anyone to.

    Small per-asset batches are the trap this creates, so two guards come with it.
    PSI bins are scaled to the batch size, and the mean-shift threshold is raised
    to whichever is larger: the practical threshold asked for, or
    `n_sigma_floor` times the sampling noise sqrt(1/n_batch + 1/n_train). A shift
    must be both big enough to act on and big enough to tell apart from luck.
    The honest cost: a half-sigma shift seen through 30 rows is reported 70% of
    the time rather than 99%, because 30 rows genuinely cannot separate it from
    chance. More rows, or wait for more.

    Every asset in EITHER frame appears in the result with a `status`:
        "checked"            - compared normally
        "insufficient_data"  - fewer than `min_rows` rows on one side
        "unknown_asset"      - in the batch but never seen in training
        "missing_from_batch" - in training but silent now, which usually means a
                               dead sensor or gateway rather than a healthy machine
    That last one is the reason this iterates the union of both frames. An asset
    that has stopped reporting is invisible to any loop over the batch alone.
    """
    if train_df is None or batch_df is None or train_df.empty or batch_df.empty:
        return {"drift_detected": False, "per_asset": {}, "assets_drifted": [], "assets_skipped": {}, "error": "Empty or None dataset."}
    if asset_col not in train_df.columns or asset_col not in batch_df.columns:
        return {"drift_detected": False, "per_asset": {}, "assets_drifted": [], "assets_skipped": {}, "error": f"Asset column {asset_col!r} is missing from one of the frames."}

    try:
        gate = max(1, int(min_rows))
        floor_sigmas = float(n_sigma_floor)
        if not np.isfinite(floor_sigmas) or floor_sigmas < 0:
            return {"drift_detected": False, "per_asset": {}, "assets_drifted": [], "assets_skipped": {}, "error": f"n_sigma_floor must be a finite non-negative number, got {n_sigma_floor!r}."}

        train_groups = {str(k): v for k, v in train_df.groupby(asset_col, observed=True)}
        batch_groups = {str(k): v for k, v in batch_df.groupby(asset_col, observed=True)}

        per_asset: dict[str, Any] = {}
        drifted: list[str] = []
        skipped: dict[str, str] = {}

        for asset in sorted(set(train_groups) | set(batch_groups)):
            t_grp = train_groups.get(asset)
            b_grp = batch_groups.get(asset)

            if b_grp is None:
                skipped[asset] = "present in training but absent from this batch"
                per_asset[asset] = {"status": "missing_from_batch", "rows_train": len(t_grp)}
                continue
            if t_grp is None:
                skipped[asset] = "not present in the training data"
                per_asset[asset] = {"status": "unknown_asset", "rows_batch": len(b_grp)}
                continue

            n_t, n_b = len(t_grp), len(b_grp)
            if n_b < gate or n_t < gate:
                skipped[asset] = f"{n_b} batch / {n_t} train rows, below min_rows={gate}"
                per_asset[asset] = {"status": "insufficient_data", "rows_batch": int(n_b), "rows_train": int(n_t)}
                continue

            noise = float(np.sqrt(1.0 / n_b + 1.0 / n_t))
            effective = max(float(threshold_pct), floor_sigmas * noise)
            bins = int(min(PSI_DEFAULT_BINS, max(2, n_b // 10)))
            report = check_data_drift(
                t_grp.drop(columns=[asset_col]),
                b_grp.drop(columns=[asset_col]),
                threshold_pct=effective,
                min_rows=gate,
                psi_bins=bins,
                include_categorical=include_categorical,
            )
            per_asset[asset] = {
                "status": "checked",
                "rows_train": int(n_t),
                "rows_batch": int(n_b),
                "effective_threshold": float(effective),
                "psi_bins": bins,
                "drift_detected": bool(report["drift_detected"]),
                "drifted_features": report["drifted_features"],
                "skipped_features": report["skipped_features"],
                "max_psi": report["max_psi"],
            }
            if report["drift_detected"]:
                drifted.append(asset)

        return {
            "asset_col": str(asset_col),
            "assets_checked": sum(1 for e in per_asset.values() if e["status"] == "checked"),
            "assets_drifted": sorted(drifted),
            "assets_skipped": skipped,
            "drift_detected": bool(drifted),
            "per_asset": per_asset,
        }
    except (ValueError, TypeError, KeyError, AttributeError, ZeroDivisionError, IndexError):
        return {"drift_detected": False, "per_asset": {}, "assets_drifted": [], "assets_skipped": {}, "error": "Per-asset drift check failed."}
