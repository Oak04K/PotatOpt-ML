"""
Sequence preparation for industrial condition-based maintenance.

Every other module in this library treats a DataFrame as a pile of independent
rows. For condition-based maintenance that is wrong twice over: the rows are a
time series per machine, and degradation is a trend, not a level. A single row
of sensor values cannot express whether a bearing is getting hotter than it used
to be; only a window of that machine's own recent history can. This module turns
a raw sensor log plus a maintenance event log into a modelling table, and it is
the only place in the library allowed to know that rows have an order.

Both functions here return (DataFrame, report_dict) and raise ValueError on
bad input rather than returning {"error": ...}. This deliberately follows
split_data() in data.py, not the JSON-safe convention used elsewhere: a
function that hands back a DataFrame cannot sit behind a tool-calling layer
anyway, so the error convention that helps there buys nothing here. The report
dict on its own is JSON-safe.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._lazy import logger
from ._utils import _require_frame

VALID_WINDOW_STATS = ("mean", "std", "slope", "delta_baseline")


def _to_jsonable_id(val: Any) -> Any:
    """Coerce an asset identifier to a JSON-serializable Python scalar."""
    if hasattr(val, "item"):
        val = val.item()
    if isinstance(val, (int, str)):
        return val
    return str(val)


def build_failure_labels(
    readings: pd.DataFrame,
    events: pd.DataFrame,
    asset_col: str,
    time_col: str,
    horizon: Any,
    event_time_col: str = "failed_at",
    event_asset_col: str | None = None,
    label_col: str = "failure",
    lead_time: Any = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Label sensor readings with binary failure targets based on forward events.

    For every reading row, this identifies the next failure event for the same asset
    occurring strictly after the reading time. The time until that failure is computed,
    and readings within the active prediction window (greater than lead_time and less
    than or equal to horizon) are assigned label 1.

    Parameters
    ----------
    readings : pd.DataFrame
        Time series of sensor measurements for one or more industrial assets.
    events : pd.DataFrame
        Log of recorded failure events.
    asset_col : str
        Column name identifying the asset in readings.
    time_col : str
        Timestamp or numeric sequence column in readings.
    horizon : Any
        Prediction window length. If time is numeric, a positive finite number;
        if time is datetime64, a duration convertible via pd.to_timedelta (e.g. "24h").
    event_time_col : str, default "failed_at"
        Timestamp or numeric sequence column in events.
    event_asset_col : str | None, default None
        Asset column in events. Defaults to asset_col if None.
    label_col : str, default "failure"
        Name of the binary integer target column added to readings.
    lead_time : Any, default 0
        Actionability buffer immediately preceding failure. A warning that arrives
        immediately before breakdown cannot be acted upon because maintenance planners
        require scheduling lead time to book bays, assign crews, and stage spare parts.
        Readings within this buffer (0 < time_to_event <= lead_time) are labelled 0 and
        tallied separately in the report as rows_inside_lead_time, indicating unserviceable
        lead time rather than normal operation.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A tuple of (labelled_frame, report).
        labelled_frame contains all original columns preserved in original order and
        index, with two columns appended: label_col (int8) and time_to_event (float).
        For datetime inputs, time_to_event is expressed as a float number of hours to
        ensure uniform numeric representation and direct JSON serializability.
        report is a JSON-safe dictionary containing summary counts, data quality flags,
        and asset-level breakdowns.
    """
    actual_event_asset = asset_col if event_asset_col is None else event_asset_col

    readings_err = _require_frame(readings, "readings", (asset_col, time_col))
    if readings_err:
        raise ValueError(readings_err)

    events_err = _require_frame(events, "events", (actual_event_asset, event_time_col))
    if events_err:
        raise ValueError(events_err)

    if label_col in readings.columns:
        raise ValueError(f"label_col {label_col!r} already present in readings.columns.")
    if "time_to_event" in readings.columns:
        raise ValueError("time_to_event already present in readings.columns.")

    r_time = readings[time_col]
    e_time = events[event_time_col]

    r_is_num = (
        pd.api.types.is_numeric_dtype(r_time)
        and not pd.api.types.is_bool_dtype(r_time)
    )
    e_is_num = (
        pd.api.types.is_numeric_dtype(e_time)
        and not pd.api.types.is_bool_dtype(e_time)
    )
    r_is_dt = pd.api.types.is_datetime64_any_dtype(r_time)
    e_is_dt = pd.api.types.is_datetime64_any_dtype(e_time)

    if r_is_num and e_is_num:
        is_datetime = False
        if isinstance(horizon, bool):
            raise ValueError(f"horizon must be a positive number, got {horizon!r}.")
        try:
            h_val = float(horizon)
        except (TypeError, ValueError):
            raise ValueError(f"horizon must be a positive number, got {horizon!r}.")
        if not np.isfinite(h_val) or h_val <= 0:
            raise ValueError(f"horizon must be a positive finite number, got {horizon!r}.")

        if isinstance(lead_time, bool):
            raise ValueError(f"lead_time must be a non-negative number, got {lead_time!r}.")
        try:
            lt_val = float(lead_time)
        except (TypeError, ValueError):
            raise ValueError(f"lead_time must be a non-negative number, got {lead_time!r}.")
        if not np.isfinite(lt_val) or lt_val < 0:
            raise ValueError(
                f"lead_time must be a non-negative finite number, got {lead_time!r}."
            )
        if lt_val >= h_val:
            raise ValueError(
                f"lead_time ({lead_time!r}) must be strictly less than horizon ({horizon!r})."
            )

        rep_horizon = str(horizon) if isinstance(horizon, pd.Timedelta) else horizon
        rep_lead_time = str(lead_time) if isinstance(lead_time, pd.Timedelta) else lead_time

    elif r_is_dt and e_is_dt:
        is_datetime = True
        try:
            h_delta = pd.to_timedelta(horizon)
        except Exception as err:  # noqa: BLE001 - any parse failure becomes the documented ValueError
            raise ValueError(
                "horizon must be convertible to a Timedelta for datetime time column, "
                f"got {horizon!r}: {err}"
            ) from None
        if pd.isna(h_delta) or h_delta <= pd.Timedelta(0):
            raise ValueError(f"horizon must be a positive duration, got {horizon!r}.")

        try:
            lt_delta = pd.to_timedelta(lead_time)
        except Exception as err:  # noqa: BLE001 - any parse failure becomes the documented ValueError
            raise ValueError(
                "lead_time must be convertible to a Timedelta for datetime time column, "
                f"got {lead_time!r}: {err}"
            ) from None
        if pd.isna(lt_delta) or lt_delta < pd.Timedelta(0):
            raise ValueError(f"lead_time must be a non-negative duration, got {lead_time!r}.")
        if lt_delta >= h_delta:
            raise ValueError(
                f"lead_time ({lead_time!r}) must be strictly less than horizon ({horizon!r})."
            )

        h_val = h_delta.total_seconds() / 3600.0
        lt_val = lt_delta.total_seconds() / 3600.0

        rep_horizon = str(horizon) if isinstance(horizon, pd.Timedelta) else horizon
        rep_lead_time = str(lead_time) if isinstance(lead_time, pd.Timedelta) else lead_time

    else:
        raise ValueError(
            f"Time column types do not match or are unsupported: "
            f"readings[{time_col!r}] has dtype {r_time.dtype} while "
            f"events[{event_time_col!r}] has dtype {e_time.dtype}."
        )

    if readings[time_col].isna().any():
        raise ValueError(f"readings[{time_col!r}] contains null values.")
    if readings[asset_col].isna().any():
        raise ValueError(f"readings[{asset_col!r}] contains null values.")

    events_clean = events[[actual_event_asset, event_time_col]].dropna().copy()
    events_clean = events_clean.rename(
        columns={actual_event_asset: asset_col, event_time_col: "_event_time_target"}
    )
    if (
        pd.api.types.is_object_dtype(readings[asset_col].dtype)
        or pd.api.types.is_string_dtype(readings[asset_col].dtype)
    ) and (
        pd.api.types.is_object_dtype(events_clean[asset_col].dtype)
        or pd.api.types.is_string_dtype(events_clean[asset_col].dtype)
    ):
        events_clean[asset_col] = events_clean[asset_col].astype(readings[asset_col].dtype)

    events_sorted = events_clean.sort_values(by="_event_time_target", kind="mergesort")

    readings_work = readings.copy()
    readings_work["_potatopt_orig_pos"] = np.arange(len(readings))
    readings_sorted = readings_work.sort_values(by=time_col, kind="mergesort")

    merged = pd.merge_asof(
        readings_sorted,
        events_sorted,
        left_on=time_col,
        right_on="_event_time_target",
        by=asset_col,
        direction="forward",
        allow_exact_matches=False,
    )

    if is_datetime:
        time_diff = merged["_event_time_target"] - merged[time_col]
        tte = time_diff.dt.total_seconds() / 3600.0
    else:
        tte = (merged["_event_time_target"] - merged[time_col]).astype(float)

    is_pos = (tte > lt_val) & (tte <= h_val)
    labels = is_pos.astype(np.int8)
    is_inside_lead = (tte > 0) & (tte <= lt_val)

    merged[label_col] = labels
    merged["time_to_event"] = tte.astype(float)

    merged = merged.sort_values(by="_potatopt_orig_pos", kind="mergesort")
    merged.index = readings.index
    labelled_frame = merged.drop(columns=["_potatopt_orig_pos", "_event_time_target"])

    rows = len(readings)
    positives = int((labelled_frame[label_col] == 1).sum())
    positive_rate = round(float(positives / rows), 6) if rows > 0 else 0.0
    assets = int(readings[asset_col].nunique())

    reading_assets = set(readings[asset_col].unique())
    events_used = int(events_clean[asset_col].isin(reading_assets).sum())

    event_assets = set(events_clean[asset_col].unique())
    no_events_list = sorted(reading_assets - event_assets, key=str)
    if len(no_events_list) > 50:
        rem = len(no_events_list) - 50
        assets_without_events = [_to_jsonable_id(a) for a in no_events_list[:50]] + [
            f"... and {rem} more"
        ]
    else:
        assets_without_events = [_to_jsonable_id(a) for a in no_events_list]

    all_event_assets = set(events[actual_event_asset].dropna().unique())
    no_readings_list = sorted(all_event_assets - reading_assets, key=str)
    if len(no_readings_list) > 50:
        rem = len(no_readings_list) - 50
        events_without_readings = [_to_jsonable_id(a) for a in no_readings_list[:50]] + [
            f"... and {rem} more"
        ]
    else:
        events_without_readings = [_to_jsonable_id(a) for a in no_readings_list]

    censored_rows = int(labelled_frame["time_to_event"].isna().sum())
    rows_inside_lead_time = int(is_inside_lead.sum())

    asset_reading_counts = readings[asset_col].value_counts()
    asset_pos_counts = (
        labelled_frame[labelled_frame[label_col] == 1][asset_col].value_counts()
    )
    asset_ev_counts = events_clean[asset_col].value_counts()

    per_asset_truncated = len(asset_reading_counts) > 200
    selected_assets = (
        asset_reading_counts.index[:200]
        if per_asset_truncated
        else asset_reading_counts.index
    )

    per_asset: dict[str | int, dict[str, int]] = {}
    for a in selected_assets:
        key = _to_jsonable_id(a)
        per_asset[key] = {
            "rows": int(asset_reading_counts.get(a, 0)),
            "positives": int(asset_pos_counts.get(a, 0)),
            "events": int(asset_ev_counts.get(a, 0)),
        }

    if positive_rate < 0.005:
        horizon_warning = (
            f"Positive rate is {positive_rate:.6f} (< 0.005); the horizon ({rep_horizon}) "
            "is likely too short to capture impending failures."
        )
        logger.warning(horizon_warning)
    elif positive_rate > 0.5:
        horizon_warning = (
            f"Positive rate is {positive_rate:.6f} (> 0.5); the horizon ({rep_horizon}) "
            "is likely too long, labelling normal operation as pre-failure."
        )
        logger.warning(horizon_warning)
    else:
        horizon_warning = None

    report: dict[str, Any] = {
        "label_col": label_col,
        "horizon": rep_horizon,
        "lead_time": rep_lead_time,
        "rows": rows,
        "positives": positives,
        "positive_rate": positive_rate,
        "assets": assets,
        "events_used": events_used,
        "assets_without_events": assets_without_events,
        "events_without_readings": events_without_readings,
        "censored_rows": censored_rows,
        "rows_inside_lead_time": rows_inside_lead_time,
        "per_asset": per_asset,
        "horizon_warning": horizon_warning,
    }
    if per_asset_truncated:
        report["per_asset_truncated"] = True

    return labelled_frame, report


def add_window_features(
    readings: pd.DataFrame,
    asset_col: str,
    time_col: str,
    value_cols: Sequence[str] | None = None,
    windows: Sequence[int] = (12, 24),
    stats: Sequence[str] = ("mean", "std", "slope", "delta_baseline"),
    baseline_n: int = 100,
    label_col: str | None = None,
    drop_incomplete: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Generate trailing window statistical features and Phase I baseline deltas per asset.

    To ensure strictly causal features, every statistic for row i is computed exclusively
    from readings of the same asset with timestamps less than or equal to row i. Future rows
    can never influence past features, guaranteeing zero lookahead leakage during training.
    Centred rolling windows are forbidden.

    Parameters
    ----------
    readings : pd.DataFrame
        Sensor measurement log.
    asset_col : str
        Column identifying individual equipment or machines.
    time_col : str
        Chronological timestamp or sequence index column.
    value_cols : Sequence[str] | None, default None
        Numeric sensor columns to process. When None, automatically selects all numeric
        columns excluding asset_col, time_col, label_col (if provided), and any column
        named 'time_to_event'.
    windows : Sequence[int], default (12, 24)
        Window lengths (row counts) for rolling statistics. Each window must be an int >= 2.
    stats : Sequence[str], default ("mean", "std", "slope", "delta_baseline")
        Statistical aggregates to construct:
        - 'mean': Trailing rolling average including the current row.
        - 'std': Trailing rolling standard deviation (unbiased sample standard deviation).
        - 'slope': Trailing mean first difference (series.diff().rolling(w).mean()),
          representing average step-to-step rate of change across the window. This is
          chosen rather than an ordinary least squares regression line because it
          computes in a single rolling pass without solving normal equations at every step.
        - 'delta_baseline': Difference between current reading and that asset's Phase I
          healthy baseline (the mean of its first baseline_n readings in time order).
          Assets operating under differing ambient temperatures or manufacturing tolerances
          exhibit natural offset levels; subtracting the asset's own initial baseline
          normalizes each asset against its own baseline rather than comparing against
          dissimilar peers, consistent with the rationale in drift detection. To prevent
          data leakage, rows at within-asset position less than baseline_n are set to NaN,
          as those readings defined the baseline.
    baseline_n : int, default 100
        Number of initial time-ordered readings per asset used to define the Phase I baseline.
    label_col : str | None, default None
        Target column name, if present in readings, to exclude from feature generation.
    drop_incomplete : bool, default True
        Whether to drop rows containing NaN in any newly generated feature column. When False,
        all rows are preserved, though the leak guard setting rows before baseline_n to NaN
        remains enforced.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        A tuple of (featured_frame, report).
        The returned DataFrame contains all original columns preserved plus the newly
        generated feature columns. The DataFrame is re-sorted by [asset_col, time_col]
        and the original row index is reset (not preserved).
        report is a JSON-safe dictionary detailing added features, row counts, dropped rows,
        skipped assets, and memory overhead in megabytes.
    """
    readings_err = _require_frame(readings, "readings", (asset_col, time_col))
    if readings_err:
        raise ValueError(readings_err)

    if label_col is not None and label_col not in readings.columns:
        raise ValueError(f"label_col {label_col!r} not found in readings.columns.")

    if not stats:
        raise ValueError(
            f"stats must not be empty. Valid stats are: {', '.join(VALID_WINDOW_STATS)}."
        )
    for s in stats:
        if s not in VALID_WINDOW_STATS:
            raise ValueError(
                f"Unknown stat {s!r} in stats. Valid stats are: {', '.join(VALID_WINDOW_STATS)}."
            )

    if not windows:
        raise ValueError("windows must not be empty.")
    for w in windows:
        if isinstance(w, bool) or not isinstance(w, (int, np.integer)) or int(w) < 2:
            raise ValueError(f"Each window must be an integer >= 2, got {w!r}.")

    if (
        isinstance(baseline_n, bool)
        or not isinstance(baseline_n, (int, np.integer))
        or int(baseline_n) < 1
    ):
        raise ValueError(f"baseline_n must be a positive integer, got {baseline_n!r}.")

    if value_cols is None:
        excluded = {asset_col, time_col, "time_to_event"}
        if label_col is not None:
            excluded.add(label_col)
        actual_value_cols = [
            c
            for c in readings.columns
            if c not in excluded
            and pd.api.types.is_numeric_dtype(readings[c])
            and not pd.api.types.is_bool_dtype(readings[c])
        ]
        if not actual_value_cols:
            raise ValueError(
                "No numeric value columns found in readings after excluding asset, time, "
                "label, and time_to_event columns."
            )
    else:
        if not value_cols:
            raise ValueError("value_cols must not be empty.")
        for c in value_cols:
            if c not in readings.columns:
                raise ValueError(f"Column {c!r} in value_cols not found in readings.columns.")
            if not pd.api.types.is_numeric_dtype(readings[c]) or pd.api.types.is_bool_dtype(
                readings[c]
            ):
                raise ValueError(
                    f"Column {c!r} in value_cols must be numeric, got dtype {readings[c].dtype}."
                )
        actual_value_cols = list(value_cols)

    if readings[asset_col].isna().any():
        raise ValueError(f"readings[{asset_col!r}] contains null values.")
    if readings[time_col].isna().any():
        raise ValueError(f"readings[{time_col!r}] contains null values.")

    df = readings.sort_values(by=[asset_col, time_col], kind="mergesort").reset_index(drop=True)

    max_w = max(windows)
    needed_rows = baseline_n + max_w
    asset_counts = df[asset_col].value_counts()

    assets_skipped_all: dict[str | int, str] = {}
    for aid in df[asset_col].unique():
        count = int(asset_counts[aid])
        if count < needed_rows:
            k = _to_jsonable_id(aid)
            assets_skipped_all[k] = (
                f"{count} readings, needs baseline_n={baseline_n} + window={max_w} = {needed_rows}"
            )

    if len(assets_skipped_all) > 50:
        rem = len(assets_skipped_all) - 50
        first_50 = dict(list(assets_skipped_all.items())[:50])
        first_50[f"... and {rem} more"] = f"... and {rem} more"
        assets_skipped = first_50
    else:
        assets_skipped = assets_skipped_all

    group_feature_frames = []
    for _, group in df.groupby(asset_col, sort=False):
        group_dict: dict[str, pd.Series] = {}
        n_rows = len(group)
        pos = np.arange(n_rows)
        for c in actual_value_cols:
            s = group[c]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s_diff = s.diff()
            for w in windows:
                if "mean" in stats:
                    group_dict[f"{c}_w{w}_mean"] = s.rolling(
                        w, min_periods=w, center=False
                    ).mean()
                if "std" in stats:
                    group_dict[f"{c}_w{w}_std"] = s.rolling(
                        w, min_periods=w, center=False
                    ).std()
                if "slope" in stats:
                    group_dict[f"{c}_w{w}_slope"] = s_diff.rolling(
                        w, min_periods=w, center=False
                    ).mean()
            if "delta_baseline" in stats:
                col_name = f"{c}_vs_baseline"
                if n_rows >= baseline_n:
                    b_val = float(s.iloc[:baseline_n].mean())
                    diff_vals = (s - b_val).to_numpy()
                    group_dict[col_name] = pd.Series(
                        np.where(pos < baseline_n, np.nan, diff_vals),
                        index=s.index,
                        dtype=float,
                    )
                else:
                    group_dict[col_name] = pd.Series(np.nan, index=s.index, dtype=float)
        group_feature_frames.append(pd.DataFrame(group_dict, index=group.index))

    features_df = pd.concat(group_feature_frames, axis=0)

    colliding = [col for col in features_df.columns if col in readings.columns]
    if colliding:
        raise ValueError(f"Feature column(s) already exist in readings: {colliding}.")

    combined = pd.concat([df, features_df], axis=1)

    features_added = sorted(features_df.columns.tolist())
    n_features_added = len(features_added)
    has_nan = features_df.isna().any(axis=1)

    if drop_incomplete:
        df_out = combined[~has_nan].reset_index(drop=True)
        incomplete_rows_kept = 0
    else:
        df_out = combined.reset_index(drop=True)
        incomplete_rows_kept = int(has_nan.sum())

    rows_in = len(readings)
    rows_out = len(df_out)
    rows_dropped = int(rows_in - rows_out)

    if features_added and len(df_out) > 0:
        bytes_used = float(
            df_out[features_added].memory_usage(index=False, deep=True).sum()
        )
        memory_mb_added = round(bytes_used / (1024 * 1024), 3)
    else:
        memory_mb_added = 0.0

    drop_rate = (rows_dropped / rows_in) if rows_in > 0 else 0.0
    if drop_rate > 0.30:
        rows_dropped_warning = (
            f"More than 30% of rows were dropped ({rows_dropped} / {rows_in}); "
            f"baseline_n ({baseline_n}) and the largest window ({max_w}) "
            "set the burn-in period required for complete features."
        )
        logger.warning(rows_dropped_warning)
    else:
        rows_dropped_warning = None

    report: dict[str, Any] = {
        "rows_in": rows_in,
        "rows_out": rows_out,
        "rows_dropped": rows_dropped,
        "features_added": features_added,
        "n_features_added": n_features_added,
        "value_cols": actual_value_cols,
        "windows": windows,
        "stats": stats,
        "baseline_n": int(baseline_n),
        "assets": int(df[asset_col].nunique()),
        "assets_skipped": assets_skipped,
        "incomplete_rows_kept": incomplete_rows_kept,
        "memory_mb_added": memory_mb_added,
        "rows_dropped_warning": rows_dropped_warning,
    }

    return df_out, report
