from __future__ import annotations

from typing import Any  # loose return-type annotations for JSON-shaped dicts

import numpy as np  # arrays + math for SPC/EWMA/CUSUM limits, downcasting, anomaly scoring
import pandas as pd  # DataFrame/Series is the data contract for every public function

from ._utils import _finite_float, to_jsonable
from .constants import (
    AUTOCORRELATION_WARN,
    CONTROL_RULES_WESTERN_ELECTRIC,
    CUSUM_DEFAULT_DECISION,
    CUSUM_DEFAULT_SLACK,
    EWMA_DEFAULT_LAMBDA,
    EWMA_DEFAULT_SIGMAS,
    MOVING_RANGE_D2,
)


def calculate_spc_limits(df: pd.DataFrame, sensor_column: str, n_sigmas: float = 3) -> dict[str, Any]:
    """
    Calculate Statistical Process Control (SPC) 3-sigma control limits for a sensor.

    Returns:
    --------
    dict:
        Upper Control Limit (UCL), Lower Control Limit (LCL), and process Mean.
    """
    if df is None or df.empty:
        return {"error": "Dataset is empty or None."}
        
    if sensor_column not in df.columns:
        return {"error": f"Sensor '{sensor_column}' not found."}
        
    s_data = df[sensor_column]
    s_series = s_data.iloc[:, 0] if isinstance(s_data, pd.DataFrame) else s_data
    series = pd.to_numeric(s_series, errors='coerce').dropna()
    
    if series.empty:
        return {"error": f"Sensor '{sensor_column}' contains no valid numeric data."}
        
    mean_val, std_val = series.mean(), series.std()
    if pd.isna(std_val):
        std_val = 0.0
        
    n_sig = float(n_sigmas) if (n_sigmas and float(n_sigmas) > 0) else 3.0
    return {
        "sensor": sensor_column,
        "mean": float(mean_val),
        "ucl": float(mean_val + (n_sig * std_val)),
        "lcl": float(mean_val - (n_sig * std_val))
    }


def _baseline_window(series: np.ndarray, baseline_n: int | None = None) -> np.ndarray:
    """
    Return the Phase I window: the whole series when baseline_n is falsy,
    otherwise the first max(2, int(baseline_n)) points. The centre, the
    spread and the autocorrelation check must all describe the same stretch
    of data, or they contradict each other.
    """
    return series if not baseline_n else series[:max(2, int(baseline_n))]


def _baseline_stats(series: np.ndarray, baseline_n: int | None = None) -> tuple[float, float]:
    """
    Estimate the in-control centre and spread for an individuals control chart.

    Sigma comes from the average MOVING RANGE divided by d2, not from the sample
    standard deviation. On a degrading process the two are not interchangeable: a
    series that trends upward inflates its own standard deviation, which widens
    the control limits and hides the very trend the chart exists to catch. On a
    reference ramp the sample standard deviation was 2.201 against a moving-range
    sigma of 0.222, and the EWMA chart signalled at sample 3 instead of sample 13.

    `baseline_n` restricts both estimates to the first N points - the Phase I
    period, before the process was allowed to wander.

    If the baseline window turns out to be perfectly flat but the wider series is
    not, sigma is re-estimated from the whole series. Only a series that never
    moves at all is left with a zero spread.
    """
    window = _baseline_window(series, baseline_n)
    centre = float(np.mean(window))
    if len(window) < 2:
        return centre, 0.0

    spread = float(np.mean(np.abs(np.diff(window))) / MOVING_RANGE_D2)
    if spread <= 0 and len(series) > 1:
        # A quantised or slow-moving sensor can hold one value for the whole
        # baseline window. Falling back to the full series keeps the chart alive
        # rather than reporting a rising process as "nothing to see".
        spread = float(np.mean(np.abs(np.diff(series))) / MOVING_RANGE_D2)
    return centre, spread


def _lag1_autocorrelation(series: np.ndarray) -> float | None:
    """
    Calculate the lag-1 Pearson autocorrelation of the series.
    
    A high lag-1 autocorrelation indicates the series behaves like a random walk,
    violating the independent-observations assumption of the moving-range sigma. 
    Measured on AI4I 2020 Process temperature [K] (600 points), lag-1 was +0.999,
    moving-range sigma was 0.0489 against a sample standard deviation of 1.4837 (30x),
    and EWMA flagged 573 of 600 points.
    """
    if len(series) < 3:
        return None
        
    x = series[:-1]
    y = series[1:]
    
    if np.var(x) == 0.0 or np.var(y) == 0.0:
        return None
        
    corr = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(corr):
        return None
        
    return corr


def calculate_ewma_chart(values: Any, lambda_weight: float = EWMA_DEFAULT_LAMBDA,
                         n_sigmas: float = EWMA_DEFAULT_SIGMAS, target: float | None = None,
                         sigma: float | None = None, baseline_n: int | None = None) -> dict[str, Any]:
    """
    Exponentially Weighted Moving Average control chart.

    Each point carries a fading memory of everything before it:

        z_i = lambda * x_i + (1 - lambda) * z_(i-1),   z_0 = target

    so a shift too small to break a 3-sigma Shewhart limit still accumulates until
    the EWMA crosses. That is the failure mode of condition monitoring: bearings,
    tools and pumps drift, they do not jump.

    The limits are the exact time-varying form,

        target +/- n_sigmas * sigma * sqrt( (lambda / (2 - lambda)) * (1 - (1 - lambda)^(2i)) )

    which is tighter than the asymptotic version for the first few samples, so an
    early fault is not masked while the chart "warms up".

    Parameters:
    -----------
    values : array-like
        The measurement series, oldest first.
    target, sigma : float or None
        In-control centre and spread. Estimated from the data when omitted - see
        `_baseline_stats` for why sigma comes from the moving range.
    baseline_n : int or None
        Estimate target and sigma from the first N points only. Use this whenever
        the series may already contain the degradation you are looking for.

    Returns:
    --------
    dict:
        JSON-ready. Returns `{"error": ...}` on unusable input rather than raising,
        matching `calculate_spc_limits`. `lag1_autocorrelation` and
        `autocorrelation_warning` describe the baseline window, because a chart
        cannot tell a random-walk sensor from a real trend without a known-good
        stretch to compare against.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if series.size == 0:
        return {"error": "No numeric values supplied."}

    lam, problem = _finite_float(lambda_weight, "lambda_weight")
    if problem:
        return {"error": problem}
    if not 0 < lam <= 1:
        return {"error": f"lambda_weight must be in (0, 1], got {lambda_weight!r}."}

    sigmas, problem = _finite_float(n_sigmas, "n_sigmas")
    if problem:
        return {"error": problem}
    if sigmas <= 0:
        return {"error": f"n_sigmas must be positive, got {n_sigmas!r}."}

    centre, spread = _baseline_stats(series, baseline_n)
    if target is not None:
        centre, problem = _finite_float(target, "target")
        if problem:
            return {"error": problem}
    if sigma is not None:
        spread, problem = _finite_float(sigma, "sigma")
        if problem:
            return {"error": problem}
        if spread < 0:
            return {"error": f"sigma must not be negative, got {sigma!r}."}

    lag1 = _lag1_autocorrelation(_baseline_window(series, baseline_n))
    autocorrelation_warning = None
    if sigma is None and lag1 is not None and abs(lag1) >= AUTOCORRELATION_WARN:
        if baseline_n:
            autocorrelation_warning = (
                f"Lag-1 autocorrelation is {lag1:.3f} inside the baseline window. "
                "Moving-range sigma assumes independent observations, so it is "
                "underestimated here and this chart will over-alarm. Chart subgroup "
                "means or model residuals instead of the raw readings, or pass an "
                "explicit sigma."
            )
        else:
            autocorrelation_warning = (
                f"Lag-1 autocorrelation is {lag1:.3f} across the whole series, which is "
                "also where sigma was estimated. If that memory belongs to the sensor - a "
                "random walk - this chart will over-alarm; if it is the degradation you "
                "are looking for, the signal is real. Re-run with baseline_n over a "
                "known-good period to tell the two apart."
            )

    smoothed = np.empty(series.size, dtype=float)
    previous = centre
    for index, value in enumerate(series):
        previous = lam * value + (1.0 - lam) * previous
        smoothed[index] = previous

    steps = np.arange(1, series.size + 1)
    half_width = sigmas * spread * np.sqrt((lam / (2.0 - lam)) * (1.0 - (1.0 - lam) ** (2 * steps)))
    upper = centre + half_width
    lower = centre - half_width

    if spread <= 0:
        # A flat baseline gives zero-width limits; reporting every wobble as a
        # violation would be noise, so say so instead.
        violations: list[int] = []
        note = "Baseline spread is zero; control limits are degenerate and no violations are reported."
    else:
        violations = [int(i) for i in np.flatnonzero((smoothed > upper) | (smoothed < lower))]
        note = None

    first = violations[0] if violations else None
    direction = None
    if first is not None:
        direction = "increasing" if smoothed[first] > centre else "decreasing"

    return to_jsonable({
        "chart": "ewma",
        "n_points": int(series.size),
        "target": centre,
        "sigma": spread,
        "lag1_autocorrelation": lag1,
        "autocorrelation_warning": autocorrelation_warning,
        "lambda_weight": lam,
        "n_sigmas": sigmas,
        "baseline_n": baseline_n,
        "ewma": smoothed,
        "ucl": upper,
        "lcl": lower,
        "violations": violations,
        "first_violation": first,
        "out_of_control": bool(violations),
        "direction": direction,
        "note": note,
    })


def calculate_cusum_chart(values: Any, target: float | None = None, sigma: float | None = None,
                          slack_k: float = CUSUM_DEFAULT_SLACK,
                          decision_h: float = CUSUM_DEFAULT_DECISION,
                          baseline_n: int | None = None) -> dict[str, Any]:
    """
    Tabular CUSUM control chart.

    Two one-sided sums accumulate deviation from target, each with a slack term
    that keeps ordinary noise from building up:

        SH_i = max(0, SH_(i-1) + (x_i - target) - k * sigma)
        SL_i = max(0, SL_(i-1) + (target - x_i) - k * sigma)

    and the chart signals when either exceeds the decision interval h * sigma.
    Because the sums reset at zero, CUSUM ignores noise indefinitely but responds
    to a sustained shift in a handful of samples - the behaviour wanted when the
    question is "has this machine started to degrade", not "was that one reading
    odd".

    `slack_k=0.5` with `decision_h=5` is the classic pairing for catching a
    sustained one-sigma shift.

    Returns:
    --------
    dict:
        JSON-ready. Returns `{"error": ...}` on unusable input rather than raising.
        `lag1_autocorrelation` and `autocorrelation_warning` describe the baseline
        window, because a chart cannot tell a random-walk sensor from a real trend
        without a known-good stretch to compare against.
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if series.size == 0:
        return {"error": "No numeric values supplied."}

    slack_value, problem = _finite_float(slack_k, "slack_k")
    if problem:
        return {"error": problem}
    if slack_value < 0:
        return {"error": f"slack_k must not be negative, got {slack_k!r}."}

    decision_value, problem = _finite_float(decision_h, "decision_h")
    if problem:
        return {"error": problem}
    if decision_value <= 0:
        return {"error": f"decision_h must be positive, got {decision_h!r}."}

    centre, spread = _baseline_stats(series, baseline_n)
    if target is not None:
        centre, problem = _finite_float(target, "target")
        if problem:
            return {"error": problem}
    if sigma is not None:
        spread, problem = _finite_float(sigma, "sigma")
        if problem:
            return {"error": problem}
        if spread < 0:
            return {"error": f"sigma must not be negative, got {sigma!r}."}

    lag1 = _lag1_autocorrelation(_baseline_window(series, baseline_n))
    autocorrelation_warning = None
    if sigma is None and lag1 is not None and abs(lag1) >= AUTOCORRELATION_WARN:
        if baseline_n:
            autocorrelation_warning = (
                f"Lag-1 autocorrelation is {lag1:.3f} inside the baseline window. "
                "Moving-range sigma assumes independent observations, so it is "
                "underestimated here and this chart will over-alarm. Chart subgroup "
                "means or model residuals instead of the raw readings, or pass an "
                "explicit sigma."
            )
        else:
            autocorrelation_warning = (
                f"Lag-1 autocorrelation is {lag1:.3f} across the whole series, which is "
                "also where sigma was estimated. If that memory belongs to the sensor - a "
                "random walk - this chart will over-alarm; if it is the degradation you "
                "are looking for, the signal is real. Re-run with baseline_n over a "
                "known-good period to tell the two apart."
            )

    slack = slack_value * spread
    interval = decision_value * spread

    high = np.empty(series.size, dtype=float)
    low = np.empty(series.size, dtype=float)
    running_high = 0.0
    running_low = 0.0
    for index, value in enumerate(series):
        running_high = max(0.0, running_high + (value - centre) - slack)
        running_low = max(0.0, running_low + (centre - value) - slack)
        high[index] = running_high
        low[index] = running_low

    if spread <= 0:
        violations_high: list[int] = []
        violations_low: list[int] = []
        note = "Baseline spread is zero; the decision interval is degenerate and no violations are reported."
    else:
        violations_high = [int(i) for i in np.flatnonzero(high > interval)]
        violations_low = [int(i) for i in np.flatnonzero(low > interval)]
        note = None

    candidates = [i for i in (
        violations_high[0] if violations_high else None,
        violations_low[0] if violations_low else None,
    ) if i is not None]
    first = min(candidates) if candidates else None

    direction = None
    if first is not None:
        direction = "increasing" if first in violations_high else "decreasing"

    return to_jsonable({
        "chart": "cusum",
        "n_points": int(series.size),
        "target": centre,
        "sigma": spread,
        "lag1_autocorrelation": lag1,
        "autocorrelation_warning": autocorrelation_warning,
        "slack_k": slack_value,
        "decision_h": decision_value,
        "decision_interval": interval,
        "baseline_n": baseline_n,
        "cusum_high": high,
        "cusum_low": low,
        "violations_high": violations_high,
        "violations_low": violations_low,
        "first_violation": first,
        "out_of_control": bool(violations_high or violations_low),
        "direction": direction,
        "note": note,
    })


_RULE_MIN_POINTS: dict[int, int] = {
    1: 1,
    2: 9,
    3: 6,
    4: 14,
    5: 3,
    6: 5,
    7: 15,
    8: 8,
}


def calculate_control_rules(
    values: Any,
    target: float | None = None,
    sigma: float | None = None,
    baseline_n: int | None = None,
    rules: tuple[int, ...] = CONTROL_RULES_WESTERN_ELECTRIC,
) -> dict[str, Any]:
    """
    Evaluate Statistical Process Control (SPC) rules on a continuous measurement series.

    Implements the eight Nelson rules (Nelson 1984) for detecting special causes,
    trends, shifts, and systematic patterns in individuals charts.

    The classic Western Electric set is the four rules (1, 2, 5, 6), with one honest
    caveat: Western Electric's own fourth rule uses eight points in a row on one side,
    while Nelson's rule 2 uses nine. This implementation adheres strictly to Nelson's
    definition without hybridizing.

    Measured False-Alarm Rates (100 in-control standard-normal points, 4,000 trials):
    --------------------------------------------------------------------------------
    - Rule 1 alone (the 3-sigma chart): 23.4% (theoretical: 1 - (1 - 0.0027)^100 = 23.7%)
    - The four Western Electric rules (1, 2, 5, 6): 59.6%
    - All eight Nelson rules (1 to 8): 74.0%

    Sensitivity is bought with false alarms, and a board that cries wolf gets
    ignored - the same argument the Andon amber state already makes.

    Nelson Rules (1-8):
    -------------------
    1. One point more than 3 sigma from centre (outlier / large shift)
    2. Nine points in a row on the same side of centre (mean shift)
    3. Six points in a row all increasing or all decreasing (trend)
    4. Fourteen points in a row alternating up and down (systematic variation)
    5. Two out of three consecutive points more than 2 sigma from centre, same side
    6. Four out of five consecutive points more than 1 sigma from centre, same side
    7. Fifteen points in a row within 1 sigma of centre, either side (stratification)
    8. Eight points in a row more than 1 sigma from centre, either side (mixture)

    Parameters:
    -----------
    values : array-like
        The measurement series, oldest first.
    target : float or None
        In-control process centre. Estimated from `baseline_n` / series if omitted.
    sigma : float or None
        In-control process spread. Estimated from moving range / d2 if omitted.
    baseline_n : int or None
        Estimate centre and sigma from the first N points (Phase I window).
    rules : tuple of int
        Rule numbers to evaluate (1-8). Defaults to Western Electric (1, 2, 5, 6).

    Returns:
    --------
    dict:
        JSON-ready dict. Never raises on invalid input; returns an error dict.
        - target, sigma, n_points
        - rules_applied: tuple of validated rule numbers
        - violations: {rule_number_as_str: [index, ...]} pointing to the last point
        - any_violation, first_violation_index, first_violation_rule
        - rules_skipped: {rule_number: reason} for series too short to evaluate
        - false_alarm_rate_note: empirical false-alarm rate sentence for the set
    """
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if series.size == 0:
        return {"error": "No numeric values supplied."}
    if series.size < 2:
        return {"error": f"At least 2 points are required to calculate control rules, got {series.size}."}

    if not isinstance(rules, (tuple, list, set)):
        return {"error": f"rules must be a sequence of integers between 1 and 8, got {rules!r}."}

    validated_rules: list[int] = []
    for r in rules:
        try:
            r_int = int(r)
        except (TypeError, ValueError):
            return {"error": f"Rule numbers must be integers between 1 and 8, got {r!r}."}
        if not (1 <= r_int <= 8):
            return {"error": f"Rule number {r_int} is outside valid range (1-8)."}
        if r_int not in validated_rules:
            validated_rules.append(r_int)

    if not validated_rules:
        return {"error": "No valid control rules specified."}

    rules_applied = tuple(validated_rules)

    centre, spread = _baseline_stats(series, baseline_n)
    if target is not None:
        centre, problem = _finite_float(target, "target")
        if problem:
            return {"error": problem}

    if sigma is not None:
        spread, problem = _finite_float(sigma, "sigma")
        if problem:
            return {"error": problem}
        if spread <= 0:
            return {"error": f"sigma must be positive, got {sigma!r}."}

    if spread <= 0:
        return {"error": "Estimated baseline sigma is zero or negative; cannot evaluate control rules."}

    n = int(series.size)
    z = (series - centre) / spread

    violations: dict[str, list[int]] = {}
    rules_skipped: dict[str, str] = {}

    for r in rules_applied:
        min_pts = _RULE_MIN_POINTS[r]
        if n < min_pts:
            rules_skipped[str(r)] = f"Series has {n} points; rule {r} requires at least {min_pts} points."
            continue

        v_list: list[int] = []
        if r == 1:
            v_list = [int(i) for i in np.flatnonzero(np.abs(z) > 3.0)]
        elif r == 2:
            for i in range(8, n):
                w = series[i - 8 : i + 1]
                if np.all(w > centre) or np.all(w < centre):
                    v_list.append(i)
        elif r == 3:
            for i in range(5, n):
                w = series[i - 5 : i + 1]
                d = np.diff(w)
                if np.all(d > 0) or np.all(d < 0):
                    v_list.append(i)
        elif r == 4:
            for i in range(13, n):
                w = series[i - 13 : i + 1]
                d = np.diff(w)
                if np.all(d != 0) and np.all(d[:-1] * d[1:] < 0):
                    v_list.append(i)
        elif r == 5:
            for i in range(2, n):
                w_z = z[i - 2 : i + 1]
                if np.sum(w_z > 2.0) >= 2 or np.sum(w_z < -2.0) >= 2:
                    v_list.append(i)
        elif r == 6:
            for i in range(4, n):
                w_z = z[i - 4 : i + 1]
                if np.sum(w_z > 1.0) >= 4 or np.sum(w_z < -1.0) >= 4:
                    v_list.append(i)
        elif r == 7:
            for i in range(14, n):
                w_z = z[i - 14 : i + 1]
                if np.all(np.abs(w_z) <= 1.0):
                    v_list.append(i)
        elif r == 8:
            for i in range(7, n):
                w_z = z[i - 7 : i + 1]
                if np.all(np.abs(w_z) > 1.0):
                    v_list.append(i)

        violations[str(r)] = v_list

    all_viols: list[tuple[int, int]] = []
    for r_str, idxs in violations.items():
        for idx in idxs:
            all_viols.append((idx, int(r_str)))

    if all_viols:
        all_viols.sort(key=lambda item: (item[0], item[1]))
        first_violation_index: int | None = all_viols[0][0]
        first_violation_rule: int | None = all_viols[0][1]
        any_violation = True
    else:
        first_violation_index = None
        first_violation_rule = None
        any_violation = False

    sorted_applied = tuple(sorted(rules_applied))
    if sorted_applied == (1,):
        false_alarm_rate_note = (
            "Measured false-alarm rate on in-control normal data (100 points, 4000 trials) "
            "for rule 1 alone is 23.4% (theoretical: 23.7%). Sensitivity is bought with false alarms."
        )
    elif sorted_applied == (1, 2, 5, 6):
        false_alarm_rate_note = (
            "Measured false-alarm rate on in-control normal data (100 points, 4000 trials) "
            "for the four Western Electric rules is 59.6%. Sensitivity is bought with false alarms."
        )
    elif sorted_applied == (1, 2, 3, 4, 5, 6, 7, 8):
        false_alarm_rate_note = (
            "Measured false-alarm rate on in-control normal data (100 points, 4000 trials) "
            "for all eight Nelson rules is 74.0%. Sensitivity is bought with false alarms."
        )
    else:
        false_alarm_rate_note = (
            "Combined control rules increase false-alarm rates (measured 23.4% for rule 1 alone, "
            "59.6% for Western Electric, and 74.0% for all 8 Nelson rules on 100 in-control points). "
            "Sensitivity is bought with false alarms."
        )

    return to_jsonable({
        "target": centre,
        "sigma": spread,
        "n_points": n,
        "rules_applied": rules_applied,
        "violations": violations,
        "any_violation": any_violation,
        "first_violation_index": first_violation_index,
        "first_violation_rule": first_violation_rule,
        "rules_skipped": rules_skipped,
        "false_alarm_rate_note": false_alarm_rate_note,
    })

