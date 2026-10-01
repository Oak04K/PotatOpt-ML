from __future__ import annotations

import math
from typing import Any  # loose return-type annotations for JSON-shaped dicts

import numpy as np  # arrays + math for SPC/EWMA/CUSUM limits, downcasting, anomaly scoring
import pandas as pd  # DataFrame/Series is the data contract for every public function
from scipy.optimize import (  # Weibull shape root; age-replacement optimum
    brentq,
    minimize_scalar,
)
from scipy.special import (  # Weibull MTTF; incomplete gamma for age replacement
    gamma,
    gammainc,
)
from scipy.stats import (  # Wilson interval; hazard fit; Crow-AMSAA; Poisson forecast
    chi2,
    norm,
    poisson,
    weibull_min,
)

from ._utils import _is_numeric_series, _require_frame
from .constants import (
    CROW_AMSAA_MIN_FAILURES,
    DEFAULT_RANDOM_STATE,
    FORECAST_INTERVAL_DEFAULT,
    OEE_WORLD_CLASS,
    PARETO_CUTOFF,
    PM_MIN_SAVING_FRACTION,
    WEIBULL_BOOTSTRAP_DEFAULT,
    WEIBULL_CI_CONFIDENCE,
    WEIBULL_MIN_INTERVALS,
)


def wilson_confidence_interval(successes: int, trials: int, confidence: float = 0.95) -> dict[str, Any]:
    """
    Wilson score interval for a binomial proportion.
    Used to report detection rate (recall) and precision with uncertainty
    bounds. Preferred over the normal approximation because industrial defect
    counts are small: with 3 defects out of 30 the normal interval extends
    below zero, while the Wilson interval always stays inside [0, 1].
    """
    try:
        conf = float(confidence)
        succ = int(successes)
        n = int(trials)

        if n <= 0 or succ < 0 or succ > n or not (0.0 < conf < 1.0):
            return {"point": None, "lower": None, "upper": None, "n": 0, "confidence": conf}

        p = succ / n
        z = norm.ppf(1.0 - (1.0 - conf) / 2.0)
        denominator = 1.0 + (z ** 2) / n
        centre = (p + (z ** 2) / (2.0 * n)) / denominator
        half_width = (z / denominator) * np.sqrt(p * (1.0 - p) / n + (z ** 2) / (4.0 * (n ** 2)))

        return {
            "point": float(p),
            "lower": float(max(0.0, centre - half_width)),
            "upper": float(min(1.0, centre + half_width)),
            "n": int(n),
            "confidence": float(conf),
        }
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        try:
            fallback_conf = float(confidence)
        except (ValueError, TypeError):
            fallback_conf = 0.95
        return {"point": None, "lower": None, "upper": None, "n": 0, "confidence": fallback_conf}


def calculate_maintenance_savings(true_positives: int, false_positives: int, false_negatives: int, cost_breakdown: float = 50000.0, cost_planned: float = 8000.0, cost_inspection: float = 1500.0) -> dict[str, Any]:
    """
    Turn a confusion matrix into the maintenance business case.

    The comparison is against RUN TO FAILURE, the honest baseline for condition
    monitoring: with no model at all, every failure that happens becomes an
    unplanned breakdown.

        run to failure = (tp + fn) * cost_breakdown
        with the model = fn * cost_breakdown              a failure that was missed
                       + tp * (cost_inspection + cost_planned)   caught, repaired on plan
                       + fp * cost_inspection            somebody looked, found nothing

    A false positive is charged the inspection only, not a part: an engineer goes
    and looks before replacing anything.

    Reporting `breakdown_avoidance_rate` next to `cost_savings` is deliberate,
    because the two disagree in the case that matters. A model that flags every
    machine reaches an avoidance rate of 1.00 and still loses money - measured on
    20 real failures among 1000 machines, flagging everything scores perfect
    recall and a saving of -660,000, because 980 pointless call-outs cost more
    than the breakdowns they prevented. Recall cannot show that; cost can.

    Returns an error dict rather than raising, so it is safe to call from a
    tool-calling layer with whatever arguments turn up.
    """
    try:
        tp, fp, fn = int(true_positives), int(false_positives), int(false_negatives)
        if tp < 0 or fp < 0 or fn < 0:
            return {"error": "Counts must not be negative."}
        costs = {"cost_breakdown": cost_breakdown, "cost_planned": cost_planned, "cost_inspection": cost_inspection}
        for name, value in costs.items():
            number = float(value)
            if not np.isfinite(number) or number < 0:
                return {"error": f"{name} must be a finite non-negative number, got {value!r}."}
        c_break, c_plan, c_insp = float(cost_breakdown), float(cost_planned), float(cost_inspection)
    except (TypeError, ValueError):
        return {"error": "Counts and costs must be numbers."}

    total_failures = tp + fn
    run_to_failure = total_failures * c_break
    with_model = (fn * c_break) + (tp * (c_insp + c_plan)) + (fp * c_insp)
    savings = run_to_failure - with_model

    return {
        "run_to_failure_cost": float(run_to_failure),
        "predictive_cost": float(with_model),
        "cost_savings": float(savings),
        "savings_percentage": float(savings / run_to_failure * 100.0) if run_to_failure > 0 else 0.0,
        "breakdowns_avoided": tp,
        "unplanned_breakdowns": fn,
        "breakdown_avoidance_rate": float(tp / total_failures) if total_failures > 0 else None,
        "wasted_inspections": fp,
        "cost_assumptions": {"cost_breakdown": c_break, "cost_planned": c_plan, "cost_inspection": c_insp},
    }


def calculate_mtbf(work_orders: pd.DataFrame, operating_hours: float, wo_type_col: str = "wo_type", breakdown_types: tuple[str, ...] = ("breakdown",)) -> dict[str, Any]:
    """
    Mean Time Between Failures over a period of running time.

        MTBF = operating_hours / number of breakdowns

    `operating_hours` is time the equipment was actually **running**, not calendar
    time. The two definitions disagree, and calendar time is the flattering one:
    it counts hours the machine sat idle or unscheduled as though they were
    trouble-free service, so a rarely used machine scores well for doing nothing.

    Only rows whose `wo_type_col` value is in `breakdown_types` raise the failure
    count. Planned repairs, inspections and predictive call-outs are maintenance
    events, not failures - counting them would penalise the very behaviour a
    condition-based programme is trying to produce.

    Zero breakdowns returns `mtbf_hours: None` rather than an error: a machine
    that has not failed is a real answer, and the caller must be able to tell it
    apart from a malformed call.

    `failure_rate_per_hour` is `1 / MTBF`, which is the instantaneous failure
    rate only when the hazard rate does not change with age. A wearing machine has
    a rising hazard, so this figure then understates the risk of failing soon;
    `calculate_weibull` tests that assumption on the times between failures.
    """
    problem = _require_frame(work_orders, "work_orders", (wo_type_col,))
    if problem:
        return {"error": problem}
    try:
        hours = float(operating_hours)
    except (TypeError, ValueError):
        return {"error": "operating_hours must be a number."}
    if not np.isfinite(hours) or hours <= 0:
        return {"error": f"operating_hours must be a finite positive number, got {operating_hours!r}."}

    wanted = {str(t) for t in breakdown_types}
    breakdowns = int(work_orders[wo_type_col].astype(str).isin(wanted).sum())
    mtbf = hours / breakdowns if breakdowns > 0 else None

    return {
        "mtbf_hours": float(mtbf) if mtbf is not None else None,
        "breakdowns": breakdowns,
        "operating_hours": hours,
        "failure_rate_per_hour": float(1.0 / mtbf) if mtbf else None,
        "failure_rate_assumes_constant_hazard": True,
        "constant_hazard_note": (
            "failure_rate_per_hour is 1 / MTBF, which is the instantaneous failure "
            "rate only when the hazard rate does not change with age. A wearing machine has "
            "a rising hazard, so this figure then understates the risk of failing soon; "
            "calculate_weibull tests that assumption on the times between failures."
        ),
    }


def calculate_mttr(work_orders: pd.DataFrame, reported_col: str = "reported_at", started_col: str = "started_at", finished_col: str = "finished_at", wo_type_col: str = "wo_type", breakdown_types: tuple[str, ...] = ("breakdown",)) -> dict[str, Any]:
    """
    Split a repair into the three durations most systems report as one number.

        wait   (MTTA) = started_at  - reported_at
        repair (MTTR) = finished_at - started_at
        down   (MDT)  = finished_at - reported_at    and MDT = MTTA + MTTR

    They are separated because they have different owners and different fixes.
    Waiting is the maintenance organisation's scheduling and spares problem;
    repair time is the technician and the job itself; downtime is what production
    actually loses. A single blended figure hides which of the three to work on.

    A row missing any timestamp, or holding a negative duration (finished before
    started - a data-entry error), is excluded from the averages and counted in
    `rows_excluded` rather than quietly averaged in.
    """
    problem = _require_frame(work_orders, "work_orders", (reported_col, started_col, finished_col, wo_type_col))
    if problem:
        return {"error": problem}

    wanted = {str(t) for t in breakdown_types}
    subset = work_orders[work_orders[wo_type_col].astype(str).isin(wanted)]
    total_rows = len(subset)
    if total_rows == 0:
        return {
            "mttr_hours": None, "mtta_hours": None, "mdt_hours": None,
            "repairs": 0, "rows_excluded": 0,
            "longest_repair_hours": None, "longest_wait_hours": None,
        }

    reported = pd.to_datetime(subset[reported_col], errors="coerce")
    started = pd.to_datetime(subset[started_col], errors="coerce")
    finished = pd.to_datetime(subset[finished_col], errors="coerce")

    hour = np.timedelta64(1, "h")
    wait = (started - reported) / hour
    repair = (finished - started) / hour
    down = (finished - reported) / hour

    # One usable-row mask for all three, so the durations stay additive:
    # averaging each column over a different set of rows would break
    # MDT = MTTA + MTTR and quietly produce a self-inconsistent report.
    usable = wait.notna() & repair.notna() & down.notna() & (wait >= 0) & (repair >= 0)
    repairs = int(usable.sum())
    if repairs == 0:
        return {
            "mttr_hours": None, "mtta_hours": None, "mdt_hours": None,
            "repairs": 0, "rows_excluded": total_rows,
            "longest_repair_hours": None, "longest_wait_hours": None,
        }

    return {
        "mttr_hours": float(repair[usable].mean()),
        "mtta_hours": float(wait[usable].mean()),
        "mdt_hours": float(down[usable].mean()),
        "repairs": repairs,
        "rows_excluded": total_rows - repairs,
        "longest_repair_hours": float(repair[usable].max()),
        "longest_wait_hours": float(wait[usable].max()),
    }


def calculate_availability(mtbf_hours: float, mttr_hours: float, mdt_hours: float | None = None) -> dict[str, Any]:
    """
    Inherent and operational availability, and the gap between them.

        inherent    A_i = MTBF / (MTBF + MTTR)
        operational A_o = MTBF / (MTBF + MDT)

    `A_i` is what the equipment is capable of; `A_o` is what the plant actually
    gets. Since MDT includes the wait before anyone starts work, `A_o` is always
    the lower of the two, and the difference is availability lost to **waiting
    rather than repairing**. That gap is not a property of the machine: no new
    equipment removes it, but scheduling, spares holding and work-study can. It
    is returned as its own key so it cannot be overlooked, which is what happens
    whenever a single availability figure is reported on its own.
    """
    values = {"mtbf_hours": mtbf_hours, "mttr_hours": mttr_hours}
    if mdt_hours is not None:
        values["mdt_hours"] = mdt_hours
    numbers = {}
    for name, value in values.items():
        try:
            number = float(value)
        except (TypeError, ValueError):
            return {"error": f"{name} must be a number, got {value!r}."}
        if not np.isfinite(number) or number < 0:
            return {"error": f"{name} must be a finite non-negative number, got {value!r}."}
        numbers[name] = number

    mtbf, mttr = numbers["mtbf_hours"], numbers["mttr_hours"]
    if mtbf + mttr <= 0:
        return {"error": "Availability is undefined when MTBF and MTTR are both zero."}

    inherent = mtbf / (mtbf + mttr)
    operational = None
    if "mdt_hours" in numbers:
        mdt = numbers["mdt_hours"]
        operational = mtbf / (mtbf + mdt) if (mtbf + mdt) > 0 else None

    return {
        "inherent_availability": float(inherent),
        "operational_availability": float(operational) if operational is not None else None,
        "availability_lost_to_waiting": float(inherent - operational) if operational is not None else None,
        "inherent_availability_pct": float(inherent * 100.0),
        "operational_availability_pct": float(operational * 100.0) if operational is not None else None,
    }


def calculate_oee(planned_time_min: float, run_time_min: float, ideal_cycle_time_min: float, total_count: int, good_count: int) -> dict[str, Any]:
    """
    Overall Equipment Effectiveness for one machine over one period.

        Availability = run_time / planned_time
        Performance  = (ideal_cycle_time * total_count) / run_time
        Quality      = good_count / total_count
        OEE          = Availability * Performance * Quality

    **The `availability` returned here is not the quantity `calculate_availability()`
    returns.** This one is a production-time ratio over a shift, with planned
    downtime already excluded from the denominator; that one is a reliability
    ratio derived from MTBF and MTTR. The two are routinely confused, quoted
    against each other, and they do not mean the same thing.

    Nothing is clamped. Performance above 1.0 means the machine beat its stated
    ideal cycle time, which almost always means the master data is wrong rather
    than that the machine is exceptional - clamping it to 1.0 would hide the
    defect instead of reporting it, so it comes back as a warning.
    """
    numbers = {}
    for name, value in (
        ("planned_time_min", planned_time_min),
        ("run_time_min", run_time_min),
        ("ideal_cycle_time_min", ideal_cycle_time_min),
        ("total_count", total_count),
        ("good_count", good_count),
    ):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return {"error": f"{name} must be a number, got {value!r}."}
        if not np.isfinite(number) or number < 0:
            return {"error": f"{name} must be a finite non-negative number, got {value!r}."}
        numbers[name] = number

    planned, run = numbers["planned_time_min"], numbers["run_time_min"]
    cycle, total, good = numbers["ideal_cycle_time_min"], numbers["total_count"], numbers["good_count"]
    if planned <= 0:
        return {"error": "planned_time_min must be greater than zero."}
    if run <= 0:
        return {"error": "run_time_min must be greater than zero."}
    if total <= 0:
        return {"error": "total_count must be greater than zero."}
    if good > total:
        return {"error": f"good_count ({good:g}) cannot exceed total_count ({total:g})."}

    warnings_found = []
    if run > planned:
        warnings_found.append(
            f"run_time_min ({run:g}) exceeds planned_time_min ({planned:g}), so availability is above 1.0."
        )
    availability = run / planned
    performance = (cycle * total) / run
    if performance > 1.0:
        warnings_found.append(
            f"Performance is {performance:.3f}, above 1.0: the machine beat its ideal cycle time of "
            f"{cycle:g} min, which usually means the ideal cycle time is set too slow."
        )
    quality = good / total
    oee = availability * performance * quality

    return {
        "oee": float(oee),
        "availability": float(availability),
        "performance": float(performance),
        "quality": float(quality),
        "oee_pct": float(oee * 100.0),
        "availability_pct": float(availability * 100.0),
        "performance_pct": float(performance * 100.0),
        "quality_pct": float(quality * 100.0),
        "world_class_benchmark": float(OEE_WORLD_CLASS),
        "meets_world_class": bool(oee >= OEE_WORLD_CLASS),
        "warnings": warnings_found,
    }


def calculate_pareto(df: pd.DataFrame, category_col: str, value_col: str | None = None, cutoff: float = PARETO_CUTOFF, top_n: int | None = None) -> dict[str, Any]:
    """
    Rank causes by share of the total and mark the vital few.

    With `value_col` left as None the categories are ranked by how **often** they
    occur; given a column such as downtime hours or cost they are ranked by how
    **much** they cost. The two rankings disagree, and the count ranking is the
    classic trap: a sensor that fails 25 times but is swapped in half an hour
    outranks a bearing that fails 10 times and stops the line for four hours
    each. Ranking by time or money is usually the one that should drive action.

    Rows with a null category are grouped under "(unknown)" rather than dropped,
    because dropping them understates the total and every remaining percentage
    with it - the same reason drift reporting lists what it could not judge
    instead of staying silent about it.
    """
    required = (category_col,) if value_col is None else (category_col, value_col)
    problem = _require_frame(df, "df", required)
    if problem:
        return {"error": problem}
    try:
        limit = float(cutoff)
    except (TypeError, ValueError):
        return {"error": f"cutoff must be a number, got {cutoff!r}."}
    if not np.isfinite(limit) or not 0 < limit <= 1:
        return {"error": f"cutoff must be greater than 0 and at most 1, got {cutoff!r}."}

    categories = df[category_col].astype("object").where(df[category_col].notna(), "(unknown)").astype(str)
    if value_col is None:
        totals = categories.value_counts()
        measured_by = "count"
    else:
        values = pd.to_numeric(df[value_col], errors="coerce")
        if (values.dropna() < 0).any():
            return {"error": f"{value_col} holds negative values; a Pareto over mixed signs has no meaning."}
        totals = values.groupby(categories).sum(min_count=1).dropna().sort_values(ascending=False)
        measured_by = str(value_col)

    grand_total = float(totals.sum())
    if grand_total <= 0:
        return {"error": f"The total of {measured_by} is zero, so no share can be calculated."}

    rows = []
    cumulative = 0.0
    reached = False
    for name, value in totals.items():
        share = float(value) / grand_total * 100.0
        cumulative += share
        # The vital few run up to and including the first category that reaches
        # the cut-off, so the group named always accounts for at least `cutoff`.
        vital = not reached
        if cumulative >= limit * 100.0:
            reached = True
        rows.append({
            "category": str(name),
            "value": float(value),
            "percentage": share,
            "cumulative_percentage": cumulative,
            "is_vital_few": vital,
        })

    if top_n is not None:
        try:
            keep = int(top_n)
        except (TypeError, ValueError):
            return {"error": f"top_n must be a whole number, got {top_n!r}."}
        if keep > 0:
            rows = rows[:keep]

    vital_few = [row["category"] for row in rows if row["is_vital_few"]]
    vital_rows = [row for row in rows if row["is_vital_few"]]

    return {
        "categories": rows,
        "total": grand_total,
        "vital_few": vital_few,
        "vital_few_count": len(vital_few),
        "vital_few_share": float(vital_rows[-1]["cumulative_percentage"]) if vital_rows else 0.0,
        "measured_by": measured_by,
        "cutoff": limit,
    }


def _weibull_mle(sample: np.ndarray) -> tuple[float, float] | None:
    """Maximum-likelihood Weibull shape and scale with the location fixed at zero.

    With loc fixed, the two-parameter likelihood collapses to a single scalar
    equation in beta, and eta follows in closed form. Solving that root directly
    is the same estimate `scipy.stats.weibull_min.fit(x, floc=0)` converges to -
    agreement is within 1e-6 relative on shapes from 0.6 to 4.0 - but a scalar
    root-find costs 0.189 ms against 5.614 ms for the general optimiser, 30x. That
    ratio is the whole reason this helper exists: the confidence interval below
    refits the sample several hundred times, so a 500-sample bootstrap goes from
    2.81 s to 0.094 s, and a per-machine hazard check stops being something a
    shop-floor page has to wait for.

    Returns None when the root cannot be bracketed (every value identical, or a
    shape outside [0.05, 50]); the caller falls back to the general optimiser.
    """
    log_x = np.log(sample)
    mean_log = float(log_x.mean())

    def score(b: float) -> float:
        # d/db of the profile log-likelihood, with eta profiled out.
        weighted = sample ** b
        return float((weighted * log_x).sum() / weighted.sum() - 1.0 / b - mean_log)

    low, high = 0.05, 50.0
    try:
        if score(low) * score(high) > 0:
            return None
        beta = float(brentq(score, low, high, xtol=1e-10, rtol=1e-12))
        eta = float((sample ** beta).mean() ** (1.0 / beta))
    except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
        return None
    if not (np.isfinite(beta) and np.isfinite(eta) and beta > 0 and eta > 0):
        return None
    return beta, eta


def calculate_time_between_failures(
    work_orders: pd.DataFrame,
    time_col: str = "reported_at",
    asset_col: str | None = None,
    wo_type_col: str = "wo_type",
    breakdown_types: tuple[str, ...] = ("breakdown",),
) -> dict[str, Any]:
    """
    Compute elapsed operating times between consecutive breakdown events.

    Times between failures (TBF) are the empirical basis for reliability modeling
    and hazard estimation (such as Weibull analysis). When `asset_col` is given,
    gaps are computed strictly within each asset and never across assets: a gap
    measured from machine A's failure to machine B's failure is not a time between
    failures of anything, but rather a reflection of maintenance queueing or
    reporting artifacts. When `asset_col` is None, the dataset is treated as the
    operational history of a single machine.

    Note that N failures on one machine give N - 1 intervals, so a fleet of
    machines with two failures each yields far fewer intervals than the failure
    count suggests. This distinction determines whether enough data exists to
    reliably fit a failure distribution or test for wear-out.

    Parameters:
    -----------
    work_orders : pd.DataFrame
        Work order records containing maintenance events.
    time_col : str, default="reported_at"
        Column holding failure timestamps (datetime or numeric).
    asset_col : str or None, default=None
        Column identifying the equipment or asset. If None, the entire table is
        treated as a single asset history.
    wo_type_col : str, default="wo_type"
        Column distinguishing breakdown events from planned or preventative work.
    breakdown_types : tuple of str, default=("breakdown",)
        Work order classifications considered genuine breakdown failures.

    Returns:
    --------
    dict:
        JSON-ready diagnostic dictionary containing:
        - "intervals": list of float time gaps between failures
        - "n_intervals": count of computed intervals
        - "n_failures": total count of breakdown events used
        - "assets": count of assets with breakdowns, or None if asset_col is None
        - "per_asset": mapping of asset identifier to interval count, or None
        - "unit": "hours" for datetime input, or "input_units" for numeric
        - "error": error message if frame is unusable
    """
    required = (time_col, wo_type_col) if asset_col is None else (time_col, wo_type_col, asset_col)
    problem = _require_frame(work_orders, "work_orders", required)
    if problem:
        return {"error": problem}

    try:
        wanted = {str(t) for t in breakdown_types}
    except TypeError:
        return {"error": f"breakdown_types must be an iterable of strings, got {breakdown_types!r}."}

    raw_wo_type = work_orders[wo_type_col]
    wo_type_series = raw_wo_type.iloc[:, 0] if isinstance(raw_wo_type, pd.DataFrame) else raw_wo_type
    subset = work_orders[wo_type_series.astype(str).isin(wanted)].copy()

    raw_time = subset[time_col]
    time_series = raw_time.iloc[:, 0] if isinstance(raw_time, pd.DataFrame) else raw_time

    is_dt = False
    if pd.api.types.is_datetime64_any_dtype(time_series.dtype):
        is_dt = True
        parsed_time = pd.to_datetime(time_series, errors="coerce")
    elif _is_numeric_series(time_series):
        is_dt = False
        parsed_time = pd.to_numeric(time_series, errors="coerce")
    else:
        num_parsed = pd.to_numeric(time_series, errors="coerce")
        if num_parsed.notna().sum() > 0 and num_parsed.notna().sum() == time_series.notna().sum():
            is_dt = False
            parsed_time = num_parsed
        else:
            dt_parsed = pd.to_datetime(time_series, errors="coerce")
            if dt_parsed.notna().sum() > 0:
                is_dt = True
                parsed_time = dt_parsed
            else:
                return {"error": f"Column '{time_col}' could not be parsed as numeric or datetime."}

    valid_mask = parsed_time.notna()
    subset = subset.loc[valid_mask].copy()
    subset["_parsed_time"] = parsed_time.loc[valid_mask]
    n_failures = len(subset)
    unit = "hours" if is_dt else "input_units"

    intervals: list[float] = []

    if asset_col is None:
        if n_failures > 1:
            sorted_sub = subset.sort_values(by="_parsed_time", kind="mergesort")
            times = sorted_sub["_parsed_time"]
            if is_dt:
                diffs = (times.diff().iloc[1:] / np.timedelta64(1, "h")).astype(float).tolist()
            else:
                diffs = times.diff().iloc[1:].astype(float).tolist()
            intervals = [float(d) for d in diffs if np.isfinite(d)]
        assets_count = None
        per_asset = None
    else:
        raw_assets = subset[asset_col]
        asset_series = raw_assets.iloc[:, 0] if isinstance(raw_assets, pd.DataFrame) else raw_assets
        unique_assets = asset_series.dropna().unique()
        try:
            sorted_assets = sorted(unique_assets)
        except TypeError:
            sorted_assets = sorted(unique_assets, key=str)

        assets_count = len(sorted_assets)
        per_asset = {}
        for a in sorted_assets:
            a_key = a.item() if hasattr(a, "item") else a
            a_key_str = str(a_key) if not isinstance(a_key, (int, str)) else a_key

            asset_mask = (asset_series == a)
            asset_df = subset.loc[asset_mask].sort_values(by="_parsed_time", kind="mergesort")
            if len(asset_df) > 1:
                times = asset_df["_parsed_time"]
                if is_dt:
                    diffs = (times.diff().iloc[1:] / np.timedelta64(1, "h")).astype(float).tolist()
                else:
                    diffs = times.diff().iloc[1:].astype(float).tolist()
                clean_diffs = [float(d) for d in diffs if np.isfinite(d)]
                intervals.extend(clean_diffs)
                per_asset[a_key_str] = len(clean_diffs)
            else:
                per_asset[a_key_str] = 0

    return {
        "intervals": intervals,
        "n_intervals": len(intervals),
        "n_failures": n_failures,
        "assets": assets_count,
        "per_asset": per_asset,
        "unit": unit,
    }


def calculate_weibull(
    intervals: Any,
    confidence: float = WEIBULL_CI_CONFIDENCE,
    n_bootstrap: int = WEIBULL_BOOTSTRAP_DEFAULT,
    random_state: int = DEFAULT_RANDOM_STATE,
    min_intervals: int = WEIBULL_MIN_INTERVALS,
) -> dict[str, Any]:
    """
    Fit a two-parameter Weibull distribution to failure intervals and test
    whether a constant failure rate is statistically defensible.

    Fixing `loc=0` is required: fitting a non-zero location parameter would imply
    that equipment is physically incapable of failing before some threshold
    operating time, an assumption unsupported by industrial maintenance data.
    The fitted shape parameter `beta` indicates the nature of failure: beta < 1
    implies infant mortality (decreasing hazard), beta > 1 implies wear-out
    (increasing hazard), and beta = 1 corresponds to an exponential distribution
    with a constant hazard rate.

    The confidence interval on `beta` is estimated via bootstrap resampling
    rather than a large-sample formula: Weibull analysis in maintenance is most
    critical when failure events are scarce, which is precisely where large-sample
    standard errors are least trustworthy.

    That interval is then shifted by the bootstrap's own estimate of its bias.
    Maximum likelihood overestimates the Weibull shape on small samples, and an
    uncorrected interval carries that error straight into the verdict: on
    exponential intervals, where the honest answer is "constant", it excluded 1.0
    on 29.3% of 8-interval samples. See the comment at the correction for the
    measured false-alarm and detection rates on both sides of the change, and for
    the pivotal interval that was measured and rejected.

    The correction is applied to the interval only. `beta` stays the raw maximum
    likelihood estimate, so that it remains the same quantity any other tool
    reports for the same data - but that estimate runs high on small samples, by a
    measured margin, and `small_sample_warning` carries the figure rather than
    leaving it to be discovered.

    Parameters:
    -----------
    intervals : array-like
        1-D sequence of failure intervals (positive numbers).
    confidence : float, default=WEIBULL_CI_CONFIDENCE (0.95)
        Two-sided confidence level for the bootstrap percentile interval.
    n_bootstrap : int, default=WEIBULL_BOOTSTRAP_DEFAULT (500)
        Number of bootstrap resamples to fit.
    random_state : int, default=DEFAULT_RANDOM_STATE (42)
        Random seed for reproducible bootstrap draws.
    min_intervals : int, default=WEIBULL_MIN_INTERVALS (8)
        Minimum count of positive finite intervals required for fitting.

    Returns:
    --------
    dict:
        JSON-ready diagnostic dictionary containing:
        - "beta": fitted shape parameter (the raw maximum-likelihood estimate,
          which runs high on small samples - see "small_sample_warning"), or None
        - "eta": fitted scale / characteristic life (63.2% failed), or None
        - "beta_ci_lower": lower bootstrap percentile bound, or None
        - "beta_ci_upper": upper bootstrap percentile bound, or None
        - "constant_hazard_is_valid": True if interval contains 1.0, else False, None if not fitted
        - "hazard_pattern": "infant_mortality", "wear_out", "constant", or None
        - "mttf": Mean Time To Failure (eta * gamma(1 + 1/beta)), or None
        - "b10_life": time by which 10% have failed, or None
        - "n_intervals": count of usable intervals fitted
        - "dropped_non_positive": count of non-positive / non-finite values removed
        - "bootstrap_fits": number of bootstrap resamples successfully fitted
        - "small_sample_warning": set below 30 intervals, with the measured
          false-alarm rate at that size; None otherwise
        - "confidence": echoed confidence level
        - "reason": plain-words interpretation of the hazard verdict
        - "error": error message on refusal, None on success
    """
    try:
        conf = float(confidence)
        if not (0.0 < conf < 1.0):
            return {
                "beta": None, "eta": None, "beta_ci_lower": None, "beta_ci_upper": None,
                "constant_hazard_is_valid": None, "hazard_pattern": None,
                "mttf": None, "b10_life": None, "n_intervals": 0,
                "dropped_non_positive": 0, "bootstrap_fits": 0,
                "confidence": None, "small_sample_warning": None,
                "reason": (
                    f"confidence must be strictly between 0 and 1, got {confidence!r}. "
                    f"Nothing was fitted; the default of {WEIBULL_CI_CONFIDENCE} was NOT "
                    f"substituted, because a silently changed confidence level makes the "
                    f"interval below mean something other than what was asked for."
                ),
                "error": f"confidence must be in (0, 1), got {confidence!r}.",
            }
    except (ValueError, TypeError):
        conf = WEIBULL_CI_CONFIDENCE

    try:
        n_boot = int(n_bootstrap)
        if n_boot < 1:
            n_boot = WEIBULL_BOOTSTRAP_DEFAULT
    except (ValueError, TypeError):
        n_boot = WEIBULL_BOOTSTRAP_DEFAULT

    try:
        m_int = int(min_intervals)
        if m_int < 1:
            m_int = WEIBULL_MIN_INTERVALS
    except (ValueError, TypeError):
        m_int = WEIBULL_MIN_INTERVALS

    try:
        r_seed = int(random_state)
    except (ValueError, TypeError):
        r_seed = DEFAULT_RANDOM_STATE

    try:
        arr = np.asarray(intervals, dtype=float).ravel()
    except (ValueError, TypeError):
        return {
            "beta": None,
            "eta": None,
            "beta_ci_lower": None,
            "beta_ci_upper": None,
            "constant_hazard_is_valid": None,
            "hazard_pattern": None,
            "mttf": None,
            "b10_life": None,
            "n_intervals": 0,
            "dropped_non_positive": 0,
            "bootstrap_fits": 0,
            "small_sample_warning": None,
            "confidence": conf,
            "reason": (
                "Input cannot be coerced to a float array. "
                "Supply a 1-D sequence (list, numpy array, or pandas Series) of positive numbers."
            ),
            "error": "Input cannot be coerced to a float array.",
        }

    valid_mask = np.isfinite(arr) & (arr > 0)
    usable = arr[valid_mask]
    dropped_non_positive = int(len(arr) - len(usable))
    n_usable = len(usable)

    if len(arr) > 0 and n_usable == 0:
        return {
            "beta": None,
            "eta": None,
            "beta_ci_lower": None,
            "beta_ci_upper": None,
            "constant_hazard_is_valid": None,
            "hazard_pattern": None,
            "mttf": None,
            "b10_life": None,
            "n_intervals": 0,
            "dropped_non_positive": dropped_non_positive,
            "bootstrap_fits": 0,
            "small_sample_warning": None,
            "confidence": conf,
            "reason": (
                "Every value in intervals is non-positive or non-finite. "
                "A Weibull distribution is undefined at or below zero. Supply strictly positive failure intervals."
            ),
            "error": "Every interval value is non-positive or non-finite.",
        }

    if n_usable < m_int:
        return {
            "beta": None,
            "eta": None,
            "beta_ci_lower": None,
            "beta_ci_upper": None,
            "constant_hazard_is_valid": None,
            "hazard_pattern": None,
            "mttf": None,
            "b10_life": None,
            "n_intervals": n_usable,
            "dropped_non_positive": dropped_non_positive,
            "bootstrap_fits": 0,
            "small_sample_warning": None,
            "confidence": conf,
            "reason": (
                f"Found {n_usable} usable interval(s), which is fewer than the minimum of {m_int} "
                f"required for a trustworthy Weibull fit. Collect at least {m_int} failure intervals before fitting."
            ),
            "error": f"Found {n_usable} usable interval(s), fewer than min_intervals={m_int}.",
        }

    try:
        fitted = _weibull_mle(usable)
        if fitted is None:
            shape, _loc, scale = weibull_min.fit(usable, floc=0)
            beta, eta = float(shape), float(scale)
        else:
            beta, eta = fitted
        if not (np.isfinite(beta) and np.isfinite(eta) and beta > 0 and eta > 0):
            raise ValueError("Fitted parameters are not positive finite numbers.")
    except Exception as exc:  # noqa: BLE001 - any fit failure becomes the documented refusal
        return {
            "beta": None,
            "eta": None,
            "beta_ci_lower": None,
            "beta_ci_upper": None,
            "constant_hazard_is_valid": None,
            "hazard_pattern": None,
            "mttf": None,
            "b10_life": None,
            "n_intervals": n_usable,
            "dropped_non_positive": dropped_non_positive,
            "bootstrap_fits": 0,
            "small_sample_warning": None,
            "confidence": conf,
            "reason": f"Weibull fit failed: {exc}. Verify interval values.",
            "error": f"Weibull fit failed: {exc}",
        }

    rng = np.random.default_rng(r_seed)
    bootstrap_betas: list[float] = []
    for _ in range(n_boot):
        sample = rng.choice(usable, size=n_usable, replace=True)
        fitted = _weibull_mle(sample)
        if fitted is not None:
            bootstrap_betas.append(fitted[0])

    bootstrap_fits = len(bootstrap_betas)
    if bootstrap_fits > 0:
        alpha = 1.0 - conf
        lower_pct = (alpha / 2.0) * 100.0
        upper_pct = (1.0 - alpha / 2.0) * 100.0
        beta_ci_lower = float(np.percentile(bootstrap_betas, lower_pct))
        beta_ci_upper = float(np.percentile(bootstrap_betas, upper_pct))

        # Maximum likelihood overestimates the Weibull shape on small samples, and
        # the raw percentile interval inherits that shift whole: on exponential
        # intervals - where beta is exactly 1 and the honest verdict is "constant"
        # - the uncorrected interval excluded 1.0 on 29.3% of 8-interval samples.
        # Subtracting the bootstrap's own estimate of the bias re-centres it.
        # Measured over 300 trials per cell, at 8 / 15 / 30 / 60 / 120 intervals:
        # that false-alarm rate falls to 9.0 / 11.3 / 7.0 / 6.0 / 5.3%.
        #
        # It is paid for in power at the smallest sizes, and the cost is real: a
        # true beta of 1.5 is caught 29.7% of the time at 8 intervals instead of
        # 78.0%, and a beta of 2.0 62.3% instead of 94.7%. Those two numbers are
        # not comparable as they stand - the 78.0% was bought at a 29.3%
        # false-alarm rate, the 29.7% at 9.0%. By 60 intervals the two rules agree
        # to within a point on every arm.
        #
        # The pivotal ("basic") interval was measured on the same draws and
        # rejected: it holds the false-alarm rate at 9.0% but detects a beta of
        # 3.0 on 0.7% of 8-interval samples, which is no test at all.
        bootstrap_bias = float(np.mean(bootstrap_betas)) - beta
        beta_ci_lower -= bootstrap_bias
        beta_ci_upper -= bootstrap_bias
    else:
        beta_ci_lower = None
        beta_ci_upper = None

    if beta_ci_lower is not None and beta_ci_upper is not None:
        if beta_ci_lower <= 1.0 <= beta_ci_upper:
            constant_hazard_is_valid = True
            hazard_pattern = "constant"
            reason = (
                f"The {round(conf * 100)}% bootstrap confidence interval for beta [{beta_ci_lower:.3f}, {beta_ci_upper:.3f}] "
                f"contains 1.0. This means the data cannot distinguish this machine from one with a constant failure rate, "
                f"which is not the same as showing that it has one — a small sample fails to distinguish almost anything. "
                f"A constant-hazard assumption is consistent with the data, but more failure observations are needed to rule out wear-out or infant mortality."
            )
        elif beta_ci_lower > 1.0:
            constant_hazard_is_valid = False
            hazard_pattern = "wear_out"
            reason = (
                f"The {round(conf * 100)}% bootstrap confidence interval for beta [{beta_ci_lower:.3f}, {beta_ci_upper:.3f}] "
                f"is entirely above 1.0, indicating a wear-out pattern with an increasing hazard rate over time. "
                f"A constant-hazard assumption is invalid; assuming a constant failure rate (1 / MTBF) understates the risk of near-term failure as the asset ages."
            )
        else:
            constant_hazard_is_valid = False
            hazard_pattern = "infant_mortality"
            reason = (
                f"The {round(conf * 100)}% bootstrap confidence interval for beta [{beta_ci_lower:.3f}, {beta_ci_upper:.3f}] "
                f"is entirely below 1.0, indicating an infant-mortality pattern with a decreasing hazard rate over time. "
                f"Failures are clustered early in operation (e.g. burn-in or installation defects); assuming a constant failure rate overstates risk once initial runtime has passed."
            )
    else:
        constant_hazard_is_valid = None
        hazard_pattern = None
        reason = "Bootstrap refits failed to produce a valid confidence interval for beta."

    try:
        mttf: float | None = float(eta * gamma(1.0 + 1.0 / beta))
        if not np.isfinite(mttf):
            mttf = None
    except (OverflowError, ValueError):
        mttf = None

    try:
        b10_life: float | None = float(eta * ((-math.log(0.9)) ** (1.0 / beta)))
        if not np.isfinite(b10_life):
            b10_life = None
    except (OverflowError, ValueError):
        b10_life = None

    # The bias correction above brings the false-alarm rate close to nominal by
    # about 60 intervals but not below it before then. A caller reading a verdict
    # off 8 or 15 intervals is entitled to know that, in the same numbers this
    # module was tuned on, rather than to discover it from a stripped good bearing.
    small_sample_warning = None
    if n_usable < 30:
        small_sample_warning = (
            f"Verdict rests on {n_usable} interval(s). Measured over 300 trials per cell on "
            f"constant-hazard data, this test still called 9.0% of 8-interval and 11.3% of "
            f"15-interval samples 'not constant', against 7.0% at 30 and 5.3% at 120. The "
            f"reported beta is also biased upward at this size and is NOT corrected: on data "
            f"whose true shape is 1.00, the median fitted beta was 1.14 at 8 intervals and 1.06 "
            f"at 15, against 1.01 at 120. Only the interval that decides the verdict is "
            f"bias-corrected. Treat a wear-out verdict here as a reason to collect more "
            f"failures, not as a finding."
        )

    return {
        "beta": beta,
        "eta": eta,
        "beta_ci_lower": beta_ci_lower,
        "beta_ci_upper": beta_ci_upper,
        "constant_hazard_is_valid": constant_hazard_is_valid,
        "hazard_pattern": hazard_pattern,
        "mttf": mttf,
        "b10_life": b10_life,
        "n_intervals": n_usable,
        "dropped_non_positive": dropped_non_positive,
        "bootstrap_fits": bootstrap_fits,
        "small_sample_warning": small_sample_warning,
        "confidence": conf,
        "reason": reason,
        "error": None,
    }


def calculate_crow_amsaa(
    failure_times: Any,
    observation_end: float,
    confidence: float = WEIBULL_CI_CONFIDENCE,
    min_failures: int = CROW_AMSAA_MIN_FAILURES,
) -> dict[str, Any]:
    """
    Assess whether a repairable machine's failure rate is improving, worsening,
    or constant over time using a power-law Non-Homogeneous Poisson Process (NHPP).

    Models cumulative failure events in a time-truncated observation window
    (0, observation_end]. The shape parameter beta indicates the trend:
    beta < 1 implies reliability growth (improving), beta > 1 implies wear-out /
    deterioration (worsening), and beta = 1 represents a homogeneous Poisson process.
    An exact conditional confidence interval is computed via Chi-square quantiles.

    Parameters:
    -----------
    failure_times : array-like
        Cumulative arrival times of failure events measured from the start of the
        observation window.
    observation_end : float
        End of the observation window (same time units as failure_times). Must be > 0.
    confidence : float, default=WEIBULL_CI_CONFIDENCE (0.95)
        Two-sided confidence level for the exact conditional interval on beta.
        Must be strictly between 0 and 1.
    min_failures : int, default=CROW_AMSAA_MIN_FAILURES (5)
        Minimum count of valid failure times required to estimate parameters.

    Returns:
    --------
    dict:
        JSON-ready diagnostic dictionary containing:
        - "beta": maximum likelihood estimate of the NHPP shape parameter, or None
        - "beta_unbiased": time-truncated bias-corrected shape ((n - 1) / n * beta), or None
        - "lambda_": scale / intensity parameter (failures per unit time^beta), or None
        - "beta_ci_lower": lower exact conditional confidence limit, or None
        - "beta_ci_upper": upper exact conditional confidence limit, or None
        - "trend": "improving", "worsening", "no_evidence", or None on refusal
        - "current_mtbf": instantaneous MTBF (1 / intensity) at observation_end, or None
        - "n_failures": count of valid failure times analyzed
        - "observation_end": echoed observation window endpoint, or None on invalid input
        - "dropped": count of non-positive, non-finite, or out-of-window values excluded
        - "confidence": echoed confidence level, or None on invalid input
        - "reason": plain-language summary of the trend verdict and confidence bounds
        - "error": error message on refusal, None on success
    """
    if isinstance(confidence, bool):
        return {
            "beta": None, "beta_unbiased": None, "lambda_": None,
            "beta_ci_lower": None, "beta_ci_upper": None, "trend": None,
            "current_mtbf": None, "n_failures": 0, "observation_end": None,
            "dropped": 0, "confidence": None,
            "reason": f"confidence must be strictly between 0 and 1, got {confidence!r}.",
            "error": f"confidence must be strictly between 0 and 1, got {confidence!r}.",
        }
    try:
        conf = float(confidence)
        if not (0.0 < conf < 1.0) or not math.isfinite(conf):
            return {
                "beta": None, "beta_unbiased": None, "lambda_": None,
                "beta_ci_lower": None, "beta_ci_upper": None, "trend": None,
                "current_mtbf": None, "n_failures": 0, "observation_end": None,
                "dropped": 0, "confidence": conf if math.isfinite(conf) else None,
                "reason": f"confidence must be strictly between 0 and 1, got {confidence!r}.",
                "error": f"confidence must be strictly between 0 and 1, got {confidence!r}.",
            }
    except (ValueError, TypeError):
        return {
            "beta": None, "beta_unbiased": None, "lambda_": None,
            "beta_ci_lower": None, "beta_ci_upper": None, "trend": None,
            "current_mtbf": None, "n_failures": 0, "observation_end": None,
            "dropped": 0, "confidence": None,
            "reason": f"confidence must be a float in (0, 1), got {confidence!r}.",
            "error": f"confidence must be a float in (0, 1), got {confidence!r}.",
        }

    if isinstance(observation_end, bool):
        return {
            "beta": None, "beta_unbiased": None, "lambda_": None,
            "beta_ci_lower": None, "beta_ci_upper": None, "trend": None,
            "current_mtbf": None, "n_failures": 0, "observation_end": None,
            "dropped": 0, "confidence": conf,
            "reason": f"observation_end must be finite and > 0, got {observation_end!r}.",
            "error": f"observation_end must be finite and > 0, got {observation_end!r}.",
        }
    try:
        obs_end = float(observation_end)
        if not (obs_end > 0.0) or not math.isfinite(obs_end):
            return {
                "beta": None, "beta_unbiased": None, "lambda_": None,
                "beta_ci_lower": None, "beta_ci_upper": None, "trend": None,
                "current_mtbf": None, "n_failures": 0,
                "observation_end": obs_end if math.isfinite(obs_end) else None,
                "dropped": 0, "confidence": conf,
                "reason": f"observation_end must be finite and > 0, got {observation_end!r}.",
                "error": f"observation_end must be finite and > 0, got {observation_end!r}.",
            }
    except (ValueError, TypeError):
        return {
            "beta": None, "beta_unbiased": None, "lambda_": None,
            "beta_ci_lower": None, "beta_ci_upper": None, "trend": None,
            "current_mtbf": None, "n_failures": 0, "observation_end": None,
            "dropped": 0, "confidence": conf,
            "reason": f"observation_end must be a number > 0, got {observation_end!r}.",
            "error": f"observation_end must be a number > 0, got {observation_end!r}.",
        }

    if isinstance(min_failures, bool):
        min_f = CROW_AMSAA_MIN_FAILURES
    else:
        try:
            min_f = int(min_failures)
            if min_f < 1:
                min_f = CROW_AMSAA_MIN_FAILURES
        except (ValueError, TypeError):
            min_f = CROW_AMSAA_MIN_FAILURES

    if isinstance(failure_times, bool):
        return {
            "beta": None, "beta_unbiased": None, "lambda_": None,
            "beta_ci_lower": None, "beta_ci_upper": None, "trend": None,
            "current_mtbf": None, "n_failures": 0, "observation_end": obs_end,
            "dropped": 0, "confidence": conf,
            "reason": f"failure_times cannot be a boolean: {failure_times!r}.",
            "error": "failure_times must be an array-like sequence of numbers.",
        }
    try:
        arr = np.asarray(failure_times, dtype=float).ravel()
    except (ValueError, TypeError):
        return {
            "beta": None, "beta_unbiased": None, "lambda_": None,
            "beta_ci_lower": None, "beta_ci_upper": None, "trend": None,
            "current_mtbf": None, "n_failures": 0, "observation_end": obs_end,
            "dropped": 0, "confidence": conf,
            "reason": f"failure_times could not be converted to a numeric array: {failure_times!r}.",
            "error": "failure_times must be an array-like sequence of numbers.",
        }

    valid_mask = np.isfinite(arr) & (arr > 0.0) & (arr <= obs_end)
    kept = arr[valid_mask]
    dropped = int(len(arr) - len(kept))
    n = len(kept)

    if n < min_f:
        return {
            "beta": None,
            "beta_unbiased": None,
            "lambda_": None,
            "beta_ci_lower": None,
            "beta_ci_upper": None,
            "trend": None,
            "current_mtbf": None,
            "n_failures": n,
            "observation_end": obs_end,
            "dropped": dropped,
            "confidence": conf,
            "reason": (
                f"Crow-AMSAA requires at least {min_f} failure events to report a trend, "
                f"but only {n} valid event(s) were found (dropped {dropped})."
            ),
            "error": f"Crow-AMSAA requires at least {min_f} failures, found {n} (dropped {dropped}).",
        }

    log_ratios = np.log(obs_end / kept)
    s_val = float(np.sum(log_ratios))
    if not math.isfinite(s_val) or s_val <= 0.0:
        return {
            "beta": None,
            "beta_unbiased": None,
            "lambda_": None,
            "beta_ci_lower": None,
            "beta_ci_upper": None,
            "trend": None,
            "current_mtbf": None,
            "n_failures": n,
            "observation_end": obs_end,
            "dropped": dropped,
            "confidence": conf,
            "reason": "All failure times occur at the observation window boundary; beta cannot be computed.",
            "error": "Sum of log(observation_end / t) is <= 0 (all failures at window end).",
        }

    beta = float(n / s_val)
    lam = float(n / (obs_end ** beta))
    beta_unbiased = float(((n - 1) / n) * beta)

    alpha = 1.0 - conf
    lower = float(chi2.ppf(alpha / 2.0, 2 * n) / (2.0 * s_val))
    upper = float(chi2.ppf(1.0 - alpha / 2.0, 2 * n) / (2.0 * s_val))

    if lower > 1.0:
        trend = "worsening"
    elif upper < 1.0:
        trend = "improving"
    else:
        trend = "no_evidence"

    try:
        current_intensity = float(lam * beta * (obs_end ** (beta - 1.0)))
        if current_intensity > 0.0 and math.isfinite(current_intensity):
            current_mtbf = float(1.0 / current_intensity)
        else:
            current_mtbf = None
    except (OverflowError, ValueError):
        current_mtbf = None

    conf_pct = f"{conf * 100:.0f}%" if (conf * 100).is_integer() else f"{conf * 100:.1f}%"
    if trend == "worsening":
        reason = (
            f"Failures are arriving faster over time "
            f"(beta {beta:.2f}, {conf_pct} CI {lower:.2f}-{upper:.2f} excludes 1)."
        )
    elif trend == "improving":
        reason = (
            f"Failures are arriving slower over time "
            f"(beta {beta:.2f}, {conf_pct} CI {lower:.2f}-{upper:.2f} excludes 1)."
        )
    else:
        reason = (
            f"No evidence the failure rate is changing "
            f"({conf_pct} CI {lower:.2f}-{upper:.2f} contains 1). "
            f"This is not proof it is constant - with few failures the test has little power."
        )

    return {
        "beta": beta,
        "beta_unbiased": beta_unbiased,
        "lambda_": lam,
        "beta_ci_lower": lower,
        "beta_ci_upper": upper,
        "trend": trend,
        "current_mtbf": current_mtbf,
        "n_failures": n,
        "observation_end": obs_end,
        "dropped": dropped,
        "confidence": conf,
        "reason": reason,
        "error": None,
    }


def forecast_failure_count(
    n_failures: int,
    lookback: float,
    horizon: float,
    interval: float = FORECAST_INTERVAL_DEFAULT,
) -> dict[str, Any]:
    """
    Forecast the number of failure events in a future time horizon based on
    a historical lookback window, under a homogeneous Poisson process assumption.

    Parameters:
    -----------
    n_failures : int
        Number of failures observed during the lookback window. Must be an integer >= 0.
    lookback : float
        Length of the historical observation window. Must be finite and > 0.
    horizon : float
        Length of the forward forecast horizon. Must be finite and > 0.
    interval : float, default=FORECAST_INTERVAL_DEFAULT (0.80)
        Central coverage probability for the Poisson prediction interval.
        Must be strictly between 0 and 1.

    Returns:
    --------
    dict:
        JSON-ready diagnostic dictionary containing:
        - "expected": expected number of failures in the horizon (rate * horizon), or None
        - "lower": lower integer bound of the Poisson forecast interval, or None
        - "upper": upper integer bound of the Poisson forecast interval, or None
        - "interval": echoed central interval coverage, or None on invalid input
        - "rate": historical failure rate per unit time (n_failures / lookback), or None
        - "prob_at_least_one": probability of observing >= 1 failure (1 - exp(-expected)), or None
        - "n_failures": echoed historical failure count, or None on invalid input
        - "lookback": echoed lookback duration, or None on invalid input
        - "horizon": echoed forecast horizon duration, or None on invalid input
        - "distribution": list of {"k": int, "probability": float} up to max(upper + 2, 4), or [] on error
        - "note": plain-language advisory on Poisson noise and rate uncertainty
        - "error": error message on refusal, None on success
    """
    if isinstance(n_failures, bool):
        return {
            "expected": None, "lower": None, "upper": None, "interval": None,
            "rate": None, "prob_at_least_one": None, "n_failures": None,
            "lookback": None, "horizon": None,
            "distribution": [],
            "note": "n_failures must be an integer >= 0, got boolean.",
            "error": f"n_failures must be an integer >= 0, got {n_failures!r}.",
        }

    try:
        if isinstance(n_failures, (int, np.integer)):
            if n_failures < 0:
                raise ValueError(f"n_failures must be >= 0, got {n_failures}")
            n_f = int(n_failures)
        elif isinstance(n_failures, (float, np.floating)):
            if not math.isfinite(n_failures) or n_failures < 0.0 or not float(n_failures).is_integer():
                raise ValueError(f"n_failures must be a finite whole number >= 0, got {n_failures}")
            n_f = int(n_failures)
        else:
            raise TypeError(f"n_failures must be an integer >= 0, got {type(n_failures).__name__}")
    except (ValueError, TypeError) as err:
        return {
            "expected": None, "lower": None, "upper": None, "interval": None,
            "rate": None, "prob_at_least_one": None, "n_failures": None,
            "lookback": None, "horizon": None,
            "distribution": [],
            "note": str(err),
            "error": str(err),
        }

    if isinstance(lookback, bool):
        return {
            "expected": None, "lower": None, "upper": None, "interval": None,
            "rate": None, "prob_at_least_one": None, "n_failures": n_f,
            "lookback": None, "horizon": None,
            "distribution": [],
            "note": "lookback must be a positive number, got boolean.",
            "error": f"lookback must be a positive number, got {lookback!r}.",
        }
    try:
        lb = float(lookback)
        if not math.isfinite(lb) or lb <= 0.0:
            raise ValueError(f"lookback must be finite and > 0, got {lookback!r}")
    except (ValueError, TypeError) as err:
        return {
            "expected": None, "lower": None, "upper": None, "interval": None,
            "rate": None, "prob_at_least_one": None, "n_failures": n_f,
            "lookback": None, "horizon": None,
            "distribution": [],
            "note": str(err),
            "error": str(err),
        }

    if isinstance(horizon, bool):
        return {
            "expected": None, "lower": None, "upper": None, "interval": None,
            "rate": None, "prob_at_least_one": None, "n_failures": n_f,
            "lookback": lb, "horizon": None,
            "distribution": [],
            "note": "horizon must be a positive number, got boolean.",
            "error": f"horizon must be a positive number, got {horizon!r}.",
        }
    try:
        hz = float(horizon)
        if not math.isfinite(hz) or hz <= 0.0:
            raise ValueError(f"horizon must be finite and > 0, got {horizon!r}")
    except (ValueError, TypeError) as err:
        return {
            "expected": None, "lower": None, "upper": None, "interval": None,
            "rate": None, "prob_at_least_one": None, "n_failures": n_f,
            "lookback": lb, "horizon": None,
            "distribution": [],
            "note": str(err),
            "error": str(err),
        }

    if isinstance(interval, bool):
        return {
            "expected": None, "lower": None, "upper": None, "interval": None,
            "rate": None, "prob_at_least_one": None, "n_failures": n_f,
            "lookback": lb, "horizon": hz,
            "distribution": [],
            "note": "interval must be strictly between 0 and 1, got boolean.",
            "error": f"interval must be strictly between 0 and 1, got {interval!r}.",
        }
    try:
        inv = float(interval)
        if not math.isfinite(inv) or not (0.0 < inv < 1.0):
            raise ValueError(f"interval must be strictly between 0 and 1, got {interval!r}")
    except (ValueError, TypeError) as err:
        return {
            "expected": None, "lower": None, "upper": None, "interval": None,
            "rate": None, "prob_at_least_one": None, "n_failures": n_f,
            "lookback": lb, "horizon": hz,
            "distribution": [],
            "note": str(err),
            "error": str(err),
        }

    rate = float(n_f / lb)
    expected = float(rate * hz)

    if expected == 0.0:
        lower = 0
        upper = 0
    else:
        alpha_half = (1.0 - inv) / 2.0
        lower = int(max(0, poisson.ppf(alpha_half, expected)))
        upper = int(max(0, poisson.ppf(1.0 - alpha_half, expected)))

    prob_at_least_one = float(1.0 - math.exp(-expected))

    k_max = max(upper + 2, 4)
    distribution: list[dict[str, Any]] = []
    if expected == 0.0:
        distribution.append({"k": 0, "probability": 1.0})
        for k in range(1, k_max + 1):
            distribution.append({"k": int(k), "probability": 0.0})
    else:
        for k in range(k_max + 1):
            distribution.append({"k": int(k), "probability": float(poisson.pmf(k, expected))})

    if n_f == 0:
        note = (
            "No failures in the lookback window. A zero forecast is not a zero risk - "
            "the window may simply be too short."
        )
    else:
        note = (
            "The interval covers Poisson noise only, not uncertainty in the rate itself, "
            "so it is narrower than the truth when the lookback holds few failures."
        )

    return {
        "expected": expected,
        "lower": lower,
        "upper": upper,
        "interval": inv,
        "rate": rate,
        "prob_at_least_one": prob_at_least_one,
        "distribution": distribution,
        "n_failures": n_f,
        "lookback": lb,
        "horizon": hz,
        "note": note,
        "error": None,
    }


def _age_replacement_cost_rate(
    t: float,
    beta: float,
    eta: float,
    cost_planned: float,
    cost_breakdown: float,
) -> float:
    """Evaluate long-run cost per unit time under an age replacement policy at interval t."""
    if t <= 0.0:
        return float("inf")
    scaled = t / eta
    z = scaled ** beta
    if z > 700.0:
        r_t = 0.0
        ginc = 1.0
    else:
        r_t = math.exp(-z)
        ginc = float(gammainc(1.0 / beta, z))
    m_t = (eta / beta) * gamma(1.0 / beta) * ginc
    if m_t <= 0.0:
        return float("inf")
    return float((cost_planned * r_t + cost_breakdown * (1.0 - r_t)) / m_t)


def calculate_optimal_pm_interval(
    beta: float,
    eta: float,
    cost_planned: float,
    cost_breakdown: float,
    min_saving_fraction: float = PM_MIN_SAVING_FRACTION,
) -> dict[str, Any]:
    """
    Calculate the cost-optimal Preventive Maintenance (PM) age-replacement interval
    for equipment subject to Weibull(beta, eta) failure behavior.

    Under an age-replacement policy, the asset is replaced/serviced upon reaching
    operating age T (at cost_planned) or upon failure (at cost_breakdown), whichever
    comes first. Each replacement restores the asset to as-good-as-new condition.
    An optimal replacement schedule exists only when equipment exhibits wear-out
    (beta > 1) and breakdown cost exceeds planned replacement cost.

    Parameters:
    -----------
    beta : float
        Weibull shape parameter. Must be finite and > 0.
    eta : float
        Weibull scale parameter (characteristic life). Must be finite and > 0.
    cost_planned : float
        Cost of scheduled preventive maintenance or replacement. Must be finite and > 0.
    cost_breakdown : float
        Total cost of an unplanned breakdown / reactive replacement. Must be finite and > 0.
    min_saving_fraction : float, default=PM_MIN_SAVING_FRACTION (0.05)
        Minimum relative cost rate reduction versus run-to-failure required to
        declare the preventive policy worthwhile.

    Returns:
    --------
    dict:
        JSON-ready diagnostic dictionary containing:
        - "interval": optimal PM age replacement interval T*, or None
        - "cost_rate": long-run cost per unit time under the optimal policy C(T*), or None
        - "cost_rate_run_to_failure": long-run cost per unit time under run-to-failure, or None
        - "saving_fraction": relative cost rate reduction 1 - C(T*) / C_rtf, or None
        - "prob_failure_before_pm": probability of breakdown before reaching PM age, or None
        - "worthwhile": True if PM saves >= min_saving_fraction and optimum is within search bounds
        - "beta": echoed Weibull shape parameter, or None on invalid input
        - "eta": echoed Weibull scale parameter, or None on invalid input
        - "cost_ratio": ratio of breakdown cost to planned cost, or None on invalid input
        - "reason": plain-language advisory explaining the recommendation
        - "error": error message on refusal, None on success
    """
    for name, val in [
        ("beta", beta),
        ("eta", eta),
        ("cost_planned", cost_planned),
        ("cost_breakdown", cost_breakdown),
    ]:
        if isinstance(val, bool):
            return {
                "interval": None, "cost_rate": None, "cost_rate_run_to_failure": None,
                "saving_fraction": None, "prob_failure_before_pm": None,
                "worthwhile": False, "beta": None, "eta": None, "cost_ratio": None,
                "reason": f"{name} must be a positive number, got boolean.",
                "error": f"{name} must be finite and > 0, got {val!r}.",
            }

    try:
        b = float(beta)
        e = float(eta)
        cp = float(cost_planned)
        cb = float(cost_breakdown)
        if not (b > 0.0 and math.isfinite(b)):
            raise ValueError(f"beta must be finite and > 0, got {beta!r}")
        if not (e > 0.0 and math.isfinite(e)):
            raise ValueError(f"eta must be finite and > 0, got {eta!r}")
        if not (cp > 0.0 and math.isfinite(cp)):
            raise ValueError(f"cost_planned must be finite and > 0, got {cost_planned!r}")
        if not (cb > 0.0 and math.isfinite(cb)):
            raise ValueError(f"cost_breakdown must be finite and > 0, got {cost_breakdown!r}")
    except (ValueError, TypeError) as err:
        return {
            "interval": None, "cost_rate": None, "cost_rate_run_to_failure": None,
            "saving_fraction": None, "prob_failure_before_pm": None,
            "worthwhile": False, "beta": None, "eta": None, "cost_ratio": None,
            "reason": str(err),
            "error": str(err),
        }

    try:
        if isinstance(min_saving_fraction, bool):
            msf = PM_MIN_SAVING_FRACTION
        else:
            msf = float(min_saving_fraction)
            if not math.isfinite(msf) or msf < 0.0:
                msf = PM_MIN_SAVING_FRACTION
    except (ValueError, TypeError):
        msf = PM_MIN_SAVING_FRACTION

    cost_ratio = float(cb / cp)
    c_rtf = float(cb / (e * gamma(1.0 + 1.0 / b)))

    if b <= 1.0:
        return {
            "interval": None,
            "cost_rate": None,
            "cost_rate_run_to_failure": c_rtf,
            "saving_fraction": 0.0,
            "prob_failure_before_pm": None,
            "worthwhile": False,
            "beta": b,
            "eta": e,
            "cost_ratio": cost_ratio,
            "reason": (
                "The failure rate is not increasing (beta <= 1), so replacing on a "
                "fixed schedule cannot beat running to failure."
            ),
            "error": None,
        }

    if cb <= cp:
        return {
            "interval": None,
            "cost_rate": None,
            "cost_rate_run_to_failure": c_rtf,
            "saving_fraction": 0.0,
            "prob_failure_before_pm": None,
            "worthwhile": False,
            "beta": b,
            "eta": e,
            "cost_ratio": cost_ratio,
            "reason": "A breakdown costs no more than a planned job, so scheduled PM cannot save money.",
            "error": None,
        }

    def cost_func(t: float) -> float:
        return _age_replacement_cost_rate(t, b, e, cp, cb)

    grid = np.geomspace(0.01 * e, 5.0 * e, 400)
    costs = [cost_func(float(t)) for t in grid]
    min_idx = int(np.argmin(costs))

    low_idx = max(0, min_idx - 1)
    high_idx = min(len(grid) - 1, min_idx + 1)
    bound_low = float(grid[low_idx])
    bound_high = float(grid[high_idx])

    try:
        res = minimize_scalar(cost_func, bounds=(bound_low, bound_high), method="bounded")
        if res.success and math.isfinite(res.x):
            optimal_interval = float(res.x)
            cost_rate = float(res.fun)
        else:
            optimal_interval = float(grid[min_idx])
            cost_rate = float(costs[min_idx])
    except Exception:  # noqa: BLE001 - fallback to best grid point if numerical solver fails
        optimal_interval = float(grid[min_idx])
        cost_rate = float(costs[min_idx])

    saving_fraction = float(1.0 - cost_rate / c_rtf)
    is_last_point = (min_idx == len(grid) - 1)
    worthwhile = bool((saving_fraction >= msf) and (not is_last_point))

    z_opt = (optimal_interval / e) ** b
    prob_failure_before_pm = float(1.0 - (0.0 if z_opt > 700.0 else math.exp(-z_opt)))

    if worthwhile:
        reason = (
            f"Replace/service every {optimal_interval:.1f} units: "
            f"long-run cost rate {saving_fraction * 100:.1f}% below running to failure; "
            f"{prob_failure_before_pm * 100:.1f}% chance of failing before the PM is due."
        )
    else:
        reason = (
            f"Scheduled PM saves {saving_fraction * 100:.1f}%, which is below the "
            f"{msf * 100:.1f}% threshold; recommend running to failure."
        )

    return {
        "interval": optimal_interval,
        "cost_rate": cost_rate,
        "cost_rate_run_to_failure": c_rtf,
        "saving_fraction": saving_fraction,
        "prob_failure_before_pm": prob_failure_before_pm,
        "worthwhile": worthwhile,
        "beta": b,
        "eta": e,
        "cost_ratio": cost_ratio,
        "reason": reason,
        "error": None,
    }


def calculate_weibull_curves(
    beta: float,
    eta: float,
    points: int = 80,
    t_max: float | None = None,
) -> dict[str, Any]:
    """
    Generate evaluation points for Weibull probability density, reliability (survival),
    and hazard rate curves across a specified time grid.

    Parameters:
    -----------
    beta : float
        Weibull shape parameter. Must be finite and > 0.
    eta : float
        Weibull scale parameter (characteristic life). Must be finite and > 0.
    points : int, default=80
        Number of grid evaluation points. Must be an integer in [10, 500].
    t_max : float | None, default=None
        Maximum evaluation time. If None, defaults to 2.5 * eta. Must be finite and > 0.

    Returns:
    --------
    dict:
        JSON-ready dictionary containing:
        - "t": list of grid evaluation times from t_max / points to t_max
        - "pdf": Weibull probability density f(t)
        - "reliability": Weibull reliability / survival R(t) = P(T > t)
        - "hazard": Weibull hazard rate h(t) = f(t) / R(t)
        - "beta": echoed Weibull shape parameter
        - "eta": echoed Weibull scale parameter
        - "t_max": maximum evaluation time
        - "hazard_shape": 'rising' (beta > 1), 'falling' (beta < 1), or 'flat' (beta == 1)
        - "reason": plain-language summary of the curve characteristics
        - "error": error message on refusal, None on success
    """
    for name, val in [("beta", beta), ("eta", eta)]:
        if isinstance(val, bool):
            return {
                "t": [], "pdf": [], "reliability": [], "hazard": [],
                "beta": None, "eta": None, "t_max": None, "hazard_shape": None,
                "reason": f"{name} must be a positive number, got boolean.",
                "error": f"{name} must be finite and > 0, got {val!r}.",
            }

    try:
        b = float(beta)
        e = float(eta)
        if not (math.isfinite(b) and b > 0.0):
            raise ValueError(f"beta must be finite and > 0, got {beta!r}")
        if not (math.isfinite(e) and e > 0.0):
            raise ValueError(f"eta must be finite and > 0, got {eta!r}")
    except (ValueError, TypeError) as err:
        return {
            "t": [], "pdf": [], "reliability": [], "hazard": [],
            "beta": None, "eta": None, "t_max": None, "hazard_shape": None,
            "reason": str(err),
            "error": str(err),
        }

    if isinstance(points, bool):
        return {
            "t": [], "pdf": [], "reliability": [], "hazard": [],
            "beta": None, "eta": None, "t_max": None, "hazard_shape": None,
            "reason": "points must be an integer in [10, 500], got boolean.",
            "error": f"points must be an integer in [10, 500], got {points!r}.",
        }

    try:
        if isinstance(points, (int, np.integer)) or isinstance(points, (float, np.floating)) and math.isfinite(points) and float(points).is_integer():
            p = int(points)
        else:
            raise TypeError(f"points must be an integer in [10, 500], got {type(points).__name__}")
        if not (10 <= p <= 500):
            raise ValueError(f"points must be an integer in [10, 500], got {points!r}")
    except (ValueError, TypeError) as err:
        return {
            "t": [], "pdf": [], "reliability": [], "hazard": [],
            "beta": None, "eta": None, "t_max": None, "hazard_shape": None,
            "reason": str(err),
            "error": str(err),
        }

    if t_max is None:
        tm = float(e * 2.5)
    else:
        if isinstance(t_max, bool):
            return {
                "t": [], "pdf": [], "reliability": [], "hazard": [],
                "beta": None, "eta": None, "t_max": None, "hazard_shape": None,
                "reason": "t_max must be a positive number, got boolean.",
                "error": f"t_max must be finite and > 0, got {t_max!r}.",
            }
        try:
            tm = float(t_max)
            if not (math.isfinite(tm) and tm > 0.0):
                raise ValueError(f"t_max must be finite and > 0, got {t_max!r}")
        except (ValueError, TypeError) as err:
            return {
                "t": [], "pdf": [], "reliability": [], "hazard": [],
                "beta": None, "eta": None, "t_max": None, "hazard_shape": None,
                "reason": str(err),
                "error": str(err),
            }

    t_arr = np.linspace(tm / p, tm, p)
    t_list = [float(val) for val in t_arr]
    pdf_list: list[float] = []
    rel_list: list[float] = []
    haz_list: list[float] = []

    for val in t_list:
        scaled = val / e
        z = scaled ** b
        rel = float(math.exp(-z)) if z < 700.0 else 0.0
        haz = float((b / e) * (scaled ** (b - 1.0)))
        pdf = float(haz * rel)
        pdf_list.append(pdf)
        rel_list.append(rel)
        haz_list.append(haz)

    if b > 1.0:
        hazard_shape = "rising"
    elif b < 1.0:
        hazard_shape = "falling"
    else:
        hazard_shape = "flat"

    return {
        "t": t_list,
        "pdf": pdf_list,
        "reliability": rel_list,
        "hazard": haz_list,
        "beta": b,
        "eta": e,
        "t_max": tm,
        "hazard_shape": hazard_shape,
        "reason": f"Weibull curves evaluated at {p} points ({hazard_shape} hazard).",
        "error": None,
    }


def calculate_pm_cost_curve(
    beta: float,
    eta: float,
    cost_planned: float,
    cost_breakdown: float,
    points: int = 80,
) -> dict[str, Any]:
    """
    Calculate the long-run cost rate curve under an age replacement policy
    across an operating interval grid from 0.05 * eta to 3.0 * eta.

    Parameters:
    -----------
    beta : float
        Weibull shape parameter. Must be finite and > 0.
    eta : float
        Weibull scale parameter (characteristic life). Must be finite and > 0.
    cost_planned : float
        Cost of scheduled preventive maintenance. Must be finite and > 0.
    cost_breakdown : float
        Cost of reactive breakdown repair. Must be finite and > 0.
    points : int, default=80
        Number of evaluation points. Must be an integer in [10, 500].

    Returns:
    --------
    dict:
        JSON-ready diagnostic dictionary containing:
        - "t": list of replacement intervals
        - "cost_rate": cost per unit time for each interval t
        - "run_to_failure_rate": cost per unit time under run-to-failure policy
        - "beta": echoed Weibull shape parameter
        - "eta": echoed Weibull scale parameter
        - "cost_ratio": ratio of breakdown cost to planned cost
        - "reason": summary of the curve
        - "error": error message on refusal, None on success
    """
    for name, val in [
        ("beta", beta),
        ("eta", eta),
        ("cost_planned", cost_planned),
        ("cost_breakdown", cost_breakdown),
    ]:
        if isinstance(val, bool):
            return {
                "t": [], "cost_rate": [], "run_to_failure_rate": None,
                "beta": None, "eta": None, "cost_ratio": None,
                "reason": f"{name} must be a positive number, got boolean.",
                "error": f"{name} must be finite and > 0, got {val!r}.",
            }

    try:
        b = float(beta)
        e = float(eta)
        cp = float(cost_planned)
        cb = float(cost_breakdown)
        if not (math.isfinite(b) and b > 0.0):
            raise ValueError(f"beta must be finite and > 0, got {beta!r}")
        if not (math.isfinite(e) and e > 0.0):
            raise ValueError(f"eta must be finite and > 0, got {eta!r}")
        if not (math.isfinite(cp) and cp > 0.0):
            raise ValueError(f"cost_planned must be finite and > 0, got {cost_planned!r}")
        if not (math.isfinite(cb) and cb > 0.0):
            raise ValueError(f"cost_breakdown must be finite and > 0, got {cost_breakdown!r}")
    except (ValueError, TypeError) as err:
        return {
            "t": [], "cost_rate": [], "run_to_failure_rate": None,
            "beta": None, "eta": None, "cost_ratio": None,
            "reason": str(err),
            "error": str(err),
        }

    if isinstance(points, bool):
        return {
            "t": [], "cost_rate": [], "run_to_failure_rate": None,
            "beta": None, "eta": None, "cost_ratio": None,
            "reason": "points must be an integer in [10, 500], got boolean.",
            "error": f"points must be an integer in [10, 500], got {points!r}.",
        }

    try:
        if isinstance(points, (int, np.integer)) or isinstance(points, (float, np.floating)) and math.isfinite(points) and float(points).is_integer():
            p = int(points)
        else:
            raise TypeError(f"points must be an integer in [10, 500], got {type(points).__name__}")
        if not (10 <= p <= 500):
            raise ValueError(f"points must be an integer in [10, 500], got {points!r}")
    except (ValueError, TypeError) as err:
        return {
            "t": [], "cost_rate": [], "run_to_failure_rate": None,
            "beta": None, "eta": None, "cost_ratio": None,
            "reason": str(err),
            "error": str(err),
        }

    cost_ratio = float(cb / cp)
    c_rtf = float(cb / (e * gamma(1.0 + 1.0 / b)))
    t_grid = np.linspace(0.05 * e, 3.0 * e, p)
    t_list = [float(val) for val in t_grid]
    cost_rate = [_age_replacement_cost_rate(val, b, e, cp, cb) for val in t_list]

    return {
        "t": t_list,
        "cost_rate": cost_rate,
        "run_to_failure_rate": c_rtf,
        "beta": b,
        "eta": e,
        "cost_ratio": cost_ratio,
        "reason": f"PM cost rate evaluated at {p} points.",
        "error": None,
    }
