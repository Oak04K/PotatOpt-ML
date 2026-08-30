from __future__ import annotations

from typing import Any  # loose return-type annotations for JSON-shaped dicts

import numpy as np  # arrays + math for SPC/EWMA/CUSUM limits, downcasting, anomaly scoring
import pandas as pd  # DataFrame/Series is the data contract for every public function
from scipy.stats import norm  # norm.ppf() - z-score for wilson_confidence_interval()

from ._utils import _require_frame
from .constants import OEE_WORLD_CLASS, PARETO_CUTOFF


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
