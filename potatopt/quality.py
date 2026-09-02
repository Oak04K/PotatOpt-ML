from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from ._utils import _finite_float, to_jsonable
from .constants import (
    CAPABILITY_BASELINE_INFLATION_K,
    CAPABILITY_OUTLIER_ALPHA,
    CAPABILITY_OUTLIER_RATE_LIMIT,
    CAPABILITY_SIGMA_RATIO_LIMIT,
    CAPABILITY_TREND_ALPHA,
    CAPABILITY_TREND_DRIFT_SIGMAS,
    CAPABILITY_TREND_LAMBDA,
    CAPABILITY_TREND_RATE_LIMIT,
    CONTROL_RULES_WESTERN_ELECTRIC,
    NORMAL_TAIL_BEYOND_3_SIGMA,
)
from .spc import _baseline_stats, calculate_control_rules, calculate_ewma_chart


def _stability_sigma(sigma_within: float, baseline_n: int | None) -> tuple[float, float]:
    """
    The sigma the stability criteria are tested against, and the factor applied.

    With no baseline window the series estimates its own sigma from every point and
    the factor is 1.0, leaving the criteria exactly as they were. With a window of m
    points the estimate carries m points' worth of error, so the test sigma is
    widened by 1 + CAPABILITY_BASELINE_INFLATION_K / sqrt(m). The reported
    `sigma_within` is never touched: capability indices describe the process, and
    this factor describes only how well the baseline pinned the process down.
    """
    if not baseline_n:
        return sigma_within, 1.0
    window = max(2, int(baseline_n))
    inflation = 1.0 + CAPABILITY_BASELINE_INFLATION_K / float(np.sqrt(window))
    return sigma_within * inflation, inflation


def _criterion_sigma_ratio(context: dict[str, Any]) -> dict[str, Any]:
    """
    Catch variation arriving between subgroups rather than within them.

    Measured on healthy data the ratio sits at 0.99-1.00 whatever the length, and a
    limit of 1.20 never fired in 400 trials at n >= 100 while catching a 3-sigma drift
    97.7% of the time and a 2-sigma step 99.3%.

    The limit is widened when sigma came from a baseline window, because the ratio is
    then partly measuring the error in its own denominator.

    Blind spot: a variance change lifts sigma_within and sigma_overall together,
    leaving the ratio near 1.
    """
    sigma_ratio = context["sigma_ratio"]
    limit = CAPABILITY_SIGMA_RATIO_LIMIT * context["stability_inflation"]
    fired = bool(sigma_ratio is not None and sigma_ratio > limit)
    reason = (
        f"sigma_overall is {sigma_ratio:.2f} times sigma_within (limit "
        f"{limit:.2f}), so variation is arriving between subgroups "
        f"rather than within them"
        if fired
        else None
    )
    return {
        "name": "sigma_ratio",
        "value": sigma_ratio,
        "limit": limit,
        "fired": fired,
        "reason": reason,
    }


def _criterion_outlier_rate(context: dict[str, Any]) -> dict[str, Any]:
    """
    Catch points beyond 3 sigma at a rate chance does not explain.

    This covers the sigma-ratio criterion's blind spot: a variance change inflates
    sigma_within and sigma_overall together, leaving the ratio near 1. Chance puts
    0.27% of points outside, and this is the only criterion that sees a mid-series
    variance doubling at all - 67.4% of the time by n=1000.

    The rate alone loses resolution on a short series: one point in 50 is already 2%,
    and chance supplies at least one such point 12.6% of the time. So the count must
    also be more than chance explains, tested against Binomial(n, 0.0027) at alpha
    CAPABILITY_OUTLIER_ALPHA. Measured over 1,000 trials per cell at
    n = 50 / 100 / 200 / 400 / 1000, that changes the 50-point column and nothing
    else: healthy false alarms go 17.4 / 3.4 / 3.0 / 0.5 / 0.0% to
    4.6 / 3.4 / 3.0 / 0.5 / 0.0%. The cost, also only at n=50: a variance doubling is
    caught 13.4% of the time rather than 46.3%, a figure that had been bought at a
    17.4% false-alarm rate on healthy data.
    """
    rules_res = calculate_control_rules(
        context["series"],
        target=context["stability_centre"],
        sigma=context["stability_sigma"],
        baseline_n=context["baseline_n"],
        rules=(1,),
    )
    series = context["series"]
    beyond_3s = rules_res.get("violations", {}).get("1", []) if "error" not in rules_res else []
    count = len(beyond_3s)
    n_points = len(series)
    outlier_rate = (count / n_points) if n_points else 0.0
    practical = bool(outlier_rate > CAPABILITY_OUTLIER_RATE_LIMIT)
    if not practical:
        fired = False
        reason = None
    else:
        tail = float(scipy.stats.binom.sf(count - 1, n_points, NORMAL_TAIL_BEYOND_3_SIGMA))
        fired = bool(tail < CAPABILITY_OUTLIER_ALPHA)
        reason = (
            f"{outlier_rate:.1%} of points lie beyond 3 sigma - {count} of {n_points} - and chance "
            f"alone produces that many with probability {tail:.1%} (limits: above "
            f"{CAPABILITY_OUTLIER_RATE_LIMIT:.0%} of points, and below {CAPABILITY_OUTLIER_ALPHA:.0%} probability)"
            if fired
            else None
        )
    return {
        "name": "outlier_rate_beyond_3_sigma",
        "value": outlier_rate,
        "limit": CAPABILITY_OUTLIER_RATE_LIMIT,
        "fired": fired,
        "reason": reason,
    }


def _criterion_ewma_trend(context: dict[str, Any]) -> dict[str, Any]:
    """
    Catch a sustained drift across the series.

    The EWMA carries a fading memory of earlier readings, accumulating a shift too
    small to break a 3-sigma limit until the chart crosses. Lambda is 0.1 because that
    is the standard choice for a shift of about 1 sigma.

    The criterion is a rate rather than "signalled at least once", which at this lambda
    fires on 4.1 / 7.6 / 18.1 / 34.3 / 66.1% of healthy series at
    n = 50 / 100 / 200 / 400 / 1000 - climbing with the amount of data until it
    condemns everything, the trap that disqualified the Western Electric rule set.

    Measured over 1,000 trials per cell, adding it moved detection of a 1-sigma drift
    from 20.6 / 7.4 / 4.1 / 1.8 / 0.2% to 25.2 / 27.0 / 43.3 / 58.1 / 80.1%, against a
    healthy false-alarm rate that went from 15.8 / 2.2 / 2.3 / 0.4 / 0.0% to
    16.5 / 2.9 / 3.0 / 0.5 / 0.0%. Detection had been falling as data accumulated,
    because a longer drift is more thoroughly absorbed into sigma_overall.

    Blind spot: a half-sigma drift, which no lambda reaches - see
    `_criterion_fitted_trend`, which exists for it.
    """
    series = context["series"]
    chart = calculate_ewma_chart(
        series,
        lambda_weight=CAPABILITY_TREND_LAMBDA,
        target=context["stability_centre"],
        sigma=context["stability_sigma"],
        baseline_n=context["baseline_n"],
    )
    if "error" in chart or len(series) == 0:
        value = None
        fired = False
        reason = None
    else:
        rate = len(chart["violations"]) / len(series)
        value = rate
        fired = bool(value > CAPABILITY_TREND_RATE_LIMIT)
        reason = (
            f"{rate:.1%} of points have an EWMA beyond its control limits (limit "
            f"{CAPABILITY_TREND_RATE_LIMIT:.0%}), the signature of a sustained drift too small "
            f"for any single point to look extreme"
            if fired
            else None
        )
    return {
        "name": "ewma_violation_rate",
        "value": value,
        "limit": CAPABILITY_TREND_RATE_LIMIT,
        "fired": fired,
        "reason": reason,
    }


def _criterion_fitted_trend(context: dict[str, Any]) -> dict[str, Any]:
    """
    Catch a fitted trend across the series.

    The EWMA cannot see a half-sigma drift at any lambda: centred on its own mean such
    a drift never leaves +/-0.25 sigma, against an EWMA limit of 0.688 sigma. A slope
    is a different instrument and carries t = 4.56 on the same data at n=1000, though
    t = 1.02 at n=50 where it really is out of reach.

    Both practical and statistical significance are required, because a long enough
    series makes a negligible slope significant. The fitted total drift must exceed
    CAPABILITY_TREND_DRIFT_SIGMAS and the slope must be significant at
    CAPABILITY_TREND_ALPHA.

    Measured over 1,000 trials per cell at n = 50 / 100 / 200 / 400 / 1000, adding it
    to the other three moved a 1-sigma drift from 14.0 / 27.0 / 43.3 / 58.1 / 80.1% to
    33.7 / 64.4 / 84.2 / 92.8 / 98.6%, while healthy false alarms went from
    3.4 / 2.9 / 3.0 / 0.5 / 0.0% to 4.6 / 3.4 / 3.0 / 0.5 / 0.0% - unchanged at
    n >= 200, the whole cost falling on the two shortest lengths.

    A 0.5-sigma drift is still called only 4.0% of the time at n=1000. That is a choice,
    not a blindness: it sits below CAPABILITY_TREND_DRIFT_SIGMAS, and lowering that to
    0.5 raises it to 53.0% while the healthy rate at n=200 goes from 3.0% to 3.7%.
    """
    series = context["series"]
    sigma = context["stability_sigma"]
    if series.size < 3 or sigma <= 0:
        value = None
        fired = False
        reason = None
    else:
        fit = scipy.stats.linregress(np.arange(series.size, dtype=float), series)
        total_drift = float(fit.slope) * (series.size - 1)
        value = abs(total_drift) / sigma
        pvalue = float(fit.pvalue)
        fired = bool(
            np.isfinite(pvalue)
            and value > CAPABILITY_TREND_DRIFT_SIGMAS
            and pvalue < CAPABILITY_TREND_ALPHA
        )
        if fired:
            direction = "rising" if total_drift > 0 else "falling"
            reason = (
                f"the series drifts {value:.2f} sigma end to end ({direction}), a slope chance explains with "
                f"probability {pvalue:.1%} (limits: above {CAPABILITY_TREND_DRIFT_SIGMAS} sigma, and below "
                f"{CAPABILITY_TREND_ALPHA:.0%} probability)"
            )
        else:
            reason = None
    return {
        "name": "fitted_trend_sigmas",
        "value": value,
        "limit": CAPABILITY_TREND_DRIFT_SIGMAS,
        "fired": fired,
        "reason": reason,
    }


# Every criterion that can call a process unstable, each carrying in its own
# docstring the false-alarm rate and the power that justify its limit. A new
# criterion is added here and measured before it is trusted - never inline in
# calculate_capability(), which is how a check drifts away from its evidence.
_STABILITY_CRITERIA = (
    _criterion_sigma_ratio,
    _criterion_outlier_rate,
    _criterion_ewma_trend,
    _criterion_fitted_trend,
)


def _assess_stability(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Run every registered stability criterion and return their verdicts in order."""
    return [criterion(context) for criterion in _STABILITY_CRITERIA]


def calculate_capability(
    values: Any,
    usl: float | None = None,
    lsl: float | None = None,
    target: float | None = None,
    baseline_n: int | None = None,
) -> dict[str, Any]:
    """
    Calculate process capability (Cp, Cpk) and process performance (Pp, Ppk) indices.

    Capability is the final link in the Quality Engineering chain:
    1. Gauge R&R ensures the measurement system is trustworthy.
    2. Control rules ensure the process is in statistical control.
    3. Capability (Cp/Cpk/Pp/Ppk) is only interpretable once both hold - a Cpk
       computed on an out-of-control process describes nothing, and a Cpk computed
       through a gauge contributing 40% of observed variation measures the gauge.

    Variance Separation:
    --------------------
    - `sigma_within`: Estimated from average moving range over d2 (Phase I baseline).
      Reflects inherent short-term variation within subgroup/consecutive samples.
    - `sigma_overall`: Estimated from sample standard deviation (s). Reflects total
      observed variation over time, including process shifts, drifts, and instability.
    - `sigma_ratio` (sigma_overall / sigma_within): A ratio well above 1.0 indicates
      variation is arriving between subgroups rather than within them - the same
      finding as `stable: False`, arriving by arithmetic instead of by rule.

    Indices:
    --------
    - Cp, Cpk use `sigma_within` (potential capability).
    - Pp, Ppk use `sigma_overall` (realized process performance).
    - With only one specification limit, two-sided Cp/Pp are undefined (`None`),
      and the one-sided Cpk/Ppk is computed from the available limit.

    Stability & Meaningfulness:
    ---------------------------
    - `stability_violations` lists which Western Electric rules fired. It is reported
      because it says WHERE to look, but it is deliberately NOT the stability test:
      measured on healthy in-control data the set signals at least once on 30.5% of
      50-point series, 60.0% at 100, 81.0% at 200, 97.5% at 400 and 99.5% at 1,000.
      A gate built on "any rule fired" would reject nearly every real data set, and a
      flag that is always raised is not a flag - the same lesson the Andon amber state
      learned when it started requiring sustained evidence.
    - `stable` uses four length-independent criteria, any of which fails it:
      `sigma_ratio` above CAPABILITY_SIGMA_RATIO_LIMIT (1.20), more than
      CAPABILITY_OUTLIER_RATE_LIMIT (1%) of points beyond 3 sigma in a count chance
      does not explain (Binomial, alpha CAPABILITY_OUTLIER_ALPHA), more than
      CAPABILITY_TREND_RATE_LIMIT (3%) of points whose EWMA (lambda
      CAPABILITY_TREND_LAMBDA, 0.1) lies outside its own control limits, or a fitted
      line rising or falling more than CAPABILITY_TREND_DRIFT_SIGMAS (0.75) sigma
      end to end with a slope significant at CAPABILITY_TREND_ALPHA (1%). Together
      they fire on 3.4% of healthy 100-point series and 0.0% of healthy 1,000-point
      ones, while catching a 3-sigma drift and a 2-sigma step every time, a 1-sigma
      drift 98.6% and a mid-series variance doubling 67.4% at n=1000.
    - **Short series are judged more carefully, not more bravely.** A 1% rate on
      50 points is one reading, which chance supplies 12.6% of the time, so the
      count carries a Binomial test as well: at n=50 the gate went from calling
      17.4% of healthy series unstable to 4.6%. Nothing at n >= 100 changed. The
      price is at n=50 only, where a variance doubling is now caught 13.4% of the
      time rather than 46.3% - a figure that had been bought at that 17.4%
      false-alarm rate.
    - Each criterion is reported in `stability_criteria` as a dict of
      `name` / `value` / `limit` / `fired` / `reason`, so the reason a process was
      called unstable can be read without re-deriving it.
    - **`baseline_n` widens the test, because a short window measures sigma
      badly.** It restricts the sigma estimate to the first N readings, which is
      right when a Phase I period has to be fenced off from a process that later
      wandered. But that estimate is unbiased and *noisy*, and a criterion compared
      against a noisy ruler reads the noise: on 300 in-control points the gate
      called healthy series unstable 24.8% of the time at N=40 and 40.4% at N=20.
      The sigma used for testing is therefore widened by
      1 + CAPABILITY_BASELINE_INFLATION_K / sqrt(N), which brings those to 2.2%
      and 7.0%. The reported `sigma_within` and every capability index are
      untouched. A 2-sigma step after the window is still caught every time; a
      1-sigma drift is caught 79.6% of the time at N=40 rather than 97.8%, a
      detection rate that had been bought at a 24.8% false-alarm rate. Twenty
      points cannot pin down a sigma and the correction does not pretend otherwise.
    - **Known blind spots, stated rather than discovered later:** a variance change
      lifts `sigma_within` and `sigma_overall` together, so the ratio stays near 1
      and neither the ratio nor the fitted line sees it - only the outlier rate
      does, at 67.4% by n=1000. A drift of 0.5 sigma end to end is called 4.0% of
      the time, which is a choice rather than a blindness: it sits below
      CAPABILITY_TREND_DRIFT_SIGMAS, and lowering that limit to 0.5 raises it to
      53.0% while lifting the false-alarm rate at n=200 from 3.0% to 3.7%. Read
      `stability_criteria` and the chart itself when the decision matters.
    - `capability_is_meaningful`: `False` whenever `stable` is `False`.
      Capability indices describe a process that repeats itself, and an out-of-control
      process has no single distribution for them to describe.

    Normality Assumption:
    ---------------------
    Normality is assumed and reported via `skewness` and `excess_kurtosis`. When
    `abs(skewness) > 1` or `abs(excess_kurtosis) > 1`, `normality_warning` is set,
    warning that Cpk on a strongly non-normal process misstates the fraction beyond
    limits. Hypothesis tests are avoided because large samples reject on negligible
    differences.

    Verdict Bands:
    --------------
    - Cpk >= 1.33: "capable"
    - 1.00 <= Cpk < 1.33: "marginal"
    - Cpk < 1.00: "not capable"
    These bands are industrial convention, not physical law.

    Parameters:
    -----------
    values : array-like
        The process measurement series.
    usl : float or None
        Upper Specification Limit.
    lsl : float or None
        Lower Specification Limit.
    target : float or None
        Nominal target value.
    baseline_n : int or None
        Number of initial points used for baseline sigma_within estimation.

    Returns:
    --------
    dict:
        JSON-ready dict. Never raises on invalid input; returns an error dict.
    """
    if usl is None and lsl is None:
        return {"error": "At least one specification limit (USL or LSL) must be provided."}

    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if series.size == 0:
        return {"error": "No numeric values supplied."}
    if series.size < 2:
        return {"error": f"At least 2 points are required to calculate capability, got {series.size}."}

    usl_val: float | None = None
    if usl is not None:
        usl_val, problem = _finite_float(usl, "usl")
        if problem:
            return {"error": problem}

    lsl_val: float | None = None
    if lsl is not None:
        lsl_val, problem = _finite_float(lsl, "lsl")
        if problem:
            return {"error": problem}

    if usl_val is not None and lsl_val is not None and usl_val <= lsl_val:
        return {"error": f"USL ({usl_val}) must be strictly greater than LSL ({lsl_val})."}

    target_val: float | None = None
    if target is not None:
        target_val, problem = _finite_float(target, "target")
        if problem:
            return {"error": problem}

    n_points = int(series.size)
    mean_val = float(np.mean(series))
    baseline_centre, sigma_within = _baseline_stats(series, baseline_n)
    sigma_overall = float(np.std(series, ddof=1))

    sigma_ratio = float(sigma_overall / sigma_within) if sigma_within > 0 else None
    sigma_ratio_note = (
        "A sigma ratio (overall / within) well above 1.0 indicates variation arriving between "
        "subgroups rather than within them (process drift or instability)."
    )

    cp: float | None = None
    cpu: float | None = None
    cpl: float | None = None
    cpk: float | None = None

    if sigma_within > 0:
        if usl_val is not None and lsl_val is not None:
            cp = float((usl_val - lsl_val) / (6.0 * sigma_within))
            cpu = float((usl_val - mean_val) / (3.0 * sigma_within))
            cpl = float((mean_val - lsl_val) / (3.0 * sigma_within))
            cpk = min(cpu, cpl)
        elif usl_val is not None:
            cpu = float((usl_val - mean_val) / (3.0 * sigma_within))
            cpk = cpu
        elif lsl_val is not None:
            cpl = float((mean_val - lsl_val) / (3.0 * sigma_within))
            cpk = cpl

    pp: float | None = None
    ppu: float | None = None
    ppl: float | None = None
    ppk: float | None = None

    if sigma_overall > 0:
        if usl_val is not None and lsl_val is not None:
            pp = float((usl_val - lsl_val) / (6.0 * sigma_overall))
            ppu = float((usl_val - mean_val) / (3.0 * sigma_overall))
            ppl = float((mean_val - lsl_val) / (3.0 * sigma_overall))
            ppk = min(ppu, ppl)
        elif usl_val is not None:
            ppu = float((usl_val - mean_val) / (3.0 * sigma_overall))
            ppk = ppu
        elif lsl_val is not None:
            ppl = float((mean_val - lsl_val) / (3.0 * sigma_overall))
            ppk = ppl

    rules_res = calculate_control_rules(
        series,
        target=target_val,
        baseline_n=baseline_n,
        rules=CONTROL_RULES_WESTERN_ELECTRIC,
    )
    # Which rules fired is reported, but "any rule fired at all" is NOT the
    # stability test. Measured on healthy in-control data, the Western Electric
    # set signals at least once on 30.5% of 50-point series, 60.0% at 100, 81.0%
    # at 200, 97.5% at 400 and 99.5% at 1,000: a gate built on it would call
    # almost every real data set unusable and would be ignored within a week -
    # the same failure the Andon amber state hit before it required sustained
    # evidence. Two length-independent criteria are used instead.
    if "error" in rules_res:
        stability_violations: list[int] = []
    else:
        raw_viols = rules_res.get("violations", {})
        stability_violations = sorted([int(r) for r, idxs in raw_viols.items() if idxs])

    stability_sigma, stability_inflation = _stability_sigma(sigma_within, baseline_n)
    stability_centre = target_val if target_val is not None else baseline_centre

    stability_criteria = _assess_stability({
        "series": series,
        "sigma_ratio": sigma_ratio,
        "baseline_n": baseline_n,
        "stability_sigma": stability_sigma,
        "stability_inflation": stability_inflation,
        "stability_centre": stability_centre,
    })
    outlier_rate = next(
        c["value"] for c in stability_criteria if c["name"] == "outlier_rate_beyond_3_sigma"
    )
    stable = not any(c["fired"] for c in stability_criteria)

    capability_is_meaningful = bool(stable and cpk is not None)
    if not stable:
        reasons = [c["reason"] for c in stability_criteria if c["fired"]]
        interpretation = (
            "Capability indices describe a process that repeats itself, and an out-of-control "
            "process has no single distribution for them to describe. Here " + "; and ".join(reasons) + "."
        )
    elif cpk is None:
        interpretation = "Capability could not be computed (zero within-subgroup spread)."
    else:
        interpretation = (
            "Process is in statistical control; capability indices are valid and describe "
            "the underlying distribution."
        )

    if n_points >= 3:
        skew_val = float(scipy.stats.skew(series, bias=False))
        kurt_val = float(scipy.stats.kurtosis(series, bias=False))
    else:
        skew_val = float(scipy.stats.skew(series))
        kurt_val = float(scipy.stats.kurtosis(series))

    if abs(skew_val) > 1.0 or abs(kurt_val) > 1.0:
        normality_warning: str | None = (
            f"Distribution shows substantial departure from normality (skewness={skew_val:.2f}, "
            f"excess kurtosis={kurt_val:.2f}). Cpk on a strongly non-normal process misstates "
            "the fraction beyond specification limits."
        )
    else:
        normality_warning = None

    if cpk is None:
        verdict = None
    elif cpk >= 1.33:
        verdict = "capable"
    elif cpk >= 1.00:
        verdict = "marginal"
    else:
        verdict = "not capable"

    return to_jsonable({
        "n_points": n_points,
        "mean": mean_val,
        "usl": usl_val,
        "lsl": lsl_val,
        "target": target_val,
        "sigma_within": sigma_within,
        "sigma_overall": sigma_overall,
        "sigma_ratio": sigma_ratio,
        "sigma_ratio_note": sigma_ratio_note,
        "cp": cp,
        "cpk": cpk,
        "cpu": cpu,
        "cpl": cpl,
        "pp": pp,
        "ppk": ppk,
        "ppu": ppu,
        "ppl": ppl,
        "stable": stable,
        "stability_violations": stability_violations,
        "outlier_rate_beyond_3_sigma": outlier_rate,
        "stability_criteria": stability_criteria,
        "capability_is_meaningful": capability_is_meaningful,
        "interpretation": interpretation,
        "skewness": skew_val,
        "excess_kurtosis": kurt_val,
        "normality_warning": normality_warning,
        "verdict": verdict,
        "control_rules": rules_res,
    })


def calculate_gauge_rr(
    df: pd.DataFrame,
    part_col: str,
    operator_col: str,
    measurement_col: str,
) -> dict[str, Any]:
    """
    Calculate Crossed Gauge Repeatability and Reproducibility (Gauge R&R) by ANOVA.

    Implements the two-way ANOVA method with part-by-operator interaction according
    to the AIAG Measurement Systems Analysis (MSA) 4th edition standard.

    Measurement System First:
    -------------------------
    A gauge contributing 30% of the observed variation inflates sigma_overall, which
    deflates every cpk computed from the same data, and it widens every control limit,
    which is why an unacceptable gauge makes the SPC charts look calm. Measure the
    measurement system first.

    ANOVA Method & Interaction Pooling:
    -----------------------------------
    The ANOVA method separates total measurement error into:
    - Repeatability (Equipment Variation, EV): Variance within repeated measurements.
    - Reproducibility (Operator Variation, AV): Variance between operators.
    - Interaction: Part-by-operator interaction.
    - Part Variation (PV): True part-to-part variation.

    Following AIAG guidelines, if the interaction term's p-value is greater than 0.25,
    it is pooled into the repeatability error term, and variance components are
    recomputed (`interaction_pooled: True`).

    Variance Components vs Study Variation:
    ---------------------------------------
    - `percent_contribution`: Based on variance components (squared standard deviations).
      Sums to 100% across components (Gauge R&R % + Part Variation % = 100%).
    - `percent_study_variation`: Based on standard deviations (spread scale).
      Does NOT sum to 100%. The AIAG acceptance bands apply to this metric.

    AIAG Acceptance Bands (% Study Variation):
    ------------------------------------------
    - < 10%: "acceptable" (measurement system is adequate)
    - 10% - 30%: "marginal" (acceptable depending on the cost of application and repair)
    - > 30%: "unacceptable" (measurement system needs remediation)

    Number of Distinct Categories (NDC):
    ------------------------------------
    ndc = int(1.41 * (pv_sd / grr_sd)). Fewer than 5 means the gauge cannot even rank
    parts reliably, whatever the percentage says.

    Parameters:
    -----------
    df : pd.DataFrame
        Balanced measurement data containing parts, operators, and replicate readings.
    part_col : str
        Column name identifying the parts/samples measured.
    operator_col : str
        Column name identifying the appraisers/operators.
    measurement_col : str
        Column name containing the numeric measurement values.

    Returns:
    --------
    dict:
        JSON-ready dict. Never raises on invalid input; returns an error dict.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return {"error": "df must be a pandas DataFrame."}
    if df.empty:
        return {"error": "DataFrame is empty."}

    req_cols = (part_col, operator_col, measurement_col)
    missing = [c for c in req_cols if c not in df.columns]
    if missing:
        return {"error": f"DataFrame is missing column(s) {missing}. Columns present: {list(df.columns)}."}

    y_raw = df[measurement_col]
    y_numeric = pd.to_numeric(y_raw, errors="coerce")
    if y_numeric.isna().any():
        return {"error": f"Column '{measurement_col}' contains missing or non-numeric values."}

    parts = df[part_col].unique()
    operators = df[operator_col].unique()
    n_parts = len(parts)
    n_operators = len(operators)

    if n_parts < 2:
        return {"error": f"At least 2 parts are required for Gauge R&R, found {n_parts}."}
    if n_operators < 2:
        return {"error": f"At least 2 operators are required for Gauge R&R, found {n_operators}."}

    cell_counts = df.groupby([part_col, operator_col], observed=False).size()
    expected_cells = n_parts * n_operators
    if len(cell_counts) != expected_cells:
        present_set = set(cell_counts.index)
        all_possible = {(p, o) for p in parts for o in operators}
        missing_combos = sorted(all_possible - present_set)
        return {
            "error": (
                f"Unbalanced design: missing part-operator combinations ({len(missing_combos)} missing). "
                f"Offending missing cells: {missing_combos[:5]}."
            )
        }

    rep_counts = cell_counts.unique()
    if len(rep_counts) != 1:
        mode_rep = cell_counts.mode().iloc[0]
        offending = {
            f"({p}, {o})": int(cnt)
            for (p, o), cnt in cell_counts.items()
            if cnt != mode_rep
        }
        return {
            "error": (
                f"Unbalanced design: all part-operator cells must have the same number of replicates. "
                f"Expected {mode_rep} replicates; offending cells: {offending}."
            )
        }

    n_replicates = int(rep_counts[0])
    if n_replicates < 2:
        return {"error": f"At least 2 replicate measurements per cell are required, found {n_replicates}."}

    a = n_parts
    b = n_operators
    n = n_replicates
    total_n = a * b * n

    calc_df = df[[part_col, operator_col]].copy()
    calc_df["_y"] = y_numeric.to_numpy(dtype=float)

    grand_mean = float(calc_df["_y"].mean())
    ss_total = float(np.sum((calc_df["_y"].to_numpy() - grand_mean) ** 2))

    part_means = calc_df.groupby(part_col, observed=False)["_y"].mean()
    ss_parts = float(b * n * np.sum((part_means.to_numpy() - grand_mean) ** 2))
    df_parts = a - 1
    ms_parts = ss_parts / df_parts if df_parts > 0 else 0.0

    operator_means = calc_df.groupby(operator_col, observed=False)["_y"].mean()
    ss_operators = float(a * n * np.sum((operator_means.to_numpy() - grand_mean) ** 2))
    df_operators = b - 1
    ms_operators = ss_operators / df_operators if df_operators > 0 else 0.0

    cell_means = calc_df.groupby([part_col, operator_col], observed=False)["_y"].mean()
    ss_subtotal = float(n * np.sum((cell_means.to_numpy() - grand_mean) ** 2))

    ss_interaction = max(0.0, float(ss_subtotal - ss_parts - ss_operators))
    df_interaction = (a - 1) * (b - 1)
    ms_interaction = ss_interaction / df_interaction if df_interaction > 0 else 0.0

    ss_error = max(0.0, float(ss_total - ss_subtotal))
    df_error = a * b * (n - 1)
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    if ms_error > 0 and df_interaction > 0 and df_error > 0:
        f_interaction = ms_interaction / ms_error
        interaction_p_value = float(scipy.stats.f.sf(f_interaction, df_interaction, df_error))
    else:
        interaction_p_value = 1.0

    notes: list[str] = []

    if interaction_p_value > 0.25:
        interaction_pooled = True
        ss_pooled_error = ss_interaction + ss_error
        df_pooled_error = df_interaction + df_error
        ms_pooled_error = ss_pooled_error / df_pooled_error if df_pooled_error > 0 else 0.0

        var_repeatability = ms_pooled_error
        var_interaction = 0.0

        raw_var_operator = (ms_operators - ms_pooled_error) / (a * n)
        if raw_var_operator < 0:
            notes.append("Negative operator variance estimate clamped to zero.")
            var_operator = 0.0
        else:
            var_operator = float(raw_var_operator)

        var_reproducibility = var_operator

        raw_var_part = (ms_parts - ms_pooled_error) / (b * n)
        if raw_var_part < 0:
            notes.append("Negative part variance estimate clamped to zero.")
            var_part = 0.0
        else:
            var_part = float(raw_var_part)
    else:
        interaction_pooled = False
        var_repeatability = ms_error

        raw_var_interaction = (ms_interaction - ms_error) / n
        if raw_var_interaction < 0:
            notes.append("Negative interaction variance estimate clamped to zero.")
            var_interaction = 0.0
        else:
            var_interaction = float(raw_var_interaction)

        raw_var_operator = (ms_operators - ms_interaction) / (a * n)
        if raw_var_operator < 0:
            notes.append("Negative operator variance estimate clamped to zero.")
            var_operator = 0.0
        else:
            var_operator = float(raw_var_operator)

        var_reproducibility = var_operator + var_interaction

        raw_var_part = (ms_parts - ms_interaction) / (b * n)
        if raw_var_part < 0:
            notes.append("Negative part variance estimate clamped to zero.")
            var_part = 0.0
        else:
            var_part = float(raw_var_part)

    var_grr = var_repeatability + var_reproducibility
    var_total = var_grr + var_part

    sd_repeatability = float(np.sqrt(var_repeatability))
    sd_reproducibility = float(np.sqrt(var_reproducibility))
    sd_operator = float(np.sqrt(var_operator))
    sd_interaction = float(np.sqrt(var_interaction))
    sd_part = float(np.sqrt(var_part))
    sd_grr = float(np.sqrt(var_grr))
    sd_total = float(np.sqrt(var_total))

    if var_total > 0:
        percent_contribution = {
            "repeatability": float(100.0 * var_repeatability / var_total),
            "reproducibility": float(100.0 * var_reproducibility / var_total),
            "operator": float(100.0 * var_operator / var_total),
            "interaction": float(100.0 * var_interaction / var_total),
            "gauge_rr": float(100.0 * var_grr / var_total),
            "part_variation": float(100.0 * var_part / var_total),
        }
    else:
        percent_contribution = {
            "repeatability": 0.0,
            "reproducibility": 0.0,
            "operator": 0.0,
            "interaction": 0.0,
            "gauge_rr": 0.0,
            "part_variation": 0.0,
        }

    if sd_total > 0:
        percent_study_variation = {
            "repeatability": float(100.0 * sd_repeatability / sd_total),
            "reproducibility": float(100.0 * sd_reproducibility / sd_total),
            "operator": float(100.0 * sd_operator / sd_total),
            "interaction": float(100.0 * sd_interaction / sd_total),
            "gauge_rr": float(100.0 * sd_grr / sd_total),
            "part_variation": float(100.0 * sd_part / sd_total),
        }
    else:
        percent_study_variation = {
            "repeatability": 0.0,
            "reproducibility": 0.0,
            "operator": 0.0,
            "interaction": 0.0,
            "gauge_rr": 0.0,
            "part_variation": 0.0,
        }

    if sd_grr > 0:
        ndc: int | None = int(1.41 * (sd_part / sd_grr))
    else:
        ndc = None

    grr_study_pct = percent_study_variation["gauge_rr"]
    if grr_study_pct < 10.0:
        verdict = "acceptable"
    elif grr_study_pct <= 30.0:
        verdict = "marginal"
    else:
        verdict = "unacceptable"

    if ndc is not None and ndc >= 5:
        ndc_verdict = "acceptable"
    else:
        ndc_verdict = "unacceptable"

    return to_jsonable({
        "n_parts": n_parts,
        "n_operators": n_operators,
        "n_replicates": n_replicates,
        "n_total": total_n,
        "interaction_p_value": interaction_p_value,
        "interaction_pooled": interaction_pooled,
        "variance_components": {
            "repeatability": float(var_repeatability),
            "reproducibility": float(var_reproducibility),
            "operator": float(var_operator),
            "interaction": float(var_interaction),
            "gauge_rr": float(var_grr),
            "part_variation": float(var_part),
            "total_variation": float(var_total),
        },
        "standard_deviations": {
            "repeatability": float(sd_repeatability),
            "reproducibility": float(sd_reproducibility),
            "operator": float(sd_operator),
            "interaction": float(sd_interaction),
            "gauge_rr": float(sd_grr),
            "part_variation": float(sd_part),
            "total_variation": float(sd_total),
        },
        "percent_contribution": percent_contribution,
        "percent_study_variation": percent_study_variation,
        "ndc": ndc,
        "verdict": verdict,
        "ndc_verdict": ndc_verdict,
        "notes": notes,
    })
