from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import scipy.stats

from ._utils import _finite_float, to_jsonable
from .constants import (
    CAPABILITY_OUTLIER_RATE_LIMIT,
    CAPABILITY_SIGMA_RATIO_LIMIT,
    CONTROL_RULES_WESTERN_ELECTRIC,
)
from .spc import _baseline_stats, calculate_control_rules


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
    - `stable` uses two length-independent criteria instead, either of which fails it:
      `sigma_ratio` above CAPABILITY_SIGMA_RATIO_LIMIT (1.20), or more than
      CAPABILITY_OUTLIER_RATE_LIMIT (1%) of points beyond 3 sigma. Together they fire
      on roughly 2% of healthy series while catching a 3-sigma drift 97.7% of the
      time, a 2-sigma step 99.3%, and a mid-series variance doubling 47.3%.
    - **`baseline_n` is not free.** It restricts the sigma estimate to the first N
      readings, which is right when a Phase I period has to be fenced off from a
      process that later wandered - but every reading outside the window is then
      judged against a sigma estimated from N points. On 300 genuinely in-control
      points, `baseline_n=40` reports `stable: False` 17.3% of the time against
      2.0% when the whole series is used, because a noisy moving-range sigma makes
      ordinary points look extreme. Use it when the process really does have a
      Phase I; leave it out when the series is its own baseline.
    - **Known blind spot, stated rather than discovered later:** a variance change
      lifts `sigma_within` and `sigma_overall` together, so the ratio stays near 1
      and only the outlier-rate criterion sees it - at 47.3%, not 100%. A slow
      1-sigma drift is missed by both. Read `stability_violations` and the chart
      itself when the decision matters.
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
    _, sigma_within = _baseline_stats(series, baseline_n)
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

    # 1. Variation arriving BETWEEN subgroups rather than within them. Measured
    #    on healthy data the ratio sits at 0.99-1.00 whatever the length, and a
    #    limit of 1.20 never fired in 400 trials at n >= 100 while catching a
    #    3-sigma drift 97.7% of the time and a 2-sigma step 99.3%.
    ratio_exceeded = sigma_ratio is not None and sigma_ratio > CAPABILITY_SIGMA_RATIO_LIMIT

    # 2. Points beyond 3 sigma at a rate chance does not explain. This covers the
    #    ratio's blind spot: a variance change inflates sigma_within and
    #    sigma_overall together, leaving the ratio near 1. Chance puts 0.27% of
    #    points outside; a 1% limit fires on 1-2.7% of healthy series and on
    #    47.3% of series whose variance doubles halfway through, where the ratio
    #    alone manages 0.7%.
    beyond_3s = rules_res.get("violations", {}).get("1", []) if "error" not in rules_res else []
    outlier_rate = (len(beyond_3s) / len(series)) if len(series) else 0.0
    outliers_exceeded = outlier_rate > CAPABILITY_OUTLIER_RATE_LIMIT

    stable = not (ratio_exceeded or outliers_exceeded)

    capability_is_meaningful = bool(stable and cpk is not None)
    if not stable:
        reasons = []
        if ratio_exceeded:
            reasons.append(
                f"sigma_overall is {sigma_ratio:.2f} times sigma_within (limit "
                f"{CAPABILITY_SIGMA_RATIO_LIMIT}), so variation is arriving between subgroups "
                f"rather than within them"
            )
        if outliers_exceeded:
            reasons.append(
                f"{outlier_rate:.1%} of points lie beyond 3 sigma (limit "
                f"{CAPABILITY_OUTLIER_RATE_LIMIT:.0%}; chance alone puts 0.27% there)"
            )
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
