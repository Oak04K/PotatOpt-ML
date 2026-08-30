from __future__ import annotations

from typing import Any  # loose return-type annotations for JSON-shaped dicts

import numpy as np  # arrays + math for SPC/EWMA/CUSUM limits, downcasting, anomaly scoring
import pandas as pd  # DataFrame/Series is the data contract for every public function

from .constants import CALIBRATION_DEFAULT_BINS, CALIBRATION_ECE_LIMIT


def check_calibration(y_true: Any, y_prob: Any, n_bins: int = CALIBRATION_DEFAULT_BINS) -> dict[str, Any]:
    """
    Ask whether a predicted probability means what it says.

    A model is CALIBRATED when, of every batch it scored 0.30, close to 30 per cent
    really did fail. Discrimination and calibration are different properties and a
    model can have one without the other: ranking every failure above every healthy
    machine gives a perfect AUC while the scores themselves are all pressed up near
    1.0, which is an excellent model with meaningless numbers on it.

    This is checked here because the cost layer depends on it. `optimize_threshold`
    picks the cut that costs least on the validation rows; the cut it finds is still
    the cheapest cut on that score, calibrated or not. What miscalibration breaks is
    everything a reader then wants to do with the number: a threshold of 0.30 cannot
    be described as "act at a 30 per cent chance of failure", the expected cost of a
    single call-out cannot be quoted, and the threshold does not survive being moved
    to a line with a different failure rate. Report ECE next to the saving so the
    reader knows which of those claims the model can carry.

    The measurement bins the predictions, and in each bin compares mean predicted
    probability against the fraction that actually turned out positive:

        ECE = sum over bins of (bin size / total) * |predicted - observed|
        MCE = the largest of those gaps in any bin

    Parameters:
    -----------
    y_true : array-like
        Binary outcomes, coerced to 0/1. Exactly two distinct values are required;
        a single-class sample cannot say anything about calibration.
    y_prob : array-like
        Predicted probability of the positive class, in [0, 1]. Pass the column of
        `predict_proba` for the positive class, not the hard 0/1 prediction.
    n_bins : int
        Number of equal-width bins across [0, 1]. Empty bins are skipped and
        `n_bins_used` reports how many carried data - with few rows and clustered
        scores, most bins are empty and ECE rests on very little.

    Returns:
    --------
    dict:
        `brier_score` (lower is better, 0 is perfect), `brier_skill_score` against
        always predicting the base rate (0 means no better than that, negative means
        worse), `expected_calibration_error`, `max_calibration_error`,
        `is_well_calibrated`, and the per-bin table behind the numbers.
        Returns `{"error": ...}` and never raises.
    """
    try:
        bins = int(n_bins)
    except (TypeError, ValueError):
        return {"error": f"n_bins must be a whole number, got {n_bins!r}."}
    if bins < 2:
        return {"error": f"n_bins must be at least 2, got {n_bins!r}."}

    try:
        truth = pd.Series(y_true).reset_index(drop=True)
        prob = pd.to_numeric(pd.Series(y_prob).reset_index(drop=True), errors="coerce")
    except (TypeError, ValueError):
        return {"error": "y_true and y_prob must both be array-like."}

    if len(truth) != len(prob):
        return {"error": f"y_true and y_prob must be the same length ({len(truth)} vs {len(prob)})."}
    if len(truth) == 0:
        return {"error": "y_true is empty."}

    # A non-numeric label set (pass/fail, OK/NG) is normal on a shop floor. Sort the
    # two labels so the mapping is deterministic and the caller can predict which
    # one became the positive class.
    labels = pd.unique(truth.dropna())
    if len(labels) != 2:
        return {"error": f"Calibration needs exactly two outcome classes, found {len(labels)}."}
    if pd.api.types.is_numeric_dtype(truth) and set(pd.Series(labels).astype(float)) <= {0.0, 1.0}:
        outcome = pd.to_numeric(truth, errors="coerce")
        positive_label = 1
    else:
        ordered = sorted(labels, key=str)
        positive_label = ordered[-1]
        outcome = (truth == positive_label).astype(float)

    usable = outcome.notna() & prob.notna() & np.isfinite(prob)
    outcome = outcome[usable].astype(float).to_numpy()
    prob = prob[usable].astype(float).to_numpy()
    if outcome.size == 0:
        return {"error": "No rows left after dropping missing or non-finite values."}
    if prob.min() < 0.0 or prob.max() > 1.0:
        return {"error": f"y_prob must lie in [0, 1], got range [{prob.min()}, {prob.max()}]."}

    n_rows = int(outcome.size)
    base_rate = float(outcome.mean())
    brier = float(np.mean((prob - outcome) ** 2))
    # Always predicting the base rate is the honest floor to beat. Its Brier score
    # is p(1-p), so the skill score below says how much the model added over
    # knowing nothing but how often the line fails.
    brier_reference = base_rate * (1.0 - base_rate)
    skill = float(1.0 - brier / brier_reference) if brier_reference > 0 else None

    edges = np.linspace(0.0, 1.0, bins + 1)
    # right=False keeps bins half-open; the top edge is folded back in so a score of
    # exactly 1.0 lands in the last bin instead of a bin of its own.
    index = np.clip(np.digitize(prob, edges[1:-1], right=False), 0, bins - 1)

    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for b in range(bins):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        predicted = float(prob[mask].mean())
        observed = float(outcome[mask].mean())
        gap = abs(predicted - observed)
        ece += (count / n_rows) * gap
        mce = max(mce, gap)
        rows.append({
            "bin_lower": float(edges[b]),
            "bin_upper": float(edges[b + 1]),
            "count": count,
            "mean_predicted": predicted,
            "observed_rate": observed,
            "gap": float(predicted - observed),
        })

    well_calibrated = bool(ece <= CALIBRATION_ECE_LIMIT)
    if well_calibrated:
        interpretation = (
            f"Predicted probabilities track observed rates to within {ece:.3f} on average; "
            f"the threshold and the per-unit cost figures can be read as probabilities."
        )
    else:
        direction = "over-confident" if sum(r["gap"] * r["count"] for r in rows) > 0 else "under-confident"
        interpretation = (
            f"Average gap between predicted and observed is {ece:.3f}, above the {CALIBRATION_ECE_LIMIT} "
            f"guideline, and the model is {direction} overall. Ranking-based results (AUC, the chosen "
            f"threshold) still hold; do not quote the scores as probabilities or move the threshold to "
            f"a line with a different failure rate without re-tuning."
        )

    return {
        "n_rows": n_rows,
        "n_bins": bins,
        "n_bins_used": len(rows),
        "positive_label": str(positive_label),
        "base_rate": base_rate,
        "brier_score": brier,
        "brier_skill_score": skill,
        "expected_calibration_error": float(ece),
        "max_calibration_error": float(mce),
        "is_well_calibrated": well_calibrated,
        "ece_limit": CALIBRATION_ECE_LIMIT,
        "bins": rows,
        "interpretation": interpretation,
    }
