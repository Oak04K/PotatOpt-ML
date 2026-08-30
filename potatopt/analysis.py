from __future__ import annotations

from typing import Any  # loose return-type annotations for JSON-shaped dicts

import numpy as np  # arrays + math for summary statistics
import pandas as pd  # DataFrame/Series is the data contract for every public function

from ._utils import to_jsonable
from .constants import DEFAULT_RANDOM_STATE, SEED_SWEEP_DEFAULT
from .data import audit_data_quality, inspect_data, split_data_three_way
from .engine import PotatOptEngine


def auto_analyze(data: str | pd.DataFrame, target: str, cost_scrap: float = 500.0,
                 cost_fa: float = 150.0, cost_insp: float = 20.0, task: str = "auto",
                 time_budget: int = 30, n_jobs: int = -1, top_features: int | None = 10,
                 save_to: str | None = None, val_size: float = 0.2,
                 test_size: float = 0.2,
                 random_state: int = DEFAULT_RANDOM_STATE) -> dict[str, Any]:
    """
    Run the whole pipeline in one call and return one JSON-ready report.

    This is the front door. It profiles the data, gates it on quality, splits it
    three ways, searches for a model, tunes the decision threshold on the
    validation partition, scores the untouched test partition, prices the result
    in money, and ranks the features that drove it.

    It hides ceremony, not engineering decisions. Splitting, encoding, scaling and
    downcasting have one correct answer and are done for you; the cost of a missed
    defect, a false alarm and an inspection are numbers only the engineer on the
    floor knows, so they stay in the signature.

    Every step here is also available on its own - `inspect_data`,
    `split_data_three_way`, `PotatOptEngine` and the rest are unchanged. Use them
    directly when you want to control the pipeline yourself.

    Parameters:
    -----------
    data : str or pandas.DataFrame
        Path to a CSV file, or a DataFrame already in memory.
    target : str
        Name of the column to predict.
    cost_scrap, cost_fa, cost_insp : float
        Cost of a missed defect, a false alarm, and one inspection. Used to pick
        the decision threshold and to price the outcome.
    save_to : str or None
        If given, the fitted engine is saved there with its SHA-256 integrity hash.
    random_state : int
        Seed for the split and the model search. One call is one seed and therefore
        one sample of the score; `run_seed_sweep` runs several and reports the
        spread, which is what tells you whether the number below is real.

    Returns:
    --------
    dict:
        Always JSON-serialisable and **never raises**. On failure `ok` is False
        and `error` carries a readable message, so a calling agent or a beginner
        gets a sentence rather than a traceback. `top_features_note` carries the
        human-readable reason when `top_features` is empty.
    """
    report = {
        "ok": False,
        "error": None,
        "task": None,
        "rows": 0,
        "features": 0,
        "data_quality": None,
        "split": None,
        "model": None,
        "metrics": None,
        "cost": None,
        "threshold": None,
        "calibration": None,
        "random_state": random_state,
        "top_features": None,
        "top_features_note": None,
        "saved_to": None,
    }

    try:
        # Accept the two shapes a caller realistically has: a file on disk, or a
        # frame already in memory.
        if isinstance(data, str):
            frame = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            frame = data.copy()
        else:
            report["error"] = (
                f"data must be a path to a CSV file or a pandas DataFrame, "
                f"got {type(data).__name__}."
            )
            return report

        if target not in frame.columns:
            report["error"] = (
                f"Target column '{target}' is not in the data. "
                f"Columns available: {list(frame.columns)[:20]}"
            )
            return report

        # Profile and quality-gate the raw data before a model ever sees it.
        profile = inspect_data(frame, target)
        audit = audit_data_quality(frame, target)

        resolved_task = task
        if resolved_task == "auto":
            resolved_task = profile.get("recommended_task", "classification")

        # Three-way split: the threshold is tuned on validation so the test
        # partition stays untouched until it is reported.
        X_train, X_val, X_test, y_train, y_val, y_test = split_data_three_way(
            frame, target, task=resolved_task, val_size=val_size, test_size=test_size,
            random_state=random_state,
        )

        engine = PotatOptEngine(
            task=resolved_task,
            time_budget=time_budget,
            cost_sensitive_weighting=True,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        engine.fit(X_train, y_train)

        # A decision threshold and a cost of quality only mean something when the
        # model is deciding between classes.
        threshold_info = None
        cost = None
        calibration = None
        if engine.task == "classification":
            optimal = engine.optimize_threshold(
                X_val, y_val, cost_scrap=cost_scrap, cost_fa=cost_fa, cost_insp=cost_insp
            )
            # Measured on the same validation rows the threshold was chosen from,
            # because that is the sample the cost argument below rests on.
            calibration = engine.check_calibration(X_val, y_val)
            threshold_info = {
                "optimal": optimal,
                "tuned_on": "validation",
                "n_validation_rows": len(X_val),
            }
            cost = engine.calculate_cost_of_quality(
                X_test, y_test, cost_scrap=cost_scrap, cost_fa=cost_fa, cost_insp=cost_insp
            )

        metrics = engine.evaluate(X_test, y_test)
        training = engine.get_training_report()
        explanation = engine.explain_predictions(X_test, top_k=top_features)

        if save_to:
            engine.save(save_to)
            report["saved_to"] = save_to

        report.update({
            "ok": True,
            "task": engine.task,
            "rows": len(frame),
            "features": X_train.shape[1],
            "data_quality": {
                "dqs": audit.get("dqs"),
                "grade": audit.get("grade"),
                "verdict": audit.get("verdict"),
                "issues": audit.get("issues"),
            },
            "split": {
                "train": len(X_train),
                "validation": len(X_val),
                "test": len(X_test),
            },
            "model": {
                "name": metrics.get("best_model_name"),
                "metric_optimized": training.get("metric_optimized"),
                "validation_score": training.get("validation_score"),
                "anomaly_fallback": engine.is_anomaly_model,
            },
            "metrics": metrics,
            "cost": cost,
            "threshold": threshold_info,
            "calibration": calibration,
            "random_state": random_state,
            "top_features": explanation.get("feature_attributions") if explanation.get("available") else [],
            "top_features_note": explanation.get("reason"),
        })
        return to_jsonable(report)
    except Exception as exc:  # noqa: BLE001
        # The front door must hand back a sentence, never a traceback: the caller
        # is a beginner or an agent, and both need something they can act on.
        report["error"] = f"{type(exc).__name__}: {exc}"
        return to_jsonable(report)


def run_seed_sweep(data: str | pd.DataFrame, target: str, seeds: Any = SEED_SWEEP_DEFAULT, **kwargs: Any) -> dict[str, Any]:
    """
    Run `auto_analyze` once per seed and report the spread, not just a number.

    One run of a pipeline is one sample. The split decides which rows are in Test
    and the search is stochastic, so the same data and the same code produce a
    range of scores - and on the small, imbalanced datasets this library targets
    that range is wide. Reporting the single seed that happened to run is how a
    result gets published that nobody can reproduce and that does not survive
    contact with next month's data.

    So the number this returns that matters is not the mean. It is the spread. A
    difference between two configurations smaller than the spread of either one is
    not a finding, and `stability_note` says so in the output rather than leaving
    the reader to work it out.

    Cost: each seed is a full run, so wall-clock is roughly `len(seeds)` times
    `time_budget`. Trim `seeds` or `time_budget` before running this on a laptop.

    Parameters:
    -----------
    data, target : as `auto_analyze`.
    seeds : sequence of int
        Seeds to run. Must be non-empty and free of duplicates - repeating a seed
        repeats an identical run and would narrow the spread for no reason.
    **kwargs :
        Passed straight through to `auto_analyze` (time_budget, costs, sizes...).
        `random_state` is not accepted here; that is what `seeds` is.

    Returns:
    --------
    dict:
        `runs` (one entry per seed, each with its seed and whether it succeeded),
        `summary` (mean / std / min / max / spread for every numeric metric that
        every successful run reported), and `stability_note`. JSON-serialisable,
        and never raises.
    """
    if "random_state" in kwargs:
        return {"error": "Pass the seeds through `seeds`; `random_state` would fix every run to one value."}

    try:
        seed_list = [int(s) for s in seeds]
    except (TypeError, ValueError):
        return {"error": f"seeds must be a sequence of whole numbers, got {seeds!r}."}
    if not seed_list:
        return {"error": "seeds is empty; at least one seed is required."}
    if len(set(seed_list)) != len(seed_list):
        return {"error": f"seeds contains duplicates {seed_list!r}; a repeated seed repeats an identical run."}

    runs: list[dict[str, Any]] = []
    collected: list[dict[str, float]] = []
    for seed in seed_list:
        result = auto_analyze(data, target, random_state=seed, **kwargs)
        runs.append({
            "seed": seed,
            "ok": bool(result.get("ok")),
            "error": result.get("error"),
            "best_model": (result.get("model") or {}).get("name"),
        })
        if not result.get("ok"):
            continue

        # Only flat numeric values are comparable across runs. Nested blocks
        # (the per-class report, the bin table) are left out rather than
        # averaged into something that reads like a metric but is not one.
        flat: dict[str, float] = {}
        for block in ("metrics", "cost"):
            section = result.get(block)
            if not isinstance(section, dict):
                continue
            for key, value in section.items():
                if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
                    continue
                if not np.isfinite(float(value)):
                    continue
                flat[f"{block}.{key}"] = float(value)
        calibration = result.get("calibration")
        if isinstance(calibration, dict) and isinstance(calibration.get("expected_calibration_error"), (int, float)):
            flat["calibration.expected_calibration_error"] = float(calibration["expected_calibration_error"])
        collected.append(flat)

    n_ok = len(collected)
    if n_ok == 0:
        return {
            "seeds": seed_list,
            "n_seeds": len(seed_list),
            "n_ok": 0,
            "runs": runs,
            "summary": {},
            "stability_note": "Every seed failed; see `runs` for the reason from each.",
        }

    # A metric only one run produced cannot be given a spread, so the summary is
    # restricted to keys present in every successful run.
    shared = set(collected[0])
    for flat in collected[1:]:
        shared &= set(flat)

    summary: dict[str, Any] = {}
    for key in sorted(shared):
        values = np.array([flat[key] for flat in collected], dtype=float)
        # ddof=1: these seeds are a sample of the seeds that could have been run,
        # not the population. With one run there is no spread to report.
        std = float(values.std(ddof=1)) if n_ok > 1 else 0.0
        summary[key] = {
            "mean": float(values.mean()),
            "std": std,
            "min": float(values.min()),
            "max": float(values.max()),
            "spread": float(values.max() - values.min()),
            "n": n_ok,
        }

    if n_ok < 2:
        note = "Only one seed succeeded, so no spread could be measured; this is still a single-seed result."
    else:
        note = (
            f"Report mean +/- std over {n_ok} seeds, not the best run. Any improvement smaller than "
            f"`spread` for the metric in question is inside the noise of the seed and is not a result."
        )

    return to_jsonable({
        "seeds": seed_list,
        "n_seeds": len(seed_list),
        "n_ok": n_ok,
        "runs": runs,
        "summary": summary,
        "stability_note": note,
    })
