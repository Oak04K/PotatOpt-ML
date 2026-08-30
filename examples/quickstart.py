"""
Runs a complete condition-based maintenance pipeline on the AI4I dataset.

This quickstart script downloads the data (if missing), profiles it, splits it,
trains a model to predict machine failures, optimises the decision threshold on
maintenance costs, scores the results, calculates financial impact, explains the
predictions, checks for fleet drift, and charts a sensor stream.

It takes about a minute on a typical laptop CPU.

Usage:
    python examples/quickstart.py
    python examples/quickstart.py --time-budget 60

Note: This script requires the `automl` and `xai` extras because it trains a
model and explains it. Install them via `pip install potatopt[automl,xai]`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Two entries, for two ways of running this file. The script's own folder makes
# `ai4i_dataset` importable no matter which directory the reader is standing in;
# the repository root covers a source checkout, where potatopt.py sits beside
# examples/ and has not been pip-installed.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

try:
    import potatopt as po
except ImportError:
    print("potatopt not found. Please install via pip install potatopt[automl,xai]")
    sys.exit(1)

from ai4i_dataset import ASSET_COLUMN, TARGET_COLUMN, load_ai4i


def section(number: int, title: str) -> None:
    """
    Prints a formatted section header to make the output readable as a report.
    """
    print()
    header = f"STEP {number} - {title}"
    print(header)
    print("-" * len(header))


def parse_args() -> argparse.Namespace:
    """
    Parses command-line arguments for the quickstart script.
    """
    parser = argparse.ArgumentParser(description="PotatOpt Quickstart on AI4I Dataset")
    parser.add_argument(
        "--time-budget",
        type=int,
        default=30,
        help="seconds FLAML may spend searching",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="path to an existing ai4i2020.csv",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="do not fetch the dataset if it is missing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep FLAML's per-iteration search log, which this script hides by default",
    )
    return parser.parse_args()


def main() -> None:
    """
    Executes the ten steps of the condition-based maintenance pipeline.
    """
    args = parse_args()

    section(1, "Load the machine log")
    raw = load_ai4i(args.data, download=not args.no_download, drop_leaky=False)
    df = load_ai4i(args.data, download=False, drop_leaky=True)
    failure_rate = (raw[TARGET_COLUMN].sum() / len(raw)) * 100
    print(f"Shape of the raw log: {raw.shape}")
    print(f"Failure rate: {failure_rate:.2f}%")
    print(
        "Seven columns were removed because identifiers memorise rows, and the five "
        "failure-mode flags are only known after the failure has happened."
    )

    section(2, "Profile it before training anything")
    profile = po.inspect_data(df, TARGET_COLUMN)
    print(json.dumps(profile, indent=2))
    print(
        "The data-quality score gates the pipeline, so a bad file stops here instead "
        "of becoming a confident model."
    )

    section(3, "Split three ways")
    X_train, X_val, X_test, y_train, y_val, y_test = po.split_data_three_way(
        df, TARGET_COLUMN
    )
    print(f"Train     : {len(X_train)} rows, {y_train.sum()} failures")
    print(f"Validation: {len(X_val)} rows, {y_val.sum()} failures")
    print(f"Test      : {len(X_test)} rows, {y_test.sum()} failures")
    print(
        "The validation set exists so the decision threshold is never tuned on the rows "
        "used to report the result."
    )

    section(4, "Train")
    try:
        engine = po.PotatOptEngine(
            task="classification",
            time_budget=args.time_budget,
            cost_sensitive_weighting=True,
            # Quiet is the library default; --verbose hands FLAML's per-iteration
            # search log back for anyone asking why this model was chosen.
            verbose=3 if args.verbose else 0,
        )
        t0 = time.perf_counter()
        engine.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        report = engine.get_training_report()
        print(f"Elapsed time: {elapsed:.1f} seconds")
        print(f"Chosen model: {report.get('best_estimator', 'unknown')}")
    except ImportError:
        print("Missing optional dependencies. Run: pip install potatopt[automl,xai]")
        sys.exit(1)

    section(5, "Tune the threshold on maintenance cost")
    threshold = engine.optimize_maintenance_threshold(X_val, y_val)
    print(f"Threshold: {threshold}")
    print(
        "0.5 is a statistical default, not an economic one - the cost of a surprise "
        "breakdown against a planned repair is what sets it."
    )

    section(6, "Score the held-out test set")
    results = engine.evaluate(X_test, y_test)
    print(f"Accuracy : {results.get('accuracy')}")
    print(f"Precision: {results.get('precision')}")
    print(f"Recall   : {results.get('recall')}")
    print(f"F1       : {results.get('f1')}")
    print(f"ROC AUC  : {results.get('roc_auc')}")
    print(f"Confusion Matrix: {results.get('confusion_matrix')}")
    print(f"Threshold Leakage Warning: {results.get('threshold_leakage_warning')}")
    print(
        "That warning is False because step 5 used the validation set."
    )

    section(7, "Convert the confusion matrix into money")
    cost = engine.calculate_maintenance_cost(X_test, y_test)
    print(f"Run-to-failure cost      : {cost.get('run_to_failure_cost')}")
    print(f"Predictive cost          : {cost.get('predictive_cost')}")
    print(f"Cost savings             : {cost.get('cost_savings')}")
    print(f"Savings percentage       : {cost.get('savings_percentage')}")
    print(f"Breakdowns avoided       : {cost.get('breakdowns_avoided')}")
    print(f"Unplanned breakdowns     : {cost.get('unplanned_breakdowns')}")
    print(f"Breakdown avoidance rate : {cost.get('breakdown_avoidance_rate')}")
    print(f"Wasted inspections       : {cost.get('wasted_inspections')}")
    print(
        "A model that flags every machine reaches perfect recall and still loses money, "
        "because each wasted inspection is paid for whether or not anything was wrong - "
        "which is why cost_savings is reported next to the recall, never instead of it."
    )

    section(8, "Ask which sensor drove the prediction")
    try:
        explanation = engine.explain_predictions(X_test, top_k=5)
        if explanation.get("available"):
            features = explanation.get("feature_attributions", [])
            for feat in features:
                name = feat.get("feature", "unknown")
                shap = feat.get("mean_abs_shap", 0.0)
                print(f"{name}: {shap:.3f}")
        else:
            print(explanation.get("reason", "Explanation unavailable"))

        if explanation.get("additivity_check_relaxed"):
            print(
                "The additivity check had to be relaxed, so read the ranking as "
                "indicative."
            )
    except ImportError:
        print("Missing optional dependencies. Run: pip install potatopt[automl,xai]")
        sys.exit(1)

    section(9, "Watch the fleet for drift")
    history = raw.head(8000)
    batch = raw.tail(2000)
    columns = [
        ASSET_COLUMN,
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]
    drift = po.check_asset_drift(history[columns], batch[columns], asset_col=ASSET_COLUMN)
    print(f"Assets checked: {drift['assets_checked']}, drifted: {drift['assets_drifted']}")
    for asset, info in drift["per_asset"].items():
        if info.get("status") != "checked":
            print(f"{asset} | not judged: {info.get('status')}")
            continue
        rows = info.get("rows_batch", 0)
        max_psi = info.get("max_psi") or 0.0
        drifted = ", ".join(info.get("drifted_features", {})) or "none"
        print(f"{asset} | Rows: {rows} | Max PSI: {max_psi:.3f} | Drifted: {drifted}")
    print(
        "When EVERY asset drifts on the same feature it is the environment that moved, "
        "not one machine breaking - here the last 2,000 rows run about 2 K cooler than "
        "the first 8,000 across all three product grades. A single asset drifting alone "
        "is the one worth a work order."
    )

    section(10, "Chart a sensor stream")
    stream_head = raw.head(600)
    for sensor in ["Torque [Nm]", "Process temperature [K]"]:
        chart = po.calculate_ewma_chart(stream_head[sensor], baseline_n=100)
        # lag-1 is None when it cannot be judged - a constant or a two-point
        # series - and None is not 0.0, so it is never formatted as one.
        lag1 = chart["lag1_autocorrelation"]
        lag1_text = "not measurable" if lag1 is None else f"{lag1:+.3f}"
        print(f"{sensor}:")
        print(f"  Violations             : {len(chart['violations'])} of {chart['n_points']}")
        print(f"  Sigma (moving range)   : {chart['sigma']:.4f}")
        print(f"  Lag-1 autocorrelation  : {lag1_text}")
        if chart["autocorrelation_warning"]:
            print(f"  Warning: {chart['autocorrelation_warning']}")

    print(
        "Torque readings are independent so the chart is quiet, while the temperature "
        "stream is a random walk (lag-1 near 1.0) where moving-range sigma is far too "
        "small and nearly every point crosses the limit - the warning is the chart "
        "telling you it is the wrong tool for that series, not a fleet of failing machines."
    )

    print()
    print(
        "Everything above came from one CSV and about sixty lines of calls. "
        "See `benchmarks/token_cost.py` for the same pipeline written by hand."
    )


if __name__ == "__main__":
    main()
