"""
Runtime cost benchmark for PotatOpt vs hand-written scikit-learn vs PyCaret.

WHAT THIS BENCHMARK CLAIMS, AND HOW IT AVOIDS LYING

Claim under test: PotatOpt does the whole predictive-maintenance job inside a
low-spec memory budget, and does it on the CURRENT Python.

Three honesty rules the design must follow:
1. Every variant trains on the SAME rows. The split is computed by one shared
   function, from the same seed, in every subprocess.
2. Wall-clock time is NOT the headline. PotatOpt's search time is whatever
   time_budget was set to, so a "faster" claim would be an artefact of the knob.
   The comparable numbers are peak memory and the score reached inside that
   budget; the table prints time as context, labelled as budget-bound.
3. Both sides run single-threaded (n_jobs=1). Parallelism changes memory and
   time on both sides and would otherwise depend on the machine's core count.

Run it:
    python benchmarks/runtime_cost.py
    python benchmarks/runtime_cost.py --time-budget 60 --json results.json

Note that it needs the `automl` extra and `psutil`, and that the AI4I 2020
dataset is downloaded on first use by examples/ai4i_dataset.py.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

try:
    import psutil
except ImportError:
    psutil = None

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

COST_BREAKDOWN = 50_000.0
COST_PLANNED = 8_000.0
COST_INSPECTION = 1_500.0
DEFAULT_TIME_BUDGET = 30
RSS_SAMPLE_SECONDS = 0.05
RESULT_PREFIX = "RESULT "

# Checked against PyPI metadata on 2026-08-28. PyCaret 3.3.2 is the latest
# release; it declares Python 3.9, 3.10 and 3.11 only, and pins
# numpy<1.27, pandas<2.2.0 and scipy<=1.11.4. None of those publish wheels
# for Python 3.13 or newer, so pip falls back to building numpy from source
# and fails on a machine without a C toolchain - which is what happened when
# this was attempted on Python 3.14.2. The comparison is therefore reported
# as an environment fact, not silently skipped.
PYCARET_FACTS = {
    "version_checked": "3.3.2",
    "checked_on": "2026-08-28",
    "declared_python": ["3.9", "3.10", "3.11"],
    "pins": ["numpy<1.27,>=1.21", "pandas<2.2.0", "scipy<=1.11.4"],
}

# Peak memory on its own does not say WHERE the memory went, and the answer
# turns out to be most of it: the imports each variant needs before a single row
# is read. Measuring them separately keeps the table from being read as "the
# library's own code is heavier", which is not what the numbers say. Each
# snippet loads exactly what its variant loads and nothing else - the PotatOpt
# one touches `AutoML` on purpose, because that attribute is what pulls FLAML in
# from the lazy backend loader.
IMPORT_SNIPPETS = {
    "potatopt": "import potatopt; potatopt.AutoML",
    "sklearn": (
        "import numpy, pandas; "
        "from sklearn.ensemble import RandomForestClassifier; "
        "from sklearn.impute import SimpleImputer; "
        "from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score; "
        "from sklearn.preprocessing import OrdinalEncoder, StandardScaler; "
        "from sklearn.utils.class_weight import compute_sample_weight"
    ),
    "pycaret": "import pycaret.classification",
}

# ------------------------------------------------------------------------------
# Shared data loading
# ------------------------------------------------------------------------------

def load_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                          pd.Series, pd.Series, pd.Series]:
    """
    Loads the AI4I dataset and splits it identically for all variants.
    
    Both variants call this and nothing else, so honesty rule 1 holds by
    construction.
    """
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "examples"))

    from ai4i_dataset import TARGET_COLUMN, load_ai4i

    import potatopt

    df = load_ai4i()
    return potatopt.split_data_three_way(df, TARGET_COLUMN)

# ------------------------------------------------------------------------------
# The shared cost objective
# ------------------------------------------------------------------------------

def maintenance_cost(tp: int, fp: int, fn: int) -> float:
    """
    Computes the total financial cost of the maintenance policy.
    
    A true positive is charged BOTH the inspection that found it and the
    planned repair that followed; a false positive is charged the inspection only,
    because an engineer looks before replacing a part. This mirrors
    potatopt.calculate_maintenance_savings exactly, so the two variants are scored
    by the same money, not by two different definitions of it.
    """
    return (tp * (COST_PLANNED + COST_INSPECTION) + 
            fn * COST_BREAKDOWN + 
            fp * COST_INSPECTION)

# ------------------------------------------------------------------------------
# Variant 1
# ------------------------------------------------------------------------------

def run_potatopt(time_budget: int) -> dict[str, Any]:
    """Runs the predictive-maintenance job using the PotatOpt step-by-step API."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    import potatopt as po

    # Both sides run single-threaded (n_jobs=1) to ensure honesty rule 3 holds.
    engine = po.PotatOptEngine(
        task="classification", 
        time_budget=time_budget,
        cost_sensitive_weighting=True, 
        n_jobs=1
    )
    
    t0 = time.time()
    engine.fit(X_train, y_train)
    fit_seconds = time.time() - t0
    
    engine.optimize_maintenance_threshold(X_val, y_val, COST_BREAKDOWN,
                                          COST_PLANNED, COST_INSPECTION)
    
    metrics = engine.evaluate(X_test, y_test)
    cost = engine.calculate_maintenance_cost(X_test, y_test, COST_BREAKDOWN,
                                             COST_PLANNED, COST_INSPECTION)
    
    total_seconds = time.time() - t0
    
    return {
        "variant": "PotatOpt step by step",
        "fit_seconds": round(fit_seconds, 1),
        "total_seconds": round(total_seconds, 1),
        "f1": metrics.get("f1", 0.0),
        "roc_auc": metrics.get("roc_auc", 0.0),
        "threshold": getattr(engine, "threshold_", 0.5),
        "cost_savings": cost["cost_savings"],
        "breakdown_avoidance_rate": cost["breakdown_avoidance_rate"],
    }

# ------------------------------------------------------------------------------
# Variant 2
# ------------------------------------------------------------------------------

def run_sklearn_by_hand(time_budget: int) -> dict[str, Any]:
    """
    The same job written the way a competent engineer writes it without the library.
    
    The time_budget is accepted and ignored here because there is no search to
    bound. This asymmetry is exactly what honesty rule 2 is about.
    """
    # The steps below must stay in sync with the BASELINE snippet in
    # benchmarks/token_cost.py, because token_cost.py counts that code and
    # this file runs it.
    X_train, X_val, X_test, y_train, y_val, y_test = load_split()
    import potatopt
    
    t0 = time.time()
    
    # split numeric and non-numeric columns
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_train.columns if c not in num_cols]
    
    # SimpleImputer(strategy="median") on numeric, most_frequent on the rest
    num_imp = SimpleImputer(strategy="median").fit(X_train[num_cols])
    cat_imp = (SimpleImputer(strategy="most_frequent").fit(X_train[cat_cols]) 
               if cat_cols else None)
    
    # OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    
    # StandardScaler; cast to float32
    scaler = StandardScaler()
    
    # fit on train only, transform val and test
    def prep(frame: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        out = frame.copy()
        out[num_cols] = num_imp.transform(out[num_cols])
        if cat_cols and cat_imp is not None:
            out[cat_cols] = cat_imp.transform(out[cat_cols])
            out[cat_cols] = (enc.fit_transform(out[cat_cols]) if fit else 
                             enc.transform(out[cat_cols]))
        out[num_cols] = (scaler.fit_transform(out[num_cols]) if fit else 
                         scaler.transform(out[num_cols]))
        return out.astype(np.float32)

    X_train_p = prep(X_train, fit=True)
    X_val_p = prep(X_val)
    X_test_p = prep(X_test)
    
    # compute_sample_weight("balanced", y_train)
    weights = compute_sample_weight("balanced", y_train)
    
    # RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=1)
    model.fit(X_train_p, y_train, sample_weight=weights)
    
    fit_seconds = time.time() - t0
    
    # sweep the threshold over np.arange(0.05, 0.95, 0.05) on the VALIDATION set,
    # picking the one that minimises maintenance_cost from the confusion matrix
    proba_val = model.predict_proba(X_val_p)[:, 1]
    best_t, best_cost = 0.5, float("inf")
    for t in np.arange(0.05, 0.95, 0.05):
        pred = (proba_val >= t).astype(int)
        _tn, fp, fn, tp = confusion_matrix(y_val, pred, labels=[0, 1]).ravel()
        cost = maintenance_cost(tp, fp, fn)
        if cost < best_cost:
            best_cost, best_t = cost, t
            
    # score the test set at that threshold
    proba_test = model.predict_proba(X_test_p)[:, 1]
    y_pred = (proba_test >= best_t).astype(int)
    _tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, proba_test)
    
    # potatopt.calculate_maintenance_savings computes cost so both variants
    # use the exact same money implementation.
    savings_dict = potatopt.calculate_maintenance_savings(
        tp, fp, fn, COST_BREAKDOWN, COST_PLANNED, COST_INSPECTION
    )
    
    total_seconds = time.time() - t0
    
    return {
        "variant": "scikit-learn by hand",
        "fit_seconds": round(fit_seconds, 1),
        "total_seconds": round(total_seconds, 1),
        "f1": test_f1,
        "roc_auc": test_roc_auc,
        "threshold": best_t,
        "cost_savings": savings_dict["cost_savings"],
        "breakdown_avoidance_rate": savings_dict["breakdown_avoidance_rate"],
    }

# ------------------------------------------------------------------------------
# Variant 3
# ------------------------------------------------------------------------------

def run_pycaret(time_budget: int) -> dict[str, Any]:
    """Runs the predictive-maintenance job using PyCaret if it is available."""
    try:
        import pycaret  # noqa: F401
        from pycaret.classification import compare_models, predict_model, setup
    except ImportError as exc:
        return {
            "variant": "PyCaret",
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "python": platform.python_version(),
            **PYCARET_FACTS
        }
        
    try:
        X_train, _X_val, X_test, y_train, _y_val, y_test = load_split()
        from ai4i_dataset import TARGET_COLUMN

        import potatopt
        
        t0 = time.time()
        
        train_frame = X_train.copy()
        train_frame[TARGET_COLUMN] = y_train
        
        setup(data=train_frame, target=TARGET_COLUMN, session_id=42, n_jobs=1,
              verbose=False, html=False)
        
        model = compare_models(n_select=1, budget_time=time_budget / 60.0)
        fit_seconds = time.time() - t0
        
        test_frame = X_test.copy()
        test_frame[TARGET_COLUMN] = y_test
        
        # PyCaret has no cost-based threshold step so this row is scored at
        # the default cut-off.
        predictions = predict_model(model, data=test_frame)
        y_pred = predictions["prediction_label"].astype(int)
        
        try:
            pred_score = predictions["prediction_score"]
            proba_test = np.where(y_pred == 1, pred_score, 1 - pred_score)
            test_roc_auc = roc_auc_score(y_test, proba_test)
        except Exception:  # noqa: BLE001
            test_roc_auc = 0.0
            
        _tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        test_f1 = f1_score(y_test, y_pred)
        
        savings_dict = potatopt.calculate_maintenance_savings(
            tp, fp, fn, COST_BREAKDOWN, COST_PLANNED, COST_INSPECTION
        )
        
        total_seconds = time.time() - t0
        
        return {
            "variant": "PyCaret",
            "fit_seconds": round(fit_seconds, 1),
            "total_seconds": round(total_seconds, 1),
            "f1": test_f1,
            "roc_auc": test_roc_auc,
            "threshold": 0.5,
            "cost_savings": savings_dict["cost_savings"],
            "breakdown_avoidance_rate": savings_dict["breakdown_avoidance_rate"],
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "variant": "PyCaret",
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "python": platform.python_version(),
            **PYCARET_FACTS
        }

# ------------------------------------------------------------------------------
# Measuring memory
# ------------------------------------------------------------------------------

def peak_memory_mb() -> float | None:
    """
    Returns the true peak resident memory of the current process.
    
    On Windows, uses psutil.Process().memory_info().peak_wset.
    Otherwise uses resource.getrusage(resource.RUSAGE_SELF).ru_maxrss.
    Note that ru_maxrss returns kilobytes on Linux, but bytes on macOS.
    Returns None when neither is available.
    """
    if platform.system() == "Windows":
        if psutil is None:
            return None
        try:
            return round(psutil.Process().memory_info().peak_wset / (1024 * 1024), 1)
        except Exception:  # noqa: BLE001
            return None
    else:
        try:
            import resource
            maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == "Darwin":
                return round(maxrss / (1024 * 1024), 1)  # bytes on macOS
            else:
                return round(maxrss / 1024, 1)           # kilobytes on Linux
        except Exception:  # noqa: BLE001
            return None

def run_child(variant: str, time_budget: int) -> int:
    """
    Runs the requested variant in-process.
    
    Adds peak memory to the result dict, prints it as a single line, and returns 0.
    Any exception is caught and reported so the parent always gets a row.
    """
    try:
        if variant == "potatopt":
            result = run_potatopt(time_budget)
        elif variant == "sklearn":
            result = run_sklearn_by_hand(time_budget)
        elif variant == "pycaret":
            result = run_pycaret(time_budget)
        else:
            raise ValueError(f"Unknown variant: {variant}")
            
        result["peak_memory_mb"] = peak_memory_mb()
        print(f"{RESULT_PREFIX}{json.dumps(result)}")
    except Exception as exc:  # noqa: BLE001
        err_res = {
            "variant": variant,
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}"
        }
        print(f"{RESULT_PREFIX}{json.dumps(err_res)}")
    return 0

def run_in_subprocess(variant: str, time_budget: int) -> dict[str, Any]:
    """
    Parent side: launches the child process to run the variant and samples its RSS.
    
    Each variant gets its own interpreter for a specific reason: import cost is
    part of what a library charges you, and measuring it inside a process that
    has already imported everything would quietly hide it.
    """
    cmd = [sys.executable, __file__, "--run", variant, "--time-budget", str(time_budget)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    
    sampled_peak = 0.0
    if psutil is not None:
        try:
            parent_proc = psutil.Process(proc.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            parent_proc = None
    else:
        parent_proc = None

    while proc.poll() is None:
        if parent_proc is not None:
            try:
                mem = parent_proc.memory_info().rss
                # Taking the maximum over the process AND its children
                for child in parent_proc.children(recursive=True):
                    mem += child.memory_info().rss
                current_mb = mem / (1024 * 1024)
                sampled_peak = max(sampled_peak, current_mb)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                # A short-lived child that exits mid-sample is normal, not an error.
                pass
        time.sleep(RSS_SAMPLE_SECONDS)
        
    stdout_text = proc.stdout.read() if proc.stdout else ""
    for line in stdout_text.splitlines():
        if line.startswith(RESULT_PREFIX):
            try:
                res = json.loads(line[len(RESULT_PREFIX):])
                if psutil is not None:
                    res["sampled_peak_mb"] = round(sampled_peak, 1)
                else:
                    res["sampled_peak_mb"] = None
                return res
            except json.JSONDecodeError:
                pass
                
    return {
        "variant": variant,
        "available": False,
        "reason": f"No RESULT line. Output: {stdout_text[-300:]}"
    }

# ------------------------------------------------------------------------------
# Output
# ------------------------------------------------------------------------------

def measure_import_peak(variant: str) -> float | None:
    """
    Peak memory of a process that only imports what `variant` needs.

    Runs in its own interpreter for the same reason the training runs do: a
    process that has already imported everything cannot measure what importing
    costs. Returns None when the imports fail, which is the normal answer for
    PyCaret on an interpreter it does not support.
    """
    root = Path(__file__).resolve().parents[1]
    code = (
        f"import sys; sys.path.insert(0, r'{root}'); "
        f"{IMPORT_SNIPPETS[variant]}; "
        "import sys as s; "
        # Same platform split as peak_memory_mb below: Windows reports a true
        # peak through psutil, POSIX through getrusage, where macOS counts bytes
        # and Linux kilobytes.
        "print(round((__import__('psutil').Process().memory_info().peak_wset / 1048576) "
        "if s.platform == 'win32' else "
        "(__import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss "
        "/ (1048576 if s.platform == 'darwin' else 1024)), 1))"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    if done.returncode != 0:
        return None
    try:
        return float(done.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def main() -> None:
    """Runs the variants and prints an aligned plain-text table."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-budget", type=int, default=DEFAULT_TIME_BUDGET)
    parser.add_argument("--json", type=str)
    parser.add_argument("--skip-pycaret", action="store_true")
    parser.add_argument("--run", type=str, choices=["potatopt", "sklearn", "pycaret"])
    args = parser.parse_args()

    if args.run:
        sys.exit(run_child(args.run, args.time_budget))
        
    variants = ["potatopt", "sklearn"]
    if not args.skip_pycaret:
        variants.append("pycaret")
        
    results = []
    for var in variants:
        res = run_in_subprocess(var, args.time_budget)
        res["import_peak_mb"] = measure_import_peak(var)
        results.append(res)

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "examples"))
    from ai4i_dataset import TARGET_COLUMN, load_ai4i
    
    full_df = load_ai4i()
    total_rows = len(full_df)
    failure_rate = full_df[TARGET_COLUMN].mean() * 100
    
    print(f"Dataset: AI4I 2020 ({total_rows} rows, {failure_rate:.1f}% failure rate)")
    print(f"Python: {platform.python_version()}")
    print(f"Time budget: {args.time_budget}s")
    print()
    
    header = (f"{'variant':<28}{'import MB':>11}{'peak MB':>10}"
              f"{'fit s':>9}{'f1':>7}{'roc_auc':>9}{'savings':>10}")
    print(header)
    print("-" * len(header))
    
    for row in results:
        if row.get("available", True) is False:
            if row["variant"] == "PyCaret" and "version_checked" in row:
                decl = row['declared_python']
                decl_str = f"{decl[0]}-{decl[-1]}" if len(decl) > 1 else decl[0]
                reason = (f"PyCaret {row['version_checked']} declares Python {decl_str}; "
                          f"this interpreter is {row['python']}")
            else:
                reason = row.get("reason", "Unavailable")
            print(f"{row['variant']:<28}{reason}")
        else:
            peak_mb = row.get("peak_memory_mb")
            if peak_mb is None:
                peak_mb = row.get("sampled_peak_mb", 0.0)
            
            fit_s = row.get("fit_seconds", 0.0)
            f1 = row.get("f1", 0.0)
            roc_auc = row.get("roc_auc", 0.0)
            savings = row.get("cost_savings", 0.0)
            
            import_mb = row.get("import_peak_mb")
            import_text = "-" if import_mb is None else f"{import_mb:.1f}"
            print(f"{row['variant']:<28}{import_text:>11}{peak_mb:>10.1f}"
                  f"{fit_s:>9.1f}{f1:>7.3f}{roc_auc:>9.3f}{savings:>10.0f}")
            
    print()
    print("Peak MB is the honest headline for memory usage, and import MB says how much")
    print("of it was spent before a single row was read. Read the two columns together:")
    print("when the import figures are close and the peaks are not, the difference is the")
    print("AutoML search training many candidate models, not a heavier dependency stack.")
    print("Fit seconds is bounded by --time-budget for the PotatOpt row "
          "and unbounded for the hand-written row.")
    print("f1 is a side effect here, not a verdict: both rows place their threshold by")
    print("maintenance cost, so compare the savings column and read f1 as context.")
    print("Both rows trained on identical splits at n_jobs=1.")

if __name__ == "__main__":
    main()
