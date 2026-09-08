"""Every figure Phase 5.0 publishes comes from this script.

Measured through the SHIPPED functions, never a design-time stand-in. Seeds are
explicit integers so a reader can re-run it and get the same table. Where a
change costs something, the cost is measured on the same series as the gain.

    python scripts/measure_phase5.py weibull     # hazard test: false alarm AND power
    python scripts/measure_phase5.py split       # what the split strategy is worth
    python scripts/measure_phase5.py all
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

import potatopt as po

HORIZON = 24


# --------------------------------------------------------------- the plant
def make_plant(seed: int, n_machines: int = 12, n_hours: int = 800) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A fleet that wears out, breaks, gets repaired, and wears out again.

    Returns the sensor log and the failure event log separately, because that is
    the shape a real plant has: readings from a historian, failures from a CMMS.
    """
    rng = np.random.default_rng(seed)
    readings, events = [], []
    for m in range(n_machines):
        mid = f"M-{m:02d}"
        offset_t, offset_v = rng.normal(0, 4.0), rng.normal(0, 0.8)
        wear_rate, wear = rng.uniform(0.010, 0.030), 0.0
        life = rng.uniform(120, 260)
        temp, vib, load = [], [], []
        for h in range(n_hours):
            wear += wear_rate * rng.uniform(0.6, 1.4)
            if wear > wear_rate * life:
                events.append({"machine_id": mid, "failed_at": h, "wo_type": "breakdown"})
                wear = 0.0
                life = rng.uniform(120, 260)
            temp.append(60 + offset_t + 40 * wear + rng.normal(0, 0.8))
            vib.append(2.0 + offset_v + 12 * wear ** 1.5 + rng.normal(0, 0.15))
            load.append(rng.uniform(40, 95))
        readings.append(pd.DataFrame({"machine_id": mid, "hour": np.arange(n_hours),
                                      "temperature": temp, "vibration": vib, "load_pct": load}))
    return pd.concat(readings, ignore_index=True), pd.DataFrame(events)


# --------------------------------------------------------------- weibull
def measure_weibull(trials: int = 300) -> None:
    """The question the project's own rule forces: what does this cost on data that is fine?

    A hazard test that calls a healthy constant-rate machine "wearing out" sends
    an engineer to strip a good bearing. That rate is measured here beside the
    detection rate, on the same sample sizes, never on its own.
    """
    print("\n=== calculate_weibull: does the constant-hazard test hold up? ===")
    print("false alarm = exponential data (beta is truly 1) called NOT constant hazard")
    print("detection   = Weibull data called NOT constant hazard\n")
    rows = []
    for n in (8, 15, 30, 60, 120, 300):
        cell = {"n_intervals": n}
        for label, beta_true in (("false_alarm_beta1.0", 1.0), ("beta1.5", 1.5),
                                 ("beta2.0", 2.0), ("beta3.0", 3.0)):
            flagged, fitted, betas = 0, 0, []
            for t in range(trials):
                rng = np.random.default_rng(100_000 * int(beta_true * 10) + 1000 * n + t)
                x = rng.weibull(beta_true, n) * 100.0
                r = po.calculate_weibull(x, random_state=t)
                if r.get("beta") is None:
                    continue
                fitted += 1
                betas.append(r["beta"])
                if r.get("constant_hazard_is_valid") is False:
                    flagged += 1
            cell[label] = round(100.0 * flagged / fitted, 1) if fitted else None
            if label == "false_alarm_beta1.0":
                cell["fitted"] = fitted
            cell[f"{label}_beta_med"] = round(float(np.median(betas)), 2) if betas else None
        rows.append(cell)
    t = pd.DataFrame(rows)
    show = ["n_intervals", "fitted", "false_alarm_beta1.0", "beta1.5", "beta2.0", "beta3.0"]
    print(t[show].to_string(index=False))
    print("\n(all figures are % of trials where constant_hazard_is_valid came back False)")
    print("median fitted beta by cell, to show the estimator is not biased:")
    print(t[["n_intervals"] + [c for c in t.columns if c.endswith("_beta_med")]].to_string(index=False))


# --------------------------------------------------------------- splits
def _arm(train: pd.DataFrame, test: pd.DataFrame, feats: list[str], name: str, seed: int) -> dict:
    n_val = max(50, int(len(train) * 0.25))
    tr, va = train.iloc[:-n_val], train.iloc[-n_val:]
    eng = po.PotatOptEngine(task="classification", time_budget=20, random_state=seed)
    eng.fit(tr[feats], tr["failure"])
    eng.optimize_maintenance_threshold(va[feats], va["failure"])
    y = test["failure"].to_numpy()
    p, proba = eng.predict(test[feats]), eng.predict_proba(test[feats])
    return {"arm": name, "seed": seed, "n_feat": len(feats), "n_test": len(y),
            "pr_auc": round(float(average_precision_score(y, proba[:, 1])), 4),
            "f1": round(float(f1_score(y, p, zero_division=0)), 4),
            "precision": round(float(precision_score(y, p, zero_division=0)), 4),
            "recall": round(float(recall_score(y, p, zero_division=0)), 4)}


def measure_split(seeds: tuple[int, ...] = (42, 7, 101, 2024, 555)) -> None:
    print("\n=== what the split strategy and the window features are worth ===")
    print(f"12 machines x 800 hours, label = failure within {HORIZON} readings, "
          f"{len(seeds)} simulated plants\n")
    RAW = ["temperature", "vibration", "load_pct"]
    rows, drops = [], []
    for seed in seeds:
        readings, events = make_plant(seed)
        lab, lrep = po.build_failure_labels(readings, events, "machine_id", "hour", horizon=HORIZON)
        fe, frep = po.add_window_features(lab, "machine_id", "hour", value_cols=RAW,
                                          label_col="failure", baseline_n=100, windows=(12, 24))
        drops.append({"seed": seed, "positive_rate": lrep["positive_rate"],
                      "rows_in": frep["rows_in"], "rows_out": frep["rows_out"],
                      "rows_dropped": frep["rows_dropped"], "n_features": frep["n_features_added"],
                      "mb_added": frep["memory_mb_added"]})
        WIN = RAW + frep["features_added"]
        held = {f"M-{m:02d}" for m in range(9, 12)}
        cut = int(800 * 0.8)
        # A: what the library did before this phase - a random stratified shuffle.
        Xtr, Xte, ytr, yte = po.split_data(lab[RAW + ["failure"]], "failure",
                                           test_size=0.2, random_state=seed)
        rows.append(_arm(Xtr.assign(failure=ytr), Xte.assign(failure=yte), RAW, "A random, raw", seed))
        rows.append(_arm(lab[~lab.machine_id.isin(held)], lab[lab.machine_id.isin(held)],
                         RAW, "B unseen machines, raw", seed))
        rows.append(_arm(lab[lab.hour < cut], lab[lab.hour >= cut], RAW, "C later period, raw", seed))
        rows.append(_arm(fe[~fe.machine_id.isin(held)], fe[fe.machine_id.isin(held)],
                         WIN, "D unseen machines, windows", seed))
        rows.append(_arm(fe[fe.hour < cut], fe[fe.hour >= cut], WIN, "E later period, windows", seed))
    t = pd.DataFrame(rows)
    print("what the window features cost in rows and memory:")
    print(pd.DataFrame(drops).to_string(index=False))
    print("\nper-arm, across seeds (mean [min-max]):")
    agg = t.groupby("arm")[["pr_auc", "f1", "precision", "recall"]].agg(["mean", "min", "max"]).round(3)
    print(agg.to_string())
    print("\nraw per-seed table:")
    print(t.to_string(index=False))


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if what in ("weibull", "all"):
        measure_weibull()
    if what in ("split", "all"):
        measure_split()
    print(f"\ntotal {time.time() - t0:.0f}s")
