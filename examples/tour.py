"""
A complete, runnable tour of the PotatOpt industrial AI and quality engineering library.

This tour exercises every public function and engine capability in PotatOpt across
an end-to-end simulated smart factory workflow:
1. Data Quality & Profiling (detect silent nulls, outliers, collinearity, DQS)
2. Measurement System Analysis (ANOVA Gauge R&R, NDC)
3. Statistical Process Control (SPC limits, EWMA, CUSUM, Nelson/Western Electric rules)
4. Process Capability (Cp, Cpk, Pp, Ppk with stability gating)
5. Drift Detection (fleet vs per-asset drift, numeric and categorical PSI)
6. Industrial Reliability (MTBF, MTTR, Availability, OEE, Pareto, CoQ)
7. Machine Learning Engine (3-way split, threshold optimization, XAI/SHAP, ISO 9001)
8. Interop & Serialization (JSON-ready outputs, audit logging, runtime versions)

Note on time budgets:
The time budgets used here (5 seconds) are deliberately tiny so this tour runs in
under a minute. This is a tour, not a benchmark; model quality achieved with these
tiny budgets does not reflect what the library achieves with production budgets.

Usage:
    python examples/tour.py
"""
from __future__ import annotations

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Support running directly from source checkout or examples directory
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parents[1]))

try:
    import potatopt as po
except ImportError:
    print("potatopt not found. Please install via pip install -e .[all]")
    sys.exit(1)


def section(number: int, title: str) -> None:
    """
    Prints a formatted section header to make the tour readable as a walk.
    """
    print()
    header = f"STEP {number} - {title}"
    print(header)
    print("-" * len(header))


def main() -> dict[str, Any]:
    """
    Executes the eight sections of the PotatOpt library tour and returns results.
    """
    results: dict[str, Any] = {}
    rng = np.random.default_rng(42)

    # =========================================================================
    # SECTION 1: Can we trust the data?
    # =========================================================================
    section(1, "Can we trust the data?")

    n_rows = 120
    temp_raw = rng.normal(65.0, 4.0, n_rows)
    vib_raw = rng.normal(2.5, 0.6, n_rows)
    pressure_raw = rng.normal(12.0, 1.5, n_rows)
    heat_idx_raw = temp_raw * 1.8 + 32.0 + rng.normal(0.0, 0.05, n_rows)

    status_col = rng.choice(["RUNNING", "IDLE", "STANDBY"], n_rows).astype(object)
    # Plant intentional silent null placeholder tokens
    status_col[8] = "N/A"
    status_col[24] = "null"
    status_col[50] = "-"

    # Plant intentional numeric sentinel (-999) and extreme physical outlier
    pressure_raw[15] = -999.0
    vib_raw[77] = 48.0

    failure_target = ((temp_raw > 72.0) | (vib_raw > 4.5)).astype(int)

    df_plant_raw = pd.DataFrame({
        "machine_id": rng.choice(["M01", "M02", "M03"], n_rows),
        "temperature_c": temp_raw,
        "vibration_mms": vib_raw,
        "pressure_bar": pressure_raw,
        "heat_index": heat_idx_raw,
        "status": status_col,
        "failure": failure_target,
    })

    silent_nulls = po.detect_silent_nulls(df_plant_raw)
    outliers = po.detect_outliers(df_plant_raw)
    correlations = po.calculate_correlations(df_plant_raw, min_abs=0.8)
    audit = po.audit_data_quality(df_plant_raw, target_col="failure")
    profile = po.inspect_data(df_plant_raw, target_col="failure")

    results["detect_silent_nulls"] = silent_nulls
    results["detect_outliers"] = outliers
    results["calculate_correlations"] = correlations
    results["audit_data_quality"] = audit
    results["inspect_data"] = profile

    print(f"Silent nulls found in columns: {list(silent_nulls.keys())}")
    for col, info in silent_nulls.items():
        print(f"  {col}: {info['count']} tokens ({info['tokens']})")
    print(
        "Silent nulls ('N/A', -999) appear non-null to standard parsers but carry "
        "no real data; catching them stops corrupting downstream imputation."
    )

    print(f"Outliers detected in columns: {list(outliers.keys())}")
    for col, info in outliers.items():
        print(f"  {col}: {info['count']} flagged (max: {info['flagged_max']:.2f})")
    print(
        "Modified z-scores use median absolute deviation so extreme outliers do not "
        "hide themselves by inflating the sample variance."
    )

    print(f"High collinearity pairs (|r| >= 0.8): {len(correlations['strong_pairs'])}")
    for pair in correlations["strong_pairs"]:
        print(f"  {pair['a']} <-> {pair['b']}: r = {pair['correlation']:.3f}")

    print(f"Data Quality Score: {audit['dqs']}/100 ({audit['grade']})")
    print(f"Recommended Task  : {profile['recommended_task']} ({profile['recommended_metric']})")

    # =========================================================================
    # SECTION 2: Can we even measure it?
    # =========================================================================
    section(2, "Can we even measure it?")

    # Balanced 2-way crossed design: 5 parts, 3 operators, 3 replicates = 45 rows
    gauge_parts = [f"Part_{i+1}" for i in range(5)]
    gauge_operators = ["Operator_A", "Operator_B", "Operator_C"]
    replicates = 3

    gauge_records = []
    for p_idx, part in enumerate(gauge_parts):
        true_dimension = 25.0 + p_idx * 1.5
        for op_idx, operator in enumerate(gauge_operators):
            op_effect = (op_idx - 1) * 0.15
            for _ in range(replicates):
                measurement = true_dimension + op_effect + rng.normal(0.0, 0.25)
                gauge_records.append({
                    "part": part,
                    "operator": operator,
                    "dimension_mm": measurement,
                })

    df_gauge = pd.DataFrame(gauge_records)
    grr = po.calculate_gauge_rr(
        df_gauge,
        part_col="part",
        operator_col="operator",
        measurement_col="dimension_mm",
    )
    results["calculate_gauge_rr"] = grr

    grr_pct = grr["percent_study_variation"]["gauge_rr"]
    ndc = grr["ndc"]
    print(f"Gauge R&R Study Variation: {grr_pct:.2f}% ({grr['verdict']})")
    print(f"Number of Distinct Categories (NDC): {ndc} ({grr['ndc_verdict']})")
    print(
        "An unacceptable gauge (>30% variation or NDC < 5) inflates total observed "
        "variance, deflating every Cpk computed downstream and widening SPC control limits."
    )

    # =========================================================================
    # SECTION 3: Is the process stable?
    # =========================================================================
    section(3, "Is the process stable?")

    # 120 points: 40 points in-control, followed by gradual progressive tool wear
    n_spc_points = 120
    baseline_window = 40
    spc_signal = rng.normal(50.0, 1.0, n_spc_points)
    wear_ramp = np.zeros(n_spc_points)
    for i in range(baseline_window, n_spc_points):
        wear_ramp[i] = (i - baseline_window) * 0.08
    spc_series = spc_signal + wear_ramp
    df_spc = pd.DataFrame({"spindle_temp": spc_series})

    spc_limits = po.calculate_spc_limits(df_spc, sensor_column="spindle_temp")
    ewma_chart = po.calculate_ewma_chart(spc_series, baseline_n=baseline_window)
    cusum_chart = po.calculate_cusum_chart(spc_series, baseline_n=baseline_window)
    control_rules = po.calculate_control_rules(spc_series, baseline_n=baseline_window)

    results["calculate_spc_limits"] = spc_limits
    results["calculate_ewma_chart"] = ewma_chart
    results["calculate_cusum_chart"] = cusum_chart
    results["calculate_control_rules"] = control_rules

    # Rule 1 on its own IS the Shewhart single-point test, so running it alone
    # gives the like-for-like comparison the sentence below rests on. Quoting the
    # full rule set instead would compare a memory chart against nine rules at
    # once, which is a different claim.
    shewhart_only = po.calculate_control_rules(
        spc_series, baseline_n=baseline_window, rules=(1,)
    )
    results["calculate_control_rules_shewhart_only"] = shewhart_only
    shewhart_first = shewhart_only["first_violation_index"]

    print(f"Shewhart 3-Sigma Limits: UCL = {spc_limits['ucl']:.2f}, LCL = {spc_limits['lcl']:.2f}")
    print(f"Shewhart single point (rule 1) first signal: sample #{shewhart_first}")
    print(f"EWMA chart first signal                    : sample #{ewma_chart['first_violation']}")
    print(f"CUSUM chart first signal                   : sample #{cusum_chart['first_violation']}")
    print(
        f"All rules, first signal                    : sample #{control_rules['first_violation_index']} "
        f"(Nelson rule {control_rules['first_violation_rule']}: "
        f"{po.CONTROL_RULE_DESCRIPTIONS[control_rules['first_violation_rule']]})"
    )
    print(
        "The wear starts at sample 40. A single point outside 3 sigma has to wait for the "
        "drift to become large; the memory charts and the run rules accumulate evidence, so "
        "they speak while the drift is still small - which is the whole reason to run them."
    )

    # =========================================================================
    # SECTION 4: Is it capable?
    # =========================================================================
    section(4, "Is it capable?")

    capability_degrading = po.calculate_capability(
        spc_series,
        usl=60.0,
        lsl=40.0,
        baseline_n=baseline_window,
    )
    # No baseline_n here, unlike the degrading series above, and the difference is
    # not cosmetic. A baseline window exists to fence off a Phase I period before
    # the process was allowed to wander; a process that never wanders has no such
    # period, and the whole series is its own baseline. Passing baseline_n=40 on
    # 300 in-control points estimates sigma from 40 readings and then judges the
    # other 260 against it: measured over 300 trials that calls a healthy process
    # unstable 17.3% of the time, against 2.0% when the whole series is used.
    clean_process = rng.normal(50.0, 1.0, 300)
    capability_clean = po.calculate_capability(clean_process, usl=54.0, lsl=46.0)

    results["calculate_capability_degrading"] = capability_degrading
    results["calculate_capability_clean"] = capability_clean

    cpk_deg = capability_degrading["cpk"]
    meaningful_deg = capability_degrading["capability_is_meaningful"]
    print(f"Degrading Process Cpk: {cpk_deg:.2f} | Meaningful: {meaningful_deg}")
    print(f"Interpretation: {capability_degrading['interpretation']}")

    cpk_clean = capability_clean["cpk"]
    meaningful_clean = capability_clean["capability_is_meaningful"]
    print(
        f"In-control process Cpk: {cpk_clean:.2f} | meaningful: {meaningful_clean} "
        f"({capability_clean['verdict']})"
    )
    # Print the reason whichever way it lands. A demo that only explains itself
    # when the answer is the expected one is not explaining itself.
    if not meaningful_clean:
        print(f"Interpretation: {capability_clean['interpretation']}")
    print(
        f"The degrading process scores the HIGHER Cpk of the two ({cpk_deg:.2f} against "
        f"{cpk_clean:.2f}) and is the one that means nothing: its limits are wide, and the "
        "moving range only ever sees the small step between neighbouring points, never the "
        "distance the process has travelled. Stability first, capability second."
    )

    # =========================================================================
    # SECTION 5: Has anything drifted since we trained?
    # =========================================================================
    section(5, "Has anything drifted since we trained?")

    n_train_fleet = 100
    n_batch_fleet = 50

    t_m1 = pd.DataFrame({"machine": "M1", "temp": rng.normal(60.0, 2.0, n_train_fleet), "shift": rng.choice(["Day", "Night"], n_train_fleet)})
    t_m2 = pd.DataFrame({"machine": "M2", "temp": rng.normal(70.0, 2.0, n_train_fleet), "shift": rng.choice(["Day", "Night"], n_train_fleet)})
    t_m3 = pd.DataFrame({"machine": "M3", "temp": rng.normal(80.0, 2.0, n_train_fleet), "shift": rng.choice(["Day", "Night"], n_train_fleet)})
    train_fleet = pd.concat([t_m1, t_m2, t_m3], ignore_index=True)

    # M1 drifts +5.0 K hotter and shifts to night runs; M2 and M3 remain stable
    b_m1 = pd.DataFrame({"machine": "M1", "temp": rng.normal(65.0, 2.0, n_batch_fleet), "shift": rng.choice(["Day", "Night"], n_batch_fleet, p=[0.1, 0.9])})
    b_m2 = pd.DataFrame({"machine": "M2", "temp": rng.normal(70.0, 2.0, n_batch_fleet), "shift": rng.choice(["Day", "Night"], n_batch_fleet)})
    b_m3 = pd.DataFrame({"machine": "M3", "temp": rng.normal(80.0, 2.0, n_batch_fleet), "shift": rng.choice(["Day", "Night"], n_batch_fleet)})
    batch_fleet = pd.concat([b_m1, b_m2, b_m3], ignore_index=True)

    pooled_drift = po.check_data_drift(train_fleet[["temp"]], batch_fleet[["temp"]])
    asset_drift = po.check_asset_drift(train_fleet, batch_fleet, asset_col="machine")
    psi_numeric = po.calculate_psi(train_fleet["temp"], batch_fleet["temp"])
    psi_categorical = po.calculate_categorical_psi(train_fleet["shift"], batch_fleet["shift"])

    results["check_data_drift"] = pooled_drift
    results["check_asset_drift"] = asset_drift
    results["calculate_psi"] = psi_numeric
    results["calculate_categorical_psi"] = psi_categorical

    print(f"Pooled Fleet Drift Detected : {pooled_drift['drift_detected']} (Max PSI: {pooled_drift['max_psi']:.3f})")
    print(f"Per-Asset Drift Detected    : {asset_drift['drift_detected']} (Drifted assets: {asset_drift['assets_drifted']})")
    for asset_name, asset_info in asset_drift["per_asset"].items():
        max_p = asset_info.get("max_psi") or 0.0
        print(f"  {asset_name}: status = {asset_info['status']}, max_psi = {max_p:.3f}, drifted = {asset_info['drift_detected']}")
    print(f"Numeric PSI (temp)         : {psi_numeric:.3f}")
    print(f"Categorical PSI (shift)    : {psi_categorical:.3f}")
    print(
        "Pooled drift dilutes single-machine degradation across fleet variance; per-asset "
        "drift isolates the single machine requiring a maintenance work order."
    )

    # =========================================================================
    # SECTION 6: What is it costing us?
    # =========================================================================
    section(6, "What is it costing us?")

    df_wo = pd.DataFrame({
        "wo_id": [f"WO_{i+1:03d}" for i in range(10)],
        "wo_type": ["breakdown", "breakdown", "pm", "breakdown", "inspection",
                    "breakdown", "pm", "breakdown", "pm", "breakdown"],
        "reported_at": [
            "2026-01-05 08:00", "2026-01-15 14:00", "2026-01-20 09:00", "2026-01-28 10:30",
            "2026-02-02 11:00", "2026-02-10 16:00", "2026-02-18 08:30", "2026-02-25 13:15",
            "2026-03-02 10:00", "2026-03-10 07:45"
        ],
        "started_at": [
            "2026-01-05 09:30", "2026-01-15 15:30", "2026-01-20 09:00", "2026-01-28 12:00",
            "2026-02-02 11:00", "2026-02-10 17:30", "2026-02-18 08:30", "2026-02-25 14:45",
            "2026-03-02 10:00", "2026-03-10 09:15"
        ],
        "finished_at": [
            "2026-01-05 14:30", "2026-01-15 19:30", "2026-01-20 11:00", "2026-01-28 17:00",
            "2026-02-02 12:00", "2026-02-10 22:00", "2026-02-18 10:30", "2026-02-25 20:15",
            "2026-03-02 12:00", "2026-03-10 15:15"
        ],
        "failure_mode": [
            "Bearing Fatigue", "Motor Overheat", "Scheduled", "Bearing Fatigue", "Routine",
            "Seal Failure", "Scheduled", "Bearing Fatigue", "Scheduled", "Seal Failure"
        ],
        "downtime_cost": [
            25000.0, 8000.0, 500.0, 28000.0, 200.0,
            6500.0, 500.0, 26000.0, 500.0, 7200.0
        ]
    })

    mtbf_res = po.calculate_mtbf(df_wo, operating_hours=2000.0, wo_type_col="wo_type", breakdown_types=("breakdown",))
    mttr_res = po.calculate_mttr(
        df_wo,
        reported_col="reported_at",
        started_col="started_at",
        finished_col="finished_at",
        wo_type_col="wo_type",
        breakdown_types=("breakdown",),
    )
    avail_res = po.calculate_availability(
        mtbf_hours=mtbf_res["mtbf_hours"],
        mttr_hours=mttr_res["mttr_hours"],
        mdt_hours=mttr_res["mdt_hours"],
    )
    oee_res = po.calculate_oee(
        planned_time_min=480.0,
        run_time_min=420.0,
        ideal_cycle_time_min=0.8,
        total_count=500,
        good_count=485,
    )
    pareto_cost = po.calculate_pareto(df_wo, category_col="failure_mode", value_col="downtime_cost")
    maint_savings = po.calculate_maintenance_savings(
        true_positives=20,
        false_positives=4,
        false_negatives=2,
        cost_breakdown=50000.0,
        cost_planned=8000.0,
        cost_inspection=1500.0,
    )
    wilson_ci = po.wilson_confidence_interval(successes=20, trials=22, confidence=0.95)

    results["calculate_mtbf"] = mtbf_res
    results["calculate_mttr"] = mttr_res
    results["calculate_availability"] = avail_res
    results["calculate_oee"] = oee_res
    results["calculate_pareto"] = pareto_cost
    results["calculate_maintenance_savings"] = maint_savings
    results["wilson_confidence_interval"] = wilson_ci

    print(f"MTBF: {mtbf_res['mtbf_hours']:.1f} hrs | MTTR: {mttr_res['mttr_hours']:.1f} hrs | MDT: {mttr_res['mdt_hours']:.1f} hrs")
    print(f"Inherent Availability   : {avail_res['inherent_availability_pct']:.2f}%")
    print(f"Operational Availability: {avail_res['operational_availability_pct']:.2f}%")
    print(f"Availability Lost to Wait: {avail_res['availability_lost_to_waiting'] * 100.0:.2f}%")
    print(f"OEE: {oee_res['oee_pct']:.1f}% (Avail: {oee_res['availability_pct']:.1f}%, Perf: {oee_res['performance_pct']:.1f}%, Qual: {oee_res['quality_pct']:.1f}%)")
    print(f"Pareto Vital Few (Cost) : {pareto_cost['vital_few']} ({pareto_cost['vital_few_share']:.1f}% of cost)")
    print(f"Maintenance Cost Savings: ${maint_savings['cost_savings']:,.2f} ({maint_savings['savings_percentage']:.1f}%)")
    print(f"Recall 95% Wilson CI    : {wilson_ci['point']:.3f} [{wilson_ci['lower']:.3f}, {wilson_ci['upper']:.3f}]")

    # =========================================================================
    # SECTION 7: Can a model help?
    # =========================================================================
    section(7, "Can a model help?")

    n_ml_samples = 220
    ml_temp = rng.normal(300.0, 3.0, n_ml_samples)
    ml_speed = rng.normal(1500.0, 80.0, n_ml_samples)
    ml_torque = rng.normal(40.0, 7.0, n_ml_samples)
    ml_wear = rng.uniform(10.0, 240.0, n_ml_samples)
    ml_vibration = rng.normal(3.0, 0.6, n_ml_samples)

    fail_probability = 1.0 / (1.0 + np.exp(-((ml_wear - 170.0) * 0.04 + (ml_torque - 45.0) * 0.2)))
    ml_failure = (rng.uniform(0.0, 1.0, n_ml_samples) < fail_probability).astype(int)

    df_ml = pd.DataFrame({
        "temperature_k": ml_temp,
        "speed_rpm": ml_speed,
        "torque_nm": ml_torque,
        "tool_wear_min": ml_wear,
        "vibration_mms": ml_vibration,
        "failure": ml_failure,
    })

    # 1. split_data (2-way split)
    X_tr2, X_te2, _y_tr2, _y_te2 = po.split_data(df_ml, target_col="failure", test_size=0.2, random_state=42)
    results["split_data"] = {"train_len": len(X_tr2), "test_len": len(X_te2)}

    # 2. split_data_three_way (3-way split: Train / Validation / Test)
    X_train, X_val, X_test, y_train, y_val, y_test = po.split_data_three_way(
        df_ml,
        target_col="failure",
        val_size=0.2,
        test_size=0.2,
        random_state=42,
    )
    results["split_data_three_way"] = {
        "train_len": len(X_train),
        "val_len": len(X_val),
        "test_len": len(X_test),
    }

    # 3. PotatOptEngine and PotatOpt alias instantiation
    assert po.PotatOpt is po.PotatOptEngine
    _engine_alias = po.PotatOpt(task="classification", time_budget=5, random_state=42, verbose=0)
    engine = po.PotatOptEngine(
        task="classification",
        time_budget=5,
        cost_sensitive_weighting=True,
        random_state=42,
        verbose=0,
    )

    # 4. Engine fit
    engine.fit(X_train, y_train)

    # 5. optimize_maintenance_threshold
    threshold = engine.optimize_maintenance_threshold(
        X_val,
        y_val,
        cost_breakdown=50000.0,
        cost_planned=8000.0,
        cost_inspection=1500.0,
    )

    # 6. check_calibration
    val_probs = engine.predict_proba(X_val)
    prob_col = val_probs[:, engine.pos_label_idx] if val_probs is not None else np.zeros(len(X_val))
    calib = po.check_calibration(y_val, prob_col)

    # 7. evaluate
    eval_metrics = engine.evaluate(X_test, y_test)

    # 8. calculate_maintenance_cost
    maint_cost = engine.calculate_maintenance_cost(
        X_test,
        y_test,
        cost_breakdown=50000.0,
        cost_planned=8000.0,
        cost_inspection=1500.0,
    )

    # 9. get_feature_importance
    feat_imp = engine.get_feature_importance()

    # 10. explain_predictions
    shap_exp = engine.explain_predictions(X_test, top_k=5)

    results["explain_predictions"] = {
        "available": shap_exp.get("available"),
        "n_rows_explained": shap_exp.get("n_rows_explained"),
        "reason": shap_exp.get("reason"),
    }

    # 11. get_training_report
    train_rep = engine.get_training_report()

    # 12. detect_drift
    drift_rep = engine.detect_drift(X_test)

    # 13. get_inference_health
    inf_health = engine.get_inference_health()

    # 14. save & load with temp directory cleanup
    temp_model_dir = tempfile.mkdtemp(prefix="potatopt_tour_model_")
    try:
        model_file = os.path.join(temp_model_dir, "model.pkl")
        saved_path = engine.save(model_file)
        loaded_engine = po.PotatOptEngine.load(saved_path, enforce_security=True)
        sample_preds = loaded_engine.predict(X_test.head(3))
    finally:
        shutil.rmtree(temp_model_dir, ignore_errors=True)

    # 15. auto_analyze
    auto_res = po.auto_analyze(df_ml, target="failure", time_budget=5, random_state=42)

    # 16. run_seed_sweep
    sweep_res = po.run_seed_sweep(df_ml, target="failure", seeds=[11, 22, 33], time_budget=5)

    results["PotatOptEngine"] = {
        "best_estimator": train_rep.get("best_estimator"),
        "threshold": threshold,
        "evaluate": eval_metrics,
        "calibration": calib,
        "maintenance_cost": maint_cost,
        "training_report": train_rep,
        "detect_drift": drift_rep,
        "inference_health": inf_health,
        "sample_preds": sample_preds,
    }
    results["check_calibration"] = calib
    results["auto_analyze"] = auto_res
    results["run_seed_sweep"] = sweep_res

    print(f"Three-Way Split: Train = {len(X_train)}, Val = {len(X_val)}, Test = {len(X_test)}")
    print(f"Best Estimator : {train_rep.get('best_estimator')}")
    print(f"Tuned Threshold: {threshold} (tuned on validation maintenance cost)")
    print(f"Test Accuracy  : {eval_metrics.get('accuracy', 0.0):.3f}")
    print(f"Test Recall    : {eval_metrics.get('recall', 0.0):.3f}")
    print(f"Test ROC AUC   : {eval_metrics.get('roc_auc')}")
    print(f"Calibration ECE: {calib.get('expected_calibration_error', 0.0):.3f}")
    print(f"Cost Savings   : ${maint_cost.get('cost_savings', 0.0):,.2f}")
    if feat_imp is not None and not feat_imp.empty:
        top_f = feat_imp.iloc[0]["feature"]
        print(f"Top Feature    : {top_f}  (by the model's own split counts)")
    # The two rankings answer different questions and can disagree: the model's
    # importance counts how often a feature was split on, SHAP measures how much
    # each feature moved the prediction. Print both rather than picking one.
    if shap_exp.get("available"):
        attributions = shap_exp.get("feature_attributions") or []
        ranked = ", ".join(
            f"{item['feature']} ({item['mean_abs_shap']:.3f})" for item in attributions[:3]
        )
        print(f"Top by SHAP    : {ranked or 'none returned'}")
        if shap_exp.get("additivity_check_relaxed"):
            print("                 (additivity check relaxed - ranking is approximate)")
    else:
        # Never a silent empty list: when SHAP declines, it says why.
        print(f"Top by SHAP    : unavailable - {shap_exp.get('reason')}")
    print(f"AutoML Sweep (3 seeds): {sweep_res.get('stability_note')}")

    # =========================================================================
    # SECTION 8: Getting the answers out
    # =========================================================================
    section(8, "Getting the answers out")

    lib_versions = po.get_library_versions()
    results["get_library_versions"] = lib_versions

    temp_audit_dir = tempfile.mkdtemp(prefix="potatopt_tour_audit_")
    audit_log_target = os.path.join(temp_audit_dir, "tour_audit.log")
    try:
        audit_path = po.enable_audit_log(audit_log_target)
        print(f"Audit log active at: {audit_path}")
        po.logger.info("Tour audit event recorded for ISO 9001 traceability.")
        results["enable_audit_log"] = audit_path
    finally:
        for handler in list(po.logger.handlers):
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == os.path.abspath(audit_log_target):
                handler.close()
                po.logger.removeHandler(handler)
        shutil.rmtree(temp_audit_dir, ignore_errors=True)

    jsonable_data = po.to_jsonable({
        "sample_array": np.array([10.0, 20.0, 30.0]),
        "nan_value": float("nan"),
        "metrics": {"auc": 0.94, "cost": 12500.0},
    })
    results["to_jsonable"] = jsonable_data

    print("Every result produced by the engine is JSON-ready via `to_jsonable` for MCP servers and web apps.")
    print(f"Runtime versions: numpy {lib_versions.get('numpy')}, pandas {lib_versions.get('pandas')}, scikit-learn {lib_versions.get('scikit-learn')}")

    print()
    print("Next steps - what is outside this single-file tour:")
    print("1. `chart_engine.py`: Publication-ready visual figures for SPC charts, EWMA, CUSUM, Pareto, and feature attributions.")
    print("3. `potatopt.mcp_server`: FastMCP server exposing this entire pipeline as tools for AI agents and LLM pair programmers.")

    return results


if __name__ == "__main__":
    main()
