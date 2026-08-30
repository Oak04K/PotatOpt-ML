"""
This test suite locks the production guardrails, metric correctness,
drift detection and ISO 9001 traceability of PotatOpt. Numeric expectations
and statistical behaviors were verified against the reference implementation.
"""

import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    import psutil
except ImportError:  # pragma: no cover - benchmark degrades to a skip
    psutil = None

import potatopt as po

# ---- Package surface ----


def test_version_is_exposed():
    assert isinstance(po.__version__, str) and po.__version__.count(".") == 2


def test_no_print_calls_in_source():
    # All user-facing output must go through `logger`, never a bare print().
    # Matched with a boundary so legitimate names ending in "print" - such as
    # _dataset_fingerprint() - are not mistaken for a print call.
    repo_root = Path(__file__).resolve().parent.parent
    scanned = sorted((repo_root / "potatopt").glob("*.py"))
    # A guard that reads no files reports a clean result forever. The package was
    # a single module until it was split; if it is ever moved or renamed again,
    # fail here rather than passing vacuously.
    assert len(scanned) >= 10, f"expected the potatopt package modules, scanned {len(scanned)} file(s)"

    offenders = {}
    for path in scanned:
        found = re.findall(r"(?<![A-Za-z0-9_.])print\s*\(", path.read_text(encoding="utf-8"))
        if found:
            offenders[path.name] = len(found)
    assert not offenders, f"Found bare print() call(s) in potatopt: {offenders}"


def test_the_print_guard_still_catches_a_real_print():
    # Guards the guard: prove the regex above is not vacuously true.
    sample = "logger.info('ok')\nprint('leaked')\nvalue = obj._dataset_fingerprint(x, y)\n"
    offenders = re.findall(r"(?<![A-Za-z0-9_.])print\s*\(", sample)
    assert len(offenders) == 1


# ---- Emoji policy ----
#
# The potato is the project's mark and is allowed. Nothing else is: decorative
# emoji in a repository that ships to GitHub and PyPI render inconsistently
# across terminals, and status glyphs in particular (a warning sign, a green
# tick) carry meaning that the surrounding sentence should be carrying instead.
# Ranges below cover pictographs, dingbats and the variation selector that turns
# a plain glyph into an emoji; typographic characters stay legal on purpose -
# arrows (U+2190-21FF), box drawing (U+2500-257F) and the geometric shapes
# (U+25A0-25FF) that the Thai README's architecture diagrams are built from.

POTATO = "\U0001f954"
EMOJI_PATTERN = re.compile(
    "["
    "\U0001f300-\U0001faff"  # pictographs, emoticons, supplemental symbols
    "\U00002600-\U000027bf"  # miscellaneous symbols and dingbats
    "\U00002b00-\U00002bff"  # arrows and stars used as emoji
    "\U0000fe0f"             # variation selector-16
    "]"
)
EMOJI_SCANNED_GLOBS = ("*.md", "*.py", "*.txt", "*.toml", "*.cfg", "*.yml", "*.yaml")
EMOJI_SCANNED_DIRS = ("", "potatopt", "tests", "benchmarks", "scripts", "examples", ".github/workflows")


def _shipped_text_files():
    repo_root = Path(__file__).resolve().parent.parent
    for folder in EMOJI_SCANNED_DIRS:
        directory = repo_root / folder if folder else repo_root
        if not directory.is_dir():
            continue
        for pattern in EMOJI_SCANNED_GLOBS:
            yield from sorted(directory.glob(pattern))


def test_no_emoji_except_the_potato():
    offenders = {}
    for path in _shipped_text_files():
        found = {ch for ch in EMOJI_PATTERN.findall(path.read_text(encoding="utf-8")) if ch != POTATO}
        if found:
            offenders[path.name] = sorted(found)
    assert not offenders, f"Emoji other than the potato found: {offenders}"


def test_the_emoji_guard_still_catches_a_real_emoji():
    # Guards the guard, twice over: a real emoji must be caught, and the
    # typography the diagrams rely on must not be. The sample is spelled with
    # escapes rather than a literal glyph so that this file does not trip the
    # scan above - the check runs over the whole repository, tests included.
    warning_sign = "\u26a0\ufe0f"
    assert EMOJI_PATTERN.findall(f"build passed {warning_sign} ok") == ["\u26a0", "\ufe0f"]
    assert EMOJI_PATTERN.findall("─└→►▼ ≥ √") == []
    assert EMOJI_PATTERN.findall(POTATO) == [POTATO]


# ---- Wilson confidence interval ----


def test_wilson_matches_reference_values():
    ci = po.wilson_confidence_interval(3, 30)
    assert ci["point"] == pytest.approx(0.1)
    assert ci["lower"] == pytest.approx(0.03460, abs=1e-4)
    assert ci["upper"] == pytest.approx(0.25621, abs=1e-4)


def test_wilson_stays_inside_unit_interval():
    ci = po.wilson_confidence_interval(50, 50)
    assert ci["upper"] == 1.0 and 0.9 < ci["lower"] < 1.0
    # The normal approximation would return a degenerate [1, 1] here.


@pytest.mark.parametrize(
    "args",
    [
        (0, 0),
        (10, 5),
        (-1, 10),
        ("x", None),
        (5, 10, 0.0),
        (5, 10, 1.0),
    ],
)
def test_wilson_rejects_invalid_input(args):
    ci = po.wilson_confidence_interval(*args)
    assert ci["point"] is None


# ---- PSI ----


def test_psi_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(3)
    train = pd.Series(rng.normal(50, 5, 5000))
    batch = pd.Series(rng.normal(50, 5, 2000))
    assert po.calculate_psi(train, batch) < po.PSI_MODERATE_SHIFT


def test_psi_flags_a_mean_shift():
    rng = np.random.default_rng(3)
    train = pd.Series(rng.normal(50, 5, 5000))
    batch = pd.Series(rng.normal(62, 5, 2000))
    assert po.calculate_psi(train, batch) > po.PSI_MAJOR_SHIFT


def test_psi_flags_variance_inflation_at_constant_mean():
    rng = np.random.default_rng(3)
    train = pd.Series(rng.normal(50, 5, 5000))
    batch = pd.Series(rng.normal(50, 15, 2000))
    assert po.calculate_psi(train, batch) > po.PSI_MODERATE_SHIFT
    # This is the tool-wear case a mean comparison cannot see.


def test_psi_returns_none_when_it_cannot_bin():
    assert po.calculate_psi([1, 2, 3], [1, 2, 3]) is None
    assert po.calculate_psi([5] * 100, [5] * 50) is None
    assert po.calculate_psi(None, None) is None


# ---- Data quality auditing ----


def test_clean_dataset_scores_well(clean_frame):
    audit = po.audit_data_quality(clean_frame)
    assert audit["grade"] in ("production_ready", "usable_with_caveats")
    assert audit["dimensions"]["completeness"]["score"] == 100.0


def test_dimension_weights_always_sum_to_one(clean_frame):
    audit_no_time = po.audit_data_quality(clean_frame)
    assert sum(d["weight"] for d in audit_no_time["dimensions"].values()) == pytest.approx(1.0)
    assert "timeliness" not in audit_no_time["dimensions"]

    df_time = clean_frame.copy()
    df_time["event_date"] = pd.date_range("2024-01-01", periods=len(clean_frame), freq="h")
    audit_with_time = po.audit_data_quality(df_time)
    assert sum(d["weight"] for d in audit_with_time["dimensions"].values()) == pytest.approx(1.0)
    assert "timeliness" in audit_with_time["dimensions"]


def test_silent_null_tokens_are_detected(clean_frame):
    df = clean_frame.copy()
    df.loc[:59, "line"] = "N/A"
    df.loc[60:79, "line"] = "-"
    report = po.detect_silent_nulls(df)
    assert report["line"]["count"] == 80
    assert set(report["line"]["tokens"]) == {"n/a", "-"}
    assert report["line"]["kind"] == "placeholder_string"


def test_real_nan_is_not_counted_as_a_silent_null(clean_frame):
    df = clean_frame.copy()
    df.loc[:9, "line"] = np.nan
    report = po.detect_silent_nulls(df)
    assert "line" not in report


def test_numeric_sentinels_are_reported_but_not_modified(clean_frame):
    df = clean_frame.copy()
    df.loc[:14, "temp_c"] = -999
    report = po.detect_silent_nulls(df)
    assert report["temp_c"]["kind"] == "numeric_sentinel"
    assert (df["temp_c"] == -999).sum() == 15
    # A real reading could legitimately be -999, so this is reported only.


def test_missing_value_severity_follows_the_playbook(clean_frame):
    df = clean_frame.copy()
    df.loc[:119, "pressure_bar"] = np.nan  # 40% -> critical
    df.loc[:14, "temp_c"] = np.nan         # 5%  -> medium
    audit = po.audit_data_quality(df)
    severities = {i["column"]: i["severity"] for i in audit["issues"] if i["dimension"] == "completeness"}
    assert severities["pressure_bar"] == "critical"
    assert severities["temp_c"] == "medium"
    assert audit["issues"][0]["severity"] == "critical"  # sorted most severe first


def test_duplicate_rows_lower_uniqueness(clean_frame):
    df = pd.concat([clean_frame.head(50)] * 3, ignore_index=True)
    audit = po.audit_data_quality(df)
    assert audit["duplicate_rows"] == 100
    assert audit["dimensions"]["uniqueness"]["score"] < 40


def test_mixed_type_column_lowers_consistency(clean_frame):
    df = clean_frame.copy()
    df["reading"] = ["12.5"] * 150 + ["ERROR"] * 150
    audit = po.audit_data_quality(df)
    assert audit["dimensions"]["consistency"]["score"] < 100.0


def test_modified_zscore_finds_extremes(clean_frame):
    df = clean_frame.copy()
    df.loc[:4, "temp_c"] = 5000.0
    found = po.detect_outliers(df)
    assert found["temp_c"]["count"] == 5
    assert found["temp_c"]["method"] == "modified_zscore"
    assert found["temp_c"]["flagged_max"] == 5000.0


def test_modified_zscore_survives_zero_mad():
    df = pd.DataFrame({"v": [5.0] * 190 + [900.0] * 10})
    assert po.detect_outliers(df)["v"]["count"] == 10
    assert po.detect_outliers(pd.DataFrame({"v": [5.0] * 200})) == {}
    # MAD collapses when most values are identical; the mean-absolute-
    # deviation fallback keeps the statistic usable.


def test_iqr_method_reports_fences(clean_frame):
    df = clean_frame.copy()
    df.loc[:4, "temp_c"] = 5000.0
    found = po.detect_outliers(df, method="iqr")
    assert found["temp_c"]["method"] == "iqr"
    assert found["temp_c"]["flagged_min"] < found["temp_c"]["flagged_max"]


@pytest.mark.parametrize("bad", [None, pd.DataFrame(), [1, 2, 3]])
def test_audit_rejects_bad_input(bad):
    assert "error" in po.audit_data_quality(bad)


def test_inspect_data_surfaces_the_quality_score(clean_frame):
    df = clean_frame.copy()
    df["defect"] = np.random.default_rng(1).integers(0, 2, len(df))
    result = po.inspect_data(df, "defect")
    assert isinstance(result["data_quality"]["dqs"], float)
    assert len(result["data_quality"]["top_issues"]) <= 3
    for key in ("total_rows", "total_columns", "missing_values", "recommended_task", "recommended_metric", "message"):
        assert key in result
    error = po.inspect_data(df, "does_not_exist")
    assert "error" in error and "data_quality" not in error


# ---- Fit-time guardrails ----


@pytest.mark.parametrize(
    ("x_val", "y_val", "match_pattern"),
    [
        (None, [1, 2], "(?i)cannot be None"),
        (pd.DataFrame({"a": [1]}), None, "(?i)cannot be None"),
        (pd.DataFrame(), pd.Series(dtype=float), "(?i)empty"),
        (pd.DataFrame(index=range(20)), pd.Series(range(20)), "(?i)No feature columns"),
        (pd.DataFrame({"a": range(5)}), pd.Series(range(5)), "(?i)minimum of 10"),
    ],
)
def test_fit_rejects_invalid_inputs(x_val, y_val, match_pattern):
    with pytest.raises(ValueError, match=match_pattern):
        po.PotatOptEngine(time_budget=3).fit(x_val, y_val)


def test_fit_rejects_data_with_no_usable_features():
    n_rows = 40
    df = pd.DataFrame({
        "all_nan": [np.nan] * n_rows,
        "constant": [7] * n_rows,
        "uuid_id": [f"id-{i}" for i in range(n_rows)],
    })
    y = pd.Series([0, 1] * 20)
    with pytest.raises(ValueError, match="(?i)removed during preprocessing"):
        po.PotatOptEngine(time_budget=3).fit(df, y)


def test_duplicate_column_names_warn_without_raising(log_capture):
    df = pd.DataFrame(np.random.default_rng(2).random((20, 3)), columns=["a", "a", "b"])
    po.PotatOptEngine()._validate_fit_inputs(df, pd.Series(range(20)))
    assert any("Duplicate column names" in m for m in log_capture)


def test_duplicate_rows_warn(log_capture, clean_frame):
    df = pd.concat([clean_frame.head(100)] * 3, ignore_index=True)
    y = pd.Series(np.random.default_rng(4).integers(0, 2, len(df)))
    po.PotatOptEngine(task="classification", time_budget=3).fit(df, y)
    assert any("duplicate training rows" in m for m in log_capture)
    # Identical records can land in both cross-validation folds.


# ---- Inference guardrails (must warn, never raise) ----


def test_clean_prediction_raises_no_warning(binary_engine, signal_frame):
    x, _ = signal_frame
    binary_engine.predict(x.head(5))
    assert binary_engine.last_predict_warnings == []


def test_schema_mismatch_warns(binary_engine, signal_frame):
    x, _ = signal_frame
    binary_engine.predict(x.head(5)[["temp_c"]])
    assert any("schema mismatch" in w.lower() for w in binary_engine.last_predict_warnings)


def test_out_of_bounds_values_warn(binary_engine, signal_frame):
    x, _ = signal_frame
    bad = x.head(10).copy()
    bad["temp_c"] = 99999.0
    binary_engine.predict(bad)
    assert any("out-of-bounds" in w.lower() for w in binary_engine.last_predict_warnings)


def test_unrelated_input_does_not_crash_predict(binary_engine):
    binary_engine.predict(pd.DataFrame({"totally_unrelated": [1, 2, 3]}))
    # A production line must not stop because one batch arrived malformed.


# ---- Metrics ----


def test_binary_evaluation_reports_ranking_metrics(binary_engine, signal_frame):
    x, y = signal_frame
    result = binary_engine.evaluate(x, y)
    for key in (
        "roc_auc",
        "pr_auc",
        "defect_base_rate",
        "mcc",
        "recall_ci_95",
        "precision_ci_95",
        "n_test_rows",
    ):
        assert key in result
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0
    assert -1.0 <= result["mcc"] <= 1.0
    assert result["defect_base_rate"] == pytest.approx(float(y.mean()))


def test_confidence_intervals_bracket_their_point_estimate(binary_engine, signal_frame):
    x, y = signal_frame
    result = binary_engine.evaluate(x, y)
    assert result["recall_ci_95"]["lower"] <= result["recall"] <= result["recall_ci_95"]["upper"]
    assert result["precision_ci_95"]["lower"] <= result["precision"] <= result["precision_ci_95"]["upper"]


def test_auc_is_none_when_the_test_set_has_one_class(binary_engine, signal_frame):
    x, _ = signal_frame
    result = binary_engine.evaluate(x.head(20), pd.Series([0] * 20))
    assert result["roc_auc"] is None
    assert "accuracy" in result
    # AUC is undefined without both classes; the call must still return.


def test_regression_reports_mae_and_mape(regression_engine, regression_frame):
    x, y = regression_frame
    result = regression_engine.evaluate(x, y)
    for key in ("mae", "mape", "n_test_rows", "r2", "rmse"):
        assert key in result
    assert result["mae"] <= result["rmse"] + 1e-9
    assert result["mape"] >= 0.0


# ---- AutoML search reporting ----


def test_binary_validation_score_is_the_roc_auc(binary_engine):
    report = binary_engine.get_training_report()
    assert report["metric_optimized"] == "roc_auc"
    assert report["validation_score"] is not None
    assert report["potatopt_version"] == po.__version__


def test_multiclass_validation_score_is_withheld():
    rng = np.random.default_rng(21)
    n_multi = 300
    x = pd.DataFrame({
        "f1": rng.normal(0, 1, n_multi),
        "f2": rng.normal(0, 1, n_multi),
        "f3": rng.normal(0, 1, n_multi),
    })
    y = pd.Series(rng.integers(0, 4, n_multi))
    engine = po.PotatOptEngine(task="classification", time_budget=3).fit(x, y)
    report = engine.get_training_report()
    assert report["metric_optimized"] == "log_loss"
    assert report["validation_score"] is None
    assert report["validation_loss"] is not None
    # FLAML minimises raw log_loss there, so 1 - loss has no meaning.


def test_regression_validation_score_is_the_r2(regression_engine):
    report = regression_engine.get_training_report()
    assert report["metric_optimized"] == "r2"
    assert report["validation_score"] is not None


def test_training_report_requires_a_fitted_engine():
    assert "error" in po.PotatOptEngine().get_training_report()


# ---- Anomaly fallback ----


def test_extreme_imbalance_switches_to_anomaly_detection(anomaly_engine):
    assert anomaly_engine.is_anomaly_model is True
    assert anomaly_engine.get_training_report()["best_estimator"] == "IsolationForest"
    assert anomaly_engine.get_training_report()["validation_score"] is None


def test_anomaly_model_still_predicts_and_evaluates(anomaly_engine):
    rng = np.random.default_rng(5)
    n_rows = 120
    x = pd.DataFrame({
        "s1": rng.normal(0, 1, n_rows),
        "s2": rng.normal(0, 1, n_rows),
        "s3": rng.normal(0, 1, n_rows),
    })
    y = pd.Series([0] * 117 + [1] * 3)
    assert len(anomaly_engine.predict(x.head(3))) == 3
    assert "accuracy" in anomaly_engine.evaluate(x, y)


# ---- Drift detection ----


def test_training_profile_is_captured(binary_engine):
    assert len(binary_engine.train_profile) > 0
    assert binary_engine.train_timestamp is not None
    assert len(binary_engine.train_data_hash) == 64


def test_stable_batch_is_not_flagged(binary_engine, signal_frame):
    x, _ = signal_frame
    result = binary_engine.detect_drift(x.sample(150, random_state=3))
    assert result["drift_detected"] is False
    assert "stable" in result["recommendation"].lower()


def test_shifted_batch_is_flagged_in_engineering_units(binary_engine, signal_frame):
    x, _ = signal_frame
    batch = x.sample(150, random_state=3).copy()
    batch["temp_c"] = batch["temp_c"] + 25.0
    result = binary_engine.detect_drift(batch)
    assert result["drift_detected"] is True
    feature = result["features"]["temp_c"]
    assert feature["severity"] == "major"
    assert 48 < feature["train_mean_raw"] < 52
    assert 71 < feature["batch_mean_raw"] < 79
    assert feature["mean_shift_sigma"] > 2.0
    assert "retrain" in result["recommendation"].lower()


def test_variance_inflation_is_caught_when_the_mean_holds(binary_engine, signal_frame):
    x, _ = signal_frame
    batch = x.sample(150, random_state=3).copy()
    centre = batch["cycle_time"].mean()
    batch["cycle_time"] = centre + (batch["cycle_time"] - centre) * 4.0
    feature = binary_engine.detect_drift(batch)["features"]["cycle_time"]
    assert feature["mean_shift_sigma"] < 0.5
    assert feature["std_ratio"] > 2.5
    assert feature["severity"] in ("moderate", "major")
    # The mean never moves, so only PSI and the variance ratio see this.


def test_drift_monitoring_sees_unclipped_values(binary_engine, signal_frame):
    x, _ = signal_frame
    extreme = x.head(5).copy()
    extreme["temp_c"] = 99999.0
    clipped = binary_engine._preprocess_transform(extreme)
    raw = binary_engine._preprocess_transform(extreme, apply_bounds_clip=False)
    assert clipped["temp_c"].max() < raw["temp_c"].max()
    # The prediction path clips to training bounds; the monitoring path
    # must not, or clipping would hide the shift it is meant to report.


def test_drift_requires_a_fitted_engine(clean_frame):
    assert "error" in po.PotatOptEngine().detect_drift(clean_frame)


def test_drift_tolerates_an_unrelated_batch(binary_engine):
    assert isinstance(binary_engine.detect_drift(pd.DataFrame({"zzz": [1, 2, 3]})), dict)


# ---- ISO 9001 traceability ----


def test_metadata_records_full_provenance(binary_engine, tmp_path):
    path = str(tmp_path / "model.pkl")
    binary_engine.save(path)
    metadata = json.loads((tmp_path / "model_metadata.json").read_text(encoding="utf-8"))
    for key in (
        "potatopt_version",
        "saved_at_utc",
        "trained_at_utc",
        "train_data_sha256",
        "n_train_rows",
        "automl_metric",
        "library_versions",
        "drift_profile_features",
        "train_data_quality",
        "model_hash_sha256",
        "scikit_learn_version",
        "features_used",
    ):
        assert key in metadata
    assert len(metadata["train_data_sha256"]) == 64
    assert metadata["library_versions"]["pandas"] is not None


def test_training_hash_changes_with_the_data(clean_frame):
    y = pd.Series(np.random.default_rng(6).integers(0, 2, len(clean_frame)))
    first = po.PotatOptEngine(task="classification", time_budget=3).fit(clean_frame, y)
    second = po.PotatOptEngine(task="classification", time_budget=3).fit(clean_frame, y)
    assert first.train_data_hash == second.train_data_hash
    altered = clean_frame.copy()
    altered.iloc[0, 0] = altered.iloc[0, 0] + 0.001
    third = po.PotatOptEngine(task="classification", time_budget=3).fit(altered, y)
    assert third.train_data_hash != first.train_data_hash


def test_drift_detection_survives_a_save_load_round_trip(binary_engine, signal_frame, tmp_path):
    x, _ = signal_frame
    batch = x.sample(150, random_state=3).copy()
    batch["temp_c"] += 25.0
    expected = binary_engine.detect_drift(batch)["max_psi"]
    path = str(tmp_path / "roundtrip.pkl")
    binary_engine.save(path)
    restored = po.PotatOptEngine.load(path)
    assert restored.detect_drift(batch)["max_psi"] == pytest.approx(expected)
    # This is the whole point - drift must be measurable after the
    # training set is gone.


def test_audit_log_persists_warnings(tmp_path, clean_frame):
    path = str(tmp_path / "audit.log")
    abs_path = os.path.abspath(path)
    try:
        assert po.enable_audit_log(path) == abs_path
        assert po.enable_audit_log(path) == abs_path  # idempotent

        rng = np.random.default_rng(5)
        n_rows = 120
        x = pd.DataFrame({
            "s1": rng.normal(0, 1, n_rows),
            "s2": rng.normal(0, 1, n_rows),
            "s3": rng.normal(0, 1, n_rows),
        })
        y = pd.Series([0] * 117 + [1] * 3)
        po.PotatOptEngine(task="classification", time_budget=3).fit(x, y)

        for handler in po.logger.handlers:
            handler.flush()

        content = (tmp_path / "audit.log").read_text(encoding="utf-8")
        assert "Extreme Imbalance Detected" in content
        matching_lines = [line for line in content.splitlines() if "Extreme Imbalance Detected" in line]
        assert any("WARNING" in line for line in matching_lines)
    finally:
        handlers_to_remove = [
            h
            for h in po.logger.handlers
            if isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", None) == abs_path
        ]
        for h in handlers_to_remove:
            h.close()
            po.logger.removeHandler(h)


# ---- Silent-null handling inside the engine ----


def test_placeholders_are_converted_during_fit(clean_frame, log_capture):
    df = clean_frame.copy()
    df.loc[:39, "line"] = "N/A"
    df.loc[40:59, "line"] = "-"
    y = pd.Series(np.random.default_rng(8).integers(0, 2, len(df)))
    engine = po.PotatOptEngine(task="classification", time_budget=3).fit(df, y)
    assert engine.silent_nulls_converted == 60
    assert any("placeholder values" in m for m in log_capture)


def test_placeholder_handling_can_be_disabled(clean_frame):
    df = clean_frame.copy()
    df.loc[:39, "line"] = "N/A"
    df.loc[40:59, "line"] = "-"
    y = pd.Series(np.random.default_rng(8).integers(0, 2, len(df)))
    engine = po.PotatOptEngine(task="classification", time_budget=3, handle_silent_nulls=False).fit(df, y)
    assert engine.silent_nulls_converted == 0


def test_an_all_placeholder_column_is_dropped(clean_frame):
    df = clean_frame.copy()
    df["operator"] = "N/A"
    y = pd.Series(np.random.default_rng(8).integers(0, 2, len(df)))
    engine = po.PotatOptEngine(task="classification", time_budget=3).fit(df, y)
    assert "operator" in engine.all_nan_cols
    # Without the conversion it would survive as a constant category.


# ---- Inference observability ----


def test_counters_start_at_zero_after_fit(clean_frame):
    y = pd.Series(np.random.default_rng(9).integers(0, 2, len(clean_frame)))
    engine = po.PotatOptEngine(task="classification", time_budget=3).fit(clean_frame, y)
    assert engine.get_inference_health()["transform_calls"] == 0


def test_counters_track_transforms_and_warnings(clean_frame):
    y = pd.Series(np.random.default_rng(10).integers(0, 2, len(clean_frame)))
    engine = po.PotatOptEngine(task="classification", time_budget=3).fit(clean_frame, y)
    engine.predict(clean_frame.head(10))
    engine.predict(clean_frame.head(5))
    health = engine.get_inference_health()
    assert health["transform_calls"] == 2
    assert health["rows_transformed"] == 15
    assert health["warning_events"] == 0

    bad = clean_frame.head(10).copy()
    bad["temp_c"] = 99999.0
    engine.predict(bad)
    health_after = engine.get_inference_health()
    assert health_after["warning_events"] == 1
    assert health_after["warning_rate"] == pytest.approx(1 / 3)


def test_inference_health_requires_a_fitted_engine():
    assert "error" in po.PotatOptEngine().get_inference_health()


# ---- Phase 2.6: split_data sizing ----


@pytest.fixture(scope="module")
def split_frame():
    rng = np.random.default_rng(21)
    return pd.DataFrame({
        "temp_c": rng.normal(50, 4, 200),
        "pressure_bar": rng.normal(10, 1, 200),
        "defect": rng.integers(0, 2, 200),
    })


def test_split_data_defaults_to_eighty_twenty(split_frame):
    x_train, x_test, _, _ = po.split_data(split_frame, "defect")
    assert (len(x_train), len(x_test)) == (160, 40)


def test_split_data_honours_a_custom_fraction(split_frame):
    x_train, x_test, _, _ = po.split_data(split_frame, "defect", test_size=0.3)
    assert (len(x_train), len(x_test)) == (140, 60)


def test_split_data_accepts_an_absolute_row_count(split_frame):
    x_train, x_test, _, _ = po.split_data(split_frame, "defect", test_size=25)
    assert (len(x_train), len(x_test)) == (175, 25)


@pytest.mark.parametrize("bad", [0, 1.0, 1.5, -0.2, "0.2", None, True])
def test_split_data_rejects_invalid_test_size(split_frame, bad):
    with pytest.raises(ValueError):
        po.split_data(split_frame, "defect", test_size=bad)


# ---- Phase 2.6: three-way split ----


def test_three_way_split_sizes_are_exact(split_frame):
    x_train, x_val, x_test, _, _, _ = po.split_data_three_way(split_frame, "defect")
    assert (len(x_train), len(x_val), len(x_test)) == (120, 40, 40)


def test_three_way_val_size_is_a_share_of_the_whole_dataset(split_frame):
    # Regression test: 0.1 / (1 - 0.4) is not exact in binary floating point and
    # used to push one extra row into the validation partition (99/21/80).
    x_train, x_val, x_test, _, _, _ = po.split_data_three_way(
        split_frame, "defect", val_size=0.1, test_size=0.4
    )
    assert (len(x_train), len(x_val), len(x_test)) == (100, 20, 80)


def test_three_way_partitions_are_disjoint_and_complete(split_frame):
    x_train, x_val, x_test, y_train, y_val, y_test = po.split_data_three_way(split_frame, "defect")
    indexes = set(x_train.index) | set(x_val.index) | set(x_test.index)
    assert len(indexes) == len(split_frame)
    for x_part, y_part in ((x_train, y_train), (x_val, y_val), (x_test, y_test)):
        assert list(x_part.columns) == ["temp_c", "pressure_bar"]
        assert (split_frame.loc[x_part.index, "defect"].to_numpy() == np.asarray(y_part)).all()


def test_three_way_forecasting_keeps_train_val_test_in_time_order():
    frame = pd.DataFrame({"t": np.arange(200.0), "y": np.arange(200.0) * 2})
    x_train, x_val, x_test, _, _, _ = po.split_data_three_way(frame, "y", task="forecasting")
    assert x_train["t"].max() < x_val["t"].min()
    assert x_val["t"].max() < x_test["t"].min()
    assert (len(x_train), len(x_val), len(x_test)) == (120, 40, 40)


def test_three_way_drops_rows_with_a_missing_target(split_frame):
    frame = split_frame.copy()
    frame.loc[frame.index[:20], "defect"] = np.nan
    x_train, x_val, x_test, _, _, _ = po.split_data_three_way(frame, "defect")
    assert len(x_train) + len(x_val) + len(x_test) == 180


@pytest.mark.parametrize("kwargs", [
    {"val_size": 0.9, "test_size": 0.2},
    {"val_size": 0.5, "test_size": 0.5},
    {"val_size": 0},
    {"val_size": 1.2},
    {"val_size": "0.2"},
])
def test_three_way_rejects_impossible_proportions(split_frame, kwargs):
    with pytest.raises(ValueError):
        po.split_data_three_way(split_frame, "defect", **kwargs)


# ---- Phase 2.6: forecasting must not shuffle time-series rows ----


def test_forecasting_sends_a_time_split_to_automl(monkeypatch):
    # FLAML resolves regression + split_type="auto" to "uniform" (a random
    # shuffle), which would validate past rows against future ones.
    captured = {}
    original_fit = po.AutoML.fit

    def spy(self, *args, **kwargs):
        captured.update(kwargs)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(po.AutoML, "fit", spy)

    rng = np.random.default_rng(31)
    x = pd.DataFrame({
        "lag_1": np.linspace(0, 10, 200) + rng.normal(0, 0.1, 200),
        "lag_2": np.linspace(0, 5, 200) + rng.normal(0, 0.1, 200),
    })
    y = pd.Series(x["lag_1"] * 3 + rng.normal(0, 0.1, 200))

    engine = po.PotatOptEngine(task="forecasting", time_budget=3, estimators=["lgbm"]).fit(x, y)
    assert engine.is_fitted
    assert captured.get("split_type") == "time"
    assert captured.get("eval_method") == "cv"


def test_plain_regression_is_not_forced_into_a_time_split(monkeypatch, regression_frame):
    captured = {}
    original_fit = po.AutoML.fit

    def spy(self, *args, **kwargs):
        captured.update(kwargs)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(po.AutoML, "fit", spy)

    x, y = regression_frame
    po.PotatOptEngine(task="regression", time_budget=3, estimators=["lgbm"]).fit(x, y)
    assert "split_type" not in captured


# ---- Phase 2.6: threshold-tuning leakage detection ----


def test_engine_has_no_tuning_fingerprint_before_tuning(binary_engine):
    assert po.PotatOptEngine().threshold_tuning_fingerprint is None
    assert hasattr(binary_engine, "threshold_tuning_fingerprint")


def test_tuning_on_a_separate_set_raises_no_leakage_flag(signal_frame):
    x, y = signal_frame
    frame = x.copy()
    frame["defect"] = y.to_numpy()
    x_train, x_val, x_test, y_train, y_val, y_test = po.split_data_three_way(frame, "defect")

    engine = po.PotatOptEngine(task="classification", time_budget=5).fit(x_train, y_train)
    engine.optimize_threshold(x_val, y_val, cost_scrap=500, cost_fa=150, cost_insp=20)

    assert engine.threshold_tuning_fingerprint is not None
    assert engine.evaluate(x_test, y_test)["threshold_leakage_warning"] is False


def test_tuning_on_the_reported_set_is_flagged_and_logged(signal_frame, log_capture):
    x, y = signal_frame
    frame = x.copy()
    frame["defect"] = y.to_numpy()
    x_train, x_test, y_train, y_test = po.split_data(frame, "defect")

    engine = po.PotatOptEngine(task="classification", time_budget=5).fit(x_train, y_train)
    engine.optimize_threshold(x_test, y_test, cost_scrap=500, cost_fa=150, cost_insp=20)
    metrics = engine.evaluate(x_test, y_test)

    assert metrics["threshold_leakage_warning"] is True
    assert any("tuned on this same dataset" in message for message in log_capture)


def test_evaluation_without_tuning_reports_no_leakage(binary_engine, signal_frame):
    x, y = signal_frame
    assert binary_engine.evaluate(x, y)["threshold_leakage_warning"] is False


def test_fingerprint_separates_different_datasets(signal_frame):
    x, y = signal_frame
    first = po.PotatOptEngine._dataset_fingerprint(x.head(100), y.head(100))
    second = po.PotatOptEngine._dataset_fingerprint(x.tail(100), y.tail(100))
    assert first is not None and second is not None
    assert first != second


def test_fingerprint_ignores_index_labels(signal_frame):
    x, y = signal_frame
    original = po.PotatOptEngine._dataset_fingerprint(x, y)
    shifted_x = x.copy()
    shifted_y = y.copy()
    shifted_x.index = shifted_x.index + 5000
    shifted_y.index = shifted_y.index + 5000
    assert po.PotatOptEngine._dataset_fingerprint(shifted_x, shifted_y) == original


@pytest.mark.parametrize("junk", [object(), None, 42, "rows"])
def test_fingerprint_returns_none_for_unusable_input(junk):
    assert po.PotatOptEngine._dataset_fingerprint(junk, junk) is None


def test_regression_engine_never_records_a_tuning_fingerprint(regression_engine, regression_frame):
    x, y = regression_frame
    assert regression_engine.optimize_threshold(x, y) == 0.5
    assert regression_engine.threshold_tuning_fingerprint is None
    assert "threshold_leakage_warning" not in regression_engine.evaluate(x, y)


def test_tuning_fingerprint_survives_a_save_load_round_trip(tmp_path, signal_frame):
    x, y = signal_frame
    frame = x.copy()
    frame["defect"] = y.to_numpy()
    x_train, x_val, x_test, y_train, y_val, y_test = po.split_data_three_way(frame, "defect")  # noqa: RUF059

    engine = po.PotatOptEngine(task="classification", time_budget=5).fit(x_train, y_train)
    engine.optimize_threshold(x_val, y_val)
    destination = tmp_path / "phase26_model.pkl"
    engine.save(str(destination))

    restored = po.PotatOptEngine.load(str(destination))
    assert restored.threshold_tuning_fingerprint == engine.threshold_tuning_fingerprint
    assert restored.evaluate(x_val, y_val)["threshold_leakage_warning"] is True


# ---- Phase 2.7: LowSpecML memory budget ----
#
# These are the tests that make "LowSpecML" falsifiable. The first measures the
# downcaster exactly (pandas reports its own footprint, so there is nothing to
# estimate). The second measures real process RSS and is therefore approximate;
# its ceiling is deliberately loose so it fails on a genuine regression - an
# accidental one-hot expansion, a stray float64 copy - and not on noise.

MEMORY_BENCHMARK_ROWS = 20000
DOWNCAST_MIN_SAVING = 0.49
PIPELINE_MIN_SAVING = 0.70
FIT_RSS_CEILING_MB = 400.0


def _benchmark_frame(rows=MEMORY_BENCHMARK_ROWS):
    # float64 sensors plus low-cardinality categoricals: the shape of a real
    # multi-machine condition-monitoring extract.
    rng = np.random.default_rng(99)
    return pd.DataFrame({
        "temp_c": rng.normal(50, 4, rows),
        "pressure_bar": rng.normal(10, 1, rows),
        "vibration_mm_s": rng.normal(2, 0.4, rows),
        "cycle_time": rng.normal(30, 3, rows),
        "rpm": rng.normal(1500, 60, rows),
        "machine_id": rng.choice(["M01", "M02", "M03", "M04"], rows),
        "shift": rng.choice(["A", "B", "C"], rows),
    })


def test_numeric_downcasting_halves_the_sensor_columns():
    # float64 -> float32 is exactly half, so this figure is deterministic. Only
    # the numeric columns are measured: _reduce_mem_usage does not touch strings.
    frame = _benchmark_frame().select_dtypes("number")
    engine = po.PotatOptEngine()
    before = frame.memory_usage(deep=True).sum()
    after = engine._reduce_mem_usage(frame.copy()).memory_usage(deep=True).sum()
    saving = 1 - (after / before)
    assert saving >= DOWNCAST_MIN_SAVING, (
        f"Numeric downcaster saved only {saving:.1%} "
        f"({before / 1024**2:.2f}MB -> {after / 1024**2:.2f}MB); LowSpecML claim regressed."
    )


def test_full_preprocessing_shrinks_a_mixed_frame_by_most_of_its_size():
    # The claim that matters: what the engine actually hands to the model. Sensor
    # floats are downcast and the string columns become `category` rather than
    # being one-hot expanded, so the frame gets several times smaller, not larger.
    frame = _benchmark_frame()
    target = pd.Series((frame["temp_c"] > 54).astype(int))
    engine = po.PotatOptEngine(task="classification", time_budget=3)
    before = frame.memory_usage(deep=True).sum()
    processed = engine._preprocess_fit_transform(frame.copy(), target)
    after = processed.memory_usage(deep=True).sum()
    saving = 1 - (after / before)
    assert saving >= PIPELINE_MIN_SAVING, (
        f"Preprocessing saved only {saving:.1%} "
        f"({before / 1024**2:.2f}MB -> {after / 1024**2:.2f}MB); LowSpecML claim regressed."
    )
    assert len(processed) == len(frame)
    assert "float64" not in set(processed.dtypes.astype(str))


def test_downcasting_is_lossless_within_float32_precision():
    frame = _benchmark_frame(rows=2000)
    reduced = po.PotatOptEngine()._reduce_mem_usage(frame.copy())
    for column in ["temp_c", "pressure_bar", "vibration_mm_s", "cycle_time", "rpm"]:
        assert np.allclose(frame[column], reduced[column], rtol=1e-6), column
    assert len(reduced) == len(frame)
    assert list(reduced.columns) == list(frame.columns)


@pytest.mark.skipif(psutil is None, reason="psutil is not installed")
def test_fit_stays_inside_the_low_spec_ram_budget():
    frame = _benchmark_frame()
    target = pd.Series((frame["temp_c"] > 54).astype(int))

    process = psutil.Process()
    engine = po.PotatOptEngine(task="classification", time_budget=5, estimators=["lgbm"], n_jobs=1)

    baseline_mb = process.memory_info().rss / 1024**2
    engine.fit(frame, target)
    peak_mb = process.memory_info().rss / 1024**2
    growth_mb = peak_mb - baseline_mb

    assert engine.is_fitted
    assert growth_mb < FIT_RSS_CEILING_MB, (
        f"fit() grew RSS by {growth_mb:.1f}MB on {len(frame)} rows, "
        f"over the {FIT_RSS_CEILING_MB:.0f}MB LowSpecML ceiling."
    )


# ---- Phase 2.7: n_jobs is the user's to choose ----


def test_n_jobs_defaults_to_every_core():
    assert po.PotatOptEngine().n_jobs == -1


@pytest.mark.parametrize("value", [1, 2, -1, -2])
def test_n_jobs_accepts_valid_worker_counts(value):
    assert po.PotatOptEngine(n_jobs=value).n_jobs == value


@pytest.mark.parametrize("bad", [0, 1.5, "2", None, True])
def test_n_jobs_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        po.PotatOptEngine(n_jobs=bad)


def test_n_jobs_reaches_automl(monkeypatch, signal_frame):
    captured = {}
    original_fit = po.AutoML.fit

    def spy(self, *args, **kwargs):
        captured.update(kwargs)
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(po.AutoML, "fit", spy)

    x, y = signal_frame
    po.PotatOptEngine(task="classification", time_budget=3, estimators=["lgbm"], n_jobs=1).fit(x, y)
    assert captured.get("n_jobs") == 1


def test_n_jobs_reaches_the_anomaly_fallback():
    rng = np.random.default_rng(77)
    x = pd.DataFrame({
        "s1": rng.normal(0, 1, 120),
        "s2": rng.normal(0, 1, 120),
    })
    y = pd.Series([0] * 117 + [1] * 3)
    engine = po.PotatOptEngine(task="classification", time_budget=3, n_jobs=1).fit(x, y)
    assert engine.is_anomaly_model
    assert engine.model.n_jobs == 1


# ---- Phase 2.7: public API surface ----


def test_all_is_declared_and_complete():
    assert hasattr(po, "__all__")
    for name in po.__all__:
        assert hasattr(po, name), f"__all__ promises {name} but the module does not define it"


@pytest.mark.parametrize("leaked", ["np", "pd", "os", "json", "re", "hashlib", "shap", "sklearn"])
def test_third_party_imports_are_not_public_api(leaked):
    assert leaked not in po.__all__


def test_star_import_exposes_only_the_declared_surface():
    namespace = {}
    exec("from potatopt import *", namespace)  # noqa: S102
    # Only `__builtins__` is injected by exec itself. Every other name present -
    # `__version__` included - came from `__all__`, so the comparison is exact.
    exported = {name for name in namespace if name != "__builtins__"}
    assert exported == set(po.__all__)


# ---- Phase 2.7: to_jsonable ----


@pytest.mark.parametrize("value,expected", [
    (np.int64(7), 7),
    (np.float64(1.5), 1.5),
    (np.bool_(True), True),
    (np.float64("nan"), None),
    (np.float64("inf"), None),
    (float("-inf"), None),
    (None, None),
    ("text", "text"),
    (True, True),
])
def test_to_jsonable_converts_scalars(value, expected):
    assert po.to_jsonable(value) == expected


def test_to_jsonable_converts_arrays_and_frames():
    assert po.to_jsonable(np.array([1, 2, 3])) == [1, 2, 3]
    assert po.to_jsonable(pd.Series([1.0, 2.0])) == [1.0, 2.0]
    assert po.to_jsonable(pd.DataFrame({"a": [1], "b": ["x"]})) == [{"a": 1, "b": "x"}]


def test_to_jsonable_recurses_through_nested_containers():
    payload = {"metrics": {"f1": np.float64(0.8)}, "rows": [np.int64(1), (np.float64("nan"), "ok")]}
    assert po.to_jsonable(payload) == {"metrics": {"f1": 0.8}, "rows": [1, [None, "ok"]]}


def test_to_jsonable_output_survives_strict_json():
    payload = {"a": np.float64("nan"), "b": np.array([1, 2]), "c": pd.Series(["x"])}
    encoded = json.dumps(po.to_jsonable(payload), allow_nan=False)
    assert json.loads(encoded) == {"a": None, "b": [1, 2], "c": ["x"]}


def test_to_jsonable_stringifies_what_it_cannot_map():
    assert po.to_jsonable(object()).startswith("<object object")


def test_to_jsonable_makes_predict_and_importance_serialisable(binary_engine, signal_frame):
    x, _ = signal_frame
    predictions = po.to_jsonable(binary_engine.predict(x.head(5)))
    assert isinstance(predictions, list) and len(predictions) == 5
    json.dumps(predictions, allow_nan=False)

    importance = po.to_jsonable(binary_engine.get_feature_importance())
    assert isinstance(importance, list)
    assert set(importance[0]) == {"feature", "importance"}
    json.dumps(importance, allow_nan=False)


# ---- Phase 2.7: explain_predictions ----


def test_explain_predictions_returns_ranked_json(binary_engine, signal_frame):
    x, _ = signal_frame
    report = binary_engine.explain_predictions(x.head(50))

    assert report["available"] is True
    assert report["reason"] is None
    assert report["n_rows_explained"] == 50
    attributions = report["feature_attributions"]
    assert len(attributions) == len(binary_engine.feature_names)
    scores = [row["mean_abs_shap"] for row in attributions]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(row["feature"], str) for row in attributions)
    json.dumps(report, allow_nan=False)


def test_explain_predictions_honours_top_k(binary_engine, signal_frame):
    x, _ = signal_frame
    report = binary_engine.explain_predictions(x.head(30), top_k=2)
    assert report["top_k"] == 2
    assert len(report["feature_attributions"]) == 2


def test_explain_predictions_caps_the_rows_it_explains(binary_engine, signal_frame):
    x, _ = signal_frame
    report = binary_engine.explain_predictions(x, max_rows=10)
    assert report["n_rows_explained"] == 10


def test_explain_predictions_reports_why_it_is_unavailable_when_unfitted():
    report = po.PotatOptEngine().explain_predictions(pd.DataFrame({"a": [1.0]}))
    assert report["available"] is False
    assert "not fitted" in report["reason"].lower()
    json.dumps(report, allow_nan=False)


def test_explain_predictions_declines_on_the_anomaly_fallback(anomaly_engine):
    report = anomaly_engine.explain_predictions(pd.DataFrame({
        "s1": [0.1, 0.2], "s2": [0.3, 0.4], "s3": [0.5, 0.6],
    }))
    assert report["available"] is False
    assert "anomaly" in report["reason"].lower()
    json.dumps(report, allow_nan=False)


def test_explain_predictions_never_raises_on_bad_input(binary_engine):
    report = binary_engine.explain_predictions(pd.DataFrame({"unrelated": [1, 2, 3]}))
    assert isinstance(report, dict)
    assert set(report) >= {"available", "reason", "feature_attributions"}
    json.dumps(report, allow_nan=False)


# ---- Phase 2.7: scikit-learn estimator compatibility ----


def test_engine_exposes_the_sklearn_parameter_protocol():
    engine = po.PotatOptEngine(task="classification", time_budget=7, n_jobs=2)
    params = engine.get_params()
    assert params["task"] == "classification"
    assert params["n_jobs"] == 2
    engine.set_params(scale_method="minmax")
    assert engine.scale_method == "minmax"


def test_engine_survives_sklearn_clone():
    from sklearn.base import clone

    engine = po.PotatOptEngine(task="regression", time_budget=11, estimators=["lgbm"], n_jobs=1)
    twin = clone(engine)
    assert twin is not engine
    assert twin.get_params() == engine.get_params()
    assert twin.is_fitted is False


def test_engine_reports_its_task_to_sklearn():
    from sklearn.base import is_classifier, is_regressor

    assert is_classifier(po.PotatOptEngine(task="classification"))
    assert is_regressor(po.PotatOptEngine(task="regression"))
    assert is_regressor(po.PotatOptEngine(task="forecasting"))
    # "auto" is unresolved until fit(), so it stays deliberately untagged.
    assert not is_classifier(po.PotatOptEngine(task="auto"))


def test_classes_is_absent_before_fit_and_present_after(binary_engine):
    assert not hasattr(po.PotatOptEngine(), "classes_")
    assert list(binary_engine.classes_) == [0, 1]


def test_sklearn_sees_the_engine_as_fitted(binary_engine):
    from sklearn.utils.validation import check_is_fitted

    assert po.PotatOptEngine().__sklearn_is_fitted__() is False
    assert binary_engine.__sklearn_is_fitted__() is True
    check_is_fitted(binary_engine)


def test_cross_val_score_runs_on_the_engine(signal_frame):
    from sklearn.model_selection import cross_val_score

    x, y = signal_frame
    engine = po.PotatOptEngine(task="classification", time_budget=2, estimators=["lgbm"], n_jobs=1)
    scores = cross_val_score(engine, x, y, cv=3, scoring="f1", error_score="raise")
    assert len(scores) == 3
    assert all(0.0 <= float(s) <= 1.0 for s in scores)


def test_grid_search_tunes_an_engine_parameter(signal_frame):
    from sklearn.model_selection import GridSearchCV

    x, y = signal_frame
    search = GridSearchCV(
        po.PotatOptEngine(task="classification", time_budget=2, estimators=["lgbm"], n_jobs=1),
        {"scale_method": ["standard", "minmax"]},
        cv=3,
        scoring="f1",
        error_score="raise",
    )
    search.fit(x, y)
    assert search.best_params_["scale_method"] in {"standard", "minmax"}


def test_engine_works_as_the_final_step_of_a_pipeline(signal_frame):
    from sklearn.pipeline import Pipeline

    x, y = signal_frame
    pipeline = Pipeline([
        ("model", po.PotatOptEngine(task="classification", time_budget=2, estimators=["lgbm"], n_jobs=1)),
    ])
    pipeline.fit(x, y)
    predictions = pipeline.predict(x.head(4))
    assert len(predictions) == 4


def test_sklearn_compatibility_does_not_break_save_and_load(tmp_path, binary_engine, signal_frame):
    x, _ = signal_frame
    destination = tmp_path / "sklearn_compat.pkl"
    binary_engine.save(str(destination))
    restored = po.PotatOptEngine.load(str(destination))
    assert restored.get_params() == binary_engine.get_params()
    assert list(restored.predict(x.head(3))) == list(binary_engine.predict(x.head(3)))


# ---- auto_analyze: the one-call front door ----


@pytest.fixture(scope="module")
def failure_frame():
    rng = np.random.default_rng(202)
    rows = 400
    frame = pd.DataFrame({
        "temp_c": rng.normal(50, 4, rows),
        "vibration_mm_s": rng.normal(2, 0.4, rows),
        "pressure_bar": rng.normal(10, 1, rows),
        "machine_id": rng.choice(["M01", "M02", "M03"], rows),
    })
    frame["failure"] = ((frame["temp_c"] > 54) | (frame["vibration_mm_s"] > 2.6)).astype(int)
    return frame


def test_auto_analyze_returns_a_complete_report(failure_frame):
    report = po.auto_analyze(failure_frame, target="failure", time_budget=5)

    assert report["ok"] is True
    assert report["error"] is None
    assert report["task"] == "classification"
    assert report["rows"] == len(failure_frame)
    assert report["features"] == 4
    assert report["split"] == {"train": 240, "validation": 80, "test": 80}
    assert report["data_quality"]["dqs"] is not None
    assert report["metrics"]["f1"] is not None
    assert report["cost"]["cost_savings"] is not None
    assert report["threshold"]["tuned_on"] == "validation"
    assert report["model"]["name"]


def test_auto_analyze_output_survives_strict_json(failure_frame):
    report = po.auto_analyze(failure_frame, target="failure", time_budget=5)
    assert json.loads(json.dumps(report, allow_nan=False))["ok"] is True


def test_auto_analyze_does_not_leak_the_tuning_set(failure_frame):
    # The whole point of the internal three-way split: the threshold is tuned on
    # validation, so the reported metrics come from rows nothing has seen.
    report = po.auto_analyze(failure_frame, target="failure", time_budget=5)
    assert report["metrics"]["threshold_leakage_warning"] is False


def test_auto_analyze_ranks_features(failure_frame):
    report = po.auto_analyze(failure_frame, target="failure", time_budget=5, top_features=2)
    assert len(report["top_features"]) == 2
    scores = [row["mean_abs_shap"] for row in report["top_features"]]
    assert scores == sorted(scores, reverse=True)


def test_auto_analyze_reads_a_csv_path(tmp_path, failure_frame):
    destination = tmp_path / "sensors.csv"
    failure_frame.to_csv(destination, index=False)
    report = po.auto_analyze(str(destination), target="failure", time_budget=5)
    assert report["ok"] is True
    assert report["rows"] == len(failure_frame)


def test_auto_analyze_can_save_the_fitted_engine(tmp_path, failure_frame):
    destination = tmp_path / "auto_model.pkl"
    report = po.auto_analyze(failure_frame, target="failure", time_budget=5, save_to=str(destination))
    assert report["saved_to"] == str(destination)
    assert destination.exists()
    assert po.PotatOptEngine.load(str(destination)).is_fitted


def test_auto_analyze_handles_regression(failure_frame):
    frame = failure_frame.drop(columns=["failure"]).copy()
    frame["wear_mm"] = frame["temp_c"] * 0.3 + frame["vibration_mm_s"]
    report = po.auto_analyze(frame, target="wear_mm", task="regression", time_budget=5)
    assert report["ok"] is True
    assert report["task"] == "regression"
    # Money and a decision threshold are meaningless without classes.
    assert report["threshold"] is None
    assert report["cost"] is None
    assert report["metrics"]["r2"] is not None


@pytest.mark.parametrize("bad_kwargs,fragment", [
    ({"data": 12345, "target": "failure"}, "pandas DataFrame"),
    ({"data": "no_such_file_anywhere.csv", "target": "failure"}, "FileNotFoundError"),
])
def test_auto_analyze_reports_failures_as_a_sentence(bad_kwargs, fragment):
    report = po.auto_analyze(**bad_kwargs)
    assert report["ok"] is False
    assert fragment in report["error"]
    json.dumps(report, allow_nan=False)


def test_auto_analyze_names_the_missing_target_column(failure_frame):
    report = po.auto_analyze(failure_frame, target="not_a_column")
    assert report["ok"] is False
    assert "not_a_column" in report["error"]
    assert "temp_c" in report["error"]


def test_auto_analyze_is_exported():
    assert "auto_analyze" in po.__all__


# ---- token cost benchmark ----


def test_token_benchmark_shows_the_facade_is_cheapest():
    import importlib.util

    benchmark_path = Path(__file__).resolve().parent.parent / "benchmarks" / "token_cost.py"
    assert benchmark_path.exists(), "benchmarks/token_cost.py is missing"

    spec = importlib.util.spec_from_file_location("token_cost", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rows, method = module.measure()
    assert len(rows) == 3
    assert isinstance(method, str) and method
    baseline, step_by_step, facade = rows
    assert baseline["tokens"] > step_by_step["tokens"] > facade["tokens"]
    assert facade["saving_pct"] > 80
    assert facade["lines"] <= 5


# ---- type hints ----


def test_every_public_function_is_annotated():
    # Type hints are what let an editor - or an agent reading the source - see
    # what a call returns without running it. A public function missing them is
    # a gap in the library's contract.
    import inspect as inspect_module

    unannotated = []
    for name in po.__all__:
        member = getattr(po, name, None)
        if not inspect_module.isfunction(member):
            continue
        signature = inspect_module.signature(member)
        if signature.return_annotation is inspect_module.Signature.empty:
            unannotated.append(f"{name} (return)")
        for parameter in signature.parameters.values():
            if parameter.annotation is inspect_module.Parameter.empty:
                unannotated.append(f"{name}.{parameter.name}")
    assert not unannotated, f"missing annotations: {unannotated}"


def test_every_public_engine_method_is_annotated():
    import inspect as inspect_module

    skip = {"get_params", "set_params", "fit_transform", "score", "get_metadata_routing", "set_fit_request"}
    unannotated = []
    for name, member in inspect_module.getmembers(po.PotatOptEngine, inspect_module.isfunction):
        if name.startswith("_") or name in skip:
            continue
        signature = inspect_module.signature(member)
        if signature.return_annotation is inspect_module.Signature.empty:
            unannotated.append(f"{name} (return)")
        for parameter in signature.parameters.values():
            if parameter.name in ("self", "cls"):
                continue
            if parameter.annotation is inspect_module.Parameter.empty:
                unannotated.append(f"{name}.{parameter.name}")
    assert not unannotated, f"missing annotations: {unannotated}"


def test_annotations_stay_lazy_strings():
    # `from __future__ import annotations` keeps every annotation unevaluated, so
    # the modern `X | None` syntax costs nothing at import time on older Pythons.
    import inspect as inspect_module

    annotation = inspect_module.signature(po.auto_analyze).parameters["data"].annotation
    assert isinstance(annotation, str)
    assert "DataFrame" in annotation


# ---- Phase 2.7: minimal core and packaging ----


def test_importing_potatopt_does_not_load_the_heavy_backends():
    # The claim that makes the extras real. Run in a clean subprocess, because
    # this test session has already imported both backends.
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    code = (
        f"import sys; sys.path.insert(0, r'{project_root}'); import potatopt; "
        "print('flaml' in sys.modules, 'shap' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False False", completed.stdout + completed.stderr


def test_the_backends_are_still_reachable_as_module_attributes():
    assert po.AutoML.__name__ == "AutoML"
    assert po.shap.__name__ == "shap"


def test_unknown_module_attributes_still_raise():
    with pytest.raises(AttributeError, match="has no attribute"):
        po.definitely_not_a_real_name  # noqa: B018


def test_pyproject_declares_a_four_package_core():
    import tomllib

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    assert pyproject_path.exists(), "pyproject.toml is missing"

    with pyproject_path.open("rb") as handle:
        config = tomllib.load(handle)

    core = config["project"]["dependencies"]
    core_names = {entry.split(">")[0].split("=")[0].strip() for entry in core}
    assert core_names == {"numpy", "pandas", "scipy", "scikit-learn"}

    # The heavy backends must be extras, never core.
    joined = " ".join(core)
    assert "flaml" not in joined
    assert "shap" not in joined
    assert "lightgbm" not in joined


def test_pyproject_extras_cover_every_optional_backend():
    import tomllib

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        config = tomllib.load(handle)

    extras = config["project"]["optional-dependencies"]
    assert {"automl", "xai", "viz", "mcp", "dev", "all"} <= set(extras)

    # Every optional backend the library itself can reach has to be installable by
    # name, and `all` has to actually mean all of them - an extra that exists but
    # is missing from `all` is a dependency nobody discovers until it raises.
    assert "app" not in extras and "sim" not in extras, (
        "the web application is not part of this distribution, so its dependencies "
        "must not be installable as an extra of the library"
    )

    everything = " ".join(extras["all"])
    for package in ["flaml", "lightgbm", "xgboost", "shap", "matplotlib", "seaborn"]:
        assert package in everything, f"{package} missing from the 'all' extra"


def test_pyproject_version_tracks_the_module():
    import tomllib

    pyproject_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
    with pyproject_path.open("rb") as handle:
        config = tomllib.load(handle)

    # The version is declared dynamic so it is read from potatopt.__version__ and
    # can never drift out of step with the module.
    assert "version" in config["project"]["dynamic"]
    assert config["tool"]["setuptools"]["dynamic"]["version"]["attr"] == "potatopt.__version__"
    packages = config["tool"]["setuptools"]["packages"]
    assert "potatopt" in packages, "the library is what the distribution is for"

    # The distribution is the library alone. `packages` naming anything else means
    # an application has been folded into a `pip install` of a computation library,
    # which is a decision, not an accident - so it fails here rather than shipping.
    assert packages == ["potatopt"], (
        f"the distribution must contain only the library, found {packages}"
    )
    package_data = config["tool"]["setuptools"]["package-data"]
    assert package_data == {"potatopt": ["py.typed"]}

    # Reading that attribute must not require importing the package. setuptools
    # parses a literal assignment straight out of the AST but falls back to a real
    # import when it is anything else - and the CI build job installs only pip,
    # build and twine, so an import fallback would fail there for want of numpy.
    package_dir = Path(__file__).resolve().parent.parent / "potatopt"
    init_source = (package_dir / "__init__.py").read_text(encoding="utf-8")
    assert re.search(r'^__version__ = "[0-9]+\.[0-9]+\.[0-9]+"$', init_source, re.MULTILINE), (
        "__version__ must stay a literal string assignment in potatopt/__init__.py"
    )

    # Exactly one module may declare it. The package split briefly copied the
    # literal into engine.py as well; both read 1.4.0 so nothing disagreed, and
    # the duplicate stayed invisible until the next bump, when get_training_report()
    # and the saved model metadata carried on stamping the old number.
    declaring = sorted(
        path.name
        for path in package_dir.glob("*.py")
        if re.search(r'^__version__ = "', path.read_text(encoding="utf-8"), re.MULTILINE)
    )
    assert declaring == ["__init__.py"], f"__version__ is declared in more than one module: {declaring}"


# ---- Phase 2.8: EWMA and CUSUM control charts ----


def test_ewma_smooths_toward_a_step_by_the_exact_recursion():
    # z = 0.2 * 12 + 0.8 * 10 = 10.4 on the first sample after the step.
    chart = po.calculate_ewma_chart([10, 10, 10, 10, 10, 12, 12, 12, 12, 12],
                                    target=10.0, sigma=1.0)
    assert chart["ewma"][:5] == pytest.approx([10.0] * 5)
    assert chart["ewma"][5] == pytest.approx(10.4)


def test_ewma_limits_use_the_exact_time_varying_form():
    # ucl_1 = 10 + 3 * 1 * sqrt((0.2/1.8) * (1 - 0.8^2)) = 10.6
    chart = po.calculate_ewma_chart([10] * 10, target=10.0, sigma=1.0)
    assert chart["ucl"][0] == pytest.approx(10.6)
    assert chart["lcl"][0] == pytest.approx(9.4)
    # The band widens as the chart accumulates memory.
    assert chart["ucl"][-1] > chart["ucl"][0]


def test_ewma_says_nothing_about_a_stable_process():
    chart = po.calculate_ewma_chart([10.0] * 20, target=10.0, sigma=1.0)
    assert chart["ewma"] == pytest.approx([10.0] * 20)
    assert chart["out_of_control"] is False
    assert chart["first_violation"] is None
    assert chart["direction"] is None


def test_ewma_catches_a_gradual_upward_drift():
    # A ramp of 0.25 per sample on sigma 1 - far too small for a Shewhart chart
    # to react to any single point.
    ramp = [10.0 + 0.25 * i for i in range(30)]
    chart = po.calculate_ewma_chart(ramp, target=10.0, sigma=1.0)
    assert chart["out_of_control"] is True
    assert chart["first_violation"] == 8
    assert chart["direction"] == "increasing"


def test_ewma_reports_a_downward_drift_as_decreasing():
    ramp = [10.0 - 0.25 * i for i in range(30)]
    chart = po.calculate_ewma_chart(ramp, target=10.0, sigma=1.0)
    assert chart["direction"] == "decreasing"
    assert chart["first_violation"] == 8


def test_cusum_accumulates_by_the_exact_recursion():
    # slack = 0.5 * 1 = 0.5, so each 13 adds (13 - 10) - 0.5 = 2.5.
    chart = po.calculate_cusum_chart([10, 10, 10, 13, 13, 13], target=10.0, sigma=1.0)
    assert chart["cusum_high"] == pytest.approx([0.0, 0.0, 0.0, 2.5, 5.0, 7.5])
    assert chart["decision_interval"] == pytest.approx(5.0)
    # The signal is strict: SH == h * sigma at index 4 is not yet a violation.
    assert chart["first_violation"] == 5
    assert chart["direction"] == "increasing"


def test_cusum_ignores_a_stable_process_indefinitely():
    chart = po.calculate_cusum_chart([10.0] * 50, target=10.0, sigma=1.0)
    assert chart["cusum_high"] == pytest.approx([0.0] * 50)
    assert chart["cusum_low"] == pytest.approx([0.0] * 50)
    assert chart["out_of_control"] is False


def test_cusum_catches_the_same_gradual_drift():
    ramp = [10.0 + 0.25 * i for i in range(30)]
    chart = po.calculate_cusum_chart(ramp, target=10.0, sigma=1.0)
    assert chart["out_of_control"] is True
    assert chart["first_violation"] == 8
    assert chart["violations_low"] == []


def test_cusum_detects_a_downward_shift_on_the_low_arm():
    chart = po.calculate_cusum_chart([10] * 3 + [7] * 5, target=10.0, sigma=1.0)
    assert chart["direction"] == "decreasing"
    assert chart["violations_high"] == []
    assert chart["violations_low"]


# ---- Phase 2.8: sigma estimation ----


def test_sigma_comes_from_the_moving_range_not_the_sample_std():
    # A pure ramp of 0.25 has a constant moving range, so sigma = 0.25 / 1.128.
    ramp = [10.0 + 0.25 * i for i in range(30)]
    chart = po.calculate_ewma_chart(ramp)
    assert chart["sigma"] == pytest.approx(0.25 / po.MOVING_RANGE_D2)
    # The sample standard deviation of the same series is ~9.9x larger: the trend
    # inflates it, which would widen the limits and hide the trend that caused it.
    inflation = float(np.std(ramp, ddof=1)) / chart["sigma"]
    assert inflation > 9


def _degrading_series():
    # A settled baseline with real noise, then a steady climb - the shape of a
    # bearing starting to wear.
    rng = np.random.default_rng(5)
    baseline = list(10 + rng.normal(0, 0.3, 15))
    return baseline + [baseline[-1] + 0.4 * i for i in range(1, 21)]


def test_baseline_n_estimates_from_the_clean_period_only():
    series = _degrading_series()
    windowed = po.calculate_ewma_chart(series, baseline_n=15)
    assert windowed["baseline_n"] == 15
    # The target reflects where the process actually sat before it drifted.
    assert windowed["target"] == pytest.approx(10.0, abs=0.2)
    assert windowed["first_violation"] == 16
    assert windowed["direction"] == "increasing"


def test_estimating_on_the_whole_series_raises_a_backwards_false_alarm():
    # Guards the reason baseline_n exists. Estimating the target across a rising
    # series drags it above where the process began, so the earliest points sit
    # below the lower limit and the chart fires "decreasing" on data that only
    # ever rises. Correctness, not speed, is what baseline_n buys.
    series = _degrading_series()
    whole = po.calculate_ewma_chart(series)
    windowed = po.calculate_ewma_chart(series, baseline_n=15)

    assert whole["first_violation"] == 0
    assert whole["direction"] == "decreasing"
    assert whole["target"] > windowed["target"] + 2
    assert windowed["direction"] == "increasing"


def test_cusum_agrees_with_ewma_on_the_same_degrading_series():
    series = _degrading_series()
    chart = po.calculate_cusum_chart(series, baseline_n=15)
    assert chart["direction"] == "increasing"
    assert chart["first_violation"] == 17
    assert chart["violations_low"] == []


def test_charts_report_a_flat_series_instead_of_crying_wolf():
    for chart in (po.calculate_ewma_chart([5.0] * 12), po.calculate_cusum_chart([5.0] * 12)):
        assert chart["sigma"] == 0.0
        assert chart["out_of_control"] is False
        assert "degenerate" in chart["note"]


def test_a_flat_baseline_window_does_not_blind_the_chart():
    # Regression test: a quantised sensor can hold one value through the whole
    # baseline window. Sigma then has to come from the wider series, or a process
    # climbing 10 -> 19.5 would be reported as perfectly in control.
    series = [10.0] * 10 + [10.0 + 0.5 * i for i in range(20)]
    chart = po.calculate_ewma_chart(series, baseline_n=10)
    assert chart["sigma"] == pytest.approx(float(np.mean(np.abs(np.diff(series)))) / po.MOVING_RANGE_D2)
    assert chart["sigma"] > 0
    assert chart["out_of_control"] is True
    assert chart["direction"] == "increasing"
    # Only a series that never moves at all stays degenerate.
    assert po.calculate_ewma_chart([10.0] * 30, baseline_n=10)["sigma"] == 0.0


# ---- Phase 2.8: chart input handling ----


@pytest.mark.parametrize("bad,fragment", [
    ({"values": []}, "No numeric values"),
    ({"values": ["a", "b"]}, "No numeric values"),
    ({"values": [1, 2, 3], "lambda_weight": 0}, "lambda_weight"),
    ({"values": [1, 2, 3], "lambda_weight": 1.5}, "lambda_weight"),
    ({"values": [1, 2, 3], "n_sigmas": 0}, "n_sigmas"),
])
def test_ewma_returns_an_error_dict_instead_of_raising(bad, fragment):
    result = po.calculate_ewma_chart(**bad)
    assert fragment in result["error"]


@pytest.mark.parametrize("bad,fragment", [
    ({"values": []}, "No numeric values"),
    ({"values": [1, 2, 3], "slack_k": -1}, "slack_k"),
    ({"values": [1, 2, 3], "decision_h": 0}, "decision_h"),
])
def test_cusum_returns_an_error_dict_instead_of_raising(bad, fragment):
    result = po.calculate_cusum_chart(**bad)
    assert fragment in result["error"]


def test_charts_drop_non_numeric_entries_rather_than_failing():
    chart = po.calculate_ewma_chart([10, None, 10, "bad", 10], target=10.0, sigma=1.0)
    assert chart["n_points"] == 3


def test_chart_output_survives_strict_json():
    for chart in (po.calculate_ewma_chart([1, 2, 3, 4, 5]), po.calculate_cusum_chart([1, 2, 3, 4, 5])):
        json.dumps(chart, allow_nan=False)


def test_the_new_chart_helpers_are_exported():
    for name in ["calculate_ewma_chart", "calculate_cusum_chart", "MOVING_RANGE_D2",
                 "EWMA_DEFAULT_LAMBDA", "CUSUM_DEFAULT_DECISION"]:
        assert name in po.__all__


# ---- Phase 2.8: control-chart parameter guards ----


@pytest.mark.parametrize("kwargs", [
    {"n_sigmas": float("nan")},
    {"n_sigmas": float("inf")},
    {"lambda_weight": float("nan")},
    {"lambda_weight": float("inf")},
    {"target": float("nan")},
    {"sigma": float("inf")},
])
def test_ewma_rejects_non_finite_parameters(kwargs):
    # NaN slips past every ordinary comparison, so it has to be excluded by name.
    result = po.calculate_ewma_chart([1, 2, 3, 4, 5], **kwargs)
    assert "error" in result
    assert "ucl" not in result


@pytest.mark.parametrize("kwargs", [
    {"slack_k": float("nan")},
    {"slack_k": float("inf")},
    {"decision_h": float("nan")},
    {"decision_h": float("inf")},
    {"target": float("nan")},
    {"sigma": float("nan")},
])
def test_cusum_rejects_non_finite_parameters(kwargs):
    # The reason this matters: max(0.0, nan) returns 0.0, so a NaN slack pins both
    # CUSUM arms at zero and the chart silently never signals again.
    result = po.calculate_cusum_chart([1, 2, 3, 4, 5], **kwargs)
    assert "error" in result
    assert "cusum_high" not in result


def test_a_nan_slack_would_have_silently_disabled_the_chart():
    # Guards the guard: without the finite check this series signals, and with a
    # NaN slack it would have reported a perfectly calm machine.
    ramp = [10.0 + 0.5 * i for i in range(30)]
    assert po.calculate_cusum_chart(ramp, target=10.0, sigma=1.0)["out_of_control"] is True
    assert "error" in po.calculate_cusum_chart(ramp, target=10.0, sigma=1.0, slack_k=float("nan"))


@pytest.mark.parametrize("chart_fn,kwargs", [
    (po.calculate_ewma_chart, {"n_sigmas": "big"}),
    (po.calculate_ewma_chart, {"target": "x"}),
    (po.calculate_ewma_chart, {"lambda_weight": None}),
    (po.calculate_cusum_chart, {"slack_k": "half"}),
    (po.calculate_cusum_chart, {"sigma": "one"}),
    (po.calculate_cusum_chart, {"decision_h": None}),
])
def test_charts_never_raise_on_a_non_numeric_parameter(chart_fn, kwargs):
    # Both docstrings promise an error dict rather than an exception; these used
    # to escape as ValueError from float("big").
    result = chart_fn([1, 2, 3, 4, 5], **kwargs)
    assert "error" in result
    assert "must be a number" in result["error"]


def test_a_negative_sigma_override_is_rejected():
    assert "error" in po.calculate_ewma_chart([1, 2, 3], sigma=-1.0)
    assert "error" in po.calculate_cusum_chart([1, 2, 3], sigma=-1.0)


def test_valid_parameters_still_produce_a_chart():
    # The guards must not have narrowed what is accepted.
    chart = po.calculate_ewma_chart([10] * 5 + [12] * 5, target=10.0, sigma=1.0,
                                    lambda_weight=0.2, n_sigmas=3.0)
    assert chart["ewma"][5] == pytest.approx(10.4)
    assert chart["n_sigmas"] == 3.0
    cusum = po.calculate_cusum_chart([10, 10, 10, 13, 13, 13], target=10.0, sigma=1.0,
                                     slack_k=0.5, decision_h=5.0)
    assert cusum["cusum_high"] == pytest.approx([0.0, 0.0, 0.0, 2.5, 5.0, 7.5])
    assert cusum["slack_k"] == 0.5
    assert cusum["decision_h"] == 5.0


# ---- Phase 2.8: per-asset drift, categorical PSI, incomplete data ----


def _fleet(rng, sizes):
    """Three machines that are healthy but run at different temperatures."""
    frames = []
    for name, n, temp in sizes:
        frames.append(pd.DataFrame({
            "machine_id": [name] * n,
            "temp": rng.normal(temp, 1.0, n),
            "vibration": rng.normal(2.0, 0.2, n),
        }))
    return pd.concat(frames, ignore_index=True)


def _fleet_train():
    return _fleet(np.random.default_rng(11), [("M-01", 200, 70.0), ("M-02", 200, 75.0), ("M-03", 200, 80.0)])


def test_categorical_psi_is_zero_when_the_mix_is_unchanged():
    train = ["A"] * 300 + ["B"] * 150 + ["C"] * 50
    batch = ["A"] * 60 + ["B"] * 30 + ["C"] * 10
    assert po.calculate_categorical_psi(train, batch) == pytest.approx(0.0)


def test_categorical_psi_flags_a_changed_mix():
    train = ["A"] * 300 + ["B"] * 150 + ["C"] * 50
    batch = ["A"] * 100 + ["B"] * 100 + ["C"] * 300
    # Hand-computed: sum((b - t) * ln(b / t)) over the three category frequencies.
    assert po.calculate_categorical_psi(train, batch) == pytest.approx(1.3759, abs=1e-4)
    assert po.calculate_categorical_psi(train, batch) > po.PSI_MAJOR_SHIFT


def test_categorical_psi_treats_an_unseen_category_as_a_major_shift():
    # A part number or operator that never appeared in training is real news.
    train = ["A"] * 300 + ["B"] * 150 + ["C"] * 50
    batch = ["A"] * 200 + ["B"] * 100 + ["D"] * 200
    assert po.calculate_categorical_psi(train, batch) == pytest.approx(4.1285, abs=1e-4)


def test_categorical_psi_declines_columns_it_cannot_judge():
    assert po.calculate_categorical_psi(["A"] * 100, ["A"] * 50) is None      # one category
    assert po.calculate_categorical_psi([f"id{i}" for i in range(200)], ["id1"] * 50) is None
    assert po.calculate_categorical_psi(["A", "B"], []) is None
    assert po.calculate_categorical_psi(None, None) is None


def test_a_categorical_column_is_ignored_unless_asked_for():
    train = pd.DataFrame({"shift": ["A"] * 300 + ["B"] * 150 + ["C"] * 50})
    batch = pd.DataFrame({"shift": ["A"] * 100 + ["B"] * 100 + ["C"] * 300})
    assert po.check_data_drift(train, batch)["drift_detected"] is False
    assert po.check_data_drift(train, batch, include_categorical=True)["drift_detected"] is True


def test_a_missing_sensor_column_is_reported_rather_than_passed_over():
    # This used to return drift_detected False with max_psi 0.0 - a confident
    # all-clear for a sensor that had stopped arriving.
    rng = np.random.default_rng(2)
    train = pd.DataFrame({"temp": rng.normal(70, 1, 200), "vibration": rng.normal(2, 0.2, 200)})
    batch = train.drop(columns=["vibration"])
    report = po.check_data_drift(train, batch)
    assert report["skipped_features"]["vibration"] == "missing from the batch"
    assert "vibration" not in report["drifted_features"]


def test_min_rows_counts_readings_not_rows():
    # 200 rows holding 4 real readings raised a false alarm 100% of the time.
    rng = np.random.default_rng(4)
    train = pd.DataFrame({"temp": rng.normal(70, 1, 200)})
    batch = pd.DataFrame({"temp": [np.nan] * 196 + list(rng.normal(70, 1, 4))})
    assert len(batch) == 200
    ungated = po.check_data_drift(train, batch)
    assert ungated["drift_detected"] is True          # the old behaviour, on 4 points
    gated = po.check_data_drift(train, batch, min_rows=po.DRIFT_MIN_ROWS)
    assert gated["drift_detected"] is False
    assert "below min_rows" in gated["skipped_features"]["temp"]


def test_pooling_machines_invents_drift_that_never_happened():
    # Nothing about any machine changed. M-01 is simply down for maintenance and
    # contributes 10 rows instead of 200, which moves the pooled mean.
    train = _fleet_train()
    batch = _fleet(np.random.default_rng(11), [("M-01", 10, 70.0), ("M-02", 200, 75.0), ("M-03", 200, 80.0)])
    pooled = po.check_data_drift(train.drop(columns="machine_id"), batch.drop(columns="machine_id"))
    assert pooled["drift_detected"] is True           # the false alarm

    per_asset = po.check_asset_drift(train, batch, "machine_id")
    assert per_asset["drift_detected"] is False
    assert per_asset["assets_drifted"] == []
    assert per_asset["per_asset"]["M-01"]["status"] == "insufficient_data"
    assert per_asset["per_asset"]["M-02"]["status"] == "checked"


def test_per_asset_drift_names_the_machine_that_actually_moved():
    train = _fleet_train()
    batch = _fleet(np.random.default_rng(11), [("M-01", 200, 70.0), ("M-02", 200, 78.0), ("M-03", 200, 80.0)])
    report = po.check_asset_drift(train, batch, "machine_id")
    assert report["assets_drifted"] == ["M-02"]
    assert report["per_asset"]["M-01"]["drift_detected"] is False
    assert report["per_asset"]["M-03"]["drift_detected"] is False
    # Pooled, the same fault measures 0.244; per asset it measures 3.08.
    magnitude = report["per_asset"]["M-02"]["drifted_features"]["temp"]["drift_magnitude"]
    assert magnitude > 3.0
    pooled = po.check_data_drift(train.drop(columns="machine_id"), batch.drop(columns="machine_id"))
    assert pooled["drifted_features"]["temp"]["drift_magnitude"] < 0.3
    assert magnitude > 10 * pooled["drifted_features"]["temp"]["drift_magnitude"]


def test_a_machine_that_stops_reporting_is_not_silently_dropped():
    # A dead gateway must never read as a healthy machine.
    train = _fleet_train()
    batch = _fleet(np.random.default_rng(11), [("M-01", 200, 70.0), ("M-02", 200, 75.0)])
    report = po.check_asset_drift(train, batch, "machine_id")
    assert report["per_asset"]["M-03"]["status"] == "missing_from_batch"
    assert "M-03" in report["assets_skipped"]


def test_a_machine_never_seen_in_training_is_flagged_as_unknown():
    train = _fleet_train()
    batch = _fleet(np.random.default_rng(11), [("M-01", 200, 70.0), ("M-99", 200, 70.0)])
    report = po.check_asset_drift(train, batch, "machine_id")
    assert report["per_asset"]["M-99"]["status"] == "unknown_asset"
    assert report["per_asset"]["M-99"]["rows_batch"] == 200


def test_the_noise_floor_raises_the_threshold_for_small_batches():
    train = _fleet_train()
    batch = _fleet(np.random.default_rng(11), [("M-01", 200, 70.0), ("M-02", 200, 75.0), ("M-03", 200, 80.0)])
    report = po.check_asset_drift(train, batch, "machine_id")
    # 3 * sqrt(1/200 + 1/200) = 3 * 0.1 = 0.3, above the 0.2 asked for.
    assert report["per_asset"]["M-01"]["effective_threshold"] == pytest.approx(0.3)
    assert report["drift_detected"] is False
    # Turning the floor off falls back to the plain practical threshold.
    plain = po.check_asset_drift(train, batch, "machine_id", n_sigma_floor=0.0)
    assert plain["per_asset"]["M-01"]["effective_threshold"] == pytest.approx(0.2)


def test_psi_bins_are_scaled_to_the_batch_size():
    train = _fleet_train()
    batch = _fleet(np.random.default_rng(11), [("M-01", 40, 70.0), ("M-02", 200, 75.0), ("M-03", 200, 80.0)])
    report = po.check_asset_drift(train, batch, "machine_id")
    # 40 // 10 = 4 bins for the small group, the full 10 for the large ones.
    assert report["per_asset"]["M-01"]["psi_bins"] == 4
    assert report["per_asset"]["M-02"]["psi_bins"] == po.PSI_DEFAULT_BINS


def test_check_asset_drift_reports_problems_instead_of_raising():
    train = _fleet_train()
    assert "error" in po.check_asset_drift(train, train, "no_such_column")
    assert "error" in po.check_asset_drift(pd.DataFrame(), train, "machine_id")
    assert "error" in po.check_asset_drift(None, train, "machine_id")
    assert "error" in po.check_asset_drift(train, train, "machine_id", n_sigma_floor=float("nan"))


def test_the_per_asset_report_survives_json_dumps():
    train = _fleet_train()
    batch = _fleet(np.random.default_rng(11), [("M-01", 200, 70.0), ("M-02", 200, 78.0)])
    report = po.check_asset_drift(train, batch, "machine_id")
    restored = json.loads(json.dumps(po.to_jsonable(report)))
    assert restored["assets_drifted"] == ["M-02"]
    assert restored["per_asset"]["M-03"]["status"] == "missing_from_batch"


# ---- Phase 2.8: the maintenance business case ----


def test_maintenance_savings_matches_the_hand_calculation():
    # 20 real failures. run to failure = 20 * 50000 = 1,000,000
    # with model = 2*50000 + 18*(1500+8000) + 25*1500 = 100000 + 171000 + 37500
    result = po.calculate_maintenance_savings(true_positives=18, false_positives=25, false_negatives=2)
    assert result["run_to_failure_cost"] == pytest.approx(1_000_000.0)
    assert result["predictive_cost"] == pytest.approx(308_500.0)
    assert result["cost_savings"] == pytest.approx(691_500.0)
    assert result["savings_percentage"] == pytest.approx(69.15)
    assert result["breakdown_avoidance_rate"] == pytest.approx(0.9)
    assert result["breakdowns_avoided"] == 18
    assert result["unplanned_breakdowns"] == 2


def test_perfect_recall_can_still_lose_money():
    # The reason this function exists. Flagging all 1000 machines catches every
    # one of the 20 failures - a perfect avoidance rate - and burns 660,000,
    # because 980 pointless call-outs cost more than the breakdowns prevented.
    result = po.calculate_maintenance_savings(true_positives=20, false_positives=980, false_negatives=0)
    assert result["breakdown_avoidance_rate"] == pytest.approx(1.0)
    assert result["cost_savings"] == pytest.approx(-660_000.0)
    assert result["savings_percentage"] == pytest.approx(-66.0)


def test_break_even_is_where_the_call_outs_cancel_the_saving():
    # 20 failures caught costs 20 * 9500 = 190,000 of the 1,000,000 avoided,
    # leaving 810,000 to spend on inspections at 1,500 each: 540 of them.
    assert po.calculate_maintenance_savings(20, 540, 0)["cost_savings"] == pytest.approx(0.0)
    assert po.calculate_maintenance_savings(20, 539, 0)["cost_savings"] > 0
    assert po.calculate_maintenance_savings(20, 541, 0)["cost_savings"] < 0


def test_a_model_that_predicts_nothing_scores_exactly_the_baseline():
    result = po.calculate_maintenance_savings(true_positives=0, false_positives=0, false_negatives=20)
    assert result["cost_savings"] == pytest.approx(0.0)
    assert result["savings_percentage"] == pytest.approx(0.0)
    assert result["breakdown_avoidance_rate"] == pytest.approx(0.0)


def test_maintenance_savings_handles_a_fleet_with_no_failures():
    result = po.calculate_maintenance_savings(0, 5, 0)
    assert result["run_to_failure_cost"] == pytest.approx(0.0)
    assert result["savings_percentage"] == 0.0
    assert result["breakdown_avoidance_rate"] is None


def test_the_engineers_own_cost_numbers_are_respected():
    result = po.calculate_maintenance_savings(10, 0, 0, cost_breakdown=1000.0, cost_planned=100.0, cost_inspection=50.0)
    assert result["run_to_failure_cost"] == pytest.approx(10_000.0)
    assert result["predictive_cost"] == pytest.approx(1_500.0)
    assert result["cost_assumptions"]["cost_breakdown"] == 1000.0


@pytest.mark.parametrize("kwargs", [
    {"cost_breakdown": -1.0},
    {"cost_breakdown": float("nan")},
    {"cost_planned": float("inf")},
    {"cost_inspection": "cheap"},
])
def test_maintenance_savings_reports_bad_costs_instead_of_raising(kwargs):
    result = po.calculate_maintenance_savings(5, 5, 5, **kwargs)
    assert "error" in result
    assert "cost_savings" not in result


def test_maintenance_savings_rejects_negative_counts():
    assert "error" in po.calculate_maintenance_savings(-1, 0, 0)
    assert "error" in po.calculate_maintenance_savings(1, 0, "two")


def test_maintenance_cost_needs_a_fitted_binary_classifier():
    assert "error" in po.PotatOptEngine().calculate_maintenance_cost(pd.DataFrame({"a": [1]}), pd.Series([1]))


def test_maintenance_cost_and_threshold_work_on_a_fitted_engine(failure_frame):
    train, test, y_train, y_test = po.split_data(failure_frame.drop(columns=["machine_id"]), "failure")
    engine = po.PotatOptEngine(task="classification", time_budget=3).fit(train, y_train)

    report = engine.calculate_maintenance_cost(test, y_test)
    assert "error" not in report
    assert report["cost_savings"] == pytest.approx(report["run_to_failure_cost"] - report["predictive_cost"])
    # Every real failure in the test set is either caught or missed, never lost.
    assert report["breakdowns_avoided"] + report["unplanned_breakdowns"] == int((y_test == 1).sum())

    threshold = engine.optimize_maintenance_threshold(train, y_train)
    assert 0.0 < threshold < 1.0
    # The fingerprint must still be recorded, or evaluate() loses its ability to
    # warn that the threshold was tuned on the rows being reported.
    assert engine.threshold_tuning_fingerprint is not None


# ---- Phase 2.9: the packaging metadata must not drift from CI ----


REPO_ROOT = Path(__file__).resolve().parent.parent


def _declared_python_versions():
    """Versions claimed by the Programming Language classifiers in pyproject.toml."""
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    return sorted(set(re.findall(r"Programming Language :: Python :: (\d+\.\d+)", text)))


def _ci_matrix_python_versions():
    """Versions the CI matrix actually installs and runs the suite on."""
    text = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    matrix_line = re.search(r"^\s*python-version:\s*\[(.+)\]\s*$", text, re.MULTILINE)
    assert matrix_line, "could not find the python-version matrix in ci.yml"
    return sorted(set(re.findall(r"\d+\.\d+", matrix_line.group(1))))


def test_every_supported_python_is_actually_tested():
    # A support claim nobody runs is a guess. This keeps the two lists honest:
    # 3.9 was declared for months after its October 2025 end of life and was
    # never once executed.
    assert _declared_python_versions() == _ci_matrix_python_versions()


def test_requires_python_matches_the_oldest_tested_version():
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    floor = re.search(r'requires-python\s*=\s*">=(\d+\.\d+)"', text)
    assert floor, "requires-python is missing or not a simple >= constraint"
    assert floor.group(1) == _ci_matrix_python_versions()[0]


def test_the_core_install_check_is_wired_into_ci():
    # scripts/verify_core_install.py is the only place the four-package claim is
    # checked, because by the time tests/ runs the extras are already installed.
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "scripts/verify_core_install.py" in workflow
    assert (REPO_ROOT / "scripts" / "verify_core_install.py").is_file()


def test_lag1_autocorrelation_detects_random_walk():
    rng = np.random.default_rng(0)
    random_walk = np.cumsum(rng.normal(0, 1, 300))
    noise = rng.normal(0, 1, 300)
    
    walk_lag1 = po._lag1_autocorrelation(random_walk)
    noise_lag1 = po._lag1_autocorrelation(noise)
    
    assert walk_lag1 is not None and walk_lag1 > 0.9
    assert noise_lag1 is not None and abs(noise_lag1) < 0.3


def test_lag1_autocorrelation_returns_none_when_undecidable():
    assert po._lag1_autocorrelation(np.array([5.0] * 100)) is None
    assert po._lag1_autocorrelation(np.array([1.0, 2.0])) is None


def test_ewma_warns_on_autocorrelated_series():
    rng = np.random.default_rng(0)
    random_walk = np.cumsum(rng.normal(0, 1, 300))
    result = po.calculate_ewma_chart(random_walk)
    
    assert "error" not in result
    assert result.get("lag1_autocorrelation") is not None and result["lag1_autocorrelation"] > 0.9
    assert isinstance(result.get("autocorrelation_warning"), str)
    assert "over-alarm" in result["autocorrelation_warning"]


def test_ewma_no_autocorrelation_warning_on_independent_series():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, 300)
    result = po.calculate_ewma_chart(noise)
    
    assert "error" not in result
    assert isinstance(result.get("lag1_autocorrelation"), float)
    assert result.get("autocorrelation_warning") is None


def test_explicit_sigma_suppresses_autocorrelation_warning():
    rng = np.random.default_rng(0)
    random_walk = np.cumsum(rng.normal(0, 1, 300))
    
    ewma_result = po.calculate_ewma_chart(random_walk, sigma=1.0)
    assert "error" not in ewma_result
    assert ewma_result.get("autocorrelation_warning") is None
    
    cusum_result = po.calculate_cusum_chart(random_walk, sigma=1.0)
    assert "error" not in cusum_result
    assert cusum_result.get("autocorrelation_warning") is None


def test_cusum_reports_autocorrelation():
    rng = np.random.default_rng(0)
    random_walk = np.cumsum(rng.normal(0, 1, 300))
    result = po.calculate_cusum_chart(random_walk)
    
    assert "error" not in result
    assert result.get("lag1_autocorrelation") is not None and result["lag1_autocorrelation"] > 0.9
    assert result.get("autocorrelation_warning") is not None
    assert isinstance(result.get("autocorrelation_warning"), str)


def test_auto_analyze_reports_reason_when_shap_unavailable(monkeypatch, failure_frame):
    def mock_explain(*args, **kwargs):
        return {
            "available": False, 
            "reason": "SHAP failed: synthetic test reason",
            "n_rows_explained": 0, 
            "top_k": None, 
            "feature_attributions": []
        }
    monkeypatch.setattr(po.PotatOptEngine, "explain_predictions", mock_explain)
    
    report = po.auto_analyze(failure_frame, target="failure", time_budget=1)
    
    assert report["ok"] is True
    assert report["top_features"] == []
    assert report["top_features_note"] == "SHAP failed: synthetic test reason"


def test_wear_ramp_with_baseline_does_not_warn_about_autocorrelation():
    # This is the regression test for a warning that used to fire on the chart's own success case.
    rng = np.random.default_rng(7)
    baseline = 10.0 + rng.normal(0, 0.2, 15)
    ramp = 10.0 + 0.5 * np.arange(1, 26) + rng.normal(0, 0.2, 25)
    series = np.concatenate([baseline, ramp])
    
    result = po.calculate_ewma_chart(series, baseline_n=15)
    
    assert result["first_violation"] == 15
    assert result["autocorrelation_warning"] is None
    assert abs(result["lag1_autocorrelation"]) < 0.5


def test_autocorrelation_warning_is_hedged_without_a_baseline():
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 1, 300))
    
    result_without = po.calculate_ewma_chart(series)
    assert "whole series" in result_without["autocorrelation_warning"]
    assert "known-good period" in result_without["autocorrelation_warning"]
    
    result_with = po.calculate_ewma_chart(series, baseline_n=100)
    assert "baseline window" in result_with["autocorrelation_warning"]
    assert "over-alarm" in result_with["autocorrelation_warning"]


def test_shap_explains_a_pandas_categorical_feature():
    rng = np.random.default_rng(11)
    line = pd.Categorical(rng.choice(["A", "B", "C"], 400))
    temp_c = rng.normal(20, 5, 400)
    vibration = rng.normal(10, 2, 400)
    
    df = pd.DataFrame({"line": line, "temp_c": temp_c, "vibration": vibration})
    median_temp = df["temp_c"].median()
    df["failure"] = ((df["line"] == "C") & (df["temp_c"] > median_temp)).astype(int)
    
    X = df.drop(columns="failure")
    y = df["failure"]
    
    engine = po.PotatOptEngine(
        task="classification",
        time_budget=5,
        estimators=["lgbm"],
        cost_sensitive_weighting=True
    )
    engine.fit(X, y)
    
    result = engine.explain_predictions(X, top_k=None)
    
    assert result["available"] is True, result.get("reason", "unknown error")
    
    features = [f["feature"] for f in result["feature_attributions"]]
    assert "line" in features
    assert "temp_c" in features
    assert "vibration" in features
    assert result["additivity_check_relaxed"] is False


def test_fit_accepts_a_pandas_categorical_feature():
    rng = np.random.default_rng(5)
    line = pd.Categorical(rng.choice(["A", "B", "C"], 200))
    temp_c = rng.normal(20, 5, 200)
    failure = ((line == "C") & (temp_c > 20)).astype(int)
    
    df = pd.DataFrame({"line": line, "temp_c": temp_c})
    y = pd.Series(failure)
    
    engine = po.PotatOptEngine(task="classification", time_budget=5)
    engine.fit(df, y)
    
    assert engine.is_fitted
    assert len(engine.predict(df)) == 200
    assert "line" in engine.categorical_cols
    assert "line" not in engine.numeric_cols


def test_inspect_data_accepts_a_categorical_target():
    rng = np.random.default_rng(5)
    line = pd.Categorical(rng.choice(["A", "B", "C"], 200))
    temp_c = rng.normal(20, 5, 200)
    failure = pd.Series(((line == "C") & (temp_c > 20)).astype(int), dtype="category")
    
    df = pd.DataFrame({"line": line, "temp_c": temp_c, "failure": failure})
    
    result = po.inspect_data(df, "failure")
    assert result["recommended_task"] == "classification"


def test_auto_analyze_survives_a_categorical_frame():
    rng = np.random.default_rng(5)
    line = pd.Categorical(rng.choice(["A", "B", "C"], 200))
    temp_c = rng.normal(20, 5, 200)
    failure = pd.Series(((line == "C") & (temp_c > 20)).astype(int), dtype="category")
    
    df = pd.DataFrame({"line": line, "temp_c": temp_c, "failure": failure})
    
    report = po.auto_analyze(df, target="failure", time_budget=5)
    assert report["ok"] is True
    assert report["error"] is None


# ---- Phase 3.0: a library should be quiet unless asked to speak ----


def _fit_in_subprocess(verbose: int) -> str:
    """Fit one small engine in a clean interpreter and return everything it wrote."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    code = (
        f"import sys; sys.path.insert(0, r'{project_root}');"
        "import numpy as np, pandas as pd, potatopt as po;"
        "rng = np.random.default_rng(4);"
        "X = pd.DataFrame({'a': rng.normal(0, 1, 200), 'b': rng.normal(0, 1, 200)});"
        "y = (X['a'] + rng.normal(0, 0.3, 200) > 0).astype(int);"
        f"po.PotatOptEngine(task='classification', time_budget=3, verbose={verbose}).fit(X, y)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    return done.stdout + done.stderr


def test_fit_is_quiet_by_default():
    # FLAML raises its own logger's level inside fit(), so a caller cannot silence
    # it from outside - only the verbose argument can. Measured before this was
    # wired up: a 3-second budget wrote 81 log lines and 9.5 KB into the caller's
    # output. A library writing that much uninvited is a defect, not a feature.
    assert "flaml.automl.logger" not in _fit_in_subprocess(0)


def test_verbose_engine_reports_its_search():
    # The other half of the contract: quiet by default must not mean unreachable.
    output = _fit_in_subprocess(3)
    if "flaml.automl.logger" not in output:
        pytest.skip("This FLAML build does not log its search where the test can see it.")
    assert "iteration" in output


# ---- Reliability and OEE metrics ----
#
# Every expectation below was computed by hand before the implementation existed,
# so a disagreement means the code is wrong, not that the number needs adjusting.


def _wo(reported, started, finished, wo_type="breakdown", mode="Bearing"):
    base = pd.Timestamp("2026-03-01 08:00")
    return {
        "machine_id": "M-01",
        "wo_type": wo_type,
        "failure_mode": mode,
        "reported_at": base + pd.Timedelta(hours=reported),
        "started_at": base + pd.Timedelta(hours=started),
        "finished_at": base + pd.Timedelta(hours=finished),
    }


@pytest.fixture
def work_order_frame():
    # wait / repair / down, in hours:
    #   0.50 / 2.0 / 2.50
    #   0.25 / 1.0 / 1.25
    #   1.00 / 3.0 / 4.00
    #   0.25 / 0.5 / 0.75
    return pd.DataFrame([
        _wo(0.0, 0.50, 2.50),
        _wo(24.0, 24.25, 25.25),
        _wo(48.0, 49.0, 52.0),
        _wo(72.0, 72.25, 72.75),
    ])


def test_mtbf_uses_operating_hours(work_order_frame):
    result = po.calculate_mtbf(work_order_frame, operating_hours=1000.0)
    assert result["mtbf_hours"] == pytest.approx(250.0)
    assert result["breakdowns"] == 4
    assert result["failure_rate_per_hour"] == pytest.approx(0.004)


def test_mtbf_counts_only_breakdowns(work_order_frame):
    # Planned and predictive work are maintenance events, not failures. Counting
    # them would penalise exactly the behaviour condition monitoring produces.
    extra = pd.DataFrame([
        _wo(96.0, 96.5, 98.0, wo_type="planned"),
        _wo(120.0, 120.5, 121.0, wo_type="inspection"),
        _wo(144.0, 144.5, 145.0, wo_type="predictive"),
    ])
    frame = pd.concat([work_order_frame, extra], ignore_index=True)
    assert po.calculate_mtbf(frame, operating_hours=1000.0)["breakdowns"] == 4


def test_mtbf_reports_a_machine_that_never_failed(work_order_frame):
    quiet = work_order_frame.assign(wo_type="planned")
    result = po.calculate_mtbf(quiet, operating_hours=1000.0)
    assert result["mtbf_hours"] is None
    assert result["failure_rate_per_hour"] is None
    assert result["breakdowns"] == 0
    assert "error" not in result


@pytest.mark.parametrize("hours", [0, -5, float("nan"), float("inf"), "many"])
def test_mtbf_rejects_unusable_operating_hours(work_order_frame, hours):
    assert "error" in po.calculate_mtbf(work_order_frame, operating_hours=hours)


def test_mttr_separates_waiting_from_repairing(work_order_frame):
    result = po.calculate_mttr(work_order_frame)
    assert result["mttr_hours"] == pytest.approx(1.625)
    assert result["mtta_hours"] == pytest.approx(0.5)
    assert result["mdt_hours"] == pytest.approx(2.125)
    assert result["repairs"] == 4
    assert result["rows_excluded"] == 0
    assert result["longest_repair_hours"] == pytest.approx(3.0)
    assert result["longest_wait_hours"] == pytest.approx(1.0)


def test_downtime_is_the_sum_of_waiting_and_repairing(work_order_frame):
    # Averaging each duration over a different set of rows would silently break
    # this identity and produce a self-inconsistent report.
    result = po.calculate_mttr(work_order_frame)
    assert result["mdt_hours"] == pytest.approx(result["mtta_hours"] + result["mttr_hours"])


def test_mttr_excludes_a_repair_that_finished_before_it_started(work_order_frame):
    broken = pd.concat([work_order_frame, pd.DataFrame([_wo(168.0, 170.0, 169.0)])], ignore_index=True)
    result = po.calculate_mttr(broken)
    assert result["rows_excluded"] == 1
    assert result["repairs"] == 4
    assert result["mttr_hours"] == pytest.approx(1.625)


def test_mttr_excludes_a_row_with_a_missing_timestamp(work_order_frame):
    holed = work_order_frame.copy()
    holed.loc[0, "finished_at"] = pd.NaT
    result = po.calculate_mttr(holed)
    assert result["rows_excluded"] == 1
    assert result["repairs"] == 3


def test_availability_separates_the_machine_from_the_organisation():
    result = po.calculate_availability(mtbf_hours=250.0, mttr_hours=1.625, mdt_hours=2.125)
    assert result["inherent_availability"] == pytest.approx(0.993540, abs=1e-5)
    assert result["operational_availability"] == pytest.approx(0.991572, abs=1e-5)
    assert result["availability_lost_to_waiting"] == pytest.approx(0.001968, abs=1e-5)
    # Waiting can only lose availability, never add it.
    assert result["operational_availability"] < result["inherent_availability"]


def test_availability_without_downtime_reports_only_the_inherent_figure():
    result = po.calculate_availability(mtbf_hours=250.0, mttr_hours=1.625)
    assert result["inherent_availability"] == pytest.approx(0.993540, abs=1e-5)
    assert result["operational_availability"] is None
    assert result["availability_lost_to_waiting"] is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mtbf_hours": 0.0, "mttr_hours": 0.0},
        {"mtbf_hours": -1.0, "mttr_hours": 2.0},
        {"mtbf_hours": 250.0, "mttr_hours": float("nan")},
        {"mtbf_hours": "long", "mttr_hours": 2.0},
    ],
)
def test_availability_returns_an_error_for_impossible_input(kwargs):
    assert "error" in po.calculate_availability(**kwargs)


def test_oee_matches_the_hand_computed_shift():
    result = po.calculate_oee(
        planned_time_min=480.0, run_time_min=420.0,
        ideal_cycle_time_min=0.5, total_count=700, good_count=660,
    )
    assert result["availability"] == pytest.approx(0.875)
    assert result["performance"] == pytest.approx(0.833333, abs=1e-6)
    assert result["quality"] == pytest.approx(0.942857, abs=1e-6)
    assert result["oee"] == pytest.approx(0.6875)
    assert result["oee_pct"] == pytest.approx(68.75)
    assert result["meets_world_class"] is False
    assert result["warnings"] == []


def test_oee_rejects_more_good_parts_than_parts():
    result = po.calculate_oee(480.0, 420.0, 0.5, total_count=700, good_count=800)
    assert "error" in result


def test_oee_warns_instead_of_hiding_an_overrun():
    result = po.calculate_oee(480.0, 500.0, 0.5, total_count=700, good_count=700)
    assert "error" not in result
    assert result["availability"] > 1.0
    assert any("planned" in w for w in result["warnings"])


def test_oee_warns_instead_of_clamping_impossible_performance():
    # Beating the ideal cycle time means the master data is wrong. Clamping to
    # 1.0 would hide the defect that the number exists to reveal.
    result = po.calculate_oee(480.0, 420.0, 2.0, total_count=700, good_count=700)
    assert result["performance"] > 1.0
    assert any("ideal cycle time" in w for w in result["warnings"])


@pytest.fixture
def failure_mode_frame():
    rows = []
    for mode, events, hours in [
        ("Bearing", 10, 40.0), ("Sensor", 25, 12.5),
        ("Belt", 5, 30.0), ("Motor", 2, 20.0), ("Other", 8, 4.0),
    ]:
        rows.extend({"failure_mode": mode, "downtime_hours": hours / events} for _ in range(events))
    return pd.DataFrame(rows)


def test_pareto_by_count_ranks_the_frequent_causes(failure_mode_frame):
    result = po.calculate_pareto(failure_mode_frame, "failure_mode")
    assert result["total"] == pytest.approx(50.0)
    assert result["measured_by"] == "count"
    assert result["vital_few"] == ["Sensor", "Bearing", "Other"]
    assert result["categories"][0]["percentage"] == pytest.approx(50.0)
    assert result["categories"][1]["cumulative_percentage"] == pytest.approx(70.0)
    assert result["categories"][2]["cumulative_percentage"] == pytest.approx(86.0)


def test_pareto_by_downtime_ranks_the_expensive_causes(failure_mode_frame):
    result = po.calculate_pareto(failure_mode_frame, "failure_mode", value_col="downtime_hours")
    assert result["total"] == pytest.approx(106.5)
    assert result["measured_by"] == "downtime_hours"
    assert result["vital_few"] == ["Bearing", "Belt", "Motor"]
    assert result["categories"][0]["percentage"] == pytest.approx(37.559, abs=1e-3)
    assert result["categories"][2]["cumulative_percentage"] == pytest.approx(84.507, abs=1e-3)


def test_counting_events_and_counting_hours_disagree(failure_mode_frame):
    # This is the whole reason value_col exists: the frequent cause and the
    # expensive cause are different, and only the second one is worth the work.
    by_count = po.calculate_pareto(failure_mode_frame, "failure_mode")["vital_few"]
    by_time = po.calculate_pareto(failure_mode_frame, "failure_mode", value_col="downtime_hours")["vital_few"]
    assert by_count != by_time
    assert by_count[0] == "Sensor" and by_time[0] == "Bearing"


def test_pareto_keeps_an_unlabelled_cause_visible():
    frame = pd.DataFrame({"failure_mode": ["Bearing", "Bearing", None, "Belt"]})
    result = po.calculate_pareto(frame, "failure_mode")
    assert result["total"] == pytest.approx(4.0)
    assert "(unknown)" in [row["category"] for row in result["categories"]]


def test_pareto_rejects_negative_values(failure_mode_frame):
    frame = failure_mode_frame.copy()
    frame.loc[0, "downtime_hours"] = -3.0
    assert "error" in po.calculate_pareto(frame, "failure_mode", value_col="downtime_hours")


@pytest.mark.parametrize("cutoff", [0, 1.5, -0.2, float("nan"), "most"])
def test_pareto_rejects_an_impossible_cutoff(failure_mode_frame, cutoff):
    assert "error" in po.calculate_pareto(failure_mode_frame, "failure_mode", cutoff=cutoff)


@pytest.mark.parametrize("junk", [None, 42, "rows", pd.DataFrame()])
def test_reliability_metrics_reject_a_frame_they_cannot_use(junk):
    assert "error" in po.calculate_mtbf(junk, operating_hours=100.0)
    assert "error" in po.calculate_mttr(junk)
    assert "error" in po.calculate_pareto(junk, "failure_mode")


def test_a_missing_column_names_the_columns_that_are_there(work_order_frame):
    result = po.calculate_mtbf(work_order_frame, operating_hours=100.0, wo_type_col="kind")
    assert "error" in result and "wo_type" in result["error"]


def test_every_reliability_metric_survives_json(work_order_frame, failure_mode_frame):
    results = [
        po.calculate_mtbf(work_order_frame, operating_hours=1000.0),
        po.calculate_mttr(work_order_frame),
        po.calculate_availability(250.0, 1.625, 2.125),
        po.calculate_oee(480.0, 420.0, 0.5, 700, 660),
        po.calculate_pareto(failure_mode_frame, "failure_mode", value_col="downtime_hours"),
    ]
    for result in results:
        assert json.loads(json.dumps(po.to_jsonable(result))) is not None


# ---- v1.4.0: scoped warning suppression ----


def test_importing_potatopt_leaves_warning_filters_intact():
    # Importing or reloading potatopt must not mutate process-wide warning filters.
    # Previously warnings.filterwarnings('ignore') ran at module level and silenced
    # caller warnings for the remainder of the Python process.
    import importlib
    import warnings as warnings_module

    before = list(warnings_module.filters)
    importlib.reload(po)
    assert warnings_module.filters == before

    with pytest.warns(UserWarning, match="caller_test_warning"):
        warnings_module.warn("caller_test_warning", UserWarning, stacklevel=1)


def test_quiet_dependency_warnings_restores_filters_on_exit_and_exception():
    # Context manager must restore the exact filter table on exit, even when an
    # unexpected exception bubbles out of third-party dependency calls.
    import warnings as warnings_module

    before = list(warnings_module.filters)
    with po._quiet_dependency_warnings():
        pass
    assert warnings_module.filters == before

    with pytest.raises(RuntimeError, match="forced_error"), po._quiet_dependency_warnings():
        raise RuntimeError("forced_error")
    assert warnings_module.filters == before


# ---- v1.4.0: docstring integrity vs authenticity guards ----


def test_load_docstring_claims_integrity_not_authenticity():
    # Model serialization with SHA-256 provides integrity verification (tamper/corruption detection)
    # but does not prove authenticity (who created the file), because pickle executes code.
    doc = (po.PotatOptEngine.load.__doc__ or "").lower()
    assert "cryptographic signature" not in doc
    assert "integrity" in doc
    assert "authenticity" in doc


def test_save_docstring_does_not_claim_hash_is_signature():
    # save() documentation must explicitly state the hash is an integrity record, not a signature.
    doc = (po.PotatOptEngine.save.__doc__ or "").lower()
    assert "cryptographic signature" not in doc
    assert "not a signature" in doc


# ---- v1.4.0: random_state parameterization ----


def test_split_data_random_state_affects_test_indices(split_frame):
    # A fixed seed must produce identical test indices across repeated splits,
    # whereas changing the seed must produce different partitions.
    _, x_test1_a, _, _ = po.split_data(split_frame, "defect", random_state=42)
    _, x_test1_b, _, _ = po.split_data(split_frame, "defect", random_state=42)
    assert list(x_test1_a.index) == list(x_test1_b.index)

    _, x_test2, _, _ = po.split_data(split_frame, "defect", random_state=999)
    assert list(x_test1_a.index) != list(x_test2.index)


@pytest.mark.parametrize("bad_seed", [True, False, "42", 3.14, None])
def test_split_data_rejects_non_integer_random_state(split_frame, bad_seed):
    # bool is an int subclass in Python and strings like '42' might parse loosely;
    # both must be rejected with ValueError to keep seed configuration strict.
    with pytest.raises(ValueError):
        po.split_data(split_frame, "defect", random_state=bad_seed)


def test_split_data_forecasting_ignores_random_state():
    # Time-series forecasting must never shuffle data, so random_state has no effect.
    frame = pd.DataFrame({"t": np.arange(100.0), "y": np.arange(100.0) * 2})
    x_tr1, x_te1, _, _ = po.split_data(frame, "y", task="forecasting", random_state=1)
    x_tr2, x_te2, _, _ = po.split_data(frame, "y", task="forecasting", random_state=999)
    assert list(x_tr1.index) == list(x_tr2.index)
    assert list(x_te1.index) == list(x_te2.index)


def test_split_data_three_way_is_reproducible(split_frame):
    # The same seed passed to split_data_three_way must yield identical partitions
    # across all six returned slices.
    r1 = po.split_data_three_way(split_frame, "defect", random_state=123)
    r2 = po.split_data_three_way(split_frame, "defect", random_state=123)
    for p1, p2 in zip(r1, r2):
        if hasattr(p1, "to_numpy"):
            np.testing.assert_array_equal(p1.to_numpy(), p2.to_numpy())
        assert list(p1.index) == list(p2.index)


def test_potatopt_engine_random_state_handling():
    # PotatOptEngine must record random_state, default to DEFAULT_RANDOM_STATE,
    # and reject boolean inputs with ValueError.
    engine_custom = po.PotatOptEngine(random_state=7)
    assert engine_custom.random_state == 7

    engine_default = po.PotatOptEngine()
    assert engine_default.random_state == po.DEFAULT_RANDOM_STATE

    with pytest.raises(ValueError):
        po.PotatOptEngine(random_state=True)


def test_potatopt_engine_get_params_includes_random_state():
    # BaseEstimator introspection requires all constructor args to be exposed in get_params()
    # so sklearn clone, Pipeline, and GridSearchCV work without error.
    engine = po.PotatOptEngine(random_state=42)
    params = engine.get_params()
    assert "random_state" in params
    assert params["random_state"] == 42


def test_training_report_includes_random_state(signal_frame):
    # A training report must include random_state for experiment auditability.
    x, y = signal_frame
    engine = po.PotatOptEngine(task="classification", time_budget=5, random_state=17).fit(x, y)
    report = engine.get_training_report()
    assert report["random_state"] == 17


# ---- v1.4.0: check_calibration (module function) ----


def test_check_calibration_perfectly_calibrated_synthetic():
    # Synthetic dataset drawn uniformly with y ~ Bernoulli(p) should have low ECE (< 0.05).
    rng = np.random.default_rng(42)
    n = 6000
    p = rng.uniform(0.0, 1.0, n)
    y = (rng.uniform(0.0, 1.0, n) < p).astype(int)
    result = po.check_calibration(y, p)
    assert result["is_well_calibrated"] is True
    assert result["expected_calibration_error"] < 0.05


def test_check_calibration_overconfident_sample():
    # Pushing probabilities toward extremes without true distinction causes high ECE
    # and triggers the over-confident interpretation warning.
    rng = np.random.default_rng(42)
    n = 2000
    y = rng.choice([0, 1], size=n, p=[0.7, 0.3])
    p = np.where(rng.uniform(0.0, 1.0, n) < 0.5, 0.01, 0.99)
    result = po.check_calibration(y, p)
    assert result["is_well_calibrated"] is False
    assert result["expected_calibration_error"] > 0.05
    assert "over-confident" in result["interpretation"]


def test_check_calibration_perfect_predictor_brier_score():
    # Exact predictions matching labels must yield a Brier score of exactly 0.0.
    y = np.array([0, 1, 1, 0, 1, 0])
    p = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    result = po.check_calibration(y, p)
    assert result["brier_score"] == pytest.approx(0.0)


def test_check_calibration_base_rate_brier_skill_score():
    # Predicting constant base rate provides zero skill relative to the reference climatology.
    y = np.array([0, 0, 1, 1])
    p = np.array([0.5, 0.5, 0.5, 0.5])
    result = po.check_calibration(y, p)
    assert result["brier_skill_score"] == pytest.approx(0.0, abs=1e-6)


def test_check_calibration_bin_bookkeeping_and_boundary():
    # All rows must be accounted for across bins, and p=1.0 must be placed into the top bin.
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    p = np.array([0.0, 0.15, 0.3, 0.5, 0.7, 0.85, 0.99, 1.0])
    result = po.check_calibration(y, p, n_bins=5)
    assert sum(b["count"] for b in result["bins"]) == result["n_rows"]
    assert result["n_rows"] == len(y)
    assert result["bins"][-1]["count"] >= 1


def test_check_calibration_accepts_non_numeric_labels():
    # String labels like 'OK' and 'NG' are sorted, mapping the sorted-last label to positive.
    y = ["OK", "NG", "OK", "NG", "OK", "NG"]
    p = [0.2, 0.8, 0.1, 0.9, 0.3, 0.7]
    result = po.check_calibration(y, p)
    assert "error" not in result
    assert result["positive_label"] == "OK"


@pytest.mark.parametrize("y_true,y_prob,n_bins,msg_frag", [
    ([0, 1], [0.5], 10, "same length"),
    ([], [], 10, "empty"),
    ([1, 1, 1], [0.5, 0.5, 0.5], 10, "two outcome classes"),
    ([0, 1], [0.2, 0.8], 1, "at least 2"),
    ([0, 1], [0.2, 0.8], "ten", "whole number"),
    ([0, 1], [0.2, 1.5], 10, "[0, 1]"),
])
def test_check_calibration_error_paths_do_not_raise(y_true, y_prob, n_bins, msg_frag):
    # Public API function must return error dictionaries rather than raising unhandled exceptions.
    result = po.check_calibration(y_true, y_prob, n_bins=n_bins)
    assert "error" in result
    assert msg_frag in result["error"]


def test_check_calibration_survives_json_dumps():
    # Calibration summary dictionaries must be strict JSON serializable for web API / agent consumers.
    y = np.array([0, 1, 0, 1, 1, 0])
    p = np.array([0.1, 0.9, 0.2, 0.8, 0.85, 0.15])
    result = po.check_calibration(y, p)
    serialized = json.dumps(po.to_jsonable(result), allow_nan=False)
    assert json.loads(serialized)["is_well_calibrated"] is not None


# ---- v1.4.0: PotatOptEngine.check_calibration ----


def test_engine_check_calibration_requires_fitted_engine():
    # Calling check_calibration on an unfitted engine must return an error dict, not raise.
    engine = po.PotatOptEngine()
    x = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y = pd.Series([0, 1])
    result = engine.check_calibration(x, y)
    assert "error" in result
    assert "not fitted" in result["error"]


def test_engine_check_calibration_rejects_regression_task(regression_engine, regression_frame):
    # Calibration is defined for classification probabilities, so regression tasks return an error dict.
    x, y = regression_frame
    result = regression_engine.check_calibration(x, y)
    assert "error" in result
    assert "regression" in result["error"]


def test_engine_check_calibration_on_binary_classifier(binary_engine, signal_frame):
    # Standard classification models report model_predict_proba as probability source.
    x, y = signal_frame
    result = binary_engine.check_calibration(x, y)
    assert "error" not in result
    assert "positive_class" in result
    assert result["probability_source"] == "model_predict_proba"
    assert "brier_score" in result


def test_engine_check_calibration_on_anomaly_fallback(anomaly_engine):
    # Anomaly fallback uses a sigmoid squashed over decision function, which is not calibrated.
    # It must be marked as not well calibrated with an explanation referring to ranking.
    rng = np.random.default_rng(5)
    n_rows = 120
    x = pd.DataFrame({
        "s1": rng.normal(0, 1, n_rows),
        "s2": rng.normal(0, 1, n_rows),
        "s3": rng.normal(0, 1, n_rows),
    })
    y = pd.Series([0] * 117 + [1] * 3)
    result = anomaly_engine.check_calibration(x, y)
    assert "error" not in result
    assert result["probability_source"] == "isolation_forest_sigmoid"
    assert result["is_well_calibrated"] is False
    assert "rank" in result["interpretation"].lower()


# ---- v1.4.0: run_seed_sweep ----


def test_run_seed_sweep_rejects_random_state_argument(failure_frame):
    # Passing random_state to run_seed_sweep would override per-seed variation, so it is rejected.
    result = po.run_seed_sweep(failure_frame, target="failure", random_state=1)
    assert "error" in result
    assert "seeds" in result["error"]


@pytest.mark.parametrize("bad_seeds,frag", [
    ([], "empty"),
    ([1, 1], "duplicates"),
    ([1, "two"], "whole numbers"),
])
def test_run_seed_sweep_rejects_invalid_seeds(failure_frame, bad_seeds, frag):
    # Seeds must be non-empty sequence of unique integers.
    result = po.run_seed_sweep(failure_frame, target="failure", seeds=bad_seeds)
    assert "error" in result
    assert frag in result["error"]


def test_run_seed_sweep_computes_spread_over_seeds(failure_frame):
    # Running multiple seeds reports statistics and spread across runs.
    small = failure_frame.head(60).copy()
    result = po.run_seed_sweep(small, target="failure", seeds=[10, 20], time_budget=1)
    assert result["n_ok"] == 2
    assert len(result["runs"]) == 2
    assert len(result["summary"]) > 0
    for stats in result["summary"].values():
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "spread" in stats
        assert stats["n"] == 2
        assert stats["spread"] >= 0.0
    serialized = json.dumps(result, allow_nan=False)
    assert json.loads(serialized)["n_ok"] == 2


def test_run_seed_sweep_handles_all_runs_failing(failure_frame):
    # When all runs fail, run_seed_sweep returns n_ok=0, empty summary, and a descriptive note.
    result = po.run_seed_sweep(failure_frame, target="not_a_column", seeds=[1, 2], time_budget=1)
    assert result["n_ok"] == 0
    assert result["summary"] == {}
    assert "every seed failed" in result["stability_note"].lower()


# ---- Phase 3: self-describing confusion matrix and calculate_correlations ----


def test_evaluate_binary_class_labels_matches_confusion_matrix(binary_engine, signal_frame):
    # class_labels must match the confusion matrix dimensions and order.
    x, y = signal_frame
    metrics = binary_engine.evaluate(x, y)
    assert "class_labels" in metrics
    labels = metrics["class_labels"]
    cm = metrics["confusion_matrix"]
    assert isinstance(labels, list)
    assert len(labels) == len(cm)
    assert len(labels) == len(cm[0])
    assert all(isinstance(l, (int, str)) for l in labels)
    total_cm = sum(sum(row) for row in cm)
    assert total_cm == len(y)
    # Plain Python serialization without numpy scalar leakage
    dumped = json.dumps(metrics, allow_nan=False)
    assert json.loads(dumped)["class_labels"] == labels


def test_evaluate_anomaly_and_multiclass_class_labels(anomaly_engine):
    # Anomaly fallback and multiclass branches also include class_labels.
    rng = np.random.default_rng(5)
    n_rows = 120
    x = pd.DataFrame({
        "s1": rng.normal(0, 1, n_rows),
        "s2": rng.normal(0, 1, n_rows),
        "s3": rng.normal(0, 1, n_rows),
    })
    y = pd.Series([0] * 117 + [1] * 3)
    metrics = anomaly_engine.evaluate(x, y)
    assert "class_labels" in metrics
    assert len(metrics["class_labels"]) == len(metrics["confusion_matrix"])
    assert isinstance(metrics["class_labels"], list)


def test_calculate_correlations_strong_pairs_unique_and_not_self():
    # Strong pairs must be reported once per pair, sorted by absolute correlation descending, and never self-paired.
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "x3": [5.0, 1.0, 4.0, 2.0, 3.0],
    })
    res = po.calculate_correlations(df, min_abs=0.7)
    assert "error" not in res
    pairs = res["strong_pairs"]
    assert len(pairs) == 1
    assert pairs[0]["a"] == "x1" and pairs[0]["b"] == "x2"
    assert pairs[0]["correlation"] == pytest.approx(1.0)
    assert all(p["a"] != p["b"] for p in pairs)


def test_calculate_correlations_skipped_columns_with_reasons():
    # Non-numeric, constant, and sparse columns must be documented in skipped_columns with reasons.
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, 4.0],
        "num2": [4.0, 3.0, 2.0, 1.0],
        "text": ["A", "B", "C", "D"],
        "constant": [7.0, 7.0, 7.0, 7.0],
        "sparse": [1.0, None, None, None],
    })
    res = po.calculate_correlations(df)
    assert "error" not in res
    assert res["columns"] == ["num1", "num2"]
    skipped = res["skipped_columns"]
    assert "text" in skipped and "numeric" in skipped["text"].lower()
    assert "constant" in skipped and "constant" in skipped["constant"].lower()
    assert "sparse" in skipped and "fewer than two" in skipped["sparse"].lower()


def test_calculate_correlations_invalid_method_returns_error_dict():
    # Invalid correlation methods return a helpful error dict naming the valid choices without raising.
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    res = po.calculate_correlations(df, method="invalid_method")
    assert "error" in res
    assert "pearson" in res["error"]
    assert "spearman" in res["error"]
    assert "kendall" in res["error"]


def test_calculate_correlations_json_serialisable():
    # Correlation results survive JSON round-trip serialization.
    df = pd.DataFrame({
        "sensor_a": [10.0, 20.0, 30.0, 40.0],
        "sensor_b": [1.0, 0.0, 1.0, 0.0],
    })
    res = po.calculate_correlations(df)
    dumped = json.dumps(po.to_jsonable(res), allow_nan=False)
    loaded = json.loads(dumped)
    assert loaded["columns"] == ["sensor_a", "sensor_b"]
    assert loaded["n_rows"] == 4
    assert len(loaded["matrix"]) == 2


def test_calculate_correlations_max_columns_truncation_note():
    # Truncating columns retains those with most non-nulls and sets the note field.
    df = pd.DataFrame({
        f"s{i}": [float(i), float(i + 1), float(i + 2)] for i in range(10)
    })
    res = po.calculate_correlations(df, max_columns=4)
    assert len(res["columns"]) == 4
    assert "note" in res
    assert "truncated" in res["note"].lower()


# ---- Quality Engineering Track: Control Rules, Capability, and Gauge R&R ----


def test_control_rule_constants_exposed():
    # Western Electric (1, 2, 5, 6) and Nelson (1-8) sets and descriptions are exported.
    assert po.CONTROL_RULES_WESTERN_ELECTRIC == (1, 2, 5, 6)
    assert po.CONTROL_RULES_NELSON == (1, 2, 3, 4, 5, 6, 7, 8)
    assert len(po.CONTROL_RULE_DESCRIPTIONS) == 8
    for r in range(1, 9):
        assert r in po.CONTROL_RULE_DESCRIPTIONS
        assert isinstance(po.CONTROL_RULE_DESCRIPTIONS[r], str)
        assert len(po.CONTROL_RULE_DESCRIPTIONS[r]) > 0


def test_control_rules_each_rule_fires_on_targeted_pattern():
    # Rule 1: One point beyond 3 sigma
    s1 = [10.0] * 10 + [14.0]
    r1 = po.calculate_control_rules(s1, target=10.0, sigma=1.0, rules=(1,))
    assert r1["violations"]["1"] == [10]
    assert r1["any_violation"] is True
    assert r1["first_violation_index"] == 10
    assert r1["first_violation_rule"] == 1

    # Rule 2: Nine consecutive points on the same side of centre
    s2 = [10.0] * 5 + [11.0] * 9
    r2 = po.calculate_control_rules(s2, target=10.0, sigma=1.0, rules=(2,))
    assert r2["violations"]["2"] == [13]
    assert r2["first_violation_index"] == 13

    # Rule 3: Six consecutive points increasing or decreasing
    s3_inc = [10.0, 10.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    r3_inc = po.calculate_control_rules(s3_inc, target=10.0, sigma=10.0, rules=(3,))
    assert r3_inc["violations"]["3"] == [7]

    # The mirror case needs a rise to break the run, not another fall. Written as
    # [10, 10, 6, 5, ...] the 10 -> 6 step is itself a decrease, so it extends the
    # descending run and the rule correctly reports two overlapping windows.
    s3_dec = [10.0, 10.0, 15.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    r3_dec = po.calculate_control_rules(s3_dec, target=10.0, sigma=10.0, rules=(3,))
    assert r3_dec["violations"]["3"] == [7]

    # A run longer than the rule needs reports every window that satisfies it,
    # not only the first. Nine descending points contain four such windows.
    s3_long = [10.0, 10.0, 15.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    r3_long = po.calculate_control_rules(s3_long, target=10.0, sigma=10.0, rules=(3,))
    assert r3_long["violations"]["3"] == [7, 8, 9, 10]

    # Rule 4: Fourteen consecutive points alternating up and down
    s4 = [10.0, 10.0] + [10.0, 12.0, 10.0, 12.0, 10.0, 12.0, 10.0, 12.0, 10.0, 12.0, 10.0, 12.0, 10.0, 12.0]
    r4 = po.calculate_control_rules(s4, target=11.0, sigma=5.0, rules=(4,))
    assert r4["violations"]["4"] == [15]

    # Rule 5: Two out of three consecutive points beyond 2 sigma (same side)
    s5 = [10.0, 12.5, 10.1, 12.5, 10.0]
    r5 = po.calculate_control_rules(s5, target=10.0, sigma=1.0, rules=(5,))
    assert 3 in r5["violations"]["5"]

    # Rule 6: Four out of five consecutive points beyond 1 sigma (same side)
    s6 = [10.0, 11.5, 11.5, 10.2, 11.5, 11.5]
    r6 = po.calculate_control_rules(s6, target=10.0, sigma=1.0, rules=(6,))
    assert 5 in r6["violations"]["6"]

    # Rule 7: Fifteen consecutive points within 1 sigma of centre
    s7 = [10.2] * 15
    r7 = po.calculate_control_rules(s7, target=10.0, sigma=1.0, rules=(7,))
    assert r7["violations"]["7"] == [14]

    # Rule 8: Eight consecutive points beyond 1 sigma on either side
    s8 = [10.0] * 2 + [12.0, 8.0, 12.0, 8.0, 12.0, 8.0, 12.0, 8.0]
    r8 = po.calculate_control_rules(s8, target=10.0, sigma=1.0, rules=(8,))
    assert r8["violations"]["8"] == [9]


def test_control_rules_rule_1_does_not_fire_on_in_control_data():
    # Clean data with small noise well within 3 sigma never trips Rule 1.
    series = [10.0 + 0.2 * (i % 3 - 1) for i in range(50)]
    res = po.calculate_control_rules(series, target=10.0, sigma=1.0, rules=(1,))
    assert res["any_violation"] is False
    assert res["violations"]["1"] == []
    assert res["first_violation_index"] is None
    assert res["first_violation_rule"] is None


def test_control_rules_short_series_reports_rules_skipped():
    # Rules that cannot be evaluated due to insufficient points are documented in rules_skipped, not violations.
    series_short = [10.0, 10.5, 9.8, 10.2, 10.1]  # 5 points
    res = po.calculate_control_rules(series_short, target=10.0, sigma=1.0, rules=po.CONTROL_RULES_NELSON)
    assert "error" not in res
    assert "2" in res["rules_skipped"]  # needs 9
    assert "3" in res["rules_skipped"]  # needs 6
    assert "4" in res["rules_skipped"]  # needs 14
    assert "7" in res["rules_skipped"]  # needs 15
    assert "8" in res["rules_skipped"]  # needs 8
    assert "4" not in res["violations"]
    assert "7" not in res["violations"]
    assert "14" in res["rules_skipped"]["4"]


def test_control_rules_false_alarm_rate_reproducibility():
    # Simulated false-alarm rates confirm Western Electric alarms far more than Rule 1 alone on in-control data.
    rng = np.random.default_rng(42)
    n_trials = 200
    n_points = 100
    r1_alarms = 0
    we_alarms = 0
    for _ in range(n_trials):
        data = rng.normal(0.0, 1.0, n_points)
        res_r1 = po.calculate_control_rules(data, target=0.0, sigma=1.0, rules=(1,))
        res_we = po.calculate_control_rules(data, target=0.0, sigma=1.0, rules=po.CONTROL_RULES_WESTERN_ELECTRIC)
        if res_r1["any_violation"]:
            r1_alarms += 1
        if res_we["any_violation"]:
            we_alarms += 1

    rate_r1 = r1_alarms / n_trials
    rate_we = we_alarms / n_trials
    assert 0.10 <= rate_r1 <= 0.40
    assert 0.40 <= rate_we <= 0.80
    assert rate_we > rate_r1 + 0.15


def test_calculate_capability_reports_meaningful_false_on_trend():
    # Capability is meaningless on an out-of-control process even when raw Cpk looks capable.
    ramp = np.linspace(10.0, 15.0, 60)
    res = po.calculate_capability(ramp, usl=40.0, lsl=-20.0)
    assert res["stable"] is False
    assert res["capability_is_meaningful"] is False
    assert "repeats itself" in res["interpretation"].lower()
    assert len(res["stability_violations"]) > 0
    assert res["cpk"] > 1.33


def test_calculate_capability_calls_healthy_data_stable():
    # The other half of the guard, and the half that is easy to lose. The first
    # implementation gated `stable` on "did any Western Electric rule fire",
    # which measures False on 97.5% of healthy 400-point series and 99.5% of
    # 1,000-point ones: the false-alarm rate climbed toward certainty as more
    # data arrived, and a flag that is always raised carries no information.
    # A long, clean, in-control series must survive the guard.
    healthy = np.random.default_rng(3).normal(100.0, 1.0, 400)
    res = po.calculate_capability(healthy, usl=106.0, lsl=94.0)

    assert res["stable"] is True, res.get("interpretation")
    assert res["capability_is_meaningful"] is True
    assert res["sigma_ratio"] < po.CAPABILITY_SIGMA_RATIO_LIMIT
    assert res["outlier_rate_beyond_3_sigma"] <= po.CAPABILITY_OUTLIER_RATE_LIMIT
    # The rule violations are still reported - they say where to look - they just
    # no longer decide the verdict on their own.
    assert "stability_violations" in res


def test_capability_limits_are_length_independent_and_documented():
    # Both limits are chosen from measured curves, so they must stay values a
    # reader can look up rather than numbers buried in a comparison.
    assert po.CAPABILITY_SIGMA_RATIO_LIMIT == 1.20
    assert po.CAPABILITY_OUTLIER_RATE_LIMIT == 0.01

    rng = np.random.default_rng(11)
    # The same healthy process at two very different lengths must reach the same
    # verdict. Under the old rule-based gate the longer series was far likelier
    # to be rejected purely for being longer.
    short = po.calculate_capability(rng.normal(50.0, 2.0, 80), usl=60.0, lsl=40.0)
    long = po.calculate_capability(rng.normal(50.0, 2.0, 800), usl=60.0, lsl=40.0)
    assert short["stable"] is True
    assert long["stable"] is True


def test_calculate_capability_sigma_within_vs_overall_spread():
    # Drifting series causes sigma_overall > sigma_within, while stable series has sigma_overall ~ sigma_within.
    drift = np.linspace(0.0, 20.0, 100) + np.random.default_rng(7).normal(0.0, 0.5, 100)
    res_drift = po.calculate_capability(drift, usl=40.0, lsl=-20.0)
    assert res_drift["sigma_overall"] > res_drift["sigma_within"] * 1.5
    assert res_drift["cp"] > res_drift["pp"] * 1.5
    assert res_drift["sigma_ratio"] > 1.5

    rng = np.random.default_rng(99)
    stable_data = rng.normal(10.0, 1.0, 400)
    res_stable = po.calculate_capability(stable_data, usl=15.0, lsl=5.0)
    assert res_stable["sigma_within"] == pytest.approx(res_stable["sigma_overall"], rel=0.15)
    assert res_stable["cp"] == pytest.approx(res_stable["pp"], rel=0.15)
    assert res_stable["sigma_ratio"] == pytest.approx(1.0, rel=0.15)
    assert res_stable["stable"] is True
    assert res_stable["capability_is_meaningful"] is True
    assert res_stable["verdict"] in ("capable", "marginal")


def test_calculate_capability_one_sided_specification():
    # One-sided spec limit returns cp: None, pp: None, with valid one-sided cpk and ppk.
    rng = np.random.default_rng(10)
    normal_data = rng.normal(10.0, 1.0, 200)

    res_usl = po.calculate_capability(normal_data, usl=14.0)
    assert res_usl["cp"] is None
    assert res_usl["pp"] is None
    assert res_usl["cpk"] is not None and res_usl["cpk"] > 1.0
    assert res_usl["ppk"] is not None and res_usl["ppk"] > 1.0
    assert res_usl["cpu"] == res_usl["cpk"]
    assert res_usl["cpl"] is None

    res_lsl = po.calculate_capability(normal_data, lsl=6.0)
    assert res_lsl["cp"] is None
    assert res_lsl["pp"] is None
    assert res_lsl["cpk"] is not None and res_lsl["cpk"] > 1.0
    assert res_lsl["ppk"] is not None and res_lsl["ppk"] > 1.0
    assert res_lsl["cpl"] == res_lsl["cpk"]
    assert res_lsl["cpu"] is None


def test_calculate_capability_normality_warning():
    # Heavy skewness or kurtosis triggers a descriptive normality warning without running hypothesis tests.
    skewed_data = [1.0] * 100 + [50.0] * 10
    res = po.calculate_capability(skewed_data, usl=100.0, lsl=0.0)
    assert res["normality_warning"] is not None
    assert "normality" in res["normality_warning"].lower()


def test_calculate_gauge_rr_high_reproducibility():
    # Large operator differences produce a high reproducibility variance share and unacceptable verdict.
    rows = []
    for part in range(5):
        for op, op_offset in [("OpA", 0.0), ("OpB", 15.0), ("OpC", -15.0)]:
            for rep in range(3):
                rows.append({
                    "part": f"P{part}",
                    "operator": op,
                    "meas": 50.0 + part * 0.5 + op_offset + (rep * 0.05),
                })
    df_op = pd.DataFrame(rows)
    res_op = po.calculate_gauge_rr(df_op, "part", "operator", "meas")
    assert "error" not in res_op
    assert res_op["n_parts"] == 5
    assert res_op["n_operators"] == 3
    assert res_op["n_replicates"] == 3
    assert res_op["percent_contribution"]["reproducibility"] > 50.0
    assert res_op["verdict"] == "unacceptable"


def test_calculate_gauge_rr_acceptable_system_and_ndc():
    # High part variation with negligible operator and equipment noise produces acceptable GRR (<10%) and ndc >= 5.
    rows_good = []
    for part in range(10):
        part_val = 10.0 + part * 10.0
        for op in ["Op1", "Op2", "Op3"]:
            for rep in range(3):
                rows_good.append({
                    "part": f"Part{part}",
                    "operator": op,
                    "meas": part_val + 0.01 * (rep - 1),
                })
    df_good = pd.DataFrame(rows_good)
    res_good = po.calculate_gauge_rr(df_good, "part", "operator", "meas")
    assert "error" not in res_good
    assert res_good["percent_study_variation"]["gauge_rr"] < 10.0
    assert res_good["verdict"] == "acceptable"
    assert res_good["ndc"] is not None and res_good["ndc"] >= 5
    assert res_good["ndc_verdict"] == "acceptable"


def test_calculate_gauge_rr_unbalanced_design_returns_error():
    # Unbalanced replicates and missing cells return error dicts naming the problem without raising.
    rows_good = []
    for part in range(4):
        for op in ["Op1", "Op2"]:
            for rep in range(3):
                rows_good.append({
                    "part": f"P{part}",
                    "operator": op,
                    "meas": float(part * 5 + rep),
                })
    df = pd.DataFrame(rows_good)

    # Unequal replicates
    df_unbal = df.iloc[:-1].copy()
    res_unbal = po.calculate_gauge_rr(df_unbal, "part", "operator", "meas")
    assert "error" in res_unbal
    assert "unbalanced" in res_unbal["error"].lower()

    # Missing cell
    df_missing = df[~((df["part"] == "P0") & (df["operator"] == "Op1"))].copy()
    res_missing = po.calculate_gauge_rr(df_missing, "part", "operator", "meas")
    assert "error" in res_missing
    assert "unbalanced" in res_missing["error"].lower() or "missing" in res_missing["error"].lower()


def test_quality_engineering_json_serialisable():
    # All three functions return structures that survive strict JSON serialization.
    res_cr = po.calculate_control_rules([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    dumped_cr = json.dumps(po.to_jsonable(res_cr), allow_nan=False)
    assert json.loads(dumped_cr)["n_points"] == 6

    res_cap = po.calculate_capability([10.0, 10.5, 9.8, 10.2, 10.1], usl=15.0, lsl=5.0)
    dumped_cap = json.dumps(po.to_jsonable(res_cap), allow_nan=False)
    assert "cpk" in json.loads(dumped_cap)

    rows = []
    for part in range(3):
        for op in ["Op1", "Op2"]:
            for rep in range(2):
                rows.append({"part": f"P{part}", "operator": op, "meas": float(part * 2 + rep)})
    res_grr = po.calculate_gauge_rr(pd.DataFrame(rows), "part", "operator", "meas")
    dumped_grr = json.dumps(po.to_jsonable(res_grr), allow_nan=False)
    assert "ndc" in json.loads(dumped_grr)


def test_quality_engineering_invalid_inputs_return_error_dicts():
    # Functions return error dicts on malformed inputs without raising exceptions.
    assert "error" in po.calculate_control_rules([])
    assert "error" in po.calculate_control_rules([10.0])
    assert "error" in po.calculate_control_rules([10.0, 11.0], sigma=-1.0)
    assert "error" in po.calculate_control_rules([10.0, 11.0], rules=(9,))

    assert "error" in po.calculate_capability([10.0, 11.0])
    assert "error" in po.calculate_capability([10.0, 11.0], usl=5.0, lsl=10.0)
    assert "error" in po.calculate_capability([], usl=10.0, lsl=5.0)

    assert "error" in po.calculate_gauge_rr(None, "part", "operator", "meas")
    assert "error" in po.calculate_gauge_rr(pd.DataFrame(), "part", "operator", "meas")
    assert "error" in po.calculate_gauge_rr(
        pd.DataFrame({"part": ["P1"], "operator": ["O1"], "meas": [1.0]}),
        "part", "operator", "meas"
    )
