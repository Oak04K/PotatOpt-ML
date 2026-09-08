"""
Tests for `potatopt.sequence` - turning a sensor log and an event log into a
modelling table.

The centre of gravity here is `test_window_features_cannot_see_the_future`. Every
other property in this file can be checked by reading the code; that one cannot.
A rolling window pointed the wrong way still produces plausible columns, trains
without complaint and reports a good score - the only way to know it is causal is
to corrupt the future and prove the past did not move.
"""

import json

import numpy as np
import pandas as pd
import pytest

import potatopt as po


@pytest.fixture(scope="module")
def sensor_log():
    """Three machines, 200 readings each, sitting at deliberately different levels."""
    rng = np.random.default_rng(42)
    frames = []
    for a in range(3):
        frames.append(pd.DataFrame({
            "machine_id": f"M-{a:02d}",
            "hour": np.arange(200),
            "temperature": 50 + a * 3 + np.linspace(0, 5, 200) + rng.normal(0, 0.5, 200),
            "vibration": 1.0 + rng.normal(0, 0.1, 200),
        }))
    return pd.concat(frames, ignore_index=True)


# ---- build_failure_labels: the evidence window ----


def test_evidence_window_is_open_at_the_failure_and_closed_at_the_horizon():
    """One event at t=5 with horizon 3 labels exactly t=2, 3, 4.

    The reading taken AT the failure is not labelled: a machine that has just
    broken is not a machine about to break, and labelling it teaches the model to
    recognise the present rather than predict the future.
    """
    readings = pd.DataFrame({"machine_id": "M-01", "hour": np.arange(10), "temp": 1.0})
    events = pd.DataFrame({"machine_id": ["M-01"], "failed_at": [5]})

    out, _ = po.build_failure_labels(readings, events, "machine_id", "hour", horizon=3)

    assert sorted(out.loc[out["failure"] == 1, "hour"].tolist()) == [2, 3, 4]
    assert int(out.loc[out["hour"] == 5, "failure"].iloc[0]) == 0


def test_each_failure_opens_its_own_window():
    readings = pd.DataFrame({"machine_id": "M-01", "hour": np.arange(20), "temp": 1.0})
    events = pd.DataFrame({"machine_id": ["M-01", "M-01"], "failed_at": [5, 12]})

    out, _ = po.build_failure_labels(readings, events, "machine_id", "hour", horizon=3)

    assert sorted(out.loc[out["failure"] == 1, "hour"].tolist()) == [2, 3, 4, 9, 10, 11]


def test_time_to_event_is_emitted_and_censored_after_the_last_failure():
    readings = pd.DataFrame({"machine_id": "M-01", "hour": np.arange(10), "temp": 1.0})
    events = pd.DataFrame({"machine_id": ["M-01"], "failed_at": [5]})

    out, report = po.build_failure_labels(readings, events, "machine_id", "hour", horizon=3)

    assert float(out.loc[out["hour"] == 3, "time_to_event"].iloc[0]) == 2.0
    assert bool(pd.isna(out.loc[out["hour"] == 7, "time_to_event"].iloc[0]))
    assert report["censored_rows"] == 5


def test_lead_time_rows_are_counted_rather_than_called_healthy():
    """A warning too late to act on is not the same as a healthy reading."""
    readings = pd.DataFrame({"machine_id": "M-01", "hour": np.arange(20), "temp": 1.0})
    events = pd.DataFrame({"machine_id": ["M-01", "M-01"], "failed_at": [5, 12]})

    out, report = po.build_failure_labels(
        readings, events, "machine_id", "hour", horizon=4, lead_time=2)

    assert sorted(out.loc[out["failure"] == 1, "hour"].tolist()) == [1, 2, 8, 9]
    assert report["rows_inside_lead_time"] == 4


def test_labels_never_cross_machines():
    readings = pd.concat(
        [pd.DataFrame({"machine_id": m, "hour": np.arange(10), "temp": 1.0})
         for m in ("M-01", "M-02")], ignore_index=True)
    events = pd.DataFrame({"machine_id": ["M-01"], "failed_at": [5]})

    out, _ = po.build_failure_labels(readings, events, "machine_id", "hour", horizon=3)

    assert out.loc[out["machine_id"] == "M-02", "failure"].sum() == 0


def test_a_machine_with_no_event_is_named_rather_than_silently_zeroed():
    """An all-zero label column is an assumption ("it never failed"), not a fact
    ("we have no record of it failing"). Both cases must be visible."""
    readings = pd.concat(
        [pd.DataFrame({"machine_id": m, "hour": np.arange(10), "temp": 1.0})
         for m in ("M-01", "M-02")], ignore_index=True)
    events = pd.DataFrame({"machine_id": ["M-01", "M-99"], "failed_at": [5, 5]})

    _, report = po.build_failure_labels(readings, events, "machine_id", "hour", horizon=3)

    assert "M-02" in report["assets_without_events"]
    assert "M-99" in report["events_without_readings"]


def test_datetime_readings_take_a_duration_horizon_and_report_hours():
    base = pd.Timestamp("2026-01-01")
    readings = pd.DataFrame({
        "m": "A",
        "ts": [base + pd.Timedelta(hours=i) for i in range(20)],
        "v": 1.0,
    })
    events = pd.DataFrame({"m": ["A"], "failed_at": [base + pd.Timedelta(hours=10)]})

    out, _ = po.build_failure_labels(readings, events, "m", "ts", horizon="3h")

    assert sorted(out.index[out["failure"] == 1].tolist()) == [7, 8, 9]
    assert float(out.loc[7, "time_to_event"]) == 3.0


def test_row_order_and_index_of_the_input_are_preserved():
    readings = pd.DataFrame(
        {"machine_id": "M-01", "hour": np.arange(20), "temp": 1.0},
        index=[f"r{i}" for i in range(20)])
    events = pd.DataFrame({"machine_id": ["M-01"], "failed_at": [5]})

    out, _ = po.build_failure_labels(readings, events, "machine_id", "hour", horizon=3)

    assert list(out.index) == list(readings.index)
    assert list(out["hour"]) == list(readings["hour"])


@pytest.mark.parametrize(("kwargs", "match"), [
    ({"horizon": 0}, "positive"),
    ({"horizon": -1}, "positive"),
    ({"horizon": 3, "lead_time": -1}, "non-negative"),
    ({"horizon": 3, "lead_time": 3}, "strictly less"),
    ({"horizon": 3, "label_col": "temp"}, "already present"),
])
def test_bad_label_arguments_raise_with_the_offending_name(kwargs, match):
    readings = pd.DataFrame({"machine_id": "M-01", "hour": np.arange(10), "temp": 1.0})
    events = pd.DataFrame({"machine_id": ["M-01"], "failed_at": [5]})

    with pytest.raises(ValueError, match=match):
        po.build_failure_labels(readings, events, "machine_id", "hour", **kwargs)


def test_mismatched_time_dtypes_are_refused_by_name():
    base = pd.Timestamp("2026-01-01")
    readings = pd.DataFrame({"m": "A", "ts": [base + pd.Timedelta(hours=i) for i in range(10)]})
    events = pd.DataFrame({"m": ["A"], "failed_at": [5]})

    with pytest.raises(ValueError, match="dtype"):
        po.build_failure_labels(readings, events, "m", "ts", horizon=3)


# ---- add_window_features: causality ----


def test_window_features_cannot_see_the_future(sensor_log):
    """Corrupt every reading after hour 150 and prove nothing before it moved.

    This is the test the module exists to pass. A centred window, or a baseline
    measured over the whole series, would fail it here rather than months later
    in a score nobody could explain.
    """
    clean, _ = po.add_window_features(
        sensor_log, "machine_id", "hour", value_cols=["temperature", "vibration"])

    corrupted = sensor_log.copy()
    future = corrupted["hour"] > 150
    corrupted.loc[future, "temperature"] = 9999.0
    corrupted.loc[future, "vibration"] = -9999.0
    dirty, _ = po.add_window_features(
        corrupted, "machine_id", "hour", value_cols=["temperature", "vibration"])

    past_clean = clean[clean["hour"] <= 150].reset_index(drop=True)
    past_dirty = dirty[dirty["hour"] <= 150].reset_index(drop=True)

    assert list(past_clean.columns) == list(past_dirty.columns)
    assert len(past_clean) == len(past_dirty)
    for col in past_clean.columns:
        if past_clean[col].dtype.kind in "fi":
            assert np.allclose(
                past_clean[col].to_numpy(float), past_dirty[col].to_numpy(float),
                equal_nan=True), f"a future reading changed {col}"


def test_one_machine_cannot_move_another_machines_features(sensor_log):
    clean, _ = po.add_window_features(
        sensor_log, "machine_id", "hour", value_cols=["temperature"])

    corrupted = sensor_log.copy()
    corrupted.loc[corrupted["machine_id"] == "M-02", "temperature"] += 500
    dirty, _ = po.add_window_features(
        corrupted, "machine_id", "hour", value_cols=["temperature"])

    a = clean[clean["machine_id"] == "M-00"].reset_index(drop=True)
    b = dirty[dirty["machine_id"] == "M-00"].reset_index(drop=True)
    for col in a.columns:
        if a[col].dtype.kind in "fi":
            assert np.allclose(a[col].to_numpy(float), b[col].to_numpy(float), equal_nan=True)


def test_shuffled_input_rows_give_the_same_features(sensor_log):
    ordered, _ = po.add_window_features(
        sensor_log, "machine_id", "hour", value_cols=["temperature"])
    shuffled, _ = po.add_window_features(
        sensor_log.sample(frac=1.0, random_state=7).reset_index(drop=True),
        "machine_id", "hour", value_cols=["temperature"])

    assert len(ordered) == len(shuffled)
    for col in ordered.columns:
        if ordered[col].dtype.kind in "fi":
            assert np.allclose(
                ordered[col].to_numpy(float), shuffled[col].to_numpy(float), equal_nan=True)


def test_the_baseline_delta_is_null_over_the_window_that_defined_it():
    """Rows inside the baseline window helped compute the baseline. Subtracting it
    from them uses their own future, so the value must be absent, not merely small.
    """
    frame = pd.concat(
        [pd.DataFrame({"m": m, "t": np.arange(60), "v": np.linspace(0, 10, 60)})
         for m in ("A", "B")], ignore_index=True)

    out, _ = po.add_window_features(
        frame, "m", "t", value_cols=["v"], windows=(5,), baseline_n=20,
        drop_incomplete=False)

    assert out.loc[out["t"] < 20, "v_vs_baseline"].isna().all()
    assert out.loc[out["t"] >= 20, "v_vs_baseline"].notna().all()


def test_keeping_incomplete_rows_does_not_switch_the_leak_guard_off():
    frame = pd.DataFrame({"m": "A", "t": np.arange(60), "v": np.linspace(0, 10, 60)})

    kept, report = po.add_window_features(
        frame, "m", "t", value_cols=["v"], windows=(5,), baseline_n=20,
        drop_incomplete=False)

    assert report["incomplete_rows_kept"] > 0
    assert kept.loc[kept["t"] < 20, "v_vs_baseline"].isna().all()


def test_the_label_column_survives_but_never_becomes_a_feature(sensor_log):
    frame = sensor_log.assign(failure=0)

    out, report = po.add_window_features(frame, "machine_id", "hour", label_col="failure")

    assert "failure" in out.columns
    assert not any("failure" in name for name in report["features_added"])
    assert "failure" not in report["value_cols"]


def test_the_row_and_memory_cost_is_reported_not_discovered(sensor_log):
    out, report = po.add_window_features(
        sensor_log, "machine_id", "hour", value_cols=["temperature"])

    assert report["rows_in"] - report["rows_out"] == report["rows_dropped"]
    assert report["rows_out"] == len(out)
    assert report["n_features_added"] == len(report["features_added"])
    assert report["memory_mb_added"] >= 0


def test_an_asset_too_short_for_the_window_is_named():
    frame = pd.concat([
        pd.DataFrame({"m": "LONG", "t": np.arange(60), "v": 1.0}),
        pd.DataFrame({"m": "TINY", "t": np.arange(10), "v": 1.0}),
    ], ignore_index=True)

    _, report = po.add_window_features(
        frame, "m", "t", value_cols=["v"], windows=(5,), baseline_n=20)

    assert "TINY" in report["assets_skipped"]
    assert "LONG" not in report["assets_skipped"]


@pytest.mark.parametrize("kwargs", [
    {"stats": ("bogus",)},
    {"stats": ()},
    {"windows": (1,)},
    {"windows": ()},
    {"baseline_n": 0},
    {"value_cols": ["missing_column"]},
])
def test_bad_window_arguments_are_refused(sensor_log, kwargs):
    kwargs.setdefault("value_cols", ["temperature"])
    with pytest.raises(ValueError):
        po.add_window_features(sensor_log, "machine_id", "hour", **kwargs)


# ---- both reports honour the package-wide JSON guarantee ----


def test_both_reports_survive_json_dumps(sensor_log):
    events = pd.DataFrame({"machine_id": ["M-00"], "failed_at": [100]})
    _, label_report = po.build_failure_labels(
        sensor_log, events, "machine_id", "hour", horizon=10)
    _, feature_report = po.add_window_features(
        sensor_log, "machine_id", "hour", value_cols=["temperature"])

    json.dumps(label_report)
    json.dumps(feature_report)
