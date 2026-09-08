"""
Tests for asset-aware and time-aware splitting in `potatopt.data`.

A random shuffle answers "how well does this do on rows like the ones it trained
on". On maintenance data almost nobody wants that answer: the machine you will be
asked about next is one the model has never met, and the month you will be asked
about is one that has not happened yet. `group_col` and `time_col` ask those two
questions instead, and this file pins the properties that make the answers mean
what they say.
"""

import numpy as np
import pandas as pd
import pytest

import potatopt as po


@pytest.fixture(scope="module")
def fleet():
    """Eight machines, 60 readings each, with a timestamp shared across machines."""
    rng = np.random.default_rng(42)
    frames = []
    for a in range(8):
        frames.append(pd.DataFrame({
            "machine_id": f"M-{a:02d}",
            "hour": np.arange(60),
            "temperature": rng.normal(50 + a, 2, 60),
            "vibration": rng.normal(1.0, 0.1, 60),
            "failure": rng.integers(0, 2, 60),
        }))
    return pd.concat(frames, ignore_index=True)


# ---- the two questions are not interchangeable ----


def test_group_and_time_together_are_refused(fleet):
    """They answer different questions, so the caller has to say which one the
    reported score is supposed to mean."""
    with pytest.raises(ValueError, match="group_col|time_col"):
        po.split_data(fleet, "failure", group_col="machine_id", time_col="hour")


def test_a_missing_group_column_is_named(fleet):
    with pytest.raises(ValueError, match="not_a_column"):
        po.split_data(fleet, "failure", group_col="not_a_column")


def test_a_single_group_cannot_be_split(fleet):
    one = fleet[fleet["machine_id"] == "M-00"]
    with pytest.raises(ValueError):
        po.split_data(one, "failure", group_col="machine_id")


# ---- group splits ----


def test_no_machine_appears_on_both_sides(fleet):
    X_train, X_test, _, _ = po.split_data(
        fleet, "failure", group_col="machine_id", random_state=1, drop_group_col=False)

    assert not set(X_train["machine_id"]) & set(X_test["machine_id"])


def test_the_group_column_is_dropped_by_default(fleet):
    """Held-out machines carry ids the encoder never saw, so the column offers the
    model nothing on the test side while handing it something to memorise on the
    train side."""
    X_train, X_test, _, _ = po.split_data(
        fleet, "failure", group_col="machine_id", random_state=1)

    assert "machine_id" not in X_train.columns
    assert "machine_id" not in X_test.columns


def test_the_group_column_can_be_kept_for_reporting(fleet):
    X_train, X_test, _, _ = po.split_data(
        fleet, "failure", group_col="machine_id", random_state=1, drop_group_col=False)

    assert "machine_id" in X_train.columns
    assert "machine_id" in X_test.columns


def test_the_draw_of_machines_follows_random_state(fleet):
    """`run_seed_sweep` measures how much the score moves between draws of
    machines. That only tells you anything if the draw actually moves."""
    draws = set()
    for seed in (1, 2, 3, 4, 5, 6):
        _, X_test, _, _ = po.split_data(
            fleet, "failure", group_col="machine_id", random_state=seed,
            drop_group_col=False)
        draws.add(tuple(sorted(set(X_test["machine_id"]))))

    assert len(draws) > 1


def test_the_same_seed_draws_the_same_machines(fleet):
    a = po.split_data(fleet, "failure", group_col="machine_id", random_state=7,
                      drop_group_col=False)
    b = po.split_data(fleet, "failure", group_col="machine_id", random_state=7,
                      drop_group_col=False)

    assert sorted(set(a[1]["machine_id"])) == sorted(set(b[1]["machine_id"]))


def test_a_group_split_always_leaves_something_to_train_on(fleet):
    _, _, y_train, _ = po.split_data(
        fleet, "failure", group_col="machine_id", test_size=0.99, random_state=1,
        drop_group_col=False)

    assert len(y_train) > 0


# ---- time splits ----


def test_training_rows_never_come_after_test_rows(fleet):
    X_train, X_test, _, _ = po.split_data(fleet, "failure", time_col="hour")

    assert X_train["hour"].max() <= X_test["hour"].min()


def test_a_timestamp_never_straddles_the_boundary(fleet):
    """Eight machines share every hour value. A cut that lands mid-hour puts one
    machine's 10:00 reading in train and another's in test - which is the leak the
    argument exists to prevent, wearing a different hat."""
    X_train, X_test, _, _ = po.split_data(fleet, "failure", time_col="hour")

    assert not set(X_train["hour"]) & set(X_test["hour"])


def test_too_few_distinct_timestamps_is_refused_rather_than_fudged():
    frame = pd.DataFrame({"hour": [1] * 20, "v": np.arange(20), "failure": [0, 1] * 10})

    with pytest.raises(ValueError):
        po.split_data(frame, "failure", time_col="hour")


# ---- the old behaviour is untouched ----


def test_the_default_split_is_unchanged(fleet):
    """Every new argument defaults to the previous behaviour; this is the test that
    says so."""
    X_train, X_test, _, _ = po.split_data(fleet, "failure", random_state=42)

    assert len(X_test) == round(0.2 * len(fleet))
    assert len(X_train) + len(X_test) == len(fleet)
    assert "machine_id" in X_train.columns

    again = po.split_data(fleet, "failure", random_state=42)
    assert X_train.index.equals(again[0].index)


# ---- three-way splits ----


def test_three_way_group_split_keeps_all_three_partitions_disjoint(fleet):
    X_tr, X_val, X_te, _, _, _ = po.split_data_three_way(
        fleet, "failure", group_col="machine_id", random_state=3, drop_group_col=False)

    tr, val, te = (set(f["machine_id"]) for f in (X_tr, X_val, X_te))
    assert not tr & val
    assert not tr & te
    assert not val & te


def test_three_way_group_split_drops_the_id_from_every_partition(fleet):
    X_tr, X_val, X_te, _, _, _ = po.split_data_three_way(
        fleet, "failure", group_col="machine_id", random_state=3)

    for frame in (X_tr, X_val, X_te):
        assert "machine_id" not in frame.columns


def test_three_way_time_split_runs_train_then_validation_then_test(fleet):
    X_tr, X_val, X_te, _, _, _ = po.split_data_three_way(
        fleet, "failure", time_col="hour")

    assert X_tr["hour"].max() <= X_val["hour"].min()
    assert X_val["hour"].max() <= X_te["hour"].min()


def test_three_way_partitions_still_account_for_every_row(fleet):
    X_tr, X_val, X_te, y_tr, y_val, y_te = po.split_data_three_way(
        fleet, "failure", group_col="machine_id", random_state=3, drop_group_col=False)

    assert len(X_tr) + len(X_val) + len(X_te) == len(fleet)
    assert len(y_tr) + len(y_val) + len(y_te) == len(fleet)
