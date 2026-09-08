"""
Tests for the Weibull hazard check in `potatopt.reliability`.

`calculate_mtbf` reports `failure_rate_per_hour` as 1 / MTBF. That is the
instantaneous failure rate only when the hazard rate does not change with age. A
machine that is wearing out has a rising hazard, so the figure understates the
risk of failing soon - quietly, and in the reassuring direction. These tests pin
the check that says whether the assumption holds, and pin the wording that admits
`calculate_mtbf` is making it.
"""

import json

import numpy as np
import pandas as pd
import pytest

import potatopt as po

# ---- shape recovery ----


@pytest.mark.parametrize(("beta_true", "pattern"), [
    (0.5, "infant_mortality"),
    (2.5, "wear_out"),
    (3.0, "wear_out"),
])
def test_a_changing_hazard_is_recovered_and_named(beta_true, pattern):
    rng = np.random.default_rng(3)
    intervals = rng.weibull(beta_true, 400) * 100.0

    result = po.calculate_weibull(intervals, n_bootstrap=200)

    assert result["beta"] == pytest.approx(beta_true, rel=0.20)
    assert result["constant_hazard_is_valid"] is False
    assert result["hazard_pattern"] == pattern


def test_exponential_intervals_are_called_constant_hazard():
    """Beta is truly 1 here. Calling this a wear-out pattern would send an engineer
    to strip a bearing that is fine."""
    rng = np.random.default_rng(3)
    intervals = rng.exponential(100.0, 400)

    result = po.calculate_weibull(intervals, n_bootstrap=200)

    assert result["constant_hazard_is_valid"] is True
    assert result["hazard_pattern"] == "constant"
    assert result["beta_ci_lower"] <= 1.0 <= result["beta_ci_upper"]


def test_the_interval_brackets_the_estimate():
    rng = np.random.default_rng(11)
    result = po.calculate_weibull(rng.weibull(2.0, 200) * 50.0, n_bootstrap=200)

    assert result["beta_ci_lower"] <= result["beta"] <= result["beta_ci_upper"]


def test_the_same_seed_gives_the_same_interval():
    rng = np.random.default_rng(5)
    intervals = rng.weibull(2.0, 120) * 80.0

    a = po.calculate_weibull(intervals, n_bootstrap=200, random_state=1)
    b = po.calculate_weibull(intervals, n_bootstrap=200, random_state=1)

    assert a["beta_ci_lower"] == b["beta_ci_lower"]
    assert a["beta_ci_upper"] == b["beta_ci_upper"]


def test_life_figures_are_ordered_the_way_the_definitions_require():
    """B10 is when a tenth have failed; eta is when 63.2% have. B10 comes first."""
    rng = np.random.default_rng(7)
    result = po.calculate_weibull(rng.weibull(2.0, 300) * 100.0, n_bootstrap=200)

    assert 0 < result["b10_life"] < result["eta"]
    assert result["mttf"] > 0


# ---- refusals are answers, not exceptions ----


def test_too_few_intervals_is_refused_with_a_reason():
    result = po.calculate_weibull([10.0, 20.0, 30.0])

    assert result["beta"] is None
    assert result["constant_hazard_is_valid"] is None
    assert "3" in result["reason"]


@pytest.mark.parametrize("bad", [
    None, "nonsense", [], [1.0], [-5.0, -3.0], [np.nan, np.nan], object(),
])
def test_bad_input_comes_back_as_a_dictionary_never_an_exception(bad):
    result = po.calculate_weibull(bad)

    assert isinstance(result, dict)
    assert result["beta"] is None
    assert result["reason"]


def test_non_positive_values_are_dropped_and_counted():
    """A Weibull is undefined at or below zero, so these cannot simply be fitted."""
    rng = np.random.default_rng(2)
    intervals = list(rng.weibull(2.0, 60) * 100.0) + [0.0, -4.0]

    result = po.calculate_weibull(intervals, n_bootstrap=100)

    assert result["dropped_non_positive"] == 2
    assert result["n_intervals"] == 60


def test_the_reason_is_always_present_and_the_report_is_json_safe():
    rng = np.random.default_rng(4)
    good = po.calculate_weibull(rng.weibull(2.0, 100) * 100.0, n_bootstrap=100)
    bad = po.calculate_weibull([1.0])

    for result in (good, bad):
        assert result["reason"]
        json.dumps(result)


def test_a_small_sample_says_so_on_the_result():
    """The bias correction brings the false-alarm rate near nominal by about 60
    intervals, not before. Below 30 the result has to carry that, because the
    caller reading it is deciding whether to open a machine."""
    rng = np.random.default_rng(21)
    few = po.calculate_weibull(rng.weibull(2.0, 12) * 100.0)
    many = po.calculate_weibull(rng.weibull(2.0, 200) * 100.0)

    assert few["small_sample_warning"] is not None
    assert "12" in few["small_sample_warning"]
    assert many["small_sample_warning"] is None


def test_an_out_of_range_confidence_is_refused_not_silently_replaced():
    """Substituting the default would make the interval mean something other than
    what the caller asked for, and nothing downstream could notice."""
    rng = np.random.default_rng(21)
    intervals = rng.weibull(2.0, 60) * 100.0

    result = po.calculate_weibull(intervals, confidence=1.5)

    assert result["beta"] is None
    assert result["confidence"] is None
    assert "0.95" in result["reason"] and "NOT" in result["reason"]


def test_the_interval_is_bias_corrected_not_a_raw_percentile():
    """On exponential data the honest verdict is "constant". The raw percentile
    interval excluded 1.0 far more often than its own confidence level allows;
    this pins that the shipped result is the corrected one."""
    flagged = 0
    for trial in range(60):
        rng = np.random.default_rng(5000 + trial)
        result = po.calculate_weibull(rng.exponential(100.0, 8), random_state=trial)
        if result["constant_hazard_is_valid"] is False:
            flagged += 1

    # measured at 9.0% over 300 trials; 60 trials here, so allow the sampling noise
    assert flagged / 60 < 0.25, f"false-alarm rate {flagged / 60:.1%} looks uncorrected"


def test_a_wide_interval_is_not_evidence_of_a_constant_hazard():
    """With few failures the interval spans 1 because the data cannot tell, not
    because the hazard was shown to be flat. The wording has to keep those apart."""
    rng = np.random.default_rng(9)
    result = po.calculate_weibull(rng.weibull(2.5, 8) * 100.0, n_bootstrap=200)

    if result["constant_hazard_is_valid"] is True:
        assert "cannot" in result["reason"].lower() or "not" in result["reason"].lower()


# ---- times between failures ----


def test_n_failures_on_one_machine_give_n_minus_one_intervals():
    work_orders = pd.DataFrame({
        "reported_at": [0.0, 100.0, 250.0, 400.0],
        "wo_type": ["breakdown"] * 4,
    })

    result = po.calculate_time_between_failures(work_orders)

    assert result["n_failures"] == 4
    assert result["n_intervals"] == 3
    assert result["intervals"] == [100.0, 150.0, 150.0]


def test_a_gap_is_never_measured_across_two_machines():
    """The time from machine A failing to machine B failing is not a time between
    failures of anything."""
    work_orders = pd.DataFrame({
        "machine_id": ["A", "B", "A", "B"],
        "reported_at": [0.0, 10.0, 100.0, 200.0],
        "wo_type": ["breakdown"] * 4,
    })

    result = po.calculate_time_between_failures(work_orders, asset_col="machine_id")

    assert result["n_intervals"] == 2
    assert sorted(result["intervals"]) == [100.0, 190.0]


def test_planned_work_is_not_a_failure():
    work_orders = pd.DataFrame({
        "reported_at": [0.0, 50.0, 100.0, 200.0],
        "wo_type": ["breakdown", "planned", "inspection", "breakdown"],
    })

    result = po.calculate_time_between_failures(work_orders)

    assert result["n_failures"] == 2
    assert result["intervals"] == [200.0]


def test_datetime_work_orders_come_back_in_hours():
    base = pd.Timestamp("2026-01-01")
    work_orders = pd.DataFrame({
        "reported_at": [base, base + pd.Timedelta(hours=12), base + pd.Timedelta(hours=36)],
        "wo_type": ["breakdown"] * 3,
    })

    result = po.calculate_time_between_failures(work_orders)

    assert result["unit"] == "hours"
    assert result["intervals"] == [12.0, 24.0]


# ---- calculate_mtbf owns up to its assumption ----


def test_mtbf_says_that_its_failure_rate_assumes_a_constant_hazard():
    work_orders = pd.DataFrame({"wo_type": ["breakdown"] * 4})

    result = po.calculate_mtbf(work_orders, operating_hours=1000.0)

    assert result["failure_rate_assumes_constant_hazard"] is True
    assert "calculate_weibull" in result["constant_hazard_note"]


def test_mtbf_keeps_every_figure_it_reported_before():
    work_orders = pd.DataFrame({"wo_type": ["breakdown"] * 4})

    result = po.calculate_mtbf(work_orders, operating_hours=1000.0)

    assert result["mtbf_hours"] == 250.0
    assert result["breakdowns"] == 4
    assert result["operating_hours"] == 1000.0
    assert result["failure_rate_per_hour"] == pytest.approx(1 / 250.0)
