"""
Tests for maintenance planning math functions:
- calculate_crow_amsaa: NHPP trend test for repairable systems
- forecast_failure_count: Homogeneous Poisson forecast with prediction interval
- calculate_optimal_pm_interval: Age-replacement policy for Weibull wear-out
"""

import itertools
import json

import numpy as np
import pytest
from scipy.integrate import quad

import potatopt as po

# ==============================================================================
# Crow-AMSAA Tests
# ==============================================================================


def test_crow_amsaa_hand_computed():
    """1. Hand-computed case: verify beta and lambda_ formulas against analytical values."""
    times = [10, 30, 60, 100, 150, 200]
    obs_end = 250.0
    n = len(times)
    s_val = sum(np.log(obs_end / np.array(times, dtype=float)))
    expected_beta = n / s_val
    expected_lambda = n / (obs_end ** expected_beta)

    result = po.calculate_crow_amsaa(times, observation_end=obs_end)

    assert result["error"] is None
    assert result["beta"] == pytest.approx(expected_beta)
    assert result["lambda_"] == pytest.approx(expected_lambda)
    assert result["n_failures"] == n
    assert result["dropped"] == 0


def test_crow_amsaa_false_alarm_rate_constant_process():
    """2. False-alarm rate on a constant-rate Poisson process (exact conditional CI should flag ~5%)."""
    rng = np.random.default_rng(42)
    replications = 2000
    flagged = 0
    obs_end = 365.0
    for _ in range(replications):
        times = rng.uniform(0.0, obs_end, size=10)
        res = po.calculate_crow_amsaa(times, observation_end=obs_end)
        if res["trend"] != "no_evidence":
            flagged += 1

    frac = flagged / replications
    assert 0.03 <= frac <= 0.07, f"Measured false-alarm rate was {frac:.4f} (flagged {flagged}/{replications})"


def test_crow_amsaa_power_worsening_and_improving():
    """3. Detection power on worsening (beta=2.5) and improving (beta=0.4) processes."""
    rng = np.random.default_rng(42)
    obs_end = 365.0
    n = 15
    replications = 500

    # Worsening process: beta = 2.5
    worsening_count = 0
    for _ in range(replications):
        u = rng.uniform(0.0, 1.0, size=n)
        times = obs_end * (u ** (1.0 / 2.5))
        res = po.calculate_crow_amsaa(times, observation_end=obs_end)
        if res["trend"] == "worsening":
            worsening_count += 1
    frac_worsening = worsening_count / replications
    assert frac_worsening > 0.5, f"Worsening fraction was {frac_worsening:.4f}, expected > 0.5"

    # Improving process: beta = 0.4
    improving_count = 0
    for _ in range(replications):
        u = rng.uniform(0.0, 1.0, size=n)
        times = obs_end * (u ** (1.0 / 0.4))
        res = po.calculate_crow_amsaa(times, observation_end=obs_end)
        if res["trend"] == "improving":
            improving_count += 1
    frac_improving = improving_count / replications
    assert frac_improving > 0.5, f"Improving fraction was {frac_improving:.4f}, expected > 0.5"


def test_crow_amsaa_refusals():
    """4. Refusals on too few failures, invalid observation_end, invalid confidence, and dropping out-of-range times."""
    # Fewer than min_failures
    res_few = po.calculate_crow_amsaa([10.0, 20.0], observation_end=100.0, min_failures=5)
    assert res_few["error"] is not None
    assert res_few["beta"] is None

    # observation_end <= 0
    res_obs = po.calculate_crow_amsaa([10.0, 20.0, 30.0, 40.0, 50.0], observation_end=0.0)
    assert res_obs["error"] is not None
    assert res_obs["beta"] is None

    # confidence = 1.0
    res_conf = po.calculate_crow_amsaa([10.0, 20.0, 30.0, 40.0, 50.0], observation_end=100.0, confidence=1.0)
    assert res_conf["error"] is not None
    assert res_conf["beta"] is None

    # Values > observation_end are dropped and counted in dropped
    times_with_excess = [10.0, 20.0, 30.0, 40.0, 50.0, 150.0, 200.0]
    res_drop = po.calculate_crow_amsaa(times_with_excess, observation_end=100.0)
    assert res_drop["error"] is None
    assert res_drop["n_failures"] == 5
    assert res_drop["dropped"] == 2


def test_crow_amsaa_json_safe():
    """5. json.dumps(result, allow_nan=False) succeeds for both success and refusal results."""
    res_ok = po.calculate_crow_amsaa([10.0, 20.0, 30.0, 40.0, 50.0], observation_end=100.0)
    dumped_ok = json.dumps(res_ok, allow_nan=False)
    assert isinstance(dumped_ok, str)

    res_err = po.calculate_crow_amsaa([10.0], observation_end=100.0)
    dumped_err = json.dumps(res_err, allow_nan=False)
    assert isinstance(dumped_err, str)


# ==============================================================================
# Poisson Forecast Tests
# ==============================================================================


def test_forecast_failure_count_normal():
    """6. Standard forecast: expected value, interval coverage, and prob_at_least_one."""
    res = po.forecast_failure_count(12, 365.0, 30.0)
    assert res["error"] is None
    assert res["expected"] == pytest.approx(12.0 / 365.0 * 30.0)
    assert res["lower"] <= res["expected"] <= res["upper"]
    assert 0.0 < res["prob_at_least_one"] < 1.0
    assert isinstance(res["lower"], int)
    assert isinstance(res["upper"], int)


def test_forecast_failure_count_zero():
    """7. Zero failures observed: expected, lower, upper are 0, note mentions 'not a zero risk'."""
    res = po.forecast_failure_count(0, 365.0, 30.0)
    assert res["error"] is None
    assert res["expected"] == 0.0
    assert res["lower"] == 0
    assert res["upper"] == 0
    assert "not a zero risk" in res["note"]


def test_forecast_coverage_check():
    """8. Coverage check for rate 0.05/day and horizon 30 over 3000 Poisson simulations (>= 80%)."""
    rng = np.random.default_rng(42)
    forecast = po.forecast_failure_count(45, 900.0, 30.0)
    assert forecast["error"] is None
    lower = forecast["lower"]
    upper = forecast["upper"]

    outcomes = rng.poisson(1.5, size=3000)
    frac_covered = float(np.mean((outcomes >= lower) & (outcomes <= upper)))
    assert frac_covered >= 0.80, f"Coverage was {frac_covered:.4f}, expected >= 0.80"


def test_forecast_failure_count_refusals():
    """9. Refusals on negative count, True boolean, NaN lookback, 0 horizon, and 0 interval."""
    assert po.forecast_failure_count(-1, 365.0, 30.0)["error"] is not None
    assert po.forecast_failure_count(True, 365.0, 30.0)["error"] is not None
    assert po.forecast_failure_count(10, float("nan"), 30.0)["error"] is not None
    assert po.forecast_failure_count(10, 365.0, 0.0)["error"] is not None
    assert po.forecast_failure_count(10, 365.0, 30.0, interval=0.0)["error"] is not None


# ==============================================================================
# Optimal PM Interval Tests
# ==============================================================================


def test_optimal_pm_interval_beta_at_or_below_one():
    """10. beta <= 1.0 yields interval=None, worthwhile=False, error=None."""
    for beta_val in (1.0, 0.7):
        res = po.calculate_optimal_pm_interval(
            beta=beta_val, eta=100.0, cost_planned=1.0, cost_breakdown=10.0
        )
        assert res["interval"] is None
        assert res["worthwhile"] is False
        assert res["error"] is None
        assert "not increasing" in res["reason"]


def test_optimal_pm_interval_breakdown_leq_planned():
    """11. cost_breakdown <= cost_planned yields interval=None, worthwhile=False, error=None."""
    res = po.calculate_optimal_pm_interval(
        beta=2.5, eta=100.0, cost_planned=10.0, cost_breakdown=5.0
    )
    assert res["interval"] is None
    assert res["worthwhile"] is False
    assert res["error"] is None
    assert "breakdown costs no more" in res["reason"].lower()


def test_optimal_pm_interval_brute_force_check():
    """12. Brute-force verification against numerical integration with scipy.integrate.quad."""
    beta = 2.5
    eta = 100.0
    cost_planned = 1.0
    cost_breakdown = 10.0

    def r_surv(t: float) -> float:
        return float(np.exp(-((t / eta) ** beta)))

    t_vals = np.linspace(1.0, 300.0, 2000)
    c_vals = []
    for t in t_vals:
        m_t, _ = quad(r_surv, 0.0, t)
        r_t = r_surv(t)
        c_t = (cost_planned * r_t + cost_breakdown * (1.0 - r_t)) / m_t
        c_vals.append(c_t)

    brute_min_cost = min(c_vals)
    brute_opt_t = t_vals[int(np.argmin(c_vals))]

    res = po.calculate_optimal_pm_interval(
        beta=beta, eta=eta, cost_planned=cost_planned, cost_breakdown=cost_breakdown
    )

    assert res["error"] is None
    assert res["worthwhile"] is True
    assert 0.0 < res["prob_failure_before_pm"] < 1.0
    assert res["cost_rate"] == pytest.approx(brute_min_cost, rel=0.005)
    assert res["interval"] == pytest.approx(brute_opt_t, rel=0.02)


def test_optimal_pm_interval_monotonicity():
    """13. Higher breakdown-to-planned cost ratio requires a shorter PM interval."""
    res_ratio20 = po.calculate_optimal_pm_interval(
        beta=2.5, eta=100.0, cost_planned=1.0, cost_breakdown=20.0
    )
    res_ratio3 = po.calculate_optimal_pm_interval(
        beta=2.5, eta=100.0, cost_planned=1.0, cost_breakdown=3.0
    )
    assert res_ratio20["interval"] is not None
    assert res_ratio3["interval"] is not None
    assert res_ratio20["interval"] < res_ratio3["interval"]


def test_optimal_pm_interval_low_saving_not_worthwhile():
    """14. Small savings (< 5%) result in worthwhile=False."""
    res = po.calculate_optimal_pm_interval(
        beta=1.05, eta=100.0, cost_planned=1.0, cost_breakdown=2.0
    )
    assert res["interval"] is not None
    assert res["worthwhile"] is False
    assert res["error"] is None
    assert "below the 5.0% threshold" in res["reason"]


def test_optimal_pm_interval_refusals_and_json_safety():
    """15. Refusals on invalid eta, NaN beta; verify JSON serializability with allow_nan=False."""
    res_eta = po.calculate_optimal_pm_interval(
        beta=2.5, eta=0.0, cost_planned=1.0, cost_breakdown=10.0
    )
    assert res_eta["error"] is not None
    assert res_eta["interval"] is None

    res_nan = po.calculate_optimal_pm_interval(
        beta=float("nan"), eta=100.0, cost_planned=1.0, cost_breakdown=10.0
    )
    assert res_nan["error"] is not None
    assert res_nan["interval"] is None

    # Verify JSON serializability with allow_nan=False
    res_success = po.calculate_optimal_pm_interval(
        beta=2.5, eta=100.0, cost_planned=1.0, cost_breakdown=10.0
    )
    assert isinstance(json.dumps(res_success, allow_nan=False), str)
    assert isinstance(json.dumps(res_eta, allow_nan=False), str)
    assert isinstance(json.dumps(res_nan, allow_nan=False), str)


# ==============================================================================
# Weibull Curves, PM Cost Curve, and Forecast Distribution Tests
# ==============================================================================


def test_weibull_curves_properties():
    """16. Weibull curves: reliability decreasing, pdf integrates to ~1 - R(t_max), hazard shapes."""
    res = po.calculate_weibull_curves(beta=2.5, eta=100.0, points=100)
    assert res["error"] is None
    # 1. reliability(t) decreasing in t
    rel = res["reliability"]
    assert all(r1 >= r2 for r1, r2 in itertools.pairwise(rel))

    # 2. pdf integrates to ~1 - R(t_max) over grid (trapezoid, rel tol 2%)
    t = res["t"]
    pdf = res["pdf"]
    trapz_integral = sum((t2 - t1) * (p1 + p2) / 2.0 for t1, t2, p1, p2 in zip(t[:-1], t[1:], pdf[:-1], pdf[1:]))
    r_tmax = rel[-1]
    expected_integral = 1.0 - r_tmax
    assert trapz_integral == pytest.approx(expected_integral, rel=0.02)

    # 3. hazard rising for beta 2.5, falling for 0.6, flat constant for 1.0
    res_rising = po.calculate_weibull_curves(beta=2.5, eta=100.0)
    assert res_rising["hazard_shape"] == "rising"
    assert all(h1 <= h2 for h1, h2 in zip(res_rising["hazard"][:-1], res_rising["hazard"][1:]))

    res_falling = po.calculate_weibull_curves(beta=0.6, eta=100.0)
    assert res_falling["hazard_shape"] == "falling"
    assert all(h1 >= h2 for h1, h2 in zip(res_falling["hazard"][:-1], res_falling["hazard"][1:]))

    res_flat = po.calculate_weibull_curves(beta=1.0, eta=100.0)
    assert res_flat["hazard_shape"] == "flat"
    assert all(h == pytest.approx(1.0 / 100.0) for h in res_flat["hazard"])


def test_weibull_curves_refusals_and_json():
    """17. Weibull curves refusals for eta <= 0, points = 5, and JSON serialization."""
    res_eta = po.calculate_weibull_curves(beta=2.5, eta=0.0)
    assert res_eta["error"] is not None
    assert res_eta["t"] == []

    res_pts = po.calculate_weibull_curves(beta=2.5, eta=100.0, points=5)
    assert res_pts["error"] is not None
    assert res_pts["t"] == []

    res_ok = po.calculate_weibull_curves(beta=2.5, eta=100.0)
    assert isinstance(json.dumps(res_ok, allow_nan=False), str)
    assert isinstance(json.dumps(res_eta, allow_nan=False), str)
    assert isinstance(json.dumps(res_pts, allow_nan=False), str)


def test_pm_cost_curve_and_json():
    """18. PM cost curve: minimum within 1% of optimal PM interval cost rate, every value > 0, JSON safe."""
    beta = 2.5
    eta = 100.0
    cp = 1.0
    cb = 10.0
    opt_res = po.calculate_optimal_pm_interval(beta=beta, eta=eta, cost_planned=cp, cost_breakdown=cb)
    assert opt_res["error"] is None

    cost_res = po.calculate_pm_cost_curve(beta=beta, eta=eta, cost_planned=cp, cost_breakdown=cb, points=80)
    assert cost_res["error"] is None
    assert all(c > 0.0 for c in cost_res["cost_rate"])
    min_cost = min(cost_res["cost_rate"])
    assert min_cost == pytest.approx(opt_res["cost_rate"], rel=0.01)

    assert isinstance(json.dumps(cost_res, allow_nan=False), str)
    err_res = po.calculate_pm_cost_curve(beta=beta, eta=-1.0, cost_planned=cp, cost_breakdown=cb)
    assert err_res["error"] is not None
    assert isinstance(json.dumps(err_res, allow_nan=False), str)


def test_forecast_distribution():
    """19. Forecast failure distribution: sum >= 0.99 for expected 1.5, 1.0 at k=0 for zero failures."""
    res_15 = po.forecast_failure_count(45, 900.0, 30.0)
    assert res_15["error"] is None
    assert "distribution" in res_15
    prob_sum = sum(d["probability"] for d in res_15["distribution"])
    assert prob_sum >= 0.99

    res_zero = po.forecast_failure_count(0, 365.0, 30.0)
    assert res_zero["error"] is None
    assert res_zero["distribution"][0]["k"] == 0
    assert res_zero["distribution"][0]["probability"] == 1.0
    for d in res_zero["distribution"][1:]:
        assert d["probability"] == 0.0

    assert isinstance(json.dumps(res_15, allow_nan=False), str)
    assert isinstance(json.dumps(res_zero, allow_nan=False), str)
    err_fc = po.forecast_failure_count(-1, 365.0, 30.0)
    assert err_fc["distribution"] == []
    assert isinstance(json.dumps(err_fc, allow_nan=False), str)

