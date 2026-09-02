from __future__ import annotations

# Production guardrail and anomaly detection thresholds
MIN_TRAIN_ROWS = 10
MISSING_SCHEMA_WARN_RATIO = 0.5
OUT_OF_BOUNDS_WARN_RATIO = 0.10

# Population Stability Index bands (industry standard for covariate shift)
PSI_MODERATE_SHIFT = 0.10
PSI_MAJOR_SHIFT = 0.25
PSI_DEFAULT_BINS = 10
DRIFT_MIN_ROWS = 30
DRIFT_NOISE_SIGMAS = 3.0
PSI_MAX_CATEGORIES = 50

# Control-chart defaults for condition monitoring.
# EWMA lambda 0.2 with 3-sigma limits is the standard compromise: small enough to
# accumulate a slow drift, large enough not to chase noise.
EWMA_DEFAULT_LAMBDA = 0.2
EWMA_DEFAULT_SIGMAS = 3.0
# CUSUM k=0.5 sigma / h=5 sigma is the classic pairing tuned to detect a
# sustained one-sigma shift quickly.
CUSUM_DEFAULT_SLACK = 0.5
CUSUM_DEFAULT_DECISION = 5.0
# d2 for subgroups of size 2, used to turn an average moving range into a sigma
# estimate for individual measurements.
MOVING_RANGE_D2 = 1.128
# Above this absolute lag-1 autocorrelation, moving-range sigma is no
# longer a safe spread estimate - consecutive readings repeat each other,
# the moving range collapses, and the chart alarms on nearly every point.
AUTOCORRELATION_WARN = 0.5

# Statistical Process Control rules (Western Electric subset and full Nelson 1-8 rules)
CONTROL_RULES_WESTERN_ELECTRIC = (1, 2, 5, 6)
CONTROL_RULES_NELSON = (1, 2, 3, 4, 5, 6, 7, 8)

CONTROL_RULE_DESCRIPTIONS: dict[int, str] = {
    1: "One point beyond 3 sigma from centre (outlier / large shift)",
    2: "Nine consecutive points on the same side of centre (mean shift)",
    3: "Six consecutive points steadily increasing or decreasing (trend)",
    4: "Fourteen consecutive points alternating up and down (systematic variation)",
    5: "Two out of three consecutive points beyond 2 sigma on the same side (moderate shift)",
    6: "Four out of five consecutive points beyond 1 sigma on the same side (small shift)",
    7: "Fifteen consecutive points within 1 sigma of centre (stratification / reduced variation)",
    8: "Eight consecutive points beyond 1 sigma on either side with none in Zone C (mixture / two populations)",
}


# Data Quality Score dimension weights (Completeness, Consistency, Validity,
# Uniqueness, Timeliness). Timeliness is dropped and the rest renormalised when
# the dataset has no datetime column.
DQS_WEIGHTS = {
    "completeness": 0.30,
    "consistency": 0.25,
    "validity": 0.20,
    "uniqueness": 0.15,
    "timeliness": 0.10,
}
DQS_PRODUCTION_READY = 85.0
DQS_USABLE = 65.0

# Placeholder strings that masquerade as data. Completeness metrics lie until
# these are counted as missing.
SILENT_NULL_TOKENS = frozenset({
    "", "-", "--", "?", "n/a", "na", "n.a.", "null", "none", "nil",
    "nan", "missing", "unknown", "undefined", "#n/a", "#value!", "#div/0!"
})

# Values industrial sensors and PLCs commonly write on a fault condition.
# Reported only, never auto-converted: a real reading could legitimately be -999.
NUMERIC_SENTINELS = (-999, -9999, -99999, 999999, -1e30, 1e30)

# Iglewicz-Hoaglin modified Z-score cut-off for outlier flagging
MODIFIED_ZSCORE_THRESHOLD = 3.5

# Share of the total that the Pareto "vital few" cut-off is drawn at.
PARETO_CUTOFF = 0.80

# The conventional world-class OEE benchmark, reported for context only.
OEE_WORLD_CLASS = 0.85

# The seed used when the caller does not choose one. A fixed default is right -
# an unseeded run is not reproducible and cannot be defended in a report - but it
# was previously hard-coded at every call site, which made the seed invisible and
# left every published figure a single-seed figure with no way to ask how much of
# it was the seed. It is a parameter now; see run_seed_sweep().
DEFAULT_RANDOM_STATE = 42

# Seeds run by run_seed_sweep() when the caller does not supply a list. Five is a
# compromise: enough spread to see whether a result is stable, few enough that the
# sweep still finishes on the low-spec hardware this library targets.
SEED_SWEEP_DEFAULT = (0, 1, 2, 3, 4)

# Bins used to sort predicted probabilities before calibration is measured, and the
# Expected Calibration Error above which the probabilities should not be read as
# probabilities. 0.05 is a convention, not a law - it is reported alongside the raw
# number so the reader can apply their own line.
CALIBRATION_DEFAULT_BINS = 10
CALIBRATION_ECE_LIMIT = 0.05

# The four criteria calculate_capability() uses to decide whether a process is
# stable enough for Cp/Cpk to describe anything. None of them is "any control rule
# fired": measured on healthy in-control data, the Western Electric set signals at
# least once on 30.5% of 50-point series and 99.5% of 1,000-point ones, so a gate
# built on it would reject almost every real data set.
#
# Every figure below comes from `measure_gate_history.py`, run against
# calculate_capability() itself over 1,000 trials per cell with fixed seeds, all
# arms scored on the same series. Lengths are n = 50 / 100 / 200 / 400 / 1000.

# 1. Variation arriving BETWEEN subgroups rather than within them. On healthy data
# the ratio sits at 0.99-1.00 whatever the length. Blind to a variance change,
# which lifts sigma_within and sigma_overall together and leaves the ratio near 1.
CAPABILITY_SIGMA_RATIO_LIMIT = 1.20

# 2. Points beyond 3 sigma at a rate chance does not explain. Chance puts 0.27% of
# points outside, so this covers the ratio's blind spot: it is the only criterion
# that sees a mid-series variance doubling, at 67.4% by n=1000.
CAPABILITY_OUTLIER_RATE_LIMIT = 0.01

# ...but a fraction has no resolution on a short series: at n=50 a single point
# beyond 3 sigma is already 2%, and chance alone puts at least one point out there
# 12.6% of the time - 1 - (1 - 0.0027)^50. So the COUNT must also be more than
# chance explains, tested against Binomial(n, NORMAL_TAIL_BEYOND_3_SIGMA). Practical
# significance and statistical significance both required, the same shape
# check_asset_drift uses for small batches.
#
# It binds only where the fraction is too coarse - at n=1000 the rate limit already
# demands 11 points where chance produces 2.7 - and the measurement says so:
#   healthy, rate alone   17.4 / 3.4 / 3.0 / 0.5 / 0.0%
#   healthy, with Binomial 4.6 / 3.4 / 3.0 / 0.5 / 0.0%
# Only the 50-point column moves. What that costs, also only at n=50: a variance
# doubling is caught 13.4% of the time instead of 46.3%, and a 1-sigma drift 33.7%
# instead of 42.2%. Both older figures were bought at a 17.4% false-alarm rate on
# healthy data, so the 50-point verdict was closer to a coin flip than a reading.
NORMAL_TAIL_BEYOND_3_SIGMA = 0.0027
CAPABILITY_OUTLIER_ALPHA = 0.05

# 3. A sustained drift, which neither criterion above can see: it widens no moving
# range and puts no single point out of bounds. The EWMA accumulates a shift too
# small to break a 3-sigma limit until the chart crosses. lambda 0.1 is the standard
# choice for a shift of about 1 sigma.
#
# The limit is a RATE, not "the EWMA signalled at least once". At this lambda that
# form fires on 4.1 / 7.6 / 18.1 / 34.3 / 66.1% of healthy series - climbing with
# the amount of data until it condemns everything, the same length-dependent trap
# that disqualified the Western Electric set.
#
# Adding it (measured with the outlier rate as a bare fraction, which is what the
# gate had at the time):
#   healthy       15.8 / 2.2 / 2.3 / 0.4 / 0.0%  ->  16.5 / 2.9 / 3.0 / 0.5 / 0.0%
#   1-sigma drift 20.6 / 7.4 / 4.1 / 1.8 / 0.2%  ->  25.2 / 27.0 / 43.3 / 58.1 / 80.1%
# The direction is what matters: detection had been FALLING as data accumulated,
# because a longer drift is more thoroughly absorbed into sigma_overall.
CAPABILITY_TREND_LAMBDA = 0.10
CAPABILITY_TREND_RATE_LIMIT = 0.03

# All four criteria are calibrated against a sigma estimated from the whole series.
# `baseline_n` estimates it from the first m points instead, which is unbiased but
# noisy - and a criterion compared against a noisy ruler reads the noise. On 300
# in-control points at m = 20 / 30 / 40 / 60 / 100 / 150 the gate called healthy
# series unstable 40.4 / 28.4 / 24.8 / 15.8 / 7.4 / 4.8% of the time.
#
# So the sigma used FOR TESTING is widened by 1 + k / sqrt(m). It never touches the
# reported sigma_within, and therefore never moves Cp, Cpk, Pp or Ppk. With k = 2.0,
# both arms on the same series:
#   healthy, before  40.4 / 28.4 / 24.8 / 15.8 / 7.4 / 4.8%
#   healthy, after    7.0 /  2.4 /  2.2 /  1.2 / 0.4 / 0.0%
# With the instability starting when the window closes: a 2-sigma step is still
# caught 100% of the time at every window; a variance doubling goes from 99.6% to
# 81.4% at m=20 and from 100% to 95.0% at m=40; a 1-sigma drift from 94.8% to 68.2%
# and from 97.8% to 79.6%. Those detections had been bought at 40.4% and 24.8%
# false-alarm rates.
#
# m=20 still runs hot at 7.0%. Twenty points cannot pin down a sigma, and no
# correction here can invent the information; the number is stated, not smoothed.
CAPABILITY_BASELINE_INFLATION_K = 2.0

# 4. A straight line fitted across the series. The EWMA cannot see a half-sigma
# drift at any lambda - centred on its own mean such a drift never leaves +/-0.25
# sigma, against an EWMA limit of 0.688 sigma - while a slope test carries t = 4.56
# on the same data at n=1000 (and t = 1.02 at n=50, where it really is out of reach).
#
# A significant slope alone is not enough: a long enough series makes a negligible
# slope significant. The fitted total drift must ALSO exceed a stated size, which is
# what makes this a statement about the process rather than about the sample.
#
# Adding it to the three above:
#   healthy        3.4 / 2.9 / 3.0 / 0.5 / 0.0%  ->  4.6 / 3.4 / 3.0 / 0.5 / 0.0%
#   1-sigma drift 14.0 / 27.0 / 43.3 / 58.1 / 80.1%  ->  33.7 / 64.4 / 84.2 / 92.8 / 98.6%
# At n >= 200 the false-alarm rate does not move at all; the whole cost is 1.2 points
# at n=50 and 0.5 at n=100. A 2-sigma step reaches 100% at every length.
#
# A 0.5-sigma drift is still called only 4.0% of the time at n=1000, and that is a
# CHOICE rather than a blind spot: it sits below the size this limit calls worth
# reporting. Lowering CAPABILITY_TREND_DRIFT_SIGMAS to 0.5 raises that to 53.0%
# while the healthy rate at n=200 goes from 3.0% to 3.7%.
CAPABILITY_TREND_DRIFT_SIGMAS = 0.75
CAPABILITY_TREND_ALPHA = 0.01
