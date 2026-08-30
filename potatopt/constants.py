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

# The two limits calculate_capability() uses to decide whether a process is stable
# enough for Cp/Cpk to describe anything. Neither is "any control rule fired":
# measured on healthy in-control data, the Western Electric set signals at least
# once on 30.5% of 50-point series and 99.5% of 1,000-point series, so a gate
# built on it would reject almost every real data set.
#
# sigma_overall / sigma_within above this means the variation is arriving BETWEEN
# subgroups rather than within them. On healthy data the ratio sits at 0.99-1.00
# at every length tested; 1.20 fired on 0.0% of 400 trials at n >= 100, while
# catching a 3-sigma drift 97.7% of the time and a 2-sigma step 99.3%.
CAPABILITY_SIGMA_RATIO_LIMIT = 1.20

# Fraction of points beyond 3 sigma that chance no longer explains. This covers
# the ratio's blind spot - a variance change lifts sigma_within and sigma_overall
# together and leaves the ratio near 1. Chance puts 0.27% of points outside; this
# limit fires on 1.0-2.7% of healthy series and on 47.3% of series whose variance
# doubles halfway through, where the ratio alone manages 0.7%.
CAPABILITY_OUTLIER_RATE_LIMIT = 0.01
