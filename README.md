# PotatOpt-ML 🥔

**Predictive maintenance that runs on a potato.**

[![CI](https://github.com/Oak04K/PotatOpt-ML/actions/workflows/ci.yml/badge.svg)](https://github.com/Oak04K/PotatOpt-ML/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-1.6.0-blue.svg)](CHANGELOG.md)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Core install](https://img.shields.io/badge/core%20install-4%20packages-orange.svg)](#why-it-is-built-this-way)

A machine-learning library for **condition-based and predictive maintenance**.
Sensor readings from factory equipment go in; a maintenance decision, and what
that decision is worth in money, come out. CPU only, no GPU. It is meant to be
driven either by an engineer on the shop floor or by an AI agent, in as few
lines as possible.

[อ่านเอกสารฉบับภาษาไทย (ละเอียดกว่า)](README.th.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

---

## Install

```bash
pip install potatopt                    # the four-package core
pip install "potatopt[automl]"          # + FLAML, needed for .fit()
pip install "potatopt[automl,xai]"      # + SHAP explanations
pip install "potatopt[mcp]"             # + the MCP server, to drive it from an AI agent
pip install "potatopt[all]"             # everything
```

## The shortest path

If you only ever read one section, read this one. A single call runs the whole
job and returns one dictionary:

```python
import potatopt as po

report = po.auto_analyze("machine_sensors.csv", target="failure")

print(report["metrics"]["f1"])            # how good the model is
print(report["cost"]["cost_savings"])     # the same answer, in money
print(report["top_features"])             # what drove it
```

`auto_analyze()` audits data quality, splits the data three ways, searches for
a model, tunes the decision threshold on the **validation** set, scores the
untouched test set, converts the result to money, and ranks the features.

It returns one JSON-ready dictionary and **never raises**. A failure comes back
as `{"ok": False, "error": "..."}` with a sentence you can actually read.

---

## The working path

The rest of the library is the same job taken one question at a time, in the
order those questions have to be asked. Each stage assumes the stage above it
was answered; skipping ahead produces numbers that look fine and mean nothing.

`examples/tour.py` walks all of it end to end on simulated data in about a
minute, and a test fails if the tour ever falls behind the public API.

| # | Question | Module | Key functions |
|---|---|---|---|
| 1 | Can I trust this data? | `data.py` | `inspect_data`, `audit_data_quality`, `detect_silent_nulls`, `detect_outliers`, `calculate_correlations` |
| 2 | Can I even measure it? | `quality.py` | `calculate_gauge_rr` |
| 3 | Is the process stable? | `spc.py` | `calculate_ewma_chart`, `calculate_cusum_chart`, `calculate_control_rules` |
| 4 | Is it capable? | `quality.py` | `calculate_capability` |
| 5 | Has it drifted since? | `drift.py` | `check_data_drift`, `check_asset_drift`, `calculate_psi` |
| 6 | What is it costing me? | `reliability.py` | `calculate_mtbf`, `calculate_mttr`, `calculate_availability`, `calculate_oee`, `calculate_pareto` |
| 7 | Can a model help? | `engine.py`, `analysis.py`, `calibration.py` | `PotatOptEngine`, `auto_analyze`, `check_calibration`, `run_seed_sweep` |

### Stage 1 — Can I trust this data?

```python
po.inspect_data(df, target_col="failure")   # shape, types, imbalance, suggested task
po.audit_data_quality(df)                   # a 0-100 score across five dimensions
po.detect_silent_nulls(df)                  # "N/A", "-", "null", and sensor codes like -999
po.detect_outliers(df, "temperature")       # modified z-score, robust to the outliers it hunts
po.calculate_correlations(df)               # which sensors carry the same information
```

The data-quality score weighs completeness, consistency, validity, uniqueness
and timeliness. `detect_silent_nulls` matters more than it sounds: a sensor
that writes `-999` when it fails is not missing data as far as pandas is
concerned, so the mean quietly moves and nothing warns you.

`calculate_correlations` also reports which columns it **skipped and why**,
rather than silently omitting them from the matrix.

### Stage 2 — Can I even measure it?

```python
po.calculate_gauge_rr(readings, "part", "operator", "measurement")
```

Before judging a process, establish that the measurement system can tell two
parts apart. A capability figure computed through a gauge that contributes 40%
of the observed variation is describing the gauge, not the machine.

### Stage 3 — Is the process stable?

```python
po.calculate_ewma_chart(readings, baseline_n=25)     # exact time-varying limits
po.calculate_cusum_chart(readings, baseline_n=25)    # two-sided tabular CUSUM
po.calculate_control_rules(series, rules=po.CONTROL_RULES_WESTERN_ELECTRIC)
```

Sigma is estimated from the **moving range divided by d2 (1.128)**, never from
the sample standard deviation. This is the single most important decision in
the module. A degrading series inflates its own standard deviation, which
widens the control limits and hides the very degradation the chart exists to
catch. Measured on a wear ramp: SD 2.201 against a moving-range sigma of 0.222,
and EWMA signalled at **sample 3 instead of sample 13**.

The moving range assumes consecutive readings are independent, and a sensor
that behaves like a random walk breaks that assumption the other way: adjacent
readings echo each other, sigma collapses, and the chart alarms nearly
everywhere. Both charts therefore return `lag1_autocorrelation`, and raise
`autocorrelation_warning` above 0.5. On the AI4I 2020 log, 600 points with
`baseline_n=100`: process temperature had lag-1 **+0.919** and was flagged at
**573 of 600 points**; torque had lag-1 -0.051 and was flagged at **0 of 600**.

The check reads the baseline window only, never the whole series. A wear ramp
reads +0.998 across the whole series but +0.042 across its in-control window,
so measuring everything would fire the warning on exactly the case the chart is
for.

### Stage 4 — Is it capable?

```python
po.calculate_capability(series, usl=110, lsl=90)
```

Capability is only meaningful on a stable process, so this function reports on
that condition rather than assuming it. On a series drifting from 97 to 103,
`cpk` comes out at **11.4** — world-class by any table — while
`capability_is_meaningful` is `False`, with the reason attached: sigma_overall
is 6.11 times sigma_within, and 70% of points sit beyond 3 sigma of the
in-control spread.

The stability test is deliberately **not** "did any control rule fire". On
healthy in-control data, the four Western Electric rules signal at least once
on 30.5% of 50-point series and **99.5% of 1,000-point series**. A gate built
on that rejects nearly every real dataset, and the flag stops meaning anything.
Two length-independent limits are used instead — a sigma ratio above 1.20, or
more than 1% of points beyond 3 sigma — which fire on about 2% of healthy
series while still catching a 3-sigma drift 96% of the time and a 2-sigma step
every time. False alarms now fall as data accumulates instead of climbing
toward certainty.

Two blind spots are written down rather than left to be discovered: a
mid-series variance change is caught 44.5% of the time, and a slow 1-sigma
drift only 4%.

### Stage 5 — Has it drifted since?

```python
report = po.check_asset_drift(train_df, batch_df, asset_col="machine_id")
report["assets_drifted"]   # ['M-02']  - names the machine to go and look at
report["assets_skipped"]   # {'M-01': '10 batch / 200 train rows, below min_rows=30'}
```

Machines of the same model still differ; one runs hotter, one sits by a door.
Pooled into a single profile, that between-machine spread *becomes the ruler*,
and it fails in both directions at once:

| Situation | Pooled | Per asset |
|---|---|---|
| One machine offline for maintenance, **nothing else changed** | `True`, PSI 1.26 — a false alarm | `False`, M-01 marked `insufficient_data` |
| One machine genuinely +3 °C | magnitude 0.244, names no machine | magnitude **3.084**, names M-02 |

That is a **12.5x dilution**, because the pooled sigma of 4.18 contains the
spread between machines while only one machine of three actually moved.

Splitting by asset creates its own trap — smaller batches make statistics
unreliable — so it ships with two guards. PSI bins scale to the batch size, and
the mean-shift threshold is raised to
`max(threshold_pct, k * sqrt(1/n_batch + 1/n_train))`: practical significance
and statistical significance, both required. At 30 rows that cuts false alarms
from 39.3% to 0.7% while still catching every 1.0-sigma shift.

### Stage 6 — What is it costing me?

```python
po.calculate_mtbf(work_orders)          # reliability
po.calculate_mttr(work_orders)          # with waiting split from repairing
po.calculate_availability(...)          # inherent vs operational, and the gap
po.calculate_oee(...)                   # availability x performance x quality
po.calculate_pareto(work_orders, "failure_mode", value_col="downtime_hours")
```

`calculate_pareto` ranks the vital few by **count or by cost**, which are
rarely the same list. The most frequent failure mode is often cheap; the
expensive one happens twice a year.

### Stage 7 — Can a model help?

```python
X_train, X_val, X_test, y_train, y_val, y_test = po.split_data_three_way(df, "failure")

engine = po.PotatOptEngine(task="classification", time_budget=60).fit(X_train, y_train)
engine.optimize_maintenance_threshold(X_val, y_val)   # tuned on validation, not test
print(engine.evaluate(X_test, y_test))
print(engine.calculate_maintenance_cost(X_test, y_test))
```

`PotatOptEngine` handles encoding, scaling, imputation, collinear-feature
pruning, memory downcasting and the AutoML search. It subclasses scikit-learn's
`BaseEstimator`, so `cross_val_score`, `GridSearchCV` and `Pipeline` work on it
directly. Every stochastic step takes a `random_state` (default `42`), so a
reported score can be reproduced or swept.

Full method list: `fit`, `predict`, `predict_proba`, `evaluate`,
`optimize_threshold`, `optimize_maintenance_threshold`, `check_calibration`,
`calculate_cost_of_quality`, `calculate_maintenance_cost`,
`get_feature_importance`, `explain_predictions`, `detect_drift`,
`get_training_report`, `get_inference_health`, `save`, `load`.

---

## Why it is built this way

Three constraints shaped every design decision. All three are measured in the
test suite rather than claimed in prose.

| Goal | How it is met | Measured |
|---|---|---|
| **Few tokens to drive** | `auto_analyze()` does the whole pipeline in one call | **91.7% fewer tokens** than the equivalent scikit-learn pipeline (674 to 56, `tiktoken cl100k_base`) |
| **Small machines** | Ordinal encoding, dtype downcasting, tree models, no GPU | preprocessing cuts memory **84.5%** (2.71 MB to 0.42 MB); `fit()` held under a 400 MB ceiling |
| **Reads like a first Python lesson** | Flat functions, plain dictionaries, no framework to learn | see the shortest path, above |

The install really is small: **four packages** — `numpy`, `pandas`, `scipy`,
`scikit-learn`. FLAML and SHAP are optional extras loaded lazily, which keeps
**1.46 seconds (44%)** of import cost off the table until you actually train
something. CI verifies this in an environment where the extras are genuinely
absent.

### Costs are stated as maintenance, not accuracy

The baseline is **run to failure**: with no model, every failure becomes an
unplanned breakdown. A caught failure costs an inspection plus a planned
repair. A false alarm costs the inspection only, because an engineer looks
before replacing a part.

```python
po.calculate_maintenance_savings(true_positives=18, false_positives=25, false_negatives=2)
# {'cost_savings': 691500.0, 'savings_percentage': 69.15,
#  'breakdown_avoidance_rate': 0.9, 'unplanned_breakdowns': 2, ...}
```

`breakdown_avoidance_rate` is reported next to `cost_savings` because the two
disagree exactly where it matters:

| Model | Recall | Saving |
|---|---|---|
| Good model (TP 18, FP 25, FN 2) | 0.90 | **+691,500** |
| Flags every machine (TP 20, FP 980, FN 0) | **1.00** | **-660,000** |
| Flags nothing (FN 20) | 0.00 | 0 — exactly the baseline |

The second row is the whole point. Perfect recall, and it destroys money: 980
pointless call-outs cost more than the breakdowns they prevented. Break-even is
540 false alarms. **Recall cannot tell you this. Cost can.**

### Leakage is caught, not merely documented

- `fit()` sends `split_type="time"` for forecasting. FLAML's `"auto"` resolves
  to a random shuffle for regression tasks, which lets future rows train the
  model.
- `optimize_threshold()` fingerprints the rows it tuned on, and `evaluate()`
  returns `threshold_leakage_warning` if you then report results on those same
  rows.

### Silence is never an answer

Every bug this project found in itself returned a confident all-clear. They are
now impossible:

- A machine that stops reporting gets status `missing_from_batch`. A dead
  gateway and a healthy machine no longer look identical.
- A sensor column missing from the batch appears in `skipped_features`, rather
  than the old `drift_detected: False, max_psi: 0.0`.
- `min_rows` counts **non-null readings**, not rows. A column that is 98% NaN
  still has 200 rows; with 4 usable readings the false-alarm rate was 100%.
- `auto_analyze()` returns `top_features_note` when SHAP declines, instead of
  an empty list and no reason.
- `explain_predictions()` returns `additivity_check_relaxed`, so an approximate
  ranking can never pass for an exact one.

### Design guarantees

1. **Public functions return errors; they do not raise.** A bad argument comes
   back as `{"error": "..."}`. These functions are meant to sit behind a
   tool-calling layer where the caller may be a language model.
2. **Everything survives `json.dumps()`.** NaN and infinity become `None`,
   because `json.dumps` otherwise emits a bare `NaN`, which is not valid JSON.
3. **Nothing heavy is imported until it is used.** Enforced by a test that runs
   `import potatopt` in a clean subprocess.
4. **New arguments default to the previous behaviour.** Adding a feature never
   changes an existing call's result.
5. **Zero ruff errors, with no warning tier.**

---

## What is built on top

| | | |
|---|---|---|
| **`potatopt/`** | the library | Everything above. Every function returns a JSON-ready dictionary and never raises. |
| **`chart_engine.py`** | the figures | Draws the dictionaries the library returned. It calculates nothing, so a chart can never disagree with the number printed beside it. |
| **`potatopt/mcp_server.py`** | the AI adapter | Seven MCP tools over stdio, so an agent can drive all of it without writing Python and without factory data leaving the machine. |

```bash
python examples/tour.py       # every capability, on simulated data, in about a minute
python examples/quickstart.py # the predictive-maintenance job on the real AI4I dataset
```

### Charts that cannot disagree with the report

```python
import chart_engine as ce

ce.plot_ewma(po.calculate_ewma_chart(readings, baseline_n=25))
ce.plot_confusion_matrix(engine.evaluate(X_test, y_test))
ce.plot_feature_importance(engine.explain_predictions(X_test))
ce.plot_pareto(po.calculate_pareto(work_orders, "failure_mode", value_col="downtime_hours"))
```

Every function takes a dictionary some `potatopt` function already returned, so
a figure can be drawn months later from a saved JSON report with no raw data
present. matplotlib is imported lazily, so requiring the module costs nothing
to a caller who only wanted the numbers.

### Driving it from an AI agent

```bash
pip install "potatopt[automl,mcp]"
potatopt-mcp                             # or: python -m potatopt.mcp_server
```

An MCP server over stdio, so it runs as a child process of the client and
factory data never touches a network:

```json
{ "mcpServers": { "potatopt": { "command": "potatopt-mcp" } } }
```

**Seven tools, not fifty-seven.** A tool surface is spent from the agent's
context window before it answers anything, so the count is a budget. Names,
descriptions and schemas together measure **1,585 tokens**; at that average the
full public API would cost around 13,000.

The guarantee that public functions **return** errors rather than raise is what
makes them fit this transport. When a tool raises, the MCP SDK replaces the
exception with `UnexpectedToolError` and the reason is lost, leaving the agent
at a dead end. Here a missing file comes back as `{"ok": false, "error": "..."}`,
which it can read and act on.

Set `POTATOPT_MCP_ROOT` to confine file reads to one directory. Left unset, the
server reads whatever the user running it can read; that is a process
permission model, not a sandbox.

---

## Try it on real data

```bash
python examples/quickstart.py            # about a minute on a laptop CPU
```

This fetches the **AI4I 2020 Predictive Maintenance Dataset** (UCI, CC BY 4.0,
SHA-256 pinned, cached and gitignored) and walks the whole job in ten printed
steps. It starts by deleting five failure-mode columns, because those labels
only exist once the machine has already failed. On the held-out 2,000 rows it
avoids 61 of 68 breakdowns and saves **2,269,500** against run-to-failure, at
the cost of 134 wasted inspections.

`python benchmarks/runtime_cost.py` measures what that costs to run — memory,
time and money — against the same pipeline written by hand in scikit-learn.

## Development

```bash
pip install -e ".[automl,xai,viz,mcp,dev]"   # the same set CI installs
python -m ruff check potatopt chart_engine.py tests benchmarks scripts examples
python -m pytest tests -q
```

CI runs the suite on Python 3.10 through 3.14 and on Windows, verifies the
core-only install, and builds the distribution. Every Python version claimed in
`pyproject.toml` is exercised by CI, and a test fails if those two lists ever
disagree.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the design constraints and the
measure-before-you-specify habit this project runs on.

## About

Built as an Industrial Engineering undergraduate thesis: DMAIC framing, SPC
comparison, SHAP mapped to Fishbone analysis, and every result converted to
money rather than left as a metric. The name is the design philosophy — a model
that runs on a potato.

MIT licensed. See [LICENSE](LICENSE).
