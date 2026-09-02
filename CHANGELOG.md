# Changelog

All notable changes to PotatOpt-ML are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Numbers quoted below were measured, not estimated. Where a change exists because
a specific failure was observed, that failure is named — a changelog that says
only *what* changed makes the same mistake easy to reintroduce.

---

## [1.7.0] - 2026-09-02

The stability gate inside `calculate_capability()` went from two criteria to four,
and the two it already had were repaired. `stable` and `capability_is_meaningful`
therefore change value on data that used to pass, which is why this is a minor
version and not a patch.

**Every figure below was measured through `calculate_capability()` itself over
1,000 trials per cell with fixed seeds, all arms scored on the same series, at
n = 50 / 100 / 200 / 400 / 1000.** Where a change costs detection, the cost is
stated beside the gain.

### Changed

- **A slow drift is no longer called stable.** A series drifting one sigma end to
  end came back `stable: True` with a Cpk over 2 - "capable", for a process walking
  away from its own mean. Neither original criterion can see that shape: a slow
  drift widens no moving range, so `sigma_ratio` stays near 1.1 against a limit of
  1.20, and no single reading is extreme, so the outlier rate stays under 1%.

  The third criterion uses an instrument the library already shipped and had never
  called from here, `calculate_ewma_chart`, at lambda 0.1 - the standard choice for
  a one-sigma shift, and the band where the ARL literature puts EWMA ahead of
  Shewhart-type rules.

  | 1-sigma drift caught | 50 | 100 | 200 | 400 | 1000 |
  |---|---|---|---|---|---|
  | before | 20.6% | 7.4% | 4.1% | 1.8% | **0.2%** |
  | after | 25.2% | 27.0% | 43.3% | 58.1% | **80.1%** |

  Healthy false alarms moved 15.8 / 2.2 / 2.3 / 0.4 / 0.0% to
  16.5 / 2.9 / 3.0 / 0.5 / 0.0%. Detection had been *falling* as data accumulated,
  because a longer drift is more thoroughly absorbed into `sigma_overall`.

  **The criterion is a rate, not "the EWMA signalled at least once."** At lambda 0.1
  that form fires on 4.1 / 7.6 / 18.1 / 34.3 / 66.1% of healthy series - the same
  length-dependent trap that disqualified the Western Electric rule set as a gate.

- **A fourth criterion: a straight line fitted across the series.** The EWMA cannot
  see a half-sigma drift at any lambda - centred on its own mean it never leaves
  +/-0.25 sigma, against an EWMA limit of 0.688 sigma - while a slope test carries
  t = 4.56 on the same data at n=1000. A significant slope alone is not enough,
  because a long enough series makes a negligible slope significant, so the fitted
  total drift must also exceed 0.75 sigma.

  | 1-sigma drift caught | 50 | 100 | 200 | 400 | 1000 |
  |---|---|---|---|---|---|
  | three criteria | 14.0% | 27.0% | 43.3% | 58.1% | 80.1% |
  | four criteria | 33.7% | 64.4% | 84.2% | 92.8% | **98.6%** |

  Healthy false alarms went 3.4 / 2.9 / 3.0 / 0.5 / 0.0% to
  4.6 / 3.4 / 3.0 / 0.5 / 0.0% - unchanged at n >= 200, the whole cost landing on
  the two shortest lengths. A 2-sigma step now reaches 100% at every length.

- **The outlier criterion stopped reading noise on short series.** A 1% rate on 50
  points is one single reading, and chance puts at least one point beyond 3 sigma
  there 12.6% of the time, so on short series the criterion had degenerated into
  "any outlier at all". The rate limit is unchanged; the count must now also be more
  than chance explains, tested against Binomial(n, 0.0027) at alpha 0.05 - practical
  and statistical significance together, the shape `check_asset_drift` already uses.

  | healthy | 50 | 100 | 200 | 400 | 1000 |
  |---|---|---|---|---|---|
  | before | 17.4% | 3.4% | 3.0% | 0.5% | 0.0% |
  | after | **4.6%** | 3.4% | 3.0% | 0.5% | 0.0% |

  Nothing at n >= 100 moved: at 1,000 points the rate limit already demands 11
  outliers where chance produces 2.7, so the test is never the deciding factor. The
  price falls entirely on 50-point series and is real - a variance doubling is caught
  13.4% of the time there instead of 46.3%, a 1-sigma drift 33.7% instead of 42.2%.
  Both older figures were bought at that 17.4% false-alarm rate.

- **A short `baseline_n` no longer makes the gate cry wolf.** The window estimates
  `sigma_within` from its first N readings: unbiased, but noisy, while every
  criterion is calibrated against a sigma taken from the whole series. A criterion
  compared against a noisy ruler reads the noise.

  | healthy, 300 points | N=20 | N=30 | N=40 | N=60 | N=100 | N=150 |
  |---|---|---|---|---|---|---|
  | before | 40.4% | 28.4% | 24.8% | 15.8% | 7.4% | 4.8% |
  | after | 7.0% | 2.4% | **2.2%** | 1.2% | 0.4% | 0.0% |

  The sigma used **for testing** is widened by `1 + 2.0 / sqrt(N)`. **It never
  touches the reported `sigma_within`, so no capability index moves** - a Cpk that
  shifted with the choice of Phase I window would be a worse defect than the false
  alarms this removes, and a test pins Cp/Cpu/Cpl to the sigma reported beside them.
  The sigma-ratio criterion widens its limit rather than its sigma, so the value in
  `stability_criteria` still equals the reported `sigma_ratio`.

  With the instability starting when the window closes, a 2-sigma step is still
  caught 100% of the time at every window; a variance doubling goes from 99.6% to
  81.4% at N=20 and 100% to 95.0% at N=40; a 1-sigma drift from 94.8% to 68.2% and
  97.8% to 79.6%. N=20 still runs hot at 7.0%: twenty points cannot pin down a
  sigma, and no correction can invent the information.

  Behaviour with no `baseline_n` is unchanged.

### Added

- **`stability_criteria` in the `calculate_capability()` result.** Every criterion
  reports itself - `name`, the `value` measured, the `limit` it was compared
  against, whether it `fired`, and the `reason` - so a verdict can be read instead
  of re-derived. Adding a criterion is one entry in a registry, with the
  false-alarm rate and power that justify its limit in its own docstring; the two
  original criteria kept their limits and wording exactly, verified against 21
  frozen cases.
- `CAPABILITY_TREND_LAMBDA` (0.10), `CAPABILITY_TREND_RATE_LIMIT` (0.03),
  `CAPABILITY_OUTLIER_ALPHA` (0.05), `CAPABILITY_BASELINE_INFLATION_K` (2.0),
  `CAPABILITY_TREND_DRIFT_SIGMAS` (0.75) and `CAPABILITY_TREND_ALPHA` (0.01),
  exported beside the limits they join so a reader can look them up.

### Known limits, stated rather than left to be found

- A **0.5-sigma drift** is called 4.0% of the time at n=1000. That is a choice, not
  a blindness: it sits below `CAPABILITY_TREND_DRIFT_SIGMAS`. Lowering that to 0.5
  raises detection to 53.0% and the healthy rate at n=200 from 3.0% to 3.7%.
- A **mid-series variance change** is seen only by the outlier criterion, at 67.4%
  by n=1000. The ratio and the fitted line are both blind to it by construction.
- **`baseline_n=20`** still reports 7.0% of healthy series as unstable.
- At **n=50** the gate calls 4.6% of healthy series unstable, against 0.0% at
  n=1000. Short series remain the weakest case.

## [1.6.1] - 2026-08-30

### Removed

- **Python 3.10 is no longer supported; the floor is 3.11.** It was the only
  job left failing after the pandas 3 fix, on a resolver combination none of the
  other four versions produce. Claiming support for a version whose CI job is
  red is worse than not claiming it, and the classifier list, `requires-python`
  and the CI matrix are held equal by a test - so the claim had to move rather
  than be quietly ignored. `ruff`'s target moves to `py311` with it, which
  brings `datetime.UTC` into range.

### Fixed

- **`fit()` failed on any frame containing text, on pandas 3.0.** pandas 3.0
  gives a plain text column the dedicated `str` dtype rather than `object`, and
  seven sites across `engine.py` and `data.py` identified text by comparing
  `dtype == "object"`. That comparison answers False under the new dtype, so a
  text column was neither ordinal-encoded nor dropped as a high-cardinality
  identifier: it reached the estimator as text and raised
  `could not convert string to float: 'M01'` - a message naming a cell rather
  than the column or the cause. A machine id, a shift letter or a lot code is
  ordinary factory data, so this broke training for most real inputs on
  pandas 3 while every test still passed on pandas 2.
- The dtype test now lives in one predicate, `_is_text_series`, routed through
  pandas' own `is_object_dtype` / `is_string_dtype` so the answer tracks pandas
  instead of a list of dtype spellings kept in step by hand. Selecting text
  columns goes through `_text_columns()` for the same reason: no `select_dtypes`
  argument list is correct on both majors, because pandas 2 raises
  `TypeError: numpy string dtypes are not allowed` on `"str"` while the
  `["object", "string"]` it does accept only still finds text on pandas 3
  through a fallback pandas 4 removes. Naming no dtype at all is the only
  version-independent answer.
- **The rule is now a test that can fail.** `test_text_dtype_is_never_tested_inline`
  reads the package's AST and rejects any comparison of a `.dtype` against
  `"object"`, `"string"` or `"str"`. It reads the AST rather than the text so
  prose explaining the old bug is not mistaken for code, and it is
  mutation-tested: restoring the comparison at a single one of the seven sites
  makes it fail.

  This was found by CI, not locally. The development machine had pandas 2.3.3
  and every test passed there; the failure only appeared on the runners, which
  resolve the newest release.

## [1.6.0] - 2026-08-29

### Added

- **The quality-engineering track: `calculate_gauge_rr`, `calculate_control_rules`
  and `calculate_capability`.** They run in that order and each reports on the
  condition the next one depends on, because that dependency is the whole point:
  a Cpk computed on a drifting process describes nothing, and a Cpk computed
  through a gauge contributing 40% of the observed variation is measuring the
  gauge.
  - `calculate_gauge_rr(df, part_col, operator_col, measurement_col)` — crossed
    Gauge R&R by the ANOVA method (AIAG MSA 4th edition), pooling the
    part-by-operator interaction when its p-value exceeds 0.25 and reporting
    `interaction_pooled` either way so the choice is visible. Returns variance
    components, standard deviations, `percent_contribution` (variance scale, sums
    to 100) and `percent_study_variation` (standard-deviation scale, does not)
    separately, because they answer different questions and only the second one
    carries the AIAG acceptance bands. `ndc` below 5 means the gauge cannot rank
    the parts whatever the percentage says. An unbalanced design is refused by
    name rather than silently computed. Verified by generating data from chosen
    variance components and checking the ANOVA recovers them: σ_error 1.0 came
    back as 0.879, σ_part 5.0 as 4.699, and ndc matched exactly.
  - `calculate_control_rules(values, ...)` — the eight **Nelson** rules, with
    `CONTROL_RULES_WESTERN_ELECTRIC = (1, 2, 5, 6)` naming the classic four. The
    naming is kept honest: Western Electric's own fourth rule uses eight points in
    a row on one side where Nelson's rule 2 uses nine, and no hybrid was invented.
    A rule the series is too short to evaluate lands in `rules_skipped` with the
    length it needed — never reported as passing.
  - `calculate_capability(values, usl, lsl, ...)` — Cp/Cpk from within-subgroup
    sigma (moving range over d2) and Pp/Ppk from the overall standard deviation,
    plus `sigma_ratio` between them. One-sided limits return `cp: None` rather
    than pretending the missing limit is infinite. `skewness` and
    `excess_kurtosis` are reported with a `normality_warning`, because Cpk on a
    strongly non-normal process misstates the fraction outside the limits; no
    hypothesis test is run, since on a few thousand points it would reject on
    differences too small to matter.

  **What the sensitivity of the extra rules costs, measured rather than asserted.**
  On in-control standard-normal data, 100 points, 4,000 trials, probability that a
  set signals at least once: rule 1 alone **23.4%** (theory says 23.7%), the four
  Western Electric rules **59.6%**, all eight Nelson rules **74.0%**.

### Fixed

- **`capability_is_meaningful` was raised on almost all healthy data.** The first
  implementation gated it on "did any Western Electric rule fire", which measures
  as False on 30.5% of healthy 50-point series, 60.0% at 100, 81.0% at 200, 97.5%
  at 400 and **99.5% at 1,000** — the false-alarm rate climbed toward certainty as
  more data arrived, which is backwards, and a flag that is always raised is not a
  flag. This is the failure the Andon amber state already hit and solved by
  requiring sustained evidence. `stable` now uses two length-independent limits,
  `CAPABILITY_SIGMA_RATIO_LIMIT` (1.20) and `CAPABILITY_OUTLIER_RATE_LIMIT` (1% of
  points beyond 3σ), both chosen from measured false-alarm and detection curves
  rather than convention. Healthy series now fail at 12.0% / 4.5% / 2.0% / 0.0% /
  0.0% across the same lengths, while a 3σ drift is still caught 96% of the time
  and a 2σ step 100%. The rule violations are still reported in
  `stability_violations`, because they say where to look. The two blind spots are
  documented instead of left to be found: a mid-series variance doubling is caught
  44.5% of the time and a slow 1σ drift only 4%.

## [1.5.0] - 2026-08-29

### Changed

- **`potatopt.py` is now the `potatopt/` package.** The library was one 4,543-line
  module; it is now eleven, split along the lines it already had — `engine.py`,
  `analysis.py`, `data.py`, `spc.py`, `drift.py`, `reliability.py`,
  `calibration.py`, `constants.py`, `_lazy.py`, `_utils.py`, and an `__init__.py`
  that re-exports the public surface. **No public API changed**: `__all__` holds
  the same 57 names in the same order, every signature is identical, and every
  constant keeps its value. The split was verified mechanically rather than by
  reading the diff — each of the 71 functions and methods was parsed before and
  after and compared as an AST, and all 71 are identical. The 443 tests pass
  unchanged.
- Models saved by earlier versions still load. `joblib.dump` records the class as
  `potatopt.PotatOptEngine`; `__init__.py` re-exports that name, so unpickling
  resolves to the same class object. Verified with a round trip: a `.pkl` written
  by the pre-split module loads under the package with its state intact. New
  saves record `potatopt.engine.PotatOptEngine`.

### Added

- **An MCP server: `potatopt/mcp_server.py`.** An AI agent can now drive the
  library over the Model Context Protocol without writing Python. Run it with
  `potatopt-mcp` or `python -m potatopt.mcp_server`; it speaks stdio, so it runs
  as a local child process of the client and factory data never touches a
  network. Install with `pip install "potatopt[mcp]"`.

  **Seven tools, not fifty-seven,** and the number is a budget rather than a
  guess: a tool surface is spent from the agent's context before it does any
  work. The seven advertised — `auto_analyze`, `inspect_data`,
  `audit_data_quality`, `check_asset_drift`, `calculate_ewma_chart`,
  `calculate_maintenance_savings`, `calculate_oee` — measure **1,585 tokens**
  (`tiktoken cl100k_base`) for names, descriptions and input schemas together.
  At that average, exposing all 57 public names would cost roughly 13,000 tokens
  per session before a single question is answered. Every tool is marked
  `readOnlyHint`, and `auto_analyze`'s `save_to` parameter is deliberately not
  exposed so that annotation stays true.

  **The library's "never raises" guarantee is what makes it fit this transport.**
  When a tool function raises, the SDK discards the exception and returns
  `UnexpectedToolError("Error executing tool <name>")` — the reason never reaches
  the agent, which is left with a dead end it cannot diagnose. Because the
  library returns `{"error": ...}` instead of raising, the reason survives as
  data; the adapter adds the same guarantee over its own work, above all reading
  a CSV. Every payload also passes through `to_jsonable()`, since `json.dumps`
  would otherwise emit a bare `NaN` token that is not valid JSON.

  Reading files is confined when you ask for it: set `POTATOPT_MCP_ROOT` and any
  path resolving outside that directory is refused by name. Unset, the server can
  read whatever the user running it can read — stated plainly in the module
  docstring rather than described as a sandbox.
- **`calculate_correlations(df, method, min_abs, max_columns)`.** Which sensors
  carry the same information. Two columns that move together are one piece of
  evidence counted twice: a model handed both splits the credit between them and
  the importance ranking gets harder to read. `PotatOptEngine` already prunes
  collinear features during `fit()`; this is the function that shows you what it
  will prune and why. Returns the matrix, the `strong_pairs` above `min_abs`
  (each pair once, never self-paired), and — following the rule this project
  keeps relearning — a `skipped_columns` entry with a reason for every column it
  did not correlate: not numeric, constant (where the coefficient is undefined
  and pandas would hand back a null nobody can explain), or fewer than two
  non-null values. `max_columns` caps the matrix and says so in a `note`, because
  a 500-column frame is 250,000 cells that no chart can draw and no agent can
  afford to read.
- **`evaluate()` now returns `class_labels`.** The confusion matrix was built
  with `labels=label_encoder.classes_` but the order was never reported, so
  nothing downstream could tell which class was row zero — and with string labels
  `LabelEncoder` sorts alphabetically, which puts `FAIL` before `OK`. Any chart
  or report drawn from the matrix had to guess. It no longer has to. An added
  key only; nothing existing changed.
- **Three charts in `chart_engine`, closing the Phase 3 list:**
  `plot_confusion_matrix` (reads `class_labels`, states which axis is actual and
  which predicted, prints the count in every cell so it survives greyscale, and
  normalises rows without dividing by zero), `plot_correlation_heatmap` (a
  diverging map pinned to -1..+1 so a weak matrix cannot autoscale into looking
  alarming, with a `None` cell left blank rather than drawn as a zero — zero
  means uncorrelated, blank means not computed), and `plot_feature_importance`
  (horizontal bars from any of the three shapes the library produces, labelling
  the axis `mean |SHAP|` or `model importance` for the one it actually drew,
  because nothing about the bars distinguishes them).
- **`py.typed` (PEP 561).** Type checkers now read the annotations that have been
  in this library since 1.2.0 but were unreachable to them — the marker has to sit
  inside a package directory, which a single-file distribution has nowhere to put.
  It ships in both the wheel and the sdist.

### Fixed

- **The `print()` guard was scanning one file by name and would have gone blind.**
  `test_no_print_calls_in_source` read `potatopt.py` directly; after the split that
  path is gone, and a guard that reads nothing reports success forever. It now
  walks every module in `potatopt/` and additionally asserts that it found at
  least ten of them, so a future move fails the test instead of quietly passing it.
- **The emoji scan never looked at the new package directory.** `EMOJI_SCANNED_DIRS`
  is an explicit list; `potatopt` was added to it.
- **The module header still advertised SHA-256 as preventing "unauthorized
  tampering."** Version 1.4.0 corrected that overclaim in `save()`, `load()` and the
  alert strings, but the module docstring was missed and kept the old wording.
  It now says what the hash actually proves: integrity, not authenticity.
- **The package split left a second copy of `__version__` in `engine.py`.** Both
  literals read `1.4.0`, so nothing disagreed and nothing caught it — until the
  bump to this release, when `get_training_report()` and the metadata written by
  `save()` carried on stamping the old number onto new models. That is a
  traceability record being quietly wrong, which is the one thing it exists not
  to be. `engine.py` now reads the single literal in `__init__.py` at call time,
  and a test fails if any other module in the package declares one.

### Internal

- `pyproject.toml` moves from `py-modules` to `packages` and declares
  `package-data` for the marker. `__version__` must stay a **literal** assignment
  in `__init__.py`: setuptools reads `attr = "potatopt.__version__"` statically
  from the AST for a literal but falls back to importing the package for anything
  else, and the CI `build` job installs only pip, build and twine — an import
  fallback would fail there for want of numpy. That constraint is now a test.
- `_load_automl` and `_load_shap` live only in `_lazy.py`. Both cache into
  `globals()`, so a single home means a single cache, which is what keeps
  `potatopt.AutoML` and the class used inside `fit()` the same object.

## [1.4.0] — 2026-08-28

A public-repository pass: a bug that silenced warnings for every caller of the
library, a security claim that overclaimed what it checked, and two gaps a
thesis reviewer would ask about first — a hard-coded random seed, and no check
that a predicted probability means what it says.

### Fixed

- **`import potatopt` silenced every warning in the importing process.** Module
  import used to call `warnings.filterwarnings('ignore')` at the top level — a
  mutation of Python's process-global warning filter list, not a local setting.
  Any application that imported this library stopped seeing its *own* warnings
  for the rest of the run, with no way to tell why. Replaced with
  `_quiet_dependency_warnings()`, a context manager scoped to the two call sites
  that are actually noisy — FLAML's search inside `fit()` and the SHAP fallback
  ladder in `get_shap_values()` — so the caller's filters are untouched outside
  those calls. Verified by importing the module and asserting
  `warnings.filters` is unchanged, and that a `UserWarning` raised afterwards by
  ordinary caller code still reaches `pytest.warns`.
- **`load()` claimed a "cryptographic signature" and said it prevented
  "unverified code execution."** Neither is true. The SHA-256 hash is written in
  plaintext next to the model file; there is no key and no signature, so it
  proves integrity (the bytes did not change) and not authenticity (who
  produced them) — and `joblib.load` is pickle, so a hostile file executes code
  regardless of whether its hash matches its own sidecar. This matters more now
  that the repository is public. `save()` and `load()` now say plainly what the
  check does and does not prove; no behaviour, exception type, or code path
  changed, only the wording.

### Added

- `random_state` on `split_data()`, `split_data_three_way()`,
  `PotatOptEngine.__init__()` and `auto_analyze()`. It replaced three separate
  hard-coded `42`s (two `IsolationForest` fallbacks and FLAML's `seed`), which
  meant every published number in this project was, silently, a single-seed
  number with no way to ask how much of it was the seed. The default is still
  `42` (`DEFAULT_RANDOM_STATE`), so existing calls behave exactly as before.
  Reported back in `get_training_report()` and in the `save()` metadata sidecar.
- `run_seed_sweep(data, target, seeds=(0,1,2,3,4), **kwargs)` — runs
  `auto_analyze` once per seed and reports mean / std / min / max / **spread**
  per metric, plus a `stability_note`. The number that matters is the spread:
  a difference between two configurations smaller than either one's spread is
  not a finding, and the note says so rather than leaving the reader to notice.
- `check_calibration(y_true, y_prob, n_bins=10)` and
  `PotatOptEngine.check_calibration(X, y, n_bins=10)` — Brier score, Brier skill
  score against the base rate, Expected and Maximum Calibration Error, and a
  per-bin table. This closes a real gap in `optimize_threshold()`: the
  threshold it picks is the cheapest cut on the model's score whether or not
  that score is a calibrated probability, and an uncalibrated score cannot
  honestly be read as "act at a 30% chance of failure," cannot be used to quote
  an expected cost per call-out, and does not survive being moved to a line
  with a different failure rate. The anomaly-detection fallback is always
  reported as uncalibrated on purpose — its `predict_proba` is a fixed sigmoid
  over `IsolationForest`'s decision function, built to rank consistently, not
  to estimate a failure rate — and the result says so via
  `probability_source: "isolation_forest_sigmoid"`.

---

## [1.3.0] — 2026-08-28

Reliability and OEE metrics — the calculation layer the shop-floor system is
built on, added to the library rather than the app so it stays stateless,
testable without a database, and usable on its own.

### Added

- `calculate_mtbf()` — MTBF over **operating** hours, not calendar hours. The two
  definitions disagree and calendar time is the flattering one: it counts idle
  hours as trouble-free service, so a rarely used machine scores well for doing
  nothing. Only `wo_type` values in `breakdown_types` raise the failure count —
  counting planned and predictive work as failures would penalise exactly the
  behaviour condition monitoring exists to produce. Zero breakdowns returns
  `mtbf_hours: None`, never an error; a machine that has not failed is an answer.
- `calculate_mttr()` — returns **three** durations most systems report as one:
  `mtta_hours` (reported → started, the organisation's scheduling and spares
  delay), `mttr_hours` (started → finished, the technician and the job), and
  `mdt_hours` (reported → finished, what production actually loses). A single
  blended figure hides which of the three to work on. All three average over one
  shared row mask, because filtering each column separately breaks the identity
  `MDT = MTTA + MTTR` silently whenever a row is missing one timestamp — a report
  that contradicts itself with nothing to show for it. A test locks the identity.
- `calculate_availability()` — inherent `MTBF/(MTBF+MTTR)` and operational
  `MTBF/(MTBF+MDT)`, plus `availability_lost_to_waiting` as its own key. That gap
  is availability lost to **waiting rather than repairing**: no new equipment
  removes it, but scheduling, spares holding and work study can. Reporting a
  single availability figure hides it.
- `calculate_oee()` — availability × performance × quality. The `availability`
  here is **not** what `calculate_availability()` returns — a production-time
  ratio over a shift versus a reliability ratio from MTBF and MTTR. The two are
  routinely confused and quoted against each other; the docstring says so.
  Nothing is clamped: performance above 1.0 means the machine beat its stated
  ideal cycle time, which almost always means the master data is wrong, so it
  comes back as a warning rather than a tidy 1.0 that hides the defect.
- `calculate_pareto()` — ranks by event count, or by any value column such as
  downtime or cost. Supporting both is the point: on the reference fixture,
  counting events says fix the sensor (25 failures, 30 minutes each) while
  counting hours says fix the bearing (10 failures, 4 hours each). Only one of
  those is worth the work, and count-Pareto is the classic trap that sends a team
  at the wrong one. A test asserts the two rankings disagree. A null category is
  grouped as `"(unknown)"` rather than dropped, since dropping it understates the
  total and skews every percentage with it.
- Constants `PARETO_CUTOFF = 0.80` and `OEE_WORLD_CLASS = 0.85`.

### Added

- `examples/ai4i_dataset.py` — downloads the **AI4I 2020 Predictive Maintenance
  Dataset** (UCI, CC BY 4.0), verifies it against a pinned SHA-256, and caches it
  under `examples/data/`, which `.gitignore` already excludes. The seven columns
  it drops are the point: `UDI` and `Product ID` are identifiers a model would
  memorise, and `TWF/HDF/PWF/OSF/RNF` are the individual failure modes — labels
  that exist only *after* the machine has failed. The dataset ships its own
  leakage trap.
- `examples/quickstart.py` — the whole condition-based-maintenance story in ten
  printed steps on real data: profile, three-way split, train, cost-tuned
  threshold, score, money, SHAP, per-asset drift, control charts. On the held-out
  2,000 rows it avoids 61 of 68 breakdowns and saves **2,269,500** against
  run-to-failure (66.75%), at the price of 134 wasted inspections.
- `benchmarks/runtime_cost.py` — the sibling of `token_cost.py`: what the same
  pipeline costs to *run*, rather than to write. Each variant is measured in its
  own interpreter, on identical splits, at `n_jobs=1`. Measured at a 60-second
  budget: PotatOpt 176 MB imports / 315 MB peak / savings 2,317,500 against
  scikit-learn by hand at 162 MB / 176 MB / 2,304,000. The imports are nearly
  equal, so the memory difference is the AutoML search training many candidate
  models — not a heavier dependency stack. **PyCaret could not be measured at
  all**: 3.3.2 declares Python 3.9–3.11 and pins `numpy<1.27`, `pandas<2.2.0`,
  `scipy<=1.11.4`, none of which publish wheels for 3.13+; the benchmark reports
  that as an environment fact instead of skipping the row.
- `lag1_autocorrelation` and `autocorrelation_warning` on `calculate_ewma_chart()`
  and `calculate_cusum_chart()`, with the constant `AUTOCORRELATION_WARN = 0.5`.
  Moving-range σ assumes independent observations; a random-walk sensor breaks
  that assumption in the *opposite* direction from the trend problem σ already
  guards against — consecutive readings repeat each other, the moving range
  collapses, and the chart alarms almost everywhere. Measured on AI4I 2020 over
  600 points with `baseline_n=100`: process temperature lag-1 **+0.919**, MR-σ
  0.0501 against SD 1.484, **573 of 600 points flagged**; torque lag-1 −0.051,
  **0 of 600**.
  The check reads the **baseline window, never the whole series**, because a
  degradation ramp and a random-walk sensor are indistinguishable over a whole
  series: a textbook wear ramp reads +0.998 overall and +0.042 across its
  in-control window. Without that restriction the warning fired on the chart's
  own success case. With no `baseline_n` to compare against, the message says so
  rather than guessing which cause it is looking at.
- `top_features_note` on `auto_analyze()` — the reason SHAP declined, instead of
  an empty list with no explanation. The caller of that facade is a beginner or
  an agent, and neither can debug `[]`.
- `verbose` on `PotatOptEngine` (default `0`, quiet). FLAML raises its own
  logger's level inside `fit()`, so a caller cannot silence it from outside.
  Measured on a 3-second budget: **81 log lines and 9.5 KB** written into the
  caller's output, now 41 bytes. `verbose=1` or higher hands the search log back.
- `additivity_check_relaxed` on `explain_predictions()` — True when SHAP's
  additivity check had to be disabled to produce any attribution at all, so an
  approximate ranking can never be mistaken for an exact one.

### Fixed

- **SHAP returned nothing for any model trained on a pandas `category`
  feature.** `get_shap_values()` flattened those columns to plain numbers before
  handing them to the explainer; LightGBM rejects that frame, TreeExplainer's
  additivity check then fails against a model output it could not reproduce,
  every fallback layer failed the same way, and `auto_analyze` reported
  `top_features: []`. The untouched frame is now tried first. Verified by calling
  both implementations **on one fitted engine** to remove training randomness: on
  AI4I 2020 the old path produced no attributions, the new one ranked all six
  features with the additivity check intact.
- **`fit()` and `inspect_data()` raised a raw numpy `TypeError` on any pandas
  `category` column** — `np.issubdtype` only understands numpy dtypes. All three
  dtype tests now go through `pd.api.types.is_numeric_dtype`, and categoricals
  are unwrapped once at the top of both preprocessing paths. This is not an
  exotic edge case for this library: `astype("category")` is the standard way to
  cut a frame's memory, so the low-spec user is the one most likely to hit it,
  and a categorical that slipped through would have been filed as numeric and
  scaled.

## [1.2.0] — 2026-08-28

Condition-based maintenance, and the project infrastructure to keep it honest.

### Added

**Statistical process control**
- `calculate_ewma_chart()` — EWMA control chart with **exact time-varying** limits
  rather than the asymptotic form, so the early samples are not covered by limits
  that are too wide.
- `calculate_cusum_chart()` — two-sided tabular CUSUM (k = 0.5σ, h = 5σ).
- Both return JSON-ready dictionaries and never raise; a bad argument comes back
  as `{"error": "..."}`.
- σ is estimated from the **moving range ÷ d₂ (1.128)**, never the sample standard
  deviation. A degrading series inflates its own SD, which widens the limits and
  hides the very trend the chart exists to catch. Measured on a wear ramp:
  SD 2.201 vs MR-σ 0.222, and EWMA signalled at sample 3 instead of sample 13.

**Per-asset drift**
- `check_asset_drift(train_df, batch_df, asset_col, ...)` — drift checked per
  machine instead of pooled. Pooling three machines running at 70/75/80 °C reads a
  change in the *reporting mix* as drift (one machine taken down for maintenance
  fired a false alarm at PSI 1.26 although nothing changed), and dilutes a genuine
  +3 °C fault from 3.084 per asset to 0.244 pooled — 12.5×, because the pooled σ of
  4.18 contains the spread *between* machines.
- Guarded against the false alarms that small per-asset batches create: PSI bins
  scale to the batch size, and the mean-shift threshold is raised to
  `max(threshold_pct, k·√(1/n_batch + 1/n_train))` — practical significance and
  statistical significance, both required. False-alarm rate at 30 rows fell from
  39.3% to 0.7% while 1.0σ shifts are still caught 100% of the time.
- `calculate_categorical_psi()` — PSI for label columns (shift, recipe, operator,
  lot). A category seen in the batch but never in training is treated as real news,
  not discarded.

**Maintenance economics**
- `calculate_maintenance_savings()` (stateless), `.calculate_maintenance_cost()`
  and `.optimize_maintenance_threshold()`. The baseline is **run to failure**, not
  manual inspection, and a false positive is charged an inspection only — an
  engineer looks before replacing a part.
- `breakdown_avoidance_rate` is reported beside `cost_savings` because the two
  disagree exactly where it matters: a model that flags every machine reaches
  perfect recall and **loses 660,000**, since 980 pointless call-outs cost more
  than the breakdowns they prevented. Break-even is 540 false alarms.

**Project infrastructure**
- GitHub Actions CI: ruff, the full suite across five Python versions plus Windows,
  a core-only install check, and a packaging build with `twine check`.
- `scripts/verify_core_install.py` — proves the four-package claim in an
  environment where the extras are genuinely absent, which no test inside `tests/`
  can do.
- Tests that fail if `pyproject.toml` classifiers and the CI matrix ever disagree.
- This changelog and `CONTRIBUTING.md`.

### Changed

- `check_data_drift()` gained `min_rows`, `psi_bins` and `include_categorical`.
  **Every default reproduces the previous behaviour exactly**, so no existing call
  changes its result.
- `requires-python` raised from `>=3.9` to `>=3.10`. Python 3.9 reached end of life
  in October 2025 and was never tested; classifiers now list 3.10 – 3.14, matching
  the CI matrix exactly.
- `calculate_cost_of_quality()` internals moved onto a shared `_binary_confusion()`
  helper. Verified byte-identical across four parameter sets by running the old and
  new implementations against the same fitted engine.

### Fixed

- **A flat baseline window silently blinded the control charts.** A coarse sensor
  holding one value through the whole Phase I window gave σ = 0, so a process
  climbing from 10 to 19.5 was reported as in control. σ now falls back to the whole
  series; only a genuinely constant series stays degenerate.
- **NaN and infinity walked through the chart parameter guards.** Every comparison
  against NaN is false, so `n_sigmas`, `slack_k` and `decision_h` all slipped past.
  The worst case was silent: `max(0.0, nan)` returns `0.0`, pinning both CUSUM arms
  at zero — a monitoring chart that has stopped monitoring.
- **Non-numeric parameters raised instead of returning an error dict**, breaking the
  documented contract (`float("big")` → `ValueError`).
- **A sensor column missing from the batch was reported as healthy.** Training on
  `[temp, vibration]` and scoring a batch holding only `[temp]` returned
  `drift_detected: False, max_psi: 0.0` — a positive assertion of stability about a
  sensor that was not there. Such columns are now listed in `skipped_features`.
- **A machine that stopped reporting was invisible**, because the scan iterated only
  the batch. A dead gateway and a healthy machine looked identical. The union of both
  frames is now walked, with a `missing_from_batch` status.
- **Row counts were mistaken for reading counts.** A column that is 98% NaN still has
  200 rows, so any `len(df)` gate passed it; measured with 4 usable readings, a batch
  drawn from the training distribution itself raised a false alarm **100% of the
  time**. `min_rows` now counts non-null values per column.

---

## [1.1.0] — 2026-08-27

Correctness fixes and the packaging work that made the library installable.

### Added

- `auto_analyze()` — the whole pipeline in one call, returning a single JSON-ready
  dictionary and never raising. **91.7% fewer tokens** than the equivalent
  hand-written scikit-learn pipeline (674 → 56, measured with `tiktoken cl100k_base`
  in `benchmarks/token_cost.py`).
- `split_data_three_way()` for Train/Validation/Test.
- `pyproject.toml`. The core install is **four packages**; FLAML and SHAP are extras,
  loaded lazily through a module `__getattr__` (PEP 562), deferring **1.46 s = 44%**
  of import cost. A subprocess test asserts `import potatopt` leaves them out of
  `sys.modules`.
- `to_jsonable()`, `.explain_predictions()`, and scikit-learn estimator compatibility
  (`BaseEstimator`, `__sklearn_is_fitted__`, `classes_`, `__sklearn_tags__`), so
  `cross_val_score`, `GridSearchCV` and `Pipeline` all work.
- `__all__`, closing a surface that had exported 11 third-party module names.
- `n_jobs` constructor argument (previously hard-coded to `-1`, which oversubscribes
  a low-core machine — the hardware this library targets).
- Memory benchmarks that make the LowSpecML claim falsifiable: numeric downcast
  **50.0%**, full preprocessing **84.5%** (2.71 MB → 0.42 MB), and a 400 MB RSS
  ceiling on `fit()`.
- Type hints on all 45 functions, verified by comparing parsed ASTs before and after
  to prove no signature changed.

### Fixed

- **Temporal leakage in forecasting.** `fit()` now sends `split_type="time"` and
  `eval_method="cv"` to FLAML. Confirmed against the FLAML source: for a regression
  task, `split_type="auto"` resolves to `"uniform"` — a random shuffle that lets
  future rows train the model.
- **Threshold-tuning leakage is now self-reporting.** `optimize_threshold()` takes
  `X_val`/`y_val` and records a SHA-1 dataset fingerprint; `evaluate()` warns and
  returns `threshold_leakage_warning` when results are reported on the same rows the
  threshold was tuned on.
- An off-by-one in three-way splitting caused by float drift; the second cut now uses
  an exact integer row count.

---

## [1.0.0] — 2026

Initial public release of the PotatOpt industrial ML engine: AutoML through FLAML,
ordinal encoding, lossless memory downcasting, cost-sensitive weighting,
SHAP explanations, PSI drift detection and cost-of-quality reporting, all designed
to run on CPU-only low-specification hardware.

[1.6.0]: https://github.com/Oak04K/PotatOpt-ML/releases/tag/v1.6.0
[1.5.0]: https://github.com/Oak04K/PotatOpt-ML/releases/tag/v1.5.0
[1.4.0]: https://github.com/Oak04K/PotatOpt-ML/releases/tag/v1.4.0
[1.3.0]: https://github.com/Oak04K/PotatOpt-ML/releases/tag/v1.3.0
[1.2.0]: https://github.com/Oak04K/PotatOpt-ML/releases/tag/v1.2.0
[1.1.0]: https://github.com/Oak04K/PotatOpt-ML/releases/tag/v1.1.0
[1.0.0]: https://github.com/Oak04K/PotatOpt-ML/releases/tag/v1.0.0
