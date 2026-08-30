# Contributing to PotatOpt-ML

PotatOpt-ML is an Industrial Engineering undergraduate thesis project as well as a
library. That shapes what a good contribution looks like: the code has to be
defensible, not merely working.

## Setting up

```bash
git clone https://github.com/Oak04K/PotatOpt-ML.git
cd PotatOpt-ML
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[automl,xai,viz,app,mcp,dev]"   # the same set CI installs
```

Python 3.11 – 3.14. Every version listed in `pyproject.toml` is exercised by CI, and
nothing is claimed that CI does not run — a test enforces that those two lists stay
identical.

## Before opening a pull request

```bash
python -m ruff check potatopt chart_engine.py tests benchmarks scripts examples
python -m pytest tests -q                                   # 484 tests, ~3 min
```

The zero-error ruff standard is not aspirational; there is no warning tier. If a
rule genuinely should not apply, suppress it with a `# noqa: RULE` **and a comment
saying why** — an unexplained suppression will be questioned.

> **Do not audit suppressions with `ruff check --select RUF100`.** `--select`
> disables every other rule, so all remaining `# noqa` comments falsely report as
> unused. Run plain `python -m ruff check`.

To check the four-package install claim, which no test in `tests/` can verify
because the extras are already present by then:

```bash
python -m venv /tmp/core && /tmp/core/bin/pip install .
/tmp/core/bin/python scripts/verify_core_install.py
```

## The design constraints

These are the reasons this library exists. A change that quietly breaks one of them
will not be merged, however good it is otherwise.

**1. It has to run on a potato.** CPU only, no GPU, 8–16 GB of RAM. That is why the
library uses ordinal rather than one-hot encoding, lossless dtype downcasting,
cost-sensitive weighting instead of SMOTE, and tree-based models. Prefer the
CPU-friendly, low-memory approach even when it is a little less elegant.

**2. The core install stays at four packages.** `numpy`, `pandas`, `scipy`,
`scikit-learn`. Anything heavier is an extra, loaded lazily through the module
`__getattr__` and never imported until the feature is actually called. Adding a
top-level `import` of a heavy dependency will fail CI.

**3. Public functions return errors, they do not raise.** The stateless utilities are
meant to sit behind `auto_analyze()` and, eventually, a tool-calling layer where the
caller may be a language model. A bad argument gets a sentence back in an
`{"error": ...}` dictionary, not a traceback. Everything returned must survive
`json.dumps()` — use `to_jsonable()` at the boundary.

**4. Results are stated in engineering and financial terms.** "Accuracy 94%" is not a
result. Which machine, how much money, how many breakdowns avoided — that is a result.

**5. Hide ceremony, never hide decisions.** `auto_analyze()` conceals splitting,
encoding and scaling because they are always done the same way. It keeps the cost
parameters in the signature because only the engineer on the floor knows what a
breakdown actually costs.

## How changes are expected to be justified

The most valuable habit in this project, and the one that has found every real bug so
far, is **measure before you write the specification.**

Prototyping first is how the FLAML temporal-leakage bug was confirmed rather than
assumed, how three undocumented scikit-learn compatibility gaps turned up, how the
10× difference between σ estimators was found, and how every threshold in the
per-asset drift guards was chosen. Guessing a threshold and then writing a test that
agrees with the guess proves nothing.

So, for anything numerical:

- **Quote the measurement in the docstring and the changelog.** Not "improves
  detection" but "signalled at sample 3 instead of 13".
- **Test the behaviour, not your assumption about it.** One test here originally
  asserted that a windowed baseline signals *earlier*. Measurement showed the
  opposite: it signals *correctly*, while whole-series estimation fires "decreasing"
  at index 0 on a series that only rises. The test now locks the real behaviour.
- **State the cost of a guard.** The sampling-noise floor cut false alarms from 39.3%
  to 0.7% at 30 rows, and it also drops a genuine 0.5σ shift from 98.8% detection to
  69.8%. Both numbers belong in the documentation.

## What tends to get things rejected

- Silence as an answer. The worst bugs found in this project all returned a confident
  all-clear: a CUSUM chart pinned at zero by a NaN, a missing sensor column reported
  as stable with `max_psi: 0.0`, a machine that had stopped reporting simply not
  appearing. If a check cannot be performed, say so — that is what `skipped_features`
  and the per-asset `status` field are for.
- Changing a default and calling it a fix. New arguments default to the existing
  behaviour so that no existing call changes its result.
- Editing an expected value to make a test pass. If a measured expectation fails, the
  measurement is information: report it.
- Refactoring unrelated code alongside a change.
- Adding `print()`. The library logs through `logger`; a test enforces this.

## Commits and branches

Conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `test:`, `chore:`) with a body
that explains *why*. Branch from `main`; CI must be green before merge.

## Project layout

```
potatopt/                       the library
  __init__.py                   the public surface: re-exports, __all__, __version__
  mcp_server.py                 the MCP adapter (7 tools, stdio); imports nothing else here
  engine.py                     PotatOptEngine
  analysis.py                   auto_analyze, run_seed_sweep
  data.py                       profiling, splitting, data quality
  spc.py                        control charts and the Nelson/Western Electric rules
  quality.py                    Cp/Cpk/Pp/Ppk and Gauge R&R (MSA)
  drift.py                      drift and PSI
  reliability.py                MTBF/MTTR/availability/OEE/Pareto
  calibration.py                check_calibration
  constants.py                  tuning constants
  _lazy.py                      the logger and the lazy FLAML/SHAP loaders
  _utils.py                     shared helpers, to_jsonable, audit log
  py.typed                      PEP 561 marker
tests/test_potatopt.py          the suite
benchmarks/token_cost.py        measures the token-efficiency claim
scripts/verify_core_install.py  checks the four-package install claim
Agent.md                        roadmap and progress log (Thai)
```

Three rules hold this layout together, and all three are enforced by tests:

- **`__init__.py` assigns `__version__` as a literal string.** `pyproject.toml` reads
  it with `attr = "potatopt.__version__"`; setuptools resolves a literal statically
  but falls back to *importing* the package for anything else, and the CI build job
  has no numpy to import it with.
- **`_load_automl` and `_load_shap` live only in `_lazy.py`.** They cache into
  `globals()`, so one home means one cache, which is what keeps `potatopt.AutoML` and
  the class used inside `fit()` the same object.
- **Nothing in the package imports `mcp_server`.** It is the one module that
  needs a dependency outside the four-package core, and it stays out of the
  import path unless somebody starts the server on purpose. Its tools must also
  never raise: the MCP SDK replaces an escaping exception with an opaque
  `UnexpectedToolError` and the reason never reaches the agent, so every tool
  returns `{"ok": False, "error": ...}` instead.
