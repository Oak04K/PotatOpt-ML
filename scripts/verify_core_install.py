"""Verify the LowSpecML install claim on a core-only environment.

The README says the core install is four packages and that FLAML and SHAP are
never imported until a feature actually needs them. That is a claim about the
installed environment, so no test inside `tests/` can check it - by the time the
suite runs, the heavy extras are present.

Run this against an environment created with `pip install .` and nothing else:

    python scripts/verify_core_install.py

Exits non-zero with a plain sentence on the first failure. CI runs it as its own
job; it is equally useful locally when checking whether a new import crept in.
"""

from __future__ import annotations

import subprocess
import sys

HEAVY_PACKAGES = ("flaml", "shap", "lightgbm", "xgboost", "matplotlib", "fastapi", "mcp")
CORE_PACKAGES = ("numpy", "pandas", "scipy", "scikit-learn")


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    sys.exit(1)


def installed_packages() -> set[str]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list", "--format=freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.split("==")[0].strip().lower() for line in result.stdout.splitlines() if "==" in line}


def check_nothing_heavy_was_installed() -> None:
    present = installed_packages()
    for package in CORE_PACKAGES:
        if package not in present:
            fail(f"core dependency {package!r} is missing from the environment")
    intruders = [package for package in HEAVY_PACKAGES if package in present]
    if intruders:
        fail(f"the core install pulled in {', '.join(intruders)} - check pyproject dependencies")
    print(f"  core dependencies present, none of {', '.join(HEAVY_PACKAGES)} installed")


def check_import_stays_light() -> None:
    # Imported inside the function so the pip listing above runs first.
    import potatopt as po

    # `mcp` is here for a different reason than the other two: it is not lazily
    # loaded at all. `potatopt/mcp_server.py` is a submodule nothing else in the
    # package imports, so it - and the SDK it needs - stay out of the import path
    # unless someone runs the server on purpose.
    for module in ("flaml", "shap", "lightgbm", "mcp"):
        if module in sys.modules:
            fail(f"importing potatopt loaded {module!r}; the lazy __getattr__ has been bypassed")
    print(f"  import potatopt v{po.__version__} left flaml, shap, lightgbm and mcp out of sys.modules")


def check_the_stateless_utilities_work() -> None:
    import potatopt as po

    chart = po.calculate_ewma_chart([10] * 5 + [12] * 5, target=10.0, sigma=1.0)
    if chart.get("out_of_control") is not True:
        fail(f"calculate_ewma_chart did not signal on a step change: {chart.get('note') or chart}")

    saving = po.calculate_maintenance_savings(18, 25, 2)
    if saving.get("cost_savings") != 691_500.0:
        fail(f"calculate_maintenance_savings returned {saving.get('cost_savings')}, expected 691500.0")

    drift = po.calculate_categorical_psi(["A"] * 300 + ["B"] * 150, ["A"] * 100 + ["B"] * 300)
    if drift is None or drift <= po.PSI_MAJOR_SHIFT:
        fail(f"calculate_categorical_psi missed a major shift: {drift}")

    print("  SPC charts, maintenance costing and categorical PSI all run on the core four")


def check_the_missing_backend_error_is_useful() -> None:
    import potatopt as po

    try:
        # Binding the result matters: this attribute access is what triggers the
        # lazy loader, and the name is reused below to describe the failure.
        automl = po.AutoML
    except ImportError as exc:
        if "pip install" not in str(exc):
            fail(f"the ImportError does not say how to fix it: {exc}")
        print(f"  asking for AutoML without it installed says: {exc}")
        return
    fail(f"expected an ImportError naming the extra to install, but AutoML resolved to {automl!r}")


def main() -> None:
    print("Verifying a core-only PotatOpt install")
    check_nothing_heavy_was_installed()
    check_import_stays_light()
    check_the_stateless_utilities_work()
    check_the_missing_backend_error_is_useful()
    print("OK: the core install is genuinely four packages and stays lazy")


if __name__ == "__main__":
    main()
