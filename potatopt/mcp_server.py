"""
PotatOpt MCP Server: Model Context Protocol adapter for the PotatOpt library.

This server provides an MCP interface over PotatOpt's core capabilities:
Automated Machine Learning (AutoML), Data Quality Auditing, Statistical Process
Control (SPC / EWMA), Asset Drift Detection, Financial Maintenance Savings,
and Overall Equipment Effectiveness (OEE) calculations.

The server communicates via standard input/output (stdio transport), ensuring
that factory telemetry, sensor readings, and maintenance records never leave
the local machine.

How to run:
----------
Using the console script:
    potatopt-mcp

Using python module execution:
    python -m potatopt.mcp_server

Path Confinement:
----------------
Set the environment variable `POTATOPT_MCP_ROOT` to restrict CSV file reading
strictly to paths within that directory. When `POTATOPT_MCP_ROOT` is unset, the
server can read any file that the operating system user running the process has
filesystem permissions to read. This is a standard process permission model,
not an isolated sandbox.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

import potatopt as po
from potatopt import __version__, to_jsonable


def _resolve_csv(path: str) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    """
    Resolve and load a CSV file safely into a pandas DataFrame.

    Returns:
    --------
    tuple[pd.DataFrame | None, dict[str, Any] | None]:
        (dataframe, None) on success, or (None, error_dict) on failure.
        The error dict always has the shape `{"ok": False, "error": "<message>"}`.

    Confinement:
    ------------
    If `POTATOPT_MCP_ROOT` is set, resolves both the root and requested path using
    `os.path.realpath` and rejects any path outside the root with an error dict
    naming the root.
    When `POTATOPT_MCP_ROOT` is unset, the server can read any file the running
    user can access on the filesystem.
    """
    if not isinstance(path, str) or not path.strip():
        return None, {"ok": False, "error": "CSV path must be a non-empty string."}

    mcp_root = os.environ.get("POTATOPT_MCP_ROOT")
    if mcp_root:
        real_root = os.path.realpath(mcp_root)
        real_path = os.path.realpath(path)
        try:
            common = os.path.commonpath([real_root, real_path])
            if (
                os.path.normcase(common) != os.path.normcase(real_root)
                or os.path.normcase(real_path) == os.path.normcase(real_root)
            ):
                return None, {
                    "ok": False,
                    "error": (
                        f"Access denied: path '{path}' resolves to '{real_path}', "
                        f"which is outside the allowed root directory '{real_root}'."
                    ),
                }
        except (ValueError, TypeError):
            return None, {
                "ok": False,
                "error": (
                    f"Access denied: path '{path}' cannot be resolved within "
                    f"the allowed root directory '{real_root}'."
                ),
            }

    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - see the note above create_server()
        return None, {"ok": False, "error": f"Failed to read CSV at '{path}': {exc}"}

    if df.empty:
        return None, {"ok": False, "error": f"CSV at '{path}' is empty."}

    return df, None


# Why every tool below catches bare `Exception` (ruff BLE001, silenced per line):
#
# When a tool function raises, this SDK discards the exception and hands the
# caller `UnexpectedToolError("Error executing tool <name>")` - the reason is
# gone. An agent on the other end is then stuck with a dead end it cannot
# diagnose or work around. Returning `{"ok": False, "error": ...}` keeps the
# reason in the payload where the agent can read it and choose what to do next.
#
# The library's own functions already return errors instead of raising, which is
# what makes them fit this transport; these guards exist for the adapter's own
# work - reading files, and any dependency that decides to raise something new.
def create_server() -> MCPServer:
    """Create and configure the PotatOpt MCP server instance."""
    server = MCPServer(
        name="potatopt",
        version=__version__,
        instructions=(
            "PotatOpt MCP Server: Model Context Protocol adapter for industrial machine learning, "
            "predictive maintenance, data quality auditing, statistical process control, and OEE analytics."
        ),
    )

    @server.tool(
        description=(
            "Run the complete end-to-end predictive maintenance ML pipeline in one call. "
            "Automatically profiles data quality, splits data three ways (train/validation/test), "
            "tunes the decision threshold on validation against financial Cost of Quality, "
            "scores an untouched test set, and reports monetary impact in the caller's currency. "
            "Runs an AutoML search bounded by time_budget seconds and will block for roughly that "
            "duration. Does not save models to disk."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def auto_analyze(
        csv_path: str,
        target: str,
        cost_scrap: float = 500.0,
        cost_fa: float = 150.0,
        cost_insp: float = 20.0,
        time_budget: int = 30,
        random_state: int = 42,
    ) -> dict[str, Any]:
        """Run the AutoML pipeline over a CSV dataset."""
        try:
            df, err = _resolve_csv(csv_path)
            if err is not None:
                return err
            res = po.auto_analyze(
                data=df,
                target=target,
                cost_scrap=cost_scrap,
                cost_fa=cost_fa,
                cost_insp=cost_insp,
                time_budget=time_budget,
                random_state=random_state,
            )
            return to_jsonable(res)
        except Exception as exc:  # noqa: BLE001 - see the note above create_server()
            return {"ok": False, "error": f"auto_analyze failed: {exc}"}

    @server.tool(
        description=(
            "Perform a rapid health check on a dataset and recommend the optimal machine "
            "learning task (classification vs. regression) and evaluation metric. Summarizes "
            "dataset shape, missing values, target class balance, and baseline quality. "
            "Useful for quick pre-flight screening before launching training, but does not "
            "train models or perform deep remediation."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def inspect_data(
        csv_path: str,
        target_col: str,
    ) -> dict[str, Any]:
        """Perform a rapid health check and task recommendation on a CSV dataset."""
        try:
            df, err = _resolve_csv(csv_path)
            if err is not None:
                return err
            res = po.inspect_data(df=df, target_col=target_col)
            return to_jsonable(res)
        except Exception as exc:  # noqa: BLE001 - see the note above create_server()
            return {"ok": False, "error": f"inspect_data failed: {exc}"}

    @server.tool(
        description=(
            "Score dataset health on five weighted dimensions (Completeness 30%, Consistency 25%, "
            "Validity 20%, Uniqueness 15%, Timeliness 10%) and generate an actionable "
            "remediation plan. Returns the Data Quality Score (dqs, 0-100) and a grade of "
            "production_ready (85+), usable_with_caveats (65-84) or remediation_required, "
            "and flags duplicate rows, silent nulls such as \"N/A\" and -999 that pandas "
            "reads as real data, and statistical outliers."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def audit_data_quality(
        csv_path: str,
        target_col: str | None = None,
    ) -> dict[str, Any]:
        """Audit dataset quality across five dimensions and produce a remediation plan."""
        try:
            df, err = _resolve_csv(csv_path)
            if err is not None:
                return err
            res = po.audit_data_quality(df=df, target_col=target_col)
            return to_jsonable(res)
        except Exception as exc:  # noqa: BLE001 - see the note above create_server()
            return {"ok": False, "error": f"audit_data_quality failed: {exc}"}

    @server.tool(
        description=(
            "Detect drift per machine between baseline training data and a live production "
            "batch. Reports Population Stability Index and mean shift separately for each "
            "named asset and returns the machines that drifted by name, rather than pooling "
            "them: a pooled profile carries the spread BETWEEN machines in its own ruler, "
            "which both hides real drift on one asset and invents false alarms when the "
            "asset mix changes. Machines with too little data come back as skipped with a "
            "reason, never as a silent all-clear."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def check_asset_drift(
        train_csv_path: str,
        batch_csv_path: str,
        asset_col: str,
        threshold_pct: float = 0.2,
        min_rows: int = 30,
    ) -> dict[str, Any]:
        """Check covariate and numerical drift per machine between train and batch CSVs."""
        try:
            train_df, err = _resolve_csv(train_csv_path)
            if err is not None:
                return err
            batch_df, err = _resolve_csv(batch_csv_path)
            if err is not None:
                return err
            res = po.check_asset_drift(
                train_df=train_df,
                batch_df=batch_df,
                asset_col=asset_col,
                threshold_pct=threshold_pct,
                min_rows=min_rows,
            )
            return to_jsonable(res)
        except Exception as exc:  # noqa: BLE001 - see the note above create_server()
            return {"ok": False, "error": f"check_asset_drift failed: {exc}"}

    @server.tool(
        description=(
            "Calculate an Exponentially Weighted Moving Average (EWMA) statistical process "
            "control chart for continuous sensor readings. Detects subtle, persistent shifts "
            "and gradual tool/bearing degradation by accumulating past evidence. Estimates "
            "sigma from the moving range so a degrading series cannot inflate its own limits; "
            "baseline_n should cover a period the process was in control."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def calculate_ewma_chart(
        readings: list[float],
        baseline_n: int | None = None,
        lambda_weight: float = 0.2,
        n_sigmas: float = 3.0,
    ) -> dict[str, Any]:
        """Calculate an EWMA control chart from a list of continuous sensor readings."""
        try:
            if not isinstance(readings, (list, tuple)):
                return {"ok": False, "error": "readings must be a list of numbers."}
            res = po.calculate_ewma_chart(
                values=readings,
                baseline_n=baseline_n,
                lambda_weight=lambda_weight,
                n_sigmas=n_sigmas,
            )
            return to_jsonable(res)
        except Exception as exc:  # noqa: BLE001 - see the note above create_server()
            return {"ok": False, "error": f"calculate_ewma_chart failed: {exc}"}

    @server.tool(
        description=(
            "Evaluate the financial business case of a predictive maintenance model "
            "compared to a run-to-failure baseline. Calculates net cost savings, savings "
            "percentage, and breakdown avoidance rate by weighing unplanned breakdown costs "
            "against planned interventions and false-alarm inspections; recall alone cannot "
            "tell you whether a model makes money."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def calculate_maintenance_savings(
        true_positives: int,
        false_positives: int,
        false_negatives: int,
        cost_breakdown: float = 50000.0,
        cost_planned: float = 8000.0,
        cost_inspection: float = 1500.0,
    ) -> dict[str, Any]:
        """Calculate financial maintenance savings from a model confusion matrix."""
        try:
            res = po.calculate_maintenance_savings(
                true_positives=true_positives,
                false_positives=false_positives,
                false_negatives=false_negatives,
                cost_breakdown=cost_breakdown,
                cost_planned=cost_planned,
                cost_inspection=cost_inspection,
            )
            return to_jsonable(res)
        except Exception as exc:  # noqa: BLE001 - see the note above create_server()
            return {"ok": False, "error": f"calculate_maintenance_savings failed: {exc}"}

    @server.tool(
        description=(
            "Compute Overall Equipment Effectiveness (OEE) and its three components: "
            "Availability (run time vs. planned time), Performance (operating speed vs. "
            "ideal cycle time), and Quality (good parts vs. total parts), benchmarked "
            "against the 85% world-class standard. Requires both maintenance downtime "
            "numbers and production piece counts over the shift."
        ),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    def calculate_oee(
        planned_time_min: float,
        run_time_min: float,
        ideal_cycle_time_min: float,
        total_count: int,
        good_count: int,
    ) -> dict[str, Any]:
        """Compute Overall Equipment Effectiveness (OEE) for an asset over a shift."""
        try:
            res = po.calculate_oee(
                planned_time_min=planned_time_min,
                run_time_min=run_time_min,
                ideal_cycle_time_min=ideal_cycle_time_min,
                total_count=total_count,
                good_count=good_count,
            )
            return to_jsonable(res)
        except Exception as exc:  # noqa: BLE001 - see the note above create_server()
            return {"ok": False, "error": f"calculate_oee failed: {exc}"}

    return server


def main() -> None:
    """Run the PotatOpt MCP server over stdio transport."""
    create_server().run(transport="stdio")


if __name__ == "__main__":
    main()
