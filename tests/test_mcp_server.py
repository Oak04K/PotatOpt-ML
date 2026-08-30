"""
Tests for the PotatOpt Model Context Protocol (MCP) server adapter.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("mcp", reason="MCP server tests require the mcp extra (mcp>=2.1,<3)")

from potatopt import __version__
from potatopt.mcp_server import _resolve_csv, create_server

EXPECTED_TOOL_NAMES = {
    "auto_analyze",
    "inspect_data",
    "audit_data_quality",
    "check_asset_drift",
    "calculate_ewma_chart",
    "calculate_maintenance_savings",
    "calculate_oee",
}


def _extract_tool_payload(call_result: Any) -> tuple[dict[str, Any], str]:
    """
    Extract the JSON-decoded payload and raw text from an MCP call_tool result.
    In mcp>=2.1, call_tool returns a sequence of TextContent objects or CallToolResult.
    """
    if hasattr(call_result, "content"):
        contents = call_result.content
    elif isinstance(call_result, (list, tuple)):
        contents = call_result
    else:
        raise TypeError(f"Unexpected call_tool return type: {type(call_result)}")

    assert len(contents) > 0, "Tool call returned empty content sequence"
    raw_text = contents[0].text
    parsed = json.loads(raw_text)
    return parsed, raw_text


def test_registered_tools_count_and_names() -> None:
    """Verify exactly seven tools are registered with their expected names."""
    server = create_server()

    async def _run() -> None:
        tools = await server.list_tools()
        names = {tool.name for tool in tools}
        assert names == EXPECTED_TOOL_NAMES
        assert len(tools) == 7

    asyncio.run(_run())


def test_all_tools_declare_readonly_hint_and_description() -> None:
    """Verify every tool declares readOnlyHint=True and has a non-empty description."""
    server = create_server()

    async def _run() -> None:
        tools = await server.list_tools()
        assert len(tools) == 7
        for tool in tools:
            assert tool.description is not None, f"Tool {tool.name} description is None"
            assert len(tool.description.strip()) > 0, f"Tool {tool.name} description is empty"
            assert tool.annotations is not None, f"Tool {tool.name} annotations is None"
            # The constructor keyword is readOnlyHint (the wire name); the field
            # reads back as read_only_hint, the same snake_case flip as
            # Tool.input_schema. Asserting the wire spelling raises AttributeError
            # rather than failing, which is why this is spelled out.
            assert tool.annotations.read_only_hint is True, f"Tool {tool.name} is not read-only"

    asyncio.run(_run())


def test_auto_analyze_schema_requirements() -> None:
    """Verify auto_analyze input_schema requires csv_path and target, and excludes save_to."""
    server = create_server()

    async def _run() -> None:
        tools = await server.list_tools()
        tool_map = {tool.name: tool for tool in tools}
        tool = tool_map["auto_analyze"]
        schema = tool.input_schema
        assert isinstance(schema, dict)

        required = schema.get("required", [])
        assert "csv_path" in required
        assert "target" in required

        properties = schema.get("properties", {})
        assert "save_to" not in properties
        assert "csv_path" in properties
        assert "target" in properties
        assert "cost_scrap" in properties
        assert "cost_fa" in properties
        assert "cost_insp" in properties
        assert "time_budget" in properties
        assert "random_state" in properties

    asyncio.run(_run())


def test_missing_csv_returns_error_dict_and_does_not_raise(tmp_path: Path) -> None:
    """Verify calling auto_analyze with a missing CSV returns an error dict without raising."""
    server = create_server()
    missing_csv = tmp_path / "does_not_exist.csv"

    async def _run() -> None:
        try:
            result = await server.call_tool(
                "auto_analyze",
                {"csv_path": str(missing_csv), "target": "defect"},
            )
        # Catching everything is the assertion, not an oversight: the SDK turns
        # any escaping exception into an opaque UnexpectedToolError, so "nothing
        # at all gets out" is precisely what this test has to prove.
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"server.call_tool raised unexpectedly: {exc}")

        data, _ = _extract_tool_payload(result)
        assert data.get("ok") is False
        assert "error" in data
        assert isinstance(data["error"], str)
        assert len(data["error"]) > 0

    asyncio.run(_run())


def test_mcp_root_confinement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify POTATOPT_MCP_ROOT restricts reading to inside the configured directory."""
    root_dir = tmp_path / "allowed_root"
    root_dir.mkdir()
    outside_dir = tmp_path / "outside_root"
    outside_dir.mkdir()

    inside_csv = root_dir / "inside.csv"
    inside_csv.write_text("vibration,defect\n0.1,0\n0.9,1\n0.2,0\n0.8,1\n", encoding="utf-8")

    outside_csv = outside_dir / "outside.csv"
    outside_csv.write_text("vibration,defect\n0.1,0\n0.9,1\n0.2,0\n0.8,1\n", encoding="utf-8")

    monkeypatch.setenv("POTATOPT_MCP_ROOT", str(root_dir))
    server = create_server()

    async def _run() -> None:
        # Path outside root must be rejected with an error mentioning the root
        outside_res = await server.call_tool(
            "inspect_data",
            {"csv_path": str(outside_csv), "target_col": "defect"},
        )
        outside_data, _ = _extract_tool_payload(outside_res)
        assert outside_data.get("ok") is False
        assert str(os.path.realpath(str(root_dir))) in outside_data.get("error", "")

        # Path inside root must load normally
        inside_res = await server.call_tool(
            "inspect_data",
            {"csv_path": str(inside_csv), "target_col": "defect"},
        )
        inside_data, _ = _extract_tool_payload(inside_res)
        assert "error" not in inside_data or inside_data.get("error") is None
        assert inside_data.get("total_rows") == 4

    asyncio.run(_run())


def test_nan_survives_as_valid_json(tmp_path: Path) -> None:
    """Verify NaNs in results are sanitized by to_jsonable and parse cleanly without bare NaN tokens."""
    csv_file = tmp_path / "data_with_nan.csv"
    csv_file.write_text(
        "sensor_a,sensor_b,defect\n10.5,,0\n,0.8,1\n12.1,0.5,0\n15.0,0.9,1\n",
        encoding="utf-8",
    )

    server = create_server()

    async def _run() -> None:
        result = await server.call_tool(
            "audit_data_quality",
            {"csv_path": str(csv_file), "target_col": "defect"},
        )
        data, raw_text = _extract_tool_payload(result)
        assert isinstance(data, dict)
        assert not re.search(r"\bNaN\b", raw_text), f"Found bare NaN token in raw JSON: {raw_text}"
        assert not re.search(r"\bInfinity\b", raw_text), f"Found bare Infinity token in raw JSON: {raw_text}"

        inspect_res = await server.call_tool(
            "inspect_data",
            {"csv_path": str(csv_file), "target_col": "defect"},
        )
        inspect_data, inspect_text = _extract_tool_payload(inspect_res)
        assert isinstance(inspect_data, dict)
        assert not re.search(r"\bNaN\b", inspect_text), f"Found bare NaN token in raw JSON: {inspect_text}"

    asyncio.run(_run())


def test_importing_potatopt_does_not_load_mcp() -> None:
    """Verify importing potatopt does not import mcp into sys.modules."""
    project_root = Path(__file__).resolve().parent.parent
    code = (
        f"import sys; sys.path.insert(0, r'{project_root}'); import potatopt; "
        "print('mcp' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == "False", completed.stdout + completed.stderr


def test_stateless_calculation_tools() -> None:
    """Verify stateless calculation tools (EWMA, maintenance savings, OEE) execute correctly."""
    server = create_server()

    async def _run() -> None:
        # EWMA chart
        ewma_res = await server.call_tool(
            "calculate_ewma_chart",
            {"readings": [10.0] * 5 + [12.0] * 5, "baseline_n": 5},
        )
        ewma_data, _ = _extract_tool_payload(ewma_res)
        assert ewma_data.get("out_of_control") is True

        # Maintenance savings
        savings_res = await server.call_tool(
            "calculate_maintenance_savings",
            {"true_positives": 18, "false_positives": 25, "false_negatives": 2},
        )
        savings_data, _ = _extract_tool_payload(savings_res)
        assert savings_data.get("cost_savings") == 691_500.0

        # OEE
        oee_res = await server.call_tool(
            "calculate_oee",
            {
                "planned_time_min": 480.0,
                "run_time_min": 420.0,
                "ideal_cycle_time_min": 0.5,
                "total_count": 800,
                "good_count": 760,
            },
        )
        oee_data, _ = _extract_tool_payload(oee_res)
        assert "oee" in oee_data
        assert 0.0 < oee_data["oee"] < 1.0

    asyncio.run(_run())


def test_check_asset_drift_tool(tmp_path: Path) -> None:
    """Verify check_asset_drift tool runs correctly over CSV files."""
    train_csv = tmp_path / "train.csv"
    train_csv.write_text(
        "machine_id,temp_c,vibration\n" + "\n".join(f"M1,50.0,{0.1 + i * 0.001}" for i in range(50)),
        encoding="utf-8",
    )
    batch_csv = tmp_path / "batch.csv"
    batch_csv.write_text(
        "machine_id,temp_c,vibration\n" + "\n".join(f"M1,52.0,{0.1 + i * 0.001}" for i in range(50)),
        encoding="utf-8",
    )

    server = create_server()

    async def _run() -> None:
        result = await server.call_tool(
            "check_asset_drift",
            {
                "train_csv_path": str(train_csv),
                "batch_csv_path": str(batch_csv),
                "asset_col": "machine_id",
            },
        )
        data, _ = _extract_tool_payload(result)
        assert isinstance(data, dict)
        # The per-machine contract, asserted by name. A pooled report would not
        # carry these keys, and naming the drifted machine is the whole point of
        # preferring this over check_data_drift.
        assert set(data) >= {
            "asset_col",
            "assets_checked",
            "assets_drifted",
            "assets_skipped",
            "drift_detected",
            "per_asset",
        }, f"unexpected keys from check_asset_drift: {sorted(data)}"
        assert data["asset_col"] == "machine_id"
        assert "M1" in data["per_asset"]
        assert data["per_asset"]["M1"]["status"] == "checked"

    asyncio.run(_run())


def test_server_metadata() -> None:
    """Verify server name and version match package metadata."""
    server = create_server()
    assert server.name == "potatopt"
    assert server.version == __version__


def test_resolve_csv_edge_cases(tmp_path: Path) -> None:
    """Verify _resolve_csv handles empty string, nonexistent path, and empty files."""
    df, err = _resolve_csv("")
    assert df is None
    assert err is not None
    assert err.get("ok") is False

    df, err = _resolve_csv(str(tmp_path / "absent.csv"))
    assert df is None
    assert err is not None
    assert err.get("ok") is False

    empty_csv = tmp_path / "empty.csv"
    empty_csv.write_text("", encoding="utf-8")
    df, err = _resolve_csv(str(empty_csv))
    assert df is None
    assert err is not None
    assert err.get("ok") is False
    assert "empty" in err.get("error", "").lower()
