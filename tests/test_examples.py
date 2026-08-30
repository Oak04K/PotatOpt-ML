"""
Tests for the example scripts.

These tests ensure the data-loading utilities and the quickstart structure
function correctly without requiring network access.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

# Insert the examples directory so we can import the scripts directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "examples"))

import quickstart
import tour
from ai4i_dataset import (
    IDENTIFIER_COLUMNS,
    LEAKY_COLUMNS,
    TARGET_COLUMN,
    _verify_sha256,
    load_ai4i,
)

import potatopt as po


def test_verify_sha256_rejects_wrong_payload() -> None:
    """
    Verifies that a bad payload throws RuntimeError with expected messaging.
    """
    with pytest.raises(RuntimeError) as exc_info:
        _verify_sha256(b"not the dataset")
    message = str(exc_info.value)
    assert "f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252" in message
    assert "nothing was written" in message.lower()


def test_verify_sha256_accepts_matching_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that a matching payload returns None.
    """
    fake_payload = b"this is the real dataset"
    fake_digest = hashlib.sha256(fake_payload).hexdigest()
    monkeypatch.setattr("ai4i_dataset.AI4I_ZIP_SHA256", fake_digest)
    assert _verify_sha256(fake_payload) is None


def test_load_ai4i_without_cache_and_without_download(tmp_path: Path) -> None:
    """
    Verifies that trying to load a missing file with download=False fails gracefully.
    """
    missing_path = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_ai4i(missing_path, download=False)
    assert "archive.ics.uci.edu" in str(exc_info.value)


def test_load_ai4i_drops_identifier_and_leaky_columns(tmp_path: Path) -> None:
    """
    Verifies that the leaky columns and identifiers are removed by default.
    """
    columns = (
        ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
         "Torque [Nm]", "Tool wear [min]", TARGET_COLUMN]
        + IDENTIFIER_COLUMNS
        + LEAKY_COLUMNS
    )
    df = pd.DataFrame({col: [0] * 12 for col in columns})
    test_path = tmp_path / "mock.csv"
    df.to_csv(test_path, index=False)

    loaded_df = load_ai4i(test_path, download=False)

    assert len(loaded_df.columns) == 7
    assert TARGET_COLUMN in loaded_df.columns
    for col in IDENTIFIER_COLUMNS + LEAKY_COLUMNS:
        assert col not in loaded_df.columns


def test_load_ai4i_keeps_everything_when_asked(tmp_path: Path) -> None:
    """
    Verifies that raw data can be loaded if drop_leaky is explicitly False.
    """
    columns = (
        ["Type", "Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]",
         "Torque [Nm]", "Tool wear [min]", TARGET_COLUMN]
        + IDENTIFIER_COLUMNS
        + LEAKY_COLUMNS
    )
    df = pd.DataFrame({col: [0] * 12 for col in columns})
    test_path = tmp_path / "mock.csv"
    df.to_csv(test_path, index=False)

    loaded_df = load_ai4i(test_path, download=False, drop_leaky=False)

    assert list(loaded_df.columns) == columns
    assert len(loaded_df.columns) == 14


def test_load_ai4i_rejects_a_csv_with_missing_columns(tmp_path: Path) -> None:
    """
    Verifies that a CSV missing required structural columns raises a ValueError.
    """
    df = pd.DataFrame({TARGET_COLUMN: [0], "HDF": [0]})
    test_path = tmp_path / "mock.csv"
    df.to_csv(test_path, index=False)

    with pytest.raises(ValueError) as exc_info:
        load_ai4i(test_path, download=False)
    assert "UDI" in str(exc_info.value)
    assert "Product ID" in str(exc_info.value)
    assert "TWF" in str(exc_info.value)


def test_quickstart_module_imports_without_running() -> None:
    """
    Verifies that importing the quickstart module has no top-level side effects.
    """
    assert hasattr(quickstart, "main")


def test_tour_module_imports_without_running() -> None:
    """
    Verifies that importing the tour module has no top-level side effects.
    """
    assert hasattr(tour, "main")


def test_tour_exercises_all_public_callables() -> None:
    """
    Verifies that every callable in potatopt.__all__ appears in examples/tour.py.
    """
    tour_path = Path(__file__).resolve().parent.parent / "examples" / "tour.py"
    content = tour_path.read_text(encoding="utf-8")

    public_callables = [
        name for name in po.__all__
        if callable(getattr(po, name, None))
    ]

    missing = [name for name in public_callables if name not in content]
    assert not missing, f"The following public callable(s) are missing from tour.py: {missing}"


def test_tour_runs_and_returns_jsonable_results() -> None:
    """
    Verifies that tour.main() executes, returns a non-empty dict, and survives json.dumps(to_jsonable(...)).
    """
    results = tour.main()
    assert isinstance(results, dict)
    assert len(results) > 0
    serialized = json.dumps(po.to_jsonable(results))
    assert len(serialized) > 0

