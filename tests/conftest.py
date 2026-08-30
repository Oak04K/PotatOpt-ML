# These fixtures are session-scoped because each fit() call spends its full AutoML time budget.

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import potatopt as po

N = 300


# Return a single session-scoped DataFrame; tests that mutate must copy themselves.
@pytest.fixture(scope="session")
def clean_frame():
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "temp_c": rng.normal(50, 4, N),
        "pressure_bar": rng.normal(10, 1, N),
        "cycle_time": rng.normal(30, 3, N),
        "line": rng.choice(["A", "B", "C"], N),
    })


@pytest.fixture(scope="session")
def signal_frame():
    rng = np.random.default_rng(7)
    sig = rng.normal(0, 1, N)
    x = pd.DataFrame({
        "temp_c": 50 + sig * 4,
        "pressure_bar": 10 + rng.normal(0, 2, N),
        "cycle_time": 30 + rng.normal(0, 3, N),
    })
    y = pd.Series((sig > 1.2).astype(int))
    return (x, y)


@pytest.fixture(scope="session")
def binary_engine(signal_frame):
    x, y = signal_frame
    return po.PotatOptEngine(task="classification", time_budget=5).fit(x, y)


@pytest.fixture(scope="session")
def regression_frame():
    rng = np.random.default_rng(13)
    x = pd.DataFrame({"x1": rng.normal(0, 1, N), "x2": rng.normal(0, 1, N)})
    y = pd.Series(100 + 5 * x["x1"] + rng.normal(0, 1, N))
    return (x, y)


@pytest.fixture(scope="session")
def regression_engine(regression_frame):
    x, y = regression_frame
    return po.PotatOptEngine(task="regression", time_budget=5).fit(x, y)


@pytest.fixture(scope="session")
def anomaly_engine():
    rng = np.random.default_rng(5)
    n_rows = 120
    x = pd.DataFrame({
        "s1": rng.normal(0, 1, n_rows),
        "s2": rng.normal(0, 1, n_rows),
        "s3": rng.normal(0, 1, n_rows),
    })
    y = pd.Series([0] * 117 + [1] * 3)
    return po.PotatOptEngine(task="classification", time_budget=5).fit(x, y)


class _ListLogHandler(logging.Handler):
    def __init__(self, target_list):
        super().__init__()
        self.target_list = target_list

    def emit(self, record):
        self.target_list.append(record.getMessage())


@pytest.fixture
def log_capture():
    captured = []
    handler = _ListLogHandler(captured)
    po.logger.addHandler(handler)
    try:
        yield captured
    finally:
        po.logger.removeHandler(handler)
