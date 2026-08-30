from __future__ import annotations

import logging  # the "potatopt" audit logger every guardrail/warning flows through
import os  # path handling in enable_audit_log
from datetime import datetime  # UTC timestamps in to_jsonable
from importlib.metadata import PackageNotFoundError
from importlib.metadata import (
    version as pkg_version,  # get_library_versions() ISO 9001 traceability record
)
from typing import Any  # loose return-type annotations for JSON-shaped dicts

import numpy as np  # arrays + math for downcasting, NaN/finite checks in to_jsonable and _finite_float
import pandas as pd  # DataFrame/Series is the data contract for every public function

from ._lazy import logger


def _is_numeric_series(series: pd.Series) -> bool:
    """
    np.issubdtype only understands numpy dtypes and raises TypeError on a pandas
    extension dtype such as CategoricalDtype, so every dtype test in this module
    goes through pandas' own predicate instead. A category column is categorical
    whatever its categories hold, which is the answer this code wants anyway.
    """
    return bool(pd.api.types.is_numeric_dtype(series))


def _is_text_series(series: pd.Series) -> bool:
    """
    True for a column of text, whichever dtype this pandas uses to hold one.

    Comparing `dtype == "object"` was correct until pandas 3.0, which gives a
    plain text column the dedicated `str` dtype instead. That comparison then
    answers False, so the column is neither encoded nor dropped and reaches the
    estimator as text - where it surfaces as
    `could not convert string to float: 'M01'`, naming a cell rather than the
    column or the cause. A machine id, a shift letter or a lot code is ordinary
    factory data, so on pandas 3 this broke `fit()` for most real inputs while
    every test on pandas 2 still passed.

    Routed through pandas' own predicates so the answer tracks pandas instead of
    a list of dtype spellings this module would have to keep in step by hand.
    """
    dtype = series.dtype
    return bool(pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype))


def _text_columns(frame: pd.DataFrame) -> list:
    """
    The frame's text columns, in order, without naming a dtype.

    `select_dtypes` cannot express this across both pandas majors: pandas 2
    rejects `"str"` outright with `TypeError: numpy string dtypes are not
    allowed`, while pandas 2's accepted `["object", "string"]` only still catches
    text on pandas 3 through a deprecated fallback that pandas 4 removes. There
    is no single argument list that is correct on both, so the selection goes
    through the same predicate as every other text test here.

    A duplicated column name makes `frame[col]` a DataFrame rather than a
    Series; the first column is taken, matching how the rest of this module
    handles that case.
    """
    columns = []
    for col in dict.fromkeys(frame.columns):
        series = frame[col]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        if _is_text_series(series):
            columns.append(col)
    return columns


def _finite_float(value: Any, name: str) -> tuple[float | None, str | None]:
    """
    Coerce a control-chart parameter to a finite float.

    Returns `(value, None)` on success or `(None, message)` on failure, so the
    caller can hand the message back in an error dict instead of raising - the
    contract both chart functions document.

    NaN and infinity are rejected explicitly because they slip past ordinary
    comparisons: every comparison against NaN is False, so a plain `<= 0` guard
    lets it through, and it then propagates silently into the limits.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{name} must be a number, got {value!r}."
    if not np.isfinite(number):
        return None, f"{name} must be a finite number, got {value!r}."
    return number, None


def _require_frame(df: Any, name: str, columns: tuple[str, ...]) -> str | None:
    """
    Return an error sentence if `df` is unusable, otherwise None.

    Shared by the reliability metrics so a missing column reports which columns
    the frame actually has - the same courtesy `auto_analyze` extends when the
    target column is wrong, and the difference between a caller who can fix the
    call and one who has to guess.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return f"{name} must be a pandas DataFrame."
    if df.empty:
        return f"{name} is empty."
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return f"{name} is missing column(s) {missing}. Columns present: {list(df.columns)}."
    return None


def enable_audit_log(filepath: str = "potatopt_audit.log", level: int = logging.INFO) -> str | None:
    """
    Persist every guardrail and drift event to a timestamped log file.
    Required for ISO 9001 traceability: the operator must be able to prove
    afterwards which production batches triggered a warning and when.
    """
    try:
        abs_path = os.path.abspath(filepath)
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == abs_path:
                return abs_path

        parent_dir = os.path.dirname(abs_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        file_handler = logging.FileHandler(abs_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        file_handler.setLevel(level)
        logger.addHandler(file_handler)
        return abs_path
    except (OSError, ValueError, TypeError):
        return None


def get_library_versions() -> dict[str, Any]:
    """
    Collect installed versions of the runtime stack for ISO 9001 reproducibility.
    """
    names = ["numpy", "pandas", "scipy", "scikit-learn", "flaml", "lightgbm", "xgboost", "shap", "joblib"]
    versions = {}
    for name in names:
        try:
            versions[name] = pkg_version(name)
        except (PackageNotFoundError, ValueError, TypeError):
            versions[name] = None
    return versions


def to_jsonable(value: Any) -> Any:
    """
    Convert NumPy / pandas output into plain Python that `json.dumps` accepts.

    Written for the tool-call boundary: an MCP server, a REST endpoint, or a
    dashboard all need the engine's output as ordinary dicts, lists, numbers and
    strings. Every stateless utility in this module already returns JSON-clean
    dicts; this covers the remaining cases - `predict()` returns an ndarray,
    `get_feature_importance()` returns a DataFrame - without changing either
    method's contract.

    NaN and Infinity become None rather than passing through, because
    `json.dumps` would otherwise emit the bare tokens `NaN` / `Infinity`, which
    are not valid JSON and are rejected by strict parsers such as
    JavaScript's `JSON.parse`.

    Anything it does not recognise is rendered with `str()` rather than raising,
    so a serialisation boundary never brings down a request.

    Returns:
    --------
    A structure built only from dict, list, str, int, float, bool and None.
    """
    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, (np.bool_,)):
        return bool(value)

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        as_float = float(value)
        # NaN/Inf are not representable in JSON.
        return as_float if np.isfinite(as_float) else None

    if isinstance(value, (datetime,)):
        return value.isoformat()

    if isinstance(value, pd.DataFrame):
        return [to_jsonable(record) for record in value.to_dict(orient="records")]

    if isinstance(value, (pd.Series, pd.Index)):
        return [to_jsonable(item) for item in value.tolist()]

    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]

    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]

    return str(value)
