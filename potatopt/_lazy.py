from __future__ import annotations

import contextlib  # contextmanager for _quiet_dependency_warnings() - scoped, not global, suppression
import logging  # the "potatopt" audit logger every guardrail/warning flows through
import warnings  # scoped suppression of third-party noise; see _quiet_dependency_warnings()
from collections.abc import Iterator  # return annotation for the context manager below


@contextlib.contextmanager
def _quiet_dependency_warnings() -> Iterator[None]:
    """
    Silence third-party warning noise for the duration of ONE call into a dependency.

    This module used to call `warnings.filterwarnings("ignore")` at import time. That
    is a process-global mutation: merely writing `import potatopt` switched off every
    warning in the importing application, including warnings raised by the caller's
    own code, for the rest of the run. A library has no business doing that, so the
    suppression now wraps only the specific calls that are noisy - FLAML's search and
    SHAP's explainers - and the caller's filters are restored on the way out.

    Known caveat: Python's warning filters are global to the process, not to the
    thread, so while this block is open another thread's warnings are suppressed too.
    That is a narrow, bounded window and is accepted deliberately; the alternative it
    replaced suppressed them permanently.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# ------------------------------------------------------------------------------
# Optional heavy backends, imported the first time they are actually needed.
#
# FLAML and SHAP together dominate both install size and import time, and neither
# is touched unless you call fit() or ask for explanations. Keeping them out of
# the module's import path is what lets the core install be numpy, pandas, scipy
# and scikit-learn - the difference between a library that runs on low-spec
# hardware and one that merely claims to.
# ------------------------------------------------------------------------------

def _load_automl():
    """
    Return FLAML's AutoML class, importing it on first use.

    Cached into module globals so `potatopt.AutoML` and the class used inside
    fit() are the same object - patching one patches the other.
    """
    automl = globals().get("AutoML")
    if automl is None:
        try:
            from flaml import AutoML as _AutoML
        except ImportError as exc:
            raise ImportError(
                "PotatOptEngine.fit() needs the FLAML AutoML backend, which is not installed. "
                "Install it with:  pip install potatopt[automl]"
            ) from exc
        globals()["AutoML"] = _AutoML
        automl = _AutoML
    return automl


def _load_shap():
    """
    Return the shap module, importing it on first use.

    Cached into module globals for the same reason as _load_automl.
    """
    module = globals().get("shap")
    if module is None:
        try:
            import shap as _shap
        except ImportError as exc:
            raise ImportError(
                "get_shap_values() needs the SHAP explainability backend, which is not installed. "
                "Install it with:  pip install potatopt[xai]"
            ) from exc
        globals()["shap"] = _shap
        module = _shap
    return module


# ISO 9001 audit trail: all guardrail events flow through one named logger so
# they can be silenced, redirected, or persisted to a file by the operator.
logger = logging.getLogger("potatopt")

logger.setLevel(logging.INFO)
if not logger.handlers:
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(_console_handler)
