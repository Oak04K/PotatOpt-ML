"""
chart_engine - draw the dictionaries `potatopt` returns.

This module renders; it does not calculate. Every function takes the dictionary
a potatopt function already produced and turns it into a figure. Nothing here
recomputes a limit, a sigma or a percentage, which is the point: a chart that
derives its own numbers can disagree with the report beside it, and that
disagreement is close to impossible to spot by eye.

    result = po.calculate_ewma_chart(readings, baseline_n=25)   # the numbers
    fig = ce.plot_ewma(result)                                   # the picture

Because the input is a plain dictionary, a chart can be drawn from a report that
was saved as JSON months ago on another machine, with no raw data present.

matplotlib is imported lazily for the same reason FLAML and SHAP are inside
potatopt: importing this module must not cost anything to a caller who only
wanted the numbers. Figures are returned rather than shown or written, so the
caller decides whether they become a PNG for a report or bytes for a web page.
"""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import cycle only matters to type checkers
    from matplotlib.figure import Figure

__all__ = [
    "ACCENT",
    "ALERT",
    "GRID",
    "INK",
    "MUTED",
    "PAPER",
    "figure_to_data_uri",
    "figure_to_png",
    "plot_availability",
    "plot_confusion_matrix",
    "plot_correlation_heatmap",
    "plot_cusum",
    "plot_ewma",
    "plot_feature_importance",
    "plot_oee",
    "plot_pareto",
]

# The Bauhaus paper palette settled on in Phase 1. Printed reports and a screen
# in a bright workshop both have to stay readable, which rules out a dark theme
# and anything that relies on hue alone to carry meaning.
PAPER = "#f2eedf"
INK = "#181818"
GRID = "#cccccc"
MUTED = "#8a8a8a"
ALERT = "#c0392b"
ACCENT = "#2c6fbb"


def _pyplot() -> Any:
    """Import matplotlib on first use, with a message that names the fix."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # No display on a factory server, and none needed.
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "chart_engine needs matplotlib, which is not installed. "
            "Install it with:  pip install potatopt[viz]"
        ) from exc
    return plt


def _canvas(width: float, height: float) -> tuple[Any, Any]:
    plt = _pyplot()
    figure, axes = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(PAPER)
    axes.set_facecolor(PAPER)
    axes.grid(True, color=GRID, linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(MUTED)
    axes.tick_params(colors=INK, labelsize=9)
    return figure, axes


def _guard(payload: Any, *required: str) -> None:
    """
    Reject a payload that is not the dictionary this plot draws.

    chart_engine is allowed to raise, unlike potatopt: its caller is a person or
    a web page, not a language model, and a ValueError naming the missing key is
    more use here than an error dictionary that has no figure to travel in.
    This exception to the library's never-raise rule is deliberate.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"Expected the dictionary a potatopt function returned, got {type(payload).__name__}.")
    if "error" in payload:
        raise ValueError(f"Cannot draw a failed result: {payload['error']}")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"This payload is missing {missing}; it does not look like the expected result.")


def plot_ewma(chart: dict[str, Any], title: str = "EWMA control chart", ylabel: str = "Reading") -> Figure:
    """
    EWMA with its exact time-varying limits and every violation marked.

    The limits widen over the first samples rather than sitting at their
    asymptotic width, which is why the early points are drawn against a curve
    and not a straight line. Flattening them would make the chart look tidier
    and hide signals in exactly the region a new process is least understood.
    """
    _guard(chart, "ewma", "ucl", "lcl", "target")
    figure, axes = _canvas(9, 4)
    x = range(len(chart["ewma"]))

    axes.plot(x, chart["ucl"], color=ALERT, linewidth=1.0, linestyle="--", label="Control limits")
    axes.plot(x, chart["lcl"], color=ALERT, linewidth=1.0, linestyle="--")
    axes.axhline(chart["target"], color=MUTED, linewidth=1.0, linestyle=":", label="Target")
    axes.plot(x, chart["ewma"], color=INK, linewidth=1.4, label="EWMA")

    violations = chart.get("violations") or []
    if violations:
        axes.scatter(violations, [chart["ewma"][i] for i in violations],
                     color=ALERT, s=42, zorder=5, label=f"Out of control ({len(violations)})")

    if chart.get("autocorrelation_warning"):
        # Loud on purpose. Moving-range sigma assumes independent readings, and
        # a random-walk sensor collapses it, so the chart alarms nearly
        # everywhere. A viewer who cannot see that warning will act on noise.
        axes.text(0.01, 0.02, chart["autocorrelation_warning"], transform=axes.transAxes,
                  fontsize=8, color=ALERT, va="bottom", wrap=True)

    axes.set_title(title, color=INK, fontsize=11, loc="left")
    axes.set_xlabel("Sample", color=INK, fontsize=9)
    axes.set_ylabel(ylabel, color=INK, fontsize=9)
    axes.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure


def plot_cusum(chart: dict[str, Any], title: str = "CUSUM control chart") -> Figure:
    """Two-sided tabular CUSUM, both arms against the decision interval."""
    _guard(chart, "cusum_high", "cusum_low", "decision_interval")
    figure, axes = _canvas(9, 4)
    x = range(len(chart["cusum_high"]))
    interval = chart["decision_interval"]

    axes.axhline(interval, color=ALERT, linewidth=1.0, linestyle="--", label="Decision interval")
    axes.plot(x, chart["cusum_high"], color=INK, linewidth=1.4, label="Upper arm (rising)")
    axes.plot(x, chart["cusum_low"], color=ACCENT, linewidth=1.4, label="Lower arm (falling)")

    for key, values, colour in (("violations_high", chart["cusum_high"], ALERT),
                                ("violations_low", chart["cusum_low"], ACCENT)):
        points = chart.get(key) or []
        if points:
            axes.scatter(points, [values[i] for i in points], color=colour, s=42, zorder=5)

    axes.set_title(title, color=INK, fontsize=11, loc="left")
    axes.set_xlabel("Sample", color=INK, fontsize=9)
    axes.set_ylabel("Cumulative deviation", color=INK, fontsize=9)
    axes.legend(frameon=False, fontsize=8, loc="upper left")
    figure.tight_layout()
    return figure


def plot_pareto(pareto: dict[str, Any], title: str | None = None, top_n: int = 12) -> Figure:
    """
    Bars for each cause, a cumulative line, and the cut-off drawn across both.

    The vital few are filled and the rest are hollow, so which causes the chart
    is actually pointing at survives being printed in black and white or
    photocopied onto a noticeboard - the two places a Pareto usually ends up.

    The subtitle names what was counted. A Pareto of events and a Pareto of
    hours rank the same plant differently, and a chart that does not say which
    one it drew invites the reader to act on the wrong one.
    """
    _guard(pareto, "categories", "total", "measured_by")
    rows = pareto["categories"][:top_n]
    if not rows:
        raise ValueError("This Pareto has no categories to draw.")

    figure, axes = _canvas(9, 4.6)
    labels = [row["category"] for row in rows]
    values = [row["value"] for row in rows]
    positions = range(len(rows))

    axes.bar(
        positions, values, zorder=3, width=0.68,
        color=[INK if row["is_vital_few"] else PAPER for row in rows],
        edgecolor=INK, linewidth=1.0,
    )
    axes.set_ylabel(f"{pareto['measured_by']}", color=INK, fontsize=9)
    axes.set_xticks(list(positions))
    axes.set_xticklabels(labels, rotation=30, ha="right", fontsize=9, color=INK)

    cumulative = axes.twinx()
    cumulative.plot(positions, [row["cumulative_percentage"] for row in rows],
                    color=ALERT, marker="o", markersize=4, linewidth=1.4, zorder=4)
    cumulative.axhline(pareto["cutoff"] * 100.0, color=MUTED, linestyle=":", linewidth=1.0)
    cumulative.set_ylim(0, 105)
    cumulative.set_ylabel("Cumulative %", color=ALERT, fontsize=9)
    cumulative.tick_params(colors=ALERT, labelsize=9)
    cumulative.grid(False)
    for side in ("top", "left"):
        cumulative.spines[side].set_visible(False)

    heading = title or "Pareto of causes"
    axes.set_title(
        f"{heading}\nmeasured by {pareto['measured_by']} - "
        f"{pareto['vital_few_count']} of {len(pareto['categories'])} causes "
        f"account for {pareto['vital_few_share']:.1f}%",
        color=INK, fontsize=11, loc="left",
    )
    figure.tight_layout()
    return figure


def plot_oee(oee: dict[str, Any], title: str = "OEE") -> Figure:
    """
    The three factors and the product, against the world-class benchmark.

    Drawing the factors beside the result rather than the result alone is the
    whole value: an OEE of 0.69 says nothing about what to fix, while the same
    figure shown as 0.88 x 0.83 x 0.94 points straight at the slow cycle.
    """
    _guard(oee, "oee", "availability", "performance", "quality")
    figure, axes = _canvas(7, 4)
    labels = ["Availability", "Performance", "Quality", "OEE"]
    values = [oee["availability"], oee["performance"], oee["quality"], oee["oee"]]
    colours = [MUTED, MUTED, MUTED, INK]

    bars = axes.bar(labels, [v * 100 for v in values], color=colours, edgecolor=INK, linewidth=1.0, zorder=3, width=0.6)
    axes.axhline(oee.get("world_class_benchmark", 0.85) * 100, color=ALERT, linestyle="--", linewidth=1.0,
                 label=f"World class ({oee.get('world_class_benchmark', 0.85) * 100:.0f}%)")

    for bar, value in zip(bars, values, strict=True):
        axes.text(bar.get_x() + bar.get_width() / 2, value * 100 + 1.5,
                  f"{value * 100:.1f}%", ha="center", fontsize=9, color=INK)

    axes.set_ylim(0, max(105, max(values) * 100 + 10))
    axes.set_ylabel("Percent", color=INK, fontsize=9)
    axes.set_title(title, color=INK, fontsize=11, loc="left")
    axes.legend(frameon=False, fontsize=8, loc="lower right")

    for warning in (oee.get("warnings") or [])[:2]:
        # A performance above 100% is master data being wrong, not a record
        # month. It has to be visible on the chart people circulate.
        axes.text(0.01, -0.28, warning, transform=axes.transAxes, fontsize=7.5, color=ALERT, va="top", wrap=True)

    figure.tight_layout()
    return figure


def plot_availability(availability: dict[str, Any], title: str = "Availability") -> Figure:
    """
    Inherent against operational, with the waiting gap called out.

    The gap is the point of the chart. It is not a property of the equipment -
    no new machine removes it - so showing the two bars side by side is what
    turns "availability is 99%" into "and half a point of it is us waiting".
    """
    _guard(availability, "inherent_availability")
    operational = availability.get("operational_availability")
    if operational is None:
        raise ValueError("Operational availability was not computed, so there is no gap to draw.")

    figure, axes = _canvas(7, 3.6)
    inherent = availability["inherent_availability"]
    axes.barh(["Operational (what the plant gets)", "Inherent (what the machine can do)"],
              [operational * 100, inherent * 100],
              color=[ACCENT, INK], edgecolor=INK, linewidth=1.0, zorder=3, height=0.5)

    lost = availability.get("availability_lost_to_waiting") or 0.0
    axes.set_xlim(min(operational, inherent) * 100 - max(1.0, lost * 300), 100.4)
    for index, value in enumerate((operational, inherent)):
        axes.text(value * 100, index, f"  {value * 100:.2f}%", va="center", fontsize=9, color=INK)

    axes.set_title(f"{title}\n{lost * 100:.2f} points lost to waiting rather than repairing",
                   color=INK, fontsize=11, loc="left")
    axes.set_xlabel("Percent", color=INK, fontsize=9)
    figure.tight_layout()
    return figure


def plot_confusion_matrix(
    evaluation: dict[str, Any],
    title: str = "Confusion matrix",
    normalize: bool = False,
) -> Figure:
    """
    Draw a confusion matrix heatmap with actual rows and predicted columns.

    Counts or row-normalized proportions are printed in every cell, in INK on
    light cells and PAPER on dark ones for readability in greyscale.
    """
    _guard(evaluation, "confusion_matrix", "class_labels")
    import matplotlib.colors as mcolors
    import numpy as np

    cm = np.asarray(evaluation["confusion_matrix"], dtype=float)
    labels = [str(lbl) for lbl in evaluation["class_labels"]]
    n = len(labels)

    if cm.ndim != 2 or cm.shape[0] != n or cm.shape[1] != n:
        raise ValueError(f"confusion_matrix shape {cm.shape} does not match class_labels count {n}.")

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display_matrix = np.divide(cm, row_sums, out=np.zeros_like(cm), where=(row_sums != 0))
        vmin, vmax = 0.0, 1.0
    else:
        display_matrix = cm
        vmin = 0.0
        vmax = max(1.0, float(np.max(display_matrix)))

    cmap = mcolors.LinearSegmentedColormap.from_list("potatopt_cm", [PAPER, INK])
    figure, axes = _canvas(max(5.5, min(9.0, 3.5 + n * 0.7)), max(4.5, min(8.0, 3.0 + n * 0.7)))
    axes.grid(False)

    axes.imshow(display_matrix, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")

    axes.set_xticks(range(n))
    axes.set_yticks(range(n))
    axes.set_xticklabels(labels, color=INK, fontsize=9)
    axes.set_yticklabels(labels, color=INK, fontsize=9)
    axes.set_xlabel("Predicted label", color=INK, fontsize=9)
    axes.set_ylabel("Actual label", color=INK, fontsize=9)
    axes.set_title(title, color=INK, fontsize=11, loc="left")

    denom = vmax - vmin if vmax > vmin else 1.0
    for i in range(n):
        for j in range(n):
            val = display_matrix[i, j]
            norm_val = (val - vmin) / denom
            text_color = PAPER if norm_val > 0.5 else INK
            text = f"{val:.2f}" if normalize else (f"{int(val)}" if val == int(val) else f"{val:.1f}")
            axes.text(j, i, text, ha="center", va="center", color=text_color, fontsize=9.5, fontweight="bold")

    figure.tight_layout()
    return figure


def plot_correlation_heatmap(
    correlations: dict[str, Any],
    title: str = "Correlation",
    annotate: bool = True,
) -> Figure:
    """
    Diverging correlation heatmap with fixed -1 to +1 scale.

    Neutral correlations (0.0) sit on PAPER background, while strong negative
    correlations move toward ACCENT and strong positive correlations toward ALERT.
    Uncomputed or non-finite cells are rendered as blank spaces rather than zero.
    """
    _guard(correlations, "columns", "matrix")
    import matplotlib.colors as mcolors
    import numpy as np

    columns = [str(c) for c in correlations["columns"]]
    raw_matrix = correlations["matrix"]
    if not isinstance(raw_matrix, list):
        # TypeError, matching _guard(): the payload is the wrong shape, not a
        # right-shaped payload carrying a bad value.
        raise TypeError(f"correlations['matrix'] must be a list of lists, got {type(raw_matrix).__name__}.")

    n = len(columns)
    if n == 0 or len(raw_matrix) != n:
        raise ValueError(f"Matrix size ({len(raw_matrix)}) does not match columns count ({n}).")

    mat = np.zeros((n, n), dtype=float)
    mask = np.zeros((n, n), dtype=bool)

    for i in range(n):
        row = raw_matrix[i]
        if not isinstance(row, list) or len(row) != n:
            raise ValueError(f"Matrix row {i} length ({len(row) if isinstance(row, list) else 'invalid'}) does not match columns count ({n}).")
        for j in range(n):
            val = row[j]
            if val is None or not np.isfinite(val):
                mask[i, j] = True
                mat[i, j] = np.nan
            else:
                mat[i, j] = float(val)

    masked_mat = np.ma.array(mat, mask=mask)
    cmap = mcolors.LinearSegmentedColormap.from_list("potatopt_corr", [ACCENT, PAPER, ALERT])
    cmap.set_bad(color=PAPER)

    figure, axes = _canvas(max(6.0, min(10.0, 3.5 + n * 0.45)), max(5.0, min(9.0, 3.0 + n * 0.45)))
    axes.grid(False)

    axes.imshow(masked_mat, cmap=cmap, vmin=-1.0, vmax=1.0, origin="upper")

    axes.set_xticks(range(n))
    axes.set_yticks(range(n))
    axes.set_xticklabels(columns, rotation=35, ha="right", fontsize=8.5, color=INK)
    axes.set_yticklabels(columns, fontsize=8.5, color=INK)
    axes.set_title(title, color=INK, fontsize=11, loc="left")

    if annotate and n <= 15:
        for i in range(n):
            for j in range(n):
                if not mask[i, j]:
                    val = mat[i, j]
                    text_color = PAPER if abs(val) > 0.65 else INK
                    axes.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=8.5)

    if correlations.get("note"):
        axes.text(
            0.0, -0.18, str(correlations["note"]),
            transform=axes.transAxes, fontsize=8, color=MUTED, va="top", wrap=True,
        )

    figure.tight_layout()
    return figure


def plot_feature_importance(
    payload: Any,
    title: str = "Feature importance",
    top_n: int = 15,
) -> Figure:
    """
    Horizontal bar chart ranking features by attribution or model importance.

    Accepts output from `explain_predictions()` (ranking by mean |SHAP|),
    `get_feature_importance()` (ranking by model importance), or a list of dicts.
    """
    import pandas as pd

    if isinstance(payload, pd.DataFrame):
        if "feature" not in payload.columns or "importance" not in payload.columns:
            raise ValueError("DataFrame must contain 'feature' and 'importance' columns.")
        records = payload[["feature", "importance"]].to_dict(orient="records")
    elif isinstance(payload, dict):
        if "error" in payload:
            raise ValueError(f"Cannot draw a failed result: {payload['error']}")
        if payload.get("available") is False:
            reason = payload.get("reason") or "Feature attributions are unavailable."
            raise ValueError(f"Feature attributions unavailable: {reason}")
        if "feature_attributions" not in payload:
            raise ValueError("Payload dictionary is missing 'feature_attributions'.")
        records = payload["feature_attributions"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise TypeError(f"Expected a dict, DataFrame, or list of feature records, got {type(payload).__name__}.")

    if not records:
        raise ValueError("No feature importance data to plot.")

    first = records[0]
    if not isinstance(first, dict):
        raise TypeError(f"Expected dict records in feature importance list, got {type(first).__name__}.")

    if "mean_abs_shap" in first:
        metric_key = "mean_abs_shap"
        xlabel = "mean |SHAP|"
    elif "importance" in first:
        metric_key = "importance"
        xlabel = "model importance"
    else:
        raise ValueError("Feature records must contain either 'mean_abs_shap' or 'importance'.")

    top_records = records[:top_n]
    features = [str(r.get("feature", f"feat_{i}")) for i, r in enumerate(top_records)]
    values = [float(r.get(metric_key, 0.0)) for r in top_records]

    figure, axes = _canvas(8, max(3.5, min(8.0, 1.2 + len(top_records) * 0.32)))

    y_positions = list(range(len(top_records) - 1, -1, -1))
    bars = axes.barh(y_positions, values, color=INK, edgecolor=INK, height=0.6, zorder=3)
    axes.set_yticks(y_positions)
    axes.set_yticklabels(features, fontsize=9, color=INK)
    axes.set_xlabel(xlabel, color=INK, fontsize=9)
    axes.set_title(title, color=INK, fontsize=11, loc="left")

    max_val = max(values) if values else 1.0
    axes.set_xlim(0, max_val * 1.15 if max_val > 0 else 1.0)
    for bar, val in zip(bars, values, strict=True):
        axes.text(val + max_val * 0.015, bar.get_y() + bar.get_height() / 2, f"{val:.4g}", va="center", fontsize=8.5, color=INK)

    figure.tight_layout()
    return figure


def figure_to_png(figure: Figure, dpi: int = 150) -> bytes:
    """Render to PNG bytes and close the figure, so a long run cannot leak them."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, facecolor=figure.get_facecolor())
    _pyplot().close(figure)
    return buffer.getvalue()


def figure_to_data_uri(figure: Figure, dpi: int = 150) -> str:
    """
    A `data:` URI for embedding straight into a page.

    Self-contained on purpose: the Andon board and the report pages then have no
    second request to make and no static file to serve, which is one less thing
    to go wrong on a machine in a plant with no internet.
    """
    return "data:image/png;base64," + base64.b64encode(figure_to_png(figure, dpi=dpi)).decode("ascii")
