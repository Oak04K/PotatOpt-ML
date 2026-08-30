"""
Tests for chart_engine.

The point of these is not that a picture appeared. It is that the picture was
drawn from the numbers potatopt produced and did not invent its own: a chart
that recomputes a limit can disagree with the report printed beside it, and
nobody catches that by looking.

matplotlib lives in the `viz` extra, so the whole module skips without it rather
than failing a run that never asked for charts.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

import potatopt as po

matplotlib = pytest.importorskip("matplotlib", reason="charts need the viz extra")

# Imported after the skip above, because it is only importable once matplotlib exists.
import chart_engine as ce


@pytest.fixture(scope="module")
def wear_ramp():
    rng = np.random.default_rng(11)
    flat = 60 + rng.normal(0, 0.4, 60)
    ramp = 60 + np.linspace(0, 8, 40) + rng.normal(0, 0.4, 40)
    return list(flat) + list(ramp)


@pytest.fixture(scope="module")
def ewma(wear_ramp):
    return po.calculate_ewma_chart(wear_ramp, baseline_n=40)


@pytest.fixture(scope="module")
def cusum(wear_ramp):
    return po.calculate_cusum_chart(wear_ramp, baseline_n=40)


@pytest.fixture(scope="module")
def pareto():
    import pandas as pd
    rows = []
    for mode, events, hours in [("Bearing", 10, 40.0), ("Sensor", 25, 12.5), ("Belt", 5, 30.0)]:
        rows.extend({"failure_mode": mode, "downtime_hours": hours / events} for _ in range(events))
    return po.calculate_pareto(pd.DataFrame(rows), "failure_mode", value_col="downtime_hours")


# ---- The chart draws what it was given ----


def test_ewma_plots_the_limits_it_was_handed(ewma):
    figure = ce.plot_ewma(ewma)
    axes = figure.axes[0]
    drawn = {tuple(round(v, 6) for v in line.get_ydata()) for line in axes.lines}
    assert tuple(round(v, 6) for v in ewma["ucl"]) in drawn
    assert tuple(round(v, 6) for v in ewma["ewma"]) in drawn
    ce.figure_to_png(figure)


def test_every_violation_is_marked(ewma):
    assert ewma["violations"], "the fixture should degrade far enough to signal"
    figure = ce.plot_ewma(ewma)
    marked = sum(len(collection.get_offsets()) for collection in figure.axes[0].collections)
    assert marked == len(ewma["violations"])
    ce.figure_to_png(figure)


def test_cusum_draws_both_arms(cusum):
    figure = ce.plot_cusum(cusum)
    drawn = {tuple(round(v, 6) for v in line.get_ydata()) for line in figure.axes[0].lines}
    assert tuple(round(v, 6) for v in cusum["cusum_high"]) in drawn
    assert tuple(round(v, 6) for v in cusum["cusum_low"]) in drawn
    ce.figure_to_png(figure)


def test_pareto_bars_match_the_values(pareto):
    figure = ce.plot_pareto(pareto)
    heights = [round(patch.get_height(), 6) for patch in figure.axes[0].patches]
    assert heights == [round(row["value"], 6) for row in pareto["categories"]]
    ce.figure_to_png(figure)


def test_the_vital_few_are_readable_without_colour(pareto):
    # Filled versus hollow, not two hues: a Pareto's usual fate is a photocopy
    # on a noticeboard, and colour alone does not survive that.
    figure = ce.plot_pareto(pareto)
    fills = [patch.get_facecolor() for patch in figure.axes[0].patches]
    expected = [row["is_vital_few"] for row in pareto["categories"]]
    for fill, vital in zip(fills, expected, strict=True):
        assert (fill[:3] == matplotlib.colors.to_rgb(ce.INK)) is vital
    ce.figure_to_png(figure)


def test_pareto_says_what_it_counted(pareto):
    figure = ce.plot_pareto(pareto)
    # Counting events and counting hours rank a plant differently, so a chart
    # that does not name its unit invites action on the wrong cause.
    assert pareto["measured_by"] in figure.axes[0].get_title(loc="left")
    ce.figure_to_png(figure)


def test_oee_shows_the_factors_beside_the_result():
    result = po.calculate_oee(480.0, 420.0, 0.5, 700, 660)
    figure = ce.plot_oee(result)
    heights = [round(patch.get_height(), 4) for patch in figure.axes[0].patches]
    assert heights == [
        round(result["availability"] * 100, 4),
        round(result["performance"] * 100, 4),
        round(result["quality"] * 100, 4),
        round(result["oee"] * 100, 4),
    ]
    ce.figure_to_png(figure)


def test_availability_names_the_waiting_gap():
    result = po.calculate_availability(250.0, 1.625, 2.125)
    figure = ce.plot_availability(result)
    assert "waiting" in figure.axes[0].get_title(loc="left")
    ce.figure_to_png(figure)


def test_availability_refuses_to_draw_a_gap_it_was_not_given():
    result = po.calculate_availability(250.0, 1.625)
    with pytest.raises(ValueError, match="Operational"):
        ce.plot_availability(result)


# ---- Refusing to draw the wrong thing ----


def test_a_failed_result_is_refused_not_drawn():
    # potatopt returns {"error": ...} instead of raising. Drawing empty axes
    # from that would put a chart of nothing in front of someone, which is worse
    # than no chart: it looks like a measurement.
    failed = po.calculate_ewma_chart("not a series")
    assert "error" in failed
    with pytest.raises(ValueError, match="failed result"):
        ce.plot_ewma(failed)


@pytest.mark.parametrize("junk", [None, 42, "chart", []])
def test_a_non_dictionary_is_refused(junk):
    with pytest.raises(TypeError):
        ce.plot_ewma(junk)


def test_a_dictionary_missing_its_keys_names_them():
    with pytest.raises(ValueError, match="missing"):
        ce.plot_ewma({"ewma": [1, 2, 3]})


# ---- Output ----


def test_png_bytes_are_a_png(ewma):
    data = ce.figure_to_png(ce.plot_ewma(ewma))
    assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_data_uri_embeds_the_same_png(ewma):
    uri = ce.figure_to_data_uri(ce.plot_ewma(ewma))
    assert uri.startswith("data:image/png;base64,")
    assert base64.b64decode(uri.split(",", 1)[1])[:8] == b"\x89PNG\r\n\x1a\n"


def test_drawing_does_not_leak_figures(ewma):
    # Every render closes its figure. A dashboard that redraws on a timer would
    # otherwise accumulate them until matplotlib starts warning, then swapping.
    import matplotlib.pyplot as plt
    plt.close("all")
    for _ in range(12):
        ce.figure_to_png(ce.plot_ewma(ewma))
    assert plt.get_fignums() == []


# ---- Phase 3 charts: confusion matrix, correlation, and feature importance ----


def test_plot_confusion_matrix_renders_counts_and_proportions():
    eval_payload = {
        "confusion_matrix": [[50, 5], [10, 35]],
        "class_labels": ["Good", "Defect"],
    }
    figure = ce.plot_confusion_matrix(eval_payload)
    axes = figure.axes[0]
    assert axes.get_xlabel() == "Predicted label"
    assert axes.get_ylabel() == "Actual label"
    ticks_x = [t.get_text() for t in axes.get_xticklabels()]
    ticks_y = [t.get_text() for t in axes.get_yticklabels()]
    assert ticks_x == ["Good", "Defect"]
    assert ticks_y == ["Good", "Defect"]
    png_bytes = ce.figure_to_png(figure)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"

    # Normalized version
    fig_norm = ce.plot_confusion_matrix(eval_payload, normalize=True)
    ce.figure_to_png(fig_norm)

    # Division by zero safety when a row has 0 count
    zero_row_payload = {
        "confusion_matrix": [[0, 0], [10, 35]],
        "class_labels": ["Empty", "Active"],
    }
    fig_zero = ce.plot_confusion_matrix(zero_row_payload, normalize=True)
    ce.figure_to_png(fig_zero)


def test_plot_confusion_matrix_guards():
    with pytest.raises(ValueError, match="missing"):
        ce.plot_confusion_matrix({"confusion_matrix": [[1, 0], [0, 1]]})
    with pytest.raises(ValueError, match="missing"):
        ce.plot_confusion_matrix({"class_labels": ["A", "B"]})
    with pytest.raises(TypeError):
        ce.plot_confusion_matrix(["not", "a", "dict"])
    with pytest.raises(ValueError, match="failed result"):
        ce.plot_confusion_matrix({"error": "Failed evaluate", "confusion_matrix": [], "class_labels": []})
    with pytest.raises(ValueError, match="match"):
        ce.plot_confusion_matrix({"confusion_matrix": [[1, 0]], "class_labels": ["A", "B"]})


def test_plot_correlation_heatmap_renders_and_leaves_none_blank():
    corr_payload = {
        "columns": ["temp", "pressure", "vibration"],
        "matrix": [
            [1.0, 0.85, None],
            [0.85, 1.0, -0.4],
            [None, -0.4, 1.0],
        ],
        "note": "Truncated matrix note",
    }
    figure = ce.plot_correlation_heatmap(corr_payload)
    axes = figure.axes[0]
    # 7 finite cells out of 9, so exactly 7 cell annotations should exist
    cell_texts = [t.get_text() for t in axes.texts if t.get_text() != "Truncated matrix note"]
    assert len(cell_texts) == 7
    assert "None" not in cell_texts and "nan" not in cell_texts
    assert any("Truncated matrix note" in t.get_text() for t in axes.texts)
    png_bytes = ce.figure_to_png(figure)
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_correlation_heatmap_skips_text_when_more_than_15_columns():
    cols = [f"feat_{i}" for i in range(16)]
    matrix = [[1.0 if i == j else 0.05 for j in range(16)] for i in range(16)]
    figure = ce.plot_correlation_heatmap({"columns": cols, "matrix": matrix})
    axes = figure.axes[0]
    assert len(axes.texts) == 0
    ce.figure_to_png(figure)


def test_plot_correlation_heatmap_guards():
    with pytest.raises(ValueError, match="missing"):
        ce.plot_correlation_heatmap({"columns": ["a", "b"]})
    with pytest.raises(ValueError, match="missing"):
        ce.plot_correlation_heatmap({"matrix": [[1.0]]})
    with pytest.raises(TypeError):
        ce.plot_correlation_heatmap("not a dict")
    with pytest.raises(ValueError, match="failed result"):
        ce.plot_correlation_heatmap({"error": "Failed corr", "columns": [], "matrix": []})
    with pytest.raises(ValueError, match="match"):
        ce.plot_correlation_heatmap({"columns": ["a", "b"], "matrix": [[1.0]]})


def test_plot_feature_importance_accepts_all_three_shapes():
    import pandas as pd

    # Shape 1: explain_predictions dict
    shap_payload = {
        "available": True,
        "reason": None,
        "feature_attributions": [
            {"feature": "temp", "mean_abs_shap": 0.82},
            {"feature": "pressure", "mean_abs_shap": 0.54},
            {"feature": "speed", "mean_abs_shap": 0.21},
        ],
    }
    # Shape 2: list of dicts
    list_payload = [
        {"feature": "temp", "importance": 0.82},
        {"feature": "pressure", "importance": 0.54},
        {"feature": "speed", "importance": 0.21},
    ]
    # Shape 3: pandas DataFrame
    df_payload = pd.DataFrame({
        "feature": ["temp", "pressure", "speed"],
        "importance": [0.82, 0.54, 0.21],
    })

    fig1 = ce.plot_feature_importance(shap_payload)
    fig2 = ce.plot_feature_importance(list_payload)
    fig3 = ce.plot_feature_importance(df_payload)

    # All three produce the same number of bars
    assert len(fig1.axes[0].patches) == 3
    assert len(fig2.axes[0].patches) == 3
    assert len(fig3.axes[0].patches) == 3

    # X-axis label reflects the actual measurement plotted
    assert fig1.axes[0].get_xlabel() == "mean |SHAP|"
    assert fig2.axes[0].get_xlabel() == "model importance"
    assert fig3.axes[0].get_xlabel() == "model importance"

    # top_n limits the number of plotted bars
    fig_top2 = ce.plot_feature_importance(shap_payload, top_n=2)
    assert len(fig_top2.axes[0].patches) == 2

    ce.figure_to_png(fig1)
    ce.figure_to_png(fig2)
    ce.figure_to_png(fig3)
    ce.figure_to_png(fig_top2)


def test_plot_feature_importance_guards_and_unavailable():
    import pandas as pd

    # Unavailable SHAP attribution raises ValueError with the reported reason
    unavail = {
        "available": False,
        "reason": "IsolationForest anomaly fallback does not support SHAP.",
        "feature_attributions": [],
    }
    with pytest.raises(ValueError, match="IsolationForest"):
        ce.plot_feature_importance(unavail)

    # DataFrame missing required columns
    with pytest.raises(ValueError, match="DataFrame must contain"):
        ce.plot_feature_importance(pd.DataFrame({"x": [1, 2], "y": [3, 4]}))

    # Empty list
    with pytest.raises(ValueError, match="No feature importance data"):
        ce.plot_feature_importance([])

    # Failed result dict
    with pytest.raises(ValueError, match="failed result"):
        ce.plot_feature_importance({"error": "Calculation error"})

    # Invalid type
    with pytest.raises(TypeError):
        ce.plot_feature_importance(12345)
