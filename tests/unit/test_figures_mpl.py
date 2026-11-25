"""Tests of the matplotlib figure renderers (tiny synthetic inputs, known-answer structure)."""

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
import yaml
from matplotlib.collections import PolyCollection

from viking_splatad.viz import figures_mpl as mpl_figs
from viking_splatad.viz.figure_runs import load_figure_settings

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "eval.yaml"
GRID_SIDE = 6


@pytest.fixture(scope="module")
def settings():
    """FigureSettings loaded from the real configs/eval.yaml."""
    return load_figure_settings(yaml.safe_load(CONFIG_PATH.read_text())["eval"])


@pytest.fixture(autouse=True)
def close_figures():
    """Close every figure after each test."""
    yield
    plt.close("all")


def data_axes(figure):
    """Return the axes tagged as data panels (colourbars and legends excluded)."""
    return [axis for axis in figure.axes if axis.get_gid() == mpl_figs.DATA_AXES_GID]


def footer_present(figure, settings):
    """True when the footer note is a text of the figure."""
    return any(text.get_text() == settings.footer_note for text in figure.texts)


def png_bytes(figure):
    """Render to PNG and return the byte count."""
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return len(buffer.getvalue())


def curve_dict():
    """Synthetic threshold_curves-style dict."""
    thresholds = np.linspace(0, 1, 5)
    return {"thresholds": thresholds, "pooled_recall": thresholds[::-1], "pooled_precision": thresholds,
            "mean_recall": np.array([1, 0.8, np.nan, 0.2, 0.0]), "mean_precision": thresholds * 0.5}


def test_threshold_curves_one_row_bands_guides_and_xlim(settings):
    """C4: one row of 2 axes (recall, precision), x limited to the grid ends, bands, guides and a caption."""
    curves = {arm: curve_dict() for arm in settings.arm_order}
    thresholds = np.linspace(settings.threshold_curve_min, settings.threshold_curve_max, 5)
    for arm_curves in curves.values():
        arm_curves["thresholds"] = thresholds
    campaign = {arm: {metric: (thresholds, thresholds * 0.5, thresholds)
                      for metric in ("recall", "precision")} for arm in settings.arm_order}
    p2_thresholds = (0.5, 0.8)
    figure = mpl_figs.threshold_curves_figure(curves, settings, p2_thresholds=p2_thresholds,
                                              campaign_bands=campaign, title="curves")
    axes = data_axes(figure)
    assert len(axes) == 2
    for axis in axes:
        assert axis.get_xlim() == (settings.threshold_curve_min, settings.threshold_curve_max)
        assert sum(isinstance(item, PolyCollection) for item in axis.collections) == len(settings.arm_order)
        # curves (2 per arm) + guide lines (one per P2 threshold)
        assert len(axis.lines) == 2 * len(settings.arm_order) + len(p2_thresholds)
    assert any(text.get_text() == mpl_figs.CURVES_CAPTION for text in figure.texts)
    assert footer_present(figure, settings)
    assert png_bytes(figure) > 0


def test_run_note_is_wrapped_to_figure_width_and_under_footer(settings):
    """The run note wraps at width_in x NOTE_CHARACTERS_PER_INCH characters and sits below the footer."""
    long_note = "drawn runs: " + "; ".join(f"arm{index} = run-identifier-{index}" for index in range(12))
    curves = {"stock": curve_dict()}
    figure = mpl_figs.threshold_curves_figure(curves, settings, p2_thresholds=(0.5,), title="t", run_note=long_note)
    note_texts = [text for text in figure.texts if text.get_text().startswith("drawn runs")]
    footer_texts = [text for text in figure.texts if text.get_text() == settings.footer_note]
    assert len(note_texts) == 1 and len(footer_texts) == 1
    wrap_width = int(figure.get_figwidth() * mpl_figs.NOTE_CHARACTERS_PER_INCH)
    lines = note_texts[0].get_text().split("\n")
    assert len(lines) > 1 and all(len(line) <= wrap_width for line in lines)
    assert note_texts[0].get_position()[1] < footer_texts[0].get_position()[1]
    assert " ".join(lines) == long_note


def test_band_layout_stacks_padding_note_footer_and_key():
    """total = padding + wrapped note lines + footer line + key band, computed from the named heights."""
    width_in = 4.0
    long_note = "drawn runs: " + "word " * 40
    band = mpl_figs.band_layout(width_in, long_note, "footer", key_band_in=0.7)
    expected_lines = len(mpl_figs.wrap_note(long_note, width_in))
    assert len(band["note_lines"]) == expected_lines and expected_lines > 1
    expected_total = (mpl_figs.BAND_PADDING_IN + expected_lines * mpl_figs.NOTE_LINE_HEIGHT_IN
                      + mpl_figs.FOOTER_LINE_HEIGHT_IN + 0.7)
    assert band["total_in"] == pytest.approx(expected_total)
    bare = mpl_figs.band_layout(width_in, None, None, key_band_in=0.0)
    assert bare["total_in"] == pytest.approx(mpl_figs.BAND_PADDING_IN) and bare["note_lines"] == []


def test_legend_columns_follow_the_drawn_arms_not_the_settings(settings):
    """With 2 of the 3 arms drawn the legend has 2 columns and each arm's pooled and mean entries share a column."""
    drawn_arms = list(settings.arm_order)[:2]
    assert len(drawn_arms) < len(settings.arm_order)
    figure = mpl_figs.threshold_curves_figure({arm: curve_dict() for arm in drawn_arms}, settings,
                                              p2_thresholds=(0.5,), title="two arms")
    legend = figure.legends[0]
    assert legend._ncols == len(drawn_arms)
    labels = [text.get_text() for text in legend.get_texts()]
    assert len(labels) == 2 * len(drawn_arms)
    # Matplotlib fills legend columns top to bottom: column of entry k = k // rows, rows = ceil(entries / columns).
    rows = -(-len(labels) // legend._ncols)
    column_of_label = {label: entry_index // rows for entry_index, label in enumerate(labels)}
    for arm in drawn_arms:
        arm_label = settings.arm_labels.get(arm, arm)
        assert (column_of_label[f"{arm_label} (pooled)"] == column_of_label[f"{arm_label} (mean over objects)"])
