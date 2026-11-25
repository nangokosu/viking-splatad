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
SECTION_CELL_COUNT = 9


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


def image(seed_offset=0.0, with_nan=True):
    """Small ramp image in [0, 1], optionally with a NaN cell."""
    ramp = np.linspace(0, 1, GRID_SIDE * GRID_SIDE).reshape(GRID_SIDE, GRID_SIDE) + seed_offset
    ramp = np.clip(ramp, 0, 1)
    if with_nan:
        ramp[0, 0] = np.nan
    return ramp


EXTENT = (0.0, 6.0, 0.0, 6.0)


def test_bev_figure_with_overlays(settings):
    """F1 draws image, boxes, path and contour; one data axes; PNG renders."""
    near = np.zeros((GRID_SIDE, GRID_SIDE), dtype=bool)
    near[2:4, 2:4] = True
    box = np.array([[1, 1], [2, 1], [2, 2], [1, 2]], dtype=float)
    figure = mpl_figs.occupancy_bev_figure(image(), EXTENT, settings, boxes_xy=[box],
                                           path_xy=np.array([[0, 0], [5, 5]], dtype=float),
                                           near_contour=near, title="bev")
    assert len(data_axes(figure)) == 1
    assert png_bytes(figure) > 0


def test_density_figure_log_scale_and_seed_legend(settings):
    """F3 uses a log norm and a seed legend entry with the count."""
    density = np.zeros((GRID_SIDE, GRID_SIDE))
    density[1, 1], density[2, 3] = 0.5, 4.0
    edges = np.arange(GRID_SIDE + 1, dtype=float)
    seeds = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    figure = mpl_figs.gaussian_density_figure(density, edges, edges, settings, seeds_xy=seeds, title="d")
    axis = data_axes(figure)[0]
    assert axis.images[0].norm.__class__.__name__ == "LogNorm"
    assert axis.images[0].norm.vmin == 0.5
    assert any("n=3" in text.get_text() for text in axis.get_legend().get_texts())
    assert png_bytes(figure) > 0


def test_arm_heat_difference_symmetric(settings):
    """C5 has one panel per arm plus a difference panel with norm vmin == -vmax; footer present."""
    images = {arm: image(0.1 * index) for index, arm in enumerate(settings.arm_order)}
    difference = np.linspace(-0.3, 0.9, GRID_SIDE * GRID_SIDE).reshape(GRID_SIDE, GRID_SIDE)
    figure = mpl_figs.arm_heat_figure(images, EXTENT, settings, difference=(difference, "init+loss - stock"),
                                      title="heat")
    axes = data_axes(figure)
    assert len(axes) == len(settings.arm_order) + 1
    norm = axes[-1].images[0].norm
    assert norm.vmin == -norm.vmax and norm.vmax == pytest.approx(0.9)
    assert footer_present(figure, settings)
    assert png_bytes(figure) > 0


def test_arm_heat_without_difference(settings):
    """Without a difference the figure has only arm panels."""
    images = {arm: image() for arm in settings.arm_order}
    figure = mpl_figs.arm_heat_figure(images, EXTENT, settings, title="heat")
    assert len(data_axes(figure)) == len(settings.arm_order)


def curve_dict():
    """Synthetic threshold_curves-style dict."""
    thresholds = np.linspace(0, 1, 5)
    return {"thresholds": thresholds, "pooled_recall": thresholds[::-1], "pooled_precision": thresholds,
            "mean_recall": np.array([1, 0.8, np.nan, 0.2, 0.0]), "mean_precision": thresholds * 0.5}


def test_threshold_curves_grid_bands_and_guides(settings):
    """C4: rows = bands, 2 columns, PolyCollections for campaign bands, guide lines per threshold."""
    curves = {band: {arm: curve_dict() for arm in settings.arm_order} for band in ("H_near", "whole scene")}
    thresholds = curve_dict()["thresholds"]
    campaign = {"H_near": {arm: {metric: (thresholds, thresholds * 0.5, thresholds)
                                 for metric in ("recall", "precision")} for arm in settings.arm_order}}
    p2_thresholds = (0.5, 0.8)
    figure = mpl_figs.threshold_curves_figure(curves, settings, p2_thresholds=p2_thresholds,
                                              campaign_bands=campaign, title="curves")
    axes = data_axes(figure)
    assert len(axes) == 4
    first_axis = axes[0]  # H_near recall
    assert sum(isinstance(item, PolyCollection) for item in first_axis.collections) == len(settings.arm_order)
    assert not any(isinstance(item, PolyCollection) for item in axes[2].collections)  # whole scene: no band
    # curves (2 per arm) + guide lines (one per P2 threshold)
    assert len(first_axis.lines) == 2 * len(settings.arm_order) + len(p2_thresholds)
    assert footer_present(figure, settings)
    assert png_bytes(figure) > 0


def test_histogram_panels_and_counts(settings):
    """One panel per arm; legend carries the sample counts."""
    rng = np.random.default_rng(0)
    samples = {arm: {"truth": rng.random(40), "off_truth": rng.random(25)} for arm in settings.arm_order}
    figure = mpl_figs.occupancy_histogram_figure(samples, settings, bins=10, title="hist")
    axes = data_axes(figure)
    assert len(axes) == len(settings.arm_order)
    legend_text = [text.get_text() for text in axes[0].get_legend().get_texts()]
    assert "on truth (n=40)" in legend_text and "off truth (n=25)" in legend_text
    assert footer_present(figure, settings)
    assert png_bytes(figure) > 0


def test_histogram_empty_series_does_not_crash(settings):
    """Empty samples are skipped rather than raising."""
    samples = {arm: {"truth": np.array([]), "off_truth": np.array([0.2])} for arm in settings.arm_order}
    figure = mpl_figs.occupancy_histogram_figure(samples, settings, bins=4, title="hist")
    assert png_bytes(figure) > 0


def make_section(settings, label):
    """Synthetic 3x3-cell section for every arm."""
    along, height = np.meshgrid(np.arange(3.0), np.arange(3.0))
    coords = np.column_stack([along.ravel(), height.ravel()])
    by_arm = {arm: {"occupancy": np.linspace(0, 1, SECTION_CELL_COUNT),
                    "gaussian_coords": np.array([[1.0, 1.0], [2.0, 0.5]]),
                    "gaussian_opacity": np.array([0.9, 0.2]),
                    "seed_coords": np.array([[0.5, 0.5]]) if arm == "radar_init" else None}
              for arm in settings.arm_order}
    return {"label": label, "cells_coords": coords,
            "evidence_labels": np.arange(SECTION_CELL_COUNT) % len(mpl_figs.EVIDENCE_NAMES),
            "box_polygon": np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float), "by_arm": by_arm}


def test_occlusion_section_grid_and_layers(settings):
    """C3: rows x arms axes, evidence + occupancy PolyCollections per panel, footer, PNG."""
    sections = [make_section(settings, "trunk 1"), make_section(settings, "trunk 2")]
    figure = mpl_figs.occlusion_section_figure(sections, settings, voxel_size=1.0, title="sections")
    axes = data_axes(figure)
    assert len(axes) == 2 * len(settings.arm_order)
    assert sum(isinstance(item, PolyCollection) for item in axes[0].collections) == 2
    assert footer_present(figure, settings)
    assert figure.legends and png_bytes(figure) > 0


def test_run_note_is_wrapped_to_figure_width_and_under_footer(settings):
    """The run note wraps at width_in x NOTE_CHARACTERS_PER_INCH characters and sits below the footer."""
    long_note = "drawn runs: " + "; ".join(f"arm{index} = run-identifier-{index}" for index in range(12))
    figure = mpl_figs.arm_heat_figure({"stock": image()}, EXTENT, settings, title="t", run_note=long_note)
    note_texts = [text for text in figure.texts if text.get_text().startswith("drawn runs")]
    footer_texts = [text for text in figure.texts if text.get_text() == settings.footer_note]
    assert len(note_texts) == 1 and len(footer_texts) == 1
    wrap_width = int(figure.get_figwidth() * mpl_figs.NOTE_CHARACTERS_PER_INCH)
    lines = note_texts[0].get_text().split("\n")
    assert len(lines) > 1 and all(len(line) <= wrap_width for line in lines)
    assert note_texts[0].get_position()[1] < footer_texts[0].get_position()[1]
    assert " ".join(lines) == long_note


def test_density_figure_with_seeds_has_second_log_panel_of_unit_weight_counts(settings):
    """Seeds give a second data panel (LogNorm) whose total equals the seed count inside the extent."""
    density = np.full((GRID_SIDE, GRID_SIDE), 0.5)
    edges = np.arange(GRID_SIDE + 1, dtype=float)
    seeds = np.array([[1.5, 1.5], [1.6, 1.4], [4.5, 4.5], [99.0, 99.0]])  # last one is outside the extent
    figure = mpl_figs.gaussian_density_figure(density, edges, edges, settings, seeds_xy=seeds, title="d")
    axes = data_axes(figure)
    assert len(axes) == 2 and not axes[1].collections
    seed_image = axes[1].images[0]
    assert seed_image.norm.__class__.__name__ == "LogNorm"
    inside_extent_count = int(((seeds >= edges[0]) & (seeds <= edges[-1])).all(axis=1).sum())
    assert np.ma.sum(seed_image.get_array()) == inside_extent_count
    no_seed_figure = mpl_figs.gaussian_density_figure(density, edges, edges, settings, title="d")
    assert len(data_axes(no_seed_figure)) == 1


def test_bev_limits_stay_on_extent_despite_far_overlays(settings):
    """A far-away path point does not widen the axis limits beyond the extent."""
    far_path = np.array([[0.0, 0.0], [500.0, 500.0]])
    figure = mpl_figs.occupancy_bev_figure(image(), EXTENT, settings, path_xy=far_path, title="bev")
    axis = data_axes(figure)[0]
    assert axis.get_xlim() == (EXTENT[0], EXTENT[1]) and axis.get_ylim() == (EXTENT[2], EXTENT[3])


def test_section_evidence_colours_come_from_settings_and_differ_from_seed_colour(settings):
    """Each evidence class has its own settings colour, none equal to the radar-seed colour."""
    evidence_rgbs = [tuple(settings.colours[f"evidence_{name}"]) for name in mpl_figs.EVIDENCE_NAMES]
    assert len(set(evidence_rgbs)) == len(mpl_figs.EVIDENCE_NAMES)
    assert tuple(settings.colours["radar_seed"]) not in evidence_rgbs
    section = make_section(settings, "trunk")
    section["evidence_labels"] = np.zeros(SECTION_CELL_COUNT, dtype=int)  # all "seen"
    figure = mpl_figs.occlusion_section_figure([section], settings, voxel_size=1.0, title="s")
    first_collection = data_axes(figure)[0].collections[0]
    expected_rgb = np.array(evidence_rgbs[0]) / mpl_figs.MAX_CHANNEL_VALUE
    np.testing.assert_allclose(first_collection.get_facecolor()[0][:3], expected_rgb)


def test_section_labels_legend_and_markers(settings):
    """x label names the sensor distance; legend lists evidence, occupancy, Gaussian and seed; seeds are x on top."""
    figure = mpl_figs.occlusion_section_figure([make_section(settings, "trunk")], settings, voxel_size=1.0,
                                               title="s")
    axes = data_axes(figure)
    assert "distance from sensor" in axes[0].get_xlabel().replace("\n", " ")
    legend_labels = [text.get_text() for text in figure.legends[0].get_texts()]
    for expected in (*mpl_figs.EVIDENCE_NAMES, "occupancy (alpha)", "radar seed"):
        assert expected in legend_labels
    assert any(label.startswith("Gaussian mean") for label in legend_labels)
    seed_axis = axes[list(settings.arm_order).index("radar_init")]
    gaussian_collection, seed_collection = seed_axis.collections[-2], seed_axis.collections[-1]
    assert seed_collection.get_zorder() > gaussian_collection.get_zorder()
    np.testing.assert_allclose(gaussian_collection.get_facecolor()[0][:3], (0, 0, 0))


def test_section_panel_size_follows_data_aspect(settings):
    """A window twice as tall as wide gives a figure at least that tall per row relative to width per column."""
    section = make_section(settings, "trunk")
    section["window"] = (np.array([0.0, 0.0]), np.array([4.0, 20.0]))
    figure = mpl_figs.occlusion_section_figure([section], settings, voxel_size=1.0, title="s")
    column_count = len(settings.arm_order)
    panel_width = figure.get_figwidth() / column_count
    assert (figure.get_figheight() - mpl_figs.TITLE_BAND_IN) > 2 * panel_width


def test_density_figure_seed_panel_uses_the_gaussian_panels_edges_even_when_uneven(settings):
    """Regression: the seed histogram uses the Gaussian panel's own edges, so both panels have the same grid shape."""
    edges_x = np.array([0.0, 1.0, 2.5])  # uneven: rebuilding bins from the first width would give 3 columns
    edges_y = np.array([0.0, 1.0, 2.0, 3.0])
    density = np.full((len(edges_y) - 1, len(edges_x) - 1), 0.5)
    seeds = np.array([[0.5, 0.5], [2.0, 2.5]])
    figure = mpl_figs.gaussian_density_figure(density, edges_x, edges_y, settings, seeds_xy=seeds, title="d")
    axes = data_axes(figure)
    seed_array = axes[1].images[0].get_array()
    assert seed_array.shape == density.shape
    assert np.ma.sum(seed_array) == len(seeds)


def test_occlusion_section_figure_rejects_no_sections(settings):
    """Regression: an empty section list is a clear ValueError, not max() of an empty sequence."""
    with pytest.raises(ValueError, match="at least one section"):
        mpl_figs.occlusion_section_figure([], settings, voxel_size=0.5, title="t")


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
