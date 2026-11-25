"""Matplotlib renderers of the Gaussian-occupancy figure suite.

Every renderer takes already-prepared numpy data and a ``FigureSettings`` and returns a
``matplotlib.figure.Figure``. None loads run data or saves a file; the caller saves.
Colours named in the settings are RGB 0-255 and are divided by 255 here. Per-arm line colours
are a presentation choice, taken from the qualitative ``tab10`` map.
"""

import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend, selected before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from viking_splatad.viz.figure_runs import FigureSettings  # noqa: E402
from viking_splatad.viz.pyvista_scene import MAX_CHANNEL_VALUE  # noqa: E402

# Tag put on every data axes (colourbar and legend-only axes lack it) so callers and tests can
# tell them apart without relying on matplotlib internals.
DATA_AXES_GID = "figure_data_panel"
# Qualitative colormap supplying arm line colours (presentation choice:
# ten distinct, colour-blind-tolerant hues, indexed by arm position).
QUALITATIVE_CMAP_NAME = "tab10"
# Figure inches of one panel (width, height); a layout choice.
PANEL_WIDTH_IN = 4.5
PANEL_HEIGHT_IN = 4.0
# Extra figure height (in) for the title and the footer band.
TITLE_BAND_IN = 0.6
# Horizontal centre of the footer, run note and legend (figure fraction) and their font size (pt).
FOOTER_X_FRACTION = 0.5
FOOTER_FONT_SIZE = 8
# Line width (pt) of the threshold curves.
CURVE_LINE_WIDTH = 1.6
# Bottom band (inches), stacked bottom to top: padding, run-identification note lines, one footer line,
# then the legend rows. Line heights are 8 pt text with leading (about 0.17 in); a legend row at 7 pt
# needs about 0.22 in.
BAND_PADDING_IN = 0.08
NOTE_LINE_HEIGHT_IN = 0.16
FOOTER_LINE_HEIGHT_IN = 0.2
LEGEND_ROW_HEIGHT_IN = 0.24
# Characters of 8 pt text that fit in one inch (average glyph about 0.5 em = 4 pt wide, so 18 per inch;
# 15 leaves a safety margin for wide glyphs); the run note wraps at figure width x this constant.
NOTE_CHARACTERS_PER_INCH = 15
# Grey edge (matplotlib grey string) of legend patches, so a white patch stays visible.
LEGEND_PATCH_EDGE = "0.6"
# Legend rows of the threshold figure: two entries per arm (pooled and mean line) laid out with one
# column per arm, hence two rows.
THRESHOLD_LEGEND_ROWS = 2
# Alpha of the campaign spread bands: the pooled line's band is stronger than the mean-over-objects line's, so
# the two bands of one arm stay distinguishable where they overlap.
POOLED_BAND_ALPHA = 0.25
MEAN_BAND_ALPHA = 0.1
# Hatch of the mean band (ties it to the dashed line) and the width (pt) of its hatch / edge lines.
MEAN_BAND_HATCH = "///"
MEAN_BAND_EDGE_WIDTH = 0.0
# Dash pattern of the mean-over-objects curves and dotted style of threshold guides.
MEAN_LINESTYLE = "--"
GUIDE_LINESTYLE = ":"
GUIDE_LINE_WIDTH = 0.8
GUIDE_COLOUR = "0.4"
# Metrics shown as columns of the threshold figure: (column key, axis label, pooled key, mean key).
THRESHOLD_METRICS = (
    ("recall", "recall", "pooled_recall", "mean_recall"),
    ("precision", "precision", "pooled_precision", "mean_precision"),
)
# Unit interval bounds of occupancy, recall and precision axes.
UNIT_MIN = 0.0
UNIT_MAX = 1.0
# Legend font size (pt) and caption font size (pt).
LEGEND_FONT_SIZE = 7
CAPTION_FONT_SIZE = 8
# Height (in) of one caption line (8 pt text with leading) and the figure-width multiple of the
# threshold figure's panels (wider panels keep the caption on one line).
CAPTION_LINE_HEIGHT_IN = 0.16
CURVE_PANEL_WIDTH_IN = 5.5
# Captions printed under the axes.
CURVES_CAPTION = ("recall = share of obstacle cells filled; precision = share of filled cells on/next to "
                  "obstacles that are obstacle; each line's band = its min\u2013max over the arm's repeats "
                  "(stronger shading = pooled, lighter = mean over objects)")


def _unit_rgb(settings: FigureSettings, colour_name: str) -> Tuple[float, float, float]:
    """Return a named settings colour scaled from 0-255 to 0-1.

    Args:
        settings: Figure settings.
        colour_name: Key of ``settings.colours``.

    Returns:
        (R, G, B) floats in [0, 1].
    """
    # Step 1: divide each channel by the channel maximum.
    red, green, blue = settings.colours[colour_name]
    return (red / MAX_CHANNEL_VALUE, green / MAX_CHANNEL_VALUE, blue / MAX_CHANNEL_VALUE)


def _qualitative_colour(position: int) -> Tuple[float, float, float, float]:
    """Return the tab10 colour at ``position`` (wrapping around its length).

    Args:
        position: Index of the arm or label.

    Returns:
        RGBA tuple.
    """
    # Step 1: look the map up by name and wrap the index.
    colormap = plt.get_cmap(QUALITATIVE_CMAP_NAME)
    return colormap(position % colormap.N)


def _arm_colour(settings: FigureSettings, arm: str) -> Tuple[float, float, float, float]:
    """Return the line colour of an arm (tab10 indexed by its position in arm_order).

    Args:
        settings: Figure settings.
        arm: Arm key.

    Returns:
        RGBA tuple.
    """
    return _qualitative_colour(settings.arm_order.index(arm))


def wrap_note(run_note: Optional[str], figure_width_in: float) -> List[str]:
    """Wrap the run-identification note to the figure width.

    Args:
        run_note: Text naming the drawn runs, or None.
        figure_width_in: Figure width (inches).

    Returns:
        Wrapped lines (empty for no note); width = inches x NOTE_CHARACTERS_PER_INCH characters.
    """
    if not run_note:
        return []
    # Run ids contain hyphens; never split an id there.
    return textwrap.wrap(run_note, width=max(1, int(figure_width_in * NOTE_CHARACTERS_PER_INCH)),
                         break_on_hyphens=False)


def band_layout(figure_width_in: float, run_note: Optional[str], footer_note: Optional[str], key_band_in: float
                ) -> Dict[str, Any]:
    """Stack the bottom band of a figure: padding, run note, footer, then a key band (bottom to top).

    Shared by every composed figure (matplotlib panels here, PyVista screenshots in ``figures_pyvista``).

    Args:
        figure_width_in: Figure width (inches), which sets the note's wrap width.
        run_note: Text naming the drawn runs, or None.
        footer_note: Footer text, or None for no footer line.
        key_band_in: Height (inches) reserved above the footer for a legend / colourbar.

    Returns:
        {"note_lines", "note_in", "footer_in", "key_in", "total_in"}; total_in is the sum of the padding, the
        note lines, the footer line and the key band.
    """
    # Step 1: wrapped note and the heights of each stacked part.
    note_lines = wrap_note(run_note, figure_width_in)
    note_in = len(note_lines) * NOTE_LINE_HEIGHT_IN
    footer_in = FOOTER_LINE_HEIGHT_IN if footer_note else 0.0
    # Step 2: total band height = padding + note + footer + key.
    return {"note_lines": note_lines, "note_in": note_in, "footer_in": footer_in, "key_in": key_band_in,
            "total_in": BAND_PADDING_IN + note_in + footer_in + key_band_in}


def place_band_texts(figure: Any, band: Mapping[str, Any], height_in: float, footer_note: Optional[str]) -> float:
    """Print the run note (lowest) and the footer (above it) into the bottom band.

    Args:
        figure: Matplotlib figure of height ``height_in`` inches.
        band: Result of ``band_layout``.
        height_in: Figure height (inches); inches / height_in gives figure fractions.
        footer_note: Footer text, or None.

    Returns:
        Figure-fraction height where the key band (legend / colourbar) starts.
    """
    # Step 1: the run note sits on the band's padding.
    note_bottom = BAND_PADDING_IN / height_in
    if band["note_lines"]:
        figure.text(FOOTER_X_FRACTION, note_bottom, "\n".join(band["note_lines"]), ha="center", va="bottom",
                    fontsize=FOOTER_FONT_SIZE, color=GUIDE_COLOUR)
    # Step 2: the footer sits above the note; the key band starts above the footer.
    footer_bottom = note_bottom + band["note_in"] / height_in
    if footer_note:
        figure.text(FOOTER_X_FRACTION, footer_bottom, footer_note, ha="center", va="bottom",
                    fontsize=FOOTER_FONT_SIZE)
    return footer_bottom + band["footer_in"] / height_in


def _new_figure(column_count: int, row_count: int, title: str, settings: FigureSettings,
                *, footer: bool, sharex: bool = False, sharey: bool = False,
                run_note: Optional[str] = None, legend_rows: int = 0,
                panel_size: Tuple[float, float] = (PANEL_WIDTH_IN, PANEL_HEIGHT_IN),
                caption: Optional[str] = None) -> Tuple[Figure, np.ndarray]:
    """Create a titled constrained-layout figure with a tagged axes grid and a bottom band.

    The bottom band stacks (bottom to top) the wrapped run note, the footer note and ``legend_rows`` legend
    rows; the legend's anchor height (figure fraction) is stored as ``figure.legend_anchor_y``.

    Args:
        column_count: Panels per row.
        row_count: Panel rows.
        title: Figure title (kept short by the caller).
        settings: Figure settings (dpi, footer text).
        footer: Whether to print ``settings.footer_note`` under the figure.
        sharex: Share x limits across panels.
        sharey: Share y limits across panels.
        run_note: Optional text naming the drawn runs, printed wrapped under the footer.
        legend_rows: Rows of legend the caller will place in the band.
        panel_size: (width, height) inches of one panel.
        caption: Optional text printed (wrapped to the figure width) above the legend rows, just under the axes.

    Returns:
        (figure, 2D axes array [row, column]).
    """
    # Step 1: size = panel grid + title band + bottom band (note, footer, legend rows, caption lines).
    width_in = panel_size[0] * column_count
    caption_lines = wrap_note(caption, width_in)
    legend_in = legend_rows * LEGEND_ROW_HEIGHT_IN
    band = band_layout(width_in, run_note, settings.footer_note if footer else None,
                       legend_in + len(caption_lines) * CAPTION_LINE_HEIGHT_IN)
    band_in = band["total_in"]
    height_in = panel_size[1] * row_count + TITLE_BAND_IN + band_in
    fig = plt.figure(figsize=(width_in, height_in), dpi=settings.dpi, layout="constrained")
    axes_grid = fig.subplots(row_count, column_count, squeeze=False, sharex=sharex, sharey=sharey)
    # Step 2: tag data axes, title, and reserve the bottom band.
    for panel_axis in axes_grid.ravel():
        panel_axis.set_gid(DATA_AXES_GID)
    fig.suptitle(title)
    band_fraction = band_in / height_in
    fig.get_layout_engine().set(rect=(UNIT_MIN, band_fraction, UNIT_MAX, UNIT_MAX - band_fraction))
    # Step 3: texts bottom-up: run note lowest, footer above it, legend anchor above the footer.
    fig.legend_anchor_y = place_band_texts(fig, band, height_in, settings.footer_note if footer else None)
    # Step 4: the caption sits above the legend rows (its bottom edge is the legend's top edge).
    if caption_lines:
        fig.text(FOOTER_X_FRACTION, fig.legend_anchor_y + legend_in / height_in, "\n".join(caption_lines),
                 ha="center", va="bottom", fontsize=CAPTION_FONT_SIZE)
    return fig, axes_grid


def _plot_threshold_panel(axis: Any, arm_curves: Mapping[str, Mapping[str, np.ndarray]],
                          settings: FigureSettings, metric: Tuple[str, str, str, str],
                          campaign_for_band: Optional[Mapping[str, Any]]) -> None:
    """Draw one band x metric panel: spread bands, pooled (solid) and mean (dashed) lines.

    Args:
        axis: Target axes.
        arm_curves: arm -> curve dict from ``threshold_curves``.
        settings: Figure settings.
        metric: One entry of THRESHOLD_METRICS.
        campaign_for_band: Optional arm -> curve key (pooled / mean key) -> (thresholds, lower, upper).

    Returns:
        None.
    """
    metric_key, metric_label, pooled_key, mean_key = metric
    for arm in settings.arm_order:
        if arm not in arm_curves:
            continue
        colour = _arm_colour(settings, arm)
        curves = arm_curves[arm]
        # Step 1: one spread band per line, drawn first so lines sit on top: strong for pooled, light and
        # hatched for the mean (the hatch colour follows the edge colour).
        arm_bands = campaign_for_band.get(arm, {}) if campaign_for_band else {}
        for curve_key, band_alpha, band_hatch in ((pooled_key, POOLED_BAND_ALPHA, None),
                                                  (mean_key, MEAN_BAND_ALPHA, MEAN_BAND_HATCH)):
            if curve_key not in arm_bands:
                continue
            band_thresholds, lower, upper = arm_bands[curve_key]
            axis.fill_between(band_thresholds, lower, upper, facecolor=colour, edgecolor=colour,
                              alpha=band_alpha, linewidth=MEAN_BAND_EDGE_WIDTH, hatch=band_hatch)
        # Step 2: pooled solid, mean-over-objects dashed.
        label = settings.arm_labels.get(arm, arm)
        axis.plot(curves["thresholds"], curves[pooled_key], color=colour,
                  linewidth=CURVE_LINE_WIDTH, label=f"{label} (pooled)")
        axis.plot(curves["thresholds"], curves[mean_key], color=colour, linestyle=MEAN_LINESTYLE,
                  linewidth=CURVE_LINE_WIDTH, label=f"{label} (mean over objects)")
    axis.set_xlabel("occupancy threshold")
    axis.set_ylabel(metric_label)
    axis.set_ylim(UNIT_MIN, UNIT_MAX)


def threshold_curves_figure(curves_by_arm: Mapping[str, Mapping[str, np.ndarray]],
                            settings: FigureSettings, *, p2_thresholds: Sequence[float],
                            campaign_bands: Optional[Mapping[str, Any]] = None,
                            title: str, run_note: Optional[str] = None) -> Figure:
    """Recall (left) and precision (right) against occupancy threshold, one row, one band.

    Args:
        curves_by_arm: arm -> ``threshold_curves`` dict of the drawn run, on one band.
        settings: Figure settings.
        p2_thresholds: Thresholds marked by vertical dotted lines.
        campaign_bands: Optional arm -> curve key (e.g. ``pooled_recall``) -> (thresholds, lower, upper) min/max
            spread of that curve over the repeats.
        title: Figure title.
        run_note: Optional text naming the drawn runs.

    Returns:
        The figure (not saved); both x axes are limited to [min, max] of the threshold grid.
    """
    fig, axes_grid = _new_figure(len(THRESHOLD_METRICS), 1, title, settings, footer=True, run_note=run_note,
                                 legend_rows=THRESHOLD_LEGEND_ROWS, caption=CURVES_CAPTION,
                                 panel_size=(CURVE_PANEL_WIDTH_IN, PANEL_HEIGHT_IN))
    # Step 1: x range = the grid's own ends (every arm shares one grid).
    grid = np.concatenate([np.asarray(curves["thresholds"], dtype=float) for curves in curves_by_arm.values()])
    for column_index, metric in enumerate(THRESHOLD_METRICS):
        panel_axis = axes_grid[0, column_index]
        # Step 2: spread bands and curves of every arm, then guides at the P2 thresholds.
        _plot_threshold_panel(panel_axis, curves_by_arm, settings, metric, campaign_bands)
        for threshold in p2_thresholds:
            panel_axis.axvline(threshold, color=GUIDE_COLOUR, linestyle=GUIDE_LINESTYLE,
                               linewidth=GUIDE_LINE_WIDTH)
        panel_axis.set_xlim(float(grid.min()), float(grid.max()))
    # Step 3: one shared legend from the first panel; one column per arm puts pooled and mean in two rows.
    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(FOOTER_X_FRACTION, fig.legend_anchor_y),
               ncol=max(1, len(curves_by_arm)), fontsize=LEGEND_FONT_SIZE)
    return fig
