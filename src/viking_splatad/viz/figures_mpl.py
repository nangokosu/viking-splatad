"""Matplotlib renderers of the Gaussian-occupancy figure suite.

Every renderer takes already-prepared numpy data and a ``FigureSettings`` and returns a
``matplotlib.figure.Figure``. None loads run data or saves a file; the caller saves.
Colours named in the settings are RGB 0-255 and are divided by 255 here. Per-arm line colours
and evidence-label colours are a presentation choice, taken from the qualitative ``tab10`` map.
"""

import textwrap
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend, selected before pyplot is imported

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import PolyCollection  # noqa: E402
from matplotlib.colors import LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from viking_splatad.eval.hidden_geometry import EVIDENCE_NAMES  # noqa: E402
from viking_splatad.viz.figure_runs import RGB_CHANNEL_COUNT, FigureSettings  # noqa: E402
from viking_splatad.viz.pyvista_scene import MAX_CHANNEL_VALUE  # noqa: E402

# Tag put on every data axes (colourbar and legend-only axes lack it) so callers and tests can
# tell them apart without relying on matplotlib internals.
DATA_AXES_GID = "figure_data_panel"
# Qualitative colormap supplying arm line colours and evidence colours (presentation choice:
# ten distinct, colour-blind-tolerant hues, indexed by arm or label position).
QUALITATIVE_CMAP_NAME = "tab10"
# Figure inches of one panel (width, height); a layout choice.
PANEL_WIDTH_IN = 4.5
PANEL_HEIGHT_IN = 4.0
# Extra figure height (in) for the title and the footer band.
TITLE_BAND_IN = 0.6
# Horizontal centre of the footer, run note and legend (figure fraction) and their font size (pt).
FOOTER_X_FRACTION = 0.5
FOOTER_FONT_SIZE = 8
# Contour level separating the False and True of a boolean image.
BOOL_CONTOUR_LEVEL = 0.5
# Line widths (pt) of outlines, paths and curves.
OUTLINE_LINE_WIDTH = 0.8
PATH_LINE_WIDTH = 1.2
CURVE_LINE_WIDTH = 1.6
CONTOUR_LINE_WIDTH = 1.2
# Marker area (pt^2) of radar seeds in the section figure.
SECTION_SEED_MARKER_AREA = 30.0
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
# Figure inches of the single-panel bird's-eye figure (F1) and of one density panel (F3).
BEV_PANEL_WIDTH_IN = 9.0
BEV_PANEL_HEIGHT_IN = 8.0
# Section figure: inches per metre of the (along, z) window, with a floor so tiny windows stay readable.
SECTION_INCHES_PER_METRE = 0.3
SECTION_MIN_PANEL_IN = 1.6
# Section Gaussian marker edge: thin white outline (pt) so black dots separate from dark occupancy shading.
GAUSSIAN_EDGE_WIDTH = 0.3
GAUSSIAN_DOT_COLOUR = "black"
GAUSSIAN_EDGE_COLOUR = "white"
# Draw order of section layers (higher on top).
SECTION_GAUSSIAN_ZORDER = 3
SECTION_SEED_ZORDER = 4
# Legend columns of the section figure: 4 evidence names + occupancy + Gaussian + seed = 7 entries,
# five per row, hence two rows.
SECTION_LEGEND_COLUMNS = 5
SECTION_LEGEND_ROWS = 2
# Grey edge (matplotlib grey string) of legend patches, so white "unobserved" stays visible.
LEGEND_PATCH_EDGE = "0.6"
# Legend rows of the threshold figure: two entries per arm (pooled and mean line) laid out with one
# column per arm, hence two rows.
THRESHOLD_LEGEND_ROWS = 2
# Seed density colourbar label.
SEED_DENSITY_LABEL = "radar seeds per bin"

# Section Gaussian marker area (pt^2) of an opacity-1 Gaussian: area scales linearly with opacity,
# so a dim Gaussian reads small; 25 pt^2 keeps a full one visible beside a 0.2 m voxel.
GAUSSIAN_AREA_PER_UNIT_OPACITY = 25.0
# Alpha of the evidence background squares, of campaign spread bands, and of histogram fills.
EVIDENCE_BACKGROUND_ALPHA = 0.55
SPREAD_BAND_ALPHA = 0.2
# Dark single colour (RGB 0-1) of occupancy squares in the section figure.
OCCUPANCY_OVERLAY_RGB = (0.1, 0.1, 0.1)
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
# Legend font size (pt) and histogram line width (pt).
LEGEND_FONT_SIZE = 7
HISTOGRAM_LINE_WIDTH = 1.4
# A voxel (cell) edge is split into two equal halves to get from a cell centre to its face.
HALF = 0.5
# An RGBA colour has the RGB channels plus alpha as its last channel.
RGBA_CHANNEL_COUNT = RGB_CHANNEL_COUNT + 1
# Colourbar shrink (fraction of the axes height).
COLORBAR_SHRINK = 0.8


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
                panel_size: Tuple[float, float] = (PANEL_WIDTH_IN, PANEL_HEIGHT_IN)
                ) -> Tuple[Figure, np.ndarray]:
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

    Returns:
        (figure, 2D axes array [row, column]).
    """
    # Step 1: size = panel grid + title band + bottom band (note, footer, legend rows).
    width_in = panel_size[0] * column_count
    band = band_layout(width_in, run_note, settings.footer_note if footer else None, legend_rows * LEGEND_ROW_HEIGHT_IN)
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
    return fig, axes_grid


def _draw_polygon_outlines(axis: Any, polygons: Sequence[np.ndarray], colour: Any) -> None:
    """Draw each (K,2) corner array as a closed outline.

    Args:
        axis: Target axes.
        polygons: Corner arrays.
        colour: Line colour.

    Returns:
        None.
    """
    for polygon in polygons:
        # Step 1: repeat the first corner so the outline closes.
        closed = np.vstack([polygon, polygon[:1]])
        axis.plot(closed[:, 0], closed[:, 1], color=colour, linewidth=OUTLINE_LINE_WIDTH)


def occupancy_bev_figure(image: np.ndarray, extent: Tuple[float, float, float, float],
                         settings: FigureSettings, *, boxes_xy: Optional[Sequence[np.ndarray]] = None,
                         path_xy: Optional[np.ndarray] = None,
                         near_contour: Optional[np.ndarray] = None, title: str,
                         run_note: Optional[str] = None) -> Figure:
    """F1: bird's-eye occupancy map with optional GT boxes, driven path and H_near boundary.

    Args:
        image: [y, x] occupancy in [0, 1] (NaN = no data).
        extent: (xmin, xmax, ymin, ymax) in metres.
        settings: Figure settings.
        boxes_xy: Optional list of (K,2) box outline polygons (metres), drawn closed.
        path_xy: Optional (N,2) driven path (metres).
        near_contour: Optional bool [y, x] image on the same extent; its boundary is drawn.
        title: Figure title.
        run_note: Optional text naming the drawn run.

    Returns:
        The figure (not saved).
    """
    fig, axes_grid = _new_figure(1, 1, title, settings, footer=False, run_note=run_note,
                                 panel_size=(BEV_PANEL_WIDTH_IN, BEV_PANEL_HEIGHT_IN))
    axis = axes_grid[0, 0]
    # Step 1: occupancy image on a fixed 0-1 scale.
    shown = axis.imshow(image, origin="lower", extent=extent, vmin=UNIT_MIN, vmax=UNIT_MAX,
                        cmap=settings.occupancy_cmap)
    # Step 2: optional overlays.
    if boxes_xy:
        _draw_polygon_outlines(axis, boxes_xy, _unit_rgb(settings, "missed"))
    if path_xy is not None:
        axis.plot(path_xy[:, 0], path_xy[:, 1], color=_unit_rgb(settings, "false_alarm"),
                  linewidth=PATH_LINE_WIDTH)
    if near_contour is not None:
        axis.contour(near_contour.astype(float), levels=[BOOL_CONTOUR_LEVEL], extent=extent,
                     origin="lower", colors=[_unit_rgb(settings, "hit")],
                     linewidths=CONTOUR_LINE_WIDTH)
    # Step 3: overlays must not widen the view: limits stay on the scored-cell extent.
    axis.set_xlim(extent[0], extent[1])
    axis.set_ylim(extent[2], extent[3])
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal")
    fig.colorbar(shown, ax=axis, shrink=COLORBAR_SHRINK, label="occupancy (0-1)")
    return fig


def _log_density_panel(fig: Figure, axis: Any, density: np.ndarray, extent: Tuple[float, float, float, float],
                       settings: FigureSettings, colourbar_label: str) -> None:
    """Draw one density image on a log colour scale with a colourbar.

    Args:
        fig: Parent figure (owns the colourbar).
        axis: Target axes.
        density: [y, x] non-negative density; non-positive bins are masked.
        extent: (xmin, xmax, ymin, ymax) in metres.
        settings: Figure settings (colormap).
        colourbar_label: Colourbar label.

    Returns:
        None.
    """
    # Step 1: log scale needs a positive floor; use the smallest positive density.
    positive_values = density[np.isfinite(density) & (density > 0)]
    log_norm = LogNorm(vmin=positive_values.min(), vmax=positive_values.max()) \
        if positive_values.size else None
    # Step 2: mask non-positive bins so the log norm never sees them.
    masked = np.ma.masked_less_equal(np.ma.masked_invalid(density), 0)
    shown = axis.imshow(masked, origin="lower", extent=extent, cmap=settings.occupancy_cmap, norm=log_norm)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.set_aspect("equal")
    fig.colorbar(shown, ax=axis, shrink=COLORBAR_SHRINK, label=colourbar_label)


def gaussian_density_figure(density: np.ndarray, edges_x: np.ndarray, edges_y: np.ndarray,
                            settings: FigureSettings, *, seeds_xy: Optional[np.ndarray] = None,
                            title: str, run_note: Optional[str] = None) -> Figure:
    """F3: opacity-weighted Gaussian density (log colour scale), plus a radar-seed density panel for seed arms.

    The seed panel counts seeds per bin on the Gaussian panel's own bin edges (a unit-weight histogram), so it
    has exactly the same columns and rows and an opaque scatter never hides the density; an arm without seeds
    keeps one panel.

    Args:
        density: [y, x] opacity-weighted density.
        edges_x: Bin edges along x (metres).
        edges_y: Bin edges along y (metres).
        settings: Figure settings.
        seeds_xy: Optional (S,2) radar seed positions (metres); seeds outside the extent are dropped.
        title: Figure title.
        run_note: Optional text naming the drawn run.

    Returns:
        The figure (not saved).
    """
    has_seeds = seeds_xy is not None and len(seeds_xy) > 0
    fig, axes_grid = _new_figure(2 if has_seeds else 1, 1, title, settings, footer=False, run_note=run_note,
                                 panel_size=(BEV_PANEL_WIDTH_IN, BEV_PANEL_HEIGHT_IN))
    extent = (edges_x[0], edges_x[-1], edges_y[0], edges_y[-1])
    # Step 1: Gaussian panel.
    gaussian_axis = axes_grid[0, 0]
    _log_density_panel(fig, gaussian_axis, density, extent, settings, "opacity-weighted Gaussians per bin")
    gaussian_axis.set_title("Gaussian means (opacity-weighted)")
    if has_seeds:
        # Step 2: seed counts on the identical bin edges; points outside the edges are dropped.
        seeds_xy = np.asarray(seeds_xy, dtype=float)
        seed_density_xy, _, _ = np.histogram2d(seeds_xy[:, 0], seeds_xy[:, 1], bins=[edges_x, edges_y])
        seed_density = seed_density_xy.T  # np.histogram2d is [x, y]; images are [y, x]
        seed_axis = axes_grid[0, 1]
        _log_density_panel(fig, seed_axis, seed_density, extent, settings, SEED_DENSITY_LABEL)
        seed_axis.set_title(f"radar seeds (n={len(seeds_xy)})")
        # Step 3: the seed count also sits in the first panel's legend (marker-less entry).
        gaussian_axis.legend(handles=[Line2D([], [], linestyle="none",
                                             label=f"radar seeds (n={len(seeds_xy)}), right panel")],
                             loc="upper right", fontsize=LEGEND_FONT_SIZE)
    return fig


def arm_heat_figure(images_by_arm: Mapping[str, np.ndarray], extent: Tuple[float, float, float, float],
                    settings: FigureSettings, *, difference: Optional[Tuple[np.ndarray, str]] = None,
                    title: str, run_note: Optional[str] = None) -> Figure:
    """C5: one occupancy panel per arm (shared 0-1 scale) plus an optional difference panel.

    Args:
        images_by_arm: arm -> [y, x] image (NaN allowed).
        extent: (xmin, xmax, ymin, ymax) in metres.
        settings: Figure settings.
        difference: Optional (image, label); drawn with a diverging norm symmetric about 0.
        title: Figure title.
        run_note: Optional text naming the drawn runs.

    Returns:
        The figure (not saved).
    """
    arms = [arm for arm in settings.arm_order if arm in images_by_arm]
    column_count = len(arms) + (1 if difference is not None else 0)
    fig, axes_grid = _new_figure(column_count, 1, title, settings, footer=True,
                                 sharex=True, sharey=True, run_note=run_note)
    # Step 1: arm panels share one scale, so one colourbar serves them all.
    shown = None
    for panel_axis, arm in zip(axes_grid[0], arms):
        shown = panel_axis.imshow(images_by_arm[arm], origin="lower", extent=extent,
                                  vmin=UNIT_MIN, vmax=UNIT_MAX, cmap=settings.occupancy_cmap)
        panel_axis.set_title(settings.arm_labels.get(arm, arm))
    if shown is not None:
        fig.colorbar(shown, ax=list(axes_grid[0, :len(arms)]), shrink=COLORBAR_SHRINK,
                     label="occupancy (0-1)")
    # Step 2: difference panel, limits +-max|difference| so zero sits at the colormap centre.
    if difference is not None:
        difference_image, difference_label = difference
        limit = float(np.nanmax(np.abs(difference_image))) if np.isfinite(difference_image).any() else UNIT_MAX
        limit = limit if limit > 0 else UNIT_MAX
        difference_axis = axes_grid[0, -1]
        diff_shown = difference_axis.imshow(
            difference_image, origin="lower", extent=extent, cmap=settings.difference_cmap,
            norm=TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit))
        difference_axis.set_title(difference_label)
        fig.colorbar(diff_shown, ax=difference_axis, shrink=COLORBAR_SHRINK, label=difference_label)
    # Step 3: axis labels with units, equal aspect.
    for panel_axis in axes_grid[0]:
        panel_axis.set_xlabel("x (m)")
        panel_axis.set_aspect("equal")
    axes_grid[0, 0].set_ylabel("y (m)")
    return fig


def _plot_threshold_panel(axis: Any, arm_curves: Mapping[str, Mapping[str, np.ndarray]],
                          settings: FigureSettings, metric: Tuple[str, str, str, str],
                          campaign_for_band: Optional[Mapping[str, Any]]) -> None:
    """Draw one band x metric panel: spread bands, pooled (solid) and mean (dashed) lines.

    Args:
        axis: Target axes.
        arm_curves: arm -> curve dict from ``threshold_curves``.
        settings: Figure settings.
        metric: One entry of THRESHOLD_METRICS.
        campaign_for_band: Optional arm -> metric -> (thresholds, lower, upper).

    Returns:
        None.
    """
    metric_key, metric_label, pooled_key, mean_key = metric
    for arm in settings.arm_order:
        if arm not in arm_curves:
            continue
        colour = _arm_colour(settings, arm)
        curves = arm_curves[arm]
        # Step 1: campaign spread band first, so lines draw on top of it.
        if campaign_for_band and arm in campaign_for_band and metric_key in campaign_for_band[arm]:
            band_thresholds, lower, upper = campaign_for_band[arm][metric_key]
            axis.fill_between(band_thresholds, lower, upper, color=colour, alpha=SPREAD_BAND_ALPHA,
                              linewidth=0)
        # Step 2: pooled solid, mean-over-objects dashed.
        label = settings.arm_labels.get(arm, arm)
        axis.plot(curves["thresholds"], curves[pooled_key], color=colour,
                  linewidth=CURVE_LINE_WIDTH, label=f"{label} (pooled)")
        axis.plot(curves["thresholds"], curves[mean_key], color=colour, linestyle=MEAN_LINESTYLE,
                  linewidth=CURVE_LINE_WIDTH, label=f"{label} (mean over objects)")
    axis.set_xlabel("occupancy threshold")
    axis.set_ylabel(metric_label)
    axis.set_ylim(UNIT_MIN, UNIT_MAX)


def threshold_curves_figure(curves_by_band_by_arm: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
                            settings: FigureSettings, *, p2_thresholds: Sequence[float],
                            campaign_bands: Optional[Mapping[str, Any]] = None,
                            title: str, run_note: Optional[str] = None) -> Figure:
    """C4: recall and precision against occupancy threshold, one row per band.

    Args:
        curves_by_band_by_arm: band -> arm -> ``threshold_curves`` dict.
        settings: Figure settings.
        p2_thresholds: Thresholds marked by vertical dotted lines.
        campaign_bands: Optional band -> arm -> metric -> (thresholds, lower, upper) spread.
        title: Figure title.
        run_note: Optional text naming the drawn runs.

    Returns:
        The figure (not saved).
    """
    bands = list(curves_by_band_by_arm)
    fig, axes_grid = _new_figure(len(THRESHOLD_METRICS), len(bands), title, settings, footer=True,
                                 run_note=run_note, legend_rows=THRESHOLD_LEGEND_ROWS)
    for row_index, band in enumerate(bands):
        for column_index, metric in enumerate(THRESHOLD_METRICS):
            panel_axis = axes_grid[row_index, column_index]
            # Step 1: one panel per (band row, metric column): spread bands and curves of every arm.
            _plot_threshold_panel(panel_axis, curves_by_band_by_arm[band], settings, metric,
                                  (campaign_bands or {}).get(band))
            # Step 2: guides at the P2 thresholds, and a band title on the first column.
            for threshold in p2_thresholds:
                panel_axis.axvline(threshold, color=GUIDE_COLOUR, linestyle=GUIDE_LINESTYLE,
                                   linewidth=GUIDE_LINE_WIDTH)
            if column_index == 0:
                panel_axis.set_title(band, loc="left")
    # Step 3: one shared legend from the first panel; one column per arm puts pooled and mean in two rows.
    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(FOOTER_X_FRACTION, fig.legend_anchor_y),
               ncol=max(1, len(settings.arm_order)), fontsize=LEGEND_FONT_SIZE)
    return fig


def occupancy_histogram_edges(bins: int) -> np.ndarray:
    """Shared bin edges of the occupancy histograms: ``bins`` equal bins over [0, 1].

    Args:
        bins: Bin count.

    Returns:
        (bins + 1,) edges; the same edges bin the numbers a caller reports.
    """
    return np.linspace(UNIT_MIN, UNIT_MAX, bins + 1)


def occupancy_histogram_figure(samples_by_arm: Mapping[str, Mapping[str, np.ndarray]],
                               settings: FigureSettings, *, bins: int, title: str,
                               run_note: Optional[str] = None) -> Figure:
    """Per-arm density histograms of occupancy on truth cells versus off-truth cells.

    Args:
        samples_by_arm: arm -> {"truth": values, "off_truth": values} in [0, 1].
        settings: Figure settings.
        bins: Bin count over [0, 1], shared by all panels.
        title: Figure title.
        run_note: Optional text naming the drawn runs.

    Returns:
        The figure (not saved).
    """
    arms = [arm for arm in settings.arm_order if arm in samples_by_arm]
    fig, axes_grid = _new_figure(len(arms), 1, title, settings, footer=True, sharex=True, sharey=True,
                                 run_note=run_note)
    # Step 1: shared bin edges so panels are comparable.
    edges = occupancy_histogram_edges(bins)
    series = (("truth", "on truth", "hit"), ("off_truth", "off truth", "false_alarm"))
    for panel_axis, arm in zip(axes_grid[0], arms):
        for key, series_label, colour_name in series:
            values = np.asarray(samples_by_arm[arm][key], dtype=float)
            values = values[np.isfinite(values)]
            # Step 2: density=True makes the area 1 whatever the count; empty sets are skipped.
            if values.size:
                panel_axis.hist(values, bins=edges, density=True, histtype="step",
                                color=_unit_rgb(settings, colour_name),
                                linewidth=HISTOGRAM_LINE_WIDTH, label=f"{series_label} (n={values.size})")
        panel_axis.set_title(settings.arm_labels.get(arm, arm))
        panel_axis.set_xlabel("occupancy (0-1)")
        panel_axis.legend(fontsize=LEGEND_FONT_SIZE)
    axes_grid[0, 0].set_ylabel("density (1 / occupancy unit)")
    return fig


def _square_vertices(cell_coords: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return axis-aligned square corners of side ``voxel_size`` centred on each cell.

    Args:
        cell_coords: (M,2) cell centres.
        voxel_size: Square side (metres).

    Returns:
        (M,4,2) corner array.
    """
    # Step 1: corner offsets are +-half a voxel around the centre.
    half = voxel_size * HALF
    offsets = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    return cell_coords[:, None, :] + offsets[None, :, :]


def _evidence_colour(settings: FigureSettings, evidence_name: str) -> Tuple[float, float, float]:
    """Unit RGB of an evidence class from the settings entry ``evidence_<name>``.

    Args:
        settings: Figure settings.
        evidence_name: An entry of EVIDENCE_NAMES.

    Returns:
        (R, G, B) in [0, 1].
    """
    return _unit_rgb(settings, f"evidence_{evidence_name}")


def _draw_section_panel(axis: Any, section: Mapping[str, Any], arm_data: Mapping[str, Any],
                        settings: FigureSettings, voxel_size: float) -> None:
    """Draw one section x arm panel (evidence squares, occupancy, GT outline, Gaussians, seeds).

    Args:
        axis: Target axes.
        section: One section dict (see ``occlusion_section_figure``).
        arm_data: The section's ``by_arm`` entry for this arm.
        settings: Figure settings.
        voxel_size: Cell side (metres).

    Returns:
        None.
    """
    cell_coords = np.asarray(section["cells_coords"], dtype=float).reshape(-1, 2)
    squares = _square_vertices(cell_coords, voxel_size)
    # Step 1: evidence-coloured background squares, colour looked up by EVIDENCE_NAMES position.
    evidence_colours = [_evidence_colour(settings, EVIDENCE_NAMES[label])
                        for label in np.asarray(section["evidence_labels"], dtype=int)]
    axis.add_collection(PolyCollection(squares, facecolors=evidence_colours, edgecolors="none",
                                       alpha=EVIDENCE_BACKGROUND_ALPHA))
    # Step 2: occupancy squares, alpha = occupancy (clipped to [0, 1], NaN = invisible).
    occupancy = np.nan_to_num(np.asarray(arm_data["occupancy"], dtype=float), nan=UNIT_MIN)
    overlay = np.zeros((len(occupancy), RGBA_CHANNEL_COUNT))
    overlay[:, :RGB_CHANNEL_COUNT] = OCCUPANCY_OVERLAY_RGB
    overlay[:, RGB_CHANNEL_COUNT] = np.clip(occupancy, UNIT_MIN, UNIT_MAX)
    axis.add_collection(PolyCollection(squares, facecolors=overlay, edgecolors="none"))
    # Step 3: GT outline, then Gaussian means (black, white edge, area ~ opacity), then seeds on top.
    _draw_polygon_outlines(axis, [np.asarray(section["box_polygon"], dtype=float)],
                           _unit_rgb(settings, "false_alarm"))
    gaussian_coords = np.asarray(arm_data["gaussian_coords"], dtype=float).reshape(-1, 2)
    gaussian_opacity = np.asarray(arm_data["gaussian_opacity"], dtype=float)
    axis.scatter(gaussian_coords[:, 0], gaussian_coords[:, 1],
                 s=gaussian_opacity * GAUSSIAN_AREA_PER_UNIT_OPACITY, color=GAUSSIAN_DOT_COLOUR,
                 edgecolors=GAUSSIAN_EDGE_COLOUR, linewidths=GAUSSIAN_EDGE_WIDTH, zorder=SECTION_GAUSSIAN_ZORDER)
    seed_coords = arm_data.get("seed_coords")
    if seed_coords is not None and len(seed_coords):
        axis.scatter(seed_coords[:, 0], seed_coords[:, 1], marker="x", s=SECTION_SEED_MARKER_AREA,
                     color=_unit_rgb(settings, "radar_seed"), zorder=SECTION_SEED_ZORDER)
    # Step 4: collections do not autoscale: use the padded window when given, else fit the squares.
    window = section.get("window")
    if window is not None:
        lower, upper = (np.asarray(bound, dtype=float) for bound in window)
        axis.set_xlim(lower[0], upper[0])
        axis.set_ylim(lower[1], upper[1])
    elif len(squares):
        axis.set_xlim(squares[..., 0].min(), squares[..., 0].max())
        axis.set_ylim(squares[..., 1].min(), squares[..., 1].max())
    axis.set_aspect("equal")


def _section_span(section: Mapping[str, Any], voxel_size: float) -> Tuple[float, float]:
    """(along, z) extent in metres of a section's drawn window (or of its cells when no window is given).

    Args:
        section: One section dict.
        voxel_size: Cell side (metres).

    Returns:
        (along span, z span).
    """
    window = section.get("window")
    if window is not None:
        lower, upper = (np.asarray(bound, dtype=float) for bound in window)
        return float(upper[0] - lower[0]), float(upper[1] - lower[1])
    coords = np.asarray(section["cells_coords"], dtype=float).reshape(-1, 2)
    if len(coords) == 0:
        return voxel_size, voxel_size
    spans = coords.max(axis=0) - coords.min(axis=0) + voxel_size
    return float(spans[0]), float(spans[1])


def occlusion_section_figure(sections: Sequence[Mapping[str, Any]], settings: FigureSettings, *,
                             voxel_size: float, title: str, run_note: Optional[str] = None) -> Figure:
    """C3: trunk-section panels (rows = sections, columns = arms) over an evidence background.

    Each section dict holds ``label``, ``cells_coords`` (M,2 distance-from-sensor/z), ``evidence_labels`` (M,)
    ints into EVIDENCE_NAMES, ``box_polygon`` (K,2), ``by_arm`` arm -> {``occupancy`` (M,),
    ``gaussian_coords`` (G,2), ``gaussian_opacity`` (G,), ``seed_coords`` (S,2) or None} and optionally
    ``window`` ((lower, upper) of the drawn limits). The panel size follows the data aspect: width and height
    are the largest window spans times SECTION_INCHES_PER_METRE (floored at SECTION_MIN_PANEL_IN).

    Args:
        sections: One dict per row.
        settings: Figure settings.
        voxel_size: Cell side (metres).
        title: Figure title.
        run_note: Optional text naming the drawn runs.

    Returns:
        The figure (not saved).

    Raises:
        ValueError: When ``sections`` is empty.
    """
    if not sections:
        raise ValueError("occlusion_section_figure needs at least one section")
    arms = list(settings.arm_order)
    # Step 1: panel inches from the largest window, so equal-aspect panels fill their slots.
    spans = [_section_span(section, voxel_size) for section in sections]
    panel_width = max(SECTION_MIN_PANEL_IN, max(span[0] for span in spans) * SECTION_INCHES_PER_METRE)
    panel_height = max(SECTION_MIN_PANEL_IN, max(span[1] for span in spans) * SECTION_INCHES_PER_METRE)
    fig, axes_grid = _new_figure(len(arms), len(sections), title, settings, footer=True, run_note=run_note,
                                 legend_rows=SECTION_LEGEND_ROWS, panel_size=(panel_width, panel_height))
    for row_index, section in enumerate(sections):
        for column_index, arm in enumerate(arms):
            panel_axis = axes_grid[row_index, column_index]
            if arm in section["by_arm"]:
                _draw_section_panel(panel_axis, section, section["by_arm"][arm], settings, voxel_size)
            # Step 2: titles on the top row, row label on the first column, units everywhere.
            if row_index == 0:
                panel_axis.set_title(settings.arm_labels.get(arm, arm))
            panel_axis.set_xlabel("distance from sensor\nalong ray (m)")
            # Same window in every column of a row, so inner columns drop their y labels (no wide gaps).
            if column_index == 0:
                panel_axis.set_ylabel(f"{section['label']}\nheight z (m)")
            else:
                panel_axis.tick_params(labelleft=False)
    # Step 3: one shared legend: evidence classes, occupancy shading, Gaussian mean, radar seed.
    handles: List[Any] = [Patch(facecolor=_evidence_colour(settings, name), edgecolor=LEGEND_PATCH_EDGE,
                                alpha=EVIDENCE_BACKGROUND_ALPHA, label=name) for name in EVIDENCE_NAMES]
    handles.append(Patch(facecolor=OCCUPANCY_OVERLAY_RGB, label="occupancy (alpha)"))
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor=GAUSSIAN_DOT_COLOUR,
                          markeredgecolor=GAUSSIAN_EDGE_COLOUR, markeredgewidth=GAUSSIAN_EDGE_WIDTH,
                          color=GAUSSIAN_DOT_COLOUR, label="Gaussian mean (size ~ opacity)"))
    handles.append(Line2D([], [], marker="x", linestyle="none", color=_unit_rgb(settings, "radar_seed"),
                          label="radar seed"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(FOOTER_X_FRACTION, fig.legend_anchor_y),
               ncol=SECTION_LEGEND_COLUMNS, fontsize=LEGEND_FONT_SIZE)
    return fig
