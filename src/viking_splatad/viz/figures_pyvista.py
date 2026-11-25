"""Off-screen PyVista renderers of the Gaussian-occupancy figure suite (F2 strata, C1 arm panels, C2 tiles).

PyVista is installed only in the isolated ``viz/`` project, so nothing here imports it at module level:
every renderer takes the imported module as ``pyvista_module`` (as ``pyvista_scene.draw_voxels`` does).
Renderers return RGB uint8 images and never write files. Voxel drawing reuses ``pyvista_scene``; the
oblique camera helper lives only here (scripts/15 uses PyVista's default and ``view_xy`` instead).
Matplotlib is imported lazily, and only inside ``compose_tile_grid``.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.viz.figure_grids import OUTCOME_STRATUM_NAMES
from viking_splatad.viz.figure_runs import FigureSettings
from viking_splatad.viz.pyvista_scene import (
    CUBOID_WIREFRAME_RGB,
    MAX_CHANNEL_VALUE,
    SOLID_VOXEL_OPACITY,
    VOXEL_AMBIENT_LIGHT,
    WIREFRAME_LINE_WIDTH,
    draw_voxels,
)

# Translucency (0-1, presentation choices): faint context, translucent false alarms and misses.
CONTEXT_OPACITY = 0.15
FALSE_ALARM_OPACITY = 0.5
MISSED_OPACITY = 0.35
# Missed-hidden grey = the settings' "missed" grey times this factor (darker, so hidden misses stand out
# from visible ones); a presentation choice.
MISSED_HIDDEN_DARKEN_FACTOR = 0.55
# Occupancy is a probability, so its colour scale is fixed to [0, 1].
OCCUPANCY_CLIM = (0.0, 1.0)
OCCUPANCY_SCALAR_NAME = "occupancy"
# Oblique camera: unit-less direction from the focal point to the camera (x right, y forward, z up), the
# vertical field of view (PyVista/VTK default 30 degrees) and a framing margin around the drawn cells.
OBLIQUE_CAMERA_DIRECTION = (1.0, -1.0, 0.9)
CAMERA_VIEW_ANGLE_DEGREES = 30.0
CAMERA_FRAMING_MARGIN = 1.15
VIEW_UP_VECTOR = (0.0, 0.0, 1.0)
# Legend / title text size (points) of the off-screen render: the window is ~1000 px wide per panel, so 8 pt
# would print about 11 px glyphs; 14 pt stays legible once the screenshot is scaled into a figure.
TEXT_FONT_SIZE = 14
# Tighter camera margin for the scene figures (C1, and the default of ``render_outcome_strata``, whose F2 caller
# passes its own margin): the bounding sphere of a flat, wide scene overstates the
# projected extent of an oblique view, so 0.9 x the sphere distance still frames the drawn cells.
TIGHT_SCENE_FRAMING_MARGIN = 0.9
# Scene-figure layout (inches): width of one scene panel when the screenshot is placed in a matplotlib figure,
# title band, and font size (pt) of its legend and titles.
SCENE_PANEL_WIDTH_IN = 6.0
SCENE_TITLE_BAND_IN = 0.45
SCENE_FONT_SIZE = 11
# Width (inches) one legend entry needs at SCENE_FONT_SIZE (swatch plus a label such as "false_alarm: 1272"):
# legends use as many columns as fit the figure width, and wrap into more rows otherwise.
SCENE_LEGEND_ENTRY_WIDTH_IN = 2.0
# Height (inches) of one legend row at SCENE_FONT_SIZE.
SCENE_LEGEND_ROW_HEIGHT_IN = 0.3
# Tile-grid key (colourbar + legend) band height and title band (inches), colourbar placement (figure fractions).
TILE_KEY_BAND_IN = 0.8
TILE_TITLE_BAND_IN = 0.4
TILE_COLOURBAR_LEFT, TILE_COLOURBAR_WIDTH = 0.08, 0.38
TILE_COLOURBAR_HEIGHT_IN = 0.14
TILE_LEGEND_X_FRACTION = 0.55
# Legend entry kinds of ``compose_scene_figure`` and tile key.
LEGEND_KIND_PATCH, LEGEND_KIND_LINE = "patch", "line"
# Digits after the decimal point in tile captions (e.g. "recall_H 0.33").
CAPTION_DECIMAL_PLACES = 2
# Font size (pt) of the grid's labels and captions, and of its title (2 pt larger so the title leads).
GRID_FONT_SIZE = 8
GRID_TITLE_FONT_SIZE = GRID_FONT_SIZE + 2
# The tile-grid colourbar's bottom edge sits at this fraction of the key band's height (its vertical middle).
KEY_BAND_MIDDLE_FRACTION = 0.5
# Settings colour names of the two difference-panel arms (hits of only the first / only the second arm).
FIRST_ARM_COLOUR_NAME = "only_radar_init_loss"
SECOND_ARM_COLOUR_NAME = "only_stock"
# Room (figure fraction) reserved at the bottom for the footer note.
FOOTER_BOTTOM_MARGIN = 0.04


def to_unit_rgb(colour_rgb: Sequence[int]) -> np.ndarray:
    """Turn a 0-255 RGB triple into the 0-1 RGB PyVista takes.

    Args:
        colour_rgb: 0-255 RGB.

    Returns:
        (3,) float array in [0, 1].
    """
    return np.asarray(colour_rgb, dtype=float) / MAX_CHANNEL_VALUE


def darken_rgb(colour_rgb: Sequence[int], factor: float) -> Tuple[int, int, int]:
    """Scale a 0-255 RGB triple toward black.

    Args:
        colour_rgb: 0-255 RGB.
        factor: Multiplier in (0, 1]; smaller is darker.

    Returns:
        Integer RGB triple.
    """
    return tuple(int(round(channel * factor)) for channel in colour_rgb)


def oblique_camera_position(cell_centres: np.ndarray, voxel_size: float,
                            margin: float = CAMERA_FRAMING_MARGIN) -> List[Tuple[float, float, float]]:
    """Camera (position, focal point, view-up) that frames cells from an oblique direction.

    The focal point is the middle of the cells' bounding box (voxel edges included). The bounding sphere has
    radius r = half the box diagonal; a camera at distance d = margin * r / tan(fov / 2) along the fixed
    oblique direction sees the whole sphere.

    Args:
        cell_centres: (V, 3) centres to frame (at least one).
        voxel_size: Voxel edge (m), added around the centres.
        margin: Distance multiplier; below 1 crops the bounding sphere, which still frames a flat wide scene.

    Returns:
        [position, focal_point, view_up], each a 3-tuple.
    """
    # Step 1: bounding box padded by half a voxel on every side.
    half_edge = voxel_size / 2.0
    lower_corner = cell_centres.min(axis=0) - half_edge
    upper_corner = cell_centres.max(axis=0) + half_edge
    focal_point = (lower_corner + upper_corner) / 2.0
    # Step 2: bounding-sphere radius and the distance that fits it in the field of view.
    sphere_radius = np.linalg.norm(upper_corner - lower_corner) / 2.0
    distance = margin * sphere_radius / np.tan(np.radians(CAMERA_VIEW_ANGLE_DEGREES) / 2.0)
    # Step 3: step back along the unit oblique direction.
    direction = np.asarray(OBLIQUE_CAMERA_DIRECTION, dtype=float)
    position = focal_point + distance * direction / np.linalg.norm(direction)
    return [tuple(position), tuple(focal_point), VIEW_UP_VECTOR]


def draw_scalar_voxels(plotter, pyvista_module, centres: np.ndarray, voxel_size: float, scalars: np.ndarray,
                       colormap: str, opacity: float = SOLID_VOXEL_OPACITY, show_scalar_bar: bool = False) -> None:
    """Draw voxels as cubes coloured by a per-voxel scalar (0-1) through a continuous colormap.

    Args:
        plotter: ``pyvista.Plotter`` with the panel's subplot active.
        pyvista_module: The imported ``pyvista``.
        centres: (V, 3) voxel centres; nothing is drawn when empty.
        voxel_size: Voxel edge (m).
        scalars: (V,) values, mapped over ``OCCUPANCY_CLIM``.
        colormap: Matplotlib colormap name.
        opacity: 0-1.
        show_scalar_bar: Whether to add a colour bar.
    """
    if centres.shape[0] == 0:
        return
    # Step 1: attach the scalar to the points; glyphing copies point data onto each cube's faces.
    cube = pyvista_module.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    cloud = pyvista_module.PolyData(centres)
    cloud[OCCUPANCY_SCALAR_NAME] = np.asarray(scalars, dtype=float)
    cubes = cloud.glyph(geom=cube, orient=False, scale=False)
    # Step 2: colour through the colormap over the fixed probability range.
    plotter.add_mesh(cubes, scalars=OCCUPANCY_SCALAR_NAME, cmap=colormap, clim=OCCUPANCY_CLIM, opacity=opacity,
                     ambient=VOXEL_AMBIENT_LIGHT, diffuse=1.0 - VOXEL_AMBIENT_LIGHT,
                     show_scalar_bar=show_scalar_bar)


def capture_image(plotter) -> np.ndarray:
    """Screenshot an off-screen plotter as an RGB array.

    Args:
        plotter: ``pyvista.Plotter`` created with ``off_screen=True``.

    Returns:
        (H, W, 3) uint8 image.
    """
    return np.asarray(plotter.screenshot(return_img=True), dtype=np.uint8)


def outcome_stratum_styles(settings: FigureSettings) -> Dict[str, Tuple[Tuple[int, int, int], float]]:
    """Colour (0-255 RGB) and opacity of each outcome stratum, shared by the drawing and its legend.

    Args:
        settings: Figure settings (colours).

    Returns:
        stratum name -> (RGB, opacity).
    """
    return {
        "hit_seen": (settings.colours["hit"], SOLID_VOXEL_OPACITY),
        "hit_hidden": (settings.colours[FIRST_ARM_COLOUR_NAME], SOLID_VOXEL_OPACITY),
        "false_alarm": (settings.colours["false_alarm"], FALSE_ALARM_OPACITY),
        "missed_seen": (settings.colours["missed"], MISSED_OPACITY),
        "missed_hidden": (darken_rgb(settings.colours["missed"], MISSED_HIDDEN_DARKEN_FACTOR), MISSED_OPACITY),
    }


def render_outcome_strata(pyvista_module, centres: np.ndarray, voxel_size: float, strata: np.ndarray,
                          settings: FigureSettings, *, window_size: Tuple[int, int], title: str,
                          draw_text_legend: bool = True,
                          framing_margin: float = TIGHT_SCENE_FRAMING_MARGIN) -> np.ndarray:
    """F2: one panel of the cells drawn per outcome stratum, optionally with a text legend of per-stratum counts.

    Colours: hit_seen = settings "hit"; hit_hidden = settings FIRST_ARM_COLOUR_NAME (blue, so a hit on a
    never-seen cell differs from a hit on a seen one); false_alarm = "false_alarm" translucent;
    missed_seen = "missed" grey translucent; missed_hidden = that grey darkened by ``MISSED_HIDDEN_DARKEN_FACTOR``.

    Args:
        pyvista_module: The imported ``pyvista``.
        centres: (V, 3) cell centres.
        voxel_size: Voxel edge (m).
        strata: (V,) int index into ``OUTCOME_STRATUM_NAMES`` (-1 = not drawn).
        settings: Figure settings (colours).
        window_size: (width, height) in pixels.
        title: First line of the legend text.
        draw_text_legend: Whether to print the title and per-stratum counts into the image; the figure
            scripts pass False and draw a colour legend with the same counts around the image instead.
        framing_margin: Camera distance multiplier (see ``oblique_camera_position``).

    Returns:
        (H, W, 3) uint8 image.
    """
    colour_by_stratum = outcome_stratum_styles(settings)
    plotter = pyvista_module.Plotter(off_screen=True, window_size=window_size)
    try:
        legend_lines = [title]
        # Step 1: one draw call per stratum; count its cells for the legend.
        for stratum_index, stratum_name in enumerate(OUTCOME_STRATUM_NAMES):
            stratum_mask = strata == stratum_index
            colour_rgb, opacity = colour_by_stratum[stratum_name]
            draw_voxels(plotter, pyvista_module, centres[stratum_mask], voxel_size, colour_rgb, opacity)
            legend_lines.append(f"{stratum_name}: {int(stratum_mask.sum())}")
        # Step 2: legend, camera on the drawn cells, then the image.
        if draw_text_legend:
            plotter.add_text("\n".join(legend_lines), font_size=TEXT_FONT_SIZE)
        drawn_centres = centres[strata >= 0]
        if drawn_centres.shape[0]:
            plotter.camera_position = oblique_camera_position(drawn_centres, voxel_size, framing_margin)
        plotter.enable_depth_peeling()
        return capture_image(plotter)
    finally:
        plotter.close()


def render_arm_panels(pyvista_module, centres: np.ndarray, voxel_size: float,
                      outcomes_by_arm: Mapping[str, Mapping[str, np.ndarray]],
                      difference_sets: Mapping[str, np.ndarray], settings: FigureSettings, *,
                      window_size: Tuple[int, int], title_by_panel: Sequence[str],
                      first_colour_name: str = FIRST_ARM_COLOUR_NAME,
                      second_colour_name: str = SECOND_ARM_COLOUR_NAME) -> np.ndarray:
    """C1: one panel per arm (context, hits, false alarms, misses) plus a difference panel, on linked cameras.

    Arm panels follow ``settings.arm_order`` (only arms present in ``outcomes_by_arm``). The last panel
    colours hits of only the first arm (``only_first``, settings colour ``first_colour_name``, blue),
    only the second (``second_colour_name``, orange) and both ("both" green).

    Args:
        pyvista_module: The imported ``pyvista``.
        centres: (V, 3) cell centres.
        voxel_size: Voxel edge (m).
        outcomes_by_arm: arm -> {"context","hit","false_alarm","missed"} (V,) bool (``hidden_cell_outcomes``).
        difference_sets: {"only_first","only_second","both"} (V,) bool (``arm_difference_sets``).
        settings: Figure settings (arm order, colours).
        window_size: (width, height) of the whole window in pixels.
        title_by_panel: One title per panel, arms first and the difference panel last.
        first_colour_name: Settings colour name of the hits only the first compared arm has.
        second_colour_name: Settings colour name of the hits only the second compared arm has.

    Returns:
        (H, W, 3) uint8 image.
    """
    arm_keys = [arm for arm in settings.arm_order if arm in outcomes_by_arm]
    panel_count = len(arm_keys) + 1
    # Step 1: camera fitted to the union of every cell any panel draws.
    drawn_mask = np.zeros(centres.shape[0], dtype=bool)
    for arm_key in arm_keys:
        for outcome_name in ("context", "hit", "false_alarm", "missed"):
            drawn_mask |= np.asarray(outcomes_by_arm[arm_key][outcome_name], dtype=bool)
    for difference_mask in difference_sets.values():
        drawn_mask |= np.asarray(difference_mask, dtype=bool)
    plotter = pyvista_module.Plotter(shape=(1, panel_count), off_screen=True, window_size=window_size)
    try:
        # Step 2: arm panels, faint context first so the solid hits sit on top.
        for panel_index, arm_key in enumerate(arm_keys):
            plotter.subplot(0, panel_index)
            outcomes = outcomes_by_arm[arm_key]
            for outcome_name, colour_name, opacity in (("context", "context", CONTEXT_OPACITY),
                                                       ("missed", "missed", MISSED_OPACITY),
                                                       ("false_alarm", "false_alarm", FALSE_ALARM_OPACITY),
                                                       ("hit", "hit", SOLID_VOXEL_OPACITY)):
                draw_voxels(plotter, pyvista_module, centres[np.asarray(outcomes[outcome_name], dtype=bool)],
                            voxel_size, settings.colours[colour_name], opacity)
            plotter.add_text(title_by_panel[panel_index], font_size=TEXT_FONT_SIZE)
        # Step 3: difference panel.
        plotter.subplot(0, panel_count - 1)
        for set_name, colour_name in (("both", "both"), ("only_first", first_colour_name),
                                      ("only_second", second_colour_name)):
            draw_voxels(plotter, pyvista_module, centres[np.asarray(difference_sets[set_name], dtype=bool)],
                        voxel_size, settings.colours[colour_name], SOLID_VOXEL_OPACITY)
        plotter.add_text(title_by_panel[panel_count - 1], font_size=TEXT_FONT_SIZE)
        # Step 4: same oblique camera and depth peeling in every panel, then link the views.
        for panel_index in range(panel_count):
            plotter.subplot(0, panel_index)
            if drawn_mask.any():
                plotter.camera_position = oblique_camera_position(centres[drawn_mask], voxel_size,
                                                                  TIGHT_SCENE_FRAMING_MARGIN)
            plotter.enable_depth_peeling()
        plotter.link_views()
        return capture_image(plotter)
    finally:
        plotter.close()


def render_object_tile(pyvista_module, centres: np.ndarray, voxel_size: float, tile: Mapping[str, Any],
                       occupancy: np.ndarray, truth: np.ndarray, hidden: np.ndarray, false_alarm: np.ndarray,
                       settings: FigureSettings, *, window_size: Tuple[int, int],
                       cuboid_wireframe: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """C2: close-up of one object. Hidden GT cells coloured by occupancy, visible GT faint, false alarms red.

    The camera looks at ``tile["centre"]`` from the oblique direction, at the distance that frames the box
    ``[extent_min, extent_max]`` (``oblique_camera_position`` on the two corners). The GT drawn is exactly the
    metric's G_k = truth restricted to the object's own box cells (``tile["inside_cells"]``; truth already lies in
    the neighbourhood), so a neighbouring object's GT cell in the grown box is not drawn as this object's and the
    drawing agrees with the caption; false alarms are the flagged cells inside the tile box (padded by half a voxel).

    Args:
        pyvista_module: The imported ``pyvista``.
        centres: (V, 3) cell centres.
        voxel_size: Voxel edge (m).
        tile: One ``object_tiles`` entry.
        occupancy: (V,) Gaussian occupancy of every cell.
        truth: (V,) bool, GT cells.
        hidden: (V,) bool, hidden flag.
        false_alarm: (V,) bool, false-alarm cells.
        settings: Figure settings (colours, occupancy colormap).
        window_size: (width, height) in pixels.
        cuboid_wireframe: Optional (corner points, VTK line cells) of the object box.

    Returns:
        (H, W, 3) uint8 image.
    """
    # Step 1: G_k = the object's inside cells that are GT, split by the hidden flag; false alarms inside the
    # padded tile box.
    inside_cells = np.asarray(tile["inside_cells"], dtype=int)
    tile_truth = inside_cells[np.asarray(truth, dtype=bool)[inside_cells]]
    hidden_truth = tile_truth[np.asarray(hidden, dtype=bool)[tile_truth]]
    visible_truth = tile_truth[~np.asarray(hidden, dtype=bool)[tile_truth]]
    half_edge = voxel_size / 2.0
    inside_box = np.all((centres >= np.asarray(tile["extent_min"]) - half_edge)
                        & (centres <= np.asarray(tile["extent_max"]) + half_edge), axis=1)
    tile_false_alarm = np.flatnonzero(np.asarray(false_alarm, dtype=bool) & inside_box)
    plotter = pyvista_module.Plotter(off_screen=True, window_size=window_size)
    try:
        # Step 2: faint visible GT, false alarms, then the occupancy-coloured hidden GT.
        draw_voxels(plotter, pyvista_module, centres[visible_truth], voxel_size, settings.colours["context"],
                    CONTEXT_OPACITY)
        draw_voxels(plotter, pyvista_module, centres[tile_false_alarm], voxel_size, settings.colours["false_alarm"],
                    FALSE_ALARM_OPACITY)
        draw_scalar_voxels(plotter, pyvista_module, centres[hidden_truth], voxel_size, np.asarray(occupancy)[hidden_truth],
                           settings.occupancy_cmap)
        if cuboid_wireframe is not None and cuboid_wireframe[0].shape[0]:
            plotter.add_mesh(pyvista_module.PolyData(cuboid_wireframe[0], lines=cuboid_wireframe[1]),
                             color=to_unit_rgb(CUBOID_WIREFRAME_RGB), line_width=WIREFRAME_LINE_WIDTH)
        # Step 3: close-up camera: framing of the tile box, re-centred exactly on the tile centre.
        framing = oblique_camera_position(np.vstack([tile["extent_min"], tile["extent_max"]]), voxel_size)
        tile_centre = np.asarray(tile["centre"], dtype=float)
        shift = tile_centre - np.asarray(framing[1])
        plotter.camera_position = [tuple(np.asarray(framing[0]) + shift), tuple(tile_centre), VIEW_UP_VECTOR]
        plotter.enable_depth_peeling()
        return capture_image(plotter)
    finally:
        plotter.close()


def paginate(items: Sequence[Any], rows_per_page: int) -> List[List[Any]]:
    """Split items into consecutive pages of at most ``rows_per_page`` items.

    Args:
        items: Rows to split.
        rows_per_page: Page capacity (positive).

    Returns:
        List of pages (lists); empty for no items; the last page holds the remainder.

    Raises:
        ValueError: If ``rows_per_page`` is not positive.
    """
    if rows_per_page < 1:
        raise ValueError("rows_per_page must be positive")
    # Ceil(len / rows_per_page) pages, each a slice.
    return [list(items[start:start + rows_per_page]) for start in range(0, len(items), rows_per_page)]


def format_tile_caption(recall_hidden: Optional[float], baseline_recall: Optional[float]) -> str:
    """Caption text of a tile, e.g. ``recall_H 0.33 · baseline 0.80`` (``n/a`` for None).

    Args:
        recall_hidden: Hidden recall of the object, or None.
        baseline_recall: Whole-scene baseline recall, or None.

    Returns:
        The caption.
    """
    def _format(value: Optional[float]) -> str:
        """Fixed-precision number text, or ``n/a``."""
        return "n/a" if value is None else f"{value:.{CAPTION_DECIMAL_PLACES}f}"
    return f"recall_H {_format(recall_hidden)} · baseline {_format(baseline_recall)}"


def _legend_handles(entries: Sequence[Tuple[str, Sequence[int], str]]) -> List[Any]:
    """Matplotlib legend handles from (label, 0-255 RGB, kind) entries.

    Args:
        entries: ``(label, colour_rgb, kind)`` with kind LEGEND_KIND_PATCH (filled square) or LEGEND_KIND_LINE.

    Returns:
        Patch / Line2D handles.
    """
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from viking_splatad.viz.figures_mpl import LEGEND_PATCH_EDGE
    handles: List[Any] = []
    for label, colour_rgb, kind in entries:
        unit_colour = tuple(to_unit_rgb(colour_rgb))
        if kind == LEGEND_KIND_LINE:
            handles.append(Line2D([], [], color=unit_colour, linewidth=WIREFRAME_LINE_WIDTH, label=label))
        else:
            handles.append(Patch(facecolor=unit_colour, edgecolor=LEGEND_PATCH_EDGE, label=label))
    return handles


def compose_scene_figure(image: np.ndarray, title: str, legend_entries: Sequence[Tuple[str, Sequence[int], str]],
                         settings: FigureSettings, *, panel_count: int = 1, run_note: Optional[str] = None):
    """Place a 3D screenshot in a matplotlib figure with a short title, a colour legend, footer and run note.

    The image keeps its aspect: the figure is ``panel_count * SCENE_PANEL_WIDTH_IN`` inches wide and the image
    axes fills the width.

    Args:
        image: (H, W, 3) uint8 screenshot.
        title: Short figure title.
        legend_entries: ``(label, 0-255 RGB, kind)`` legend entries (kind LEGEND_KIND_PATCH or LEGEND_KIND_LINE).
        settings: Figure settings (footer note).
        panel_count: Number of side-by-side panels in the screenshot (sets the figure width).
        run_note: Optional text naming the drawn runs.

    Returns:
        A matplotlib ``Figure`` (Agg backend).
    """
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    from viking_splatad.viz import figures_mpl

    # Step 1: geometry in inches: image height follows the screenshot aspect; the legend takes as many columns
    # as fit the width, and the bottom band holds its rows (ceil(entries / columns)).
    width_in = SCENE_PANEL_WIDTH_IN * panel_count
    image_height_in = width_in * image.shape[0] / image.shape[1]
    legend_columns = max(1, min(len(legend_entries), int(width_in // SCENE_LEGEND_ENTRY_WIDTH_IN)))
    legend_rows = -(-len(legend_entries) // legend_columns)  # ceiling division
    band = figures_mpl.band_layout(width_in, run_note, settings.footer_note, legend_rows * SCENE_LEGEND_ROW_HEIGHT_IN)
    height_in = image_height_in + SCENE_TITLE_BAND_IN + band["total_in"]
    figure = Figure(figsize=(width_in, height_in), dpi=settings.dpi)
    # Step 2: image axes between the title band and the bottom band.
    image_axes = figure.add_axes([0.0, band["total_in"] / height_in, 1.0, image_height_in / height_in])
    image_axes.imshow(image)
    image_axes.set_axis_off()
    figure.suptitle(title, y=1.0, va="top", fontsize=SCENE_FONT_SIZE)
    # Step 3: texts and the colour legend above the footer.
    legend_bottom = figures_mpl.place_band_texts(figure, band, height_in, settings.footer_note)
    figure.legend(handles=_legend_handles(legend_entries), loc="lower center",
                  bbox_to_anchor=(figures_mpl.FOOTER_X_FRACTION, legend_bottom), ncol=legend_columns,
                  fontsize=SCENE_FONT_SIZE, frameon=False)
    return figure


def compose_tile_grid(tile_images: Sequence[Sequence[Optional[np.ndarray]]], row_labels: Sequence[str],
                      column_labels: Sequence[str], caption_by_tile: Sequence[Sequence[str]],
                      settings: FigureSettings, *, title: Optional[str] = None, run_note: Optional[str] = None,
                      show_key: bool = False):
    """Lay tile images in a rows x arms grid with labels, per-tile captions and the footer note.

    With ``show_key`` a horizontal occupancy colourbar (``settings.occupancy_cmap``, 0-1) and a legend of the
    false-alarm, visible-GT and GT-box drawing styles are added at the bottom.

    Args:
        tile_images: ``tile_images[row][column]`` RGB image, or None for an empty cell.
        row_labels: One label per row (y-axis label of the first column).
        column_labels: One label per column (title of the first row; arm labels only).
        caption_by_tile: ``caption_by_tile[row][column]`` text drawn under the image.
        settings: Figure settings (tile_px, dpi, footer_note).
        title: Optional short figure title above the column headers.
        run_note: Optional text naming the drawn runs, printed wrapped under the footer.
        show_key: Whether to draw the colourbar and legend band.

    Returns:
        A matplotlib ``Figure`` (Agg backend) with rows x columns image axes (plus the colourbar axes
        when ``show_key``).
    """
    # Lazy import: matplotlib is needed only here, and Agg keeps it headless.
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure

    from viking_splatad.viz import figures_mpl

    row_count, column_count = len(tile_images), len(column_labels)
    # Step 1: figure size = tiles at tile_px pixels each at the settings' dpi, plus title and bottom bands.
    tile_inches = settings.tile_px / settings.dpi
    width_in = tile_inches * column_count
    band = figures_mpl.band_layout(width_in, run_note, settings.footer_note, TILE_KEY_BAND_IN if show_key else 0.0)
    title_in = TILE_TITLE_BAND_IN if title else 0.0
    grid_in = tile_inches * row_count / (1.0 - FOOTER_BOTTOM_MARGIN)
    height_in = grid_in + band["total_in"] + title_in
    figure = Figure(figsize=(width_in, height_in), dpi=settings.dpi)
    axes_grid = figure.subplots(row_count, column_count, squeeze=False)
    # Step 2: one image per axes, captions as x labels, row/column labels on the edges.
    for row_index in range(row_count):
        for column_index in range(column_count):
            axes = axes_grid[row_index][column_index]
            image = tile_images[row_index][column_index]
            if image is not None:
                axes.imshow(image)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_xlabel(caption_by_tile[row_index][column_index], fontsize=GRID_FONT_SIZE)
            if column_index == 0:
                axes.set_ylabel(row_labels[row_index], fontsize=GRID_FONT_SIZE)
            if row_index == 0:
                axes.set_title(column_labels[column_index], fontsize=GRID_FONT_SIZE)
    # Step 3: reserve the bands, then the texts (run note lowest, footer above, key above that).
    bottom_fraction = band["total_in"] / height_in
    figure.tight_layout(rect=(0.0, bottom_fraction, 1.0, 1.0 - title_in / height_in))
    if title:
        figure.suptitle(title, y=1.0, va="top", fontsize=GRID_TITLE_FONT_SIZE)
    key_bottom = figures_mpl.place_band_texts(figure, band, height_in, settings.footer_note)
    if show_key:
        # Step 4: colourbar (occupancy 0-1) on the left of the key band, drawing-style legend on the right.
        colourbar_axes = figure.add_axes([TILE_COLOURBAR_LEFT, key_bottom + band["key_in"] / height_in * KEY_BAND_MIDDLE_FRACTION,
                                          TILE_COLOURBAR_WIDTH, TILE_COLOURBAR_HEIGHT_IN / height_in])
        mappable = ScalarMappable(norm=Normalize(*OCCUPANCY_CLIM), cmap=settings.occupancy_cmap)
        colourbar = figure.colorbar(mappable, cax=colourbar_axes, orientation="horizontal")
        colourbar.set_label("occupancy of hidden GT cells", fontsize=GRID_FONT_SIZE)
        colourbar.ax.tick_params(labelsize=GRID_FONT_SIZE)
        key_entries = [("false alarm", settings.colours["false_alarm"], LEGEND_KIND_PATCH),
                       ("visible GT cell (context)", settings.colours["context"], LEGEND_KIND_PATCH),
                       ("GT box", CUBOID_WIREFRAME_RGB, LEGEND_KIND_LINE)]
        figure.legend(handles=_legend_handles(key_entries), loc="lower left",
                      bbox_to_anchor=(TILE_LEGEND_X_FRACTION, key_bottom), fontsize=GRID_FONT_SIZE, frameon=False)
    return figure
