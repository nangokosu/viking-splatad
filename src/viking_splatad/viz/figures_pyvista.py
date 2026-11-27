"""Off-screen PyVista renderers of the Gaussian-occupancy figure suite (the per-arm outcome panels).

PyVista is installed only in the isolated ``viz/`` project, so nothing here imports it at module level:
every renderer takes the imported module as ``pyvista_module`` (as ``pyvista_scene.draw_voxels`` does).
Renderers return RGB uint8 images and never write files. Voxel drawing reuses ``pyvista_scene``; the
oblique camera is fitted exactly by ``camera_framing`` (scripts/15 uses PyVista's default and ``view_xy`` instead).
``render_difference_panel`` draws the hit-difference panel alone, zoomed to its cells.
Matplotlib is imported lazily, and only inside ``compose_scene_figure`` and the legend helper.
"""

import itertools
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.labels.cuboids import compute_cuboid_geometry
from viking_splatad.viz.camera_framing import (  # noqa: F401  (the camera constants are re-exported for callers)
    CAMERA_VIEW_ANGLE_DEGREES,
    OBLIQUE_CAMERA_DIRECTION,
    VIEW_UP_VECTOR,
    tight_oblique_camera,
    tight_projected_aspect,
)
from viking_splatad.viz.figure_runs import FigureSettings
from viking_splatad.viz.pyvista_scene import (
    CORNER_SIGN_TRIPLES,
    MAX_CHANNEL_VALUE,
    SOLID_VOXEL_OPACITY,
    WIREFRAME_LINE_WIDTH,
    cuboid_wireframe_arrays,
    draw_voxels,
)

# Translucency (0-1, presentation choices): faint context, translucent false alarms and misses.
CONTEXT_OPACITY = 0.15
FALSE_ALARM_OPACITY = 0.5
MISSED_OPACITY = 0.35
# Camera distance multiplier of the scene figure (1 = the closest camera that keeps every drawn cell and GT box
# corner in view): 5 % of room (also covers the half-voxel of the outermost cubes) so no edge voxel touches the panel border.
SCENE_CAMERA_PADDING = 1.05
# Scene-figure grid: row 0 holds the ground-truth panel and the difference panel, row 1 one panel per arm.
SCENE_GRID_ROWS, SCENE_GRID_COLUMNS = 2, 3
# Grid column of the ground-truth panel and of the difference panel in row 0 (the middle cell stays empty).
GROUND_TRUTH_PANEL_COLUMN, DIFFERENCE_PANEL_COLUMN = 0, SCENE_GRID_COLUMNS - 1
# Row of the arm panels.
ARM_PANEL_ROW = 1
# Off-screen PyVista background (the library's default theme is white). A pixel is scene content when any of its
# channels differs from the background by more than the tolerance (0-255 levels): 1 level ignores nothing
# visible (the faintest context cell differs by about 7 levels) but absorbs 8-bit rounding.
BACKGROUND_RGB = (255, 255, 255)
CONTENT_TOLERANCE = 1
# Margin kept around the union content box, as a fraction of its height / width, and the blank title band above
# every grid row, as a fraction of the cropped panel height (the titles are drawn there by ``compose_scene_figure``).
CROP_MARGIN_FRACTION = 0.03
TITLE_BAND_FRACTION = 0.14
# The stand-alone difference panel has no panel title, so it keeps no title band (only the figure title above).
SINGLE_PANEL_TITLE_BAND_FRACTION = 0.0
# Colour name (settings.colours) of the cuboid edges.
CUBOID_EDGE_COLOUR_NAME = "cuboid_edge"
# Corners of one box face as sign pairs on the face's two free axes, in ring order (each consecutive pair, and the
# last with the first, shares an edge), so the four corners form a planar, non-self-crossing quad.
FACE_RING_SIGN_PAIRS = ((-1, -1), (1, -1), (1, 1), (-1, 1))
# VTK "faces" cell layout: a leading corner count (a quad has four corners) then the corner indices.
QUAD_CORNER_COUNT = 4
# Corners of an axis-aligned cube as (x, y, z) sign triples: every combination of -1 / +1 on the three axes (8 rows).
CUBE_CORNER_SIGNS = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
# Scene-figure layout (inches): width of one scene panel when the screenshot is placed in a matplotlib figure,
# title band, and font size (pt) of its legend and titles.
SCENE_PANEL_WIDTH_IN = 6.0
SCENE_TITLE_BAND_IN = 0.45
SCENE_FONT_SIZE = 11
# Left padding of a panel title, as a fraction of the cell width.
SCENE_TITLE_PAD_FRACTION = 0.01
# Width (inches) one legend entry needs at SCENE_FONT_SIZE (swatch plus a label such as "false_alarm: 1272"):
# legends use as many columns as fit the figure width, and wrap into more rows otherwise.
SCENE_LEGEND_ENTRY_WIDTH_IN = 2.0
# Height (inches) of one legend row at SCENE_FONT_SIZE.
SCENE_LEGEND_ROW_HEIGHT_IN = 0.3
# Legend entry kinds of ``compose_scene_figure``.
LEGEND_KIND_PATCH, LEGEND_KIND_LINE = "patch", "line"
# Settings colour names of the two difference-panel arms (hits of only the first / only the second arm).
FIRST_ARM_COLOUR_NAME = "only_radar_init_loss"
SECOND_ARM_COLOUR_NAME = "only_stock"


def to_unit_rgb(colour_rgb: Sequence[int]) -> np.ndarray:
    """Turn a 0-255 RGB triple into the 0-1 RGB PyVista takes.

    Args:
        colour_rgb: 0-255 RGB.

    Returns:
        (3,) float array in [0, 1].
    """
    return np.asarray(colour_rgb, dtype=float) / MAX_CHANNEL_VALUE


def capture_image(plotter) -> np.ndarray:
    """Screenshot an off-screen plotter as an RGB array.

    Args:
        plotter: ``pyvista.Plotter`` created with ``off_screen=True``.

    Returns:
        (H, W, 3) uint8 image.
    """
    return np.asarray(plotter.screenshot(return_img=True), dtype=np.uint8)


def cuboid_face_cells() -> np.ndarray:
    """VTK ``faces`` array of one box (six quads) over the corner order of ``compute_cuboid_geometry``.

    Corner k has the sign triple ``CORNER_SIGN_TRIPLES[k]``. The face on side s of axis a holds the four corners whose
    triple has s on axis a; they are listed in ring order over the two other axes (``FACE_RING_SIGN_PAIRS``).

    Returns:
        (6 * 5,) int array [4, a, b, c, d, 4, ...].
    """
    face_cells: List[int] = []
    for fixed_axis in range(CORNER_SIGN_TRIPLES.shape[1]):
        # Step 1: the two axes that vary on this face.
        free_axes = [axis for axis in range(CORNER_SIGN_TRIPLES.shape[1]) if axis != fixed_axis]
        for side in (-1, 1):
            face_cells.append(QUAD_CORNER_COUNT)
            for first_sign, second_sign in FACE_RING_SIGN_PAIRS:
                # Step 2: the corner whose sign triple is (side on the fixed axis, the ring pair on the free ones).
                target_triple = np.zeros(CORNER_SIGN_TRIPLES.shape[1], dtype=CORNER_SIGN_TRIPLES.dtype)
                target_triple[fixed_axis] = side
                target_triple[free_axes[0]], target_triple[free_axes[1]] = first_sign, second_sign
                face_cells.append(int(np.flatnonzero(np.all(CORNER_SIGN_TRIPLES == target_triple, axis=1))[0]))
    return np.asarray(face_cells, dtype=np.int64)


def draw_class_cuboids(plotter, pyvista_module, cuboids: Sequence[Any], colour_by_class_id: Mapping[int, Sequence[int]],
                       opacity: float, edge_rgb: Sequence[int]) -> None:
    """Draw GT boxes as translucent solids coloured by class, with dark edges (display only).

    Args:
        plotter: ``pyvista.Plotter`` with the panel's subplot active.
        pyvista_module: The imported ``pyvista``.
        cuboids: Boxes with ``class_id`` in the map frame; nothing is drawn when empty.
        colour_by_class_id: class id -> 0-255 RGB (every drawn class needs an entry).
        opacity: 0-1 opacity of the solid faces.
        edge_rgb: 0-255 RGB of the 12 edges of every box.
    """
    if not cuboids:
        return
    face_cells = cuboid_face_cells()
    for cuboid in cuboids:
        # Step 1: one solid mesh per box (its 8 map-frame corners, six quad faces) in its class colour.
        corners = compute_cuboid_geometry(cuboid).corners
        plotter.add_mesh(pyvista_module.PolyData(corners, faces=face_cells),
                         color=to_unit_rgb(colour_by_class_id[cuboid.class_id]), opacity=opacity)
    # Step 2: all edges as one line mesh (``pyvista_scene.cuboid_wireframe_arrays``).
    corner_points, line_cells = cuboid_wireframe_arrays(list(cuboids))
    plotter.add_mesh(pyvista_module.PolyData(corner_points, lines=line_cells), color=to_unit_rgb(edge_rgb),
                     line_width=WIREFRAME_LINE_WIDTH)


def scene_camera_points(centres: np.ndarray, outcomes_by_arm: Mapping[str, Mapping[str, np.ndarray]],
                        difference_sets: Mapping[str, np.ndarray], cuboids: Sequence[Any]) -> np.ndarray:
    """Points every scene panel must keep in view: the drawn cells' centres and the GT box corners.

    Args:
        centres: (V, 3) cell centres.
        outcomes_by_arm: arm -> outcome masks (scripts/26 ``scene_outcomes``).
        difference_sets: Difference masks (``arm_difference_sets``).
        cuboids: GT boxes drawn in the ground-truth panel.

    Returns:
        (N, 3) array (N = 0 when nothing is drawn).
    """
    # Step 1: a cell is drawn when it carries any outcome of any arm or any difference set.
    drawn_mask = np.zeros(centres.shape[0], dtype=bool)
    for outcomes in outcomes_by_arm.values():
        for outcome_mask in outcomes.values():
            drawn_mask |= np.asarray(outcome_mask, dtype=bool)
    for difference_mask in difference_sets.values():
        drawn_mask |= np.asarray(difference_mask, dtype=bool)
    # Step 2: add every corner of every GT box.
    point_blocks = [centres[drawn_mask]] + [compute_cuboid_geometry(cuboid).corners for cuboid in cuboids]
    return np.vstack(point_blocks)


def content_bounding_box(image: np.ndarray, background_rgb: Sequence[int],
                         tolerance: int) -> Optional[Tuple[int, int, int, int]]:
    """Bounding box of the pixels that differ from the background.

    A pixel is content when ``max over channels |pixel - background| > tolerance``.

    Args:
        image: (H, W, 3) uint8 image.
        background_rgb: 0-255 background colour.
        tolerance: Largest per-channel difference (0-255 levels) still counted as background.

    Returns:
        (row_start, row_stop, column_start, column_stop) with exclusive stops, or None when the whole image is
        background.
    """
    # Step 1: per-pixel largest channel difference (int16 avoids uint8 wrap-around).
    difference = np.abs(image.astype(np.int16) - np.asarray(background_rgb, dtype=np.int16)).max(axis=2)
    content_mask = difference > tolerance
    if not content_mask.any():
        return None
    # Step 2: first and last content row / column.
    content_rows = np.flatnonzero(content_mask.any(axis=1))
    content_columns = np.flatnonzero(content_mask.any(axis=0))
    return (int(content_rows[0]), int(content_rows[-1]) + 1, int(content_columns[0]), int(content_columns[-1]) + 1)


def crop_grid_to_content(image: np.ndarray, grid_shape: Tuple[int, int], drawn_cells: Sequence[Tuple[int, int]],
                         background_rgb: Sequence[int], tolerance: int, margin_fraction: float,
                         title_band_fraction: float) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Crop every panel of a screenshot grid to the union of the panels' content boxes and reassemble the grid.

    Each cell of the (rows, columns) screenshot is ``(H // rows, W // columns)`` pixels. The content box of every
    drawn cell is found (``content_bounding_box``) in cell-local coordinates and the union box over all of them is
    grown by ``margin_fraction`` of its size per side (kept inside the cell). Every cell is cropped by that same
    box, so the scale of all panels stays comparable. The new grid has a blank band of
    ``title_band_fraction`` x the cropped height above each row; cells not drawn stay background.

    Args:
        image: (H, W, 3) uint8 screenshot of the whole grid.
        grid_shape: (rows, columns).
        drawn_cells: (row, column) of the cells that hold a panel.
        background_rgb: 0-255 background colour.
        tolerance: Background tolerance of ``content_bounding_box``.
        margin_fraction: Margin around the union box, as a fraction of its height (rows) and width (columns).
        title_band_fraction: Title band height as a fraction of the cropped cell height.

    Returns:
        (new image, report). The report holds ``union_box`` (cell-local row/column start and stop), ``cell_shape``
        (cropped (height, width)), ``title_band_px`` and ``fill_by_cell`` (``"row,column"`` -> share of the cell's
        area that the cell's own content box covers, before and after the crop).
    """
    rows, columns = grid_shape
    cell_height, cell_width = image.shape[0] // rows, image.shape[1] // columns
    # Step 1: content box of every drawn cell, in cell-local pixels.
    boxes = {}
    for row_index, column_index in drawn_cells:
        cell = image[row_index * cell_height:(row_index + 1) * cell_height,
                     column_index * cell_width:(column_index + 1) * cell_width]
        box = content_bounding_box(cell, background_rgb, tolerance)
        if box is not None:
            boxes[(row_index, column_index)] = box
    # Step 2: union box (the whole cell when nothing is drawn), grown by the margin and clamped to the cell.
    if boxes:
        row_start = min(box[0] for box in boxes.values())
        row_stop = max(box[1] for box in boxes.values())
        column_start = min(box[2] for box in boxes.values())
        column_stop = max(box[3] for box in boxes.values())
        margin_rows = int(round(margin_fraction * (row_stop - row_start)))
        margin_columns = int(round(margin_fraction * (column_stop - column_start)))
        row_start, row_stop = max(0, row_start - margin_rows), min(cell_height, row_stop + margin_rows)
        column_start, column_stop = max(0, column_start - margin_columns), min(cell_width, column_stop + margin_columns)
    else:
        row_start, row_stop, column_start, column_stop = 0, cell_height, 0, cell_width
    cropped_height, cropped_width = row_stop - row_start, column_stop - column_start
    # Step 3: reassemble on a background canvas with a title band above every row.
    title_band = int(round(title_band_fraction * cropped_height))
    row_height = title_band + cropped_height
    canvas = np.empty((rows * row_height, columns * cropped_width, image.shape[2]), dtype=image.dtype)
    canvas[:] = np.asarray(background_rgb, dtype=image.dtype)
    for row_index, column_index in drawn_cells:
        source = image[row_index * cell_height + row_start:row_index * cell_height + row_stop,
                       column_index * cell_width + column_start:column_index * cell_width + column_stop]
        top = row_index * row_height + title_band
        canvas[top:top + cropped_height, column_index * cropped_width:(column_index + 1) * cropped_width] = source
    # Step 4: how much of its cell / crop each panel's own content covers.
    fill_by_cell = {f"{row_index},{column_index}": {
        "before_crop": (box[1] - box[0]) * (box[3] - box[2]) / (cell_height * cell_width),
        "after_crop": (box[1] - box[0]) * (box[3] - box[2]) / (cropped_height * cropped_width)}
        for (row_index, column_index), box in boxes.items()}
    return canvas, {"union_box": [row_start, row_stop, column_start, column_stop],
                    "cell_shape": [cropped_height, cropped_width], "title_band_px": title_band,
                    "fill_by_cell": fill_by_cell}


def scene_panel_titles(arm_keys: Sequence[str], title_by_panel: Sequence[str],
                       gt_title: str) -> Dict[Tuple[int, int], str]:
    """Titles of the scene grid by (row, column): ground truth, difference, then one per arm.

    Args:
        arm_keys: Arms drawn, in panel order.
        title_by_panel: One title per arm (in ``arm_keys`` order), then the difference panel's.
        gt_title: Title of the ground-truth panel.

    Returns:
        (row, column) -> title.
    """
    titles = {(0, GROUND_TRUTH_PANEL_COLUMN): gt_title, (0, DIFFERENCE_PANEL_COLUMN): title_by_panel[len(arm_keys)]}
    titles.update({(ARM_PANEL_ROW, column_index): title_by_panel[column_index]
                   for column_index in range(len(arm_keys))})
    return titles


def scene_camera_aspect(points: np.ndarray) -> float:
    """Width / height a panel needs to frame the points from the scene's oblique direction.

    Args:
        points: (N, 3) points from ``scene_camera_points`` (N >= 1).

    Returns:
        ``camera_framing.tight_projected_aspect`` of the points.
    """
    return tight_projected_aspect(points, OBLIQUE_CAMERA_DIRECTION, VIEW_UP_VECTOR)


def draw_difference_cells(plotter, pyvista_module, centres: np.ndarray, voxel_size: float,
                          difference_sets: Mapping[str, np.ndarray], colours: Mapping[str, Sequence[int]],
                          first_colour_name: str = FIRST_ARM_COLOUR_NAME,
                          second_colour_name: str = SECOND_ARM_COLOUR_NAME) -> None:
    """Draw the hit-difference cells into the active subplot (shared by the grid and the stand-alone panel).

    Args:
        plotter: ``pyvista.Plotter`` whose active subplot receives the cells.
        pyvista_module: The imported ``pyvista``.
        centres: (V, 3) cell centres.
        voxel_size: Voxel edge (m).
        difference_sets: {"only_first","only_second","both"} (V,) bool (``arm_difference_sets``).
        colours: Colour name -> 0-255 RGB (``FigureSettings.colours``).
        first_colour_name: Colour name of the hits only the first compared arm has.
        second_colour_name: Colour name of the hits only the second compared arm has.
    """
    # Solid cubes, "both" first so the single-arm hits sit on top.
    for set_name, colour_name in (("both", "both"), ("only_first", first_colour_name),
                                  ("only_second", second_colour_name)):
        draw_voxels(plotter, pyvista_module, centres[np.asarray(difference_sets[set_name], dtype=bool)],
                    voxel_size, colours[colour_name], SOLID_VOXEL_OPACITY)


def difference_camera_points(centres: np.ndarray, difference_sets: Mapping[str, np.ndarray]) -> np.ndarray:
    """Centres of the cells in any difference mask: the points a difference-only camera must keep in view.

    Args:
        centres: (V, 3) cell centres.
        difference_sets: Difference masks (``arm_difference_sets``).

    Returns:
        (N, 3) array (N = 0 when no difference cell exists); no GT box corners and no other outcomes.
    """
    # Union of the difference masks.
    difference_mask = np.zeros(centres.shape[0], dtype=bool)
    for set_mask in difference_sets.values():
        difference_mask |= np.asarray(set_mask, dtype=bool)
    return centres[difference_mask]


def cell_corner_points(cell_centres: np.ndarray, voxel_size: float) -> np.ndarray:
    """The 8 corners of every cubic cell: centre + (±voxel_size / 2) on each axis.

    Framing corners instead of centres keeps the outermost cubes whole, and gives any non-empty set of cells a
    projected extent of at least one voxel on both screen axes, so the framing aspect is finite and > 0 even
    for one cell or a line of cells.

    Args:
        cell_centres: (N, 3) cell centres.
        voxel_size: Voxel edge (m).

    Returns:
        (8 N, 3) corner points (empty for N = 0).
    """
    # Broadcast (N, 1, 3) centres against (1, 8, 3) half-edge offsets, then flatten to (8 N, 3).
    half_edge_offsets = CUBE_CORNER_SIGNS * (voxel_size / 2.0)
    return (cell_centres[:, None, :] + half_edge_offsets[None, :, :]).reshape(-1, CUBE_CORNER_SIGNS.shape[1])


def difference_frame_points(centres: np.ndarray, difference_sets: Mapping[str, np.ndarray],
                            voxel_size: float) -> np.ndarray:
    """Corners of the difference cells: what the stand-alone difference panel frames (camera and window aspect).

    Args:
        centres: (V, 3) cell centres.
        difference_sets: Difference masks (``arm_difference_sets``).
        voxel_size: Voxel edge (m).

    Returns:
        (8 N, 3) corner points of the N difference cells (empty when there is none).
    """
    return cell_corner_points(difference_camera_points(centres, difference_sets), voxel_size)


def render_arm_panels(pyvista_module, centres: np.ndarray, voxel_size: float,
                      outcomes_by_arm: Mapping[str, Mapping[str, np.ndarray]],
                      difference_sets: Mapping[str, np.ndarray], settings: FigureSettings, *,
                      window_size: Tuple[int, int],
                      first_colour_name: str = FIRST_ARM_COLOUR_NAME,
                      second_colour_name: str = SECOND_ARM_COLOUR_NAME,
                      gt_cuboids: Sequence[Any] = (), gt_colour_by_class_id: Optional[Mapping[int, Sequence[int]]] = None,
                      layout_report: Optional[Dict[str, Any]] = None) -> np.ndarray:
    """A (2, 3) grid on one linked camera: ground truth and hit difference above, one panel per arm below.

    Row 0 holds the GT panel (column 0: faint context and the GT boxes, ``draw_class_cuboids``) and the difference
    panel (column 2; the middle cell stays empty). Row 1 holds the arms in ``settings.arm_order`` (only arms in
    ``outcomes_by_arm``; context, misses, false alarms, hits). The difference panel colours hits of only the
    first arm (``first_colour_name``), only the second (``second_colour_name``) and both ("both"). Every panel
    gets the same camera, fitted exactly to the drawn cells and box corners (``tight_oblique_camera``) for the
    panel aspect ``(window width / columns) / (window height / rows)``. The screenshot is then cropped so the
    panels lose their empty margins (``crop_grid_to_content``); panel titles are not drawn here but by
    ``compose_scene_figure`` (``scene_panel_titles`` gives their cells).

    Args:
        pyvista_module: The imported ``pyvista``.
        centres: (V, 3) cell centres.
        voxel_size: Voxel edge (m).
        outcomes_by_arm: arm -> {"context","hit","false_alarm","missed"} (V,) bool (scripts/26 ``scene_outcomes``).
        difference_sets: {"only_first","only_second","both"} (V,) bool (``arm_difference_sets``).
        settings: Figure settings (arm order, colours, box opacity).
        window_size: (width, height) of the whole window in pixels.
        first_colour_name: Settings colour name of the hits only the first compared arm has.
        second_colour_name: Settings colour name of the hits only the second compared arm has.
        gt_cuboids: GT boxes of the ground-truth panel.
        gt_colour_by_class_id: class id -> 0-255 RGB of the boxes (required when ``gt_cuboids`` is not empty).
        layout_report: Optional dict that receives ``crop_grid_to_content``'s report.

    Returns:
        (H, W, 3) uint8 image of the cropped grid with a blank title band above each row.

    Raises:
        ValueError: When there are more arms than grid columns, or boxes without a colour map.
    """
    arm_keys = [arm for arm in settings.arm_order if arm in outcomes_by_arm]
    if len(arm_keys) > SCENE_GRID_COLUMNS:
        raise ValueError(f"{len(arm_keys)} arms do not fit the {SCENE_GRID_COLUMNS}-column grid")
    if gt_cuboids and gt_colour_by_class_id is None:
        raise ValueError("gt_colour_by_class_id is required when gt_cuboids is given")
    drawn_outcomes = {arm: outcomes_by_arm[arm] for arm in arm_keys}
    # Step 1: camera fitted to everything any panel draws, for the panel aspect the window gives.
    camera_points = scene_camera_points(centres, drawn_outcomes, difference_sets, gt_cuboids)
    panel_aspect = (window_size[0] / SCENE_GRID_COLUMNS) / (window_size[1] / SCENE_GRID_ROWS)
    camera_position = (tight_oblique_camera(camera_points, panel_aspect, SCENE_CAMERA_PADDING,
                                            OBLIQUE_CAMERA_DIRECTION, VIEW_UP_VECTOR, CAMERA_VIEW_ANGLE_DEGREES)
                       if camera_points.shape[0] else None)
    plotter = pyvista_module.Plotter(shape=(SCENE_GRID_ROWS, SCENE_GRID_COLUMNS), off_screen=True,
                                     window_size=window_size, border=False)
    # Plain background so the content crop's background assumption (BACKGROUND_RGB) always holds.
    plotter.set_background(to_unit_rgb(BACKGROUND_RGB))
    drawn_panels = [(0, GROUND_TRUTH_PANEL_COLUMN), (0, DIFFERENCE_PANEL_COLUMN)]
    try:
        # Step 2: ground-truth panel: faint context of the first arm, then the class-coloured boxes.
        plotter.subplot(0, GROUND_TRUTH_PANEL_COLUMN)
        if arm_keys:
            draw_voxels(plotter, pyvista_module, centres[np.asarray(drawn_outcomes[arm_keys[0]]["context"], dtype=bool)],
                        voxel_size, settings.colours["context"], CONTEXT_OPACITY)
        draw_class_cuboids(plotter, pyvista_module, gt_cuboids, gt_colour_by_class_id or {}, settings.cuboid_opacity,
                           settings.colours[CUBOID_EDGE_COLOUR_NAME] if gt_cuboids else ())
        # Step 3: difference panel.
        plotter.subplot(0, DIFFERENCE_PANEL_COLUMN)
        draw_difference_cells(plotter, pyvista_module, centres, voxel_size, difference_sets, settings.colours,
                              first_colour_name, second_colour_name)
        # Step 4: arm panels (bottom row), faint context first so the solid hits sit on top.
        for column_index, arm_key in enumerate(arm_keys):
            plotter.subplot(ARM_PANEL_ROW, column_index)
            drawn_panels.append((ARM_PANEL_ROW, column_index))
            outcomes = outcomes_by_arm[arm_key]
            for outcome_name, colour_name, opacity in (("context", "context", CONTEXT_OPACITY),
                                                       ("missed", "missed", MISSED_OPACITY),
                                                       ("false_alarm", "false_alarm", FALSE_ALARM_OPACITY),
                                                       ("hit", "hit", SOLID_VOXEL_OPACITY)):
                draw_voxels(plotter, pyvista_module, centres[np.asarray(outcomes[outcome_name], dtype=bool)],
                            voxel_size, settings.colours[colour_name], opacity)
        # Step 5: the same camera and depth peeling in every drawn panel, then link the views.
        for row_index, column_index in drawn_panels:
            plotter.subplot(row_index, column_index)
            if camera_position is not None:
                plotter.camera_position = camera_position
            plotter.enable_depth_peeling()
        plotter.link_views()
        screenshot = capture_image(plotter)
    finally:
        plotter.close()
    # Step 6: crop every panel to the union content box and reassemble (titles are added by compose_scene_figure).
    cropped, report = crop_grid_to_content(screenshot, (SCENE_GRID_ROWS, SCENE_GRID_COLUMNS), drawn_panels,
                                           BACKGROUND_RGB, CONTENT_TOLERANCE, CROP_MARGIN_FRACTION,
                                           TITLE_BAND_FRACTION)
    if layout_report is not None:
        layout_report.update(report)
    return cropped


def render_difference_panel(pyvista_module, centres: np.ndarray, voxel_size: float,
                            difference_sets: Mapping[str, np.ndarray], colours: Mapping[str, Sequence[int]],
                            window_size: Tuple[int, int], *,
                            first_colour_name: str = FIRST_ARM_COLOUR_NAME,
                            second_colour_name: str = SECOND_ARM_COLOUR_NAME) -> Tuple[np.ndarray, Dict[str, Any]]:
    """The hit-difference panel alone, zoomed so the difference cells fill the frame.

    One off-screen plotter (no grid) of ``window_size``; the cells are drawn by ``draw_difference_cells`` and the
    camera is fitted exactly to the difference cells' cube corners only (``difference_frame_points``) for the
    window aspect ``width / height``. The screenshot is cropped to its content (``crop_grid_to_content`` on a (1, 1) grid) with
    no per-panel title band: the lone panel has no panel title, ``compose_scene_figure`` puts the figure title above.

    Args:
        pyvista_module: The imported ``pyvista``.
        centres: (V, 3) cell centres.
        voxel_size: Voxel edge (m).
        difference_sets: {"only_first","only_second","both"} (V,) bool (``arm_difference_sets``).
        colours: Colour name -> 0-255 RGB (``FigureSettings.colours``).
        window_size: (width, height) of the window in pixels.
        first_colour_name: Colour name of the hits only the first compared arm has.
        second_colour_name: Colour name of the hits only the second compared arm has.

    Returns:
        (cropped (H, W, 3) uint8 image, ``crop_grid_to_content`` report).

    Raises:
        ValueError: When no cell is in any difference mask (nothing to frame).
    """
    # Step 1: camera fitted to the difference cells' cube corners for the window aspect (width / height).
    camera_points = difference_frame_points(centres, difference_sets, voxel_size)
    if camera_points.shape[0] == 0:
        raise ValueError("no difference cell to frame")
    window_aspect = window_size[0] / window_size[1]
    camera_position = tight_oblique_camera(camera_points, window_aspect, SCENE_CAMERA_PADDING,
                                           OBLIQUE_CAMERA_DIRECTION, VIEW_UP_VECTOR, CAMERA_VIEW_ANGLE_DEGREES)
    plotter = pyvista_module.Plotter(off_screen=True, window_size=window_size, border=False)
    # Plain background so the content crop's background assumption (BACKGROUND_RGB) always holds.
    plotter.set_background(to_unit_rgb(BACKGROUND_RGB))
    try:
        # Step 2: draw, then camera and depth peeling as in the grid panels.
        draw_difference_cells(plotter, pyvista_module, centres, voxel_size, difference_sets, colours,
                              first_colour_name, second_colour_name)
        plotter.camera_position = camera_position
        plotter.enable_depth_peeling()
        screenshot = capture_image(plotter)
    finally:
        plotter.close()
    # Step 3: crop to content, with no title band (the figure title sits above the image).
    return crop_grid_to_content(screenshot, (1, 1), [(0, 0)], BACKGROUND_RGB, CONTENT_TOLERANCE,
                                CROP_MARGIN_FRACTION, SINGLE_PANEL_TITLE_BAND_FRACTION)


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
                         settings: FigureSettings, *, panel_count: int = 1, run_note: Optional[str] = None,
                         panel_titles: Optional[Mapping[Tuple[int, int], str]] = None, grid_rows: int = 1):
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
        panel_titles: Optional (row, column) -> title, drawn at the top-left of that grid cell (the screenshot's
            blank title band, see ``crop_grid_to_content``).
        grid_rows: Rows of the screenshot grid (``panel_count`` is its column count).

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
    # Titles sit at the top-left of their cell, in image pixel coordinates (rows grow downward in imshow).
    cell_width_px, row_height_px = image.shape[1] / panel_count, image.shape[0] / grid_rows
    for (row_index, column_index), panel_title in (panel_titles or {}).items():
        image_axes.text(column_index * cell_width_px + SCENE_TITLE_PAD_FRACTION * cell_width_px,
                        row_index * row_height_px, panel_title, ha="left", va="top", fontsize=SCENE_FONT_SIZE)
    figure.suptitle(title, y=1.0, va="top", fontsize=SCENE_FONT_SIZE)
    # Step 3: texts and the colour legend above the footer.
    legend_bottom = figures_mpl.place_band_texts(figure, band, height_in, settings.footer_note)
    figure.legend(handles=_legend_handles(legend_entries), loc="lower center",
                  bbox_to_anchor=(figures_mpl.FOOTER_X_FRACTION, legend_bottom), ncol=legend_columns,
                  fontsize=SCENE_FONT_SIZE, frameon=False)
    return figure
