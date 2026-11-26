"""Tests of the off-screen figure renderers (viz/figures_pyvista.py) against a recording stand-in for pyvista."""

import dataclasses
import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from viking_splatad.viz.figure_runs import FigureSettings
from viking_splatad.viz.pyvista_scene import MAX_CHANNEL_VALUE
from viking_splatad.viz.camera_framing import (
    CAMERA_VIEW_ANGLE_DEGREES,
    OBLIQUE_CAMERA_DIRECTION,
    VIEW_UP_VECTOR,
    tight_oblique_camera,
)
from viking_splatad.viz.figures_pyvista import (
    DIFFERENCE_PANEL_COLUMN,
    GROUND_TRUTH_PANEL_COLUMN,
    SCENE_CAMERA_PADDING,
    SCENE_GRID_COLUMNS,
    SCENE_GRID_ROWS,
    cuboid_face_cells,
    cell_corner_points,
    difference_camera_points,
    difference_frame_points,
    draw_class_cuboids,
    render_difference_panel,
    scene_camera_aspect,
    render_arm_panels,
    BACKGROUND_RGB,
    content_bounding_box,
    crop_grid_to_content,
    scene_panel_titles,
    scene_camera_points,
)
from viking_splatad.labels.cuboids import Cuboid, compute_cuboid_geometry

VOXEL_EDGE_METERS = 0.5
WINDOW_SIZE = (120, 60)
STUB_IMAGE_SIDE = 4
CHANNELS = 3
COLOURS = {"hit": (10, 160, 10), "false_alarm": (200, 0, 0), "missed": (150, 150, 150), "context": (210, 210, 210),
           "only_radar_init_loss": (30, 100, 180), "only_stock": (250, 120, 10), "both": (40, 160, 70),
           "radar_seed": (140, 100, 190)}
TILE_PX, DPI = 20, 10
FOOTER = "footer text"
SETTINGS = FigureSettings(
    arm_order=("stock", "radar_init_loss"), arm_labels={"stock": "Stock", "radar_init_loss": "Init+loss"},
    campaign_profiles={}, dpi=DPI,
    tile_px=TILE_PX, colours=COLOURS,
    footer_note=FOOTER, representative_metric="recall_H")


class RecordingPolyData:
    """Stand-in for ``pyvista.PolyData``: remembers points, scalars and glyph arguments."""

    def __init__(self, points, lines=None, faces=None):
        """Store points (and line or face cells)."""
        self.points = np.asarray(points)
        self.lines = lines
        self.faces = faces
        self.point_data = {}

    def __setitem__(self, name, values):
        """Record a point scalar array."""
        self.point_data[name] = np.asarray(values)

    def glyph(self, **glyph_kwargs):
        """Return self (the cubes keep the scalars)."""
        return self


class RecordingPlotter:
    """Stand-in for ``pyvista.Plotter`` that records the calls the renderers use."""

    instances = []

    def __init__(self, **kwargs):
        """Record constructor arguments and register the instance."""
        self.kwargs = kwargs
        self.meshes = []
        self.texts = []
        self.subplots = []
        self.linked = False
        self.closed = False
        self.camera_position = None
        self.background = None
        RecordingPlotter.instances.append(self)

    def add_mesh(self, mesh, **mesh_kwargs):
        """Record one mesh."""
        self.meshes.append((mesh, mesh_kwargs))

    def add_text(self, text, **text_kwargs):
        """Record one text."""
        self.texts.append(text)

    def subplot(self, row, column):
        """Record a subplot activation."""
        self.subplots.append((row, column))

    def set_background(self, colour):
        """Record the background colour."""
        self.background = np.asarray(colour)

    def link_views(self):
        """Record that views were linked."""
        self.linked = True

    def enable_depth_peeling(self):
        """No-op."""

    def screenshot(self, return_img=False):
        """Return a dummy uint8 image."""
        return np.zeros((STUB_IMAGE_SIDE, STUB_IMAGE_SIDE, CHANNELS), dtype=np.uint8)

    def close(self):
        """Record the close."""
        self.closed = True


class FakePyvista:
    """Stand-in for the ``pyvista`` module."""

    PolyData = RecordingPolyData
    Plotter = RecordingPlotter

    @staticmethod
    def Cube(**cube_kwargs):
        """Return the cube arguments as a namespace."""
        return SimpleNamespace(**cube_kwargs)


def last_plotter():
    """The most recently created fake plotter."""
    return RecordingPlotter.instances[-1]


def unit_rgb(colour):
    """Expected PyVista colour of a 0-255 triple, computed here."""
    return np.array(colour) / MAX_CHANNEL_VALUE


def mesh_by_colour(plotter, colour):
    """Meshes drawn in a given 0-255 colour."""
    return [mesh for mesh, kwargs in plotter.meshes if "color" in kwargs
            and np.allclose(kwargs["color"], unit_rgb(colour))]


def test_arm_panels_count_linked_views_and_difference_colours():
    """Arms + 1 panels, linked views, difference panel in blue / orange / green."""
    cell_count = 5
    centres = np.arange(cell_count * 3, dtype=float).reshape(cell_count, 3)

    def outcomes(hit_mask):
        """Outcome dict with the given hits and nothing else."""
        none = np.zeros(cell_count, dtype=bool)
        return {"context": np.ones(cell_count, dtype=bool), "hit": hit_mask, "false_alarm": none, "missed": none}

    hit_stock = np.array([True, True, False, False, False])
    hit_init = np.array([False, True, True, True, False])
    difference = {"only_first": hit_init & ~hit_stock, "only_second": hit_stock & ~hit_init,
                  "both": hit_init & hit_stock}
    titles = ["stock", "init", "diff"]
    render_arm_panels(FakePyvista, centres, VOXEL_EDGE_METERS, {"stock": outcomes(hit_stock),
                      "radar_init_loss": outcomes(hit_init)}, difference, SETTINGS, window_size=WINDOW_SIZE)
    plotter = last_plotter()
    assert plotter.kwargs["shape"] == (SCENE_GRID_ROWS, SCENE_GRID_COLUMNS)
    assert plotter.linked and plotter.closed
    # The crop assumes BACKGROUND_RGB, so the plotter background is set to exactly that colour (0-1 units).
    assert plotter.background is not None and np.allclose(plotter.background, unit_rgb(BACKGROUND_RGB))
    # Titles are drawn by compose_scene_figure now, not by PyVista (the content crop would include them).
    assert plotter.texts == []
    assert scene_panel_titles(["stock", "radar_init_loss"], titles, "ground truth") == {
        (0, GROUND_TRUTH_PANEL_COLUMN): "ground truth", (0, DIFFERENCE_PANEL_COLUMN): "diff", (1, 0): "stock",
        (1, 1): "init"}
    # Difference colours: counts computed from the masks.
    assert mesh_by_colour(plotter, COLOURS["only_radar_init_loss"])[0].points.shape[0] == int(difference["only_first"].sum())
    assert mesh_by_colour(plotter, COLOURS["only_stock"])[0].points.shape[0] == int(difference["only_second"].sum())
    assert plotter.subplots[:2] == [(0, GROUND_TRUTH_PANEL_COLUMN), (0, DIFFERENCE_PANEL_COLUMN)]


def test_compose_scene_figure_has_image_legend_footer_and_note():
    """One image axes, a legend with one entry per input, the footer and the wrapped run note."""
    from viking_splatad.viz.figures_pyvista import compose_scene_figure
    image = np.zeros((STUB_IMAGE_SIDE, STUB_IMAGE_SIDE, CHANNELS), dtype=np.uint8)
    entries = [("hit: 3", COLOURS["hit"], "patch"), ("box", COLOURS["missed"], "line")]
    figure = compose_scene_figure(image, "short title", entries, SETTINGS, run_note="drawn run: abc")
    assert len(figure.axes) == 1 and len(figure.axes[0].images) == 1
    assert [text.get_text() for text in figure.legends[0].get_texts()] == ["hit: 3", "box"]
    texts = [text.get_text() for text in figure.texts]
    assert FOOTER in texts and "drawn run: abc" in texts


# ---- GT boxes, grid layout, tight camera (synthetic known-answer inputs) ----
TRUNK_RGB, ROCK_RGB, EDGE_RGB = (140, 90, 30), (240, 228, 66), (40, 40, 40)
BOX_OPACITY = 0.35
CORNERS_PER_BOX = 8


def make_box(box_id, class_id, class_name, centre, size=(1.0, 2.0, 3.0)):
    """Axis-aligned unit-quaternion Cuboid at ``centre``."""
    return Cuboid(box_id=box_id, class_id=class_id, class_name=class_name, center_xyz=tuple(centre),
                  dimensions_xyz=tuple(size), quaternion_xyzw=(0.0, 0.0, 0.0, 1.0))


def test_cuboid_face_cells_are_six_quads_over_the_eight_corners():
    """Six quads of four distinct corners; each corner is on exactly three faces; quads are planar."""
    face_cells = cuboid_face_cells().reshape(6, 5)
    assert (face_cells[:, 0] == 4).all()
    corner_indices = face_cells[:, 1:]
    assert np.bincount(corner_indices.ravel(), minlength=CORNERS_PER_BOX).tolist() == [3] * CORNERS_PER_BOX
    corners = compute_cuboid_geometry(make_box(1, 1, "tree_trunk", (0, 0, 0))).corners
    for quad in corner_indices:
        assert len(set(quad.tolist())) == 4
        # Planar: one coordinate axis is constant over the four corners of a box face (axis-aligned box).
        assert (np.ptp(corners[quad], axis=0) == 0).sum() == 1
        # Ring order: consecutive corners are one box edge apart (differ along exactly one axis).
        for position in range(4):
            step = corners[quad[position]] - corners[quad[(position + 1) % 4]]
            assert np.count_nonzero(step) == 1


def test_draw_class_cuboids_one_coloured_mesh_per_box_and_edges():
    """One translucent solid per box in its class colour, plus one dark edge mesh with 12 edges per box."""
    boxes = [make_box(1, 1, "tree_trunk", (0, 0, 0)), make_box(2, 3, "rock", (10, 0, 0)),
             make_box(3, 1, "tree_trunk", (20, 0, 0))]
    plotter = RecordingPlotter()
    draw_class_cuboids(plotter, FakePyvista, boxes, {1: TRUNK_RGB, 3: ROCK_RGB}, BOX_OPACITY, EDGE_RGB)
    solids = [(mesh, kwargs) for mesh, kwargs in plotter.meshes if mesh.faces is not None]
    assert len(solids) == len(boxes)
    for (mesh, kwargs), box in zip(solids, boxes):
        expected_colour = TRUNK_RGB if box.class_id == 1 else ROCK_RGB
        np.testing.assert_allclose(kwargs["color"], unit_rgb(expected_colour))
        assert kwargs["opacity"] == BOX_OPACITY
        np.testing.assert_allclose(mesh.points, compute_cuboid_geometry(box).corners)
    edge_meshes = mesh_by_colour(plotter, EDGE_RGB)
    assert len(edge_meshes) == 1
    # VTK lines layout [2, a, b] per edge: 3 values x 12 edges x boxes.
    assert edge_meshes[0].lines.size == 3 * 12 * len(boxes)


def test_draw_class_cuboids_draws_nothing_without_boxes():
    """No box, no mesh."""
    plotter = RecordingPlotter()
    draw_class_cuboids(plotter, FakePyvista, [], {}, BOX_OPACITY, EDGE_RGB)
    assert plotter.meshes == []


class CameraRecordingPlotter(RecordingPlotter):
    """Plotter that records the camera set in each subplot."""

    def __init__(self, **kwargs):
        """Start with no active subplot and no cameras."""
        super().__init__(**kwargs)
        self.cameras = {}

    def subplot(self, row, column):
        """Remember the active subplot."""
        super().subplot(row, column)
        self.active = (row, column)

    @property
    def camera_position(self):
        """Camera of the active subplot."""
        return self.cameras.get(self.active)

    @camera_position.setter
    def camera_position(self, value):
        """Record the camera set in the active subplot."""
        if hasattr(self, "active"):
            self.cameras[self.active] = value


class CameraFakePyvista(FakePyvista):
    """Fake pyvista whose plotter records one camera per subplot."""

    Plotter = CameraRecordingPlotter


def test_scene_grid_gt_panel_first_and_every_panel_shares_the_fitted_camera():
    """(2, 3) grid, GT panel first, and every drawn panel has the camera fitted to cells + box corners."""
    centres = np.array([[0.0, 0.0, 0.0], [30.0, 5.0, 1.0], [10.0, -4.0, 0.5]])
    none = np.zeros(3, dtype=bool)
    outcomes = {"context": np.ones(3, dtype=bool), "hit": np.array([True, False, False]), "false_alarm": none,
                "missed": none}
    difference = {"only_first": none, "only_second": np.array([True, False, False]), "both": none}
    boxes = [make_box(1, 1, "tree_trunk", (40.0, 8.0, 1.0))]
    window_size = (900, 300)
    box_settings = dataclasses.replace(SETTINGS, colours={**COLOURS, "cuboid_edge": EDGE_RGB})
    render_arm_panels(CameraFakePyvista, centres, VOXEL_EDGE_METERS, {"stock": outcomes, "radar_init_loss": outcomes},
                      difference, box_settings, window_size=window_size,
                      gt_cuboids=boxes, gt_colour_by_class_id={1: TRUNK_RGB})
    plotter = last_plotter()
    assert plotter.kwargs["shape"] == (2, 3)
    assert plotter.subplots[0] == (0, GROUND_TRUTH_PANEL_COLUMN)
    # Expected camera, computed from the union of every cell centre drawn and the box corners.
    union = np.vstack([centres, compute_cuboid_geometry(boxes[0]).corners])
    panel_aspect = (window_size[0] / 3) / (window_size[1] / 2)
    expected = tight_oblique_camera(union, panel_aspect, SCENE_CAMERA_PADDING,
                                    OBLIQUE_CAMERA_DIRECTION, VIEW_UP_VECTOR, CAMERA_VIEW_ANGLE_DEGREES)
    assert set(plotter.cameras) == {(0, 0), (0, 2), (1, 0), (1, 1)}
    for camera in plotter.cameras.values():
        np.testing.assert_allclose(np.asarray(camera[0]), np.asarray(expected[0]))
        np.testing.assert_allclose(np.asarray(camera[1]), np.asarray(expected[1]))
    # The GT box is drawn in the GT panel's subplot (the first subplot activated).
    assert mesh_by_colour(plotter, TRUNK_RGB)


def test_scene_camera_points_union_of_drawn_cells_and_box_corners():
    """Only cells with some outcome or difference flag count; every box corner is added."""
    centres = np.arange(12, dtype=float).reshape(4, 3)
    none = np.zeros(4, dtype=bool)
    outcomes = {"stock": {"context": np.array([True, False, False, False]), "hit": none, "false_alarm": none,
                          "missed": none}}
    difference = {"both": np.array([False, False, True, False])}
    box = make_box(1, 1, "tree_trunk", (100.0, 0.0, 0.0))
    points = scene_camera_points(centres, outcomes, difference, [box])
    expected = np.vstack([centres[[0, 2]], compute_cuboid_geometry(box).corners])
    np.testing.assert_allclose(points, expected)


def test_scene_window_follows_the_aspect():
    """Panel width / height of the window grid equals the requested aspect to within one pixel of rounding."""
    import importlib.util
    from pathlib import Path
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "26_occupancy_figures.py"
    module_spec = importlib.util.spec_from_file_location("script26_for_window_test", script_path)
    script = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(script)
    context = SimpleNamespace(settings=SETTINGS)
    aspect = 2.5
    window_width, window_height = script.FigureContext.scene_window(context, aspect)
    panel_width, panel_height = window_width / SCENE_GRID_COLUMNS, window_height / SCENE_GRID_ROWS
    assert panel_width == TILE_PX * script.SCENE_PANEL_TILES
    assert abs(panel_width / panel_height - aspect) < aspect / panel_height


# ---- content bounding box and grid crop (synthetic images) ----
WHITE = BACKGROUND_RGB


def blank_image(height, width):
    """Background-coloured (height, width, 3) uint8 image."""
    return np.full((height, width, CHANNELS), WHITE, dtype=np.uint8)


def test_content_bounding_box_known_rectangle():
    """A 3 x 2 red block at rows 4-6, columns 7-8 gives exclusive stops (7, 9)."""
    image = blank_image(12, 15)
    image[4:7, 7:9] = (200, 10, 10)
    assert content_bounding_box(image, WHITE, 0) == (4, 7, 7, 9)


def test_content_bounding_box_blank_is_none():
    """No content pixel, no box."""
    assert content_bounding_box(blank_image(5, 5), WHITE, 0) is None


def test_content_bounding_box_tolerance_ignores_faint_pixels():
    """A pixel 2 levels off the background is content at tolerance 1 but not at tolerance 2; no uint8 wrap."""
    image = blank_image(6, 6)
    image[2, 3] = (253, 255, 255)
    assert content_bounding_box(image, WHITE, 1) == (2, 3, 3, 4)
    assert content_bounding_box(image, WHITE, 2) is None
    dark = np.zeros((3, 3, CHANNELS), dtype=np.uint8)
    dark[1, 1] = (255, 255, 255)  # difference 255 from a black background must not wrap to 0
    assert content_bounding_box(dark, (0, 0, 0), 0) == (1, 2, 1, 2)


def test_crop_grid_uses_the_union_box_for_every_cell_and_reassembles():
    """Two cells with different content share one crop box; the empty cell stays background; band is reserved."""
    rows, columns, cell_height, cell_width = 2, 3, 20, 30
    image = blank_image(rows * cell_height, columns * cell_width)
    # Cell (0, 0): content at local rows 4-7, columns 5-9. Cell (1, 2): local rows 8-13, columns 12-19.
    image[4:8, 5:10] = (10, 10, 200)
    image[cell_height + 8:cell_height + 14, 2 * cell_width + 12:2 * cell_width + 20] = (200, 10, 10)
    margin_fraction, band_fraction = 0.0, 0.5
    cropped, report = crop_grid_to_content(image, (rows, columns), [(0, 0), (1, 2)], WHITE, 0, margin_fraction,
                                           band_fraction)
    # Union box (local): rows 4-14, columns 5-20 -> 10 x 15 pixels; band = round(0.5 * 10) = 5 pixels.
    assert report["union_box"] == [4, 14, 5, 20]
    assert report["cell_shape"] == [10, 15] and report["title_band_px"] == 5
    assert cropped.shape == (rows * (5 + 10), columns * 15, CHANNELS)
    # Blue block: cell (0, 0) rows 4-7 -> cropped rows 0-3 below the band (top = 5), columns 0-4.
    assert (cropped[5:9, 0:5] == (10, 10, 200)).all()
    # Red block: cell (1, 2) rows 8-13 -> crop rows 4-9, columns 7-14, in row 1 (top = 15 + 5) and column 2 (x0 = 30).
    assert (cropped[20 + 4:20 + 10, 30 + 7:30 + 15] == (200, 10, 10)).all()
    # Empty cells (0, 1), (0, 2) and the bands are background.
    assert (cropped[:, 15:] == WHITE)[:15].all()
    assert (cropped[:5] == WHITE).all()
    # Fill: blue block 4 x 5 of the 20 x 30 cell, and of the 10 x 15 crop.
    assert report["fill_by_cell"]["0,0"] == {"before_crop": 20 / 600, "after_crop": 20 / 150}


def test_crop_grid_margin_grows_the_box_and_clamps_to_the_cell():
    """A 10 % margin adds round(0.1 x extent) per side, never past the cell."""
    image = blank_image(20, 20)
    image[8:12, 8:12] = (0, 0, 0)
    _, report = crop_grid_to_content(image, (1, 1), [(0, 0)], WHITE, 0, 0.5, 0.0)
    # Extent 4 -> margin round(2.0) = 2 per side.
    assert report["union_box"] == [6, 14, 6, 14]
    image[0, 0] = (0, 0, 0)
    _, clamped = crop_grid_to_content(image, (1, 1), [(0, 0)], WHITE, 0, 0.5, 0.0)
    assert clamped["union_box"][0] == 0 and clamped["union_box"][2] == 0


def test_crop_grid_without_content_keeps_the_whole_cell():
    """Nothing drawn: the crop is the full cell."""
    cropped, report = crop_grid_to_content(blank_image(8, 12), (2, 3), [(0, 0)], WHITE, 0, 0.1, 0.0)
    assert report["cell_shape"] == [4, 4] and cropped.shape == (8, 12, CHANNELS)


# ---- stand-alone difference panel ----
def test_difference_camera_points_are_the_union_of_the_difference_masks():
    """Exactly the centres of cells in any difference mask, in cell order."""
    centres = np.arange(15, dtype=float).reshape(5, 3)
    difference = {"only_first": np.array([True, False, False, False, False]),
                  "only_second": np.array([False, False, True, False, False]),
                  "both": np.array([False, False, False, False, False])}
    expected = centres[[0, 2]]
    np.testing.assert_allclose(difference_camera_points(centres, difference), expected)


def test_difference_camera_ignores_cells_outside_every_difference_mask():
    """A far cell in no mask leaves the fitted camera unchanged."""
    near = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 0.5]])
    centres_with_far = np.vstack([near, [[500.0, 400.0, 30.0]]])
    mask_near = np.array([True, True])
    mask_with_far = np.array([True, True, False])
    points_without = difference_camera_points(near, {"both": mask_near})
    points_with = difference_camera_points(centres_with_far, {"both": mask_with_far})
    aspect = 2.0
    camera_without = tight_oblique_camera(points_without, aspect, SCENE_CAMERA_PADDING, OBLIQUE_CAMERA_DIRECTION,
                                          VIEW_UP_VECTOR, CAMERA_VIEW_ANGLE_DEGREES)
    camera_with = tight_oblique_camera(points_with, aspect, SCENE_CAMERA_PADDING, OBLIQUE_CAMERA_DIRECTION,
                                       VIEW_UP_VECTOR, CAMERA_VIEW_ANGLE_DEGREES)
    for camera_part_without, camera_part_with in zip(camera_without, camera_with):
        np.testing.assert_allclose(np.asarray(camera_part_without), np.asarray(camera_part_with))


class BlockScreenshotPlotter(RecordingPlotter):
    """Plotter whose screenshot is a white window with a block coloured like the only_stock cells."""

    def screenshot(self, return_img=False):
        """White window_size image with a centred only_stock-coloured block."""
        window_width, window_height = self.kwargs["window_size"]
        image = np.full((window_height, window_width, CHANNELS), BACKGROUND_RGB, dtype=np.uint8)
        quarter_height, quarter_width = window_height // 4, window_width // 4
        image[quarter_height:window_height - quarter_height,
              quarter_width:window_width - quarter_width] = COLOURS["only_stock"]
        return image


class BlockFakePyvista(FakePyvista):
    """Fake pyvista whose plotter returns the block screenshot."""

    Plotter = BlockScreenshotPlotter


def test_render_difference_panel_returns_a_cropped_uint8_image_with_the_difference_colour():
    """(H, W, 3) uint8, content kept, width within the window, one camera on one plotter without a grid."""
    centres = np.array([[0.0, 0.0, 0.0], [3.0, 1.0, 0.5], [9.0, 2.0, 0.0]])
    none = np.zeros(3, dtype=bool)
    difference = {"only_first": none, "only_second": np.array([True, True, False]), "both": none}
    window_size = (200, 100)
    image, report = render_difference_panel(BlockFakePyvista, centres, VOXEL_EDGE_METERS, difference, COLOURS,
                                            window_size)
    plotter = last_plotter()
    assert "shape" not in plotter.kwargs and plotter.closed
    assert image.dtype == np.uint8 and image.ndim == 2 + 1 and image.shape[2] == CHANNELS
    assert image.shape[1] <= window_size[0]
    # Height = the cropped panel alone: the lone panel keeps no title band.
    cell_height, cell_width = report["cell_shape"]
    assert report["title_band_px"] == 0
    assert image.shape[:2] == (cell_height, cell_width)
    assert cell_height <= window_size[1]
    # Only-stock colour survives the crop.
    assert (image == np.asarray(COLOURS["only_stock"], dtype=np.uint8)).all(axis=2).any()
    assert content_bounding_box(image, BACKGROUND_RGB, 1) is not None
    # Camera is the one fitted to the difference cells' cube corners (centre ± half edge per axis) at the window aspect.
    half_edge = VOXEL_EDGE_METERS / 2.0
    corners = np.array([centre + half_edge * np.array(signs) for centre in centres[:2]
                        for signs in itertools.product((-1.0, 1.0), repeat=3)])
    expected = tight_oblique_camera(corners, window_size[0] / window_size[1], SCENE_CAMERA_PADDING,
                                    OBLIQUE_CAMERA_DIRECTION, VIEW_UP_VECTOR, CAMERA_VIEW_ANGLE_DEGREES)
    for camera_part, expected_part in zip(plotter.camera_position, expected):
        np.testing.assert_allclose(np.asarray(camera_part), np.asarray(expected_part))


def test_cell_corner_points_are_centre_plus_minus_half_edge_on_every_axis():
    """Eight corners per cell, each centre ± voxel / 2 per axis; empty in, empty out."""
    centres = np.array([[1.0, 2.0, 3.0], [-4.0, 0.0, 5.0]])
    voxel_edge = 0.4
    corners = cell_corner_points(centres, voxel_edge)
    corners_per_cell = 2 ** 3
    assert corners.shape == (centres.shape[0] * corners_per_cell, 3)
    for cell_index, centre in enumerate(centres):
        cell_corners = corners[cell_index * corners_per_cell:(cell_index + 1) * corners_per_cell]
        # Every corner is exactly half an edge from the centre on each axis, and all 8 are distinct.
        np.testing.assert_allclose(np.abs(cell_corners - centre), voxel_edge / 2.0)
        assert len({tuple(corner) for corner in np.round(cell_corners, 12)}) == corners_per_cell
    assert cell_corner_points(np.zeros((0, 3)), voxel_edge).shape == (0, 3)


@pytest.mark.parametrize("centres", [np.array([[2.0, 3.0, 1.0]]),
                                     np.array([[0.0, 0.0, z_metres] for z_metres in (0.0, 1.0, 2.0, 3.0)]),
                                     np.array([[x_metres, 0.0, 0.0] for x_metres in (0.0, 1.0, 2.0, 3.0)])],
                         ids=["one cell", "vertical line", "horizontal line"])
def test_difference_frame_aspect_is_finite_and_positive_for_degenerate_cell_sets(centres):
    """One cell or a line of cells has a zero extent on a screen axis by centres; the cube corners frame it."""
    difference = {"only_first": np.ones(centres.shape[0], dtype=bool),
                  "only_second": np.zeros(centres.shape[0], dtype=bool),
                  "both": np.zeros(centres.shape[0], dtype=bool)}
    aspect = scene_camera_aspect(difference_frame_points(centres, difference, VOXEL_EDGE_METERS))
    assert np.isfinite(aspect) and aspect > 0


def test_render_difference_panel_frames_a_single_cell():
    """A single difference cell still renders (no NaN aspect or empty frame)."""
    centres = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]])
    difference = {"only_first": np.zeros(2, dtype=bool), "only_second": np.array([True, False]),
                  "both": np.zeros(2, dtype=bool)}
    aspect = scene_camera_aspect(difference_frame_points(centres, difference, VOXEL_EDGE_METERS))
    window_width = WINDOW_SIZE[0]
    window_size = (window_width, max(1, round(window_width / aspect)))
    image, _ = render_difference_panel(BlockFakePyvista, centres, VOXEL_EDGE_METERS, difference, COLOURS, window_size)
    assert image.dtype == np.uint8 and image.size > 0


def test_render_difference_panel_without_difference_cells_raises():
    """Nothing to frame: ValueError."""
    centres = np.arange(6, dtype=float).reshape(2, 3)
    none = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError):
        render_difference_panel(FakePyvista, centres, VOXEL_EDGE_METERS,
                                {"only_first": none, "only_second": none, "both": none}, COLOURS, WINDOW_SIZE)
