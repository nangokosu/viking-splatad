"""Tests of the off-screen figure renderers (viz/figures_pyvista.py) against a recording stand-in for pyvista."""

from types import SimpleNamespace

import numpy as np

from viking_splatad.viz.figure_grids import OUTCOME_STRATUM_NAMES
from viking_splatad.viz.figure_runs import FigureSettings
from viking_splatad.viz.pyvista_scene import MAX_CHANNEL_VALUE
from viking_splatad.viz.figures_pyvista import (
    CAPTION_DECIMAL_PLACES,
    MISSED_HIDDEN_DARKEN_FACTOR,
    OCCUPANCY_CLIM,
    compose_tile_grid,
    darken_rgb,
    draw_scalar_voxels,
    format_tile_caption,
    paginate,
    render_arm_panels,
    render_object_tile,
    render_outcome_strata,
)

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
    campaign_profiles={}, section_trunk_count=1, rows_per_page=2, dpi=DPI,
    tile_px=TILE_PX, bev_bin_m=None, difference_cmap="RdBu", occupancy_cmap="viridis", colours=COLOURS,
    footer_note=FOOTER, representative_metric="recall_H")


class RecordingPolyData:
    """Stand-in for ``pyvista.PolyData``: remembers points, scalars and glyph arguments."""

    def __init__(self, points, lines=None):
        """Store points (and line cells)."""
        self.points = np.asarray(points)
        self.lines = lines
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


def test_outcome_strata_colours_counts_and_close():
    """Each stratum is drawn in its colour with its cell count, the legend lists the counts, the plotter closes."""
    centres = np.arange(18, dtype=float).reshape(6, 3)
    strata = np.array([0, 1, 1, 2, 3, 4])
    image = render_outcome_strata(FakePyvista, centres, VOXEL_EDGE_METERS, strata, SETTINGS,
                                  window_size=WINDOW_SIZE, title="F2")
    plotter = last_plotter()
    assert image.dtype == np.uint8 and image.shape[2] == CHANNELS
    assert plotter.kwargs["off_screen"] is True and plotter.closed
    hidden_hit_meshes = mesh_by_colour(plotter, COLOURS["only_radar_init_loss"])
    assert len(hidden_hit_meshes) == 1 and hidden_hit_meshes[0].points.shape[0] == 2
    assert mesh_by_colour(plotter, COLOURS["hit"])[0].points.shape[0] == 1
    darker_grey = darken_rgb(COLOURS["missed"], MISSED_HIDDEN_DARKEN_FACTOR)
    assert darker_grey != COLOURS["missed"] and len(mesh_by_colour(plotter, darker_grey)) == 1
    expected_counts = {name: int((strata == index).sum()) for index, name in enumerate(OUTCOME_STRATUM_NAMES)}
    legend = plotter.texts[0]
    assert all(f"{name}: {count}" in legend for name, count in expected_counts.items())


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
                      "radar_init_loss": outcomes(hit_init)}, difference, SETTINGS, window_size=WINDOW_SIZE,
                      title_by_panel=titles)
    plotter = last_plotter()
    expected_panel_count = len(SETTINGS.arm_order) + 1
    assert plotter.kwargs["shape"] == (1, expected_panel_count)
    assert plotter.linked and plotter.closed
    assert plotter.texts == titles
    # Difference colours: counts computed from the masks.
    assert mesh_by_colour(plotter, COLOURS["only_radar_init_loss"])[0].points.shape[0] == int(difference["only_first"].sum())
    assert mesh_by_colour(plotter, COLOURS["only_stock"])[0].points.shape[0] == int(difference["only_second"].sum())
    assert plotter.subplots[-1][1] == expected_panel_count - 1


def test_draw_scalar_voxels_passes_scalars_with_unit_clim():
    """The occupancy values ride on the mesh and the colour limits are 0-1."""
    plotter = RecordingPlotter()
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    values = np.array([0.2, 0.9])
    draw_scalar_voxels(plotter, FakePyvista, centres, VOXEL_EDGE_METERS, values, "viridis")
    mesh, mesh_kwargs = plotter.meshes[0]
    assert mesh_kwargs["clim"] == OCCUPANCY_CLIM == (0.0, 1.0)
    assert mesh_kwargs["cmap"] == "viridis"
    np.testing.assert_array_equal(mesh.point_data[mesh_kwargs["scalars"]], values)


def test_draw_scalar_voxels_empty_draws_nothing():
    """No centres: no mesh."""
    plotter = RecordingPlotter()
    draw_scalar_voxels(plotter, FakePyvista, np.zeros((0, 3)), VOXEL_EDGE_METERS, np.zeros(0), "viridis")
    assert plotter.meshes == []


def test_object_tile_camera_focuses_on_tile_centre_and_colours_cells():
    """Focal point = tile centre; hidden GT is scalar-coloured, visible GT faint, false alarm red."""
    centres = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.2, 0.0], [9.0, 9.0, 9.0]])
    truth = np.array([True, True, True, False, False])
    hidden = np.array([True, True, False, False, False])
    false_alarm = np.array([False, False, False, True, True])
    tile = {"cells": np.array([0, 1, 2]), "inside_cells": np.array([0, 1, 2]), "centre": np.array([0.4, 0.1, 0.0]),
            "extent_min": np.array([0.0, 0.0, 0.0]), "extent_max": np.array([1.0, 0.0, 0.0])}
    occupancy = np.array([0.1, 0.8, 0.5, 0.0, 0.0])
    render_object_tile(FakePyvista, centres, VOXEL_EDGE_METERS, tile, occupancy, truth, hidden, false_alarm, SETTINGS,
                       window_size=WINDOW_SIZE)
    plotter = last_plotter()
    np.testing.assert_allclose(plotter.camera_position[1], tile["centre"])
    scalar_meshes = [mesh for mesh, kwargs in plotter.meshes if "scalars" in kwargs]
    assert len(scalar_meshes) == 1 and scalar_meshes[0].points.shape[0] == int((truth & hidden).sum())
    # The far false alarm lies outside the tile box; only the one inside is drawn.
    assert mesh_by_colour(plotter, COLOURS["false_alarm"])[0].points.shape[0] == 1
    assert mesh_by_colour(plotter, COLOURS["context"])[0].points.shape[0] == 1
    assert plotter.closed


def test_object_tile_draws_only_its_own_gt_not_a_neighbours_in_the_grown_box():
    """Regression: a GT cell in the grown box but outside this object's own box (a neighbour's) is not drawn."""
    centres = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]])
    truth = np.array([True, True, True])
    hidden = np.array([True, True, False])
    false_alarm = np.zeros(3, dtype=bool)
    # The grown box holds cells 0, 1, 2 but only cell 0 lies inside the object's own box.
    tile = {"cells": np.array([0, 1, 2]), "inside_cells": np.array([0]), "centre": np.array([0.0, 0.0, 0.0]),
            "extent_min": np.array([0.0, 0.0, 0.0]), "extent_max": np.array([1.0, 0.0, 0.0])}
    render_object_tile(FakePyvista, centres, VOXEL_EDGE_METERS, tile, np.array([0.9, 0.9, 0.9]), truth, hidden,
                       false_alarm, SETTINGS, window_size=WINDOW_SIZE)
    plotter = last_plotter()
    scalar_meshes = [mesh for mesh, kwargs in plotter.meshes if "scalars" in kwargs]
    # Only cell 0 (own, hidden) is scalar-coloured; cell 1 (hidden, a neighbour's) and cell 2 (visible) are not drawn.
    assert len(scalar_meshes) == 1 and scalar_meshes[0].points.shape[0] == 1
    assert np.allclose(scalar_meshes[0].points, centres[[0]])
    assert mesh_by_colour(plotter, COLOURS["context"]) == []


def test_paginate_exact_multiple_remainder_and_empty():
    """Pages hold rows_per_page items, the last the remainder, and no items give no pages."""
    assert paginate([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]
    assert paginate([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert paginate([], 3) == []


def test_format_tile_caption_precision_and_missing():
    """Numbers use the named precision; None prints n/a."""
    recall, baseline = 1 / 3, 0.8
    expected = f"recall_H {recall:.{CAPTION_DECIMAL_PLACES}f} · baseline {baseline:.{CAPTION_DECIMAL_PLACES}f}"
    assert format_tile_caption(recall, baseline) == expected
    assert "n/a" in format_tile_caption(None, baseline)


def test_compose_tile_grid_has_rows_by_columns_image_axes_and_footer():
    """One axes per tile, each with an image, and the footer note as figure text."""
    row_count, column_count = 2, 3
    image = np.zeros((STUB_IMAGE_SIDE, STUB_IMAGE_SIDE, CHANNELS), dtype=np.uint8)
    tile_images = [[image] * column_count for _ in range(row_count)]
    captions = [[f"c{row}{column}" for column in range(column_count)] for row in range(row_count)]
    figure = compose_tile_grid(tile_images, ["r0", "r1"], ["a", "b", "c"], captions, SETTINGS)
    assert len(figure.axes) == row_count * column_count
    assert all(len(axes.images) == 1 for axes in figure.axes)
    assert FOOTER in [text.get_text() for text in figure.texts]


def test_tighter_margin_moves_camera_closer_by_the_margin_ratio():
    """Distance from the focal point scales linearly with the margin argument."""
    from viking_splatad.viz.figures_pyvista import CAMERA_FRAMING_MARGIN, oblique_camera_position
    cells = np.array([[0.0, 0.0, 0.0], [10.0, 4.0, 2.0]])
    default_view = oblique_camera_position(cells, VOXEL_EDGE_METERS)
    tight_margin = CAMERA_FRAMING_MARGIN / 2.0
    tight_view = oblique_camera_position(cells, VOXEL_EDGE_METERS, tight_margin)
    default_distance = np.linalg.norm(np.subtract(default_view[0], default_view[1]))
    tight_distance = np.linalg.norm(np.subtract(tight_view[0], tight_view[1]))
    np.testing.assert_allclose(tight_distance, default_distance * tight_margin / CAMERA_FRAMING_MARGIN)


def test_outcome_strata_text_legend_can_be_switched_off():
    """With draw_text_legend=False the image carries no text; the styles table feeds the colour legend."""
    from viking_splatad.viz.figures_pyvista import outcome_stratum_styles
    centres = np.arange(6, dtype=float).reshape(2, 3)
    render_outcome_strata(FakePyvista, centres, VOXEL_EDGE_METERS, np.array([0, 4]), SETTINGS,
                          window_size=WINDOW_SIZE, title="", draw_text_legend=False)
    assert last_plotter().texts == []
    assert set(outcome_stratum_styles(SETTINGS)) == set(OUTCOME_STRATUM_NAMES)


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


def test_tile_grid_key_adds_colourbar_and_legend_but_default_does_not():
    """show_key adds exactly one colourbar axes and the three key entries; the title is the suptitle."""
    row_count, column_count = 1, 2
    image = np.zeros((STUB_IMAGE_SIDE, STUB_IMAGE_SIDE, CHANNELS), dtype=np.uint8)
    tile_images = [[image] * column_count]
    captions = [["c0", "c1"]]
    keyed = compose_tile_grid(tile_images, ["r"], ["a", "b"], captions, SETTINGS, title="title", show_key=True,
                              run_note="drawn runs: x")
    assert len(keyed.axes) == row_count * column_count + 1
    assert [text.get_text() for text in keyed.legends[0].get_texts()] == [
        "false alarm", "visible GT cell (context)", "GT box"]
    assert keyed._suptitle.get_text() == "title"
    plain = compose_tile_grid(tile_images, ["r"], ["a", "b"], captions, SETTINGS)
    assert len(plain.axes) == row_count * column_count and not plain.legends
