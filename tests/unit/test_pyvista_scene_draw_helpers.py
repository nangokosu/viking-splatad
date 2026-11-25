"""Tests of the PyVista drawing helpers (viz/pyvista_scene.py) against a recording stand-in for pyvista."""

from types import SimpleNamespace

import numpy as np

from viking_splatad.labels.cuboids import Cuboid
from viking_splatad.viz.pyvista_scene import (
    CUBOID_WIREFRAME_RGB,
    CORNERS_PER_CUBOID,
    VOXEL_AMBIENT_LIGHT,
    WIREFRAME_LINE_WIDTH,
    draw_scored_cuboid_wireframes,
    draw_voxels,
)

# Full RGB scale, to compute the expected 0-1 colour from first principles.
MAX_COLOUR_LEVEL = 255.0
VOXEL_EDGE_METERS = 0.5
TEST_RGB = (51, 102, 255)
TEST_OPACITY = 0.25
WIREFRAME_RADIUS_METERS = 5.0
TRUNK_CLASS_ID = 1
OTHER_CLASS_ID = 2
# Identity quaternion (x, y, z, w) and a small box size for the synthetic cuboids.
IDENTITY_QUATERNION_XYZW = (0.0, 0.0, 0.0, 1.0)
BOX_DIMENSIONS = (0.5, 0.5, 2.0)
# Box centres: two near the origin lidar position, one 100 m away.
NEAR_CENTRES = ((1.0, 0.0, 1.0), (0.0, 2.0, 1.0))
FAR_CENTRE = (100.0, 0.0, 1.0)


class RecordingPolyData:
    """Stand-in for ``pyvista.PolyData`` that remembers its constructor arguments and glyph call."""

    def __init__(self, points, lines=None):
        """Store the points (and optional line cells).

        Args:
            points: (N, 3) points.
            lines: Optional VTK line cells.
        """
        self.points = np.asarray(points)
        self.lines = lines
        self.glyph_kwargs = None

    def glyph(self, **glyph_kwargs):
        """Record the glyph arguments and return self.

        Args:
            **glyph_kwargs: Glyph options.

        Returns:
            This object.
        """
        self.glyph_kwargs = glyph_kwargs
        return self


class FakePyvista:
    """Stand-in for the ``pyvista`` module: ``Cube`` and ``PolyData`` only."""

    PolyData = RecordingPolyData

    @staticmethod
    def Cube(**cube_kwargs):
        """Return the cube arguments as a namespace.

        Args:
            **cube_kwargs: Cube edge lengths.

        Returns:
            Namespace of the arguments.
        """
        return SimpleNamespace(**cube_kwargs)


class RecordingPlotter:
    """Stand-in for ``pyvista.Plotter`` that records every ``add_mesh`` call."""

    def __init__(self):
        """Start with no recorded calls."""
        self.add_mesh_calls = []

    def add_mesh(self, mesh, **mesh_kwargs):
        """Record one call.

        Args:
            mesh: The mesh drawn.
            **mesh_kwargs: Display options.
        """
        self.add_mesh_calls.append((mesh, mesh_kwargs))


def make_box(center_xyz, class_id) -> Cuboid:
    """An axis-aligned synthetic Cuboid of the given class.

    Args:
        center_xyz: Box centre.
        class_id: Class id.

    Returns:
        The cuboid.
    """
    return Cuboid("box", class_id, "name", tuple(center_xyz), BOX_DIMENSIONS, IDENTITY_QUATERNION_XYZW)


def test_draw_voxels_adds_one_mesh_with_scaled_colour_and_opacity():
    """One glyph mesh: colour = RGB / 255, the given opacity, ambient + diffuse sum to 1, one cube per centre."""
    plotter = RecordingPlotter()
    centres = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 0.0, 1.0]])
    draw_voxels(plotter, FakePyvista, centres, VOXEL_EDGE_METERS, TEST_RGB, TEST_OPACITY)
    assert len(plotter.add_mesh_calls) == 1
    mesh, mesh_kwargs = plotter.add_mesh_calls[0]
    np.testing.assert_allclose(mesh_kwargs["color"], np.array(TEST_RGB) / MAX_COLOUR_LEVEL)
    assert mesh_kwargs["opacity"] == TEST_OPACITY
    assert mesh_kwargs["ambient"] == VOXEL_AMBIENT_LIGHT
    assert mesh_kwargs["ambient"] + mesh_kwargs["diffuse"] == 1.0
    np.testing.assert_array_equal(mesh.points, centres)
    assert mesh.glyph_kwargs["geom"].x_length == VOXEL_EDGE_METERS


def test_draw_voxels_draws_nothing_without_centres():
    """An empty centre array adds no mesh."""
    plotter = RecordingPlotter()
    draw_voxels(plotter, FakePyvista, np.zeros((0, 3)), VOXEL_EDGE_METERS, TEST_RGB, TEST_OPACITY)
    assert plotter.add_mesh_calls == []


def test_wireframes_draw_one_mesh_of_the_selected_nearby_boxes_only():
    """Two near trunk boxes are drawn as one line mesh; the far trunk and the other class are skipped."""
    scene = SimpleNamespace(scored_cuboids=[make_box(NEAR_CENTRES[0], TRUNK_CLASS_ID),
                                            make_box(NEAR_CENTRES[1], TRUNK_CLASS_ID),
                                            make_box(FAR_CENTRE, TRUNK_CLASS_ID),
                                            make_box(NEAR_CENTRES[0], OTHER_CLASS_ID)])
    plotter = RecordingPlotter()
    draw_scored_cuboid_wireframes(plotter, FakePyvista, scene, [TRUNK_CLASS_ID], np.array([[0.0, 0.0]]),
                                  WIREFRAME_RADIUS_METERS)
    assert len(plotter.add_mesh_calls) == 1
    mesh, mesh_kwargs = plotter.add_mesh_calls[0]
    # Two boxes -> two sets of corners.
    assert mesh.points.shape[0] == len(NEAR_CENTRES) * CORNERS_PER_CUBOID
    np.testing.assert_allclose(mesh_kwargs["color"], np.array(CUBOID_WIREFRAME_RGB) / MAX_COLOUR_LEVEL)
    assert mesh_kwargs["line_width"] == WIREFRAME_LINE_WIDTH


def test_wireframes_draw_nothing_when_no_box_is_nearby():
    """No selected box within the radius: no mesh is added."""
    scene = SimpleNamespace(scored_cuboids=[make_box(FAR_CENTRE, TRUNK_CLASS_ID)])
    plotter = RecordingPlotter()
    draw_scored_cuboid_wireframes(plotter, FakePyvista, scene, [TRUNK_CLASS_ID], np.array([[0.0, 0.0]]),
                                  WIREFRAME_RADIUS_METERS)
    assert plotter.add_mesh_calls == []
