"""Tests of the pure PyVista-viewer helpers (viz/pyvista_scene.py) on synthetic boxes, no PyVista needed."""

import math
from pathlib import Path

import numpy as np

from viking_splatad.labels.cuboids import Cuboid, compute_cuboid_geometry
from viking_splatad.viz.pyvista_scene import (
    CORNERS_PER_CUBOID,
    INSIDE_CUBOID_RGB,
    OUTSIDE_CUBOID_RGB,
    VTK_POINTS_PER_EDGE,
    cuboid_edge_index_pairs,
    cuboid_wireframe_arrays,
    inside_outside_colors,
    viz_command,
)

# A cube has 8 corners, each joined to 3 neighbours: 8 * 3 / 2 edges.
EDGES_OF_A_BOX = 12


def yawed_box(center, dimensions, yaw_radians, box_id="box") -> Cuboid:
    """A yaw-only Cuboid (class 1) with FULL sizes ``dimensions``."""
    quaternion_xyzw = (0.0, 0.0, math.sin(yaw_radians / 2.0), math.cos(yaw_radians / 2.0))
    return Cuboid(box_id, 1, "tree_trunk", tuple(center), tuple(dimensions), quaternion_xyzw)


def test_a_box_has_twelve_edges_and_every_corner_has_three():
    """Edges join corners that differ in exactly one sign: 12 pairs, each corner used 3 times."""
    edge_pairs = cuboid_edge_index_pairs()
    assert edge_pairs.shape == (EDGES_OF_A_BOX, 2)
    corner_degrees = np.bincount(edge_pairs.reshape(-1), minlength=CORNERS_PER_CUBOID)
    assert np.all(corner_degrees == 3)
    assert len({tuple(pair) for pair in edge_pairs}) == EDGES_OF_A_BOX


def test_wireframe_of_a_rotated_box_follows_its_axes():
    """A 0.5 x 0.8 x 7 box yawed by 0.4 rad at (3, -2, 3.5): every edge has one of the three
    full sizes (4 edges each), and the corners are the T30 corners."""
    box = yawed_box((3.0, -2.0, 3.5), (0.5, 0.8, 7.0), 0.4)
    corner_points, line_cells = cuboid_wireframe_arrays([box])
    np.testing.assert_allclose(corner_points, compute_cuboid_geometry(box).corners)
    edge_cells = line_cells.reshape(-1, VTK_POINTS_PER_EDGE + 1)
    assert np.all(edge_cells[:, 0] == VTK_POINTS_PER_EDGE)
    edge_lengths = np.linalg.norm(corner_points[edge_cells[:, 1]] - corner_points[edge_cells[:, 2]], axis=1)
    for full_size in box.dimensions_xyz:
        assert np.count_nonzero(np.isclose(edge_lengths, full_size)) == 4


def test_two_boxes_stack_their_corners_and_shift_their_edge_indices():
    """Box k owns corner rows 8k..8k+7: the second box's edge indices all lie in 8..15."""
    boxes = [yawed_box((0, 0, 0), (1, 1, 1), 0.0, "a"), yawed_box((5, 0, 0), (1, 2, 3), 1.0, "b")]
    corner_points, line_cells = cuboid_wireframe_arrays(boxes)
    assert corner_points.shape == (2 * CORNERS_PER_CUBOID, 3)
    edge_cells = line_cells.reshape(2, EDGES_OF_A_BOX, VTK_POINTS_PER_EDGE + 1)
    assert edge_cells[0, :, 1:].max() < CORNERS_PER_CUBOID
    assert edge_cells[1, :, 1:].min() >= CORNERS_PER_CUBOID
    np.testing.assert_allclose(corner_points[CORNERS_PER_CUBOID:], compute_cuboid_geometry(boxes[1]).corners)


def test_no_boxes_give_empty_arrays():
    """Nothing to draw -> empty corner and line arrays, not an error."""
    corner_points, line_cells = cuboid_wireframe_arrays([])
    assert corner_points.shape == (0, 3) and line_cells.size == 0


def test_inside_points_get_the_inside_colour():
    """True -> inside colour, False -> outside colour, as uint8 rows."""
    colors = inside_outside_colors(np.array([True, False, True]))
    assert colors.dtype == np.uint8
    np.testing.assert_array_equal(colors, [INSIDE_CUBOID_RGB, OUTSIDE_CUBOID_RGB, INSIDE_CUBOID_RGB])


def test_viz_command_names_every_run_and_the_isolated_project():
    """The command runs scripts/15 in viz/, passes each run, the output folder, config and threshold."""
    command = viz_command([Path("/r/B"), Path("/r/R")], Path("/out"), Path("/cfg/eval.yaml"), 0.5, "render-x")
    assert command[:5] == ["uv", "run", "--project", "viz", "python"]
    assert command[5].endswith("15_view_pose_pyvista.py")
    assert [command[index + 1] for index, token in enumerate(command) if token == "--run"] == ["/r/B", "/r/R"]
    assert command[command.index("--output-dir") + 1] == "/out"
    assert float(command[command.index("--accumulation-threshold") + 1]) == 0.5
    assert command[command.index("--rerender") + 1] == "render-x"
    assert "--rerender" not in viz_command([Path("/r/B")], Path("/out"), Path("/cfg/eval.yaml"), 0.5)
    assert "--radius-meters" not in command
    with_radius = viz_command([Path("/r/B")], Path("/out"), Path("/cfg/eval.yaml"), 0.5, radius_meters=15)
    assert float(with_radius[with_radius.index("--radius-meters") + 1]) == 15.0


def test_viz_command_passes_the_class_filter_only_when_given():
    """--classes carries the comma-joined names; absent when no filter is asked for."""
    command = viz_command([Path("/r/B")], Path("/out"), Path("/cfg/eval.yaml"), 0.5, class_names=["tree_trunk", "rock"])
    assert command[command.index("--classes") + 1] == "tree_trunk,rock"
    assert "--classes" not in viz_command([Path("/r/B")], Path("/out"), Path("/cfg/eval.yaml"), 0.5)
