"""Tests for the scored-cell builder (synthetic scenes, expected values from geometry)."""

import numpy as np

from viking_splatad.eval.detection import range_limit_from_real_points
from viking_splatad.eval.scored_cells import (
    RANGE_HISTOGRAM_BIN_M,
    build_scored_cells,
    object_membership,
    range_quantile_from_histogram,
    scan_sweeps,
    scored_cell_set,
    scored_neighbourhoods,
    streaming_max_range,
)
from viking_splatad.eval.session_rays import SessionSweep
from viking_splatad.labels.cuboids import Cuboid

IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
RANDOM_SEED = 7
SWEEP_COUNT = 5
POINTS_PER_SWEEP = 50
RANGE_SCALE_M = 40.0
BOX_EDGE_M = 2.0
# Synthetic session for the build test: cubic cells of this edge on a grid of GRID_SHAPE cells.
BUILD_VOXEL_SIZE_M = 1.0
GRID_SHAPE = (24, 8, 8)
GRID_ORIGIN = np.array([0.0, -4.0, -4.0])
# One box of class BUILD_CLASS_ID centred at BUILD_BOX_CENTRE; the sweeps stand at the origin.
BUILD_CLASS_ID = 1
BUILD_CLASS_NAME = "trunk"
BUILD_BOX_CENTRE_X_M = 10.0
# Return ranges (m) of the synthetic sweeps: all well beyond the box so the box is in range.
BUILD_RETURN_RANGES_M = (15.0, 20.0, 25.0)
BUILD_VERTICAL_HALF_FOV_RAD = 0.5
BUILD_RANGE_QUANTILE = 1.0


def make_sweep(stem, returns_xyz):
    """Sweep with identity pose at the origin."""
    return SessionSweep(stem=stem, split="train", timestamp_ns=0, returns_xyz=np.asarray(returns_xyz, dtype=float),
                        lidar_to_map=np.eye(4))


def make_box(box_id, class_id, centre_x):
    """Axis-aligned cube of edge BOX_EDGE_M at (centre_x, 0, 0)."""
    return Cuboid(box_id=box_id, class_id=class_id, class_name="c", center_xyz=(centre_x, 0.0, 0.0),
                  dimensions_xyz=(BOX_EDGE_M,) * 3, quaternion_xyzw=IDENTITY_QUATERNION)


def test_object_membership_shared_cell_and_inside_flags():
    # Box A spans x in [-1, 1] and is grown by 0.5 m to [-1.5, 1.5]; box B spans [0.5, 2.5], no growth.
    box_a, box_b = make_box("A", 1, 0.0), make_box("B", 3, 1.5)
    neighbourhoods = {"a": {"class_id": 1, "cuboids": [box_a], "margin": 0.5},
                      "b": {"class_id": 3, "cuboids": [box_b], "margin": 0.0}}
    # x = -0.5 only A; 0.75 both; 1.25 A grown (not original) and B; 2.0 only B; 3.5 none.
    centres = np.array([[-0.5, 0, 0], [0.75, 0, 0], [1.25, 0, 0], [2.0, 0, 0], [3.5, 0, 0]], dtype=float)
    cell_indices = np.arange(len(centres) * 3).reshape(-1, 3)
    box_ids, class_ids, offsets, cells, inside = object_membership(neighbourhoods, cell_indices, centres, {})
    assert box_ids.tolist() == ["A", "B"] and class_ids.tolist() == [1, 3]
    a_slice, b_slice = slice(offsets[0], offsets[1]), slice(offsets[1], offsets[2])
    assert sorted(cells[a_slice].tolist()) == [0, 1, 2]
    assert sorted(cells[b_slice].tolist()) == [1, 2, 3]
    inside_a = dict(zip(cells[a_slice].tolist(), inside[a_slice].tolist()))
    inside_b = dict(zip(cells[b_slice].tolist(), inside[b_slice].tolist()))
    assert inside_a == {0: True, 1: True, 2: False}
    assert inside_b == {1: True, 2: True, 3: True}


def test_streaming_max_range_equals_range_limit_from_real_points():
    random_state = np.random.default_rng(RANDOM_SEED)
    sweeps = [random_state.normal(scale=RANGE_SCALE_M, size=(POINTS_PER_SWEEP, 3)) for _ in range(SWEEP_COUNT)]
    sweeps.append(np.zeros((0, 3)))  # an empty sweep is skipped by both
    assert streaming_max_range(iter(sweeps)) == range_limit_from_real_points(sweeps)
    assert streaming_max_range(iter([])) == range_limit_from_real_points([]) == 0.0


def test_scored_cell_set_picks_grid_labels():
    grid_class_ids = np.zeros((3, 3, 1), dtype=np.uint8)
    grid_class_ids[0, 1, 0], grid_class_ids[2, 2, 0] = 1, 4
    mask_a, mask_b = np.zeros((3, 3, 1), dtype=bool), np.zeros((3, 3, 1), dtype=bool)
    mask_a[0, 1, 0] = mask_a[1, 1, 0] = True
    mask_b[2, 2, 0] = True
    grid = {"origin": np.array([10.0, 20.0, 30.0]), "voxel_size": 2.0, "class_ids": grid_class_ids}
    neighbourhoods = {"a": {"neighbourhood": mask_a}, "b": {"neighbourhood": mask_b}}
    cell_indices, centres, by_class, gt_ids = scored_cell_set(neighbourhoods, grid)
    # argwhere order: (0,1,0), (1,1,0), (2,2,0).
    assert cell_indices.tolist() == [[0, 1, 0], [1, 1, 0], [2, 2, 0]]
    assert gt_ids.tolist() == [1, 0, 4]
    assert by_class["a"].tolist() == [True, True, False] and by_class["b"].tolist() == [False, False, True]
    # centre = origin + (index + 0.5) * voxel_size.
    assert centres[0].tolist() == [10.0 + 0.5 * 2.0, 20.0 + 1.5 * 2.0, 30.0 + 0.5 * 2.0]


def test_range_quantile_from_histogram_and_scan_histogram():
    """The streamed range histogram gives the upper bin edge of the requested share of returns."""
    # Ten returns along +x at 1, 2, ..., 10 m (whole numbers of bins), split over two sweeps.
    ranges_m = np.arange(1, 11, dtype=np.float64)
    returns = [np.column_stack([ranges_m[:5], np.zeros(5), np.zeros(5)]),
               np.column_stack([ranges_m[5:], np.zeros(5), np.zeros(5)])]
    sweeps = [SessionSweep(stem=f"sweep{index}", split="train", timestamp_ns=index, returns_xyz=points,
                           lidar_to_map=np.eye(4))
              for index, points in enumerate(returns)]
    scan = scan_sweeps(sweeps)
    assert int(scan.range_histogram_counts.sum()) == len(ranges_m)
    assert scan.max_range_m == ranges_m.max()
    # 0.7 of 10 returns = the 7th smallest (7 m); it sits in bin floor(7 / w), whose upper edge is 7 m + w.
    seventh_range_m = ranges_m[6]
    expected_limit_m = (np.floor(seventh_range_m / RANGE_HISTOGRAM_BIN_M) + 1) * RANGE_HISTOGRAM_BIN_M
    assert range_quantile_from_histogram(scan.range_histogram_counts, 0.7) == expected_limit_m
    # Quantile 1 reaches the farthest return's bin.
    farthest_limit_m = (np.floor(ranges_m.max() / RANGE_HISTOGRAM_BIN_M) + 1) * RANGE_HISTOGRAM_BIN_M
    assert range_quantile_from_histogram(scan.range_histogram_counts, 1.0) == farthest_limit_m
    assert range_quantile_from_histogram(np.zeros(3, dtype=np.int64), 0.5) == 0.0


def build_sweeps():
    """Synthetic sweeps at the origin, one return along +x per range in BUILD_RETURN_RANGES_M."""
    return [make_sweep(f"sweep{index}", [[range_m, 0.0, 0.0]]) for index, range_m in enumerate(BUILD_RETURN_RANGES_M)]


def build_scene():
    """Box, GT grid (the box's cells labelled with its class id) and scored class ids of the build tests."""
    box = make_box("trunk_0", BUILD_CLASS_ID, BUILD_BOX_CENTRE_X_M)
    grid_class_ids = np.zeros(GRID_SHAPE, dtype=np.uint8)
    # The box spans x in [9, 11], y and z in [-1, 1]; with origin (0, -4, -4) and edge 1 its cells are
    # x indices 9..10, y and z indices 3..4 (cell centres 9.5, 10.5 / -0.5, 0.5).
    grid_class_ids[9:11, 3:5, 3:5] = BUILD_CLASS_ID
    grid = {"origin": GRID_ORIGIN, "voxel_size": BUILD_VOXEL_SIZE_M, "class_ids": grid_class_ids}
    return [box], grid, {BUILD_CLASS_NAME: BUILD_CLASS_ID}


def test_build_scored_cells_matches_the_step_by_step_composition():
    """The one-call build equals scan -> range -> neighbourhoods -> cells -> object lists, bit for bit."""
    cuboids, grid, scored_class_ids = build_scene()
    result = build_scored_cells(
        build_sweeps(), cuboids, scored_class_ids, grid, BUILD_VERTICAL_HALF_FOV_RAD, BUILD_RANGE_QUANTILE)
    # Expected, composed from the retained building blocks.
    scan = scan_sweeps(build_sweeps())
    expected_range_m = range_quantile_from_histogram(scan.range_histogram_counts, BUILD_RANGE_QUANTILE)
    neighbourhoods = scored_neighbourhoods(
        cuboids, scored_class_ids, scan.sweep_poses, expected_range_m, BUILD_VERTICAL_HALF_FOV_RAD, grid)
    cell_indices, centres, by_class, gt_class_ids = scored_cell_set(neighbourhoods, grid)
    object_lists = object_membership(neighbourhoods, cell_indices, centres, grid)
    assert result["scored_range_m"] == expected_range_m
    assert np.array_equal(result["cell_indices"], cell_indices) and np.array_equal(result["centres"], centres)
    assert np.array_equal(result["gt_class_ids"], gt_class_ids)
    assert np.array_equal(result["neighbourhood_by_class"][BUILD_CLASS_NAME], by_class[BUILD_CLASS_NAME])
    for result_key, expected_array in zip(
            ("object_box_ids", "object_class_ids", "object_cell_offsets", "object_cells", "object_inside"),
            object_lists):
        assert np.array_equal(result[result_key], expected_array)
    assert result["scan"].sweep_count_by_split == {"train": len(BUILD_RETURN_RANGES_M)}


def test_build_scored_cells_scores_the_box_cells_from_first_principles():
    """The 2 x 2 x 2 cells of the box are scored, labelled with its class, and inside its own object list."""
    cuboids, grid, scored_class_ids = build_scene()
    result = build_scored_cells(
        build_sweeps(), cuboids, scored_class_ids, grid, BUILD_VERTICAL_HALF_FOV_RAD, BUILD_RANGE_QUANTILE)
    box_cell_count = int(np.count_nonzero(grid["class_ids"]))
    # Every labelled cell is in the scored set (the neighbourhood is the box grown by a margin >= 0).
    assert int(np.count_nonzero(result["gt_class_ids"] == BUILD_CLASS_ID)) == box_cell_count
    # The scored range is the quantile-1 bin edge: the farthest return's bin upper edge.
    farthest_range_m = max(BUILD_RETURN_RANGES_M)
    assert result["scored_range_m"] == (np.floor(farthest_range_m / RANGE_HISTOGRAM_BIN_M) + 1) * RANGE_HISTOGRAM_BIN_M
    # The one object owns the cells whose centre is in its original box: exactly the labelled cells.
    assert result["object_box_ids"].tolist() == ["trunk_0"]
    assert int(np.count_nonzero(result["object_inside"])) == box_cell_count


def test_build_scored_cells_reads_the_sweeps_once():
    """A one-shot iterator gives the same result as a list: the build makes a single pass."""
    cuboids, grid, scored_class_ids = build_scene()
    from_list = build_scored_cells(
        build_sweeps(), cuboids, scored_class_ids, grid, BUILD_VERTICAL_HALF_FOV_RAD, BUILD_RANGE_QUANTILE)
    from_iterator = build_scored_cells(
        iter(build_sweeps()), cuboids, scored_class_ids, grid, BUILD_VERTICAL_HALF_FOV_RAD, BUILD_RANGE_QUANTILE)
    assert from_iterator["scan"].sweep_count_by_split == {"train": len(BUILD_RETURN_RANGES_M)}
    assert np.array_equal(from_iterator["cell_indices"], from_list["cell_indices"])
    assert np.array_equal(from_iterator["object_cells"], from_list["object_cells"])
