"""Tests for the session-wide hidden space builder (synthetic scenes, expected values from geometry)."""

import numpy as np

from viking_splatad.eval.detection import range_limit_from_real_points
from viking_splatad.eval.hidden_geometry import hidden_space_mask
from viking_splatad.eval.session_hidden_space import (
    accumulate_session_evidence,
    object_membership,
    scored_cell_set,
    streaming_max_range,
)
from viking_splatad.eval.session_rays import SessionSweep
from viking_splatad.labels.cuboids import Cuboid

VOXEL_SIZE_M = 1.0
RANGE_MARGIN_M = 0.1
MAX_RANGE_M = 30.0
# Cell centre of the scene, relative to the first sweep's origin.
CELL_CENTRE = np.array([[10.0, 0.0, 0.0]])
# The wall return of sweep 1 is 5 m away: entry of the cell (9.5 m) minus margin is far beyond it.
WALL_DEPTH_M = 5.0
# Sweep 2 stands 10 m beside the cell and looks along +y; its return lies exactly at the cell centre.
SIDE_ORIGIN = np.array([10.0, -10.0, 0.0])
SIDE_DEPTH_M = 10.0
IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
WORKER_COUNTS = (1, 2)
RANDOM_SEED = 7
SWEEP_COUNT = 5
POINTS_PER_SWEEP = 50
RANGE_SCALE_M = 40.0
# Distance past the reach of every sweep.
BEYOND_REACH_M = 50.0
BOX_EDGE_M = 2.0


def make_sweep(stem, returns_xyz, translation):
    """Sweep with identity rotation at ``translation``, no no-return beams."""
    pose = np.eye(4)
    pose[:3, 3] = translation
    return SessionSweep(stem=stem, split="train", timestamp_ns=0, returns_xyz=np.asarray(returns_xyz, dtype=float),
                        no_return_directions=np.zeros((0, 3)), lidar_to_map=pose)


def wall_sweep():
    """Sweep 1: at the origin, one return along +x at the wall depth."""
    return make_sweep("wall", [[WALL_DEPTH_M, 0.0, 0.0]], np.zeros(3))


def side_sweep():
    """Sweep 2: beside the cell, one return at the cell centre."""
    return make_sweep("side", [[0.0, SIDE_DEPTH_M, 0.0]], SIDE_ORIGIN)


def test_wall_hides_cell_until_second_view_sees_it():
    for worker_count in WORKER_COUNTS:
        wall_only = accumulate_session_evidence(
            [wall_sweep()], CELL_CENTRE, VOXEL_SIZE_M, RANGE_MARGIN_M, MAX_RANGE_M, worker_count)
        both = accumulate_session_evidence(
            [wall_sweep(), side_sweep()], CELL_CENTRE, VOXEL_SIZE_M, RANGE_MARGIN_M, MAX_RANGE_M, worker_count)
        # Wall alone: blocked, not seen -> hidden. With the side view: seen -> not hidden.
        assert hidden_space_mask(wall_only).tolist() == [True]
        assert hidden_space_mask(both).tolist() == [False]
        assert both[0, 0]


def test_worker_counts_agree():
    sweeps = [wall_sweep(), side_sweep()] * 3
    serial = accumulate_session_evidence(sweeps, CELL_CENTRE, VOXEL_SIZE_M, RANGE_MARGIN_M, MAX_RANGE_M, 1)
    parallel = accumulate_session_evidence(sweeps, CELL_CENTRE, VOXEL_SIZE_M, RANGE_MARGIN_M, MAX_RANGE_M, 2)
    assert np.array_equal(serial, parallel)


def test_cells_beyond_reach_get_no_evidence():
    far_centre = np.array([[MAX_RANGE_M + BEYOND_REACH_M, 0.0, 0.0]])
    evidence = accumulate_session_evidence([wall_sweep()], far_centre, VOXEL_SIZE_M, RANGE_MARGIN_M, MAX_RANGE_M, 1)
    assert not evidence.any()


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
    from viking_splatad.eval.session_hidden_space import (
        RANGE_HISTOGRAM_BIN_M, range_quantile_from_histogram, scan_sweeps)

    # Ten returns along +x at 1, 2, ..., 10 m (whole numbers of bins), split over two sweeps.
    ranges_m = np.arange(1, 11, dtype=np.float64)
    returns = [np.column_stack([ranges_m[:5], np.zeros(5), np.zeros(5)]),
               np.column_stack([ranges_m[5:], np.zeros(5), np.zeros(5)])]
    sweeps = [SessionSweep(stem=f"sweep{index}", split="train", timestamp_ns=index, returns_xyz=points,
                           no_return_directions=np.zeros((0, 3)), lidar_to_map=np.eye(4))
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


def test_near_band_hidden_mask_keeps_hidden_cells_within_the_band():
    """H_near = H AND horizontal distance to the nearest sweep origin <= band (z ignored)."""
    from viking_splatad.eval.session_hidden_space import near_band_hidden_mask, nearest_horizontal_distance

    band_m = 20.0
    sweep_origins = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 5.0]])
    # Cells at horizontal distances 3-4-5 = 5 m, exactly the band, just beyond it, and near the second origin.
    centres = np.array([[3.0, 4.0, 50.0], [0.0, band_m, 0.0], [0.0, band_m + 0.5, 0.0], [100.0, 7.0, -3.0]])
    expected_distances = np.array([5.0, band_m, band_m + 0.5, 7.0])
    distances = nearest_horizontal_distance(centres, sweep_origins)
    assert np.allclose(distances, expected_distances)
    hidden = np.array([True, True, True, False])
    assert near_band_hidden_mask(hidden, distances, band_m).tolist() == [True, True, False, False]
