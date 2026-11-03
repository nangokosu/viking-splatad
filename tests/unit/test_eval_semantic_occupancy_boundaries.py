"""Boundary tests of P1 semantic occupancy (T45, T46): thresholds, margins, halos, grids, visibility, voxel E2E.

Oracles are brute-force loops or hand-worked numbers, independent of the functions under test.
"""

import math

import numpy as np
import pytest

from tests.unit.test_eval_semantic_occupancy import (
    EVAL_SETTINGS,
    TRUNK_CLASS_ID,
    load_eval_script,
    make_box,
    write_fake_run,
    write_fake_session,
)
from viking_splatad.eval.detection import (
    cuboid_detection_rate,
    cuboid_voxel_mask,
    grow_cuboid_horizontally,
    median_half_footprint,
    n_min_from_real_counts,
    neighbourhood_point_precision,
    visible_from_any_pose,
)


# ---------------------------------------------------------------- T45 thresholds


@pytest.mark.covers("T45")
def test_headline_threshold_counts_boxes_with_exactly_n_min_points():
    """Counts [2, 3, 4], n_min = 3: the box holding exactly 3 detects (>=, not >) -> 2 / 3."""
    result = cuboid_detection_rate(np.array([2, 3, 4]), n_min=3)
    assert result.detection_rate == pytest.approx(2 / 3)
    # The curve at n = 3 is the same quantity: boxes with >= 3 points.
    assert result.rate_by_min_points[2] == pytest.approx(2 / 3)


@pytest.mark.covers("T45")
def test_every_box_exactly_at_threshold_is_detected():
    """All boxes hold exactly n_min = 5 points -> rate 1.0; n_min = 6 -> 0.0."""
    counts = np.array([5, 5, 5])
    assert cuboid_detection_rate(counts, n_min=5).detection_rate == 1.0
    assert cuboid_detection_rate(counts, n_min=6).detection_rate == 0.0


def test_n_min_quantile_returns_an_observed_count_below_the_interpolation():
    """Hit counts [1, 2, 3, 4, 5]: the element at floor(q (len - 1)) is returned.

    q = 0.7 -> floor(2.8) = index 2 -> 3 ("nearest" would give 4); q = 1.0 -> 5; q = 0 -> 1.
    Hit counts [1, 10], q = 0.5 -> floor(0.5) = index 0 -> 1 (linear interpolation would give 5.5).
    """
    hit_counts = np.array([1, 2, 3, 4, 5])
    assert n_min_from_real_counts(hit_counts, 0.7) == 3
    assert n_min_from_real_counts(hit_counts, 1.0) == 5
    assert n_min_from_real_counts(hit_counts, 0.0) == 1
    assert n_min_from_real_counts(np.array([1, 10]), 0.5) == 1


# ---------------------------------------------------------------- T46 margin and halo


def test_median_half_footprint_uses_both_horizontal_sizes():
    """(dx, dy) = (2, 6), (3, 1), (5, 5): mean widths 4, 2, 5 -> median 4 -> margin 2.

    Using only dx would give median 3 -> 1.5; only dy would give 5 -> 2.5.
    """
    boxes = [make_box(TRUNK_CLASS_ID, (0, 0, 0), dimensions) for dimensions in ((2, 6, 1), (3, 1, 1), (5, 5, 1))]
    assert median_half_footprint(boxes) == 2.0


def test_median_half_footprint_of_an_even_count_averages_the_middle_pair():
    """Widths 2 and 4 -> median 3 -> margin 1.5."""
    boxes = [make_box(TRUNK_CLASS_ID, (0, 0, 0), (width, width, 1.0)) for width in (2.0, 4.0)]
    assert median_half_footprint(boxes) == 1.5


def test_growth_adds_two_margins_to_x_and_y_and_none_to_z():
    """Box 1 x 3 x 2 with margin 0.5 -> 2 x 4 x 2, same centre."""
    grown = grow_cuboid_horizontally(make_box(TRUNK_CLASS_ID, (1.0, 2.0, 3.0), (1.0, 3.0, 2.0)), 0.5)
    assert grown.dimensions_xyz == (2.0, 4.0, 2.0)
    assert grown.center_xyz == (1.0, 2.0, 3.0)


@pytest.mark.covers("T46")
def test_halo_along_x_and_on_the_diagonal_counts_and_beyond_does_not():
    """Box 1 x 1 x 2 at the origin, margin 0.5: neighbourhood x, y in [-1, 1] (half size 0.5 + 0.5).

    Points (all at z = 1, mid-height): inside (0, 0); x halo (0.9, 0) in and (1.1, 0) out;
    diagonal halo (0.9, 0.9) in (corner of the grown square) and (1.1, 1.1) out; y halo (0, -0.9) in.
    Neighbourhood holds (0, 0), (0.9, 0), (0.9, 0.9) and (0, -0.9) -> 4 points; only (0, 0) is inside the box -> 1 / 4.
    """
    trunk = make_box(TRUNK_CLASS_ID, (0.0, 0.0, 1.0), (1.0, 1.0, 2.0))
    points = np.array([
        [0.0, 0.0, 1.0], [0.9, 0.0, 1.0], [1.1, 0.0, 1.0],
        [0.9, 0.9, 1.0], [1.1, 1.1, 1.0], [0.0, -0.9, 1.0],
    ])
    result = neighbourhood_point_precision(points, [trunk], margin=0.5)
    assert (result.inside_count, result.neighbourhood_count, result.precision) == (1, 4, 0.25)


@pytest.mark.covers("T46")
def test_halo_of_a_non_square_box_follows_each_axis_and_the_yaw():
    """Box 4 x 1 x 2 (x long), margin 0.5: grown to 5 x 2, i.e. |x| <= 2.5 and |y| <= 1.0.

    (2.4, 0) in; (2.6, 0) out; (0, 0.9) in; (0, 1.1) out. Yawed 90 deg the long side is along y, so
    (0, 2.4) is in and (2.4, 0) is out.
    """
    unrotated = make_box(TRUNK_CLASS_ID, (0.0, 0.0, 1.0), (4.0, 1.0, 2.0))
    points = np.array([[2.4, 0.0, 1.0], [2.6, 0.0, 1.0], [0.0, 0.9, 1.0], [0.0, 1.1, 1.0], [0.0, 2.4, 1.0]])
    # Unrotated: (2.4, 0) and (0, 0.9) are in the neighbourhood; the three others are not.
    assert neighbourhood_point_precision(points, [unrotated], 0.5).neighbourhood_count == 2
    rotated = make_box(TRUNK_CLASS_ID, (0.0, 0.0, 1.0), (4.0, 1.0, 2.0), yaw_radians=math.pi / 2)
    # Rotated: footprint is 1 (x) by 4 (y); grown to 2 by 5. (0, 2.4) in, (0, 0.9) in, (2.4, 0) out,
    # (0, 1.1) in, (2.6, 0) out.
    rotated_result = neighbourhood_point_precision(points, [rotated], 0.5)
    assert rotated_result.neighbourhood_count == 3


# ---------------------------------------------------------------- cuboid_voxel_mask


def brute_force_voxel_mask(box_center, box_sizes, yaw_radians, grid_origin, voxel_size, grid_shape):
    """Voxels whose centre is inside a yawed box, by looping over every voxel (independent oracle).

    Args:
        box_center: (3,) box centre.
        box_sizes: (3,) full sizes.
        yaw_radians: Yaw about +z.
        grid_origin: (3,) grid minimum corner.
        voxel_size: Voxel edge.
        grid_shape: (nx, ny, nz).

    Returns:
        Set of (ix, iy, iz) index tuples.
    """
    cosine, sine = math.cos(yaw_radians), math.sin(yaw_radians)
    inside = set()
    for index_x in range(grid_shape[0]):
        for index_y in range(grid_shape[1]):
            for index_z in range(grid_shape[2]):
                centre = [grid_origin[axis] + (index + 0.5) * voxel_size
                          for axis, index in enumerate((index_x, index_y, index_z))]
                offset = [centre[axis] - box_center[axis] for axis in range(3)]
                # Rotate the offset into the box frame: R(-yaw) applied to (dx, dy).
                local_x = cosine * offset[0] + sine * offset[1]
                local_y = -sine * offset[0] + cosine * offset[1]
                if (abs(local_x) <= box_sizes[0] / 2 and abs(local_y) <= box_sizes[1] / 2
                        and abs(offset[2]) <= box_sizes[2] / 2):
                    inside.add((index_x, index_y, index_z))
    return inside


def test_voxel_mask_on_a_non_cubic_grid_with_an_off_centre_box():
    """Grid origin (-1, 2, 0.5), shape (6, 4, 3), voxel 0.5; box centre (0.6, 3.3, 1.2), sizes 1.0 x 0.8 x 0.6.

    Centres: x -0.75, -0.25, 0.25, 0.75, 1.25, 1.75; y 2.25, 2.75, 3.25, 3.75; z 0.75, 1.25, 1.75.
    Box x in [0.1, 1.1] -> indices 2, 3; y in [2.9, 3.7] -> index 2; z in [0.9, 1.5] -> index 1.
    """
    origin, grid_shape = np.array([-1.0, 2.0, 0.5]), (6, 4, 3)
    box = make_box(TRUNK_CLASS_ID, (0.6, 3.3, 1.2), (1.0, 0.8, 0.6))
    mask = cuboid_voxel_mask([box], origin, 0.5, grid_shape)
    assert mask.shape == grid_shape
    assert set(map(tuple, np.argwhere(mask))) == {(2, 2, 1), (3, 2, 1)}


@pytest.mark.parametrize("yaw_degrees", [0.0, 90.0, 30.0])
def test_voxel_mask_matches_brute_force_on_a_non_cubic_grid_for_yawed_boxes(yaw_degrees):
    """Same grid and box as above, yawed: the set equals the brute-force loop over every voxel."""
    origin, grid_shape = np.array([-1.0, 2.0, 0.5]), (6, 4, 3)
    box = make_box(TRUNK_CLASS_ID, (0.6, 3.3, 1.2), (1.0, 0.8, 0.6), math.radians(yaw_degrees))
    expected = brute_force_voxel_mask((0.6, 3.3, 1.2), (1.0, 0.8, 0.6), math.radians(yaw_degrees), origin, 0.5, grid_shape)
    mask = cuboid_voxel_mask([box], origin, 0.5, grid_shape)
    assert set(map(tuple, np.argwhere(mask))) == expected
    # A 90-degree box swaps its x and y extents: x [0.2, 1.0] and y [2.8, 3.8] -> 2 x 2 voxels.
    if yaw_degrees == 90.0:
        assert len(expected) == 4


def test_voxel_mask_ignores_boxes_outside_the_grid():
    """A box entirely beyond the grid marks nothing and does not raise."""
    box = make_box(TRUNK_CLASS_ID, (100.0, 100.0, 100.0), (1.0, 1.0, 1.0))
    assert not cuboid_voxel_mask([box], np.zeros(3), 0.5, (6, 4, 3)).any()


# ---------------------------------------------------------------- visible_from_any_pose


def cyclic_pose():
    """Pose with R mapping sensor x -> map y, sensor y -> map z, sensor z -> map x, translation (1, 2, 3)."""
    pose = np.eye(4)
    pose[:3, :3] = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    pose[:3, 3] = [1.0, 2.0, 3.0]
    return pose


def test_visibility_with_a_rotated_translated_pose_and_a_point_at_exactly_max_range():
    """Sensor-frame points are mapped to the map as R s + t with the cyclic R above.

    sensor (10, 0, 0) -> map (1, 12, 3): range exactly 10 = max_range, elevation 0 -> visible.
    sensor (10.5, 0, 0) -> map (1, 12.5, 3): beyond the range -> not visible.
    sensor (0, 0, 10) -> map (11, 2, 3): straight up in the sensor frame (90 deg > 0.1 rad) -> not visible.
    An implementation using R instead of R^T, or skipping the rotation, would flip these.
    """
    map_points = np.array([[1.0, 12.0, 3.0], [1.0, 12.5, 3.0], [11.0, 2.0, 3.0]])
    visible = visible_from_any_pose(map_points, [cyclic_pose()], max_range=10.0, vertical_half_fov=0.1)
    np.testing.assert_array_equal(visible, [True, False, False])


def test_visibility_is_the_union_over_poses():
    """A point outside the first pose's range but inside the second pose's range is visible."""
    far_pose = np.eye(4)
    far_pose[:3, 3] = [100.0, 0.0, 0.0]
    near_pose = np.eye(4)
    visible = visible_from_any_pose(np.array([[101.0, 0.0, 0.0], [50.0, 0.0, 0.0]]), [near_pose, far_pose], 5.0, 0.5)
    np.testing.assert_array_equal(visible, [True, False])


# ---------------------------------------------------------------- scripts/13 voxel metrics end to end


def test_script_voxel_iou_precision_recall_match_a_hand_derived_grid(tmp_path):
    """Voxel metrics of the fake run, derived from the geometry with plain loops.

    Grid origin (0, -5, -1), voxel 0.5, shape (40, 20, 10); GT trunk voxels x 19..20, y 9..10, z 2..5.
    Lidar at (0, 0, 1); rays at elevation 0 land on map z = 1 (voxel z index 4). Ungated points are
    (10 cos a, 10 sin a) for a = 0, 1, 2, -1 deg and (10, 0.8): voxels (20, 10), (19, 10), (19, 10), (19, 9), (20, 11).
    Gated = rays 0, 1 and the halo ray: voxels (20, 10), (19, 10), (20, 11).
    Voxels at x index 20 (centre x 10.25) lie beyond the farthest real return and are not scored.
    Neighbourhood: margin 0.5 -> grown box x [9, 11], y [-1, 1], z [0, 2], restricted to voxel centres
    within the farthest real return (sqrt(10.2^2 + 0.1^2)) of the lidar (the vertical FOV, 0.5 rad, never binds).
    """
    eval_script = load_eval_script()
    manifest_sha256 = write_fake_session(tmp_path, "may_x", "2024_05", "may_x_smoke")
    run_dir = write_fake_run(tmp_path, "run-may", "may_x_smoke", manifest_sha256)
    _, report = eval_script.evaluate_run(run_dir, None, EVAL_SETTINGS, None, tmp_path)

    # Step 1: neighbourhood and GT voxels by brute force.
    voxel_size, origin, lidar_position = 0.5, (0.0, -5.0, -1.0), (0.0, 0.0, 1.0)
    max_range = math.hypot(math.hypot(10.2, 0.1), 0.0)
    neighbourhood, ground_truth = set(), set()
    for index_x in range(40):
        for index_y in range(20):
            for index_z in range(10):
                centre = [origin[axis] + (index + 0.5) * voxel_size for axis, index in enumerate((index_x, index_y, index_z))]
                distance = math.dist(centre, lidar_position)
                in_grown_box = 9.0 <= centre[0] <= 11.0 and -1.0 <= centre[1] <= 1.0 and 0.0 <= centre[2] <= 2.0
                if in_grown_box and distance <= max_range:
                    neighbourhood.add((index_x, index_y, index_z))
                    if 19 <= index_x <= 20 and 9 <= index_y <= 10 and 2 <= index_z <= 5:
                        ground_truth.add((index_x, index_y, index_z))
    # Sanity of the oracle: every voxel at x index 20 has centre x = 10.25 > 10.2005, hence out of range;
    # only the 2 (y) x 4 (z) GT voxels at x index 19 (centre 9.75, farthest corner ~9.8) remain.
    assert ground_truth == {(19, y, z) for y in (9, 10) for z in range(2, 6)}

    # Step 2: metrics of each point set.
    occupied_by_set = {
        "ungated": {(20, 10, 4), (19, 10, 4), (19, 9, 4), (20, 11, 4)},
        "gated": {(20, 10, 4), (19, 10, 4), (20, 11, 4)},
    }
    for set_name, occupied in occupied_by_set.items():
        occupied_in_region = occupied & neighbourhood
        true_positives = len(occupied_in_region & ground_truth)
        false_positives = len(occupied_in_region - ground_truth)
        false_negatives = len(ground_truth - occupied_in_region)
        voxel = report["point_sets"][set_name]["classes"]["tree_trunk"]["voxel"]
        assert (voxel["true_positives"], voxel["false_positives"], voxel["false_negatives"]) == (
            true_positives, false_positives, false_negatives), set_name
        assert voxel["iou"] == pytest.approx(true_positives / (true_positives + false_positives + false_negatives))
        assert voxel["precision"] == pytest.approx(true_positives / (true_positives + false_positives))
        assert voxel["recall"] == pytest.approx(true_positives / len(ground_truth))
    # Hand-checked headline. Ungated: occupied voxels inside the region are (19, 10, 4) and (19, 9, 4)
    # -> TP 2, FP 0, FN 8 - 2 = 6. Gated: only (19, 10, 4) -> TP 1, FN 7. The voxels at x index 20 are
    # out of range and so not scored (neither as hit nor as false positive).
    ungated_voxel = report["point_sets"]["ungated"]["classes"]["tree_trunk"]["voxel"]
    gated_voxel = report["point_sets"]["gated"]["classes"]["tree_trunk"]["voxel"]
    assert (ungated_voxel["true_positives"], ungated_voxel["false_positives"], ungated_voxel["false_negatives"]) == (2, 0, 6)
    assert (gated_voxel["true_positives"], gated_voxel["false_positives"], gated_voxel["false_negatives"]) == (1, 0, 7)
    assert ungated_voxel["iou"] == pytest.approx(0.25) and ungated_voxel["recall"] == pytest.approx(0.25)
    assert ungated_voxel["precision"] == pytest.approx(1.0)
