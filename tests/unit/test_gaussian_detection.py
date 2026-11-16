"""Known-answer tests for P2 cuboid detection (T79) and present-but-hidden counts (T80)."""

import numpy as np
import pytest

from viking_splatad.eval.gaussian_detection import (
    cuboid_occupied_fractions, detection_curve, f_min_grid, present_but_hidden, threshold_sweep,
)
from viking_splatad.eval.hidden_geometry import EVIDENCE_NAMES
from viking_splatad.labels.cuboids import DEFAULT_BOUNDARY_TOLERANCE_METERS, Cuboid

# Synthetic scene sizes: chosen so the yawed box covers part of the grid and
# leaves cells both inside and outside it; any such values would do.
GRID_SPACING = 0.3
GRID_OFFSET = 0.1
GRID_HALF_SPAN = 1.5
BOX_YAW = np.pi / 6
BOX_CENTER = (0.2, -0.1, 0.0)
BOX_DIMENSIONS = (2.0, 1.0, 1.0)
# Second layer above the box (box half height is 0.5), so all of it is outside.
OUTSIDE_LAYER_Z = 2.0
# Far-away box that contains no grid cell: its fraction must be undefined.
EMPTY_BOX_CENTER = (50.0, 50.0, 0.0)
# Hand-set patterns: every TRUTH_GAP-th cell is not GT; every OCCUPIED_STRIDE-th is occupied
# (odd, so it alternates across the two z layers interleaved in the flattened grid).
TRUTH_GAP = 5
OCCUPIED_STRIDE = 3


def yaw_quaternion_xyzw(yaw):
    """Return the unit quaternion (x, y, z, w) of a rotation by yaw about map z.

    q = (0, 0, sin(yaw/2), cos(yaw/2)) by the axis-angle formula.

    Args:
        yaw: Rotation angle in radians.
    Returns:
        Tuple (x, y, z, w).
    """
    return (0.0, 0.0, float(np.sin(yaw / 2)), float(np.cos(yaw / 2)))


def synthetic_grid():
    """Return [V,3] centres: a regular xy grid on two z layers (z=0 and above the box).

    Returns:
        numpy.ndarray [V,3].
    """
    axis_values = np.arange(-GRID_HALF_SPAN, GRID_HALF_SPAN, GRID_SPACING) + GRID_OFFSET
    grid_x, grid_y, grid_z = np.meshgrid(axis_values, axis_values, [0.0, OUTSIDE_LAYER_Z], indexing="ij")
    return np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)


@pytest.mark.covers("T79")
def test_yawed_cuboid_fraction_matches_its_analytic_extent_and_empty_box_is_undefined():
    """Inside cells follow the analytic yawed box; a box with no GT cell gives NaN."""
    centres = synthetic_grid()
    cell_indices = np.arange(len(centres))
    truth = cell_indices % TRUTH_GAP != 0
    occupied = cell_indices % OCCUPIED_STRIDE == 0
    cuboids = [Cuboid("box", 0, "trunk", BOX_CENTER, BOX_DIMENSIONS, yaw_quaternion_xyzw(BOX_YAW)),
               Cuboid("empty", 0, "trunk", EMPTY_BOX_CENTER, BOX_DIMENSIONS, yaw_quaternion_xyzw(BOX_YAW))]
    # First principles: p_box = Rz(yaw)^T (p - c); inside iff |p_box| <= dims/2 on every axis.
    cosine, sine = np.cos(BOX_YAW), np.sin(BOX_YAW)
    rotation_map_box = np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])
    local_points = (centres - np.asarray(BOX_CENTER)) @ rotation_map_box
    half_extents = np.asarray(BOX_DIMENSIONS) / 2
    # No cell sits on a face, so the boundary tolerance cannot change the answer.
    assert np.min(np.abs(np.abs(local_points) - half_extents)) > DEFAULT_BOUNDARY_TOLERANCE_METERS
    inside = np.all(np.abs(local_points) <= half_extents, axis=1)
    expected_gt = int(np.count_nonzero(inside & truth))
    expected_occupied = int(np.count_nonzero(inside & truth & occupied))
    # The scene must exercise both a non-trivial inside set and a partial fraction.
    assert 0 < expected_occupied < expected_gt < len(centres)

    result = cuboid_occupied_fractions(cuboids, centres, truth, occupied)

    np.testing.assert_array_equal(result["gt_cells"], [expected_gt, 0])
    np.testing.assert_array_equal(result["occupied_cells"], [expected_occupied, 0])
    assert result["fraction"][0] == pytest.approx(expected_occupied / expected_gt)
    assert np.isnan(result["fraction"][1])


@pytest.mark.covers("T79")
def test_cuboid_fraction_rejects_non_boolean_or_mismatched_masks():
    """Probability-valued or wrong-length masks are refused at the boundary."""
    centres = synthetic_grid()
    cuboids = [Cuboid("box", 0, "trunk", BOX_CENTER, BOX_DIMENSIONS, yaw_quaternion_xyzw(BOX_YAW))]
    truth = np.ones(len(centres), dtype=bool)
    with pytest.raises(ValueError, match="occupied"):
        cuboid_occupied_fractions(cuboids, centres, truth, np.ones(len(centres)))
    with pytest.raises(ValueError, match="truth"):
        cuboid_occupied_fractions(cuboids, centres, truth[:-1], truth)


def test_detection_curve_counts_defined_cuboids_and_reports_undefined():
    """Rates are counts over defined cuboids only; the NaN cuboid is reported, not scored."""
    point_count = 5
    fractions = np.array([0.0, 0.25, 0.5, 1.0, np.nan])
    grid = f_min_grid(point_count)
    # Hand count of fractions >= f_min for f_min = 0, 1/4, 1/2, 3/4, 1 among 4 defined cuboids.
    expected_detected = [4, 3, 2, 1, 1]
    defined_count = 4

    curve = detection_curve(fractions, grid)

    assert curve["defined_cuboids"] == defined_count
    assert curve["undefined_cuboids"] == 1
    assert curve["detected_count"] == expected_detected
    np.testing.assert_allclose(curve["detection_rate"], np.array(expected_detected) / defined_count)
    np.testing.assert_allclose(curve["f_min"], grid)


def test_detection_curve_with_no_defined_cuboid_has_no_rate():
    """All-undefined input gives None rates rather than a division by zero."""
    curve = detection_curve(np.array([np.nan, np.nan]), f_min_grid(3))
    assert curve["detection_rate"] == [None, None, None]
    assert curve["defined_cuboids"] == 0 and curve["undefined_cuboids"] == 2


def test_f_min_grid_spans_unit_interval_evenly_and_validates_count():
    """Spacing is 1/(n-1) with both ends included; n < 2 or a bool is refused."""
    point_count = 11
    grid = f_min_grid(point_count)
    assert grid[0] == 0.0 and grid[-1] == 1.0 and len(grid) == point_count
    np.testing.assert_allclose(np.diff(grid), 1.0 / (point_count - 1))
    with pytest.raises(ValueError):
        f_min_grid(1)
    with pytest.raises(ValueError):
        f_min_grid(True)


@pytest.mark.covers("T80")
def test_present_but_hidden_counts_per_exclusive_evidence_stratum():
    """Hand-made cells: hidden = GT & P2 & ~P1, each GT cell in exactly one stratum."""
    # Columns: seen, blocked, no_return, passed_through. Row 3 is a conflict (seen wins).
    evidence = np.array([
        [True, False, False, False],   # cell 0: seen
        [False, True, False, False],   # cell 1: blocked
        [False, True, False, False],   # cell 2: blocked
        [True, True, False, False],    # cell 3: seen and blocked -> seen
        [False, False, True, False],   # cell 4: no_return
        [False, False, False, True],   # cell 5: passed_through
        [False, False, False, False],  # cell 6: unobserved
        [False, True, False, False],   # cell 7: blocked, but not GT
    ])
    truth = np.array([True, True, True, True, True, True, True, False])
    p2_occupied = np.array([True, True, True, False, True, False, True, True])
    p1_hit = np.array([True, False, True, False, False, False, False, False])
    # Expected per stratum, from the row comments: GT cells, GT&P2, GT&P1, GT&P2&~P1.
    expected_strata = {
        "seen": {"gt_cells": 2, "p2_occupied": 1, "p1_hit": 1, "present_but_hidden": 0},
        "blocked": {"gt_cells": 2, "p2_occupied": 2, "p1_hit": 1, "present_but_hidden": 1},
        "no_return": {"gt_cells": 1, "p2_occupied": 1, "p1_hit": 0, "present_but_hidden": 1},
        "passed_through": {"gt_cells": 1, "p2_occupied": 0, "p1_hit": 0, "present_but_hidden": 0},
        "unobserved": {"gt_cells": 1, "p2_occupied": 1, "p1_hit": 0, "present_but_hidden": 1},
    }
    # Totals are the stratum sums because the strata partition the GT cells.
    expected_totals = {key: sum(stratum[key] for stratum in expected_strata.values())
                       for key in expected_strata["seen"]}

    result = present_but_hidden(p2_occupied, p1_hit, truth, evidence)

    assert set(result["strata"]) == set(EVIDENCE_NAMES)
    assert result["strata"] == expected_strata
    assert result["totals"] == expected_totals
    assert result["totals"]["present_but_hidden"] == int(np.count_nonzero(truth & p2_occupied & ~p1_hit))
    with pytest.raises(ValueError, match="evidence"):
        present_but_hidden(p2_occupied, p1_hit, truth, evidence[:, :3])


def test_threshold_sweep_recall_is_monotone_and_matches_counts():
    """occupied = p >= t; recall over 4 GT cells falls as t rises."""
    probabilities = np.array([0.9, 0.6, 0.3, 0.1, 0.8, 0.05])
    truth = np.array([True, True, True, True, False, False])
    evidence = np.zeros((len(truth), 4), dtype=bool)
    thresholds = np.array([0.0, 0.2, 0.5, 0.7, 0.95])
    gt_count = int(np.count_nonzero(truth))
    # Hand count of GT cells with p >= t: t=0 -> 4, 0.2 -> 3, 0.5 -> 2, 0.7 -> 1, 0.95 -> 0.
    expected_recall = np.array([4, 3, 2, 1, 0]) / gt_count

    sweep = threshold_sweep(probabilities, truth, evidence, thresholds)

    recalls = np.array([entry["metrics"]["recall"] for entry in sweep])
    np.testing.assert_allclose(recalls, expected_recall)
    assert np.all(np.diff(recalls) <= 0)
    assert [entry["threshold"] for entry in sweep] == list(thresholds)
    # All GT cells are unobserved, so that stratum's recall equals the overall recall.
    np.testing.assert_allclose([entry["metrics"]["strata"]["unobserved"]["recall"] for entry in sweep],
                               expected_recall)


def test_exact_ratio_on_a_grid_value_is_detected():
    """3 of 10 cells occupied is exactly f_min = 3/10 on the 11-point grid, so it counts as detected there
    (numpy.linspace's 0.30000000000000004 would have rejected it)."""
    grid = f_min_grid(11)
    occupied_cells, gt_cells = 3, 10
    curve = detection_curve(np.array([occupied_cells / gt_cells]), grid)
    grid_index_of_fraction = 3
    assert curve["detected_count"][grid_index_of_fraction] == 1
    assert curve["detected_count"][grid_index_of_fraction + 1] == 0
