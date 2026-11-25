"""Known-answer tests for the figure grid helpers (synthetic inputs, expectations derived by hand)."""

import numpy as np
import pytest

from viking_splatad.eval.detection import voxel_centers
from viking_splatad.eval.hidden_geometry import EVIDENCE_NAMES
from viking_splatad.viz.figure_grids import (
    NO_OUTCOME_LABEL, OUTCOME_STRATUM_NAMES, arm_difference_sets, column_max_grid,
    opacity_weighted_density, outcome_strata)

VOXEL_SIZE = 0.5
GRID_ORIGIN = np.array([10.0, -4.0, 1.0])


def test_column_max_grid_values_nan_and_extent():
    """3x3x2 grid: two z-layers per column, one empty column, one masked-out cell."""
    # Cells at columns (x,y): (0,0) z0,z1; (2,0); (1,1); (0,2) z0 masked out; (2,2); (1,2) absent.
    cell_indices = np.array([[0, 0, 0], [0, 0, 1], [2, 0, 0], [1, 1, 1], [0, 2, 0], [2, 2, 1]])
    values = np.array([0.2, 0.9, 0.4, 0.7, 5.0, 0.1])
    mask = np.array([True, True, True, True, False, True])
    image, extent = column_max_grid(cell_indices, values, mask, GRID_ORIGIN, VOXEL_SIZE)
    expected = np.full((3, 3), np.nan)  # [y_row, x_col]
    expected[0, 0] = max(0.2, 0.9)       # two z-layers: max wins
    expected[0, 2] = 0.4
    expected[1, 1] = 0.7
    expected[2, 2] = 0.1                 # the masked-out 5.0 at (0,2) must not appear
    np.testing.assert_array_equal(np.isnan(image), np.isnan(expected))
    np.testing.assert_allclose(image[~np.isnan(image)], expected[~np.isnan(expected)])
    # Extent: min corner of voxel 0 to max corner of voxel 2, per axis.
    assert extent == pytest.approx((GRID_ORIGIN[0], GRID_ORIGIN[0] + 3 * VOXEL_SIZE,
                                    GRID_ORIGIN[1], GRID_ORIGIN[1] + 3 * VOXEL_SIZE))


def test_column_max_grid_extent_matches_voxel_centres_convention():
    """A single cell's pixel must be centred on the repo's voxel centre."""
    cell_indices = np.array([[3, 5, 0]])
    image, extent = column_max_grid(cell_indices, np.array([1.0]), np.array([True]), GRID_ORIGIN, VOXEL_SIZE)
    centre = voxel_centers(cell_indices, GRID_ORIGIN, VOXEL_SIZE)[0]
    assert image.shape == (1, 1)
    assert (extent[0] + extent[1]) / 2 == pytest.approx(centre[0])
    assert (extent[2] + extent[3]) / 2 == pytest.approx(centre[1])


def test_column_max_grid_range_follows_mask_only():
    """Cells outside the mask do not widen the image."""
    cell_indices = np.array([[1, 1, 0], [9, 9, 0]])
    image, _ = column_max_grid(cell_indices, np.array([1.0, 2.0]), np.array([True, False]), GRID_ORIGIN, VOXEL_SIZE)
    assert image.shape == (1, 1)


def test_column_max_grid_empty_mask_raises():
    with pytest.raises(ValueError):
        column_max_grid(np.zeros((2, 3), dtype=int), np.zeros(2), np.zeros(2, dtype=bool), GRID_ORIGIN, VOXEL_SIZE)


def test_density_mass_and_bin_placement():
    """Points on exact bin centres/edges; the outside point is dropped; mass = sum of inside opacities."""
    extent = (0.0, 2.0, 10.0, 11.0)
    bin_size = 0.5
    means_xy = np.array([[0.25, 10.25], [0.25, 10.25], [1.75, 10.75], [2.0, 11.0], [5.0, 10.5], [0.5, 9.0]])
    opacities = np.array([0.5, 0.25, 1.0, 0.125, 0.9, 0.9])
    density, edges_x, edges_y = opacity_weighted_density(means_xy, opacities, extent, bin_size)
    # Inside: first four points (the point on the upper corner joins the last bin, as in np.histogram).
    expected_mass = 0.5 + 0.25 + 1.0 + 0.125
    assert density.sum() == pytest.approx(expected_mass)
    assert density.shape == (2, 4)  # [y, x]: span 1 m / 0.5 = 2 rows, span 2 m / 0.5 = 4 columns
    assert edges_x[0] == extent[0] and edges_x[-1] == pytest.approx(extent[1])
    assert edges_y[0] == extent[2] and edges_y[-1] == pytest.approx(extent[3])
    assert density[0, 0] == pytest.approx(0.75)
    assert density[1, 3] == pytest.approx(1.0 + 0.125)


def test_arm_difference_sets_partition():
    first = np.array([True, True, False, False])
    second = np.array([True, False, True, False])
    sets = arm_difference_sets(first, second)
    np.testing.assert_array_equal(sets["only_first"], [False, True, False, False])
    np.testing.assert_array_equal(sets["only_second"], [False, False, True, False])
    np.testing.assert_array_equal(sets["both"], [True, False, False, False])
    # Partition: disjoint and union equals first | second.
    total = sets["only_first"].astype(int) + sets["only_second"] + sets["both"]
    np.testing.assert_array_equal(total, (first | second).astype(int))


def test_outcome_strata_five_cells():
    """Cells: hit seen, hit passed-through, false alarm, missed seen, missed unobserved; plus one in no outcome."""
    evidence = np.zeros((6, 4), dtype=bool)
    evidence[0, EVIDENCE_NAMES.index("seen")] = True
    evidence[1, EVIDENCE_NAMES.index("passed_through")] = True
    evidence[3, EVIDENCE_NAMES.index("seen")] = True
    # Cell 4 has no flag (unobserved); cell 2 evidence is irrelevant for false alarms; cell 5 is in no outcome.
    outcomes = {
        "hit": np.array([True, True, False, False, False, False]),
        "false_alarm": np.array([False, False, True, False, False, False]),
        "missed": np.array([False, False, False, True, True, False]),
    }
    labels = outcome_strata(outcomes, evidence)
    expected = [OUTCOME_STRATUM_NAMES.index(name) for name in
                ("hit_seen", "hit_hidden", "false_alarm", "missed_seen", "missed_hidden")]
    np.testing.assert_array_equal(labels, expected + [NO_OUTCOME_LABEL])
