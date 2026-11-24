"""Known-answer tests for the lidar range margin (T92)."""

import numpy as np
import pytest

from viking_splatad.eval.range_margin import (
    cross_sweep_plane_residuals,
    points_within_range,
    range_margin,
    sweep_pair_indices,
)

GRID_SIDE = 21
GRID_SPACING = 0.1
OFFSET_DELTA = 0.013
NEIGHBOUR_COUNT = 8


def _flat_grid() -> np.ndarray:
    """Dense grid on z = 0, shape (GRID_SIDE^2, 3)."""
    axis_values = np.arange(GRID_SIDE) * GRID_SPACING
    grid_x, grid_y = np.meshgrid(axis_values, axis_values)
    return np.stack([grid_x.ravel(), grid_y.ravel(), np.zeros(grid_x.size)], axis=1)


@pytest.mark.covers("T92")
def test_residual_equals_offset_above_flat_grid():
    """Points at z = +/-delta over a z = 0 grid have residual delta exactly."""
    grid_points = _flat_grid()
    # Query points between grid nodes, away from the border, alternating sign.
    query_xy = np.array([[0.55, 0.55], [0.95, 1.05], [1.25, 0.75], [0.65, 1.15]])
    signs = np.array([1.0, -1.0, 1.0, -1.0])
    query_points = np.column_stack([query_xy, signs * OFFSET_DELTA])
    residuals = cross_sweep_plane_residuals(query_points, grid_points, NEIGHBOUR_COUNT, 0.0)
    assert residuals.shape == (len(query_points),)
    assert residuals.dtype == np.float64
    np.testing.assert_allclose(residuals, OFFSET_DELTA, atol=1e-12)


@pytest.mark.covers("T92")
def test_isotropic_cluster_rejected_by_planarity():
    """Cube corners are isotropic (planarity 0): a planarity threshold rejects the point."""
    cube_corners = np.array([[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)])
    centred = cube_corners - cube_corners.mean(axis=0)
    eigenvalues = np.sort(np.linalg.eigvalsh(centred.T @ centred / len(cube_corners)))[::-1]
    expected_planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
    assert expected_planarity < 1e-9
    query_point = np.array([[0.0, 0.0, 0.3]])
    kept = cross_sweep_plane_residuals(query_point, cube_corners, len(cube_corners), 0.5)
    assert kept.size == 0
    # With the threshold at the (zero) planarity the point is kept.
    permissive = cross_sweep_plane_residuals(query_point, cube_corners, len(cube_corners), 0.0)
    assert permissive.size == 1


@pytest.mark.covers("T92")
def test_too_few_reference_points_returns_empty():
    """Fewer B points than neighbour_count gives an empty array."""
    result = cross_sweep_plane_residuals(np.zeros((2, 3)), np.zeros((3, 3)), NEIGHBOUR_COUNT, 0.0)
    assert result.shape == (0,)


def test_range_margin_linear_quantile():
    """Linear quantile of [1..5] at 0.9 is 1 + 0.9 * (5 - 1) = 4.6."""
    residuals = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    assert range_margin(residuals, 0.9) == pytest.approx(1.0 + 0.9 * (5.0 - 1.0))


def test_sweep_pair_indices():
    """(10 sweeps, stride 3, gap 2) -> k = 0, 3, 6 (k = 9 has no partner)."""
    assert sweep_pair_indices(10, 3, 2) == [(0, 2), (3, 5), (6, 8)]


def test_points_within_range():
    """3-4-5 triangle: norm 5 is inside max_range 5, norm 6 outside."""
    points = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 6.0]])
    np.testing.assert_array_equal(points_within_range(points, 5.0), [True, False])


@pytest.mark.covers("T92")
def test_value_errors():
    """Invalid shapes and parameters raise ValueError."""
    good = np.zeros((4, 3))
    with pytest.raises(ValueError):
        cross_sweep_plane_residuals(np.zeros((4, 2)), good, 3, 0.5)
    with pytest.raises(ValueError):
        cross_sweep_plane_residuals(good, good, 2, 0.5)
    with pytest.raises(ValueError):
        cross_sweep_plane_residuals(good, good, 3, 1.5)
    with pytest.raises(ValueError):
        range_margin(np.array([]), 0.5)
    with pytest.raises(ValueError):
        range_margin(np.array([1.0]), 1.0)
    with pytest.raises(ValueError):
        sweep_pair_indices(10, 0, 2)
    with pytest.raises(ValueError):
        sweep_pair_indices(10, 2, 0)
    with pytest.raises(ValueError):
        points_within_range(np.zeros((3, 2)), 1.0)
