"""First-principles P2 probability and recorded observation evidence cases."""

import numpy as np
import pytest

from viking_splatad.eval.hidden_geometry import (
    exclusive_evidence_labels, gaussian_occupancy_at_centres, recorded_ray_evidence, occupied_cell_metrics,
)


@pytest.mark.covers("T57")
def test_product_occupancy_overlap_and_inclusive_three_sigma_boundary():
    """Two coincident half-opaque Gaussians give 3/4, not one; 3 sigma is included."""
    means = np.zeros((2, 3))
    covariances = np.repeat(np.eye(3)[None], len(means), axis=0)
    centres = np.array([[0., 0, 0], [3., 0, 0], [4., 0, 0]])
    alpha = 0.5 * np.exp(-0.5 * 3 ** 2)
    expected = [1 - (1 - 0.5) ** 2, 1 - (1 - alpha) ** 2, 0]
    actual = gaussian_occupancy_at_centres(centres, means, covariances, np.full(2, 0.5), 1)
    np.testing.assert_allclose(actual, expected)
    np.testing.assert_array_equal(actual, gaussian_occupancy_at_centres(centres, means, covariances, np.full(2, 0.5), 8))


@pytest.mark.covers("T57")
def test_rotated_anisotropic_gaussian_and_exact_unit_opacity():
    """A quarter-turn exchanges long and short axes; zero survival remains exact."""
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
    covariance = rotation @ np.diag([4., 1., 1.]) @ rotation.T
    centres = np.array([[0., 0, 0], [0., 6, 0], [6., 0, 0]])
    actual = gaussian_occupancy_at_centres(centres, np.zeros((1, 3)), covariance[None], np.ones(1), 2)
    np.testing.assert_allclose(actual, [1, np.exp(-0.5 * (6 / 2) ** 2), 0])
    with pytest.raises(ValueError, match="positive-definite"):
        gaussian_occupancy_at_centres(centres, np.zeros((1, 3)), np.zeros((1, 3, 3)), np.ones(1), 2)


@pytest.mark.covers("T78")
def test_recorded_beams_distinguish_seen_blocked_no_return_passed_and_unobserved():
    """Only beams crossing each cell supply evidence; empty directions are not no-returns."""
    centres = np.array([[2., 0, 0], [4., 0, 0], [0., 4, 0], [0., 0, 4], [-4., 0, 0]])
    rays = np.eye(3)
    evidence = recorded_ray_evidence(centres, 1., rays, np.array([2., 10., 2000.]), np.array([True, True, False]))
    expected = np.array([[True, False, False, False], [False, True, False, False],
                         [False, False, False, True], [False, False, True, False], [False, False, False, False]])
    np.testing.assert_array_equal(evidence, expected)
    np.testing.assert_array_equal(exclusive_evidence_labels(evidence), [0, 1, 3, 2, 4])


@pytest.mark.covers("T78")
def test_evidence_parallel_miss_boundary_and_multiple_observations():
    """An exact face return is seen; parallel off-cell beams supply no evidence."""
    centres = np.array([[2., 0, 0], [2., 2, 0]])
    directions = np.repeat([[1., 0, 0]], 3, axis=0)
    evidence = recorded_ray_evidence(centres, 1., directions, np.array([1.5, 1., 2000.]),
                                     np.array([True, True, False]))
    np.testing.assert_array_equal(evidence, [[True, True, True, False], [False, False, False, False]])
    np.testing.assert_array_equal(exclusive_evidence_labels(evidence), [0, 4])


def test_visibility_follows_world_grid_axes_after_rotating_sweep():
    """Rotate a sensor's x beam into map y before intersecting map-axis cells."""
    rotation = np.array([[0., -1, 0], [1., 0, 0], [0., 0, 1]])
    world_rays = np.array([[1., 0, 0]]) @ rotation.T
    evidence = recorded_ray_evidence(np.array([[0., 2, 0]]), 1., world_rays, np.array([2.]), np.array([True]))
    np.testing.assert_array_equal(evidence, [[True, False, False, False]])


def test_recall_gain_does_not_hide_a_neighbourhood_precision_loss():
    """Recovering one blocked GT cell plus a phantom raises recall but lowers precision."""
    truth = np.array([True, True, False])
    evidence = np.array([[True, False, False, False], [False, True, False, False], [False, False, False, False]])
    baseline = occupied_cell_metrics([True, False, False], truth, evidence)
    radar = occupied_cell_metrics([True, True, True], truth, evidence)
    assert baseline["precision"] == 1 and baseline["recall"] == 1 / 2
    assert radar["precision"] == 2 / 3 and radar["recall"] == 1
    assert baseline["strata"]["blocked"]["recall"] == 0
    assert radar["strata"]["blocked"]["recall"] == 1
    assert radar["strata"]["no_return"]["recall"] is None
