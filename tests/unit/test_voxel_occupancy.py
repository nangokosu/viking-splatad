"""First-principles P2 Gaussian occupancy probability cases."""

import numpy as np
import pytest

from viking_splatad.eval.voxel_occupancy import gaussian_occupancy_at_centres


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
