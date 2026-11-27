"""Stable thin-Gaussian occupancy without regularizing or clipping the trained model."""

import numpy as np
import pytest

from viking_splatad.eval.gaussian_occupancy import checkpoint_gaussians_to_map_frame, save_gaussian_factors
from viking_splatad.eval.voxel_occupancy import gaussian_occupancy_at_centres


@pytest.mark.covers("T57")
def test_factorized_occupancy_handles_covariance_destroyed_by_float32_rounding():
    """A 45-degree thin Gaussian has analytic unit Mahalanobis distance along its short axis."""
    cosine = 1 / np.sqrt(2)
    rotation = np.array([[cosine, -cosine, 0], [cosine, cosine, 0], [0, 0, 1.]])
    scales = np.array([1000., 0.001, 0.0001])
    covariance32 = (rotation @ np.diag(scales ** 2) @ rotation.T).astype(np.float32)
    assert np.linalg.eigvalsh(covariance32.astype(np.float64))[0] <= 0
    centres = np.stack([np.zeros(3), rotation[:, 1] * scales[1], rotation[:, 2] * scales[2]])
    actual = gaussian_occupancy_at_centres(centres, np.zeros((1, 3)), None, np.array([0.5]), 2,
                                         principal_axes=rotation[None], axis_std=scales[None])
    np.testing.assert_allclose(actual, [0.5, 0.5 * np.exp(-0.5), 0.5 * np.exp(-0.5)])


@pytest.mark.covers("T56")
def test_checkpoint_factors_preserve_axes_scales_and_do_not_change_legacy_archive(tmp_path):
    """Known z quarter-turn and exponential scales survive the separate float64 factor archive."""
    quaternion = np.array([[1 / np.sqrt(2), 0., 0., 1 / np.sqrt(2)]])
    scales = np.array([[2., 1., 0.5]])
    decoded = checkpoint_gaussians_to_map_frame(np.zeros((1, 3)), np.log(scales), quaternion,
        np.zeros((1, 1)), np.zeros((1, 1)), 0, np.eye(4), 1.)
    path = tmp_path / "factors.npz"
    save_gaussian_factors(path, decoded, 20)
    with np.load(path, allow_pickle=False) as archive:
        np.testing.assert_allclose(archive["axis_std"], scales)
        np.testing.assert_allclose(archive["principal_axes"][0] @ np.array([1., 0., 0.]), [0., 1., 0.], atol=np.finfo(float).eps * 2)
        assert archive["principal_axes"].dtype == np.float64
        assert int(archive["checkpoint_step"]) == 20
        assert "covariances" not in archive.files
