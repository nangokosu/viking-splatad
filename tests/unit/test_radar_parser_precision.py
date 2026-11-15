"""Encoded float32 parser rotations remain rigid within their actual input precision."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.radar_init.anchors import require_rigid_transform
from viking_splatad.radar_init.artifact import transform_anchor_arrays


def test_float32_parser_rotation_is_preserved_without_silent_reorthogonalization():
    """A general rounded rotation is accepted, applied as encoded, and never projected to another matrix."""
    rotation = Rotation.from_euler("xyz", [np.pi / 7, np.pi / 11, np.pi / 13]).as_matrix().astype(np.float32)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = [1., 2., 3.]
    result = require_rigid_transform(matrix)
    np.testing.assert_array_equal(result, matrix.astype(np.float64))
    arrays = {"means": np.array([[2., 3., 4.]]), "sensor_origins": np.zeros((1, 3)),
              "covariances": np.eye(3)[None]}
    transformed = transform_anchor_arrays(arrays, matrix[:3])
    expected_mean = rotation.astype(np.float64) @ arrays["means"][0] + matrix[:3, 3]
    np.testing.assert_allclose(transformed["means"][0], expected_mean)
    np.testing.assert_allclose(transformed["covariances"][0], rotation.astype(np.float64) @ rotation.astype(np.float64).T)


def test_precision_aware_validation_still_rejects_scale_and_reflection():
    """Actual non-rigid transforms cannot hide behind float32 encoding roundoff."""
    scaled = np.eye(4, dtype=np.float32)
    scaled[0, 0] = 1.01
    reflected = np.eye(4, dtype=np.float32)
    reflected[0, 0] = -1.
    double_scaled = np.eye(4)
    double_scaled[0, 0] = 1 + np.sqrt(np.finfo(np.float64).eps)
    for matrix in (scaled, reflected, double_scaled):
        with pytest.raises(ValueError, match="unit scale"):
            require_rigid_transform(matrix)
