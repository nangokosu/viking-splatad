"""Analytic tests of loss-only radar geometry, selection and frame/window gates."""

import numpy as np
import pytest

from viking_splatad.extract.pose import Trajectory
from viking_splatad.radar_init.anchors import (
    accumulate_radar, persistent_anchors, resolution_covariances, static_doppler_mask,
)
from viking_splatad.radar_init.artifact import (
    crop_anchor_arrays, corrected_radar_times, transform_anchor_arrays,
)


@pytest.mark.covers("T34")
def test_clock_corrected_radar_composition_and_no_extrapolation():
    """A unit-speed base, translated sensor and known offset give analytic map points."""
    trajectory = Trajectory(np.array([0., 2.]), np.array([[0., 0., 0.], [2., 0., 0.]]),
                            np.array([[0., 0., 0., 1.], [0., 0., 0., 1.]]))
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [0., 1., 0.]
    points, origins, times = accumulate_radar(np.array([[1., 0., 0.]]), np.array([0.5]),
                                              trajectory, extrinsic, 0.25, 0.0, 0.0)
    np.testing.assert_allclose(times, [0.5 + 0.25])
    np.testing.assert_allclose(origins, [[0.5 + 0.25, 1., 0.]])
    np.testing.assert_allclose(points, [[1. + 0.5 + 0.25, 1., 0.]])
    with pytest.raises(ValueError, match="trajectory support"):
        accumulate_radar(np.array([[1., 0., 0.]]), np.array([2.]), trajectory, extrinsic, 0.1, 0., 0.)


@pytest.mark.covers("T36")
def test_covariance_distinguishes_full_cell_width_from_sigma():
    """Uniform angular-cell variance is width squared / 12, rotated with the ray basis."""
    distance, radial_sigma, azimuth_width, elevation_width = 10., 0.2, 0.1, 0.2
    covariance = resolution_covariances(np.array([[distance, 0., 0.], [0., distance, 0.]]),
                                        radial_sigma, azimuth_width, elevation_width)
    expected = np.diag([radial_sigma**2, (distance * azimuth_width)**2 / 12,
                        (distance * elevation_width)**2 / 12])
    np.testing.assert_allclose(covariance[0], expected)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    np.testing.assert_allclose(covariance[1], rotation @ expected @ rotation.T)


def test_static_doppler_filter_uses_signed_ego_projection():
    """Three stationary returns agree with -vx; a moving return is rejected."""
    points = np.tile([10., 0., 0.], (4, 1))
    kept, scale = static_doppler_mask(points, np.array([-1., -1., -1., 2.]),
                                     np.array([1., 0., 0.]), 0.1, 0.95)
    np.testing.assert_array_equal(kept, [True, True, True, False])
    assert scale == 0.1


@pytest.mark.covers("T35")
def test_persistence_counts_distinct_sweeps_not_duplicate_points():
    """Only the repeated voxel survives; covariance includes observed centroid spread."""
    points = np.array([[0.1, 0.1, 0.1], [0.3, 0.1, 0.1], [2.1, 0.1, 0.1], [2.2, 0.1, 0.1]])
    result = persistent_anchors(points, np.tile(np.eye(3), (4, 1, 1)), np.zeros_like(points),
                                 np.array([0., 1., 2., 2.]), np.array([10, 11, 12, 12]),
                                 np.ones(3), 0.5)
    np.testing.assert_allclose(result["means"], [[(0.1 + 0.3) / 2, 0.1, 0.1]])
    np.testing.assert_allclose(result["covariances"], [np.diag([1 + ((0.3 - 0.1) / 2)**2, 1, 1])])
    np.testing.assert_array_equal(result["support_count"], [2])
    np.testing.assert_array_equal(result["source_stamps_ns"], [10, 11])


@pytest.mark.covers("T39")
def test_anchor_frame_transform_rotates_covariance_and_actual_origins():
    """A yaw plus translation applies to positions and origins, never to covariance translation."""
    arrays = {"means": np.array([[1., 2., 3.]]), "sensor_origins": np.array([[0., 0., 0.]]),
              "covariances": np.array([np.diag([1., 2., 3.])])}
    matrix = np.array([[0., -1., 0., 10.], [1., 0., 0., 20.], [0., 0., 1., 30.]])
    result = transform_anchor_arrays(arrays, matrix)
    np.testing.assert_allclose(result["means"], [[8., 21., 33.]])
    np.testing.assert_allclose(result["sensor_origins"], [[10., 20., 30.]])
    np.testing.assert_allclose(result["covariances"], [np.diag([2., 1., 3.])])
    np.testing.assert_array_equal(arrays["means"], [[1., 2., 3.]])
    with pytest.raises(ValueError, match="unit"):
        transform_anchor_arrays(arrays, matrix, scale=2.)


@pytest.mark.covers("T40")
def test_anchor_crop_is_inclusive_and_shifts_only_retained_time_spans():
    """One anchor is kept; one is outside the box and another outside the time window."""
    arrays = {"means": np.array([[0., 0., 0.], [2., 0., 0.], [0., 0., 0.]]),
              "covariances": np.tile(np.eye(3), (3, 1, 1)), "sensor_origins": np.zeros((3, 3)),
              "support_count": np.array([2, 2, 2]), "time_spans": np.array([[10., 11.], [10., 11.], [9., 10.]])}
    result = crop_anchor_arrays(arrays, (10., 11.), np.array([[-1., -1., -1.], [1., 1., 1.]]), 10.)
    np.testing.assert_array_equal(result["time_spans"], [[0., 1.]])
    assert result["dropped_count"] == 2
    assert result["retained_fraction"] == 1 / 3


def test_clock_conversion_subtracts_epoch_in_integers():
    """Nanosecond epoch precision is preserved before applying the measured clock model."""
    reference = 1_700_000_000_000_000_000
    calibration = {"offset_seconds": 0.1, "drift_rate": 0.01, "clock_reference_seconds": 0.0}
    result = corrected_radar_times(np.array([reference + 1], dtype=np.int64), calibration, reference)
    np.testing.assert_allclose(result, [1e-9 + 0.1 + 0.01 * 1e-9], rtol=0, atol=np.finfo(float).eps)
