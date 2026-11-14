"""Synthetic known-answer tests for radar Gaussian initialization (T34-T36)."""

import numpy as np
import pytest

from viking_splatad.extract.pose import Trajectory
from viking_splatad.radar_init.accumulate import AccumulatedRadarPoints, accumulate_radar_points
from viking_splatad.radar_init.seeds import (
    NEUTRAL_GREY_RGB,
    build_seed_gaussians,
    paint_visible_seed_colors,
)
from viking_splatad.radar_init.select import RadarCandidateVoxels, select_radar_candidates


@pytest.mark.covers("T34")
def test_accumulation_uses_only_train_scans_and_removes_static_ego_doppler():
    """A moving radar observes one fixed map point with zero residual after compensation."""
    timestamps = np.array([0.0, 1.0, 2.0])
    trajectory = Trajectory(
        timestamps,
        np.column_stack([timestamps, np.zeros((3, 2))]),
        np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)),
    )
    scans = [
        np.array([[10.0 - scan_time, 0.0, 0.0, 2.0, -1.0]])
        for scan_time in timestamps
    ]
    accumulated = accumulate_radar_points(
        scans, timestamps, np.array([True, False, True]), trajectory, np.eye(4)
    )
    np.testing.assert_allclose(accumulated.points_map, [[10.0, 0.0, 0.0]] * 2)
    np.testing.assert_array_equal(accumulated.scan_indices, [0, 2])
    np.testing.assert_allclose(accumulated.doppler_residuals_mps, 0.0, atol=1e-12)


@pytest.mark.covers("T35")
def test_selection_keeps_persistent_trunk_height_lidar_sparse_voxel():
    """Three-scan trunk support survives while a one-scan return fails persistence."""
    points = np.array([
        [5.00, 0.00, 1.00], [5.00, 0.00, 1.01], [5.00, 0.01, 0.99],
        [8.00, 0.00, 1.00],
    ])
    origins = np.zeros_like(points)
    accumulated = AccumulatedRadarPoints(
        points_map=points,
        sensor_origins_map=origins,
        corrected_times_s=np.array([0.0, 1.0, 2.0, 0.0]),
        scan_indices=np.array([0, 1, 2, 0]),
        powers=np.array([4.0, 5.0, 6.0, 1.0]),
        ranges_m=np.linalg.norm(points, axis=1),
        doppler_residuals_mps=np.zeros(4),
    )
    lidar_points = np.array([
        [5.0, 0.0, 0.0], [5.0, 0.0, 4.0],
        [8.0, 0.0, 0.0], [8.0, 0.0, 4.0],
    ])
    candidates = select_radar_candidates(
        accumulated, lidar_points, 0.1, np.deg2rad(1.25), np.deg2rad(1.7),
        3.0, 75.0, 75.0,
    )
    assert candidates.means_map.shape == (1, 3)
    np.testing.assert_allclose(candidates.means_map[0], points[:3].mean(axis=0))
    assert candidates.support_counts.tolist() == [3]


@pytest.mark.covers("T36")
def test_seed_covariance_matches_radar_resolution_in_ray_basis():
    """An x-axis ray produces the three analytical variances on x/y/z."""
    candidates = RadarCandidateVoxels(
        means_map=np.array([[10.0, 0.0, 0.0]]),
        sensor_origins_map=np.zeros((1, 3)),
        support_counts=np.array([3]),
        time_spans_s=np.array([[1.0, 2.0]]),
        median_powers=np.array([5.0]),
    )
    range_resolution = 0.1
    azimuth_resolution = np.deg2rad(1.25)
    elevation_resolution = np.deg2rad(1.7)
    seeds = build_seed_gaussians(
        candidates, np.array([[10, 20, 30]], dtype=np.uint8), range_resolution,
        azimuth_resolution, elevation_resolution,
    )
    expected_variances = np.square([
        range_resolution,
        10.0 * np.tan(azimuth_resolution),
        10.0 * np.tan(elevation_resolution),
    ])
    np.testing.assert_allclose(seeds.covariances[0], np.diag(expected_variances), atol=1e-12)


@pytest.mark.covers("T36")
def test_camera_painting_uses_grey_for_lidar_occlusion():
    """A closer LiDAR point inside the radar footprint blocks camera color leakage."""
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2] = [12, 34, 56]
    intrinsics = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    colors, painted = paint_visible_seed_colors(
        np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
        np.eye(4), intrinsics, image, np.array([[0.0, 0.0, 3.0]]), 0.25, 0.1,
    )
    np.testing.assert_array_equal(colors[0], NEUTRAL_GREY_RGB)
    np.testing.assert_array_equal(colors[1], image[2, 2])
    np.testing.assert_array_equal(painted, [False, True])
