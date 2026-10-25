"""Tests of viking_splatad.splatad.rendered_lidar (T50 and sweep naming), synthetic known answers."""

import math

import numpy as np
import pytest

from viking_splatad.splatad.rendered_lidar import (
    lidar_pose_in_map_frame,
    match_scan_times_to_frame_indices,
    sorted_sweep_stems,
)


def rotation_about_z(angle_radians: float) -> np.ndarray:
    """3x3 rotation by ``angle_radians`` about +z (right-handed).

    Args:
        angle_radians: Rotation angle.

    Returns:
        (3, 3) rotation matrix.
    """
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.covers("T50")
def test_map_pose_undoes_world_to_mean_transform():
    """Lidar at map position c with yaw a; w2m = yaw b, shift t.

    By construction lidar_to_world_dp = W @ lidar_to_map, so T50 must return lidar_to_map:
    rotation yaw(a), translation c, built independently of W.
    """
    # Known map-frame lidar pose: yaw a, position c.
    lidar_yaw = math.radians(30.0)
    lidar_position_in_map = np.array([12.0, -4.0, 1.5])
    expected_lidar_to_map = np.eye(4)
    expected_lidar_to_map[:3, :3] = rotation_about_z(lidar_yaw)
    expected_lidar_to_map[:3, 3] = lidar_position_in_map
    # Known w2m: yaw b and a translation, as ADDataParser stores it (3x4).
    world_to_mean = np.zeros((3, 4))
    world_to_mean[:, :3] = rotation_about_z(math.radians(-75.0))
    world_to_mean[:, 3] = [-10.0, 3.0, -0.5]
    world_to_mean_4x4 = np.vstack([world_to_mean, [0.0, 0.0, 0.0, 1.0]])
    # The dataparser-frame pose the model sees (3x4, as in Lidars.lidar_to_worlds).
    lidar_to_world_dataparser = (world_to_mean_4x4 @ expected_lidar_to_map)[:3, :]

    lidar_to_map = lidar_pose_in_map_frame(lidar_to_world_dataparser, world_to_mean)

    np.testing.assert_allclose(lidar_to_map, expected_lidar_to_map, atol=1e-12)


@pytest.mark.covers("T50")
def test_lidar_point_reaches_its_map_position():
    """A point 5 m ahead of a lidar yawed 90 deg at c lies at c + (0, 5, 0) in the map.

    yaw(90 deg) maps lidar +x to map +y, so p_map = c + 5 * e_y, whatever w2m is.
    """
    lidar_position_in_map = np.array([2.0, 1.0, 0.0])
    lidar_to_map_truth = np.eye(4)
    lidar_to_map_truth[:3, :3] = rotation_about_z(math.pi / 2.0)
    lidar_to_map_truth[:3, 3] = lidar_position_in_map
    world_to_mean_4x4 = np.eye(4)
    world_to_mean_4x4[:3, :3] = rotation_about_z(math.radians(40.0))
    world_to_mean_4x4[:3, 3] = [7.0, -2.0, 0.3]
    lidar_to_world_dataparser = world_to_mean_4x4 @ lidar_to_map_truth

    lidar_to_map = lidar_pose_in_map_frame(lidar_to_world_dataparser, world_to_mean_4x4)
    point_in_lidar = np.array([5.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose((lidar_to_map @ point_in_lidar)[:3], lidar_position_in_map + [0.0, 5.0, 0.0],
                               atol=1e-12)


def test_scan_times_match_nearest_sweeps_within_half_period():
    """Sweeps every period P from t0; scans jittered by < P/2 map back to their own sweep."""
    sweep_period = 0.1
    frame_times = 100.0 + sweep_period * np.arange(10)
    chosen_frames = np.array([7, 0, 3, 9])
    # Jitters strictly inside +-P/2, both signs.
    jitters = np.array([0.04, -0.049, 0.0, -0.01])
    scan_times = frame_times[chosen_frames] + jitters

    matched = match_scan_times_to_frame_indices(scan_times, frame_times, sweep_period / 2.0)

    np.testing.assert_array_equal(matched, chosen_frames)


def test_scan_beyond_tolerance_raises():
    """A scan 0.06 s after a sweep, tolerance 0.05 s, has no matching sweep (time base off)."""
    frame_times = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="nearest sweep"):
        match_scan_times_to_frame_indices(np.array([0.26]), frame_times, 0.05)


def test_two_scans_on_one_sweep_raise():
    """Two scans 0.01 s either side of sweep 1 both match it, which must not pass silently."""
    frame_times = np.array([0.0, 0.1, 0.2])
    with pytest.raises(ValueError, match="same input sweep"):
        match_scan_times_to_frame_indices(np.array([0.09, 0.11]), frame_times, 0.05)


def test_sweep_stems_are_ordered_by_index(tmp_path):
    """Files written out of order come back as index 0, 1, 2 with their timestamps."""
    stems = ["000002_300", "000000_100", "000001_200"]
    for stem in stems:
        (tmp_path / f"{stem}.npy").write_bytes(b"")

    assert sorted_sweep_stems(tmp_path) == ["000000_100", "000001_200", "000002_300"]


def test_sweep_stems_with_index_gap_raise(tmp_path):
    """Indices 0 and 2 only: position 1 would point at the wrong poses.npz entry."""
    for stem in ["000000_100", "000002_300"]:
        (tmp_path / f"{stem}.npy").write_bytes(b"")
    with pytest.raises(ValueError, match="contiguous"):
        sorted_sweep_stems(tmp_path)


@pytest.mark.covers("T50")
def test_scaled_dataparser_transform_raises():
    """W = 2 * I has R^T R = 4 I: the rigid inverse would be off by 4x, so it must be refused."""
    scaled_world_to_mean = np.hstack([2.0 * np.eye(3), np.zeros((3, 1))])
    with pytest.raises(ValueError, match="not rigid"):
        lidar_pose_in_map_frame(np.hstack([np.eye(3), np.zeros((3, 1))]), scaled_world_to_mean)
