"""Unit tests for extract/deskew.py (T13 motion compensation)."""

import numpy as np
import pytest

from viking_splatad.extract.deskew import deskew_points

# Synthetic scene constants (test fixtures, not tuned parameters).
SWEEP_PERIOD_SECONDS = 0.1  # a 10 Hz spinning lidar sweep
RIG_SPEED_METRES_PER_SECOND = 4.0
WALL_X_WORLD = 20.0  # static plane x = 20 in the world frame
POINT_COUNT = 400


def make_pose_at(velocity_x):
    """Pose lookup for a rig moving at constant velocity along world x (no rotation)."""

    def pose_at(times):
        poses = np.tile(np.eye(4), (len(times), 1, 1))
        poses[:, 0, 3] = velocity_x * np.asarray(times)
        return poses

    return pose_at


def build_wall_scene():
    """Sample wall points at random capture times, expressed in the moving sensor frame.

    Returns:
        (raw_points, capture_times, reference_time): sensor-frame points measured at
        each point's own time, their absolute times, and the sweep mid time.
    """
    rng = np.random.default_rng(seed=3)
    sweep_start = 100.0
    capture_times = sweep_start + rng.uniform(0, SWEEP_PERIOD_SECONDS, size=POINT_COUNT)
    wall_y = rng.uniform(-10, 10, size=POINT_COUNT)
    wall_z = rng.uniform(-2, 2, size=POINT_COUNT)
    # Sensor origin in world at capture time: (v*t, 0, 0); pure translation, so the
    # sensor-frame point is the world point minus the sensor origin.
    sensor_x = WALL_X_WORLD - RIG_SPEED_METRES_PER_SECOND * capture_times
    raw_points = np.stack([sensor_x, wall_y, wall_z], axis=1)
    reference_time = sweep_start + SWEEP_PERIOD_SECONDS / 2
    return raw_points, capture_times, reference_time


def plane_fit_rms(points):
    """RMS distance of points to their best-fit plane (smallest singular value / sqrt(n))."""
    centred = points - points.mean(axis=0)
    smallest_singular_value = np.linalg.svd(centred, compute_uv=False)[-1]
    return smallest_singular_value / np.sqrt(len(points))


@pytest.mark.covers("T13")
def test_deskew_makes_wall_planar_and_t_is_signed():
    """Raw wall points are non-planar; deskewed are planar at x = wall - v*t_ref."""
    raw_points, capture_times, reference_time = build_wall_scene()
    pose_at = make_pose_at(RIG_SPEED_METRES_PER_SECOND)
    deskewed, relative_time = deskew_points(raw_points, capture_times, pose_at, reference_time)

    # Raw spread in x is v * sweep period = 0.4 m, so raw RMS is clearly nonzero.
    assert plane_fit_rms(raw_points) > 0.5 * RIG_SPEED_METRES_PER_SECOND * SWEEP_PERIOD_SECONDS / 4
    assert plane_fit_rms(deskewed) < 1e-9

    # Analytic plane position in the reference sensor frame.
    expected_x = WALL_X_WORLD - RIG_SPEED_METRES_PER_SECOND * reference_time
    np.testing.assert_allclose(deskewed[:, 0], expected_x, atol=1e-9)
    # y, z unchanged for a pure x translation.
    np.testing.assert_allclose(deskewed[:, 1:], raw_points[:, 1:], atol=1e-12)

    # t = t_point - t_ref: signed, bounded by the sweep period, matches definition.
    np.testing.assert_allclose(relative_time, capture_times - reference_time, atol=1e-12)
    assert relative_time.min() < 0 < relative_time.max()
    assert np.abs(relative_time).max() <= SWEEP_PERIOD_SECONDS


@pytest.mark.covers("T13")
def test_deskew_wrong_motion_sign_is_detected():
    """A pose model moving the wrong way must NOT yield a planar wall (sign guard)."""
    raw_points, capture_times, reference_time = build_wall_scene()
    wrong_pose_at = make_pose_at(-RIG_SPEED_METRES_PER_SECOND)
    deskewed, _ = deskew_points(raw_points, capture_times, wrong_pose_at, reference_time)
    # Residual x error is 2*v*(t - t_ref), RMS ~ 2*v*period/sqrt(12) ~ 0.23 m.
    assert plane_fit_rms(deskewed) > 0.1
    # And swapping the composition order would also break planarity: check the explicit
    # formula against an independently built p_ref = inv(T_ref) @ T_pt @ p.
    pose_at = make_pose_at(RIG_SPEED_METRES_PER_SECOND)
    deskewed_right, _ = deskew_points(raw_points, capture_times, pose_at, reference_time)
    t_ref = pose_at(np.array([reference_time]))[0]
    t_points = pose_at(capture_times)
    homogeneous = np.concatenate([raw_points, np.ones((len(raw_points), 1))], axis=1)
    expected = np.einsum("ab,nbc,nc->na", np.linalg.inv(t_ref), t_points, homogeneous)[:, :3]
    np.testing.assert_allclose(deskewed_right, expected, atol=1e-12)


@pytest.mark.covers("T13")
def test_deskew_rotation_about_z_known_answer():
    """Pure yaw rate: a world-fixed point is mapped by the analytic relative rotation."""
    yaw_rate = 0.5  # rad/s fixture
    reference_time = 0.0
    capture_time = 0.2

    def pose_at(times):
        yaw = yaw_rate * np.asarray(times)
        poses = np.tile(np.eye(4), (len(yaw), 1, 1))
        poses[:, 0, 0], poses[:, 0, 1] = np.cos(yaw), -np.sin(yaw)
        poses[:, 1, 0], poses[:, 1, 1] = np.sin(yaw), np.cos(yaw)
        return poses

    # World point (3,0,0) seen at capture time sits at R(-yaw_c) @ p_world in sensor frame.
    yaw_capture = yaw_rate * capture_time
    point_sensor = np.array([[3 * np.cos(yaw_capture), -3 * np.sin(yaw_capture), 0.0]])
    deskewed, relative_time = deskew_points(
        point_sensor, np.array([capture_time]), pose_at, reference_time
    )
    # At the reference time yaw = 0 so the sensor frame equals the world frame.
    np.testing.assert_allclose(deskewed, [[3.0, 0.0, 0.0]], atol=1e-12)
    assert relative_time[0] == pytest.approx(capture_time - reference_time)
