"""Unit tests for extract/pose.py (T3, T4)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.extract.pose import (
    TfEdge,
    Trajectory,
    check_world_z_up,
    compose_tf_chain,
    interpolate_pose_se3,
    mean_up_vector_from_orientations,
)

VELOCITY = np.array([1.0, -2.0, 0.5])  # m/s, arbitrary non-axis-aligned
YAW_RATE = 0.3  # rad/s, small enough that adjacent samples are < pi apart


def analytic_pose(time_seconds: float) -> np.ndarray:
    """Ground-truth 4x4 pose of constant-velocity, constant-yaw-rate motion."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", YAW_RATE * time_seconds).as_matrix()
    pose[:3, 3] = VELOCITY * time_seconds
    return pose


def build_trajectory() -> Trajectory:
    """Trajectory sampled at 0..4 s every 1 s from the analytic motion."""
    timestamps = np.arange(5, dtype=np.float64)
    positions = np.stack([VELOCITY * time_value for time_value in timestamps])
    quaternions = Rotation.from_rotvec(np.outer(YAW_RATE * timestamps, [0.0, 0.0, 1.0])).as_quat()
    return Trajectory(timestamps, positions, quaternions)


@pytest.mark.covers("T3")
def test_interpolation_matches_analytic_pose():
    """Mid-interval query equals the analytic pose (slerp is exact for constant yaw rate)."""
    trajectory = build_trajectory()
    query_times = np.array([0.0, 0.25, 2.5, 3.9])
    poses = interpolate_pose_se3(trajectory, query_times)
    for query_index, query_time in enumerate(query_times):
        np.testing.assert_allclose(poses[query_index], analytic_pose(query_time), atol=1e-9)
        rotation = poses[query_index][:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.covers("T3")
def test_out_of_range_raises_unless_extrapolation_allowed():
    """Queries outside the span raise; with allow_extrapolation they clamp to the end pose."""
    trajectory = build_trajectory()
    with pytest.raises(ValueError):
        interpolate_pose_se3(trajectory, np.array([-0.1]))
    with pytest.raises(ValueError):
        interpolate_pose_se3(trajectory, np.array([4.1]))
    clamped = interpolate_pose_se3(trajectory, np.array([10.0]), allow_extrapolation=True)
    np.testing.assert_allclose(clamped[0], analytic_pose(4.0), atol=1e-9)


@pytest.mark.covers("T4")
def test_tf_chain_composes_to_expected_matrix():
    """world->base (translate + yaw 90deg) then base->cam (translate), plus an inverted edge."""
    yaw_quaternion = Rotation.from_euler("z", np.pi / 2).as_quat()
    identity_quaternion = np.array([0.0, 0.0, 0.0, 1.0])
    world_base = TfEdge(np.array([10.0, 0.0, 0.0]), yaw_quaternion)
    base_cam = TfEdge(np.array([1.0, 0.0, 0.0]), identity_quaternion)
    composed = compose_tf_chain([world_base, base_cam])
    # Origin of cam in world: base position + R_yaw90 @ [1,0,0] = [10,0,0] + [0,1,0].
    np.testing.assert_allclose(composed[:3, 3], [10.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(composed[:3, :3], Rotation.from_euler("z", np.pi / 2).as_matrix(), atol=1e-12)
    # An edge followed by its own inverse cancels to identity.
    inverse_edge = TfEdge(world_base.translation, world_base.quaternion_xyzw, invert=True)
    np.testing.assert_allclose(compose_tf_chain([world_base, inverse_edge]), np.eye(4), atol=1e-12)


@pytest.mark.covers("T4")
def test_tilted_world_is_flagged_and_level_world_passes():
    """Orientations rolled 20deg about x give tilt 20deg; tolerance 5deg raises, 30deg passes."""
    tilt_radians = np.deg2rad(20.0)
    quaternions = np.tile(Rotation.from_euler("x", tilt_radians).as_quat(), (4, 1))
    mean_up = mean_up_vector_from_orientations(quaternions)
    # First principles: roll about x maps body z to (0, -sin(a), cos(a)); angle to world z is a.
    np.testing.assert_allclose(mean_up, [0.0, -np.sin(tilt_radians), np.cos(tilt_radians)], atol=1e-12)
    assert check_world_z_up(mean_up, tolerance_radians=np.deg2rad(30.0)) == pytest.approx(tilt_radians)
    with pytest.raises(ValueError):
        check_world_z_up(mean_up, tolerance_radians=np.deg2rad(5.0))
    level = mean_up_vector_from_orientations(np.tile([0.0, 0.0, 0.0, 1.0], (3, 1)))
    assert check_world_z_up(level, tolerance_radians=1e-6) == pytest.approx(0.0)
