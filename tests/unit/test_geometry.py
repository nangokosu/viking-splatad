"""Unit tests for extract/geometry.py (T9a, T9b, T18, T20)."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.extract.geometry import (
    apply_opencv_to_nerfstudio,
    assign_sensor_indices,
    camera_to_world_opencv,
    lidar_to_world_at_reference_time,
)
from viking_splatad.extract.pose import Trajectory


def make_pose(yaw_radians: float, translation) -> np.ndarray:
    """4x4 pose with yaw about z and given translation."""
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", yaw_radians).as_matrix()
    pose[:3, 3] = translation
    return pose


@pytest.mark.covers("T9a")
def test_camera_to_world_is_base_pose_times_extrinsic():
    """Base at (5,0,0) yaw 90deg; camera 2 m ahead of base -> camera at (5,2,0)."""
    base_pose = make_pose(np.pi / 2, [5.0, 0.0, 0.0])
    extrinsic = make_pose(0.0, [2.0, 0.0, 0.0])
    c2w_float32, c2w_float64 = camera_to_world_opencv(base_pose, extrinsic)
    assert c2w_float32.shape == (3, 4) and c2w_float32.dtype == np.float32
    assert c2w_float64.shape == (4, 4) and c2w_float64.dtype == np.float64
    np.testing.assert_allclose(c2w_float64[:3, 3], [5.0, 2.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(c2w_float64[:3, :3], Rotation.from_euler("z", np.pi / 2).as_matrix(), atol=1e-12)
    np.testing.assert_allclose(c2w_float32, c2w_float64[:3, :], atol=1e-6)


@pytest.mark.covers("T9b")
def test_nerfstudio_flip_is_involution_and_flips_optical_axis():
    """Applying twice is identity; OpenCV +z axis point lands on -z of the new camera frame."""
    c2w_opencv = make_pose(0.7, [1.0, 2.0, 3.0])
    flipped = apply_opencv_to_nerfstudio(c2w_opencv)
    np.testing.assert_allclose(flipped[:3, 3], c2w_opencv[:3, 3])
    np.testing.assert_allclose(apply_opencv_to_nerfstudio(flipped), c2w_opencv, atol=1e-12)
    # World point 1 m along the OpenCV optical axis (camera +z).
    point_world = c2w_opencv @ np.array([0.0, 0.0, 1.0, 1.0])
    # Express in the nerfstudio camera frame through the inverse of the new c2w.
    point_nerfstudio_camera = np.linalg.inv(flipped) @ point_world
    np.testing.assert_allclose(point_nerfstudio_camera[:3], [0.0, 0.0, -1.0], atol=1e-12)
    # 3x4 input stays 3x4.
    assert apply_opencv_to_nerfstudio(c2w_opencv[:3, :]).shape == (3, 4)


@pytest.mark.covers("T18")
def test_lidar_reference_time_start_mid_end_give_distinct_analytic_poses():
    """Linear motion along x at 2 m/s; sweep starts at 10 s, lasts 0.1 s."""
    timestamps = np.array([0.0, 20.0])
    speed = 2.0
    positions = np.array([[0.0, 0.0, 0.0], [speed * 20.0, 0.0, 0.0]])
    quaternions = np.tile([0.0, 0.0, 0.0, 1.0], (2, 1))
    trajectory = Trajectory(timestamps, positions, quaternions)
    lidar_extrinsic = make_pose(0.0, [0.0, 0.0, 1.5])
    sweep_start, sweep_duration = 10.0, 0.1
    expected_times = {"start": 10.0, "mid": 10.05, "end": 10.1}
    for mode, expected_time in expected_times.items():
        pose, reference_time = lidar_to_world_at_reference_time(
            trajectory, lidar_extrinsic, sweep_start, sweep_duration, mode
        )
        assert reference_time == pytest.approx(expected_time)
        np.testing.assert_allclose(pose[:3, 3], [speed * expected_time, 0.0, 1.5], atol=1e-9)
    with pytest.raises(ValueError):
        lidar_to_world_at_reference_time(trajectory, lidar_extrinsic, sweep_start, sweep_duration, "bogus")


@pytest.mark.covers("T20")
def test_sensor_indices_one_camera_one_lidar():
    """1 camera + 1 lidar: camera 0, lidar offset by n_cam -> 1."""
    mapping, names = assign_sensor_indices(["cam_front"], ["lidar_top"])
    assert mapping == {"cam_front": 0, "lidar_top": 1}
    assert names == ["cam_front", "lidar_top"]


@pytest.mark.covers("T20")
def test_sensor_indices_two_cameras_one_lidar():
    """2 cameras + 1 lidar: lidar index 2."""
    mapping, names = assign_sensor_indices(["cam_a", "cam_b"], ["lidar_top"])
    assert mapping == {"cam_a": 0, "cam_b": 1, "lidar_top": 2}
    assert names == ["cam_a", "cam_b", "lidar_top"]
