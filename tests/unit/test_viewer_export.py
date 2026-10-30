"""Tests of viking_splatad.viz.viewer_export (T69, T70) with synthetic known answers."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.viz.viewer_export import (
    intensity_colors,
    pack_rgb_float32,
    quaternion_to_fiftyone_rotation,
    read_binary_pcd,
    scale_to_unit_range,
    sweep_points_to_map,
    write_binary_pcd,
)


def axis_rotation_matrix(axis_index: int, angle_radians: float) -> np.ndarray:
    """Right-handed 3x3 rotation about one coordinate axis, from the textbook formula.

    Args:
        axis_index: 0 = x, 1 = y, 2 = z.
        angle_radians: Rotation angle.

    Returns:
        (3, 3) rotation matrix.
    """
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    # Rotation in the plane of the two other axes: [[c, -s], [s, c]] placed on those axes.
    first_axis, second_axis = [(1, 2), (2, 0), (0, 1)][axis_index]
    matrix = np.eye(3)
    matrix[first_axis, first_axis], matrix[first_axis, second_axis] = cosine, -sine
    matrix[second_axis, first_axis], matrix[second_axis, second_axis] = sine, cosine
    return matrix


@pytest.mark.covers("T69")
def test_yaw_only_quaternion_gives_rotation_about_z_only():
    """A box turned by yaw about +z has quaternion (0, 0, sin(yaw/2), cos(yaw/2)) -> Euler (0, 0, yaw)."""
    yaw = 0.7
    quaternion = [0.0, 0.0, np.sin(yaw / 2), np.cos(yaw / 2)]
    np.testing.assert_allclose(quaternion_to_fiftyone_rotation(quaternion), [0.0, 0.0, yaw], atol=1e-12)


@pytest.mark.covers("T69")
def test_euler_triple_rebuilds_the_original_rotation_matrix_as_rx_ry_rz():
    """Intrinsic XYZ means R = Rx(a) Ry(b) Rz(c); the returned angles must rebuild the input rotation."""
    angle_x, angle_y, angle_z = 0.3, -0.5, 1.1
    expected_matrix = (
        axis_rotation_matrix(0, angle_x) @ axis_rotation_matrix(1, angle_y) @ axis_rotation_matrix(2, angle_z)
    )
    quaternion = Rotation.from_matrix(expected_matrix).as_quat()
    euler_angles = quaternion_to_fiftyone_rotation(quaternion)
    np.testing.assert_allclose(euler_angles, [angle_x, angle_y, angle_z], atol=1e-12)
    rebuilt_matrix = (
        axis_rotation_matrix(0, euler_angles[0])
        @ axis_rotation_matrix(1, euler_angles[1])
        @ axis_rotation_matrix(2, euler_angles[2])
    )
    np.testing.assert_allclose(rebuilt_matrix, expected_matrix, atol=1e-12)


@pytest.mark.covers("T69")
def test_unnormalised_quaternion_is_accepted():
    """Scaling the quaternion must not change the rotation (it is normalised first)."""
    quaternion = np.array([0.0, 0.0, np.sin(0.25), np.cos(0.25)])
    np.testing.assert_allclose(
        quaternion_to_fiftyone_rotation(3.0 * quaternion), quaternion_to_fiftyone_rotation(quaternion), atol=1e-12
    )


@pytest.mark.covers("T70")
def test_points_follow_rotation_then_translation():
    """+90 deg about z sends (1, 0, 0) to (0, 1, 0); the translation (10, 20, 30) is added after."""
    pose = np.eye(4)
    pose[:3, :3] = axis_rotation_matrix(2, np.pi / 2)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    points_map = sweep_points_to_map(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), pose)
    np.testing.assert_allclose(points_map, [[10.0, 21.0, 30.0], [10.0, 20.0, 30.0]], atol=1e-12)


def test_binary_pcd_round_trips_points_and_colors(tmp_path):
    """Written coordinates and 8-bit colours come back exactly (float32 coordinates)."""
    points = np.array([[1.5, -2.25, 3.0], [0.0, 0.125, -7.5]], dtype=np.float32)
    colors = np.array([[255, 0, 7], [1, 128, 254]], dtype=np.uint8)
    pcd_path = tmp_path / "cloud.pcd"
    write_binary_pcd(pcd_path, points, colors)
    read_points, read_colors = read_binary_pcd(pcd_path)
    np.testing.assert_array_equal(read_points, points)
    np.testing.assert_array_equal(read_colors, colors)
    # The header must announce the same point count as the payload holds.
    assert b"POINTS 2\n" in pcd_path.read_bytes()


def test_pack_rgb_matches_the_pcl_integer_formula():
    """Packed colour bits equal r*65536 + g*256 + b."""
    packed = pack_rgb_float32(np.array([[1, 2, 3]], dtype=np.uint8)).view(np.uint32)[0]
    assert packed == 1 * 65536 + 2 * 256 + 3


def test_pcd_writer_rejects_mismatched_inputs(tmp_path):
    """Wrong point shape or a colour count different from the point count is refused."""
    with pytest.raises(ValueError):
        write_binary_pcd(tmp_path / "a.pcd", np.zeros((2, 4)), np.zeros((2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        write_binary_pcd(tmp_path / "b.pcd", np.zeros((2, 3)), np.zeros((3, 3), dtype=np.uint8))


def test_intensity_colors_span_the_full_byte_range_in_every_channel():
    """Min intensity -> 0, max -> 255, midpoint -> round(127.5); a constant column gives zeros."""
    colors = intensity_colors(np.array([2.0, 4.0, 6.0]))
    np.testing.assert_array_equal(colors[:, 0], [0, 128, 255])
    np.testing.assert_array_equal(colors[:, 0], colors[:, 2])
    np.testing.assert_array_equal(scale_to_unit_range(np.array([5.0, 5.0])), [0.0, 0.0])
