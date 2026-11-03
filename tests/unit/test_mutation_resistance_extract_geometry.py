"""Mutation-resistance tests for extract/{deskew,geometry,pose}.py (T3, T4, T9a, T9b, T13, T18, T20).

Existing tests use translation-only poses, yaw-only rotations or reference_time = 0, so
composition-order, inverse and transpose bugs stay invisible. Here every rigid transform is a
seeded random rotation (random axis and angle) plus translation, rotation and translation do not
commute, and every expected value is built from explicit rotation-matrix algebra (R, t pieces),
never by calling the function under test or reusing its formula.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.extract.deskew import deskew_points
from viking_splatad.extract.geometry import (
    apply_opencv_to_nerfstudio,
    assign_sensor_indices,
    camera_to_world_opencv,
    lidar_to_world_at_reference_time,
)
from viking_splatad.extract.pose import (
    TfEdge,
    Trajectory,
    check_world_z_up,
    compose_tf_chain,
    interpolate_pose_se3,
    mean_up_vector_from_orientations,
)

# Arbitrary fixed seed: only makes the "random" transforms reproducible, no value is tuned to it.
RANDOM_SEED = 20240611
# Float64 algebra with a handful of products: 1e-9 leaves ~6 orders of margin over rounding.
ALGEBRA_TOLERANCE = 1e-9


def random_rotation(rng, maximum_angle=np.pi - 0.3):
    """Rotation about a random unit axis by a random angle in (0.3, maximum_angle).

    Args:
        rng: numpy Generator.
        maximum_angle: Upper bound of the angle in radians.

    Returns:
        scipy Rotation. The 0.3 rad floor keeps it far from identity so a transpose is detectable.
    """
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * rng.uniform(0.3, maximum_angle))


def random_pose(rng):
    """Random rigid (rotation, translation) pair plus the 4x4 matrix built from them."""
    rotation = random_rotation(rng)
    translation = rng.uniform(-5.0, 5.0, size=3)
    matrix = np.eye(4)
    matrix[:3, :3] = rotation.as_matrix()
    matrix[:3, 3] = translation
    return rotation.as_matrix(), translation, matrix


# --------------------------------------------------------------------------- T13 deskew


@pytest.mark.covers("T13")
def test_deskew_rotating_translating_rig_with_nonzero_reference_rotation():
    """Points of a world-static scene map to the reference frame via R_ref^T (R_p p + t_p - t_ref)."""
    rng = np.random.default_rng(RANDOM_SEED)
    # Pose model T(t) = (R0 * Rot(axis, rate*t), velocity*t + origin): non-zero rotation at
    # every time (R0 != I), rotation and translation both vary, and they do not commute.
    base_rotation = random_rotation(rng)
    spin_axis = rng.normal(size=3)
    spin_axis /= np.linalg.norm(spin_axis)
    spin_rate = 0.7  # rad/s fixture
    velocity = np.array([3.0, -1.5, 0.8])
    origin = np.array([2.0, -4.0, 1.0])

    def rotation_at(time_value):
        """Pose rotation matrix at one time."""
        return (base_rotation * Rotation.from_rotvec(spin_axis * spin_rate * time_value)).as_matrix()

    def pose_at(times):
        """T_world_lidar for each time."""
        poses = np.tile(np.eye(4), (len(times), 1, 1))
        for pose_index, time_value in enumerate(times):
            poses[pose_index, :3, :3] = rotation_at(time_value)
            poses[pose_index, :3, 3] = origin + velocity * time_value
        return poses

    point_count = 12
    reference_time = 7.3  # non-zero reference
    capture_times = reference_time + rng.uniform(-0.3, 0.3, size=point_count)
    world_points = rng.uniform(-10, 10, size=(point_count, 3))
    # Sensor-frame measurement at capture time: R_p^T (p_world - t_p).
    raw_points = np.stack([
        rotation_at(time_value).T @ (world_point - (origin + velocity * time_value))
        for world_point, time_value in zip(world_points, capture_times)
    ])
    deskewed, relative_time = deskew_points(raw_points, capture_times, pose_at, reference_time)

    # Expected: the same world point seen from the reference pose. Rotation of the sweep is
    # tiny so a correct answer must also differ from the raw points (guards a no-op).
    reference_rotation = rotation_at(reference_time)
    reference_origin = origin + velocity * reference_time
    expected = np.stack([reference_rotation.T @ (world_point - reference_origin) for world_point in world_points])
    np.testing.assert_allclose(deskewed, expected, atol=ALGEBRA_TOLERANCE)
    assert np.abs(deskewed - raw_points).max() > 0.1
    np.testing.assert_allclose(relative_time, capture_times - reference_time, atol=1e-12)


# --------------------------------------------------------------------------- T9a / T9b


@pytest.mark.covers("T9a")
def test_camera_to_world_with_random_rotations_on_both_transforms():
    """c2w rotation = R_base R_ext and translation = R_base t_ext + t_base (order matters)."""
    rng = np.random.default_rng(RANDOM_SEED + 1)
    base_rotation, base_translation, base_pose = random_pose(rng)
    extrinsic_rotation, extrinsic_translation, extrinsic = random_pose(rng)
    c2w_float32, c2w_float64 = camera_to_world_opencv(base_pose, extrinsic)
    np.testing.assert_allclose(c2w_float64[:3, :3], base_rotation @ extrinsic_rotation, atol=ALGEBRA_TOLERANCE)
    np.testing.assert_allclose(
        c2w_float64[:3, 3], base_rotation @ extrinsic_translation + base_translation, atol=ALGEBRA_TOLERANCE
    )
    np.testing.assert_allclose(c2w_float64[3], [0, 0, 0, 1], atol=0)
    # A point at the camera origin must land at the c2w translation (extrinsic applied first).
    np.testing.assert_allclose(c2w_float32[:, 3], c2w_float64[:3, 3], atol=1e-5)
    # The swapped product would give a different matrix: the fixture must be able to tell.
    assert not np.allclose(base_pose @ extrinsic, extrinsic @ base_pose, atol=1e-3)


@pytest.mark.covers("T9b")
def test_nerfstudio_flip_maps_all_three_axes_for_random_rotation():
    """Columns (right, down, forward) become (right, up, backward): x kept, y and z negated."""
    rng = np.random.default_rng(RANDOM_SEED + 2)
    rotation, translation, camera_to_world = random_pose(rng)
    flipped = apply_opencv_to_nerfstudio(camera_to_world)
    np.testing.assert_allclose(flipped[:3, 0], rotation[:, 0], atol=ALGEBRA_TOLERANCE)
    np.testing.assert_allclose(flipped[:3, 1], -rotation[:, 1], atol=ALGEBRA_TOLERANCE)
    np.testing.assert_allclose(flipped[:3, 2], -rotation[:, 2], atol=ALGEBRA_TOLERANCE)
    np.testing.assert_allclose(flipped[:3, 3], translation, atol=ALGEBRA_TOLERANCE)
    # Still a proper rotation (det +1), not a reflection.
    assert np.linalg.det(flipped[:3, :3]) == pytest.approx(1.0)
    # Input untouched, 3x4 float32 input keeps shape and dtype and gets the same column map.
    assert np.array_equal(camera_to_world[:3, :3], rotation)
    flipped_float32 = apply_opencv_to_nerfstudio(camera_to_world[:3, :].astype(np.float32))
    assert flipped_float32.shape == (3, 4) and flipped_float32.dtype == np.float32
    np.testing.assert_allclose(flipped_float32[:, :3], flipped[:3, :3], atol=1e-6)


# --------------------------------------------------------------------------- T18


@pytest.mark.covers("T18")
def test_lidar_to_world_with_rotating_base_and_rotated_extrinsic():
    """T_world_lidar = T_world_base(t_ref) T_base_lidar with both rotations non-identity."""
    rng = np.random.default_rng(RANDOM_SEED + 3)
    start_rotation = random_rotation(rng, maximum_angle=1.0)
    spin_axis = rng.normal(size=3)
    spin_axis /= np.linalg.norm(spin_axis)
    spin_rate = 0.05  # rad/s: small so the slerp segment is far from antipodal
    velocity = np.array([1.5, -0.5, 0.2])
    # Two keyframes bound the motion; slerp on a constant-rate spin is exact at any time.
    timestamps = np.array([0.0, 40.0])
    positions = np.stack([velocity * time_value for time_value in timestamps])
    quaternions = np.stack([
        (start_rotation * Rotation.from_rotvec(spin_axis * spin_rate * time_value)).as_quat()
        for time_value in timestamps
    ])
    trajectory = Trajectory(timestamps, positions, quaternions)
    extrinsic_rotation, extrinsic_translation, extrinsic = random_pose(rng)
    sweep_start, sweep_duration = 12.0, 0.4
    for mode, fraction in {"start": 0.0, "mid": 0.5, "end": 1.0}.items():
        pose, reference_time = lidar_to_world_at_reference_time(
            trajectory, extrinsic, sweep_start, sweep_duration, mode
        )
        time_value = sweep_start + fraction * sweep_duration
        assert reference_time == pytest.approx(time_value)
        base_rotation = (start_rotation * Rotation.from_rotvec(spin_axis * spin_rate * time_value)).as_matrix()
        base_translation = velocity * time_value
        np.testing.assert_allclose(pose[:3, :3], base_rotation @ extrinsic_rotation, atol=1e-7)
        np.testing.assert_allclose(
            pose[:3, 3], base_rotation @ extrinsic_translation + base_translation, atol=1e-7
        )


# --------------------------------------------------------------------------- T20


@pytest.mark.covers("T20")
def test_sensor_indices_two_lidars_with_cameras_shift_all_lidars():
    """With cameras present every lidar index is offset by the camera count, order preserved."""
    mapping, names = assign_sensor_indices(["cam_a", "cam_b", "cam_c"], ["lidar_x", "lidar_y"])
    assert mapping == {"cam_a": 0, "cam_b": 1, "cam_c": 2, "lidar_x": 3, "lidar_y": 4}
    assert names == ["cam_a", "cam_b", "cam_c", "lidar_x", "lidar_y"]
    # Two cameras, two lidars: lidars start at 2 (offset equals camera count, not 1).
    mapping, names = assign_sensor_indices(["cam_a", "cam_b"], ["lidar_x", "lidar_y"])
    assert mapping["lidar_x"] == 2 and mapping["lidar_y"] == 3
    assert names == ["cam_a", "cam_b", "lidar_x", "lidar_y"]
    # No cameras: nothing to collide with, lidars keep their own 0..n-1.
    mapping, names = assign_sensor_indices([], ["lidar_x", "lidar_y"])
    assert mapping == {"lidar_x": 0, "lidar_y": 1} and names == ["lidar_x", "lidar_y"]


# --------------------------------------------------------------------------- T3


@pytest.mark.covers("T3")
def test_interpolation_nonuniform_spacing_piecewise_path_and_knots():
    """Unequal knot gaps and per-segment rates: expected from np.interp on angle and position."""
    rng = np.random.default_rng(RANDOM_SEED + 4)
    spin_axis = rng.normal(size=3)
    spin_axis /= np.linalg.norm(spin_axis)
    start_rotation = random_rotation(rng)
    knot_times = np.array([0.0, 0.5, 2.0, 2.2, 5.0])  # gaps 0.5, 1.5, 0.2, 2.8
    # Piecewise path: angle and position change at different rates on each segment
    # (accelerating then reversing), so index-based or global-linear interpolation is wrong.
    knot_angles = np.array([0.0, 0.1, 0.9, 0.5, 1.2])
    knot_positions = rng.uniform(-3, 3, size=(5, 3))
    quaternions = np.stack([
        (start_rotation * Rotation.from_rotvec(spin_axis * angle)).as_quat() for angle in knot_angles
    ])
    trajectory = Trajectory(knot_times, knot_positions, quaternions)
    query_times = np.array([0.0, 0.2, 0.5, 1.0, 1.9, 2.0, 2.1, 2.2, 3.5, 4.99, 5.0])
    poses = interpolate_pose_se3(trajectory, query_times)
    for pose_index, query_time in enumerate(query_times):
        # np.interp is the independent piecewise-linear reference for angle and position.
        angle = np.interp(query_time, knot_times, knot_angles)
        position = np.array([np.interp(query_time, knot_times, knot_positions[:, axis]) for axis in range(3)])
        expected_rotation = (start_rotation * Rotation.from_rotvec(spin_axis * angle)).as_matrix()
        np.testing.assert_allclose(poses[pose_index, :3, :3], expected_rotation, atol=1e-9)
        np.testing.assert_allclose(poses[pose_index, :3, 3], position, atol=1e-9)
    # Exact-knot queries reproduce the keyframes bit-for-bit within rounding.
    knot_poses = interpolate_pose_se3(trajectory, knot_times)
    np.testing.assert_allclose(knot_poses[:, :3, 3], knot_positions, atol=1e-12)
    np.testing.assert_allclose(
        knot_poses[:, :3, :3], Rotation.from_quat(quaternions).as_matrix(), atol=1e-9
    )


@pytest.mark.covers("T3")
def test_interpolation_range_boundaries_and_clamping():
    """First and last knot are in range; just outside raises; extrapolation clamps to the end poses."""
    rng = np.random.default_rng(RANDOM_SEED + 5)
    knot_times = np.array([1.0, 1.5, 4.0])
    knot_positions = rng.uniform(-3, 3, size=(3, 3))
    quaternions = Rotation.from_rotvec(rng.normal(size=(3, 3)) * 0.3).as_quat()
    trajectory = Trajectory(knot_times, knot_positions, quaternions)
    interpolate_pose_se3(trajectory, np.array([knot_times[0], knot_times[-1]]))  # must not raise
    for outside in (np.nextafter(knot_times[0], -np.inf), np.nextafter(knot_times[-1], np.inf)):
        with pytest.raises(ValueError):
            interpolate_pose_se3(trajectory, np.array([outside]))
    clamped = interpolate_pose_se3(trajectory, np.array([-100.0, 100.0]), allow_extrapolation=True)
    np.testing.assert_allclose(clamped[0, :3, 3], knot_positions[0], atol=1e-12)
    np.testing.assert_allclose(clamped[1, :3, 3], knot_positions[-1], atol=1e-12)
    np.testing.assert_allclose(clamped[0, :3, :3], Rotation.from_quat(quaternions[0]).as_matrix(), atol=1e-9)
    np.testing.assert_allclose(clamped[1, :3, :3], Rotation.from_quat(quaternions[-1]).as_matrix(), atol=1e-9)


# --------------------------------------------------------------------------- T4


def edge_matrix(rotation_matrix, translation):
    """4x4 from rotation and translation (test-side builder)."""
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    matrix[:3, 3] = translation
    return matrix


@pytest.mark.covers("T4")
def test_tf_chain_with_inverted_edge_differs_from_forward_edge_random_rotations():
    """[A, inv(B)] with A != B and rotated edges equals A @ inv(B); inv via general np.linalg.inv."""
    rng = np.random.default_rng(RANDOM_SEED + 6)
    rotation_a, translation_a, matrix_a = random_pose(rng)
    rotation_b, translation_b, matrix_b = random_pose(rng)
    rotation_c, translation_c, matrix_c = random_pose(rng)
    edge_a = TfEdge(translation_a, Rotation.from_matrix(rotation_a).as_quat())
    edge_b_inverted = TfEdge(translation_b, Rotation.from_matrix(rotation_b).as_quat(), invert=True)
    edge_c = TfEdge(translation_c, Rotation.from_matrix(rotation_c).as_quat())
    np.testing.assert_allclose(compose_tf_chain([edge_a, edge_b_inverted]), matrix_a @ np.linalg.inv(matrix_b), atol=1e-9)
    # Three edges with an inverted middle edge: pins left-to-right (root first) order.
    np.testing.assert_allclose(
        compose_tf_chain([edge_a, edge_b_inverted, edge_c]),
        matrix_a @ np.linalg.inv(matrix_b) @ matrix_c,
        atol=1e-9,
    )
    # Reordering gives a different matrix, so the assertion above can tell the order.
    assert not np.allclose(matrix_a @ np.linalg.inv(matrix_b), np.linalg.inv(matrix_b) @ matrix_a, atol=1e-3)
    # Empty chain is identity.
    np.testing.assert_allclose(compose_tf_chain([]), np.eye(4), atol=0)


@pytest.mark.covers("T4")
def test_mean_up_vector_over_distinct_orientations_is_normalised_mean():
    """Rolls about x and pitches about y give known up vectors; the mean is renormalised."""
    roll_angles = np.deg2rad([10.0, -30.0])
    pitch_angles = np.deg2rad([40.0, 5.0])
    quaternions = np.concatenate([
        Rotation.from_euler("x", roll_angles).as_quat(),
        Rotation.from_euler("y", pitch_angles).as_quat(),
    ])
    # First principles: roll a about x maps z to (0, -sin a, cos a); pitch b about y to (sin b, 0, cos b).
    up_vectors = np.concatenate([
        np.stack([np.zeros(2), -np.sin(roll_angles), np.cos(roll_angles)], axis=1),
        np.stack([np.sin(pitch_angles), np.zeros(2), np.cos(pitch_angles)], axis=1),
    ])
    raw_mean = up_vectors.mean(axis=0)
    mean_up = mean_up_vector_from_orientations(quaternions)
    np.testing.assert_allclose(mean_up, raw_mean / np.linalg.norm(raw_mean), atol=1e-12)
    assert np.linalg.norm(mean_up) == pytest.approx(1.0)
    # Non-default body axis: body +x under yaw a about z is (cos a, sin a, 0).
    yaw_angles = np.deg2rad([0.0, 90.0])
    mean_x = mean_up_vector_from_orientations(
        Rotation.from_euler("z", yaw_angles).as_quat(), body_up_axis=(1.0, 0.0, 0.0)
    )
    np.testing.assert_allclose(mean_x, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0), atol=1e-12)


@pytest.mark.covers("T4")
def test_check_world_z_up_normalises_input_and_tolerance_boundary():
    """Scaled input has the same tilt; tolerance just below the tilt raises, just above passes."""
    tilt = np.deg2rad(12.0)
    direction = np.array([np.sin(tilt) * np.cos(0.4), np.sin(tilt) * np.sin(0.4), np.cos(tilt)])
    # A non-unit vector (norm 5) must give the same tilt as the unit one.
    assert check_world_z_up(5.0 * direction, tolerance_radians=tilt + 1e-6) == pytest.approx(tilt, abs=1e-12)
    with pytest.raises(ValueError):
        check_world_z_up(5.0 * direction, tolerance_radians=tilt - 1e-6)
    # An upside-down vector is a tilt of pi, not 0 (no abs() on the dot product).
    with pytest.raises(ValueError):
        check_world_z_up(np.array([0.0, 0.0, -2.0]), tolerance_radians=3.0)
    assert check_world_z_up(np.array([0.0, 0.0, 3.0]), tolerance_radians=1e-9) == pytest.approx(0.0)
