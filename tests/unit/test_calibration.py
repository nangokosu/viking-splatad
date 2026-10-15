"""Unit tests for extract/calibration.py (T5 extrinsics parsing, T6 transform comparison)."""

from pathlib import Path

import numpy as np
import pytest

from viking_splatad.extract.calibration import (
    compare_transforms,
    load_extrinsics,
    parse_extrinsics_text,
)

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "extrinsics_real.txt"
ABSOLUTE_TOLERANCE = 1e-9


def quaternion_to_matrix(quaternion_xyzw):
    """Independent textbook quaternion -> rotation matrix (normalised first)."""
    qx, qy, qz, qw = np.asarray(quaternion_xyzw, dtype=np.float64) / np.linalg.norm(
        quaternion_xyzw
    )
    return np.array(
        [
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)],
        ]
    )


def make_transform(rotation, translation):
    """Assemble a 4x4 homogeneous transform."""
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation
    transform_matrix[:3, 3] = translation
    return transform_matrix


def rotation_about_axis(axis_unit, angle_rad):
    """Rodrigues rotation matrix for a unit axis."""
    axis_x, axis_y, axis_z = axis_unit
    cross_matrix = np.array([[0, -axis_z, axis_y], [axis_z, 0, -axis_x], [-axis_y, axis_x, 0]])
    return np.eye(3) + np.sin(angle_rad) * cross_matrix + (1 - np.cos(angle_rad)) * (
        cross_matrix @ cross_matrix
    )


@pytest.fixture(scope="module")
def real_text():
    """Real extrinsics.txt text."""
    return FIXTURE_PATH.read_text()


@pytest.mark.covers("T5")
def test_parse_real_file_entries(real_text):
    """All expected frame pairs are present and are proper rigid 4x4 matrices."""
    parsed = parse_extrinsics_text(real_text, "short_grass")
    assert set(parsed) == {
        "base_link->camera", "base_link->emlid_gps", "base_link->hugin_radar",
        "base_link->imu", "base_link->os_sensor", "os_sensor->os_lidar",
        "os_sensor->os_imu",
    }
    for matrix in parsed.values():
        assert matrix.shape == (4, 4) and matrix.dtype == np.float64
        rotation = matrix[:3, :3]
        # Renormalisation must make rotations orthonormal with det +1 despite 3-decimal text.
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=ABSOLUTE_TOLERANCE)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        np.testing.assert_array_equal(matrix[3], [0, 0, 0, 1])


@pytest.mark.covers("T5")
def test_os_sensor_is_pure_translation(real_text):
    """base_link->os_sensor: translation (0.02, 0, 0.225), identity rotation."""
    matrix = parse_extrinsics_text(real_text, "short_grass")["base_link->os_sensor"]
    np.testing.assert_allclose(matrix, make_transform(np.eye(3), [0.02, 0.0, 0.225]))


@pytest.mark.covers("T5")
def test_camera_rotation_maps_optical_axes_to_base_axes(real_text):
    """q=(-.5,.5,-.5,.5) -> optical z (forward) = base +x, optical x (right) = base -y,
    optical y (down) = base -z, derived from first principles (REP-103 / optical frame)."""
    matrix = parse_extrinsics_text(real_text, "short_grass")["base_link->camera"]
    rotation = matrix[:3, :3]
    np.testing.assert_allclose(rotation @ [0, 0, 1], [1, 0, 0], atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, -1, 0], atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(rotation @ [0, 1, 0], [0, 0, -1], atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(matrix[:3, 3], [0.18, -0.01, 0.14])
    # Cross-check against the independent quaternion formula.
    np.testing.assert_allclose(
        rotation, quaternion_to_matrix([-0.5, 0.5, -0.5, 0.5]), atol=ABSOLUTE_TOLERANCE
    )


@pytest.mark.covers("T5")
def test_non_base_entry_and_lidar_flip(real_text):
    """os_sensor->os_lidar: 180 degree yaw (q=(0,0,1,0)): x->-x, y->-y, z->z."""
    matrix = parse_extrinsics_text(real_text, "short_grass")["os_sensor->os_lidar"]
    np.testing.assert_allclose(matrix[:3, :3], np.diag([-1.0, -1.0, 1.0]), atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(matrix[:3, 3], [0.0, 0.0, 0.036])


@pytest.mark.covers("T5")
@pytest.mark.parametrize(
    "variant,quaternion_xyzw",
    [("short_grass", [0.701, 0.712, 0.028, 0.027]), ("high_grass", [0.703, 0.711, 0.004, 0.004])],
)
def test_radar_variant_selection(real_text, variant, quaternion_xyzw):
    """The variant picks its own hugin quaternion; translation is shared."""
    matrix = parse_extrinsics_text(real_text, variant)["base_link->hugin_radar"]
    np.testing.assert_allclose(
        matrix[:3, :3], quaternion_to_matrix(quaternion_xyzw), atol=ABSOLUTE_TOLERANCE
    )
    np.testing.assert_allclose(matrix[:3, 3], [0.2, 0.17, 0.08])


@pytest.mark.covers("T5")
def test_variants_differ(real_text):
    """May and June radar rotations are genuinely different matrices."""
    short_matrix = parse_extrinsics_text(real_text, "short_grass")["base_link->hugin_radar"]
    high_matrix = parse_extrinsics_text(real_text, "high_grass")["base_link->hugin_radar"]
    assert not np.allclose(short_matrix, high_matrix, atol=1e-4)


@pytest.mark.covers("T5")
def test_missing_variant_raises(real_text):
    """Unknown variant, or text lacking the requested variant, raises."""
    with pytest.raises(ValueError, match="mystery_grass"):
        parse_extrinsics_text(real_text, "mystery_grass")
    without_high = real_text.split('"high_grass__ros1__"')[0]
    with pytest.raises(ValueError, match="high_grass"):
        parse_extrinsics_text(without_high, "high_grass")


@pytest.mark.covers("T5")
def test_load_extrinsics_reads_file(tmp_path, real_text):
    """load_extrinsics reads calibration_dir/extrinsics/extrinsics.txt."""
    (tmp_path / "extrinsics").mkdir()
    (tmp_path / "extrinsics" / "extrinsics.txt").write_text(real_text)
    loaded = load_extrinsics(tmp_path, "short_grass")
    expected = parse_extrinsics_text(real_text, "short_grass")
    assert set(loaded) == set(expected)
    for key in expected:
        np.testing.assert_array_equal(loaded[key], expected[key])


@pytest.mark.covers("T6")
def test_compare_equal_transforms_is_zero():
    """Identical transforms -> zero error."""
    transform_matrix = make_transform(rotation_about_axis([0, 0, 1], 0.7), [1.0, 2.0, 3.0])
    translation_error, rotation_error = compare_transforms(transform_matrix, transform_matrix)
    assert translation_error == pytest.approx(0.0, abs=ABSOLUTE_TOLERANCE)
    assert rotation_error == pytest.approx(0.0, abs=1e-7)


@pytest.mark.covers("T6")
@pytest.mark.parametrize("angle_rad", [0.01, 0.5, 1.5, 3.0])
def test_compare_recovers_known_perturbation(angle_rad):
    """Known translation delta (3-4-0 -> 5 m) and rotation angle are recovered."""
    base_rotation = rotation_about_axis(np.array([1.0, 1.0, 0.0]) / np.sqrt(2), 0.3)
    transform_a = make_transform(base_rotation, [1.0, 2.0, 3.0])
    perturbation = rotation_about_axis([0.0, 0.0, 1.0], angle_rad)
    transform_b = make_transform(base_rotation @ perturbation, [1.0 + 3.0, 2.0 + 4.0, 3.0])
    translation_error, rotation_error = compare_transforms(transform_a, transform_b)
    assert translation_error == pytest.approx(5.0)
    assert rotation_error == pytest.approx(angle_rad, abs=1e-6)
