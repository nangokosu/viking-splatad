"""Tests for the SuperSplat PLY writer/reader and the axes -> quaternion conversion (T82, T83)."""

import numpy as np
import pytest
from scipy.stats import special_ortho_group

from viking_splatad.eval.gaussian_occupancy import quaternions_to_rotations
from viking_splatad.export.supersplat_ply import (
    read_supersplat_ply, rotations_to_quaternions_wxyz, write_supersplat_ply)

SEED = 20261121
ROTATION_COUNT = 64
GAUSSIAN_COUNT = 7
SH_DEGREE = 3
# K = (degree + 1)^2 - 1 higher-order coefficients per channel.
REST_PER_CHANNEL = (SH_DEGREE + 1) ** 2 - 1


def expected_header(count: int, rest_per_channel: int) -> str:
    """Header text built from the format rules, independently of the module's helper.

    Args:
        count: Number of vertex rows (N) announced by ``element vertex``.
        rest_per_channel: K higher-order SH coefficients per colour channel (0 = no f_rest).

    Returns:
        The full ASCII header, ``end_header`` line and trailing newline included.
    """
    names = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"]
    # f_rest is channel-major: index channel * K + coefficient.
    for channel in range(3):
        for coefficient in range(rest_per_channel):
            names.append(f"f_rest_{channel * rest_per_channel + coefficient}")
    names += ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    lines = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    lines += [f"property float {name}" for name in names]
    lines.append("end_header")
    return "\n".join(lines) + "\n"


def random_inputs(rest_per_channel: int):
    """Seeded random Gaussian parameters; sh_rest is None when K = 0.

    Args:
        rest_per_channel: K higher-order SH coefficients per colour channel.

    Returns:
        (means (N, 3), opacity (N,), scales (N, 3), quaternions (N, 4), sh_dc (N, 3),
        sh_rest (N, 3, K) or None), N = GAUSSIAN_COUNT, float64. Values are arbitrary
        normals: the round trip only checks bit-exact float32 storage, not validity.
    """
    rng = np.random.default_rng(SEED)
    sh_rest = (rng.normal(size=(GAUSSIAN_COUNT, 3, rest_per_channel))
               if rest_per_channel else None)
    return (rng.normal(size=(GAUSSIAN_COUNT, 3)), rng.normal(size=GAUSSIAN_COUNT),
            rng.normal(size=(GAUSSIAN_COUNT, 3)), rng.normal(size=(GAUSSIAN_COUNT, 4)),
            rng.normal(size=(GAUSSIAN_COUNT, 3)), sh_rest)


@pytest.mark.covers("T83")
def test_quaternion_round_trip_with_mirrored_axes():
    """quaternion -> rotation returns the axes, third column flipped where det was -1."""
    rotations = special_ortho_group.rvs(3, size=ROTATION_COUNT, random_state=SEED)
    rng = np.random.default_rng(SEED)
    # Mirror roughly half of the matrices by negating the second axis (det becomes -1).
    mirrored_mask = rng.random(ROTATION_COUNT) < 0.5
    axes = rotations.copy()
    axes[mirrored_mask, :, 1] *= -1.0
    assert np.all(np.linalg.det(axes)[mirrored_mask] < 0)
    # Output is canonical (w >= 0) and unit length.
    quaternions = rotations_to_quaternions_wxyz(axes)
    assert np.all(quaternions[:, 0] >= 0.0)
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=1e-12)
    # Converting back must give the input axes, with the third column negated on mirrored
    # matrices (the module's det fix), an independent quaternion -> matrix implementation.
    expected = axes.copy()
    expected[mirrored_mask, :, 2] *= -1.0
    np.testing.assert_allclose(quaternions_to_rotations(quaternions), expected, atol=1e-10)


@pytest.mark.covers("T83")
def test_quaternion_rejects_bad_shape():
    """Non (N, 3, 3) input is refused."""
    with pytest.raises(ValueError):
        rotations_to_quaternions_wxyz(np.zeros((4, 3)))


@pytest.mark.covers("T82")
@pytest.mark.parametrize("rest_per_channel", [0, REST_PER_CHANNEL])
def test_write_read_round_trip_and_header(tmp_path, rest_per_channel):
    """Fields read back equal the float32 inputs; header and size match the format rules."""
    means, opacity, scales, quaternions, sh_dc, sh_rest = random_inputs(rest_per_channel)
    path = tmp_path / "out.ply"
    written = write_supersplat_ply(path, means, opacity, scales, quaternions, sh_dc, sh_rest)
    header = expected_header(GAUSSIAN_COUNT, rest_per_channel)
    # Fields per row: position 3 + DC 3 + rest 3K + opacity 1 + scale 3 + quaternion 4;
    # file size = header + N rows * fields * 4 bytes (float32).
    field_count = 3 + 3 + 3 * rest_per_channel + 1 + 3 + 4
    assert written == path.stat().st_size == len(header) + GAUSSIAN_COUNT * field_count * 4
    assert path.read_bytes().startswith(header.encode("ascii"))
    # Reader returns the fields in header order.
    fields = read_supersplat_ply(path)
    assert list(fields) == [line[len("property float "):] for line in header.splitlines()
                            if line.startswith("property float ")]
    # Every field equals its input column rounded to float32 (exact: no other transform).
    for axis, name in enumerate(("x", "y", "z")):
        np.testing.assert_array_equal(fields[name], means[:, axis].astype(np.float32))
    for channel in range(3):
        np.testing.assert_array_equal(fields[f"f_dc_{channel}"],
                                      sh_dc[:, channel].astype(np.float32))
        for coefficient in range(rest_per_channel):
            np.testing.assert_array_equal(
                fields[f"f_rest_{channel * rest_per_channel + coefficient}"],
                sh_rest[:, channel, coefficient].astype(np.float32))
    np.testing.assert_array_equal(fields["opacity"], opacity.astype(np.float32))
    for axis in range(3):
        np.testing.assert_array_equal(fields[f"scale_{axis}"], scales[:, axis].astype(np.float32))
    for part in range(4):
        np.testing.assert_array_equal(fields[f"rot_{part}"],
                                      quaternions[:, part].astype(np.float32))


@pytest.mark.covers("T82")
def test_write_rejects_inconsistent_shapes(tmp_path):
    """A wrong-length opacity vector is refused before any file is written."""
    means, opacity, scales, quaternions, sh_dc, _ = random_inputs(0)
    with pytest.raises(ValueError):
        write_supersplat_ply(tmp_path / "bad.ply", means, opacity[:-1], scales, quaternions,
                             sh_dc, None)
    assert not (tmp_path / "bad.ply").exists()


# Named constants for the added tests.
TRAILING_BYTE_COUNT = 4  # one float32 worth of garbage
TRUNCATED_BYTE_COUNT = 1  # dropping any byte of the last row must be detected
NONFINITE_ROW_INDEX = 2  # arbitrary valid row to poison
NON_ORTHONORMAL_SCALE = 1.01  # 1% scale error, far above the float32-level tolerance
OVERFLOW_FACTOR = 10.0  # multiple of the float32 maximum


def write_valid_ply(path, rest_per_channel=0):
    """Write a valid PLY from the seeded random inputs.

    Args:
        path: Destination file.
        rest_per_channel: K higher-order SH coefficients per channel.

    Returns:
        None.
    """
    write_supersplat_ply(path, *random_inputs(rest_per_channel))


@pytest.mark.covers("T82")
def test_read_rejects_truncated_file(tmp_path):
    """A file missing its last byte is refused."""
    path = tmp_path / "t.ply"
    write_valid_ply(path)
    path.write_bytes(path.read_bytes()[:-TRUNCATED_BYTE_COUNT])
    with pytest.raises(ValueError):
        read_supersplat_ply(path)


@pytest.mark.covers("T82")
def test_read_rejects_trailing_bytes(tmp_path):
    """Extra bytes after the last row are refused."""
    path = tmp_path / "t.ply"
    write_valid_ply(path)
    path.write_bytes(path.read_bytes() + bytes(TRAILING_BYTE_COUNT))
    with pytest.raises(ValueError):
        read_supersplat_ply(path)


@pytest.mark.covers("T82")
def test_read_rejects_missing_format_line(tmp_path):
    """A header without the binary_little_endian line is refused."""
    path = tmp_path / "t.ply"
    write_valid_ply(path)
    data = path.read_bytes().replace(b"format binary_little_endian 1.0\n", b"", 1)
    path.write_bytes(data)
    with pytest.raises(ValueError):
        read_supersplat_ply(path)


@pytest.mark.covers("T82")
def test_write_rejects_2d_sh_rest(tmp_path):
    """A 2-D sh_rest raises ValueError (not IndexError) and writes nothing."""
    means, opacity, scales, quaternions, sh_dc, _ = random_inputs(0)
    with pytest.raises(ValueError):
        write_supersplat_ply(tmp_path / "b.ply", means, opacity, scales, quaternions, sh_dc,
                             np.zeros((GAUSSIAN_COUNT, 3)))
    assert not (tmp_path / "b.ply").exists()


@pytest.mark.covers("T82")
def test_write_rejects_nan_and_leaves_no_file(tmp_path):
    """NaN in any input raises and leaves neither the target nor a temp file."""
    means, opacity, scales, quaternions, sh_dc, _ = random_inputs(0)
    means[NONFINITE_ROW_INDEX, 0] = np.nan
    with pytest.raises(ValueError, match="means"):
        write_supersplat_ply(tmp_path / "n.ply", means, opacity, scales, quaternions, sh_dc, None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.covers("T82")
def test_write_rejects_float32_overflow(tmp_path):
    """A finite float64 above the float32 maximum raises, naming the field."""
    means, opacity, scales, quaternions, sh_dc, _ = random_inputs(0)
    opacity[NONFINITE_ROW_INDEX] = np.finfo(np.float32).max * OVERFLOW_FACTOR
    with pytest.raises(ValueError, match="opacity_logits"):
        write_supersplat_ply(tmp_path / "o.ply", means, opacity, scales, quaternions, sh_dc, None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.covers("T83")
def test_quaternion_rejects_non_orthonormal():
    """A scaled (non-orthonormal) matrix is refused instead of silently projected."""
    rotations = special_ortho_group.rvs(3, size=ROTATION_COUNT, random_state=SEED)
    rotations[0] *= NON_ORTHONORMAL_SCALE
    with pytest.raises(ValueError, match="orthonormal"):
        rotations_to_quaternions_wxyz(rotations)


@pytest.mark.covers("T83")
def test_quaternion_half_turn_canonical_sign_and_round_trip():
    """180 degrees about +z and -z: w == 0, first non-zero of (x, y, z) is positive; round trips."""
    half_turn_about_z = np.diag([-1.0, -1.0, 1.0])
    # Rotation by 180 degrees about any axis n is 2 n n^T - I (computed from first principles).
    axis_matrices = np.stack([2.0 * np.outer(axis, axis) - np.eye(3) for axis in
                              (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0]),
                               np.array([1.0, 0.0, 0.0]), np.array([0.0, -1.0, 0.0]))])
    np.testing.assert_allclose(axis_matrices[0], half_turn_about_z)
    quaternions = rotations_to_quaternions_wxyz(axis_matrices)
    # w is exactly zero for a half turn, and the first non-zero vector component is positive.
    np.testing.assert_array_equal(quaternions[:, 0], 0.0)
    for quaternion in quaternions:
        first_nonzero = quaternion[1:][np.nonzero(quaternion[1:])[0][0]]
        assert first_nonzero > 0.0
    np.testing.assert_allclose(quaternions_to_rotations(quaternions), axis_matrices, atol=1e-12)


@pytest.mark.covers("T82")
def test_written_file_mode_follows_the_umask(tmp_path):
    """The final ply has mode 0o666 & ~umask (as a plain open() would), not the temp file's 0o600."""
    import os

    from viking_splatad.export.supersplat_ply import DEFAULT_NEW_FILE_MODE

    test_umask = 0o027
    previous_umask = os.umask(test_umask)
    try:
        output_path = tmp_path / "mode.ply"
        write_supersplat_ply(output_path, *random_inputs(0))
    finally:
        os.umask(previous_umask)
    expected_mode = DEFAULT_NEW_FILE_MODE & ~test_umask
    assert (output_path.stat().st_mode & 0o777) == expected_mode
