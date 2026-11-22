"""SuperSplat / INRIA 3DGS binary PLY writer and reader (T82, T83).

Layout: ``ply`` / ``format binary_little_endian 1.0`` / ``element vertex N`` / one
``property float <name>`` per field / ``end_header``, then N rows of little-endian float32 in the
order ``x y z f_dc_0..2 [f_rest_0..3K-1] opacity scale_0..2 rot_0..3``.

Conventions: opacity = logit(alpha), scale = log(std) per axis, rot = unit quaternion (w, x, y, z),
``f_rest`` channel-major (``f_rest_{channel * K + coefficient}``), as the INRIA exporter writes
``features_rest.transpose(1, 2).flatten``.

Why hand-written: the format is a fixed float32 table, and ``plyfile`` is not declared in
``pyproject.toml`` (checked). Adding it would need a change of ``docker/constraints.txt`` and the
Dockerfile recipe plus an image rebuild, which this small writer does not justify.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from viking_splatad.registry import transform

# Little-endian float32: the one dtype of every PLY property (format fixed by the 3DGS PLY spec).
PLY_FLOAT_DTYPE = np.dtype("<f4")
# Header lines, fixed by the PLY format.
PLY_MAGIC_LINE = "ply"
PLY_FORMAT_LINE = "format binary_little_endian 1.0"
PLY_END_HEADER_LINE = "end_header"
PLY_VERTEX_PREFIX = "element vertex "
PLY_PROPERTY_PREFIX = "property float "
# Structural sizes of the 3DGS record (position, colour channels, scale axes, quaternion parts).
POSITION_DIMENSIONS = 3
COLOUR_CHANNELS = 3
SCALE_DIMENSIONS = 3
QUATERNION_PARTS = 4
# SuperSplat/INRIA field names that are not indexed families.
POSITION_FIELD_NAMES = ("x", "y", "z")
OPACITY_FIELD_NAME = "opacity"
# A rotation matrix is square with the spatial dimension on both axes.
ROTATION_MATRIX_SHAPE = (POSITION_DIMENSIONS, POSITION_DIMENSIONS)
# scipy's ``as_quat`` is scalar-last (x, y, z, w): the scalar sits at the last quaternion index and
# the vector part is everything before it.
SCIPY_SCALAR_INDEX = QUATERNION_PARTS - 1
SCIPY_SCALAR_SLICE = slice(SCIPY_SCALAR_INDEX, QUATERNION_PARTS)
SCIPY_VECTOR_SLICE = slice(0, SCIPY_SCALAR_INDEX)
# Orthonormality tolerance = FLOAT32_ROUNDOFF_MULTIPLE * float32 eps, written as a multiple of the
# float64 eps. Justification: callers may build the axes in float32, and each entry of R^T R is a
# 3-term dot product, so a few float32 roundings (8 is a generous bound) must still pass, while a
# real scale or shear error (>> 1e-6) is rejected.
FLOAT32_ROUNDOFF_MULTIPLE = 8
ORTHONORMALITY_EPS_FACTOR = (FLOAT32_ROUNDOFF_MULTIPLE * np.finfo(np.float32).eps
                             / np.finfo(np.float64).eps)
ORTHONORMALITY_TOLERANCE = ORTHONORMALITY_EPS_FACTOR * np.finfo(np.float64).eps
# Largest finite value a float32 PLY property can store.
FLOAT32_MAX = float(np.finfo(np.float32).max)

# Mode of a new file before the umask is applied (what open() uses): read/write for owner, group, other.
DEFAULT_NEW_FILE_MODE = 0o666


def build_field_names(rest_coefficients_per_channel: int) -> list[str]:
    """Ordered PLY field names for K higher-order SH coefficients per colour channel.

    Args:
        rest_coefficients_per_channel: K = (degree + 1)^2 - 1, 0 when there is no f_rest.

    Returns:
        Names in file order: x y z, f_dc_*, f_rest_*, opacity, scale_*, rot_*.
    """
    # Step 1: position, then the DC colour term per channel.
    names = list(POSITION_FIELD_NAMES)
    names += [f"f_dc_{channel}" for channel in range(COLOUR_CHANNELS)]
    # Step 2: channel-major higher-order SH, 3K names in total.
    names += [f"f_rest_{rest_index}"
              for rest_index in range(COLOUR_CHANNELS * rest_coefficients_per_channel)]
    # Step 3: opacity, scales, quaternion.
    names.append(OPACITY_FIELD_NAME)
    names += [f"scale_{axis}" for axis in range(SCALE_DIMENSIONS)]
    names += [f"rot_{part}" for part in range(QUATERNION_PARTS)]
    return names


@transform("T83", "principal axes -> unit quaternion (w,x,y,z) for the SuperSplat PLY")
def rotations_to_quaternions_wxyz(rotations: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z), canonical sign, of matrices whose columns are principal axes.

    A Gaussian's covariance R diag(s^2) R^T is unchanged when one axis flips sign (the flip is
    absorbed by R -> R D, D = diag(+-1), D^2 = I), so a det = -1 matrix is made a proper rotation
    by negating its third column. q and -q give the same rotation; the sign is made canonical:
    w > 0, or, when w is exactly 0 (a 180 degree rotation), the first non-zero of (x, y, z) > 0.
    Values of w that are only numerically close to 0 are not tie-broken.

    Args:
        rotations: (N, 3, 3) matrices, columns = principal axes (orthonormal).

    Returns:
        (N, 4) float64 unit quaternions (w, x, y, z).

    Raises:
        ValueError: If the input is not (N, 3, 3), or some R^T R differs from I by more than
            ``ORTHONORMALITY_TOLERANCE`` (max abs entry).
    """
    matrices = np.array(rotations, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1:] != ROTATION_MATRIX_SHAPE:
        raise ValueError(f"rotations must be (N, 3, 3), got {matrices.shape}")
    # Step 1: orthonormality. Columns are orthonormal iff R^T R = I; scipy would otherwise
    # silently project a scaled or sheared matrix onto the nearest rotation.
    gram_matrices = np.swapaxes(matrices, 1, 2) @ matrices
    identity = np.eye(POSITION_DIMENSIONS)
    if matrices.size and not np.all(
            np.abs(gram_matrices - identity).max(axis=(1, 2)) <= ORTHONORMALITY_TOLERANCE):
        raise ValueError(
            f"rotations are not orthonormal (|R^T R - I| > {ORTHONORMALITY_TOLERANCE:.3g})")
    # Step 2: make every matrix a proper rotation (det +1) by negating the third axis if needed.
    mirrored_mask = np.linalg.det(matrices) < 0.0
    matrices[mirrored_mask, :, 2] *= -1.0
    # Step 3: scipy returns (x, y, z, w); reorder to (w, x, y, z).
    quaternions_xyzw = Rotation.from_matrix(matrices).as_quat()
    quaternions_wxyz = np.concatenate(
        [quaternions_xyzw[:, SCIPY_SCALAR_SLICE], quaternions_xyzw[:, SCIPY_VECTOR_SLICE]], axis=1)
    # Step 4: canonical sign. Flip when w < 0; when w == 0 both signs have w >= 0, so flip when the
    # first non-zero vector component is negative.
    scalar_parts = quaternions_wxyz[:, 0]
    vector_parts = quaternions_wxyz[:, 1:]
    first_nonzero_index = np.argmax(vector_parts != 0.0, axis=1)
    first_nonzero_values = vector_parts[np.arange(len(vector_parts)), first_nonzero_index]
    flip_mask = (scalar_parts < 0.0) | ((scalar_parts == 0.0) & (first_nonzero_values < 0.0))
    quaternions_wxyz[flip_mask] *= -1.0
    return quaternions_wxyz


def check_shape(name: str, array: np.ndarray, expected_shape: tuple) -> None:
    """Raise ValueError unless ``array.shape`` equals ``expected_shape``.

    Args:
        name: Argument name for the message.
        array: Array to check.
        expected_shape: Required shape.

    Returns:
        None.
    """
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")


def check_float32_storable(name: str, array: np.ndarray) -> None:
    """Raise ValueError if ``array`` has a non-finite value or one beyond the float32 range.

    Args:
        name: Field name for the message.
        array: Numeric array to check.

    Returns:
        None.
    """
    if array.size == 0:
        return
    # Step 1: NaN and +-inf would be written as-is and break the viewer.
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    # Step 2: finite float64 beyond float32's maximum would silently become +-inf on the cast.
    if np.abs(array).max() > FLOAT32_MAX:
        raise ValueError(f"{name} has values outside the float32 range (|v| > {FLOAT32_MAX:.3g})")


@transform("T82", "SuperSplat 3DGS binary PLY writer")
def write_supersplat_ply(output_path: Path, means: np.ndarray, opacity_logits: np.ndarray,
                         log_scales: np.ndarray, quaternions_wxyz: np.ndarray,
                         sh_dc: np.ndarray, sh_rest: np.ndarray | None) -> int:
    """Write Gaussians as an INRIA 3DGS binary little-endian PLY, atomically.

    The file is written to a temporary file in the destination directory and moved into place with
    ``os.replace``, so a failure never leaves a partial file at ``output_path``.

    Args:
        output_path: Destination file.
        means: (N, 3) positions.
        opacity_logits: (N,) logit(alpha).
        log_scales: (N, 3) log std per axis.
        quaternions_wxyz: (N, 4) unit quaternions (w, x, y, z).
        sh_dc: (N, 3) DC spherical-harmonic coefficients per RGB channel.
        sh_rest: (N, 3, K) higher-order SH, axes (gaussian, colour channel, coefficient), or None
            for K = 0. Channel-major: nerfstudio-style (N, K, 3) must be transposed by the caller
            (``swapaxes(1, 2)``); this cannot be detected when K == 3, so it is not checked.

    Returns:
        Bytes written (header + N * fields * 4).

    Raises:
        ValueError: If a shape disagrees with N or K, sh_rest is not 3-D, or a value is
            non-finite or outside the float32 range.
    """
    # Step 1: validate every shape once, at the boundary (sh_rest ndim before reading shape[2]).
    count = means.shape[0]
    check_shape("means", means, (count, POSITION_DIMENSIONS))
    check_shape("opacity_logits", opacity_logits, (count,))
    check_shape("log_scales", log_scales, (count, SCALE_DIMENSIONS))
    check_shape("quaternions_wxyz", quaternions_wxyz, (count, QUATERNION_PARTS))
    check_shape("sh_dc", sh_dc, (count, COLOUR_CHANNELS))
    if sh_rest is not None and sh_rest.ndim != 3:
        raise ValueError(f"sh_rest must be (N, 3, K), got {sh_rest.ndim}-D shape {sh_rest.shape}")
    rest_per_channel = 0 if sh_rest is None else sh_rest.shape[2]
    if sh_rest is not None:
        check_shape("sh_rest", sh_rest, (count, COLOUR_CHANNELS, rest_per_channel))
    # Step 2: validate values before any file is created.
    named_inputs = {"means": means, "opacity_logits": opacity_logits, "log_scales": log_scales,
                    "quaternions_wxyz": quaternions_wxyz, "sh_dc": sh_dc}
    if sh_rest is not None:
        named_inputs["sh_rest"] = sh_rest
    for input_name, input_array in named_inputs.items():
        check_float32_storable(input_name, input_array)
    # Step 3: fill one structured float32 array column by column, straight from the inputs (no
    # float64 concatenation). sh_rest (N, 3, K) reshaped to (N, 3K) is channel-major: column
    # channel * K + coefficient.
    field_names = build_field_names(rest_per_channel)
    records = np.empty(count, dtype=[(name, PLY_FLOAT_DTYPE) for name in field_names])
    column_blocks = [means, sh_dc]
    if sh_rest is not None:
        column_blocks.append(sh_rest.reshape(count, COLOUR_CHANNELS * rest_per_channel))
    column_blocks += [opacity_logits.reshape(count, 1), log_scales, quaternions_wxyz]
    field_cursor = 0
    for block in column_blocks:
        for block_column in range(block.shape[1]):
            records[field_names[field_cursor]] = block[:, block_column]
            field_cursor += 1
    # Step 4: header then raw rows, into a temp file beside the target, then atomic rename.
    header_lines = [PLY_MAGIC_LINE, PLY_FORMAT_LINE, f"{PLY_VERTEX_PREFIX}{count}"]
    header_lines += [f"{PLY_PROPERTY_PREFIX}{name}" for name in field_names]
    header_lines.append(PLY_END_HEADER_LINE)
    header_bytes = ("\n".join(header_lines) + "\n").encode("ascii")
    output_path = Path(output_path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent, prefix=output_path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(file_descriptor, "wb") as output_file:
            output_file.write(header_bytes)
            records.tofile(output_file)
        # mkstemp creates the file with mode 0600; give the final file the mode a plain open() would:
        # 0o666 & ~umask. The umask can only be read by setting it, so set 0 and restore at once.
        current_umask = os.umask(0)
        os.umask(current_umask)
        os.chmod(temporary_name, DEFAULT_NEW_FILE_MODE & ~current_umask)
        os.replace(temporary_name, output_path)
    except BaseException:
        # Never leave the partial temp file behind.
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return len(header_bytes) + records.nbytes


def read_supersplat_ply(path: Path) -> dict[str, np.ndarray]:
    """Parse a PLY written by ``write_supersplat_ply``.

    Only this writer's output is supported: ``comment`` / ``obj_info`` lines, other property types
    and other elements are rejected. The file size must equal header length + N * row bytes.

    Args:
        path: PLY file.

    Returns:
        Field name -> (N,) float32 array, in file order.

    Raises:
        ValueError: If the header is not the expected binary little-endian float32 layout (the
            format line must precede the element line), or the file is truncated, has trailing
            bytes, or holds a different record count than ``element vertex N``.
    """
    field_names: list[str] = []
    count = None
    format_seen = False
    with open(path, "rb") as input_file:
        # Step 1: parse header lines up to end_header.
        if input_file.readline().decode("ascii").strip() != PLY_MAGIC_LINE:
            raise ValueError("not a PLY file")
        while True:
            line = input_file.readline().decode("ascii").strip()
            if not line:
                raise ValueError("PLY header ended before end_header")
            if line == PLY_END_HEADER_LINE:
                break
            if line == PLY_FORMAT_LINE:
                format_seen = True
            elif line.startswith(PLY_VERTEX_PREFIX):
                if not format_seen:
                    raise ValueError(f"missing {PLY_FORMAT_LINE!r} line before element vertex")
                count = int(line[len(PLY_VERTEX_PREFIX):])
            elif line.startswith(PLY_PROPERTY_PREFIX):
                field_names.append(line[len(PLY_PROPERTY_PREFIX):])
            else:
                raise ValueError(f"unsupported PLY header line: {line!r}")
        if count is None or count < 0:
            raise ValueError("PLY header has no valid element vertex line")
        if not format_seen:
            raise ValueError(f"missing {PLY_FORMAT_LINE!r} line")
        # Step 2: exact size check, header + N rows of one float32 per field.
        record_dtype = np.dtype([(name, PLY_FLOAT_DTYPE) for name in field_names])
        header_length = input_file.tell()
        expected_size = header_length + count * record_dtype.itemsize
        actual_size = os.fstat(input_file.fileno()).st_size
        if actual_size != expected_size:
            raise ValueError(f"PLY size {actual_size} != expected {expected_size} "
                             f"(header {header_length} + {count} x {record_dtype.itemsize}): "
                             "truncated or trailing bytes")
        # Step 3: the rest of the file is exactly count rows.
        records = np.fromfile(input_file, dtype=record_dtype, count=count)
        if len(records) != count:
            raise ValueError(f"read {len(records)} records, header announces {count}")
    return {name: records[name] for name in field_names}
