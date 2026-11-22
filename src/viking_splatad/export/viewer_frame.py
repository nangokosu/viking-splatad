"""Map frame -> SuperSplat viewer frame (T88), and the sidecar record of the exported frame.

Our Gaussians live in the map frame: right-handed, up = +z, the scene tens of metres from the
origin. SuperSplat (PlayCanvas engine, y-up) rotates every loaded PLY by 180 degrees about z at load
time; its exporter undoes it with ``mat.setFromEulerAngles(0, 0, -180)`` under the comment "we must
undo the transform we apply at load time" (playcanvas/supersplat ``src/splat-serialize.ts``). So a
file is expected in the INRIA/COLMAP convention, up = -y: after Rz(180) = diag(-1, -1, 1) the file's
-y becomes the engine's +y.

The "viewer frame" (the file frame) is therefore
    p_viewer = M (p_map - origin),   M = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]
so map +z (up) -> viewer -y, map +y -> viewer +z, map +x -> viewer +x, and det M = +1 (a proper
rotation: handedness is kept, no mirrored scene). ``origin`` is the mean of the train sensor
positions (the driven path), so the viewer opens on the scene instead of tens of metres away.
"""

import json
import os
import tempfile
from pathlib import Path

import numpy as np

from viking_splatad.export.supersplat_ply import DEFAULT_NEW_FILE_MODE, check_shape
from viking_splatad.registry import transform

# Spatial dimension of positions and directions.
SPATIAL_DIMENSIONS = 3
VECTOR_SHAPE = (SPATIAL_DIMENSIONS,)
MATRIX_SHAPE = (SPATIAL_DIMENSIONS, SPATIAL_DIMENSIONS)

# Map frame -> file frame, rows are the viewer axes written in map coordinates: viewer x = map x,
# viewer y = -map z (map up is file -y), viewer z = map y. det = +1 by construction (checked in the
# tests). Justification for up = -y: SuperSplat applies Rz(180 deg) at load (see module docstring),
# which flips x and y, so the file must carry up on -y for the engine to see +y.
SUPERSPLAT_FROM_MAP_ROTATION = np.array([[1.0, 0.0, 0.0],
                                         [0.0, 0.0, -1.0],
                                         [0.0, 1.0, 0.0]], dtype=np.float64)
SUPERSPLAT_FROM_MAP_ROTATION.setflags(write=False)

# What SuperSplat applies to every loaded PLY: a 180 degree rotation about z,
# Rz(180) = [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]] = diag(-1, -1, 1).
SUPERSPLAT_LOAD_ROTATION = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)
SUPERSPLAT_LOAD_ROTATION.setflags(write=False)

# Tolerance on |R^T R - I| and |det R - 1|. Justification: entries of R^T R are 3-term float64 dot
# products, so honest input is within a few float64 eps; 1e3 eps (~2e-13) leaves margin for rotations
# that were composed or saved as decimal text, while any real scale or shear (>> 1e-9) is rejected.
FLOAT64_ROUNDOFF_MULTIPLE = 1000
ROTATION_TOLERANCE = FLOAT64_ROUNDOFF_MULTIPLE * float(np.finfo(np.float64).eps)

# Sidecar record naming (format chosen by this module).
VIEWER_FRAME_FILE_NAME = "supersplat_frame.json"
VIEWER_FRAME_NAME = "supersplat_viewer"
VIEWER_FRAME_CONVENTION = ("p_viewer = rotation @ (p_map - origin_map); up = -y "
                           "(SuperSplat rotates 180 deg about z at load)")
# A map-frame export is recorded too, so that "no sidecar" can only mean a legacy export. Its transform is
# the identity: origin 0 and rotation I.
MAP_FRAME_NAME = "supersplat_map"
MAP_FRAME_CONVENTION = "p_file = p_map; up = +z (no recentring, no rotation)"
FRAME_NAMES = (VIEWER_FRAME_NAME, MAP_FRAME_NAME)
FRAME_CONVENTIONS = {VIEWER_FRAME_NAME: VIEWER_FRAME_CONVENTION, MAP_FRAME_NAME: MAP_FRAME_CONVENTION}
# Keys every record has. ``ply_bytes`` and ``vertex_count`` are optional on read (records written before
# they existed, or by a caller that has no PLY): a record without ``ply_bytes`` cannot be paired with a file.
RECORD_KEYS = ("frame", "convention", "origin_map", "rotation_viewer_from_map", "crop")
PLY_BYTES_KEY, VERTEX_COUNT_KEY = "ply_bytes", "vertex_count"
# JSON indentation, for human-readable sidecars.
JSON_INDENT = 2


def check_finite(name: str, array: np.ndarray) -> None:
    """Raise ValueError if ``array`` holds NaN or infinity.

    Args:
        name: Argument name for the message.
        array: Array to check.

    Returns:
        None.
    """
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")


def check_proper_rotation(name: str, rotation: np.ndarray) -> None:
    """Raise ValueError unless ``rotation`` is (3, 3), orthonormal and has det +1.

    Orthonormal columns <=> R^T R = I; with that, det R = +-1, and +1 excludes a mirror.

    Args:
        name: Argument name for the message.
        rotation: Candidate rotation matrix.

    Returns:
        None.
    """
    check_shape(name, rotation, MATRIX_SHAPE)
    check_finite(name, rotation)
    # Step 1: orthonormality (max abs entry of R^T R - I).
    gram_error = np.abs(rotation.T @ rotation - np.eye(SPATIAL_DIMENSIONS)).max()
    if gram_error > ROTATION_TOLERANCE:
        raise ValueError(f"{name} is not orthonormal (|R^T R - I| = {gram_error:.3g})")
    # Step 2: handedness.
    determinant = np.linalg.det(rotation)
    if abs(determinant - 1.0) > ROTATION_TOLERANCE:
        raise ValueError(f"{name} is not a proper rotation (det = {determinant:.6g})")


def path_origin(sensor_positions_map: np.ndarray) -> np.ndarray:
    """Origin of the viewer frame: the mean of the train sensor positions (the driven path).

    Args:
        sensor_positions_map: (P, 3) sensor positions in the map frame, P >= 1.

    Returns:
        (3,) float64 mean position.

    Raises:
        ValueError: If the shape is not (P, 3) with P >= 1, or a value is not finite.
    """
    positions = np.asarray(sensor_positions_map, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != SPATIAL_DIMENSIONS or positions.shape[0] < 1:
        raise ValueError(f"sensor_positions_map must be (P, 3) with P >= 1, got {positions.shape}")
    check_finite("sensor_positions_map", positions)
    # Mean over positions: origin = (1/P) sum_p position_p.
    return positions.mean(axis=0)


@transform("T88", "Map frame -> SuperSplat viewer frame: recentre on the train path, rotate "
                  "up=+z to up=-y (p' = M(p - o), R' = M R)")
def map_to_viewer_frame(means_map: np.ndarray, rotations_map: np.ndarray, origin_map: np.ndarray,
                        rotation: np.ndarray = SUPERSPLAT_FROM_MAP_ROTATION
                        ) -> tuple[np.ndarray, np.ndarray]:
    """Express Gaussian means and principal axes in the SuperSplat viewer frame.

    Means move as points: p' = M (p - o). Rotation columns are principal axes, i.e. vectors, so
    each rotates like a vector: R' = M R. The covariance R diag(s^2) R^T then becomes
    M R diag(s^2) R^T M^T = M Sigma M^T, which is the covariance of the transformed Gaussian
    (scales are unchanged, M being a rotation).

    Spherical-harmonic coefficients of degree >= 1 are NOT rotated here. The caller must fit them
    with viewer-frame view directions (scripts/22 does, via ``rotate_directions_to_viewer``).

    Args:
        means_map: (N, 3) Gaussian centres in the map frame.
        rotations_map: (N, 3, 3) matrices whose columns are the principal axes in the map frame.
        origin_map: (3,) map-frame point that becomes the viewer origin.
        rotation: (3, 3) proper rotation M, viewer from map.

    Returns:
        (means_viewer (N, 3), rotations_viewer (N, 3, 3)), float64.

    Raises:
        ValueError: On a wrong shape, a non-finite value, or a non-proper ``rotation``.
    """
    # Step 1: validate everything at the boundary.
    means = np.asarray(means_map, dtype=np.float64)
    axes = np.asarray(rotations_map, dtype=np.float64)
    origin = np.asarray(origin_map, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != SPATIAL_DIMENSIONS:
        raise ValueError(f"means_map must be (N, 3), got {means.shape}")
    check_shape("rotations_map", axes, (means.shape[0], SPATIAL_DIMENSIONS, SPATIAL_DIMENSIONS))
    check_shape("origin_map", origin, VECTOR_SHAPE)
    for argument_name, argument_array in (("means_map", means), ("rotations_map", axes),
                                          ("origin_map", origin)):
        check_finite(argument_name, argument_array)
    check_proper_rotation("rotation", matrix)
    # Step 2: p' = M (p - o), batched as row vectors: (p - o) M^T.
    means_viewer = (means - origin) @ matrix.T
    # Step 3: R' = M R for every Gaussian (broadcast matmul over the leading axis).
    rotations_viewer = matrix @ axes
    return means_viewer, rotations_viewer


@transform("T89", "SuperSplat viewer frame -> map frame for points: p = Mᵀ p' + o (inverse of T88)")
def viewer_to_map_points(means_viewer: np.ndarray, origin_map: np.ndarray,
                         rotation: np.ndarray = SUPERSPLAT_FROM_MAP_ROTATION) -> np.ndarray:
    """Inverse of the point part of ``map_to_viewer_frame``: p = M^T p' + o (M^-1 = M^T).

    Args:
        means_viewer: (N, 3) points in the viewer frame.
        origin_map: (3,) map-frame origin used forward.
        rotation: (3, 3) proper rotation M used forward.

    Returns:
        (N, 3) float64 points in the map frame.

    Raises:
        ValueError: On a wrong shape, a non-finite value, or a non-proper ``rotation``.
    """
    points = np.asarray(means_viewer, dtype=np.float64)
    origin = np.asarray(origin_map, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != SPATIAL_DIMENSIONS:
        raise ValueError(f"means_viewer must be (N, 3), got {points.shape}")
    check_shape("origin_map", origin, VECTOR_SHAPE)
    check_finite("means_viewer", points)
    check_finite("origin_map", origin)
    check_proper_rotation("rotation", matrix)
    # Row-vector form of M^T p': p' M.
    return points @ matrix + origin


@transform("T90", "Map -> SuperSplat viewer frame for viewing directions: d' = M d, renormalised (SH fit frame)")
def rotate_directions_to_viewer(directions_map: np.ndarray,
                                rotation: np.ndarray = SUPERSPLAT_FROM_MAP_ROTATION
                                ) -> np.ndarray:
    """Map-frame directions to the viewer frame: d' = M d, no translation, renormalised.

    Args:
        directions_map: (D, 3) non-zero direction vectors in the map frame.
        rotation: (3, 3) proper rotation M.

    Returns:
        (D, 3) unit vectors in the viewer frame.

    Raises:
        ValueError: On a wrong shape, a non-finite or zero-length direction, or a non-proper
            ``rotation``.
    """
    directions = np.asarray(directions_map, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != SPATIAL_DIMENSIONS:
        raise ValueError(f"directions_map must be (D, 3), got {directions.shape}")
    check_finite("directions_map", directions)
    check_proper_rotation("rotation", matrix)
    # Step 1: rotate (row-vector form of M d).
    rotated = directions @ matrix.T
    # Step 2: renormalise; a rotation keeps length, so this only removes input non-unit length.
    lengths = np.linalg.norm(rotated, axis=1, keepdims=True)
    if np.any(lengths == 0.0):
        raise ValueError("directions_map contains a zero-length direction")
    return rotated / lengths


def identity_frame_transform() -> tuple[np.ndarray, np.ndarray]:
    """Origin and rotation of the map-frame record: no shift (origin 0) and no rotation (identity).

    Returns:
        ((3,) zeros, (3, 3) identity), both float64.
    """
    return np.zeros(SPATIAL_DIMENSIONS), np.eye(SPATIAL_DIMENSIONS)


def check_map_frame_is_identity(origin: np.ndarray, rotation: np.ndarray) -> None:
    """Raise ValueError unless a map-frame record carries the identity transform (origin 0, R = I).

    Args:
        origin: (3,) origin of the record.
        rotation: (3, 3) rotation of the record.

    Returns:
        None.
    """
    expected_origin, expected_rotation = identity_frame_transform()
    if not (np.array_equal(origin, expected_origin) and np.array_equal(rotation, expected_rotation)):
        raise ValueError(f"a {MAP_FRAME_NAME!r} record must have origin 0 and the identity rotation")


def stage_viewer_frame_record(path: Path, origin_map: np.ndarray, rotation: np.ndarray, crop: dict,
                              frame_name: str = VIEWER_FRAME_NAME, ply_bytes: int | None = None,
                              vertex_count: int | None = None) -> Path:
    """Write the sidecar JSON to a temp file beside ``path`` and return the temp path (not yet in place).

    Two-step publication lets the caller order it against the PLY write: stage the sidecar, write the
    PLY, then ``os.replace(staged, path)`` (``write_viewer_frame_record`` does stage + replace in one go).
    The temp file's mode is 0o666 & ~umask, as ``open()`` gives (same approach as ``write_supersplat_ply``).

    Args:
        path: Final destination (usually ``VIEWER_FRAME_FILE_NAME`` beside the PLY); only its directory
            and name are used to place and name the temp file.
        origin_map: (3,) map-frame origin of the frame (zeros for the map frame).
        rotation: (3, 3) proper rotation M, file frame from map (identity for the map frame).
        crop: JSON-serialisable description of the crop applied to the export.
        frame_name: ``VIEWER_FRAME_NAME`` or ``MAP_FRAME_NAME``.
        ply_bytes: Byte size of the PLY this record describes, or None when unknown.
        vertex_count: Number of Gaussians in that PLY, or None when unknown.

    Returns:
        Path of the staged temp file. The caller must replace it into place or delete it.

    Raises:
        ValueError: On a wrong shape, a non-finite value (also inside ``crop``), a non-proper
            ``rotation``, a ``crop`` that is not a dict, an unknown ``frame_name``, a map-frame record
            that is not the identity, or a negative / non-integer ``ply_bytes`` or ``vertex_count``.
    """
    # Step 1: validate, so no malformed record is ever written.
    origin = np.asarray(origin_map, dtype=np.float64)
    matrix = np.asarray(rotation, dtype=np.float64)
    check_shape("origin_map", origin, VECTOR_SHAPE)
    check_finite("origin_map", origin)
    check_proper_rotation("rotation", matrix)
    if not isinstance(crop, dict):
        raise ValueError(f"crop must be a dict, got {type(crop).__name__}")
    if frame_name not in FRAME_NAMES:
        raise ValueError(f"frame_name must be one of {FRAME_NAMES}, got {frame_name!r}")
    if frame_name == MAP_FRAME_NAME:
        check_map_frame_is_identity(origin, matrix)
    for count_name, count_value in ((PLY_BYTES_KEY, ply_bytes), (VERTEX_COUNT_KEY, vertex_count)):
        check_optional_count(count_name, count_value)
    record = {"frame": frame_name, "convention": FRAME_CONVENTIONS[frame_name],
              "origin_map": origin.tolist(), "rotation_viewer_from_map": matrix.tolist(),
              "crop": crop, PLY_BYTES_KEY: ply_bytes, VERTEX_COUNT_KEY: vertex_count}
    # Step 2: temp file beside the target.
    path = Path(path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as record_file:
            # allow_nan=False: NaN/Infinity are not valid JSON and raise ValueError (temp file is removed).
            json.dump(record, record_file, indent=JSON_INDENT, allow_nan=False)
            record_file.write("\n")
        # mkstemp makes mode 0600; the umask can only be read by setting it, so set 0 and restore.
        current_umask = os.umask(0)
        os.umask(current_umask)
        os.chmod(temporary_name, DEFAULT_NEW_FILE_MODE & ~current_umask)
    except BaseException:
        # Never leave the temp file behind.
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return Path(temporary_name)


def check_optional_count(count_name: str, count_value) -> None:
    """Raise ValueError unless ``count_value`` is None or a non-negative integer (bool excluded).

    Args:
        count_name: Key name for the message.
        count_value: Candidate value.

    Returns:
        None.
    """
    if count_value is None:
        return
    # bool is an int subclass; True would silently be read as 1 byte.
    if isinstance(count_value, bool) or not isinstance(count_value, (int, np.integer)) or count_value < 0:
        raise ValueError(f"{count_name} must be None or a non-negative integer, got {count_value!r}")


def write_viewer_frame_record(path: Path, origin_map: np.ndarray, rotation: np.ndarray, crop: dict,
                              frame_name: str = VIEWER_FRAME_NAME, ply_bytes: int | None = None,
                              vertex_count: int | None = None) -> None:
    """Write the sidecar JSON describing the exported frame (viewer or map), atomically.

    ``stage_viewer_frame_record`` then ``os.replace``, so a failure never leaves a partial file.

    Args:
        path: Destination file (usually ``VIEWER_FRAME_FILE_NAME`` beside the PLY).
        origin_map: (3,) map-frame origin of the frame (zeros for the map frame).
        rotation: (3, 3) proper rotation M, file frame from map (identity for the map frame).
        crop: JSON-serialisable description of the crop applied to the export.
        frame_name: ``VIEWER_FRAME_NAME`` (default) or ``MAP_FRAME_NAME``.
        ply_bytes: Byte size of the PLY this record describes, or None when unknown.
        vertex_count: Number of Gaussians in that PLY, or None when unknown.

    Returns:
        None.

    Raises:
        ValueError: As ``stage_viewer_frame_record``.
    """
    staged_path = stage_viewer_frame_record(path, origin_map, rotation, crop, frame_name, ply_bytes,
                                            vertex_count)
    try:
        os.replace(staged_path, path)
    except BaseException:
        staged_path.unlink(missing_ok=True)
        raise


def read_viewer_frame_record(path: Path) -> dict:
    """Read a sidecar written by ``write_viewer_frame_record``, strictly.

    Args:
        path: Sidecar JSON file.

    Returns:
        Dict with ``frame`` (``VIEWER_FRAME_NAME`` or ``MAP_FRAME_NAME``), ``origin_map`` ((3,) array),
        ``rotation`` ((3, 3) array, file frame from map), ``crop`` (dict), and ``ply_bytes`` /
        ``vertex_count`` (int, or None when the record does not hold them).

    Raises:
        ValueError: On a missing key, an unknown frame name, a wrong shape, a non-finite value, a
            non-proper rotation, a map-frame record that is not the identity, or a bad count.
    """
    with open(path, encoding="utf-8") as record_file:
        record = json.load(record_file)
    # Step 1: every key must be present.
    if not isinstance(record, dict):
        raise ValueError("frame record must be a JSON object")
    missing_keys = [key for key in RECORD_KEYS if key not in record]
    if missing_keys:
        raise ValueError(f"frame record is missing keys: {missing_keys}")
    # Step 2: the record must describe one of the frames this module defines.
    if record["frame"] not in FRAME_NAMES:
        raise ValueError(f"frame record 'frame' must be one of {FRAME_NAMES}, got {record['frame']!r}")
    # Step 3: shapes and values.
    origin = np.asarray(record["origin_map"], dtype=np.float64)
    rotation = np.asarray(record["rotation_viewer_from_map"], dtype=np.float64)
    check_shape("origin_map", origin, VECTOR_SHAPE)
    check_finite("origin_map", origin)
    check_proper_rotation("rotation_viewer_from_map", rotation)
    if record["frame"] == MAP_FRAME_NAME:
        check_map_frame_is_identity(origin, rotation)
    if not isinstance(record["crop"], dict):
        raise ValueError("frame record 'crop' must be an object")
    for count_name in (PLY_BYTES_KEY, VERTEX_COUNT_KEY):
        check_optional_count(count_name, record.get(count_name))
    return {"frame": record["frame"], "origin_map": origin, "rotation": rotation, "crop": record["crop"],
            PLY_BYTES_KEY: record.get(PLY_BYTES_KEY), VERTEX_COUNT_KEY: record.get(VERTEX_COUNT_KEY)}


def check_record_matches_ply(record: dict, ply_path: Path) -> None:
    """Raise ValueError unless the PLY's byte size is the one the sidecar was written for.

    The export stages the new sidecar, replaces the PLY and only then the sidecar. A crash between the
    two renames leaves the OLD sidecar next to the NEW PLY; the sizes then differ (a different crop, frame
    or SH degree changes the row count or the row width, so the byte size), and this check catches the
    pair. Equal sizes are necessary, not sufficient, for a match (a same-size re-export with another frame
    would pass), which is why the sidecar is replaced last and not skipped.

    Args:
        record: Output of ``read_viewer_frame_record``.
        ply_path: The PLY beside the sidecar.

    Returns:
        None. A record without ``ply_bytes`` (written before the field existed) is not checkable and passes.

    Raises:
        ValueError: If the file size differs from ``ply_bytes`` (message tells to re-export).
    """
    recorded_bytes = record.get(PLY_BYTES_KEY)
    if recorded_bytes is None:
        return
    actual_bytes = Path(ply_path).stat().st_size
    if actual_bytes != recorded_bytes:
        raise ValueError(f"{ply_path} is {actual_bytes} bytes but its sidecar describes a {recorded_bytes}-byte "
                         "PLY: the pair is inconsistent (interrupted or mixed export); re-export")
