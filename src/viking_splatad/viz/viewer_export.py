"""Pure-numpy helpers that turn SplatAD run outputs into FiftyOne inputs.

Nothing here imports FiftyOne, so the tests run in the reference image. The viewer is display
only: GT cuboids become labels, never SplatAD inputs, seeds or loss targets (CLAUDE.md rule 6).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from viking_splatad.registry import transform

# Binary PCD layout FiftyOne reads: x, y, z as float32 plus one float32 whose 4 bytes are the packed
# 0x00RRGGBB colour. For lidar FiftyOne treats the r channel as intensity (its documented convention).
PCD_FIELD_NAMES = ("x", "y", "z", "rgb")
PCD_BYTES_PER_POINT = 4 * len(PCD_FIELD_NAMES)
# Euler order of FiftyOne's 3D ``rotation``: upper-case = intrinsic x-y-z, as in FiftyOne's own
# grouped-dataset example (``Rotation.from_euler("XYZ", rotation)``).
FIFTYONE_EULER_ORDER = "XYZ"


@transform("T69", "GT cuboid quaternion (x, y, z, w) -> FiftyOne 3D Detection rotation (intrinsic XYZ Euler, radians)")
def quaternion_to_fiftyone_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Convert a box orientation to the Euler triple a FiftyOne 3D ``Detection`` stores.

    The box's rotation matrix is R = Rx(a) Ry(b) Rz(c) for intrinsic XYZ angles (a, b, c); scipy
    inverts that factorisation. Angles lie in [-pi, pi] as FiftyOne expects.

    Args:
        quaternion_xyzw: (4,) unit quaternion of R_map_box (scipy's x, y, z, w order).

    Returns:
        (3,) float64 Euler angles (radians) for ``fo.Detection(rotation=...)``.
    """
    # Step 1: normalise defensively; scipy rejects zero-norm input.
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    unit_quaternion = quaternion / np.linalg.norm(quaternion)
    # Step 2: factorise into intrinsic XYZ Euler angles.
    return Rotation.from_quat(unit_quaternion).as_euler(FIFTYONE_EULER_ORDER)


@transform("T70", "Sweep points (lidar frame) -> map frame with lidar_to_map, for display")
def sweep_points_to_map(points_lidar: np.ndarray, lidar_to_map: np.ndarray) -> np.ndarray:
    """Move points from a sweep's lidar frame to the map frame shared with the GT cuboids.

    p_map = R p + t, applied to row vectors as p @ R^T + t (the same rule as T68).

    Args:
        points_lidar: (N, 3) points, lidar frame.
        lidar_to_map: (4, 4) rigid pose of the sweep.

    Returns:
        (N, 3) float64 map-frame points.
    """
    pose = np.asarray(lidar_to_map, dtype=np.float64)
    return np.asarray(points_lidar, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]


def scale_to_unit_range(values: np.ndarray) -> np.ndarray:
    """Min-max scale values to [0, 1] (all zeros when the values are constant or empty).

    Args:
        values: (N,) numbers.

    Returns:
        (N,) float64 in [0, 1].
    """
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    value_range = values.max() - values.min()
    # A constant column has no contrast to show; avoid dividing by zero.
    if value_range == 0:
        return np.zeros_like(values)
    return (values - values.min()) / value_range


def pack_rgb_float32(colors_uint8: np.ndarray) -> np.ndarray:
    """Pack (N, 3) uint8 RGB into the float32 'rgb' field of a PCD file.

    The 24-bit colour (r << 16 | g << 8 | b) is stored in a uint32 whose bytes are reinterpreted
    as float32 (PCL's convention).

    Args:
        colors_uint8: (N, 3) uint8 red, green, blue.

    Returns:
        (N,) float32 whose bit pattern is the packed colour.
    """
    colors = np.asarray(colors_uint8, dtype=np.uint32)
    packed_colors = (colors[:, 0] << 16) | (colors[:, 1] << 8) | colors[:, 2]
    return packed_colors.astype(np.uint32).view(np.float32)


def write_binary_pcd(pcd_path: Path, points_xyz: np.ndarray, colors_uint8: np.ndarray) -> None:
    """Write a binary PCD v0.7 file with x, y, z and packed rgb (FiftyOne's format).

    Args:
        pcd_path: Output file.
        points_xyz: (N, 3) coordinates.
        colors_uint8: (N, 3) uint8 colours (for lidar, intensity goes in the red channel).

    Raises:
        ValueError: If the two arrays disagree in length or the points are not (N, 3).
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float32)
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N, 3), got {points_xyz.shape}")
    if len(colors_uint8) != len(points_xyz):
        raise ValueError(f"{len(colors_uint8)} colours for {len(points_xyz)} points")
    point_count = len(points_xyz)
    # Step 1: header; WIDTH x HEIGHT = N x 1 describes an unorganised cloud.
    header_lines = [
        "# .PCD v0.7 - Point Cloud Data file format",
        "VERSION 0.7",
        "FIELDS " + " ".join(PCD_FIELD_NAMES),
        "SIZE " + " ".join(["4"] * len(PCD_FIELD_NAMES)),
        "TYPE F F F F",
        "COUNT " + " ".join(["1"] * len(PCD_FIELD_NAMES)),
        f"WIDTH {point_count}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {point_count}",
        "DATA binary",
    ]
    # Step 2: one structured record per point, then raw bytes after the header.
    records = np.empty(point_count, dtype=[(name, "<f4") for name in PCD_FIELD_NAMES])
    records["x"], records["y"], records["z"] = points_xyz[:, 0], points_xyz[:, 1], points_xyz[:, 2]
    records["rgb"] = pack_rgb_float32(colors_uint8)
    with open(pcd_path, "wb") as pcd_file:
        pcd_file.write(("\n".join(header_lines) + "\n").encode("ascii"))
        pcd_file.write(records.tobytes())


def read_binary_pcd(pcd_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read back a file written by ``write_binary_pcd`` (used by the tests and for checks).

    Args:
        pcd_path: PCD file with the x, y, z, rgb layout.

    Returns:
        ((N, 3) float32 points, (N, 3) uint8 colours).
    """
    raw_bytes = Path(pcd_path).read_bytes()
    # The header ends after the line "DATA binary\n"; everything after is the point records.
    data_marker = b"DATA binary\n"
    payload_start = raw_bytes.index(data_marker) + len(data_marker)
    records = np.frombuffer(raw_bytes[payload_start:], dtype=[(name, "<f4") for name in PCD_FIELD_NAMES])
    points_xyz = np.stack([records["x"], records["y"], records["z"]], axis=1)
    packed_colors = records["rgb"].copy().view(np.uint32)
    colors = np.stack([(packed_colors >> 16) & 255, (packed_colors >> 8) & 255, packed_colors & 255], axis=1)
    return points_xyz, colors.astype(np.uint8)


def intensity_colors(intensity: np.ndarray) -> np.ndarray:
    """Grey colours whose red channel carries the scaled intensity (FiftyOne reads r as intensity).

    Args:
        intensity: (N,) raw intensities.

    Returns:
        (N, 3) uint8, intensity min-max scaled to 0..255 in all three channels.
    """
    scaled_intensity = np.round(scale_to_unit_range(intensity) * np.iinfo(np.uint8).max).astype(np.uint8)
    return np.repeat(scaled_intensity[:, None], 3, axis=1)
