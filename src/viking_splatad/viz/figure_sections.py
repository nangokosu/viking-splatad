"""Vertical "occlusion section" geometry for the hidden-trunk figures.

A section is the vertical plane through a box centre that contains the horizontal ray from the
nearest lidar sweep origin (the sensor) to the box. Points inside a thin slab around that plane
are projected onto it: x = distance along the ray, y = height. Pure numpy/scipy; no I/O except
``session_sweep_origins``, which reads only the poses file.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy.spatial import cKDTree

from viking_splatad.eval.session_hidden_space import HORIZONTAL_AXIS_COUNT
from viking_splatad.eval.session_rays import DEFAULT_LIDAR_SENSOR_NAME
from viking_splatad.splatad.viking_conversions import open_dataset_view

# Space dimension and the translation column of a 4x4 pose.
SPACE_DIMENSION = 3
POSE_TRANSLATION_COLUMN = 3
# Index of the vertical (z) coordinate of a map-frame point.
Z_AXIS_INDEX = 2
# Horizontal distance (m) below which the sensor-to-box direction is undefined. 1 micrometre is
# far below any lidar/box position uncertainty yet well above float64 round-off at map scale.
MINIMUM_HORIZONTAL_DISTANCE_M = 1e-6
# World up axis of the map frame (z up).
UP_AXIS = np.array([0.0, 0.0, 1.0])


def slice_frame(box_centre: np.ndarray, sweep_origin: np.ndarray) -> Dict[str, np.ndarray]:
    """Orthonormal right-handed frame of the vertical section from the sensor to the box.

    along = horizontal unit vector origin -> box; across = up x along; up = +z. Then
    (along, across, up) satisfies along x across = up (right-handed) and all are orthonormal
    because along is horizontal, hence perpendicular to up.

    Args:
        box_centre: (3,) box centre, map frame.
        sweep_origin: (3,) sensor position, map frame.

    Returns:
        dict with ``origin`` (3,), ``along`` (3,), ``across`` (3,), ``up`` (3,).

    Raises:
        ValueError: If the horizontal distance is below ``MINIMUM_HORIZONTAL_DISTANCE_M``.
    """
    # Step 1: horizontal offset from sensor to box (z dropped).
    origin = np.asarray(sweep_origin, dtype=np.float64).reshape(SPACE_DIMENSION)
    offset = np.asarray(box_centre, dtype=np.float64).reshape(SPACE_DIMENSION) - origin
    offset[Z_AXIS_INDEX] = 0.0
    distance = float(np.linalg.norm(offset))
    if distance < MINIMUM_HORIZONTAL_DISTANCE_M:
        raise ValueError(f"box centre is {distance:.3g} m horizontally from the sweep origin; direction undefined")
    # Step 2: normalise, then across = up x along.
    along = offset / distance
    across = np.cross(UP_AXIS, along)
    return {"origin": origin, "along": along, "across": across, "up": UP_AXIS.copy()}


def in_slab(points: np.ndarray, frame: Dict[str, np.ndarray], half_width: float) -> Tuple[np.ndarray, np.ndarray]:
    """Select points within a slab around the section plane and project them onto it.

    Signed lateral offset = (p - origin) . across; kept when |offset| <= half_width.
    Section coordinates are (u, v) = ((p - origin) . along, p_z).

    Args:
        points: (N,3) map-frame points.
        frame: output of ``slice_frame``.
        half_width: slab half thickness (m), >= 0.

    Returns:
        ``(mask (N,) bool, coordinates (M,2))`` with M = mask.sum(), in input order.
    """
    # Step 1: relative positions and lateral offset.
    relative = np.asarray(points, dtype=np.float64).reshape(-1, SPACE_DIMENSION) - frame["origin"]
    lateral_offset = relative @ frame["across"]
    # Step 2: slab mask.
    mask = np.abs(lateral_offset) <= half_width
    # Step 3: (distance along ray, absolute height) of the kept points.
    along_distance = relative[mask] @ frame["along"]
    height = np.asarray(points, dtype=np.float64).reshape(-1, SPACE_DIMENSION)[mask, Z_AXIS_INDEX]
    return mask, np.stack([along_distance, height], axis=1)


def nearest_origin(query_points: np.ndarray, sweep_origins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Nearest sweep origin by horizontal distance (same convention as the hidden-space score).

    Args:
        query_points: (Q,2) or (Q,3) points; only x, y are used.
        sweep_origins: (S,2) or (S,3) origins; only x, y are used.

    Returns:
        ``(indices (Q,) int, origins (Q,3 or as given))`` of the nearest origin per query.
    """
    # Step 1: KD-tree on the horizontal axes of the origins.
    origins = np.asarray(sweep_origins, dtype=np.float64)
    tree = cKDTree(origins[:, :HORIZONTAL_AXIS_COUNT])
    # Step 2: nearest neighbour of each query.
    queries = np.atleast_2d(np.asarray(query_points, dtype=np.float64))
    _, indices = tree.query(queries[:, :HORIZONTAL_AXIS_COUNT])
    return indices, origins[indices]


def session_sweep_origins(dataset_dir: Path, lidar_sensor_name: str = DEFAULT_LIDAR_SENSOR_NAME) -> np.ndarray:
    """Map-frame lidar positions of every recorded sweep, without loading any point cloud.

    Uses ``open_dataset_view`` (manifest + poses.npz + file listing, no frame decoded), the same
    pose source ``iter_session_sweeps`` uses for ``lidar_to_map``. Includes all recorded sweeps
    (train, test and guard), as script 24 does.

    Args:
        dataset_dir: Extracted dataset directory.
        lidar_sensor_name: Lidar stream name.

    Returns:
        (S,3) float64 translations of ``poses_world_sensor``, in time order.
    """
    # Step 1: poses of the lidar stream; the translation column of each 4x4 is the origin.
    view = open_dataset_view(Path(dataset_dir))
    poses = np.asarray(view.sensor_poses[lidar_sensor_name], dtype=np.float64)
    return poses[:, :SPACE_DIMENSION, POSE_TRANSLATION_COLUMN].copy()
