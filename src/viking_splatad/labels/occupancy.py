"""Cuboid voxelisation (T31) and per-frame ground-truth crop (T32).

Grids live in the lidar MAP frame (z-up). Voxel ``(ix, iy, iz)`` has its centre at
``origin + (index + 0.5) * voxel_size``. Class id 0 means empty / ground.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from viking_splatad.labels.cuboids import Cuboid, compute_cuboid_geometry, points_in_cuboid
from viking_splatad.registry import transform

# Per-point labelling priority of the original tools: the first class in this list
# that contains a point wins (rock, lamp/sign, trunk, car, bush, building, canopy).
# Small/foreground objects precede large enclosing ones such as canopy boxes.
DEFAULT_CLASS_PRIORITY: Tuple[int, ...] = (3, 7, 1, 5, 4, 6, 2)

# Value stored in empty voxels (ground is implicit, never a box).
EMPTY_CLASS_ID = 0


@dataclass(frozen=True)
class OccupancyGrid:
    """Class-id occupancy grid in the map frame.

    Attributes:
        class_ids: (nx, ny, nz) integer array, 0 = empty.
        origin: (3,) map-frame coordinates of the grid's minimum corner.
        voxel_size: Edge length of a voxel (meters).
    """

    class_ids: np.ndarray
    origin: np.ndarray
    voxel_size: float

    def occupied_voxel_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return map-frame centres and class ids of all occupied voxels.

        Returns:
            ((N, 3) centres, (N,) class ids) in C-order of the grid.
        """
        occupied_indices = np.argwhere(self.class_ids != EMPTY_CLASS_ID)
        # Centre = origin + (index + half a voxel) * voxel_size.
        centers = self.origin + (occupied_indices + 0.5) * self.voxel_size
        class_ids = self.class_ids[tuple(occupied_indices.T)]
        return centers, class_ids


def compute_default_bounds(cuboids: Sequence[Cuboid], margin_meters: float) -> Tuple[np.ndarray, np.ndarray]:
    """Union of box axis-aligned extents, padded by a margin.

    Args:
        cuboids: Non-empty boxes.
        margin_meters: Padding added on every side.

    Returns:
        (minimum corner, maximum corner), each (3,).
    """
    # Stack the 8 corners of every box; the AABB of a rotated box is the min/max of its corners.
    all_corners = np.concatenate([compute_cuboid_geometry(box).corners for box in cuboids])
    # Per-axis min/max over all corners = union AABB, then pad each side.
    return all_corners.min(axis=0) - margin_meters, all_corners.max(axis=0) + margin_meters


@transform("T31", "Voxelise cuboids into a class-id occupancy grid in the map frame")
def voxelize_cuboids(
    cuboids: Sequence[Cuboid],
    voxel_size: float,
    bounds: Optional[Tuple[Sequence[float], Sequence[float]]] = None,
    margin: Optional[float] = None,
    class_priority: Sequence[int] = DEFAULT_CLASS_PRIORITY,
) -> OccupancyGrid:
    """Rasterise boxes into a class-id grid; a voxel is set if its centre is inside a box.

    Complexity: for each box only voxels inside its axis-aligned bounding box are
    tested, so cost is sum over boxes of (AABB volume / voxel_size^3), not
    (#boxes x grid size). Discretisation error: only voxels whose centre-vs-surface
    classification flips are wrong; those lie within sqrt(3)*voxel_size of the
    surface, so |volume error| <= O(surface_area * voxel_size).

    Overlap rule: each voxel takes the class with the lowest rank in
    ``class_priority`` among boxes containing its centre. Classes missing from the
    priority list rank after all listed ones, ordered by class id.

    Args:
        cuboids: Boxes in the map frame.
        voxel_size: Voxel edge length (meters), > 0.
        bounds: Optional (min_xyz, max_xyz); default = union of box extents + margin.
        margin: Padding for default bounds; default one voxel so boundary voxels fit.
        class_priority: Class ids, highest priority first.

    Returns:
        OccupancyGrid covering the bounds.

    Raises:
        ValueError: If voxel_size <= 0 or no boxes and no bounds are given.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    if bounds is None:
        if not cuboids:
            raise ValueError("bounds are required when there are no cuboids")
        padding = voxel_size if margin is None else margin
        bounds_min, bounds_max = compute_default_bounds(cuboids, padding)
    else:
        bounds_min = np.asarray(bounds[0], dtype=np.float64)
        bounds_max = np.asarray(bounds[1], dtype=np.float64)

    # Grid shape: ceil so the grid fully covers [min, max] along each axis.
    grid_shape = np.ceil((bounds_max - bounds_min) / voxel_size).astype(int)
    class_ids = np.full(tuple(grid_shape), EMPTY_CLASS_ID, dtype=np.int32)
    # Rank of the class currently stored per voxel (lower wins); infinity = empty.
    best_rank = np.full(tuple(grid_shape), np.inf)
    rank_by_class = {class_id: rank for rank, class_id in enumerate(class_priority)}

    for box in cuboids:
        # Unlisted classes rank after every listed one, ordered by class id.
        box_rank = rank_by_class.get(box.class_id, len(class_priority) + box.class_id)
        geometry = compute_cuboid_geometry(box)
        # Voxel index range of the box's AABB, clipped to the grid.
        index_low = np.maximum(np.floor((geometry.corners.min(axis=0) - bounds_min) / voxel_size).astype(int), 0)
        index_high = np.minimum(np.ceil((geometry.corners.max(axis=0) - bounds_min) / voxel_size).astype(int), grid_shape)
        if np.any(index_high <= index_low):
            continue  # box entirely outside the grid
        # Enumerate every voxel index in the AABB window as an (nx, ny, nz, 3) mesh.
        axis_index_ranges = [np.arange(low, high) for low, high in zip(index_low, index_high)]
        index_mesh = np.stack(np.meshgrid(*axis_index_ranges, indexing="ij"), axis=-1)
        # Voxel centres in the map frame, then exact point-in-box test (box may be rotated).
        centers = bounds_min + (index_mesh + 0.5) * voxel_size
        inside_mask = points_in_cuboid(geometry, centers.reshape(-1, 3)).reshape(index_mesh.shape[:3])
        # Write only voxels inside the box that this box outranks.
        window = tuple(slice(low, high) for low, high in zip(index_low, index_high))
        wins = inside_mask & (box_rank < best_rank[window])
        class_ids[window][wins] = box.class_id
        best_rank[window][wins] = box_rank
    return OccupancyGrid(class_ids, bounds_min, float(voxel_size))


@transform("T32", "Crop occupancy grid to a sensor's range/FOV, in sensor-frame coordinates")
def crop_occupancy_to_sensor_fov(
    grid: OccupancyGrid,
    T_map_sensor: np.ndarray,
    max_range: float,
    horizontal_fov_radians: float,
    vertical_fov_radians: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return occupied voxels visible in a sensor's field of view.

    Sensor frame: x forward, y left, z up. A voxel centre p_s = T_sensor_map @ p_map
    is kept when |p_s| <= max_range, |atan2(y, x)| <= horizontal_fov/2 and
    |atan2(z, hypot(x, y))| <= vertical_fov/2 (a full 2*pi horizontal FOV keeps all
    azimuths: atan2 returns values in [-pi, pi], so |azimuth| <= pi always holds).
    No occlusion reasoning is done. Cost is O(occupied voxels).

    Args:
        grid: Map-frame occupancy grid.
        T_map_sensor: (4, 4) float64 sensor pose in the map frame.
        max_range: Maximum Euclidean range (meters).
        horizontal_fov_radians: Total horizontal field of view.
        vertical_fov_radians: Total vertical field of view.

    Returns:
        ((N, 3) sensor-frame voxel centres, (N,) class ids).
    """
    centers_map, class_ids = grid.occupied_voxel_centers()
    # T_sensor_map = inverse of the rigid transform: R^T and -R^T t.
    rotation_map_sensor = T_map_sensor[:3, :3]
    translation_map_sensor = T_map_sensor[:3, 3]
    centers_sensor = (centers_map - translation_map_sensor) @ rotation_map_sensor
    # Row-vector form: (p - t) @ R == R^T (p - t) for each centre p.
    forward, left, up = centers_sensor[:, 0], centers_sensor[:, 1], centers_sensor[:, 2]
    # Range test, azimuth (yaw about z) test, elevation (angle above the xy plane) test.
    range_ok = np.linalg.norm(centers_sensor, axis=1) <= max_range
    azimuth_ok = np.abs(np.arctan2(left, forward)) <= horizontal_fov_radians / 2.0
    elevation_ok = np.abs(np.arctan2(up, np.hypot(forward, left))) <= vertical_fov_radians / 2.0
    keep = range_ok & azimuth_ok & elevation_ok
    return centers_sensor[keep], class_ids[keep]
