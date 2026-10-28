"""Semantic occupancy of map-frame points against GT cuboids (T45, T46), numpy only.

PLAN §5 P1 metrics, per scored class (trunk, young tree, rock):

* **T45 detection rate:** fraction of the class's scored cuboids holding at least ``n_min``
  points, plus the whole curve over n.
* **T46 (part a) neighbourhood point precision:** of the points inside the class's cuboids grown
  horizontally by a margin, the fraction inside a class cuboid. Points far from any object
  (ground, open grass) do not enter. A halo, blur or phantom next to an object lowers it.
* **Voxel IoU / precision / recall** (T33's ``compute_binary_metrics``) inside the same
  neighbourhood, on the GT occupancy grid.

A cuboid or voxel is scored only when it lies within lidar range and vertical FOV of at least one
evaluated sweep pose (as T32, horizontal FOV 360°). No occlusion reasoning is done here.
"""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from viking_splatad.labels.cuboids import Cuboid, compute_cuboid_geometry, points_in_cuboid
from viking_splatad.labels.eval_occupancy import OccupancyMetrics, compute_binary_metrics
from viking_splatad.registry import transform

# Indices of the box's horizontal (x, y) and vertical (z) axes in dimensions_xyz. Every GT box is
# yaw-only (phase1_findings.md: 0 boxes with nonzero qx/qy), so box z is map z.
HORIZONTAL_AXES = (0, 1)


@dataclass(frozen=True)
class DetectionResult:
    """T45 result for one class.

    Attributes:
        cuboid_count: Scored cuboids of the class.
        n_min: Point-count threshold used for ``detection_rate``.
        detection_rate: Fraction of scored cuboids with >= n_min points (0.0 when none are scored).
        rate_by_min_points: rate_by_min_points[k] = fraction with >= k + 1 points, k = 0..max count − 1.
    """

    cuboid_count: int
    n_min: int
    detection_rate: float
    rate_by_min_points: List[float]


@dataclass(frozen=True)
class PrecisionResult:
    """T46 (part a) result for one class.

    Attributes:
        inside_count: Neighbourhood points inside a class cuboid.
        neighbourhood_count: Points in the neighbourhood (class cuboids grown by the margin).
        precision: inside / neighbourhood (None when the neighbourhood holds no point).
    """

    inside_count: int
    neighbourhood_count: int
    precision: Optional[float]


def load_cuboids_json(cuboids_path: Path) -> List[Cuboid]:
    """Read ``cuboids.json`` written by scripts/05_extract_cuboids.py.

    Args:
        cuboids_path: File holding a list of {box_id, class_id, class_name, center_xyz,
            dimensions_xyz (full sizes), quaternion_xyzw, ...}.

    Returns:
        Cuboids in file order.
    """
    return [
        Cuboid(
            box_id=entry["box_id"],
            class_id=int(entry["class_id"]),
            class_name=str(entry["class_name"]),
            center_xyz=tuple(entry["center_xyz"]),
            dimensions_xyz=tuple(entry["dimensions_xyz"]),
            quaternion_xyzw=tuple(entry["quaternion_xyzw"]),
        )
        for entry in json.loads(Path(cuboids_path).read_text())
    ]


def median_half_footprint(class_cuboids: Sequence[Cuboid]) -> float:
    """Median horizontal half-size of a class: median over boxes of (dx + dy) / 4.

    (dx + dy) / 2 is a box's mean footprint width; half of it is the radius-like size used as the
    neighbourhood margin, so the neighbourhood is as wide again as a typical object on each side.

    Args:
        class_cuboids: Boxes of one class (non-empty).

    Returns:
        Margin in metres.
    """
    footprint_widths = [
        (cuboid.dimensions_xyz[HORIZONTAL_AXES[0]] + cuboid.dimensions_xyz[HORIZONTAL_AXES[1]]) / 2.0
        for cuboid in class_cuboids
    ]
    return float(np.median(footprint_widths)) / 2.0


def grow_cuboid_horizontally(cuboid: Cuboid, margin: float) -> Cuboid:
    """Same box with its x and y full sizes grown by 2·margin (margin on each face), z unchanged.

    Args:
        cuboid: Box.
        margin: Growth per face (m).

    Returns:
        Grown box.
    """
    grown_dimensions = list(cuboid.dimensions_xyz)
    for axis_index in HORIZONTAL_AXES:
        grown_dimensions[axis_index] += 2.0 * margin
    return replace(cuboid, dimensions_xyz=tuple(grown_dimensions))


def cuboid_point_masks(points_map: np.ndarray, cuboids: Sequence[Cuboid]) -> List[np.ndarray]:
    """Index sets of the points inside each cuboid.

    Each box first keeps only the points in its axis-aligned bounding box (min/max of its corners),
    then runs the exact rotated test (``points_in_cuboid``). Cost O(boxes · N) comparisons for the
    prefilter, exact test only on the few survivors.

    Args:
        points_map: (N, 3) map-frame points.
        cuboids: Boxes.

    Returns:
        Per box, (k,) int64 indices of its points.
    """
    point_indices_per_cuboid = []
    for cuboid in cuboids:
        geometry = compute_cuboid_geometry(cuboid)
        # Step 1: axis-aligned bounding-box prefilter (a superset of the rotated box).
        lower_corner = geometry.corners.min(axis=0)
        upper_corner = geometry.corners.max(axis=0)
        candidate_indices = np.flatnonzero(np.all((points_map >= lower_corner) & (points_map <= upper_corner), axis=1))
        # Step 2: exact point-in-rotated-box test on the candidates.
        inside_mask = points_in_cuboid(geometry, points_map[candidate_indices])
        point_indices_per_cuboid.append(candidate_indices[inside_mask])
    return point_indices_per_cuboid


def cuboid_point_counts(points_map: np.ndarray, cuboids: Sequence[Cuboid]) -> np.ndarray:
    """Number of points inside each cuboid.

    Args:
        points_map: (N, 3) map-frame points.
        cuboids: Boxes.

    Returns:
        (B,) int64 counts.
    """
    return np.array([indices.size for indices in cuboid_point_masks(points_map, cuboids)], dtype=np.int64)


def points_in_any_cuboid(points_map: np.ndarray, cuboids: Sequence[Cuboid]) -> np.ndarray:
    """Which points lie inside at least one of the cuboids.

    Args:
        points_map: (N, 3) map-frame points.
        cuboids: Boxes.

    Returns:
        (N,) bool.
    """
    inside_any = np.zeros(points_map.shape[0], dtype=bool)
    for point_indices in cuboid_point_masks(points_map, cuboids):
        inside_any[point_indices] = True
    return inside_any


def visible_from_any_pose(
    points_map: np.ndarray, sensor_to_map_poses: Sequence[np.ndarray], max_range: float, vertical_half_fov: float
) -> np.ndarray:
    """Which map-frame points lie within range and vertical FOV of at least one sensor pose.

    Same test as T32 with a 360° horizontal FOV: p_s = Rᵀ (p − t) for each pose, kept when
    |p_s| <= max_range and |atan2(z_s, hypot(x_s, y_s))| <= vertical_half_fov. O(poses · N).

    Args:
        points_map: (N, 3) points.
        sensor_to_map_poses: (4, 4) sensor poses in the map frame.
        max_range: Range limit (m).
        vertical_half_fov: Half the vertical FOV (rad).

    Returns:
        (N,) bool.
    """
    visible = np.zeros(points_map.shape[0], dtype=bool)
    for sensor_to_map in sensor_to_map_poses:
        # Row-vector form of R^T (p - t): (p - t) @ R.
        points_sensor = (points_map - sensor_to_map[:3, 3]) @ sensor_to_map[:3, :3]
        horizontal_distance = np.hypot(points_sensor[:, 0], points_sensor[:, 1])
        in_range = np.linalg.norm(points_sensor, axis=1) <= max_range
        in_fov = np.abs(np.arctan2(points_sensor[:, 2], horizontal_distance)) <= vertical_half_fov
        visible |= in_range & in_fov
    return visible


def n_min_from_real_counts(real_point_counts: np.ndarray, quantile: float) -> Optional[int]:
    """Detection floor from the real lidar: a quantile of the per-cuboid real point counts.

    Only cuboids the real sweeps hit (count >= 1) enter; ``method="lower"`` returns an observed count,
    so the threshold is an integer the real sensor actually produced.

    Args:
        real_point_counts: (B,) real-sweep points per scored cuboid of one class.
        quantile: Quantile in [0, 1] (config ``eval.n_min_quantile``).

    Returns:
        The threshold, or None when the real sweeps hit no cuboid of the class.
    """
    hit_counts = real_point_counts[real_point_counts > 0]
    if hit_counts.size == 0:
        return None
    return int(np.quantile(hit_counts, quantile, method="lower"))


@transform("T45", "Per-class cuboid detection rate: fraction of scored cuboids with >= n_min points (+ curve)")
def cuboid_detection_rate(point_counts: np.ndarray, n_min: int) -> DetectionResult:
    """Detection rate of one class's scored cuboids, and the rate as a function of the threshold.

    rate(n) = |{b : count_b >= n}| / B. rate is non-increasing in n and reaches 0 above the maximum
    count, so the curve stops at n = max count.

    Args:
        point_counts: (B,) points inside each scored cuboid.
        n_min: Threshold for the headline rate (>= 1).

    Returns:
        DetectionResult.
    """
    cuboid_count = int(point_counts.size)
    if cuboid_count == 0:
        return DetectionResult(0, n_min, 0.0, [])
    # rate(n) for n = 1..max count, by counting boxes at or above each threshold.
    max_count = int(point_counts.max())
    rate_by_min_points = [float(np.count_nonzero(point_counts >= threshold)) / cuboid_count
                          for threshold in range(1, max_count + 1)]
    detection_rate = float(np.count_nonzero(point_counts >= n_min)) / cuboid_count
    return DetectionResult(cuboid_count, n_min, detection_rate, rate_by_min_points)


@transform("T46", "Neighbourhood point precision: points in class cuboids / points in cuboids grown by a margin")
def neighbourhood_point_precision(
    points_map: np.ndarray, class_cuboids: Sequence[Cuboid], margin: float
) -> PrecisionResult:
    """Fraction of the points near a class's objects that are inside one of them.

    Neighbourhood = union of the class boxes, each grown horizontally by ``margin`` per face (same z
    span). precision = |points in a box| / |points in the neighbourhood|; every box is inside its grown
    box, so the numerator is a subset of the denominator and precision is in [0, 1].

    Args:
        points_map: (N, 3) map-frame points.
        class_cuboids: Scored boxes of one class.
        margin: Horizontal growth per face (m).

    Returns:
        PrecisionResult.
    """
    # Step 1: neighbourhood membership, then box membership among those points only.
    grown_cuboids = [grow_cuboid_horizontally(cuboid, margin) for cuboid in class_cuboids]
    neighbourhood_points = points_map[points_in_any_cuboid(points_map, grown_cuboids)]
    inside_count = int(np.count_nonzero(points_in_any_cuboid(neighbourhood_points, class_cuboids)))
    neighbourhood_count = int(neighbourhood_points.shape[0])
    # Step 2: ratio; undefined (None) with an empty neighbourhood.
    precision = inside_count / neighbourhood_count if neighbourhood_count else None
    return PrecisionResult(inside_count, neighbourhood_count, precision)


def cuboid_voxel_mask(
    cuboids: Sequence[Cuboid], grid_origin: np.ndarray, voxel_size: float, grid_shape: Sequence[int]
) -> np.ndarray:
    """Voxels of a grid whose centre lies inside at least one cuboid (T31's centre rule).

    Each box only tests the voxels of its corner bounding box, so the cost is the boxes' volume in
    voxels, not the grid's.

    Args:
        cuboids: Boxes.
        grid_origin: (3,) grid minimum corner.
        voxel_size: Voxel edge (m).
        grid_shape: (nx, ny, nz).

    Returns:
        Bool array of ``grid_shape``.
    """
    mask = np.zeros(tuple(grid_shape), dtype=bool)
    shape_array = np.asarray(grid_shape)
    for cuboid in cuboids:
        geometry = compute_cuboid_geometry(cuboid)
        # Step 1: index range of the corner bounding box, clipped to the grid.
        lower_index = np.clip(np.floor((geometry.corners.min(axis=0) - grid_origin) / voxel_size).astype(int), 0, shape_array)
        upper_index = np.clip(np.floor((geometry.corners.max(axis=0) - grid_origin) / voxel_size).astype(int) + 1, 0, shape_array)
        if np.any(upper_index <= lower_index):
            continue
        # Step 2: centres of that block, exact box test, write back.
        block_indices = np.stack(np.meshgrid(
            *[np.arange(lower_index[axis], upper_index[axis]) for axis in range(3)], indexing="ij"
        ), axis=-1).reshape(-1, 3)
        block_centers = grid_origin + (block_indices + 0.5) * voxel_size
        inside_indices = block_indices[points_in_cuboid(geometry, block_centers)]
        mask[inside_indices[:, 0], inside_indices[:, 1], inside_indices[:, 2]] = True
    return mask


def neighbourhood_voxel_metrics(
    occupied: np.ndarray, ground_truth_class: np.ndarray, neighbourhood: np.ndarray
) -> OccupancyMetrics:
    """Voxel IoU / precision / recall of one class inside its neighbourhood region.

    Recall = |occupied ∩ GT| / |GT| (GT ⊂ neighbourhood, so it is the class's full recall within the
    scored region); precision and IoU count only false positives inside the neighbourhood.

    Args:
        occupied: Bool grid of voxels holding rendered points.
        ground_truth_class: Bool grid of the class's GT voxels (already restricted to the scored region).
        neighbourhood: Bool grid of the scored neighbourhood region.

    Returns:
        OccupancyMetrics from T33's ``compute_binary_metrics``.
    """
    return compute_binary_metrics(occupied[neighbourhood], ground_truth_class[neighbourhood])


def scored_cuboids_by_class(
    cuboids: Sequence[Cuboid], class_ids: Sequence[int], cuboid_visible: np.ndarray
) -> Dict[int, List[Cuboid]]:
    """Visible cuboids of each scored class.

    Args:
        cuboids: All GT boxes.
        class_ids: Scored class ids.
        cuboid_visible: (B,) bool, box centre within range/FOV of an evaluated pose.

    Returns:
        class id -> its visible boxes (possibly empty).
    """
    return {
        class_id: [cuboid for cuboid, visible in zip(cuboids, cuboid_visible) if visible and cuboid.class_id == class_id]
        for class_id in class_ids
    }


def class_margin_table(cuboids: Sequence[Cuboid], class_ids: Sequence[int]) -> Dict[int, Optional[float]]:
    """Neighbourhood margin per class from all of the session's boxes of that class.

    Args:
        cuboids: All GT boxes.
        class_ids: Scored class ids.

    Returns:
        class id -> ``median_half_footprint`` (None when the session has no box of the class).
    """
    margins: Dict[int, Optional[float]] = {}
    for class_id in class_ids:
        class_cuboids = [cuboid for cuboid in cuboids if cuboid.class_id == class_id]
        margins[class_id] = median_half_footprint(class_cuboids) if class_cuboids else None
    return margins


def vertical_half_fov_from_elevations(elevation_mapping_radians: Dict[str, float]) -> float:
    """Half vertical FOV = largest |beam elevation| of the manifest's elevation mapping.

    Args:
        elevation_mapping_radians: ring -> elevation (rad), ``manifest.lidar.elevation_mapping``.

    Returns:
        Half FOV (rad).
    """
    return float(max(abs(float(elevation)) for elevation in elevation_mapping_radians.values()))


def cuboid_centers(cuboids: Sequence[Cuboid]) -> np.ndarray:
    """(B, 3) centres of the boxes (zeros-shaped (0, 3) when empty).

    Args:
        cuboids: Boxes.

    Returns:
        Centres in the map frame.
    """
    return np.array([cuboid.center_xyz for cuboid in cuboids], dtype=np.float64).reshape(-1, 3)


def summarise_metrics(metrics: OccupancyMetrics) -> Dict[str, float]:
    """Plain dict of an OccupancyMetrics (for JSON).

    Args:
        metrics: T33 metrics.

    Returns:
        Field name -> value.
    """
    return {
        "true_positives": metrics.true_positives, "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives, "iou": metrics.iou,
        "precision": metrics.precision, "recall": metrics.recall,
    }


def range_limit_from_real_points(real_points_per_sweep: Sequence[np.ndarray]) -> float:
    """Lidar range limit = farthest real return over the evaluated sweeps (lidar frame norms).

    Args:
        real_points_per_sweep: (N_s, 3) real returns per sweep.

    Returns:
        Max range (m); 0.0 without any return.
    """
    ranges = [np.linalg.norm(points, axis=1).max() for points in real_points_per_sweep if points.size]
    return float(max(ranges)) if ranges else 0.0


def visible_cuboid_mask(
    cuboids: Sequence[Cuboid], sensor_to_map_poses: Sequence[np.ndarray], max_range: float, vertical_half_fov: float
) -> np.ndarray:
    """Which boxes have their centre within range/FOV of an evaluated pose.

    Args:
        cuboids: Boxes.
        sensor_to_map_poses: Evaluated sweep poses.
        max_range: Range limit (m).
        vertical_half_fov: Half vertical FOV (rad).

    Returns:
        (B,) bool.
    """
    return visible_from_any_pose(cuboid_centers(cuboids), sensor_to_map_poses, max_range, vertical_half_fov)


def voxel_centers(voxel_indices: np.ndarray, grid_origin: np.ndarray, voxel_size: float) -> np.ndarray:
    """Map-frame centres of voxels: origin + (index + 0.5) · voxel_size.

    Args:
        voxel_indices: (V, 3) indices.
        grid_origin: (3,) grid minimum corner.
        voxel_size: Voxel edge (m).

    Returns:
        (V, 3) centres.
    """
    return grid_origin + (voxel_indices + 0.5) * voxel_size


def restrict_mask_to_visible(
    mask: np.ndarray, grid_origin: np.ndarray, voxel_size: float,
    sensor_to_map_poses: Sequence[np.ndarray], max_range: float, vertical_half_fov: float,
) -> np.ndarray:
    """Keep only the voxels of a mask whose centre is visible from an evaluated pose.

    Args:
        mask: Bool grid.
        grid_origin: (3,) grid minimum corner.
        voxel_size: Voxel edge (m).
        sensor_to_map_poses: Evaluated sweep poses.
        max_range: Range limit (m).
        vertical_half_fov: Half vertical FOV (rad).

    Returns:
        Bool grid, a subset of ``mask``.
    """
    set_indices = np.argwhere(mask)
    visible = visible_from_any_pose(
        voxel_centers(set_indices, grid_origin, voxel_size), sensor_to_map_poses, max_range, vertical_half_fov
    )
    restricted = np.zeros_like(mask)
    kept_indices = set_indices[visible]
    restricted[kept_indices[:, 0], kept_indices[:, 1], kept_indices[:, 2]] = True
    return restricted
