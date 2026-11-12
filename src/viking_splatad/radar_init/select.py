"""Label-free selection of persistent trunk-region radar candidates (T35)."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from viking_splatad.radar_init.accumulate import AccumulatedRadarPoints
from viking_splatad.registry import transform

# Gaussian-equivalent scale of a normal distribution from its median absolute deviation.
MAD_TO_STANDARD_DEVIATION = 1.4826
# Persistence means recurrence, so one scan can never establish it; two is the
# mathematical minimum number of distinct observations rather than a tuned threshold.
MINIMUM_PERSISTENT_SCAN_COUNT = 2


@dataclass(frozen=True)
class RadarCandidateVoxels:
    """Persistent radar voxels retained as label-free trunk candidates."""

    means_map: np.ndarray
    sensor_origins_map: np.ndarray
    support_counts: np.ndarray
    time_spans_s: np.ndarray
    median_powers: np.ndarray


def derived_radar_voxel_size(
    ranges_m: np.ndarray,
    range_resolution_m: float,
    azimuth_resolution_rad: float,
    elevation_resolution_rad: float,
) -> float:
    """Derive one accumulation voxel edge from the radar resolution at median range.

    Args:
        ranges_m: Positive radar ranges.
        range_resolution_m: Range resolution in metres.
        azimuth_resolution_rad: Azimuth angular resolution in radians.
        elevation_resolution_rad: Elevation angular resolution in radians.

    Returns:
        Maximum of range and median-range lateral/elevation resolution, metres.
    """
    ranges_m = np.asarray(ranges_m, dtype=np.float64)
    if ranges_m.size == 0 or not np.isfinite(ranges_m).all() or np.any(ranges_m <= 0.0):
        raise ValueError("ranges_m must contain finite positive values")
    if min(range_resolution_m, azimuth_resolution_rad, elevation_resolution_rad) <= 0.0:
        raise ValueError("radar resolutions must be positive")
    median_range = float(np.median(ranges_m))
    return float(max(
        range_resolution_m,
        median_range * np.tan(azimuth_resolution_rad),
        median_range * np.tan(elevation_resolution_rad),
    ))


def _static_doppler_mask(
    scan_indices: np.ndarray,
    residuals_mps: np.ndarray,
    sigma_multiplier: float,
) -> np.ndarray:
    """Robust per-scan Doppler inlier mask centred on each scan's median residual."""
    static_mask = np.zeros(len(residuals_mps), dtype=bool)
    for scan_index in np.unique(scan_indices):
        scan_mask = scan_indices == scan_index
        scan_residuals = residuals_mps[scan_mask]
        residual_median = np.median(scan_residuals)
        robust_sigma = MAD_TO_STANDARD_DEVIATION * np.median(
            np.abs(scan_residuals - residual_median)
        )
        if robust_sigma == 0.0:
            static_mask[scan_mask] = scan_residuals == residual_median
        else:
            static_mask[scan_mask] = (
                np.abs(scan_residuals - residual_median) <= sigma_multiplier * robust_sigma
            )
    return static_mask


def _voxel_group_indices(points: np.ndarray, voxel_size_m: float) -> tuple:
    """Return inverse group ids and lexicographically ordered integer voxel coordinates."""
    voxel_coordinates = np.floor(points / voxel_size_m).astype(np.int64)
    unique_coordinates, inverse_indices = np.unique(voxel_coordinates, axis=0, return_inverse=True)
    return inverse_indices, unique_coordinates


def _local_lidar_ground(
    lidar_points_map: np.ndarray, query_xy: np.ndarray, cell_size_m: float
) -> tuple:
    """Nearest occupied XY cell's minimum LiDAR height and global above-ground heights."""
    lidar_xy_cells = np.floor(lidar_points_map[:, :2] / cell_size_m).astype(np.int64)
    unique_cells, inverse_cells = np.unique(lidar_xy_cells, axis=0, return_inverse=True)
    ground_heights = np.full(len(unique_cells), np.inf, dtype=np.float64)
    np.minimum.at(ground_heights, inverse_cells, lidar_points_map[:, 2])
    cell_centres = (unique_cells.astype(np.float64) + 0.5) * cell_size_m
    ground_tree = cKDTree(cell_centres)
    _, query_ground_indices = ground_tree.query(query_xy)
    point_ground_heights = ground_heights[inverse_cells]
    positive_heights = lidar_points_map[:, 2] - point_ground_heights
    return ground_heights[query_ground_indices], positive_heights[positive_heights > 0.0]


@transform(
    "T35",
    "Map-frame radar points -> static, persistent, trunk-height, LiDAR-sparse candidate voxels",
)
def select_radar_candidates(
    accumulated: AccumulatedRadarPoints,
    train_lidar_points_map: np.ndarray,
    range_resolution_m: float,
    azimuth_resolution_rad: float,
    elevation_resolution_rad: float,
    doppler_sigma_multiplier: float,
    support_percentile: float,
    canopy_height_percentile: float,
) -> RadarCandidateVoxels:
    """Select label-free radar voxels likely to represent occluded trunk structure.

    The threshold values are derived from the supplied data: voxel size comes from radar
    resolution at median range, persistence is a percentile of distinct-scan support,
    ground is the minimum LiDAR height per XY cell, and canopy onset is a percentile of
    positive LiDAR heights above those local ground cells. LiDAR sparsity uses each
    candidate's range-dependent angular resolution instead of a fixed radius.

    Args:
        accumulated: T34 output containing train radar measurements.
        train_lidar_points_map: Finite map-frame training LiDAR returns ``[L,3]``.
        range_resolution_m: Radar range resolution in metres.
        azimuth_resolution_rad: Radar azimuth resolution in radians.
        elevation_resolution_rad: Radar elevation resolution in radians.
        doppler_sigma_multiplier: Robust per-scan static inlier multiplier.
        support_percentile: Percentile ``[0,100]`` of voxel distinct-scan counts.
        canopy_height_percentile: Percentile ``(0,100)`` of positive LiDAR heights.

    Returns:
        Candidate voxel centroids, origins, support, time spans and median power.

    Raises:
        ValueError: On invalid configuration/data or when a gate removes every candidate.
    """
    lidar_points = np.asarray(train_lidar_points_map, dtype=np.float64)
    if lidar_points.ndim != 2 or lidar_points.shape[1] != 3 or lidar_points.shape[0] == 0:
        raise ValueError("train_lidar_points_map must have shape [L,3] with L > 0")
    if not np.isfinite(lidar_points).all():
        raise ValueError("train_lidar_points_map must be finite")
    if doppler_sigma_multiplier <= 0.0:
        raise ValueError("doppler_sigma_multiplier must be positive")
    if not 0.0 <= support_percentile <= 100.0:
        raise ValueError("support_percentile must be in [0,100]")
    if not 0.0 < canopy_height_percentile < 100.0:
        raise ValueError("canopy_height_percentile must be in (0,100)")

    static_mask = _static_doppler_mask(
        accumulated.scan_indices,
        accumulated.doppler_residuals_mps,
        doppler_sigma_multiplier,
    )
    if not static_mask.any():
        raise ValueError("Doppler gate removed every radar point")
    voxel_size_m = derived_radar_voxel_size(
        accumulated.ranges_m[static_mask],
        range_resolution_m,
        azimuth_resolution_rad,
        elevation_resolution_rad,
    )
    static_points = accumulated.points_map[static_mask]
    static_origins = accumulated.sensor_origins_map[static_mask]
    static_times = accumulated.corrected_times_s[static_mask]
    static_scans = accumulated.scan_indices[static_mask]
    static_powers = accumulated.powers[static_mask]
    inverse_voxels, unique_voxels = _voxel_group_indices(static_points, voxel_size_m)
    voxel_count = len(unique_voxels)

    means = np.zeros((voxel_count, 3), dtype=np.float64)
    origins = np.zeros((voxel_count, 3), dtype=np.float64)
    support_counts = np.zeros(voxel_count, dtype=np.int64)
    time_spans = np.zeros((voxel_count, 2), dtype=np.float64)
    median_powers = np.zeros(voxel_count, dtype=np.float64)
    for voxel_index in range(voxel_count):
        voxel_mask = inverse_voxels == voxel_index
        means[voxel_index] = static_points[voxel_mask].mean(axis=0)
        origins[voxel_index] = static_origins[voxel_mask].mean(axis=0)
        support_counts[voxel_index] = np.unique(static_scans[voxel_mask]).size
        time_spans[voxel_index] = [static_times[voxel_mask].min(), static_times[voxel_mask].max()]
        median_powers[voxel_index] = np.median(static_powers[voxel_mask])

    support_threshold = max(
        MINIMUM_PERSISTENT_SCAN_COUNT,
        np.percentile(support_counts, support_percentile, method="higher"),
    )
    persistent_mask = support_counts >= support_threshold
    ground_heights, positive_lidar_heights = _local_lidar_ground(
        lidar_points, means[:, :2], voxel_size_m
    )
    if positive_lidar_heights.size == 0:
        raise ValueError("LiDAR points contain no positive height above local ground")
    canopy_onset_m = np.percentile(positive_lidar_heights, canopy_height_percentile)
    height_above_ground = means[:, 2] - ground_heights
    trunk_height_mask = (
        (height_above_ground >= range_resolution_m)
        & (height_above_ground <= canopy_onset_m)
    )

    candidate_ranges = np.linalg.norm(means - origins, axis=1)
    lateral_resolution = np.maximum(
        candidate_ranges * np.tan(azimuth_resolution_rad),
        candidate_ranges * np.tan(elevation_resolution_rad),
    )
    nearest_lidar_distances, _ = cKDTree(lidar_points).query(means)
    lidar_sparse_mask = nearest_lidar_distances > lateral_resolution
    keep_mask = persistent_mask & trunk_height_mask & lidar_sparse_mask
    if not keep_mask.any():
        raise ValueError("persistence, height and LiDAR-sparsity gates retained no radar voxel")
    return RadarCandidateVoxels(
        means_map=means[keep_mask],
        sensor_origins_map=origins[keep_mask],
        support_counts=support_counts[keep_mask],
        time_spans_s=time_spans[keep_mask],
        median_powers=median_powers[keep_mask],
    )
