"""Accumulate train-split radar scans in the map frame (T34)."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from viking_splatad.extract.pose import Trajectory, interpolate_pose_se3
from viking_splatad.registry import transform

# Extracted radar rows follow configs/sessions/*.yaml: x, y, z, power, doppler.
RADAR_XYZ_COLUMNS = slice(0, 3)
RADAR_POWER_COLUMN = 3
RADAR_DOPPLER_COLUMN = 4
RADAR_MINIMUM_COLUMN_COUNT = RADAR_DOPPLER_COLUMN + 1


@dataclass(frozen=True)
class AccumulatedRadarPoints:
    """Map-frame radar points and the measurement context used by T35.

    Attributes:
        points_map: float64 ``[N,3]`` radar targets in the session map frame.
        sensor_origins_map: float64 ``[N,3]`` radar origins paired with the targets.
        corrected_times_s: float64 ``[N]`` clock-corrected absolute seconds.
        scan_indices: int64 ``[N]`` source scan index for persistence/Doppler gates.
        powers: float64 ``[N]`` radar return power.
        ranges_m: float64 ``[N]`` target range from the radar origin.
        doppler_residuals_mps: float64 ``[N]`` measured minus predicted static Doppler.
    """

    points_map: np.ndarray
    sensor_origins_map: np.ndarray
    corrected_times_s: np.ndarray
    scan_indices: np.ndarray
    powers: np.ndarray
    ranges_m: np.ndarray
    doppler_residuals_mps: np.ndarray


def corrected_radar_times(
    scan_times_s: np.ndarray,
    clock_offset_at_zero_s: float,
    clock_drift_rate: float,
) -> np.ndarray:
    """Apply the T24 linear clock model without multiplying drift by Unix epoch time.

    Args:
        scan_times_s: Absolute radar header times ``[S]`` in seconds.
        clock_offset_at_zero_s: Clock offset at the first scan, seconds.
        clock_drift_rate: Linear drift in seconds per elapsed second.

    Returns:
        Corrected absolute times ``t + a + b*(t-t0)`` as float64 ``[S]``.
    """
    scan_times_s = np.asarray(scan_times_s, dtype=np.float64)
    if scan_times_s.ndim != 1 or scan_times_s.size == 0:
        raise ValueError("scan_times_s must be a non-empty one-dimensional array")
    elapsed_seconds = scan_times_s - scan_times_s[0]
    return scan_times_s + float(clock_offset_at_zero_s) + float(clock_drift_rate) * elapsed_seconds


def _radar_origin_velocities(sensor_origins_map: np.ndarray, scan_times_s: np.ndarray) -> np.ndarray:
    """Differentiate radar origins to obtain map-frame sensor linear velocities.

    Args:
        sensor_origins_map: Radar origins ``[S,3]``.
        scan_times_s: Strictly increasing corrected scan times ``[S]``.

    Returns:
        Linear velocity ``[S,3]`` using centred differences internally and one-sided ends.
    """
    if len(scan_times_s) == 1:
        return np.zeros_like(sensor_origins_map)
    if not np.all(np.diff(scan_times_s) > 0.0):
        raise ValueError("corrected radar scan times must be strictly increasing")
    return np.column_stack([
        np.gradient(sensor_origins_map[:, axis_index], scan_times_s, edge_order=1)
        for axis_index in range(sensor_origins_map.shape[1])
    ])


@transform(
    "T34",
    "Train radar scans -> clock-corrected map-frame targets, origins and static-Doppler residuals",
)
def accumulate_radar_points(
    radar_scans: Sequence[np.ndarray],
    scan_times_s: np.ndarray,
    train_scan_mask: np.ndarray,
    base_trajectory: Trajectory,
    transform_base_radar: np.ndarray,
    clock_offset_at_zero_s: float = 0.0,
    clock_drift_rate: float = 0.0,
) -> AccumulatedRadarPoints:
    """Transform selected radar scans to the map frame and model static-target Doppler.

    For a static target with map-frame unit ray ``u`` and radar velocity ``v``, radial
    speed is ``-v dot u``. Radar Doppler sign conventions differ by firmware, so each
    scan chooses the sign (``+`` or ``-``) whose prediction has the smaller median
    absolute residual. This changes only the static filter diagnostic, never geometry.

    Args:
        radar_scans: Sequence of float arrays ``[P_s,>=5]`` ordered by time.
        scan_times_s: Absolute radar header times ``[S]``.
        train_scan_mask: Boolean ``[S]``; false scans are excluded before accumulation.
        base_trajectory: Map-frame base poses covering every corrected selected time.
        transform_base_radar: Rigid ``T_base_radar`` ``[4,4]`` for this session.
        clock_offset_at_zero_s: T24 offset at the first radar scan, seconds.
        clock_drift_rate: T24 drift slope, seconds per elapsed second.

    Returns:
        Concatenated ``AccumulatedRadarPoints`` containing only train scans.

    Raises:
        ValueError: On mismatched inputs, malformed scans/transforms, non-finite values,
            no selected scan, or corrected times outside the base trajectory.
    """
    scan_times_s = np.asarray(scan_times_s, dtype=np.float64)
    train_scan_mask = np.asarray(train_scan_mask, dtype=bool)
    scan_count = len(radar_scans)
    if scan_times_s.shape != (scan_count,) or train_scan_mask.shape != (scan_count,):
        raise ValueError("radar scans, times and train mask must have the same length")
    if not train_scan_mask.any():
        raise ValueError("at least one radar scan must belong to the training split")
    transform_base_radar = np.asarray(transform_base_radar, dtype=np.float64)
    if transform_base_radar.shape != (4, 4) or not np.isfinite(transform_base_radar).all():
        raise ValueError("transform_base_radar must be a finite 4x4 matrix")

    corrected_times = corrected_radar_times(
        scan_times_s, clock_offset_at_zero_s, clock_drift_rate
    )
    selected_scan_indices = np.flatnonzero(train_scan_mask)
    selected_times = corrected_times[selected_scan_indices]
    base_poses = interpolate_pose_se3(base_trajectory, selected_times)
    radar_poses = base_poses @ transform_base_radar
    scan_origins = radar_poses[:, :3, 3]
    scan_velocities = _radar_origin_velocities(scan_origins, selected_times)

    point_blocks = []
    origin_blocks = []
    time_blocks = []
    index_blocks = []
    power_blocks = []
    range_blocks = []
    residual_blocks = []
    for selected_position, source_scan_index in enumerate(selected_scan_indices):
        scan = np.asarray(radar_scans[source_scan_index], dtype=np.float64)
        if scan.ndim != 2 or scan.shape[1] < RADAR_MINIMUM_COLUMN_COUNT:
            raise ValueError(f"radar scan {source_scan_index} must have shape [N,>=5]")
        if not np.isfinite(scan[:, :RADAR_MINIMUM_COLUMN_COUNT]).all():
            raise ValueError(f"radar scan {source_scan_index} contains non-finite values")
        if scan.shape[0] == 0:
            continue
        pose = radar_poses[selected_position]
        points_map = scan[:, RADAR_XYZ_COLUMNS] @ pose[:3, :3].T + pose[:3, 3]
        origins_map = np.repeat(pose[None, :3, 3], scan.shape[0], axis=0)
        ray_vectors = points_map - origins_map
        ranges_m = np.linalg.norm(ray_vectors, axis=1)
        nonzero_range = ranges_m > 0.0
        unit_rays = np.zeros_like(ray_vectors)
        unit_rays[nonzero_range] = ray_vectors[nonzero_range] / ranges_m[nonzero_range, None]
        predicted_doppler = -(unit_rays @ scan_velocities[selected_position])
        measured_doppler = scan[:, RADAR_DOPPLER_COLUMN]
        positive_residual = measured_doppler - predicted_doppler
        negative_residual = measured_doppler + predicted_doppler
        residuals = (
            positive_residual
            if np.median(np.abs(positive_residual)) <= np.median(np.abs(negative_residual))
            else negative_residual
        )
        point_blocks.append(points_map[nonzero_range])
        origin_blocks.append(origins_map[nonzero_range])
        time_blocks.append(np.full(nonzero_range.sum(), selected_times[selected_position]))
        index_blocks.append(np.full(nonzero_range.sum(), source_scan_index, dtype=np.int64))
        power_blocks.append(scan[nonzero_range, RADAR_POWER_COLUMN])
        range_blocks.append(ranges_m[nonzero_range])
        residual_blocks.append(residuals[nonzero_range])
    if not point_blocks:
        raise ValueError("selected radar scans contain no nonzero-range points")
    return AccumulatedRadarPoints(
        points_map=np.concatenate(point_blocks).astype(np.float64),
        sensor_origins_map=np.concatenate(origin_blocks).astype(np.float64),
        corrected_times_s=np.concatenate(time_blocks).astype(np.float64),
        scan_indices=np.concatenate(index_blocks).astype(np.int64),
        powers=np.concatenate(power_blocks).astype(np.float64),
        ranges_m=np.concatenate(range_blocks).astype(np.float64),
        doppler_residuals_mps=np.concatenate(residual_blocks).astype(np.float64),
    )
