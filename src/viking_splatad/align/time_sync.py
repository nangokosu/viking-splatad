"""Time-offset estimation between sensor streams and radar drift / extrinsic checks.

Conventions: timestamps float64 seconds; poses 4x4 float64 ``T_<target>_<source>``.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.signal import correlate
from scipy.spatial import cKDTree

from viking_splatad.registry import transform


def speed_signal_from_poses(
    times: np.ndarray, poses_world_sensor: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Speed magnitude of a pose stream by finite differences (offset-estimation input).

    Args:
        times: [N] float64 seconds, strictly increasing.
        poses_world_sensor: [N,4,4] float64 ``T_world_sensor``.

    Returns:
        (midpoint_times [N-1], speed [N-1]) with speed = |dp| / dt.
    """
    # Translation column of each pose is the sensor position in the world frame.
    positions = poses_world_sensor[:, :3, 3]
    time_steps = np.diff(times)
    # speed_k = ||p_{k+1} - p_k|| / (t_{k+1} - t_k), attributed to the interval midpoint.
    speed = np.linalg.norm(np.diff(positions, axis=0), axis=1) / time_steps
    midpoint_times = 0.5 * (times[:-1] + times[1:])
    return midpoint_times, speed


@transform("T22", "constant time offset between two 1-D signals via cross-correlation")
def estimate_time_offset(
    times_a: np.ndarray,
    values_a: np.ndarray,
    times_b: np.ndarray,
    values_b: np.ndarray,
    upsample_factor: int = 4,
    max_abs_offset: float = None,
) -> Tuple[float, float]:
    """Estimate delta such that ``b(t) ~ a(t - delta)``.

    Both signals are resampled onto one fine grid, mean-removed, zero-padded outside
    their support, and cross-correlated. The discrete peak is refined with a parabola.

    Args:
        times_a, values_a: Stream A samples (float64 seconds, signal values).
        times_b, values_b: Stream B samples.
        upsample_factor: Grid step = finest median sample spacing / this factor.
            Default 4 gives sub-sample lag resolution before parabolic refinement.
        max_abs_offset: Optional search bound [s]; None searches all lags.

    Returns:
        (offset_seconds, confidence). Confidence = normalised peak minus the median
        absolute normalised correlation in the search window (peak sharpness, <= 1).
    """
    # Common fine grid spanning both streams.
    grid_step = min(np.median(np.diff(times_a)), np.median(np.diff(times_b))) / upsample_factor
    grid_start = min(times_a[0], times_b[0])
    grid_stop = max(times_a[-1], times_b[-1])
    grid_times = np.arange(grid_start, grid_stop + grid_step, grid_step)

    def resample_zero_mean(sample_times: np.ndarray, sample_values: np.ndarray) -> np.ndarray:
        """Linear-resample onto the grid, remove the in-support mean, zero outside support."""
        resampled = np.interp(grid_times, sample_times, sample_values)
        inside_support = (grid_times >= sample_times[0]) & (grid_times <= sample_times[-1])
        # Zero-mean inside the support so DC does not dominate the correlation.
        resampled = np.where(inside_support, resampled - resampled[inside_support].mean(), 0.0)
        return resampled

    grid_a = resample_zero_mean(times_a, values_a)
    grid_b = resample_zero_mean(times_b, values_b)

    # corr[k] = sum_n a[n] * b[n + lag_k]; peak at lag where b(t) = a(t - lag*step).
    correlation = correlate(grid_b, grid_a, mode="full", method="fft")
    lag_samples = np.arange(-(grid_a.size - 1), grid_b.size)
    # Normalise by energies so the peak is a correlation coefficient in [-1, 1].
    energy_product = np.sqrt(np.sum(grid_a**2) * np.sum(grid_b**2))
    correlation = correlation / energy_product

    # Restrict to the allowed lag window.
    search_mask = np.ones_like(lag_samples, dtype=bool)
    if max_abs_offset is not None:
        search_mask = np.abs(lag_samples * grid_step) <= max_abs_offset
    window_indices = np.flatnonzero(search_mask)
    peak_index = window_indices[np.argmax(correlation[window_indices])]

    # Parabolic interpolation through (-1, 0, +1) neighbours:
    # delta = 0.5 (y- - y+) / (y- - 2 y0 + y+), the vertex of the fitted parabola.
    fractional_lag = 0.0
    if 0 < peak_index < correlation.size - 1:
        left, centre, right = correlation[peak_index - 1 : peak_index + 2]
        curvature_denominator = left - 2.0 * centre + right
        if curvature_denominator != 0.0:
            fractional_lag = 0.5 * (left - right) / curvature_denominator

    offset_seconds = float((lag_samples[peak_index] + fractional_lag) * grid_step)
    confidence = float(
        correlation[peak_index] - np.median(np.abs(correlation[window_indices]))
    )
    return offset_seconds, confidence


@dataclass(frozen=True)
class RadarDriftModel:
    """Linear clock-drift model ``offset(t) = offset_at_zero + drift_rate * t``.

    Attributes:
        offset_at_zero: Intercept a [s].
        drift_rate: Slope b [s/s].
        residual_rms: RMS of (measured - model) [s].
    """

    offset_at_zero: float
    drift_rate: float
    residual_rms: float


@transform("T24", "radar clock drift model offset(t)=a+b*t by least squares")
def fit_radar_drift_model(
    window_times: np.ndarray, measured_offsets: np.ndarray
) -> RadarDriftModel:
    """Least-squares line fit of per-window offsets against window time.

    Minimises sum_k (a + b t_k - o_k)^2 via the normal equations (np.linalg.lstsq).

    Args:
        window_times: [W] window centre times [s].
        measured_offsets: [W] offsets measured on each window [s].

    Returns:
        RadarDriftModel with intercept, slope and residual RMS.
    """
    # Design matrix columns: [1, t] -> solution [a, b].
    design_matrix = np.column_stack([np.ones_like(window_times), window_times])
    (intercept, slope), *_ = np.linalg.lstsq(design_matrix, measured_offsets, rcond=None)
    residuals = measured_offsets - design_matrix @ np.array([intercept, slope])
    return RadarDriftModel(
        float(intercept), float(slope), float(np.sqrt(np.mean(residuals**2)))
    )


def radar_extrinsic_residual(
    radar_points: np.ndarray, lidar_points: np.ndarray, T_lidar_radar: np.ndarray
) -> float:
    """Mean nearest-neighbour distance from radar points (in lidar frame) to the lidar cloud.

    Args:
        radar_points: [N,3] radar points in the radar frame.
        lidar_points: [M,3] lidar points in the lidar frame.
        T_lidar_radar: 4x4 float64 extrinsic mapping radar -> lidar.

    Returns:
        Mean over radar points of the distance to the closest lidar point [m].
    """
    # p_lidar = R p_radar + t, applied row-wise as p @ R^T + t.
    rotation, translation = T_lidar_radar[:3, :3], T_lidar_radar[:3, 3]
    radar_in_lidar = radar_points @ rotation.T + translation
    # KD-tree nearest neighbour: O((N + M) log M).
    nearest_distances, _ = cKDTree(lidar_points).query(radar_in_lidar)
    return float(np.mean(nearest_distances))
