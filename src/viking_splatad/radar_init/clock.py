"""Doppler/odometry clock calibration with an explicit, conservative train-only search domain."""

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from viking_splatad.registry import transform


@transform("T73", "Robust radar ego velocity and residual uncertainty from static-majority Doppler")
def doppler_ego_velocity(points: np.ndarray, doppler: np.ndarray):
    """Fit radar-frame ego velocity robustly from stationary majority radial returns.

    Args:
        points: Radar xyz [N,3]; doppler: signed receding-positive radial speeds [N].

    Returns:
        Velocity [3], radial residual sigma and conservative velocity sigma [m/s].
        The sign is tested separately against odometry rather than assumed correct.
    """
    points, doppler = np.asarray(points, dtype=np.float64), np.asarray(doppler, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or doppler.shape != (len(points),):
        raise ValueError("Doppler fitting requires matching xyz and speed vectors")
    ranges = np.linalg.norm(points, axis=1)
    if len(points) < points.shape[1] or not np.isfinite(points).all() or not np.isfinite(doppler).all() or np.any(ranges <= 0):
        raise ValueError("finite nonzero radar returns are required")
    directions = points / ranges[:, None]
    singular_values = np.linalg.svd(directions, compute_uv=False)
    if singular_values[-1] <= np.finfo(float).eps * max(directions.shape) * singular_values[0]:
        raise ValueError("radar ray geometry cannot identify 3D ego velocity")
    initial, *_ = np.linalg.lstsq(directions, -doppler, rcond=None)
    initial_residual = directions @ initial + doppler
    floor = np.finfo(float).eps * max(1., np.abs(doppler).max())
    sigma = max(floor, float(np.median(np.abs(initial_residual - np.median(initial_residual))) / norm.ppf(0.75)))

    def residual(velocity):
        """Return radial model mismatch; stationary targets obey Doppler = -u dot v."""
        return directions @ velocity + doppler

    fit = least_squares(residual, initial, loss="soft_l1", f_scale=sigma)
    if not fit.success:
        raise ValueError("Doppler ego-velocity fitting did not converge")
    radial_residual = residual(fit.x)
    sigma = max(floor, float(np.median(np.abs(radial_residual - np.median(radial_residual))) / norm.ppf(0.75)))
    return fit.x, sigma, sigma / singular_values[-1]


@transform("T75", "Radar calibration observations that stay training throughout the affine clock search")
def safe_clock_fit_mask(radar_times: np.ndarray, lidar_frames, offset_limit: float,
                        drift_limit: float, clock_reference_seconds: float) -> np.ndarray:
    """Keep observations that are training under every clock candidate in the search box.

    Args:
        radar_times: Uncorrected relative seconds [N]; lidar_frames: sorted reference split.
        offset_limit, drift_limit: Nonnegative absolute parameter search bounds.
        clock_reference_seconds: Affine-clock centre used by drift.

    Returns:
        Mask [N]. Prefix sums reject any interval that can intersect a test/guard cell;
        boundary ties and reference tails are conservatively excluded.
    """
    radar_times = np.asarray(radar_times, dtype=np.float64)
    reference = np.array([frame.time_s for frame in lidar_frames])
    if (len(reference) < 2 or np.any(np.diff(reference) <= 0)
            or not np.isfinite(radar_times).all()
            or not np.isfinite([offset_limit, drift_limit, clock_reference_seconds]).all()
            or offset_limit <= 0 or drift_limit <= 0):
        raise ValueError("ordered reference frames and positive finite clock search limits required")
    boundaries = (reference[:-1] + reference[1:]) / 2
    excursion = offset_limit + drift_limit * np.abs(radar_times - clock_reference_seconds)
    lower, upper = radar_times - excursion, radar_times + excursion
    left_cells = np.searchsorted(boundaries, lower, side="left")
    right_cells = np.searchsorted(boundaries, upper, side="right")
    blocked_prefix = np.cumsum([0] + [int(frame.split != "train") for frame in lidar_frames])
    return ((lower > reference[0]) & (upper < reference[-1])
            & ((blocked_prefix[right_cells + 1] - blocked_prefix[left_cells]) == 0))


@transform("T74", "Bounded affine radar clock fit against odometry with identifiability/uncertainty gates")
def fit_affine_clock(radar_times: np.ndarray, observed_velocity: np.ndarray,
                      velocity_sigma: np.ndarray, reference_times: np.ndarray,
                      reference_velocity: np.ndarray, offset_limit: float,
                      drift_limit: float, clock_reference_seconds: float,
                      confidence_quantile: float):
    """Fit corrected_time = radar_time + offset + drift*(radar_time-centre).

    Args:
        radar_times, observed_velocity, velocity_sigma: Conservatively train-only
            samples [N], Doppler velocities [N,3], positive uncertainties [N].
        reference_times, reference_velocity: Odometry radar-origin velocities [M], [M,3].
        offset_limit, drift_limit: Explicit restricted search bounds; no extrapolation.
        clock_reference_seconds: Affine-clock centre; confidence_quantile: two-sided gate.

    Returns:
        Measured clock parameters, fit covariance, timing uncertainty and diagnostics.
        An unidentifiable/boundary-truncated fit fails instead of publishing zero offsets.
    """
    radar_times, observed_velocity, velocity_sigma, reference_times, reference_velocity = (
        np.asarray(value, dtype=np.float64) for value in
        (radar_times, observed_velocity, velocity_sigma, reference_times, reference_velocity))
    if (radar_times.ndim != 1 or len(radar_times) <= 2 or observed_velocity.shape != (len(radar_times), 3)
            or velocity_sigma.shape != radar_times.shape or np.any(velocity_sigma <= 0)
            or reference_velocity.shape != (len(reference_times), 3) or len(reference_times) < 2
            or np.any(np.diff(reference_times) <= 0) or not 0 < confidence_quantile < 1
            or not all(np.isfinite(value).all() for value in
                       (radar_times, observed_velocity, velocity_sigma, reference_times, reference_velocity))
            or not np.isfinite([offset_limit, drift_limit, clock_reference_seconds]).all()
            or offset_limit <= 0 or drift_limit <= 0):
        raise ValueError("invalid clock calibration inputs/search bounds")
    excursion = offset_limit + drift_limit * np.abs(radar_times - clock_reference_seconds)
    if ((radar_times - excursion).min() < reference_times[0]
            or (radar_times + excursion).max() > reference_times[-1]):
        raise ValueError("clock search extends outside measured odometry")

    def residual(parameters):
        """Whiten velocity mismatch at candidate corrected times, using measured per-scan noise."""
        corrected = radar_times + parameters[0] + parameters[1] * (radar_times - clock_reference_seconds)
        predicted = np.column_stack([np.interp(corrected, reference_times, reference_velocity[:, axis])
                                     for axis in range(reference_velocity.shape[1])])
        return ((predicted - observed_velocity) / velocity_sigma[:, None]).reshape(-1)

    limits = np.array([offset_limit, drift_limit])
    fit = least_squares(residual, np.zeros(len(limits)), bounds=(-limits, limits),
                         x_scale=limits, loss="soft_l1", f_scale=1.)
    if not fit.success or np.linalg.matrix_rank(fit.jac) < len(limits):
        raise ValueError("clock fit is unidentifiable or did not converge")
    degrees_of_freedom = fit.fun.size - len(limits)
    covariance = np.linalg.inv(fit.jac.T @ fit.jac) * (fit.fun @ fit.fun) / degrees_of_freedom
    uncertainty = np.sqrt(np.maximum(0., np.diag(covariance)))
    confidence_multiplier = norm.ppf((1 + confidence_quantile) / 2)
    numerical_margin = np.sqrt(np.finfo(float).eps) * limits
    if np.any(np.abs(fit.x) + confidence_multiplier * uncertainty >= limits - numerical_margin):
        raise ValueError("clock fit/uncertainty touches the restricted search boundary; independent calibration required")
    clock_design = np.column_stack([np.ones(len(radar_times)), radar_times - clock_reference_seconds])
    timing_variance = np.einsum("ni,ij,nj->n", clock_design, covariance, clock_design)
    return {"offset_seconds": float(fit.x[0]), "drift_rate": float(fit.x[1]),
            "clock_reference_seconds": float(clock_reference_seconds),
            "parameter_covariance": covariance.tolist(),
            "residual_rms_seconds": float(np.sqrt(np.maximum(0., timing_variance).mean())),
            "normalized_velocity_rms": float(np.sqrt(np.mean(fit.fun**2))),
            "cost": float(fit.cost), "offset_search_limit_seconds": float(offset_limit),
            "drift_search_limit": float(drift_limit)}
