"""Training-only radar geometry, uncertainty and persistent occupancy anchors.

Anchors are fixed loss targets, not semantic trunk labels or additional Gaussians.
Every threshold is supplied explicitly or derived from the measured uncertainty.
"""

import numpy as np
from scipy.stats import norm

from viking_splatad.extract.pose import Trajectory, interpolate_pose_se3
from viking_splatad.registry import transform


def require_rigid_transform(matrix: np.ndarray) -> np.ndarray:
    """Validate a finite proper SE(3) matrix and return its float64 representation.

    Args:
        matrix: Homogeneous transform [4,4].

    Returns:
        Validated matrix; reflections, scaling and non-homogeneous rows are rejected.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("expected finite homogeneous transform [4,4]")
    rotation = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(rotation.T @ rotation, np.eye(3))
            or not np.isclose(np.linalg.det(rotation), 1.0)):
        raise ValueError("transform must have unit scale and proper rotation")
    return matrix


@transform("T34", "Clock-corrected radar detections and sensor origins into the map frame")
def accumulate_radar(points: np.ndarray, radar_times: np.ndarray, trajectory: Trajectory,
                     base_from_radar: np.ndarray, offset_seconds: float,
                     drift_rate: float, clock_reference_seconds: float):
    """Apply map<-base(t+offset+drift*(t-reference))<-radar without extrapolation.

    Args:
        points: Radar xyz [N,3]; radar_times: matching session-relative seconds [N].
        trajectory: Base poses in the same session-relative reference clock.
        base_from_radar: June-specific rigid extrinsic [4,4].
        offset_seconds, drift_rate, clock_reference_seconds: Measured affine clock model.

    Returns:
        Map points [N,3], actual map-frame radar origins [N,3], corrected times [N].
    """
    points, radar_times = np.asarray(points, dtype=np.float64), np.asarray(radar_times, dtype=np.float64)
    extrinsic = require_rigid_transform(base_from_radar)
    if (points.shape != (len(radar_times), 3) or radar_times.ndim != 1
            or not np.isfinite(points).all() or not np.isfinite(radar_times).all()
            or not np.isfinite([offset_seconds, drift_rate, clock_reference_seconds]).all()
            or drift_rate <= -1):
        raise ValueError("invalid radar points, timestamps or monotonic affine clock")
    corrected_times = radar_times + offset_seconds + drift_rate * (radar_times - clock_reference_seconds)
    if corrected_times.size and (corrected_times.min() < trajectory.timestamps[0]
                                 or corrected_times.max() > trajectory.timestamps[-1]):
        raise ValueError("corrected radar timestamps exceed trajectory support")
    if not len(points):
        return points.copy(), points.copy(), corrected_times
    # T3 normally clamps query times; the bounds check above prevents silent extrapolation here.
    map_from_radar = interpolate_pose_se3(trajectory, corrected_times) @ extrinsic
    map_points = np.einsum("nij,nj->ni", map_from_radar[:, :3, :3], points) + map_from_radar[:, :3, 3]
    return map_points, map_from_radar[:, :3, 3], corrected_times


@transform("T36", "Radar resolution-cell covariance for loss anchors; no colour or Gaussian creation")
def resolution_covariances(points: np.ndarray, range_std_m: float,
                           azimuth_cell_radians: float, elevation_cell_radians: float) -> np.ndarray:
    """Return radar-frame covariance under an explicitly uniform angular-cell model.

    Args:
        points: Nonzero finite radar xyz [N,3].
        range_std_m: Measured radial standard deviation, not resolution/beamwidth.
        azimuth_cell_radians, elevation_cell_radians: Full angular cell widths in radians.

    Returns:
        SPD matrices [N,3,3]. A uniform cell of full width w has variance w**2/12;
        this is a quantization model, not an undocumented FWHM-to-sigma conversion.
    """
    points = np.asarray(points, dtype=np.float64)
    widths = np.asarray([range_std_m, azimuth_cell_radians, elevation_cell_radians])
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not np.isfinite(widths).all() or np.any(widths <= 0)):
        raise ValueError("finite xyz and positive measured uncertainty/cell widths required")
    ranges = np.linalg.norm(points, axis=1)
    if np.any(ranges <= 0):
        raise ValueError("zero-range radar return has no defined ray basis")
    radial = points / ranges[:, None]
    azimuth = np.column_stack([-radial[:, 1], radial[:, 0], np.zeros(len(points))])
    horizontal_norm = np.linalg.norm(azimuth, axis=1)
    # At a pole azimuth is undefined; choose x as a deterministic orthogonal tangent.
    azimuth[horizontal_norm == 0] = [1, 0, 0]
    azimuth /= np.linalg.norm(azimuth, axis=1)[:, None]
    elevation = np.cross(radial, azimuth)
    basis = np.stack([radial, azimuth, elevation], axis=-1)
    # 12 is the analytic variance denominator of Uniform[-w/2,w/2].
    variances = np.column_stack([np.full(len(points), range_std_m**2),
                                 (ranges * azimuth_cell_radians)**2 / 12,
                                 (ranges * elevation_cell_radians)**2 / 12])
    return np.einsum("nij,nj,nkj->nik", basis, variances, basis)


def static_doppler_mask(points: np.ndarray, doppler: np.ndarray, radar_velocity: np.ndarray,
                        measured_noise_std: float, quantile: float, doppler_sign: int = 1):
    """Return static-return mask and measured residual scale using ego radial velocity.

    Args:
        points: Radar xyz [N,3]; doppler: measured radial speeds [N].
        radar_velocity: Radar-origin translational velocity in radar axes [3].
        measured_noise_std: Positive calibrated noise floor [m/s].
        quantile: Probability for a two-sided normal residual gate.
        doppler_sign: +1 if positive means receding, -1 for the opposite convention.

    Returns:
        Boolean [N], residual std. Stationary prediction is -unit_ray dot velocity.
    """
    points, doppler, radar_velocity = (np.asarray(value, dtype=np.float64)
                                     for value in (points, doppler, radar_velocity))
    if (points.ndim != 2 or points.shape[1] != 3 or doppler.shape != (len(points),)
            or radar_velocity.shape != (3,) or doppler_sign not in (-1, 1)
            or not np.isfinite(measured_noise_std) or measured_noise_std <= 0
            or not 0 < quantile < 1 or not all(np.isfinite(value).all()
                                            for value in (points, doppler, radar_velocity))):
        raise ValueError("invalid radar Doppler inputs or calibrated residual gate")
    ranges = np.linalg.norm(points, axis=1)
    if np.any(ranges <= 0):
        raise ValueError("Doppler projection requires nonzero ranges")
    residual = doppler_sign * doppler + (points / ranges[:, None]) @ radar_velocity
    # Median absolute deviation divided by Phi^-1(.75) estimates normal sigma.
    # Do not subtract the median from the gate: a biased clock/ego model must not be hidden.
    scale = max(measured_noise_std, float(np.median(np.abs(residual - np.median(residual)))
                                         / norm.ppf(0.75))) if len(points) else measured_noise_std
    cutoff = norm.ppf((1 + quantile) / 2) * scale
    return np.abs(residual) <= cutoff, scale


@transform("T35", "Persistent static-return voxel anchors counted by distinct training radar sweeps")
def persistent_anchors(points: np.ndarray, covariances: np.ndarray, origins: np.ndarray,
                       times: np.ndarray, sweep_ids: np.ndarray, voxel_size: np.ndarray,
                       persistence_quantile: float) -> dict:
    """Pool static training returns and retain genuinely repeated voxels without labels.

    Args:
        points, covariances, origins: Map xyz [N,3], uncertainty [N,3,3], sensor origins [N,3].
        times, sweep_ids: Corrected seconds [N], source radar stamp ns [N].
        voxel_size: Positive XYZ cell widths derived/configured before selection.
        persistence_quantile: Configured quantile of distinct-sweep counts in [0,1].

    Returns:
        Arrays of anchor means/covariances/origins, support counts, time spans and
        ragged supporting sweep IDs (source_offsets/source_stamps_ns), plus threshold.
        Storage/work are O(N), O(N log N); all qualifying anchors are retained,
        including those with no nearby baseline Gaussian.
    """
    points, covariances, origins, times, voxel_size = (
        np.asarray(value, dtype=np.float64) for value in (points, covariances, origins, times, voxel_size))
    sweep_ids = np.asarray(sweep_ids)
    count = len(points)
    if (points.shape != (count, 3) or covariances.shape != (count, 3, 3)
            or origins.shape != points.shape or times.shape != (count,) or sweep_ids.shape != (count,)
            or sweep_ids.dtype.kind not in "iu" or voxel_size.shape != (3,)
            or not all(np.isfinite(value).all() for value in (points, covariances, origins, times, voxel_size))
            or np.any(voxel_size <= 0) or not 0 <= persistence_quantile <= 1 or count == 0):
        raise ValueError("nonempty finite anchor observations and valid voxel/persistence settings required")
    if not np.allclose(covariances, covariances.swapaxes(-1, -2)) or np.any(np.linalg.eigvalsh(covariances) <= 0):
        raise ValueError("observation covariance must be symmetric positive definite")
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    _, groups = np.unique(voxel_keys, axis=0, return_inverse=True)
    ordered = np.argsort(groups, kind="stable")
    boundaries = np.flatnonzero(np.diff(groups[ordered])) + 1
    members = np.split(ordered, boundaries)
    supports = np.array([len(np.unique(sweep_ids[rows])) for rows in members])
    # Persistence means repetition: two distinct sweeps is the semantic minimum.
    threshold = max(2, int(np.ceil(np.quantile(supports, persistence_quantile))))
    kept = [rows for rows, support in zip(members, supports) if support >= threshold]
    if not kept:
        raise ValueError("no persistent radar anchors; no fallback targets generated")
    means, uncertainties, source_ids, spans, sensor_origins = [], [], [], [], []
    for rows in kept:
        mean = points[rows].mean(axis=0)
        centered = points[rows] - mean
        # Mixture covariance conservatively retains sensor uncertainty and observed
        # spread; repeated correlated detections do not spuriously shrink uncertainty.
        covariance = covariances[rows].mean(axis=0) + centered.T @ centered / len(rows)
        means.append(mean)
        uncertainties.append(covariance)
        source_ids.append(np.unique(sweep_ids[rows]))
        spans.append([times[rows].min(), times[rows].max()])
        sensor_origins.append(origins[rows[0]])
    offsets = np.cumsum([0] + [len(stamps) for stamps in source_ids])
    return {"means": np.asarray(means), "covariances": np.asarray(uncertainties),
            "sensor_origins": np.asarray(sensor_origins), "time_spans": np.asarray(spans),
            "support_count": np.diff(offsets), "source_offsets": offsets,
            "source_stamps_ns": np.concatenate(source_ids), "persistence_threshold": np.array(threshold)}
