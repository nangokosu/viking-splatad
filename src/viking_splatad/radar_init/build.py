"""Build immutable loss-only radar anchors on a Slurm CPU node, without touching baseline inputs."""

import json
from pathlib import Path
import tempfile

import numpy as np
from scipy.spatial.transform import Rotation

from viking_splatad.dataset.run_split import file_sha256, split_digest
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.extract.pose import Trajectory, interpolate_pose_se3
from viking_splatad.radar_init.anchors import accumulate_radar, persistent_anchors, resolution_covariances, static_doppler_mask
from viking_splatad.radar_init.artifact import (
    ARTIFACT_SCHEMA, CLOCK_SCHEMA, anchor_split, corrected_radar_times,
    load_verified_anchors, object_digest, train_radar_mask, validate_clock_report,
)
from viking_splatad.radar_init.clock import (
    clock_search_domain, doppler_ego_velocity, fit_affine_clock, regularize_velocity_noise, safe_clock_fit_mask,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view


def radar_records(dataset_directory: Path):
    """Return timestamp-ordered (stamp_ns, path) radar sidecars with unique source IDs."""
    records = []
    for path in (Path(dataset_directory) / "sidecar/radar").glob("*.npy"):
        match = FRAME_FILE_PATTERN.match(path.name)
        if match:
            records.append((int(match.group(2)), path))
    records.sort()
    if not records or len({stamp for stamp, _ in records}) != len(records):
        raise ValueError("unique extracted radar source stamps required")
    return records


def source_digest(records) -> str:
    """Hash exact timestamp-to-file-content identities, with no local/account paths."""
    return object_digest({str(stamp): file_sha256(path) for stamp, path in records})


def load_radar_scan(path: Path, fields):
    """Return finite, nonzero radar xyz/Doppler from the manifest's explicit field layout."""
    scan = np.load(path, allow_pickle=False)
    if scan.ndim != 2 or scan.shape[1] != len(fields):
        raise ValueError("radar sidecar does not match its manifest field layout")
    points = np.asarray(scan[:, [fields.index(axis) for axis in ("x", "y", "z")]], dtype=np.float64)
    doppler = np.asarray(scan[:, fields.index("doppler")], dtype=np.float64)
    kept = np.isfinite(points).all(axis=1) & np.isfinite(doppler) & (np.linalg.norm(points, axis=1) > 0)
    return points[kept], doppler[kept]


def radar_motion(view, session_config):
    """Recover base trajectory and radar-origin velocity from recorded LiDAR poses/extrinsics.

    Args:
        view: Baseline dataset view; session_config: June-specific frames and sensor names.

    Returns:
        Base Trajectory, base<-radar matrix, pose times and radar-frame ego velocities.
        Lever-arm rotation is included by differentiating the actual radar origins.
    """
    lidar_name = session_config["lidar"]["name"]
    frames = session_config["frames"]
    extrinsics = view.manifest["extrinsics"]
    base_from_lidar = np.asarray(extrinsics[f"{frames['base']}->{frames['lidar']}"], dtype=np.float64)
    base_from_radar = np.asarray(extrinsics[f"{frames['base']}->{frames['radar']}"], dtype=np.float64)
    map_from_base = view.sensor_poses[lidar_name] @ np.linalg.inv(base_from_lidar)
    times = view.sensor_times[lidar_name]
    if len(times) < 3:
        raise ValueError("radar clock calibration needs a differentiated trajectory of at least three poses")
    trajectory = Trajectory(times, map_from_base[:, :3, 3], Rotation.from_matrix(map_from_base[:, :3, :3]).as_quat())
    map_from_radar = map_from_base @ base_from_radar
    # Three samples are the minimum for the second-order endpoint derivative.
    world_velocity = np.gradient(map_from_radar[:, :3, 3], times, axis=0, edge_order=2)
    radar_velocity = np.einsum("nji,nj->ni", map_from_radar[:, :3, :3], world_velocity)
    return trajectory, base_from_radar, times, radar_velocity


def measure_clock(records, fields, view, lidar_frames, motion, options):
    """Fit clock/sign only with observations safe throughout a bounded candidate search.

    Args:
        records, fields: Radar source identities/layout; view, lidar_frames: exact split.
        motion: radar_motion outputs; options: recorded search/confidence settings.

    Returns:
        Measured affine clock report and Doppler residual noise. The search is
        explicitly local to header clock agreement, not proof of arbitrary clock sync.
        A boundary/uncertain/unidentifiable fit stops artifact construction.
    """
    reference_ns = int(view.manifest["pipeline"]["reference_ns"])
    stamps = np.array([stamp for stamp, _ in records], dtype=np.int64)
    raw_times = (stamps - reference_ns).astype(np.float64) / 1_000_000_000
    centre, offset_limit, drift_limit = clock_search_domain(
        raw_times, lidar_frames, options["clock_fit_retained_fraction"],
    )
    safe = safe_clock_fit_mask(raw_times, lidar_frames, offset_limit, drift_limit, centre)
    fit_records, velocities, radial_noise, velocity_noise = [], [], [], []
    observed_cell_widths = []
    for record, kept in zip(records, safe):
        if not kept:
            continue
        points, doppler = load_radar_scan(record[1], fields)
        # A rank-deficient scan cannot contribute a velocity observation.
        try:
            velocity, sigma_radial, sigma_velocity = doppler_ego_velocity(points, doppler)
        except ValueError:
            continue
        fit_records.append(record)
        velocities.append(velocity)
        radial_noise.append(sigma_radial)
        velocity_noise.append(sigma_velocity)
        gaps = np.diff(np.unique(doppler))
        if len(gaps):
            observed_cell_widths.append(float(np.median(gaps)))
    if len(fit_records) <= 2:
        raise ValueError("insufficient conservatively training-only Doppler observations for clock fitting")
    common_noise, velocity_noise = regularize_velocity_noise(radial_noise, velocity_noise, observed_cell_widths)
    fit_stamps = np.array([stamp for stamp, _ in fit_records], dtype=np.int64)
    fit_times = (fit_stamps - reference_ns).astype(np.float64) / 1_000_000_000
    _, _, reference_times, reference_velocity = motion
    candidates, failures = [], []
    for sign in (1, -1):
        try:
            report = fit_affine_clock(fit_times, sign * np.asarray(velocities), np.asarray(velocity_noise),
                                       reference_times, reference_velocity, offset_limit, drift_limit,
                                       centre, options["confidence_quantile"])
            candidates.append(dict(report, doppler_sign=sign))
        except ValueError as error:
            failures.append(str(error))
    if not candidates:
        raise ValueError("radar timing/sign calibration rejected: " + "; ".join(failures))
    candidates.sort(key=lambda candidate: candidate["cost"])
    if len(candidates) > 1 and np.isclose(candidates[0]["cost"], candidates[1]["cost"]):
        raise ValueError("Doppler sign is ambiguous under the measured motion")
    report = dict(candidates[0], schema=CLOCK_SCHEMA, session=view.manifest["session"]["name"],
                  reference_ns=reference_ns, method="train-safe Doppler/odometry bounded affine fit",
                  fit_stamps_ns=fit_stamps.tolist(), fit_source_sha256=source_digest(fit_records),
                  doppler_noise_std_mps=common_noise,
                  observed_doppler_cell_width_mps=float(np.median(observed_cell_widths)),
                  doppler_noise_model="max_training_median_MAD_and_observed_uniform_cell_std",
                  search_domain_policy="quantile_of_training_time_corridor_margins",
                  clock_fit_retained_fraction=options["clock_fit_retained_fraction"],
                  timing_uncertainty_definition="residual_rms_seconds_is_RMS_posterior_timing_std_from_velocity_fit_covariance")
    return report


def build_loss_only_artifact(dataset_directory: Path, session_config: dict, split_settings: dict,
                             options: dict, destination: Path, calibration=None):
    """Build and contract-check immutable fixed anchors; never edit the baseline manifest.

    Args:
        dataset_directory: Published June inputs; session_config: canonical session YAML.
        split_settings: Exact profile split/window kwargs; options: source-documented radar model.
        destination: New artifact directory, must not exist.
        calibration: Optional independently measured report; None measures a train-safe local fit.

    Returns:
        Artifact summary after checksum/split/source/frame validation and atomic publication.
    """
    dataset_directory, destination = Path(dataset_directory), Path(destination)
    if destination.exists():
        raise FileExistsError("radar artifact already exists; refusing to overwrite")
    view = open_dataset_view(dataset_directory)
    session = view.manifest["session"]["name"]
    if session_config["session"]["name"] != session:
        raise ValueError("session config does not describe this dataset")
    frames, lidar_frames, bounds = anchor_split(view, split_settings)
    motion = radar_motion(view, session_config)
    records = radar_records(dataset_directory)
    fields = view.manifest["pipeline"]["radar"]["fields"]
    reference_ns = int(view.manifest["pipeline"]["reference_ns"])
    if calibration is None:
        calibration = measure_clock(records, fields, view, lidar_frames, motion, options)
    validate_clock_report(calibration, session, reference_ns, lidar_frames, bounds)
    fit_ids = set(calibration["fit_stamps_ns"])
    fit_records = [record for record in records if record[0] in fit_ids]
    if len(fit_records) != len(fit_ids) or source_digest(fit_records) != calibration["fit_source_sha256"]:
        raise ValueError("clock report does not match the exact radar fit source files")
    stamps = np.array([stamp for stamp, _ in records], dtype=np.int64)
    training = train_radar_mask(stamps, calibration, reference_ns, lidar_frames, bounds)
    corrected = corrected_radar_times(stamps, calibration, reference_ns)
    trajectory, extrinsic, reference_times, reference_velocity = motion
    point_groups, covariance_groups, origin_groups, time_groups, stamp_groups = [], [], [], [], []
    accepted_records = []
    # The source paper reports range/angular resolution, not measurement sigma.
    # This explicitly assumes uniform cells; range sigma = full bin width/sqrt(12).
    radial_sigma = options["range_cell_m"] / np.sqrt(12)
    for record, kept, time in zip(records, training, corrected):
        if not kept:
            continue
        points, doppler = load_radar_scan(record[1], fields)
        velocity = np.array([np.interp(time, reference_times, reference_velocity[:, axis])
                             for axis in range(reference_velocity.shape[1])])
        static, _ = static_doppler_mask(points, doppler, velocity, calibration["doppler_noise_std_mps"],
                                         options["confidence_quantile"], calibration["doppler_sign"])
        points = points[static & (np.linalg.norm(points, axis=1) <= options["max_range_m"])]
        if not len(points):
            continue
        raw_seconds = (record[0] - reference_ns) / 1_000_000_000
        map_points, origins, times = accumulate_radar(points, np.full(len(points), raw_seconds), trajectory,
                                                       extrinsic, calibration["offset_seconds"],
                                                       calibration["drift_rate"], calibration["clock_reference_seconds"])
        rotation = (interpolate_pose_se3(trajectory, np.array([time]))[0] @ extrinsic)[:3, :3]
        covariance = resolution_covariances(points, radial_sigma, np.deg2rad(options["azimuth_cell_degrees"]),
                                             np.deg2rad(options["elevation_cell_degrees"]))
        covariance = rotation @ covariance @ rotation.T
        # Include measured clock-fit uncertainty along sensor translation. Angular
        # beam uncertainty remains conservative; the cell model is explicitly recorded.
        world_velocity = rotation @ velocity
        covariance += np.outer(world_velocity, world_velocity) * calibration["residual_rms_seconds"]**2
        point_groups.append(map_points)
        covariance_groups.append(covariance)
        origin_groups.append(origins)
        time_groups.append(times)
        stamp_groups.append(np.full(len(points), record[0], dtype=np.int64))
        accepted_records.append(record)
    if not point_groups:
        raise ValueError("no eligible static training radar returns; no artifact published")
    points, covariance, origins, times, source_stamps = [np.concatenate(groups) for groups in
        (point_groups, covariance_groups, origin_groups, time_groups, stamp_groups)]
    # World-aligned cells use the median full uncertainty width, derived before
    # persistence filtering. No GT, Gaussian support or target-size tuning enters.
    voxel_size = np.median(np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1)), axis=0) * np.sqrt(12)
    arrays = persistent_anchors(points, covariance, origins, times, source_stamps, voxel_size,
                                 options["persistence_quantile"])
    metadata = {"schema": ARTIFACT_SCHEMA, "session": session, "frame": "map",
                "time_base": "session_reference_seconds", "gaussian_initialization": False,
                "dataset_manifest_sha256": file_sha256(dataset_directory / "manifest.json"),
                "dataset_poses_sha256": file_sha256(dataset_directory / "poses.npz"),
                "session_config_sha256": object_digest(session_config), "split_settings": split_settings,
                "split_sha256": split_digest(frames), "clock_calibration": calibration,
                "radar_model": options, "uncertainty_model": "uniform_resolution_cell_plus_clock_translation",
                "voxel_size_m": voxel_size.tolist(), "source_files_sha256": source_digest(accepted_records),
                "training_source_stamps_ns": [stamp for stamp, _ in accepted_records],
                "training_radar_sweeps": len(accepted_records), "static_training_returns": len(points),
                "anchor_count": len(arrays["means"]), "selection": "static_and_persistent_occupancy_not_semantic_trunks"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="radar-anchors-", dir=destination.parent) as temporary:
        output = Path(temporary) / "artifact"
        output.mkdir()
        np.savez_compressed(output / "radar_seeds.npz", **arrays)
        metadata["artifact_sha256"] = file_sha256(output / "radar_seeds.npz")
        (output / "radar_inputs.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
        load_verified_anchors(output, dataset_directory, session, split_settings)
        output.rename(destination)
    return metadata
