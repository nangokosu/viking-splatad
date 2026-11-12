"""Versioned fixed radar-anchor artifacts and train/window/frame contract gates."""

import hashlib
import json
from pathlib import Path

import numpy as np

from viking_splatad.dataset.run_split import (
    build_run_split_with_cores, file_sha256, nearest_reference_positions, split_digest,
)
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.radar_init.anchors import require_rigid_transform
from viking_splatad.registry import transform
from viking_splatad.splatad.viking_conversions import open_dataset_view

ARTIFACT_SCHEMA = "viking-radar-loss-anchors/1"
CLOCK_SCHEMA = "viking-radar-clock/1"


def object_digest(value: dict) -> str:
    """Return canonical JSON SHA256 for configuration/provenance, rejecting NaN."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def anchor_split(view, settings: dict):
    """Return canonical split frames and reference LiDAR membership for one exact profile.

    Args:
        view: Lightweight dataset view; settings: kwargs accepted by build_run_split_with_cores.

    Returns:
        Split frames, sorted LiDAR frames, and inclusive profile time bounds.
    """
    frames, bounds, _, _ = build_run_split_with_cores(view, **settings)
    lidar_frames = sorted((frame for frame in frames if frame.kind == "lidar"), key=lambda frame: frame.time_s)
    if not lidar_frames or len({frame.sensor for frame in lidar_frames}) != 1:
        raise ValueError("radar anchor assignment requires exactly one reference LiDAR")
    if bounds is None:
        sensor_times = np.concatenate(list(view.sensor_times.values()))
        bounds = (float(sensor_times.min()), float(sensor_times.max()))
    return frames, lidar_frames, bounds


def parser_split_settings(config) -> dict:
    """Return the actual parser's profile settings for strict artifact/split equality."""
    return {"start_fraction": config.dataset_start_fraction, "end_fraction": config.dataset_end_fraction,
            "train_fraction": config.train_split_fraction, "test_block_count": config.test_block_count,
            "test_block_core_fraction": config.test_block_core_fraction,
            "test_split_method": config.test_split_method, "test_render_fraction": config.test_render_fraction}


def corrected_radar_times(stamps_ns: np.ndarray, calibration: dict, reference_ns: int) -> np.ndarray:
    """Apply an explicit measured affine clock model after integer nanosecond subtraction.

    Args:
        stamps_ns: Original integer radar header stamps; calibration: clock report.
        reference_ns: Dataset's pipeline time origin.

    Returns:
        Float64 seconds in the dataset's reference clock, never absolute float32 time.
    """
    stamps_ns = np.asarray(stamps_ns)
    if stamps_ns.dtype.kind not in "iu" or stamps_ns.ndim != 1:
        raise ValueError("original radar stamps must be an integer vector")
    values = [calibration[key] for key in ("offset_seconds", "drift_rate", "clock_reference_seconds")]
    if not np.isfinite(values).all() or calibration["drift_rate"] <= -1:
        raise ValueError("clock model must be finite and monotonic")
    seconds = (stamps_ns.astype(np.int64) - int(reference_ns)).astype(np.float64) / 1_000_000_000
    return seconds + calibration["offset_seconds"] + calibration["drift_rate"] * (
        seconds - calibration["clock_reference_seconds"])


def train_radar_mask(stamps_ns: np.ndarray, calibration: dict, reference_ns: int,
                     lidar_frames, bounds) -> np.ndarray:
    """Assign corrected radar observations to nearest reference instants, earlier on ties.

    Args:
        stamps_ns, calibration, reference_ns: Clock conversion inputs.
        lidar_frames: Sorted windowed reference LiDAR SplitFrame records.
        bounds: Inclusive camera/LiDAR profile window.

    Returns:
        Boolean vector; test/guard and out-of-reference-support observations are false.
    """
    times = corrected_radar_times(stamps_ns, calibration, reference_ns)
    reference_times = np.array([frame.time_s for frame in lidar_frames])
    positions = nearest_reference_positions(reference_times, times)
    train = np.array([frame.split == "train" for frame in lidar_frames])
    # Do not extend the first/last reference instant into unobserved recording tails.
    inside = ((times >= max(bounds[0], reference_times[0]))
              & (times <= min(bounds[1], reference_times[-1])))
    return inside & train[positions]


def validate_clock_report(calibration: dict, session: str, reference_ns: int,
                          lidar_frames, bounds) -> None:
    """Require measured calibration evidence whose fit observations do not leak holdouts.

    Args:
        calibration: Versioned report including source IDs, fit method and residual RMS.
        session, reference_ns, lidar_frames, bounds: Exact target dataset/profile identity.

    Raises:
        ValueError: Absent evidence, wrong clock/session, invalid fit or held-out source IDs.
    """
    if (calibration.get("schema") != CLOCK_SCHEMA or calibration.get("session") != session
            or calibration.get("reference_ns") != reference_ns or not calibration.get("method")):
        raise ValueError("measured radar clock report with matching session/reference is required")
    source_digest = calibration.get("fit_source_sha256", "")
    if len(source_digest) != hashlib.sha256().digest_size * 2 or any(character not in "0123456789abcdef" for character in source_digest):
        raise ValueError("clock fit must pin its observation-source SHA256")
    residual = calibration.get("residual_rms_seconds")
    if not isinstance(residual, (int, float)) or not np.isfinite(residual) or residual < 0:
        raise ValueError("measured nonnegative clock fit residual is required")
    stamps = np.asarray(calibration.get("fit_stamps_ns", []))
    if not len(stamps) or not train_radar_mask(stamps, calibration, reference_ns, lidar_frames, bounds).all():
        raise ValueError("clock calibration fit contains non-training or unsupported observations")


def validate_anchor_arrays(arrays: dict) -> None:
    """Check finite SPD anchors and exact ragged source support/count/time shapes.

    Args:
        arrays: NPZ arrays: means, covariances, sensor_origins, time_spans,
            support_count, source_offsets, source_stamps_ns.

    Raises:
        ValueError: Invalid arrays, including duplicate/miscounted source sweeps.
    """
    count = len(arrays["means"])
    shapes = {"means": (count, 3), "covariances": (count, 3, 3), "sensor_origins": (count, 3),
              "time_spans": (count, 2), "support_count": (count,), "source_offsets": (count + 1,)}
    if count == 0 or any(arrays[key].shape != shape or not np.isfinite(arrays[key]).all()
                         for key, shape in shapes.items()):
        raise ValueError("nonempty finite anchor arrays with matching shapes required")
    covariance = arrays["covariances"]
    if not np.allclose(covariance, covariance.swapaxes(-1, -2)) or np.any(np.linalg.eigvalsh(covariance) <= 0):
        raise ValueError("anchor covariances must be symmetric positive definite")
    offsets, stamps = arrays["source_offsets"], arrays["source_stamps_ns"]
    if (offsets.dtype.kind not in "iu" or stamps.dtype.kind not in "iu" or stamps.ndim != 1
            or arrays["support_count"].dtype.kind not in "iu" or offsets[0] != 0
            or offsets[-1] != len(stamps) or np.any(np.diff(offsets) < 2)
            or not np.array_equal(np.diff(offsets), arrays["support_count"])
            or np.any(arrays["time_spans"][:, 0] > arrays["time_spans"][:, 1])):
        raise ValueError("anchor support must record at least two distinct training sweeps")
    for anchor_index in range(count):
        sources = stamps[offsets[anchor_index]:offsets[anchor_index + 1]]
        if len(np.unique(sources)) != len(sources):
            raise ValueError("duplicate sweep IDs inflate anchor persistence")


def load_verified_anchors(directory: Path, dataset_directory: Path, session: str,
                          expected_settings: dict):
    """Verify complete radar contract against actual data and exact training split.

    Args:
        directory: Immutable artifact directory; dataset_directory: baseline inputs.
        session: Exact session identifier; expected_settings: actual profile split kwargs.

    Returns:
        Validated arrays and provenance. Legacy optional-metadata artifacts are refused.
    """
    directory, dataset_directory = Path(directory), Path(dataset_directory)
    metadata = json.loads((directory / "radar_inputs.json").read_text())
    if (metadata.get("schema") != ARTIFACT_SCHEMA or metadata.get("session") != session
            or metadata.get("frame") != "map" or metadata.get("time_base") != "session_reference_seconds"
            or metadata.get("gaussian_initialization") is not False):
        raise ValueError("loss-only radar artifact schema/session/frame/time tags do not match")
    for filename, key in (("radar_seeds.npz", "artifact_sha256"),):
        if metadata.get(key) != file_sha256(directory / filename):
            raise ValueError("radar artifact checksum mismatch")
    for filename, key in (("manifest.json", "dataset_manifest_sha256"), ("poses.npz", "dataset_poses_sha256")):
        if metadata.get(key) != file_sha256(dataset_directory / filename):
            raise ValueError("radar artifact dataset provenance mismatch")
    view = open_dataset_view(dataset_directory)
    frames, lidar_frames, bounds = anchor_split(view, expected_settings)
    if metadata.get("split_settings") != expected_settings or metadata.get("split_sha256") != split_digest(frames):
        raise ValueError("radar artifact does not match this profile's training/held-out split")
    reference_ns = int(view.manifest["pipeline"]["reference_ns"])
    calibration = metadata["clock_calibration"]
    validate_clock_report(calibration, session, reference_ns, lidar_frames, bounds)
    source_paths = {}
    for path in (dataset_directory / "sidecar/radar").glob("*.npy"):
        match = FRAME_FILE_PATTERN.match(path.name)
        if match:
            stamp = int(match.group(2))
            if stamp in source_paths:
                raise ValueError("duplicate radar source timestamps")
            source_paths[stamp] = path
    training_sources = np.asarray(metadata["training_source_stamps_ns"])
    if (len(np.unique(training_sources)) != len(training_sources)
            or not train_radar_mask(training_sources, calibration, reference_ns, lidar_frames, bounds).all()):
        raise ValueError("pinned radar source observations must be unique training sweeps")
    for source_ids, expected_hash in ((calibration["fit_stamps_ns"], calibration["fit_source_sha256"]),
                                      (metadata["training_source_stamps_ns"], metadata["source_files_sha256"])):
        if any(int(stamp) not in source_paths for stamp in source_ids):
            raise ValueError("radar artifact source observation is missing")
        actual_hash = object_digest({str(stamp): file_sha256(source_paths[int(stamp)]) for stamp in source_ids})
        if actual_hash != expected_hash:
            raise ValueError("radar source file provenance checksum mismatch")
    with np.load(directory / "radar_seeds.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    validate_anchor_arrays(arrays)
    if not set(arrays["source_stamps_ns"].tolist()) <= set(metadata["training_source_stamps_ns"]):
        raise ValueError("anchor support IDs are absent from the pinned source observations")
    if not train_radar_mask(arrays["source_stamps_ns"], calibration, reference_ns, lidar_frames, bounds).all():
        raise ValueError("anchor artifact contains held-out radar observations")
    times = corrected_radar_times(arrays["source_stamps_ns"], calibration, reference_ns)
    for anchor_index in range(len(arrays["means"])):
        selected = times[arrays["source_offsets"][anchor_index]:arrays["source_offsets"][anchor_index + 1]]
        if not np.allclose(arrays["time_spans"][anchor_index], [selected.min(), selected.max()], rtol=0,
                           atol=np.finfo(np.float64).eps * max(1.0, np.abs(selected).max())):
            raise ValueError("anchor time span differs from its recorded source observations")
    return arrays, metadata


@transform("T39", "Map-frame radar anchors/covariances/origins into an unscaled dataparser frame")
def transform_anchor_arrays(arrays: dict, world_to_parser: np.ndarray, scale: float = 1.0) -> dict:
    """Return copied anchors using R*mean+t, R*Sigma*R.T, and R*origin+t.

    Args:
        arrays: Validated artifact arrays; world_to_parser: [3,4] rigid transform.
        scale: Dataparser scale, required exactly one for metre-valued radar targets.

    Returns:
        Transformed copy preserving temporal/source provenance without modifying inputs.
    """
    if scale != 1.0:
        raise ValueError("radar supervision requires unit dataparser scale")
    matrix = np.eye(4)
    matrix[:3] = np.asarray(world_to_parser)
    matrix = require_rigid_transform(matrix)
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    result = {key: value.copy() for key, value in arrays.items()}
    result["means"] = arrays["means"] @ rotation.T + translation
    result["sensor_origins"] = arrays["sensor_origins"] @ rotation.T + translation
    result["covariances"] = rotation @ arrays["covariances"] @ rotation.T
    return result


@transform("T40", "Radar anchor time/scene-box crop with explicit retained fraction and shifted time spans")
def crop_anchor_arrays(arrays: dict, time_bounds, scene_bounds: np.ndarray, time_offset: float):
    """Return model loss targets inside the complete training window and scene box.

    Args:
        arrays: Transformed anchor arrays; time_bounds: inclusive session seconds.
        scene_bounds: Dataparser box [2,3]; time_offset: subtracted session time origin.

    Returns:
        Means, covariances, origins, relative time spans, counts, and retained fraction.
        Ragged source arrays remain in the verified artifact, not model metadata.
    """
    scene_bounds = np.asarray(scene_bounds)
    if (scene_bounds.shape != (2, 3) or not np.isfinite(scene_bounds).all()
            or np.any(scene_bounds[0] > scene_bounds[1]) or not np.isfinite(time_offset)
            or not np.isfinite(time_bounds).all() or time_bounds[0] > time_bounds[1]):
        raise ValueError("valid finite scene/time bounds required")
    spans, means = arrays["time_spans"], arrays["means"]
    kept = ((spans[:, 0] >= time_bounds[0]) & (spans[:, 1] <= time_bounds[1])
            & (means >= scene_bounds[0]).all(axis=1) & (means <= scene_bounds[1]).all(axis=1))
    result = {key: arrays[key][kept].copy() for key in ("means", "covariances", "sensor_origins", "support_count")}
    result["time_spans"] = spans[kept] - time_offset
    result["retained_fraction"] = np.array(float(kept.mean()) if len(kept) else 0.0)
    result["dropped_count"] = np.array(int((~kept).sum()))
    return result
