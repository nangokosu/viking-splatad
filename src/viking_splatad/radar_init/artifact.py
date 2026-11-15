"""Versioned fixed radar-anchor artifacts and train/window/frame contract gates."""

import hashlib
import json
from pathlib import Path

import numpy as np

from viking_splatad.dataset.run_split import (
    build_run_split_with_cores, file_sha256, nearest_reference_positions, split_digest,
)
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.extract.timebase import NANOSECONDS_PER_SECOND
from viking_splatad.radar_init.anchors import require_rigid_transform
from viking_splatad.radar_init.seeds import NEUTRAL_GREY_RGB, UNPAINTED_CAMERA_STAMP_NS
from viking_splatad.registry import transform
from viking_splatad.splatad.viking_conversions import open_dataset_view

ARTIFACT_SCHEMA = "viking-radar-loss-anchors/1"
# Painted variant: adds per-anchor `rgb` and a `painting` record; enables Gaussian initialisation.
PAINTED_ARTIFACT_SCHEMA = "viking-radar-loss-anchors/2"
# Colour channels per anchor (RGB): a property of the format, not a tunable.
RGB_CHANNEL_COUNT = 3
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
    seconds = (stamps_ns.astype(np.int64) - int(reference_ns)).astype(np.float64) / NANOSECONDS_PER_SECOND
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
            support_count, source_offsets, source_stamps_ns; optional rgb (uint8 [S,3]).

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
    # Optional painted colours: one uint8 RGB triple per anchor.
    if "rgb" in arrays:
        rgb = arrays["rgb"]
        if rgb.dtype != np.uint8 or rgb.shape != (count, RGB_CHANNEL_COUNT):
            raise ValueError("anchor rgb must be uint8 with shape [anchor_count, 3]")
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


def validate_painting_record(painting, anchor_count: int, frames, rgb: np.ndarray) -> None:
    """Require a /2 painting record whose stamps are train frames of its own camera and whose
    unpainted rows are neutral grey.

    Args:
        painting: The metadata `painting` record (or None); must hold `camera` and `camera_stamps_ns`.
        anchor_count: Number of anchors S; the stamp list must have exactly S entries.
        frames: Split frames from `anchor_split`; only train frames of `painting["camera"]` are legal.
        rgb: uint8 [S,3] stored colours, row-aligned with the stamps.

    Raises:
        ValueError: Missing record, wrong length, a stamp that is not a train frame of the recorded
            camera (a held-out, unknown or other-camera stamp would leak imagery or misattribute
            colours), or a stamp -1 row that is not exactly NEUTRAL_GREY_RGB.
    """
    if (not isinstance(painting, dict) or "camera_stamps_ns" not in painting
            or "camera" not in painting):
        raise ValueError("schema /2 radar artifact requires a painting record with camera and camera_stamps_ns")
    stamps = painting["camera_stamps_ns"]
    if not isinstance(stamps, list) or len(stamps) != anchor_count:
        raise ValueError("painting camera_stamps_ns must have one entry per anchor")
    # Legal stamps: the unpainted marker, or the timestamp of a train frame of the recorded camera.
    train_camera_stamps = {frame.timestamp_ns for frame in frames
                           if frame.kind == "camera" and frame.split == "train"
                           and frame.sensor == painting["camera"]}
    illegal = {stamp for stamp in stamps
               if stamp != UNPAINTED_CAMERA_STAMP_NS and stamp not in train_camera_stamps}
    if illegal:
        raise ValueError("painting references camera frames that are not train frames of this split")
    # An unpainted anchor carries no image information: its colour must be exactly neutral grey.
    unpainted_rows = np.asarray(stamps, dtype=np.int64) == UNPAINTED_CAMERA_STAMP_NS
    if np.any(np.asarray(rgb)[unpainted_rows] != NEUTRAL_GREY_RGB):
        raise ValueError("anchors with stamp -1 must hold exactly neutral grey rgb")


def load_verified_anchors(directory: Path, dataset_directory: Path, session: str,
                          expected_settings: dict):
    """Verify complete radar contract against actual data and exact training split.

    Args:
        directory: Immutable artifact directory; dataset_directory: baseline inputs.
        session: Exact session identifier; expected_settings: actual profile split kwargs.

    Returns:
        Validated arrays and provenance. Legacy optional-metadata artifacts are refused.
        Schema /1 (no rgb, gaussian_initialization false) and schema /2 (rgb, flag true,
        painting record naming only train camera frames) are the only accepted combinations.
    """
    directory, dataset_directory = Path(directory), Path(dataset_directory)
    metadata = json.loads((directory / "radar_inputs.json").read_text())
    # Step 1: schema tag must be one of the two known versions; the flag is checked against it below.
    schema = metadata.get("schema")
    painted_schema = schema == PAINTED_ARTIFACT_SCHEMA
    if (schema not in (ARTIFACT_SCHEMA, PAINTED_ARTIFACT_SCHEMA) or metadata.get("session") != session
            or metadata.get("frame") != "map" or metadata.get("time_base") != "session_reference_seconds"
            or metadata.get("gaussian_initialization") is not painted_schema):
        raise ValueError(
            f"radar artifact schema/session/frame/time tags do not match (schema {schema!r}, expected "
            f"{ARTIFACT_SCHEMA!r} with gaussian_initialization false or {PAINTED_ARTIFACT_SCHEMA!r} "
            f"with gaussian_initialization true)")
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
    # Step 2: rgb presence must agree with the schema/flag (/2 requires it, /1 forbids it).
    if ("rgb" in arrays) != painted_schema:
        raise ValueError("radar artifact rgb presence contradicts its schema/gaussian_initialization flag")
    if painted_schema:
        validate_painting_record(metadata.get("painting"), len(arrays["means"]), frames, arrays["rgb"])
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
    encoded = np.asarray(world_to_parser)
    matrix = np.eye(4, dtype=encoded.dtype if encoded.dtype.kind == "f" else np.float64)
    matrix[:3] = encoded
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
        Means, covariances, origins, relative time spans, counts, retained fraction and,
        when the input carries them, the retained rgb rows.
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
    # Painted colours stay row-aligned with the retained anchors; no key when unpainted.
    if "rgb" in arrays:
        result["rgb"] = arrays["rgb"][kept].copy()
    result["time_spans"] = spans[kept] - time_offset
    result["retained_fraction"] = np.array(float(kept.mean()) if len(kept) else 0.0)
    result["dropped_count"] = np.array(int((~kept).sum()))
    return result
