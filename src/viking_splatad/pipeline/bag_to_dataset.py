"""Config-driven, incremental ROS2 MCAP bag chunks -> intermediate dataset pipeline.

Flow of ``run_pipeline``:
  1. choose chunk files; skip chunks already recorded in ``extract_state.json``;
  2. read ``/icp_odom`` of ALL selected chunks (cached per chunk) into one trajectory;
  3. on the first run only, derive the global parameters (T14-T17, camera model, /tf
     cross-check) and cache them in ``derived.json``; later runs reload them;
  4. stream each new chunk once, writing camera / lidar / radar frames as they are decoded
     (no scan list is ever held in memory);
  5. rewrite ``manifest.json`` / ``poses.npz`` (T26) and the state file after each chunk.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from viking_splatad.dataset.schema import (
    SCHEMA_VERSION,
    CameraIntrinsics,
    ClockInfo,
    IntermediateDataset,
    LidarInfo,
    RollingShutterParams,
    SensorStream,
    SessionInfo,
    WindowInfo,
)
from viking_splatad.dataset.splits import linspace_train_eval_split, split_problems
from viking_splatad.dataset.writer import (
    FRAME_FILE_PATTERN,
    INDEX_DIGITS,
    NO_RETURN_DIRECTORY,
    POSE_KEY_SEPARATOR,
    write_dataset,
    write_frame_file,
)
from viking_splatad.extract.calibration import compare_transforms, load_extrinsics
from viking_splatad.extract.camera import (
    CameraIntrinsics as SourceCameraIntrinsics,
    decode_and_resize_image,
    parse_camera_info,
    rolling_shutter_parameters,
    undistort_image,
)
from viking_splatad.extract.deskew import deskew_points
from viking_splatad.extract.geometry import (
    SWEEP_REFERENCE_FRACTIONS,
    assign_sensor_indices,
    camera_to_world_opencv,
    lidar_to_world_at_reference_time,
)
from viking_splatad.extract.no_return import (
    NO_RETURN_TIME_COLUMN,
    BeamDirectionModel,
    beam_unit_directions,
    derive_beam_direction_model,
    no_return_cells,
    no_return_sweep_rows,
)
from viking_splatad.extract.lidar import (
    assemble_scan_array,
    assign_channels,
    decode_pointcloud2,
    normalise_intensity,
)
from viking_splatad.extract.pose import (
    TfEdge,
    Trajectory,
    check_world_z_up,
    compose_tf_chain,
    interpolate_pose_se3,
    mean_up_vector_from_orientations,
)
from viking_splatad.extract.radar import radar_cloud_to_array
from viking_splatad.extract.timebase import NANOSECONDS_PER_SECOND, ros_stamps_to_increasing_seconds
from viking_splatad.io.bag_metadata import chunk_index, select_bag_chunks
from viking_splatad.io.mcap_reader import make_decoded_reader
from viking_splatad.pipeline.derived import (
    DEFAULT_EGO_SEARCH_FRACTION,
    DERIVED_FILE_NAME,
    DerivedParameters,
    LidarScanSample,
    calibration_tolerance,
    covering_sweep_period,
    derive_lidar_parameters,
    load_derived,
    save_derived,
)

LOGGER = logging.getLogger(__name__)

# File names / directories inside the dataset output directory.
STATE_FILE_NAME = "extract_state.json"
# State key: largest |t| (seconds) over every lidar sweep written so far (T49 input). Kept in the
# state so incremental runs bound the sweeps of earlier runs without re-reading them.
LONGEST_POINT_TIME_STATE_KEY = "longest_point_time_seconds"
# Column of the per-point time t in the written [x, y, z, intensity, t, channel] scan array.
SCAN_TIME_COLUMN = 4
ODOMETRY_CACHE_DIRECTORY = "cache"
MANIFEST_FILE_NAME = "manifest.json"
POSES_FILE_NAME = "poses.npz"
# Entries removed by --force (everything this pipeline owns in the output directory).
OWNED_OUTPUT_ENTRIES = (
    "camera", "lidar", NO_RETURN_DIRECTORY, "sidecar", ODOMETRY_CACHE_DIRECTORY,
    MANIFEST_FILE_NAME, POSES_FILE_NAME, DERIVED_FILE_NAME, STATE_FILE_NAME, "contract_report.md",
)
# Beam-direction fit gate (T51), in azimuth columns: a no-return placeholder off by less than half
# a column still falls in its own raster column. Config key lidar.beam_fit_max_median_residual_columns.
DEFAULT_BEAM_FIT_MAX_MEDIAN_RESIDUAL_COLUMNS = 0.5
# Diagnostic percentile of the fit residual stored in derived.json (conventional robust max).
# Config key lidar.beam_fit_residual_percentile.
DEFAULT_BEAM_FIT_RESIDUAL_PERCENTILE = 99.0
# Lidar model name from the plan (LidarType patch for neurad-studio); config key lidar.type overrides.
DEFAULT_LIDAR_TYPE = "OUSTER_OS0_32"
# Rolling shutter scan direction is not documented for the IDS uEye; top-to-bottom is the
# usual raster order and only matters when readout_seconds > 0 (currently 0 in the config).
DEFAULT_SHUTTER_DIRECTION = "top_to_bottom"
# Extra seconds of bag log time read beyond --max-seconds so that odometry covers the last
# requested sensor stamp: bag log time lags header stamps by 0.09-0.13 s (Phase 1 report)
# and a lidar sweep lasts ~0.1 s; 1 s leaves an order-of-magnitude safety margin.
DEFAULT_POSE_LOOKAHEAD_SECONDS = 1.0
# Largest mean tilt of the robot up-vector from world +z accepted as "z-up map" (T4 check).
# ICP odometry on gently sloped ground tilts a few degrees; >10 degrees would mean the map
# frame is not gravity aligned and SplatAD's z-up assumption would be violated.
DEFAULT_WORLD_UP_TOLERANCE_DEGREES = 10.0
# World up axis stored in the session info (the check above guarantees it).
WORLD_UP_AXIS = (0.0, 0.0, 1.0)


@dataclass
class PipelineResult:
    """Outcome of one ``run_pipeline`` call.

    Attributes:
        output_dir: Dataset directory.
        processed_chunks: Chunk file names read in this run.
        skipped_chunks: Chunk file names skipped as already extracted.
        frames_written: Frames written in this run per stream ("camera", "lidar", "radar").
        frames_total: Frames now in the dataset per stream.
        skipped_counts: Reasons frames were dropped, counted over this run.
        warnings: Human-readable warnings (e.g. extrinsic mismatch).
    """

    output_dir: Path
    processed_chunks: List[str] = field(default_factory=list)
    skipped_chunks: List[str] = field(default_factory=list)
    frames_written: Dict[str, int] = field(default_factory=dict)
    frames_total: Dict[str, int] = field(default_factory=dict)
    skipped_counts: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


@dataclass
class SensorAccumulator:
    """Per-sensor bookkeeping of already-written frames (poses are small; frames are on disk).

    Attributes:
        name: Sensor name.
        kind: "camera" or "lidar".
        times: Frame times in seconds on the session reference clock.
        poses: 4x4 T_world_sensor per frame.
        stamps_ns: Absolute header stamps per frame.
    """

    name: str
    kind: str
    times: List[float] = field(default_factory=list)
    poses: List[np.ndarray] = field(default_factory=list)
    stamps_ns: List[int] = field(default_factory=list)

    @property
    def last_stamp_ns(self) -> int:
        """Latest written stamp, or a value below any real stamp when empty."""
        return self.stamps_ns[-1] if self.stamps_ns else -1

    def add(self, seconds: float, pose: np.ndarray, stamp_ns: int) -> int:
        """Append one frame record.

        Args:
            seconds: Frame time on the reference clock.
            pose: 4x4 T_world_sensor.
            stamp_ns: Absolute header stamp.

        Returns:
            The zero-based frame index just assigned.
        """
        self.times.append(seconds)
        self.poses.append(pose)
        self.stamps_ns.append(stamp_ns)
        return len(self.times) - 1


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------


def stamp_to_total_ns(stamp: Any) -> int:
    """Exact integer nanoseconds of a ROS ``builtin_interfaces/Time``.

    Args:
        stamp: Decoded stamp with ``sec`` and ``nanosec``.

    Returns:
        sec * 1e9 + nanosec as a Python int.
    """
    return int(stamp.sec) * NANOSECONDS_PER_SECOND + int(stamp.nanosec)


def stamp_to_seconds(stamp: Any, reference_ns: int) -> float:
    """Seconds since ``reference_ns`` for one stamp via the T2 conversion.

    Args:
        stamp: Decoded ROS time.
        reference_ns: Session reference clock origin (integer ns).

    Returns:
        float64 seconds (exact integer subtraction happens before the float conversion).
    """
    conversion = ros_stamps_to_increasing_seconds(
        np.array([stamp.sec]), np.array([stamp.nanosec]), reference_ns
    )
    return float(conversion.seconds[0])


def open_chunk_messages(
    chunk_path: Path, topics: Sequence[str], start_ns: Optional[int] = None, end_ns: Optional[int] = None
):
    """Generator over decoded messages of the given topics in log-time order.

    Args:
        chunk_path: MCAP file.
        topics: Topic names to read.
        start_ns, end_ns: Optional bag log-time bounds in nanoseconds.

    Yields:
        (topic, decoded_message) tuples; the file is closed when the generator ends.
    """
    with open(chunk_path, "rb") as handle:
        reader = make_decoded_reader(handle)
        for _, channel, _, decoded in reader.iter_decoded_messages(
            topics=list(topics), start_time=start_ns, end_time=end_ns, log_time_order=True
        ):
            yield channel.topic, decoded


def chunk_log_time_start_ns(chunk_path: Path) -> int:
    """First message log time of a chunk, from the MCAP summary statistics.

    Args:
        chunk_path: MCAP file.

    Returns:
        Log time in nanoseconds.
    """
    with open(chunk_path, "rb") as handle:
        summary = make_decoded_reader(handle).get_summary()
    return int(summary.statistics.message_start_time)


def find_chunk_files(config: Dict[str, Any], explicit_names: Optional[Sequence[str]]) -> List[Path]:
    """Resolve which chunk files to process, sorted by chunk index.

    Priority: explicit CLI names, then ``window.chunk_files``, then (when a window offset is
    set and metadata.yaml exists) T1 chunk selection, else every ``*.mcap`` in the bag dir.

    Args:
        config: Session config.
        explicit_names: File names from the command line, or None.

    Returns:
        Existing chunk paths ordered by chunk index.

    Raises:
        FileNotFoundError: If a named chunk is missing or no chunk is found.
    """
    bag_dir = Path(config["session"]["bag_dir"])
    window = config["window"]
    names: Optional[Sequence[str]] = explicit_names or window.get("chunk_files")
    metadata_path = bag_dir / "metadata.yaml"
    has_offsets = window.get("start_offset_s") is not None or window.get("end_offset_s") is not None
    if not names and has_offsets and metadata_path.is_file():
        names = select_bag_chunks(
            metadata_path.read_text(), window.get("start_offset_s"), window.get("end_offset_s")
        )
    if not names:
        names = [path.name for path in bag_dir.glob("*.mcap")]
    paths = [bag_dir / name for name in names]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Chunk file not found: {path}")
    if not paths:
        raise FileNotFoundError(f"No .mcap chunks in {bag_dir}")
    return sorted(paths, key=lambda path: chunk_index(path.name))


# --------------------------------------------------------------------------------------
# Odometry / trajectory
# --------------------------------------------------------------------------------------


def read_odometry(chunk_path: Path, pose_topic: str, end_ns: Optional[int]) -> Dict[str, np.ndarray]:
    """Read every odometry message (map->base_link) of a chunk.

    Args:
        chunk_path: MCAP file.
        pose_topic: nav_msgs/Odometry topic.
        end_ns: Optional bag log-time upper bound.

    Returns:
        Dict with int64 ``seconds``/``nanoseconds`` stamps, ``positions`` [n,3] and
        ``quaternions_xyzw`` [n,4].
    """
    stamp_seconds, stamp_nanoseconds, positions, quaternions = [], [], [], []
    for _, message in open_chunk_messages(chunk_path, [pose_topic], end_ns=end_ns):
        stamp_seconds.append(message.header.stamp.sec)
        stamp_nanoseconds.append(message.header.stamp.nanosec)
        position, orientation = message.pose.pose.position, message.pose.pose.orientation
        positions.append((position.x, position.y, position.z))
        quaternions.append((orientation.x, orientation.y, orientation.z, orientation.w))
    return {
        "seconds": np.array(stamp_seconds, dtype=np.int64),
        "nanoseconds": np.array(stamp_nanoseconds, dtype=np.int64),
        "positions": np.array(positions, dtype=np.float64).reshape(-1, 3),
        "quaternions_xyzw": np.array(quaternions, dtype=np.float64).reshape(-1, 4),
    }


def load_odometry_cached(
    chunk_path: Path, pose_topic: str, output_dir: Path, end_ns: Optional[int]
) -> Dict[str, np.ndarray]:
    """Odometry of a chunk, from the per-chunk cache when the read is not windowed.

    Windowed reads (``end_ns`` set) are partial and therefore never cached.

    Args:
        chunk_path: MCAP file.
        pose_topic: Odometry topic.
        output_dir: Dataset directory holding the cache.
        end_ns: Optional log-time upper bound (disables caching).

    Returns:
        Same dict as ``read_odometry``.
    """
    cache_path = output_dir / ODOMETRY_CACHE_DIRECTORY / f"odom_{chunk_path.stem}.npz"
    if end_ns is None and cache_path.is_file():
        with np.load(cache_path) as cached:
            return {key: cached[key] for key in cached.files}
    odometry = read_odometry(chunk_path, pose_topic, end_ns)
    if end_ns is None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **odometry)
    return odometry


def build_trajectory(
    odometry_per_chunk: Sequence[Dict[str, np.ndarray]], reference_ns: int
) -> Trajectory:
    """Merge odometry of several chunks into one T2-converted trajectory.

    Args:
        odometry_per_chunk: Outputs of ``read_odometry``.
        reference_ns: Session reference clock origin.

    Returns:
        Trajectory with strictly increasing float64 seconds.
    """
    merged = {
        key: np.concatenate([chunk[key] for chunk in odometry_per_chunk], axis=0)
        for key in odometry_per_chunk[0]
    }
    conversion = ros_stamps_to_increasing_seconds(merged["seconds"], merged["nanoseconds"], reference_ns)
    kept = conversion.kept_indices
    return Trajectory(conversion.seconds, merged["positions"][kept], merged["quaternions_xyzw"][kept])


# --------------------------------------------------------------------------------------
# Camera
# --------------------------------------------------------------------------------------


def process_camera_jpeg(
    jpeg_bytes: bytes, source_intrinsics: SourceCameraIntrinsics, resize_factor: float
) -> Tuple[np.ndarray, SourceCameraIntrinsics]:
    """Decode (T8), then undistort and crop (T7c) one camera JPEG.

    Args:
        jpeg_bytes: Encoded image.
        source_intrinsics: Distorted intrinsics from camera_info (T7a).
        resize_factor: Pre-resize factor from config.

    Returns:
        (BGR uint8 image ready for the writer, undistorted intrinsics with zero distortion).
    """
    rgb_image, scaled_intrinsics = decode_and_resize_image(jpeg_bytes, source_intrinsics, resize_factor)
    undistorted_rgb, undistorted_intrinsics = undistort_image(rgb_image, scaled_intrinsics)
    # The writer uses cv2.imwrite, which expects BGR channel order.
    return cv2.cvtColor(undistorted_rgb, cv2.COLOR_RGB2BGR), undistorted_intrinsics


def intrinsics_to_dict(intrinsics: SourceCameraIntrinsics) -> Dict[str, Any]:
    """Serialisable pinhole parameters (distortion included as a list).

    Args:
        intrinsics: Camera intrinsics.

    Returns:
        Dict with width, height, fx, fy, cx, cy and distortion.
    """
    return {
        "width": int(intrinsics.width), "height": int(intrinsics.height),
        "fx": float(intrinsics.fx), "fy": float(intrinsics.fy),
        "cx": float(intrinsics.cx), "cy": float(intrinsics.cy),
        "distortion": [float(value) for value in intrinsics.distortion],
    }


def derive_camera_model(
    camera_info_message: Any, resize_factor: float
) -> Tuple[SourceCameraIntrinsics, Dict[str, Any]]:
    """Parse camera_info (T7a) and derive the undistorted intrinsics via a black test image.

    The new intrinsics depend only on the calibration and image size, so a black frame of
    the declared size reproduces exactly what every real frame will yield.

    Args:
        camera_info_message: Decoded CameraInfo.
        resize_factor: Pre-resize factor from config.

    Returns:
        (source intrinsics, undistorted-intrinsics dict).
    """
    source_intrinsics = parse_camera_info(camera_info_message)
    black_image = np.zeros((source_intrinsics.height, source_intrinsics.width, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", black_image)
    if not encoded_ok:
        raise RuntimeError("Could not encode the black calibration image")
    _, undistorted_intrinsics = process_camera_jpeg(encoded.tobytes(), source_intrinsics, resize_factor)
    return source_intrinsics, intrinsics_to_dict(undistorted_intrinsics)


# --------------------------------------------------------------------------------------
# Lidar
# --------------------------------------------------------------------------------------


def decode_cloud_fields(message: Any) -> Dict[str, np.ndarray]:
    """T11 decode of a PointCloud2 message.

    Args:
        message: Decoded sensor_msgs/PointCloud2.

    Returns:
        Field name -> array.
    """
    return decode_pointcloud2(
        message.fields, message.point_step, message.row_step, message.width,
        message.height, message.is_bigendian, bytes(message.data),
    )


def lidar_sample_from_fields(
    fields: Dict[str, np.ndarray], column_count: int, stamp_ns: int, lidar_config: Dict[str, Any]
) -> LidarScanSample:
    """Keep the valid (range > 0, finite) points of a decoded sweep with physical units.

    Args:
        fields: Output of ``decode_cloud_fields``.
        column_count: Cloud width (azimuth columns per ring).
        stamp_ns: Header stamp in ns.
        lidar_config: ``lidar`` config section.

    Returns:
        LidarScanSample in the sensor frame.
    """
    range_meters = fields[lidar_config["range_field"]].astype(np.float64) * lidar_config["range_unit_meters"]
    coordinates = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)
    # Ouster marks no-returns with range 0 (deskewed no-returns carry tiny junk coordinates).
    valid_mask = (range_meters > 0) & np.all(np.isfinite(coordinates), axis=1)
    return LidarScanSample(
        points_xyz=coordinates[valid_mask],
        ring=fields[lidar_config["ring_field"]][valid_mask].astype(np.int64),
        intensity=fields[lidar_config["intensity_field"]][valid_mask].astype(np.float64),
        point_time_seconds=(
            fields[lidar_config["point_time_field"]][valid_mask].astype(np.float64)
            * lidar_config["point_time_unit_seconds"]
        ),
        column_count=int(column_count),
        stamp_ns=int(stamp_ns),
    )


def lidar_topic_for_source(config: Dict[str, Any]) -> str:
    """Topic of the configured lidar source.

    Args:
        config: Session config.

    Returns:
        ``topics.lidar_deskewed`` for source "deskewed", ``topics.lidar_raw`` for "raw_own_deskew".

    Raises:
        ValueError: On an unknown source or a reference the deskewed source cannot honour.
    """
    source = config["lidar"]["source"]
    if source == "deskewed":
        # The dataset's deskewed topic is compensated to the sweep start (Phase 1), so only
        # a start reference keeps t consistent with the points.
        if config["lidar"]["reference"] != "start":
            raise ValueError("lidar.source 'deskewed' requires lidar.reference 'start'")
        return config["topics"]["lidar_deskewed"]
    if source == "raw_own_deskew":
        return config["topics"]["lidar_raw"]
    raise ValueError(f"Unknown lidar.source {source!r}")


def sample_lidar_scans(
    chunk_path: Path, config: Dict[str, Any], scan_count: int
) -> List[LidarScanSample]:
    """Read the first ``scan_count`` sweeps of the configured lidar topic.

    Args:
        chunk_path: MCAP file.
        config: Session config.
        scan_count: Number of sweeps (``lidar.model_derivation_scan_count``).

    Returns:
        Samples in stream order.
    """
    samples: List[LidarScanSample] = []
    generator = open_chunk_messages(chunk_path, [lidar_topic_for_source(config)])
    for _, message in generator:
        fields = decode_cloud_fields(message)
        samples.append(
            lidar_sample_from_fields(fields, message.width, stamp_to_total_ns(message.header.stamp), config["lidar"])
        )
        if len(samples) >= scan_count:
            break
    generator.close()
    return samples


def sample_raw_beam_returns(
    chunk_path: Path, config: Dict[str, Any], scan_count: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Returns of the first ``scan_count`` RAW sweeps with their (ring, column) cell.

    The raw topic is used even when lidar.source is "deskewed": only before motion compensation
    does every cell keep a fixed direction in the sensor frame (T51 input).

    Args:
        chunk_path: MCAP file.
        config: Session config.
        scan_count: Number of sweeps (``lidar.model_derivation_scan_count``).

    Returns:
        (points_xyz [n,3] metres, ring [n], column [n], column_count) over all sampled sweeps.

    Raises:
        ValueError: If no sweep is found or the sweeps disagree on their width.
    """
    lidar_config = config["lidar"]
    points_per_sweep, rings_per_sweep, columns_per_sweep, widths = [], [], [], set()
    generator = open_chunk_messages(chunk_path, [config["topics"]["lidar_raw"]])
    for _, message in generator:
        fields = decode_cloud_fields(message)
        coordinates = np.stack([fields["x"], fields["y"], fields["z"]], axis=1).astype(np.float64)
        # Same validity rule as the returns (range > 0 and finite coordinates).
        returned = (fields[lidar_config["range_field"]] > 0) & np.all(np.isfinite(coordinates), axis=1)
        # Row-major organized cloud: cell k is in column k mod width (T53 checks the layout).
        cell_column = np.arange(returned.size) % int(message.width)
        points_per_sweep.append(coordinates[returned])
        rings_per_sweep.append(fields[lidar_config["ring_field"]][returned].astype(np.int64))
        columns_per_sweep.append(cell_column[returned])
        widths.add(int(message.width))
        if len(points_per_sweep) >= scan_count:
            break
    generator.close()
    # The model is indexed by column, so all sweeps must share one width; an empty set means
    # no raw sweep was found.
    if len(widths) != 1:
        raise ValueError(f"raw lidar sweeps in {chunk_path.name}: expected one width, found {sorted(widths)}")
    return (
        np.concatenate(points_per_sweep), np.concatenate(rings_per_sweep),
        np.concatenate(columns_per_sweep), widths.pop(),
    )


def derive_session_beam_model(
    config: Dict[str, Any], chunk_path: Path, elevation_mapping_deg: Dict[str, float], azimuth_resolution_deg: float,
) -> Dict[str, Any]:
    """Fit the beam-direction model (T51) on the first raw sweeps of a chunk.

    Args:
        config: Session config.
        chunk_path: Chunk to sample (the first processed chunk).
        elevation_mapping_deg: Derived elevation mapping; its keys are the rings to fit.
        azimuth_resolution_deg: Derived degrees per column (scales the residual gate).

    Returns:
        ``BeamDirectionModel.to_dict()`` for ``derived.json``.

    Raises:
        ValueError: From the raw sampling (no sweep, mixed widths) or the T51 fit (a ring without
            returns or with too few distinct ranges, median residual above the gate).
    """
    lidar_config = config["lidar"]
    # Returns of the raw cloud, with the (ring, column) cell of each.
    points_xyz, ring, column, column_count = sample_raw_beam_returns(
        chunk_path, config, int(lidar_config["model_derivation_scan_count"])
    )
    max_residual_columns = float(lidar_config.get(
        "beam_fit_max_median_residual_columns", DEFAULT_BEAM_FIT_MAX_MEDIAN_RESIDUAL_COLUMNS))
    # Fit with the gate converted to degrees: residual_deg = residual_columns * degrees per column.
    model = derive_beam_direction_model(
        points_xyz, ring, column, column_count,
        expected_rings=[int(channel) for channel in elevation_mapping_deg],
        max_median_residual_deg=max_residual_columns * float(azimuth_resolution_deg),
        residual_percentile=float(lidar_config.get(
            "beam_fit_residual_percentile", DEFAULT_BEAM_FIT_RESIDUAL_PERCENTILE)),
    )
    return model.to_dict()


# --------------------------------------------------------------------------------------
# /tf cross-check
# --------------------------------------------------------------------------------------


def find_bag_extrinsics(
    chunk_path: Path, tf_topic: str, frame_pairs: Dict[str, Tuple[str, str]]
) -> Dict[str, np.ndarray]:
    """Compose (T4) the first /tf transform found for each wanted parent->child pair.

    Args:
        chunk_path: MCAP file.
        tf_topic: tf2_msgs/TFMessage topic.
        frame_pairs: key ("parent->child") -> (parent, child) frame ids.

    Returns:
        key -> 4x4 T_parent_child for the pairs found (reading stops once all are found).
    """
    found: Dict[str, np.ndarray] = {}
    generator = open_chunk_messages(chunk_path, [tf_topic])
    for _, message in generator:
        for stamped in message.transforms:
            for key, (parent, child) in frame_pairs.items():
                if key in found or stamped.header.frame_id != parent or stamped.child_frame_id != child:
                    continue
                translation, rotation = stamped.transform.translation, stamped.transform.rotation
                found[key] = compose_tf_chain([
                    TfEdge(
                        np.array([translation.x, translation.y, translation.z]),
                        np.array([rotation.x, rotation.y, rotation.z, rotation.w]),
                    )
                ])
        if len(found) == len(frame_pairs):
            break
    generator.close()
    return found


def cross_check_extrinsics(
    calibration: Dict[str, np.ndarray], bag_extrinsics: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, Any]]:
    """Compare calibration-file extrinsics with the bag's /tf (T6) against print-precision tolerances.

    Args:
        calibration: key -> 4x4 from extrinsics.txt (T5).
        bag_extrinsics: key -> 4x4 from /tf (T4).

    Returns:
        key -> {translation_delta_m, rotation_delta_rad, within_tolerance}.
    """
    translation_tolerance, rotation_tolerance = calibration_tolerance()
    report: Dict[str, Dict[str, Any]] = {}
    for key, bag_transform in bag_extrinsics.items():
        translation_delta, rotation_delta = compare_transforms(calibration[key], bag_transform)
        report[key] = {
            "translation_delta_m": translation_delta,
            "rotation_delta_rad": rotation_delta,
            "translation_tolerance_m": translation_tolerance,
            "rotation_tolerance_rad": rotation_tolerance,
            "within_tolerance": bool(
                translation_delta <= translation_tolerance and rotation_delta <= rotation_tolerance
            ),
        }
    return report


# --------------------------------------------------------------------------------------
# Output directory state
# --------------------------------------------------------------------------------------


def load_state(output_dir: Path, session_name: str) -> Dict[str, Any]:
    """Load the extraction state (extracted chunk names, radar frame count).

    Args:
        output_dir: Dataset directory.
        session_name: Session that must own the state.

    Returns:
        State dict (fresh when no file exists).

    Raises:
        ValueError: If the state belongs to another session.
    """
    state_path = output_dir / STATE_FILE_NAME
    if not state_path.is_file():
        return {"session_name": session_name, "extracted_chunks": [], "radar_frame_count": 0}
    state = json.loads(state_path.read_text())
    if state["session_name"] != session_name:
        raise ValueError(f"{state_path} belongs to session {state['session_name']!r}, not {session_name!r}")
    return state


def load_accumulator(output_dir: Path, name: str, kind: str) -> SensorAccumulator:
    """Restore a sensor's written-frame records from poses.npz and the frame file names.

    Frame files beyond the recorded count (left by an interrupted run) are deleted so the
    next frame index is never reused by a stale file. For a lidar this includes its no-return
    sidecar files, which share the frame indices.

    Args:
        output_dir: Dataset directory.
        name: Sensor name.
        kind: "camera" or "lidar".

    Returns:
        Accumulator holding the existing frames (empty for a fresh dataset).
    """
    accumulator = SensorAccumulator(name, kind)
    poses_path = output_dir / POSES_FILE_NAME
    if not poses_path.is_file():
        return accumulator
    with np.load(poses_path) as archive:
        times = archive[f"{name}{POSE_KEY_SEPARATOR}times"]
        poses = archive[f"{name}{POSE_KEY_SEPARATOR}poses_world_sensor"]
    frame_files = []
    for path in (output_dir / kind / name).glob("*"):
        match = FRAME_FILE_PATTERN.match(path.name)
        if match:
            frame_files.append((int(match.group(1)), int(match.group(2)), path))
    frame_files.sort(key=lambda record: record[0])
    for frame_index, _, path in frame_files:
        if frame_index >= len(times):
            path.unlink()
    # A lidar's no-return files share its frame indices, so their orphans go too.
    if kind == "lidar":
        for path in (output_dir / NO_RETURN_DIRECTORY / name).glob("*"):
            match = FRAME_FILE_PATTERN.match(path.name)
            if match and int(match.group(1)) >= len(times):
                path.unlink()
    accumulator.times = [float(value) for value in times]
    accumulator.poses = [pose for pose in poses]
    accumulator.stamps_ns = [record[1] for record in frame_files if record[0] < len(times)]
    return accumulator


def longest_point_time_in_frames(output_dir: Path, name: str, frame_count: int) -> float:
    """Largest |t| over the lidar frame files already on disk (state files from before T49).

    Datasets this old predate the no-return sidecar too, so only the returns files are read.

    Reads every recorded sweep once: O(total points), only when the state lacks the value.

    Args:
        output_dir: Dataset directory.
        name: Lidar sensor name.
        frame_count: Number of recorded frames (files with a higher index are ignored).

    Returns:
        max |t| in seconds, or 0.0 when no frame exists.
    """
    longest = 0.0
    for path in (output_dir / "lidar" / name).glob("*.npy"):
        match = FRAME_FILE_PATTERN.match(path.name)
        # Only recorded frames count; orphans of an interrupted run are deleted by load_accumulator.
        if match and int(match.group(1)) < frame_count:
            scan = np.load(path)
            if scan.shape[0]:
                longest = max(longest, float(np.abs(scan[:, SCAN_TIME_COLUMN]).max()))
    return longest


def remove_orphan_radar_files(output_dir: Path, valid_count: int) -> None:
    """Delete radar sidecar files whose index is >= the recorded count (interrupted run).

    Args:
        output_dir: Dataset directory.
        valid_count: Radar frames recorded in the state file.
    """
    for path in (output_dir / "sidecar" / "radar").glob("*.npy"):
        match = FRAME_FILE_PATTERN.match(path.name)
        if match and int(match.group(1)) >= valid_count:
            path.unlink()


# --------------------------------------------------------------------------------------
# The extractor
# --------------------------------------------------------------------------------------


class DatasetExtractor:
    """Streams chunks into the dataset directory, frame by frame.

    Holds only small per-frame records (time, pose, stamp); image and point data are
    written to disk immediately.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        derived: DerivedParameters,
        trajectory: Trajectory,
        extrinsics: Dict[str, np.ndarray],
        state: Dict[str, Any],
        cutoff_seconds: float,
    ) -> None:
        """Prepare accumulators and fixed per-session values.

        Args:
            config: Session config.
            derived: Cached global parameters.
            trajectory: T_world_base trajectory over all selected chunks.
            extrinsics: Calibration-file transforms (T5).
            state: Extraction state (radar count is read and updated).
            cutoff_seconds: Frames with a later stamp (reference clock) are ignored; inf = none.
        """
        self.config = config
        self.derived = derived
        self.trajectory = trajectory
        self.state = state
        self.cutoff_seconds = cutoff_seconds
        self.output_dir = Path(config["session"]["output_dir"])
        frames = config["frames"]
        self.camera_name = config["camera"]["name"]
        self.lidar_name = config["lidar"]["name"]
        self.T_base_camera = extrinsics[f"{frames['base']}->{frames['camera']}"]
        self.T_base_lidar = extrinsics[f"{frames['base']}->{frames['lidar']}"]
        self.camera_accumulator = load_accumulator(self.output_dir, self.camera_name, "camera")
        self.lidar_accumulator = load_accumulator(self.output_dir, self.lidar_name, "lidar")
        # Longest |t| over all sweeps written so far (T49 input): from the state, or recomputed
        # from the recorded frames for datasets extracted before the state kept it.
        if LONGEST_POINT_TIME_STATE_KEY in state:
            self.longest_point_time_seconds = float(state[LONGEST_POINT_TIME_STATE_KEY])
        else:
            self.longest_point_time_seconds = longest_point_time_in_frames(
                self.output_dir, self.lidar_name, len(self.lidar_accumulator.times)
            )
        remove_orphan_radar_files(self.output_dir, state["radar_frame_count"])
        self.radar_last_stamp_ns = self._last_radar_stamp_ns()
        self.source_intrinsics = SourceCameraIntrinsics(
            fx=derived.camera_source["fx"], fy=derived.camera_source["fy"],
            cx=derived.camera_source["cx"], cy=derived.camera_source["cy"],
            width=derived.camera_source["width"], height=derived.camera_source["height"],
            distortion=np.array(derived.camera_source["distortion"]), model="plumb_bob",
        )
        self.reference_fraction = SWEEP_REFERENCE_FRACTIONS[config["lidar"]["reference"]]
        self.elevation_mapping = {
            int(channel): angle for channel, angle in derived.elevation_mapping_deg.items()
        }
        self.min_lidar_dist = tuple(derived.min_lidar_dist)
        self.beam_model = BeamDirectionModel.from_dict(derived.beam_direction_model)
        self.frames_written = {"camera": 0, "lidar": 0, "radar": 0}
        self.skipped: Dict[str, int] = {}
        self.clipped_values = 0
        self.total_values = 0

    def _last_radar_stamp_ns(self) -> int:
        """Largest stamp among recorded radar sidecar files (-1 if none)."""
        stamps = [
            int(FRAME_FILE_PATTERN.match(path.name).group(2))
            for path in (self.output_dir / "sidecar" / "radar").glob("*.npy")
            if FRAME_FILE_PATTERN.match(path.name)
        ]
        return max(stamps) if stamps else -1

    def _skip(self, reason: str) -> None:
        """Count a dropped frame under ``reason``."""
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def _admit(self, stamp: Any, last_stamp_ns: int) -> Optional[Tuple[float, int]]:
        """Common stamp gate: strictly increasing and within the requested time window.

        Args:
            stamp: Header stamp.
            last_stamp_ns: Latest already-written stamp of the stream.

        Returns:
            (seconds on reference clock, absolute ns) or None when the frame is dropped.
        """
        total_ns = stamp_to_total_ns(stamp)
        # Strict monotonicity also makes re-reading an already-extracted span a no-op.
        if total_ns <= last_stamp_ns:
            self._skip("not_after_previous_stamp")
            return None
        seconds = stamp_to_seconds(stamp, self.derived.reference_ns)
        if seconds > self.cutoff_seconds:
            self._skip("after_max_seconds")
            return None
        return seconds, total_ns

    def handle_camera(self, message: Any) -> None:
        """Process one CompressedImage: pose, undistortion, write.

        Args:
            message: Decoded sensor_msgs/CompressedImage.
        """
        admitted = self._admit(message.header.stamp, self.camera_accumulator.last_stamp_ns)
        if admitted is None:
            return
        seconds, total_ns = admitted
        try:
            T_world_base = interpolate_pose_se3(self.trajectory, np.array([seconds]))[0]
        except ValueError:
            self._skip("camera_outside_trajectory")
            return
        # T9a: camera pose in OpenCV axes (the nerfstudio flip belongs to the dataparser).
        _, T_world_camera = camera_to_world_opencv(T_world_base, self.T_base_camera)
        image, undistorted = process_camera_jpeg(
            bytes(message.data), self.source_intrinsics, self.derived.resize_factor
        )
        # Every frame must reproduce the cached intrinsics, otherwise frames would be inconsistent.
        cached = self.derived.camera_undistorted
        if not np.allclose(
            [undistorted.fx, undistorted.fy, undistorted.cx, undistorted.cy, undistorted.width, undistorted.height],
            [cached["fx"], cached["fy"], cached["cx"], cached["cy"], cached["width"], cached["height"]],
        ):
            raise RuntimeError("Undistorted intrinsics differ from the cached derived.json values")
        frame_index = self.camera_accumulator.add(seconds, T_world_camera, total_ns)
        write_frame_file(self.output_dir, "camera", self.camera_name, frame_index, total_ns, image)
        self.frames_written["camera"] += 1

    def handle_lidar(self, message: Any) -> None:
        """Process one lidar sweep: mask, intensity, deskew (if configured), pose, write.

        Also writes the sweep's no-return file (Phase B, T53/T52/T54) with the same index and stamp.

        Args:
            message: Decoded sensor_msgs/PointCloud2 of the configured source topic.

        Raises:
            ValueError: If the cloud is not organized (T53), so no-return cells cannot be located.
        """
        admitted = self._admit(message.header.stamp, self.lidar_accumulator.last_stamp_ns)
        if admitted is None:
            return
        seconds, total_ns = admitted
        lidar_config = self.config["lidar"]
        sweep_period = self.derived.sweep_period_seconds
        fields = decode_cloud_fields(message)
        sample = lidar_sample_from_fields(fields, message.width, total_ns, lidar_config)
        if sample.points_xyz.shape[0] == 0:
            self._skip("lidar_no_valid_points")
            return
        try:
            # T18: pose of the lidar at the sweep reference time (stamp = sweep start).
            T_world_lidar, reference_time = lidar_to_world_at_reference_time(
                self.trajectory, self.T_base_lidar, seconds, sweep_period, lidar_config["reference"]
            )
            def pose_at(query_times: np.ndarray) -> np.ndarray:
                """T_world_lidar at the given times (batched)."""
                return interpolate_pose_se3(self.trajectory, query_times) @ self.T_base_lidar

            if lidar_config["source"] == "raw_own_deskew":
                # T13: move every point to the reference pose using the interpolated trajectory.
                points_xyz, relative_time = deskew_points(
                    sample.points_xyz, seconds + sample.point_time_seconds, pose_at, reference_time
                )
            else:
                # Points are already at the sweep start; t is the offset from the reference
                # (= point time minus fraction * period; fraction is 0 for 'start').
                points_xyz = sample.points_xyz
                relative_time = sample.point_time_seconds - self.reference_fraction * sweep_period
        except ValueError:
            self._skip("lidar_outside_trajectory")
            return
        # T14 with the cached scale (dtype_max overrides the percentile): clip(value/scale, 0, 1).
        intensity_result = normalise_intensity([sample.intensity], dtype_max=self.derived.intensity_scale)
        self.clipped_values += intensity_result.clipped_count
        self.total_values += intensity_result.total_count
        # T19: ring ids are the channel keys of the cached elevation mapping.
        channel = assign_channels(self.elevation_mapping, ring=sample.ring)
        # T15 ego-box filter + column assembly [x, y, z, intensity, t, channel].
        scan_array = assemble_scan_array(
            points_xyz, intensity_result.normalised_scans[0], relative_time, channel, self.min_lidar_dist
        )
        if scan_array.shape[0] == 0:
            self._skip("lidar_empty_after_filter")
            return
        # No-returns (T53 -> T52 -> T54): cell, far-field raw direction, rotation to the reference
        # pose with our trajectory (also for the "deskewed" source, whose no-return cells carry no
        # usable coordinates). The column times can run past the last odometry stamp at the end of
        # the session; there the end pose is held. Directions use only the rotation, so the error
        # is at most the turn during one sweep. T53 raises ValueError on an unorganized cloud.
        no_return_ring, no_return_column, no_return_time = no_return_cells(
            fields, message.width, message.height, lidar_config)

        def pose_at_held(query_times: np.ndarray) -> np.ndarray:
            """T_world_lidar at the given times, holding the end poses outside the trajectory."""
            return interpolate_pose_se3(self.trajectory, query_times, allow_extrapolation=True) @ self.T_base_lidar

        no_return_rows = no_return_sweep_rows(
            beam_unit_directions(self.beam_model, no_return_ring, no_return_column),
            seconds + no_return_time, pose_at_held, reference_time,
            assign_channels(self.elevation_mapping, ring=no_return_ring), no_return_column,
        )
        frame_index = self.lidar_accumulator.add(seconds, T_world_lidar, total_ns)
        write_frame_file(self.output_dir, "lidar", self.lidar_name, frame_index, total_ns, scan_array)
        write_frame_file(self.output_dir, NO_RETURN_DIRECTORY, self.lidar_name, frame_index, total_ns, no_return_rows)
        # Track the longest |t| actually written; the manifest sweep period must cover it (T49).
        # No-returns count too (they are SplatAD rows with add_missing_points): the latest column
        # of a sweep can be all no-returns, or its returns removed by the ego filter. scan_array is
        # non-empty here, so the max is defined even when the sweep has no no-returns.
        written_times = [scan_array[:, SCAN_TIME_COLUMN], no_return_rows[:, NO_RETURN_TIME_COLUMN]]
        self.longest_point_time_seconds = max(
            self.longest_point_time_seconds, float(np.abs(np.concatenate(written_times)).max())
        )
        self.frames_written["lidar"] += 1

    def handle_radar(self, message: Any) -> None:
        """Write one radar scan to the sidecar (never part of the SplatAD input).

        Args:
            message: Decoded sensor_msgs/PointCloud2 from the radar topic.
        """
        admitted = self._admit(message.header.stamp, self.radar_last_stamp_ns)
        if admitted is None:
            return
        _, total_ns = admitted
        radar_array = radar_cloud_to_array(decode_cloud_fields(message), self.config["radar"]["fields"])
        frame_index = self.state["radar_frame_count"]
        radar_directory = self.output_dir / "sidecar" / "radar"
        radar_directory.mkdir(parents=True, exist_ok=True)
        np.save(radar_directory / f"{frame_index:0{INDEX_DIGITS}d}_{total_ns}.npy", radar_array)
        self.state["radar_frame_count"] = frame_index + 1
        self.radar_last_stamp_ns = total_ns
        self.frames_written["radar"] += 1

    def earliest_resume_log_time_ns(self) -> Optional[int]:
        """Log time from which an already-read chunk must be re-read to recover its tail.

        Frames at the end of a chunk can lack odometry until the next chunk arrives. They
        were dropped earlier, so the previous chunk is re-read from the oldest last-written
        stamp; the strictly-increasing stamp gate makes everything already written a no-op.
        Bag log time is never earlier than the header stamp, so this start loses nothing.

        Returns:
            Log-time start in ns, or None if nothing was written yet (re-read all).
        """
        last_stamps = [
            stamp for stamp in (
                self.camera_accumulator.last_stamp_ns,
                self.lidar_accumulator.last_stamp_ns,
                self.radar_last_stamp_ns,
            ) if stamp >= 0
        ]
        return min(last_stamps) if last_stamps else None

    def process_chunk(
        self, chunk_path: Path, end_ns: Optional[int], start_ns: Optional[int] = None
    ) -> None:
        """Stream one chunk, dispatching camera / lidar / radar messages.

        Args:
            chunk_path: MCAP file.
            end_ns: Optional bag log-time upper bound (for --max-seconds).
            start_ns: Optional bag log-time lower bound (tail re-read of a previous chunk).
        """
        topics = self.config["topics"]
        lidar_topic = lidar_topic_for_source(self.config)
        handlers = {
            topics["camera_image"]: self.handle_camera,
            lidar_topic: self.handle_lidar,
            topics["radar"]: self.handle_radar,
        }
        for topic, message in open_chunk_messages(chunk_path, list(handlers), start_ns=start_ns, end_ns=end_ns):
            handlers[topic](message)


# --------------------------------------------------------------------------------------
# Manifest / metadata
# --------------------------------------------------------------------------------------


def write_metadata(
    extractor: DatasetExtractor,
    extrinsics: Dict[str, np.ndarray],
    extracted_chunks: List[str],
    warnings_list: List[str],
) -> None:
    """Write manifest.json and poses.npz (T26, no frames) plus pipeline extras and the state file.

    Args:
        extractor: Extractor holding accumulators, config and derived parameters.
        extrinsics: Calibration transforms (camera, lidar, radar entries are stored).
        extracted_chunks: Fully extracted chunk names (written to the state file).
        warnings_list: Warnings to record in the manifest.
    """
    config, derived = extractor.config, extractor.derived
    output_dir = extractor.output_dir
    frames = config["frames"]
    camera_cfg = config["camera"]

    def build_stream(accumulator: SensorAccumulator) -> SensorStream:
        """SensorStream (without frame payloads) from an accumulator."""
        count = len(accumulator.times)
        return SensorStream(
            sensor_name=accumulator.name, kind=accumulator.kind,
            times=np.array(accumulator.times, dtype=np.float64),
            poses_world_sensor=np.array(accumulator.poses, dtype=np.float64).reshape(count, 4, 4),
            frame_indices=np.arange(count, dtype=np.int64),
            frame_timestamps_ns=np.array(accumulator.stamps_ns, dtype=np.int64),
            frames=[],
        )

    streams = {
        extractor.camera_name: build_stream(extractor.camera_accumulator),
        extractor.lidar_name: build_stream(extractor.lidar_accumulator),
    }
    all_times = np.concatenate([stream.times for stream in streams.values()])
    camera_dict = derived.camera_undistorted
    rolling_shutter_time, time_to_center_pixel = rolling_shutter_parameters(
        camera_cfg["timestamp_convention"], camera_cfg["readout_seconds"]
    )
    dataset = IntermediateDataset(
        schema_version=SCHEMA_VERSION,
        cameras={extractor.camera_name: CameraIntrinsics(
            width=int(camera_dict["width"]), height=int(camera_dict["height"]),
            fx=camera_dict["fx"], fy=camera_dict["fy"], cx=camera_dict["cx"], cy=camera_dict["cy"],
            distortion=None,  # images are undistorted offline
        )},
        extrinsics={
            f"{frames['base']}->{frames[role]}": extrinsics[f"{frames['base']}->{frames[role]}"]
            for role in ("camera", "lidar", "radar")
        },
        clock=ClockInfo(
            reference_sensor=extractor.lidar_name,
            sensor_offsets_seconds={extractor.camera_name: 0.0, extractor.lidar_name: 0.0,
                                    config["radar"]["name"]: 0.0},
            radar_drift_offset=0.0, radar_drift_rate=0.0,
        ),
        rolling_shutter={extractor.camera_name: RollingShutterParams(
            readout_seconds=rolling_shutter_time, direction=DEFAULT_SHUTTER_DIRECTION)},
        lidar=LidarInfo(
            lidar_type=config["lidar"].get("type", DEFAULT_LIDAR_TYPE),
            # LidarInfo documents radians; T16 yields degrees.
            elevation_mapping={int(channel): float(np.radians(angle))
                               for channel, angle in derived.elevation_mapping_deg.items()},
            azimuth_resolution=float(np.radians(derived.azimuth_resolution_deg)),
            azimuth_tile_width=derived.azimuth_tile_width,
            min_lidar_dist=list(derived.min_lidar_dist),
            intensity_scale=derived.intensity_scale,
            # T49: the derived (first-chunk sample) period, raised to the longest |t| written.
            sweep_period=covering_sweep_period(derived.sweep_period_seconds, extractor.longest_point_time_seconds),
        ),
        window=WindowInfo(
            start_time=float(all_times.min()) if all_times.size else 0.0,
            end_time=float(all_times.max()) if all_times.size else 0.0,
        ),
        session=SessionInfo(
            name=config["session"]["name"], world_up=np.array(WORLD_UP_AXIS),
            train_fraction=config["split"]["train_fraction"],
        ),
        streams=streams,
        radar_sidecar={},
    )
    write_dataset(output_dir, dataset)
    # T20 sensor indices and T21 per-sensor splits, recorded for the dataparser.
    name_to_index, index_to_name = assign_sensor_indices([extractor.camera_name], [extractor.lidar_name])
    splits, split_issues = {}, []
    for name, stream in streams.items():
        train_indices, eval_indices = linspace_train_eval_split(
            len(stream.times), config["split"]["train_fraction"]
        )
        splits[name] = {"train": train_indices.tolist(), "eval": eval_indices.tolist()}
        split_issues += [f"{name}: {issue}" for issue in
                         split_problems(train_indices, eval_indices, len(stream.times),
                                        config["split"]["train_fraction"])]
    manifest_path = output_dir / MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["pipeline"] = {
        "reference_ns": derived.reference_ns,
        "extracted_chunks": extracted_chunks,
        "sensor_name_to_index": name_to_index,
        "sensor_idx_to_name": index_to_name,
        "splits": splits,
        "split_problems": split_issues,
        "rolling_shutter_time": rolling_shutter_time,
        "time_to_center_pixel": time_to_center_pixel,
        "timestamp_convention": camera_cfg["timestamp_convention"],
        "lidar_source": config["lidar"]["source"],
        "lidar_reference": config["lidar"]["reference"],
        "intensity_clipped_fraction": (
            extractor.clipped_values / extractor.total_values if extractor.total_values else 0.0
        ),
        "extrinsic_check": derived.extrinsic_check,
        "radar": {"name": config["radar"]["name"], "fields": list(config["radar"]["fields"]),
                  "extrinsic_key": f"{frames['base']}->{frames['radar']}",
                  "frame_count": extractor.state["radar_frame_count"]},
        "warnings": warnings_list,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    extractor.state["extracted_chunks"] = extracted_chunks
    extractor.state[LONGEST_POINT_TIME_STATE_KEY] = extractor.longest_point_time_seconds
    (output_dir / STATE_FILE_NAME).write_text(json.dumps(extractor.state, indent=2))


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------


def derive_session_parameters(
    config: Dict[str, Any], first_chunk: Path, reference_ns: int, calibration: Dict[str, np.ndarray],
    output_dir: Path,
) -> Tuple[DerivedParameters, List[str]]:
    """Derive all global parameters from the first processed chunk and cache them.

    Args:
        config: Session config.
        first_chunk: First chunk being processed.
        reference_ns: Session reference clock origin.
        calibration: Calibration-file transforms.
        output_dir: Dataset directory (``derived.json`` is written here).

    Returns:
        (derived parameters, warnings).
    """
    warnings_list: List[str] = []
    lidar_config, topics, frames = config["lidar"], config["topics"], config["frames"]
    # Camera model from the first camera_info message.
    camera_info = next(m for _, m in open_chunk_messages(first_chunk, [topics["camera_info"]]))
    resize_factor = float(config["camera"]["resize_factor"])
    source_intrinsics, undistorted = derive_camera_model(camera_info, resize_factor)
    # Lidar model from the first N sweeps (T14-T17).
    samples = sample_lidar_scans(first_chunk, config, int(lidar_config["model_derivation_scan_count"]))
    lidar_derived = derive_lidar_parameters(
        samples, lidar_config["intensity_percentile"], lidar_config["ego_margin_m"],
        lidar_config.get("ego_search_fraction", DEFAULT_EGO_SEARCH_FRACTION),
    )
    if lidar_derived["min_lidar_dist_fallback"]:
        warnings_list.append("min_lidar_dist: no radial gap found, ego filtering disabled (0,0,0)")
    # Beam directions of every cell, for the no-return sidecar (T51).
    lidar_derived["beam_direction_model"] = derive_session_beam_model(
        config, first_chunk, lidar_derived["elevation_mapping_deg"], lidar_derived["azimuth_resolution_deg"]
    )
    # /tf cross-check of the three sensor extrinsics (T4 + T6).
    pairs = {
        f"{frames['base']}->{frames[role]}": (frames["base"], frames[role])
        for role in ("camera", "lidar", "radar")
    }
    bag_extrinsics = find_bag_extrinsics(first_chunk, topics["tf"], pairs)
    extrinsic_check = cross_check_extrinsics(calibration, bag_extrinsics)
    for key, entry in extrinsic_check.items():
        if not entry["within_tolerance"]:
            warnings_list.append(
                f"extrinsic {key}: file vs /tf differ by {entry['translation_delta_m']:.4g} m, "
                f"{entry['rotation_delta_rad']:.4g} rad (tolerance {entry['translation_tolerance_m']:.4g} m, "
                f"{entry['rotation_tolerance_rad']:.4g} rad)"
            )
    for key in pairs:
        if key not in bag_extrinsics:
            warnings_list.append(f"extrinsic {key}: not found in {topics['tf']}")
    derived = DerivedParameters(
        session_name=config["session"]["name"], reference_ns=reference_ns,
        source_chunk=first_chunk.name, scan_count=len(samples),
        camera_source=intrinsics_to_dict(source_intrinsics), camera_undistorted=undistorted,
        resize_factor=resize_factor,
        bag_extrinsics={key: value.tolist() for key, value in bag_extrinsics.items()},
        extrinsic_check=extrinsic_check, **lidar_derived,
    )
    save_derived(output_dir / DERIVED_FILE_NAME, derived)
    return derived, warnings_list


def run_pipeline(
    config: Dict[str, Any],
    chunk_names: Optional[Sequence[str]] = None,
    max_seconds: Optional[float] = None,
    force: bool = False,
) -> PipelineResult:
    """Extract new bag chunks into the session's intermediate dataset (idempotent).

    Args:
        config: Session config from ``load_session_config``.
        chunk_names: Optional chunk file names overriding the config selection.
        max_seconds: Only extract frames stamped within this many seconds of the session
            start (debug/test runs). Chunks read this way are not marked extracted.
        force: Delete previous outputs of this pipeline and extract everything again.

    Returns:
        PipelineResult.

    Raises:
        ValueError: If a new chunk precedes already extracted chunks (needs ``force``), or
            the world frame is not z-up.
    """
    session_name = config["session"]["name"]
    output_dir = Path(config["session"]["output_dir"])
    pipeline_config = config.get("pipeline", {})
    lookahead_seconds = pipeline_config.get("pose_lookahead_seconds", DEFAULT_POSE_LOOKAHEAD_SECONDS)
    tilt_tolerance = np.radians(pipeline_config.get("world_up_tolerance_degrees", DEFAULT_WORLD_UP_TOLERANCE_DEGREES))
    result = PipelineResult(output_dir=output_dir)
    # Step 1: force clears this pipeline's outputs (and only those).
    if force:
        for entry in OWNED_OUTPUT_ENTRIES:
            target = output_dir / entry
            if target.is_dir():
                shutil.rmtree(target)
            elif target.exists():
                target.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    # Step 2: pick chunks; split into already extracted and new.
    selected_chunks = find_chunk_files(config, chunk_names)
    state = load_state(output_dir, session_name)
    already = set(state["extracted_chunks"])
    new_chunks = [path for path in selected_chunks if path.name not in already]
    result.skipped_chunks = [path.name for path in selected_chunks if path.name in already]
    if not new_chunks:
        LOGGER.info("All selected chunks already extracted; nothing to do")
        return result
    if already:
        newest_done = max(chunk_index(name) for name in already)
        if chunk_index(new_chunks[0].name) < newest_done:
            raise ValueError(
                f"Chunk {new_chunks[0].name} precedes already extracted chunk index {newest_done}; "
                "frames must be appended in time order (use --force to rebuild)"
            )
    # Step 3: read-window end (bag log time) when --max-seconds is given.
    derived = load_derived(output_dir / DERIVED_FILE_NAME, session_name)
    window_origin_ns = derived.reference_ns if derived else chunk_log_time_start_ns(selected_chunks[0])
    end_ns = (
        None if max_seconds is None
        else int(window_origin_ns + (max_seconds + lookahead_seconds) * NANOSECONDS_PER_SECOND)
    )
    # Step 4: odometry of ALL selected chunks (extracted ones come from the cache).
    pose_topic = config["topics"]["pose"]
    odometry_per_chunk = []
    for chunk_path in selected_chunks:
        if end_ns is not None and chunk_log_time_start_ns(chunk_path) > end_ns:
            continue
        odometry_per_chunk.append(load_odometry_cached(chunk_path, pose_topic, output_dir, end_ns))
    odometry_per_chunk = [chunk for chunk in odometry_per_chunk if chunk["seconds"].size]
    # Reference clock: first odometry stamp of the session, fixed once in derived.json.
    reference_ns = derived.reference_ns if derived else int(min(
        (chunk["seconds"] * NANOSECONDS_PER_SECOND + chunk["nanoseconds"]).min()
        for chunk in odometry_per_chunk
    ))
    trajectory = build_trajectory(odometry_per_chunk, reference_ns)
    # z-up check (T4): mean body up-vector vs world +z.
    check_world_z_up(mean_up_vector_from_orientations(trajectory.quaternions_xyzw), tilt_tolerance)
    # Step 5: calibration and one-time derivation.
    calibration = load_extrinsics(config["session"]["calibration_dir"], config["session"]["extrinsics_variant"])
    result.warnings = []
    if derived is None:
        derived, result.warnings = derive_session_parameters(
            config, new_chunks[0], reference_ns, calibration, output_dir
        )
    elif derived.beam_direction_model is None:
        # Cache from before Phase B: fit the beam model once and keep it. Frames extracted
        # earlier have no no-return files; the contract reports the gap (re-extract with --force).
        derived.beam_direction_model = derive_session_beam_model(
            config, new_chunks[0], derived.elevation_mapping_deg, derived.azimuth_resolution_deg
        )
        save_derived(output_dir / DERIVED_FILE_NAME, derived)
    # Step 6: stream each new chunk.
    # max_seconds is measured on the session reference clock (first odometry stamp = 0 s).
    cutoff_seconds = float("inf") if max_seconds is None else float(max_seconds)
    extractor = DatasetExtractor(config, derived, trajectory, calibration, state, cutoff_seconds)
    extracted_chunks = list(state["extracted_chunks"])
    for chunk_path in new_chunks:
        if end_ns is not None and chunk_log_time_start_ns(chunk_path) > end_ns:
            continue
        LOGGER.info("Extracting %s", chunk_path.name)
        # Recover the tail of the directly preceding chunk, whose last frames had no odometry
        # before this chunk's poses became available.
        preceding_chunks = [
            path for path in selected_chunks
            if chunk_index(path.name) < chunk_index(chunk_path.name)
        ]
        if preceding_chunks:
            extractor.process_chunk(preceding_chunks[-1], end_ns, extractor.earliest_resume_log_time_ns())
        extractor.process_chunk(chunk_path, end_ns)
        result.processed_chunks.append(chunk_path.name)
        # Partial (windowed) reads never mark a chunk as complete.
        if max_seconds is None:
            extracted_chunks.append(chunk_path.name)
        write_metadata(extractor, calibration, extracted_chunks, result.warnings)
    result.frames_written = dict(extractor.frames_written)
    result.frames_total = {
        "camera": len(extractor.camera_accumulator.times),
        "lidar": len(extractor.lidar_accumulator.times),
        "radar": state["radar_frame_count"],
    }
    result.skipped_counts = dict(extractor.skipped)
    return result
