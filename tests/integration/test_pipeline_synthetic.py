"""End-to-end test of the bag -> dataset pipeline on a tiny synthetic ROS2 MCAP bag.

Scene: the robot drives along +x at a constant speed with identity orientation; a static
set of world points is seen by a 32-ring x 64-column lidar; one camera sees a synthetic
image through radial distortion; radar and /tf are present. Every expected number below is
computed analytically from these definitions.
"""

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from mcap_ros2.writer import Writer

from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.dataset.writer import read_dataset
from viking_splatad.io.mcap_reader import allow_legacy_field_names
from viking_splatad.pipeline.bag_to_dataset import run_pipeline
from viking_splatad.pipeline.derived import DERIVED_FILE_NAME

NANOSECONDS_PER_SECOND = 1_000_000_000  # SI definition
FIXTURE_EXTRINSICS = Path(__file__).parent.parent / "fixtures" / "extrinsics_real.txt"

# ---- synthetic scene definition (all test-local; justified by being a toy world) ----
EPOCH_BASE_SECONDS = 1_714_571_311          # realistic epoch stamp: exercises int64 T2 precision
ROBOT_SPEED = 1.0                           # m/s along +x
CHUNK_DURATION = 3.0                        # s of sensor data per chunk file
ODOMETRY_PERIOD = 0.1                       # s
CAMERA_PERIOD, CAMERA_OFFSET = 0.1, 0.05    # s
LIDAR_PERIOD, LIDAR_OFFSET = 0.1, 0.02      # s
RADAR_PERIOD = 0.05                         # s
LOG_LAG = 0.1                               # s, bag log time lags the header stamp as in the real bag
RING_COUNT, COLUMN_COUNT = 32, 64           # 32 = multiple of 8, 64 = multiple of 32 (tile rule)
IMAGE_WIDTH, IMAGE_HEIGHT = 160, 120
CAMERA_FOCAL, CAMERA_K1 = 150.0, -0.05
RADAR_POINT_COUNT = 5
LAYOUT_TOLERANCE = 1e-9                     # s, float64 time comparison
# Extrinsics of the fixture file (base_link frame): camera, lidar translations.
CAMERA_TRANSLATION = np.array([0.18, -0.01, 0.14])
LIDAR_TRANSLATION = np.array([0.02, 0.0, 0.225])

HEADER_MSG = "builtin_interfaces/Time stamp\nstring frame_id\n"
TIME_MSG = "int32 sec\nuint32 nanosec\n"
SEPARATOR = "=" * 80 + "\n"
SCHEMAS = {
    "sensor_msgs/msg/CompressedImage": (
        "std_msgs/Header header\nstring format\nuint8[] data\n"
        f"{SEPARATOR}MSG: std_msgs/Header\n{HEADER_MSG}{SEPARATOR}MSG: builtin_interfaces/Time\n{TIME_MSG}"
    ),
    # Legacy uppercase D/K/R/P exactly like the real bag.
    "sensor_msgs/msg/CameraInfo": (
        "std_msgs/Header header\nuint32 height\nuint32 width\nstring distortion_model\n"
        "float64[] D\nfloat64[9] K\nfloat64[9] R\nfloat64[12] P\n"
        f"{SEPARATOR}MSG: std_msgs/Header\n{HEADER_MSG}{SEPARATOR}MSG: builtin_interfaces/Time\n{TIME_MSG}"
    ),
    "sensor_msgs/msg/PointCloud2": (
        "std_msgs/Header header\nuint32 height\nuint32 width\nsensor_msgs/PointField[] fields\n"
        "bool is_bigendian\nuint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense\n"
        f"{SEPARATOR}MSG: std_msgs/Header\n{HEADER_MSG}{SEPARATOR}MSG: builtin_interfaces/Time\n{TIME_MSG}"
        f"{SEPARATOR}MSG: sensor_msgs/PointField\nstring name\nuint32 offset\nuint8 datatype\nuint32 count\n"
    ),
    "nav_msgs/msg/Odometry": (
        "std_msgs/Header header\nstring child_frame_id\ngeometry_msgs/PoseWithCovariance pose\n"
        f"{SEPARATOR}MSG: std_msgs/Header\n{HEADER_MSG}{SEPARATOR}MSG: builtin_interfaces/Time\n{TIME_MSG}"
        f"{SEPARATOR}MSG: geometry_msgs/PoseWithCovariance\ngeometry_msgs/Pose pose\nfloat64[36] covariance\n"
        f"{SEPARATOR}MSG: geometry_msgs/Pose\ngeometry_msgs/Point position\ngeometry_msgs/Quaternion orientation\n"
        f"{SEPARATOR}MSG: geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z\n"
        f"{SEPARATOR}MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"
    ),
    "tf2_msgs/msg/TFMessage": (
        "geometry_msgs/TransformStamped[] transforms\n"
        f"{SEPARATOR}MSG: geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\n"
        f"geometry_msgs/Transform transform\n"
        f"{SEPARATOR}MSG: std_msgs/Header\n{HEADER_MSG}{SEPARATOR}MSG: builtin_interfaces/Time\n{TIME_MSG}"
        f"{SEPARATOR}MSG: geometry_msgs/Transform\ngeometry_msgs/Vector3 translation\n"
        f"geometry_msgs/Quaternion rotation\n"
        f"{SEPARATOR}MSG: geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z\n"
        f"{SEPARATOR}MSG: geometry_msgs/Quaternion\nfloat64 x\nfloat64 y\nfloat64 z\nfloat64 w\n"
    ),
}
# sensor_msgs/PointField datatype ids (ROS definition).
FLOAT32, UINT16, UINT32 = 7, 4, 6


def stamp_dict(seconds_since_start: float) -> dict:
    """ROS time dict for an offset from the epoch base, exact to the nanosecond.

    Args:
        seconds_since_start: Offset in seconds.

    Returns:
        {"sec", "nanosec"}.
    """
    total_ns = EPOCH_BASE_SECONDS * NANOSECONDS_PER_SECOND + round(seconds_since_start * NANOSECONDS_PER_SECOND)
    return {"sec": total_ns // NANOSECONDS_PER_SECOND, "nanosec": total_ns % NANOSECONDS_PER_SECOND}


def header_dict(seconds_since_start: float, frame_id: str) -> dict:
    """Header message dict.

    Args:
        seconds_since_start: Stamp offset.
        frame_id: Frame id string.
    """
    return {"stamp": stamp_dict(seconds_since_start), "frame_id": frame_id}


def lidar_scene_start_frame() -> np.ndarray:
    """Static scene points in the lidar frame AT SWEEP START, shape [rings, columns, 3].

    Direction per (ring, column): azimuth = 2*pi*column/columns, elevation spread over
    [-20, +20] degrees across rings. Every 8th column is a "robot body" return at 0.4 m,
    the rest are scene returns at ranges 5..20 m (seeded). Ring 0 / column 3 is a no-return.

    Returns:
        float64 array of lidar-frame coordinates (0 for the no-return).
    """
    rng = np.random.default_rng(0)
    ring_ids = np.arange(RING_COUNT)[:, None]
    column_ids = np.arange(COLUMN_COUNT)[None, :]
    elevation = np.radians(-20.0 + 40.0 * ring_ids / (RING_COUNT - 1))
    azimuth = 2.0 * np.pi * column_ids / COLUMN_COUNT
    scene_range = rng.uniform(5.0, 20.0, size=(RING_COUNT, COLUMN_COUNT))
    ranges = np.where(column_ids % 8 == 0, 0.4, scene_range)
    points = np.stack(
        [ranges * np.cos(elevation) * np.cos(azimuth),
         ranges * np.cos(elevation) * np.sin(azimuth),
         ranges * np.sin(elevation) + 0 * azimuth], axis=-1)
    points[0, 3] = 0.0
    return points


def build_pointcloud_message(stamp_seconds: float, source: str) -> dict:
    """PointCloud2 dict for one lidar sweep ("deskewed" or "raw").

    The world is static. A point seen at sweep-relative time t in a lidar that moved
    ROBOT_SPEED*t along +x is p_start - (ROBOT_SPEED*t, 0, 0) in the raw (capture-time) frame
    and p_start in the deskewed (sweep-start) frame.

    Args:
        stamp_seconds: Sweep start offset.
        source: "deskewed" or "raw".

    Returns:
        Message dict with x,y,z,intensity,t,reflectivity,ring,ambient,range fields (Ouster layout).
    """
    start_frame = lidar_scene_start_frame()
    column_ids = np.broadcast_to(np.arange(COLUMN_COUNT)[None, :], (RING_COUNT, COLUMN_COUNT))
    ring_ids = np.broadcast_to(np.arange(RING_COUNT)[:, None], (RING_COUNT, COLUMN_COUNT))
    # Per-point time offset in integer ns: column * (period / columns); exact for 0.1 s / 64.
    period_ns = round(LIDAR_PERIOD * NANOSECONDS_PER_SECOND)
    point_time_ns = (column_ids * (period_ns // COLUMN_COUNT)).astype(np.uint32)
    valid = np.linalg.norm(start_frame, axis=-1) > 0
    points = start_frame.copy()
    if source == "raw":
        points[..., 0] -= ROBOT_SPEED * point_time_ns * 1e-9
    points[~valid] = 0.0
    range_mm = (np.linalg.norm(points, axis=-1) * 1000.0).astype(np.uint32)
    range_mm[~valid] = 0
    reflectivity = ((ring_ids * COLUMN_COUNT + column_ids) % 1000).astype(np.uint16)
    point_dtype = np.dtype({
        "names": ["x", "y", "z", "intensity", "t", "reflectivity", "ring", "ambient", "range"],
        "formats": ["<f4", "<f4", "<f4", "<f4", "<u4", "<u2", "<u2", "<u2", "<u4"],
        "offsets": [0, 4, 8, 16, 20, 24, 26, 28, 32], "itemsize": 48,
    })
    cloud = np.zeros(RING_COUNT * COLUMN_COUNT, dtype=point_dtype)
    cloud["x"], cloud["y"], cloud["z"] = (points[..., axis].ravel() for axis in range(3))
    cloud["intensity"] = reflectivity.ravel()
    cloud["t"], cloud["reflectivity"] = point_time_ns.ravel(), reflectivity.ravel()
    cloud["ring"], cloud["range"] = ring_ids.ravel(), range_mm.ravel()
    field_specs = [("x", 0, FLOAT32), ("y", 4, FLOAT32), ("z", 8, FLOAT32), ("intensity", 16, FLOAT32),
                   ("t", 20, UINT32), ("reflectivity", 24, UINT16), ("ring", 26, UINT16),
                   ("ambient", 28, UINT16), ("range", 32, UINT32)]
    return {
        "header": header_dict(stamp_seconds, "os_sensor"), "height": RING_COUNT, "width": COLUMN_COUNT,
        "fields": [{"name": name, "offset": offset, "datatype": datatype, "count": 1}
                   for name, offset, datatype in field_specs],
        "is_bigendian": False, "point_step": 48, "row_step": 48 * COLUMN_COUNT,
        "data": cloud.tobytes(), "is_dense": True,
    }


def build_radar_message(stamp_seconds: float) -> dict:
    """Radar PointCloud2 dict with 5 points, the last having a NaN doppler (must be dropped).

    Args:
        stamp_seconds: Stamp offset.
    """
    field_names = ["x", "y", "z", "range", "elevation", "azimuth", "power", "doppler"]
    values = np.arange(RADAR_POINT_COUNT * len(field_names), dtype=np.float32).reshape(RADAR_POINT_COUNT, -1)
    values[-1, field_names.index("doppler")] = np.nan
    return {
        "header": header_dict(stamp_seconds, "hugin_radar"), "height": 1, "width": RADAR_POINT_COUNT,
        "fields": [{"name": name, "offset": 4 * index, "datatype": FLOAT32, "count": 1}
                   for index, name in enumerate(field_names)],
        "is_bigendian": False, "point_step": 4 * len(field_names),
        "row_step": 4 * len(field_names) * RADAR_POINT_COUNT,
        "data": values.astype("<f4").tobytes(), "is_dense": False,
    }


def synthetic_image() -> np.ndarray:
    """Distorted-looking test image: a bright grid on a gradient (BGR uint8)."""
    rows = np.arange(IMAGE_HEIGHT)[:, None]
    columns = np.arange(IMAGE_WIDTH)[None, :]
    gradient = (rows * 255 // IMAGE_HEIGHT + 0 * columns).astype(np.uint8)
    image = np.stack([gradient, gradient[::-1], np.full_like(gradient, 90)], axis=-1)
    image[::10, :] = 255
    image[:, ::10] = 255
    return image


def camera_info_message() -> dict:
    """CameraInfo dict with plumb_bob distortion (k1 only)."""
    camera_matrix = [CAMERA_FOCAL, 0.0, IMAGE_WIDTH / 2, 0.0, CAMERA_FOCAL, IMAGE_HEIGHT / 2, 0.0, 0.0, 1.0]
    return {
        "header": header_dict(0.0, "camera"), "height": IMAGE_HEIGHT, "width": IMAGE_WIDTH,
        "distortion_model": "plumb_bob", "D": [CAMERA_K1, 0.0, 0.0, 0.0, 0.0], "K": camera_matrix,
        "R": [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0], "P": camera_matrix + [0.0] * 3,
    }


def transform_entry(child: str, translation, quaternion_xyzw) -> dict:
    """One geometry_msgs/TransformStamped dict base_link -> child.

    Args:
        child: Child frame id.
        translation: (3,) translation.
        quaternion_xyzw: (4,) rotation.
    """
    return {
        "header": header_dict(0.0, "base_link"), "child_frame_id": child,
        "transform": {
            "translation": dict(zip("xyz", map(float, translation))),
            "rotation": dict(zip("xyzw", map(float, quaternion_xyzw))),
        },
    }


def write_chunk(path: Path, chunk_number: int) -> None:
    """Write one synthetic chunk covering [n*D, (n+1)*D) of sensor time.

    Args:
        path: Output MCAP path.
        chunk_number: Zero-based chunk index.
    """
    allow_legacy_field_names()
    window_start = chunk_number * CHUNK_DURATION
    window_end = window_start + CHUNK_DURATION

    def stamps(period: float, offset: float) -> np.ndarray:
        """Stamps k*period+offset inside the chunk window (rounded to ns to avoid float drift)."""
        count = int(np.ceil((window_end + period) / period))
        candidates = np.round((np.arange(count) * period + offset) * NANOSECONDS_PER_SECOND) / NANOSECONDS_PER_SECOND
        return candidates[(candidates >= window_start - 1e-12) & (candidates < window_end - 1e-12)]

    import io
    jpeg_ok, jpeg_buffer = cv2.imencode(".jpg", synthetic_image())
    assert jpeg_ok
    messages = []  # (stamp, topic, datatype, payload)
    messages.append((window_start, "/ids_camera/image_raw/camera_info", "sensor_msgs/msg/CameraInfo", camera_info_message()))
    # Same transforms as extrinsics_real.txt (hugin quaternion normalised, as a real tf would be).
    hugin_quaternion = np.array([0.701, 0.712, 0.028, 0.027])
    messages.append((window_start, "/tf", "tf2_msgs/msg/TFMessage", {"transforms": [
        transform_entry("camera", CAMERA_TRANSLATION, [-0.5, 0.5, -0.5, 0.5]),
        transform_entry("os_sensor", LIDAR_TRANSLATION, [0, 0, 0, 1]),
        transform_entry("hugin_radar", [0.2, 0.17, 0.08], hugin_quaternion / np.linalg.norm(hugin_quaternion)),
    ]}))
    for stamp in stamps(ODOMETRY_PERIOD, 0.0):
        messages.append((stamp, "/icp_odom", "nav_msgs/msg/Odometry", {
            "header": header_dict(stamp, "map"), "child_frame_id": "base_link",
            "pose": {"pose": {"position": {"x": ROBOT_SPEED * stamp, "y": 0.0, "z": 0.0},
                              "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0}},
                     "covariance": [0.0] * 36}}))
    for stamp in stamps(CAMERA_PERIOD, CAMERA_OFFSET):
        messages.append((stamp, "/ids_camera/image_raw/compressed", "sensor_msgs/msg/CompressedImage", {
            "header": header_dict(stamp, "camera"), "format": "rgb8; jpeg compressed bgr8",
            "data": jpeg_buffer.tobytes()}))
    for stamp in stamps(LIDAR_PERIOD, LIDAR_OFFSET):
        messages.append((stamp, "/ouster/points", "sensor_msgs/msg/PointCloud2", build_pointcloud_message(stamp, "raw")))
        messages.append((stamp, "/point_cloud_deskewed", "sensor_msgs/msg/PointCloud2", build_pointcloud_message(stamp, "deskewed")))
    for stamp in stamps(RADAR_PERIOD, 0.0):
        messages.append((stamp, "/hugin_raf_1/radar_data", "sensor_msgs/msg/PointCloud2", build_radar_message(stamp)))
    messages.sort(key=lambda record: record[0])
    with open(path, "wb") as handle:
        writer = Writer(handle)
        schema_by_type = {datatype: writer.register_msgdef(datatype, text) for datatype, text in SCHEMAS.items()}
        for stamp, topic, datatype, payload in messages:
            log_time = EPOCH_BASE_SECONDS * NANOSECONDS_PER_SECOND + round((stamp + LOG_LAG) * NANOSECONDS_PER_SECOND)
            writer.write_message(topic, schema_by_type[datatype], payload, log_time=log_time)
        writer.finish()


def expected_stamps(period: float, offset: float, last_allowed: float, chunk_count: int) -> np.ndarray:
    """Stamps of a sensor over ``chunk_count`` chunks that have odometry (<= last_allowed).

    Args:
        period, offset: Sensor timing.
        last_allowed: Latest time with a pose.
        chunk_count: Chunks present.
    """
    total = chunk_count * CHUNK_DURATION
    candidates = np.round((np.arange(int(total / period) + 2) * period + offset) * NANOSECONDS_PER_SECOND) / NANOSECONDS_PER_SECOND
    return candidates[(candidates < total - 1e-12) & (candidates <= last_allowed + 1e-12)]


def make_config(tmp_path: Path, lidar_source: str = "deskewed", output_name: str = "out") -> dict:
    """Session config dict mirroring the YAML structure, rooted in ``tmp_path``.

    Args:
        tmp_path: Temp directory containing ``bag`` and ``calibration``.
        lidar_source: "deskewed" or "raw_own_deskew".
        output_name: Output directory name.
    """
    return {
        "session": {"name": "synthetic_session", "extrinsics_variant": "short_grass",
                    "bag_dir": tmp_path / "bag", "calibration_dir": tmp_path / "calibration",
                    "output_dir": tmp_path / output_name, "labels_json": tmp_path / "labels.json",
                    "labels_sample_key": "x", "gt_dir": tmp_path / "gt"},
        "window": {"chunk_files": None, "start_offset_s": None, "end_offset_s": None},
        "topics": {"camera_image": "/ids_camera/image_raw/compressed",
                   "camera_info": "/ids_camera/image_raw/camera_info", "lidar_raw": "/ouster/points",
                   "lidar_deskewed": "/point_cloud_deskewed", "radar": "/hugin_raf_1/radar_data",
                   "pose": "/icp_odom", "tf": "/tf"},
        "frames": {"base": "base_link", "camera": "camera", "lidar": "os_sensor",
                   "radar": "hugin_radar", "map": "map"},
        "camera": {"name": "ids_camera", "timestamp_convention": "center", "readout_seconds": 0.0,
                   "undistort_alpha": 0.0, "resize_factor": 1.0, "jpeg_quality": 95},
        "lidar": {"name": "ouster", "source": lidar_source, "reference": "start", "point_time_field": "t",
                  "point_time_unit_seconds": 1e-9, "range_field": "range", "range_unit_meters": 1e-3,
                  "intensity_field": "reflectivity", "intensity_percentile": 99.0, "ring_field": "ring",
                  "ego_margin_m": 0.05, "model_derivation_scan_count": 5},
        "radar": {"name": "hugin", "fields": ["x", "y", "z", "power", "doppler"]},
        "split": {"train_fraction": 0.8},
        "splatad": {"load_cuboids": False, "add_missing_points": False},
    }


@pytest.fixture
def synthetic_session(tmp_path):
    """tmp_path holding calibration plus chunk 0 only (chunk 1 is 'downloaded later')."""
    (tmp_path / "bag").mkdir()
    (tmp_path / "calibration" / "extrinsics").mkdir(parents=True)
    shutil.copy(FIXTURE_EXTRINSICS, tmp_path / "calibration" / "extrinsics" / "extrinsics.txt")
    write_chunk(tmp_path / "bag" / "short_grass__ros2_0.mcap", 0)
    return tmp_path


def directory_fingerprint(root: Path) -> dict:
    """Relative path -> content hash for every file below ``root`` (detects any rewrite)."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*")) if path.is_file()
    }


def test_single_chunk_passes_contract_with_analytic_values(synthetic_session):
    """Chunk 0 alone: contract passes, frame counts / poses / intrinsics match the analytic scene."""
    config = make_config(synthetic_session)
    result = run_pipeline(config)
    output_dir = config["session"]["output_dir"]
    report = validate_dataset(output_dir)
    assert report.passed, report.to_markdown()
    last_pose_time = CHUNK_DURATION - ODOMETRY_PERIOD  # last odometry stamp of chunk 0
    camera_stamps = expected_stamps(CAMERA_PERIOD, CAMERA_OFFSET, last_pose_time, 1)
    lidar_stamps = expected_stamps(LIDAR_PERIOD, LIDAR_OFFSET, last_pose_time, 1)
    radar_stamps = expected_stamps(RADAR_PERIOD, 0.0, np.inf, 1)
    assert result.frames_total == {"camera": len(camera_stamps), "lidar": len(lidar_stamps), "radar": len(radar_stamps)}
    dataset = read_dataset(output_dir)
    camera_stream, lidar_stream = dataset.streams["ids_camera"], dataset.streams["ouster"]
    assert np.allclose(camera_stream.times, camera_stamps, atol=LAYOUT_TOLERANCE)
    assert np.allclose(lidar_stream.times, lidar_stamps, atol=LAYOUT_TOLERANCE)
    # Analytic poses: T_world_base = translate(v*t); camera/lidar add the fixed extrinsic offsets.
    assert np.allclose(camera_stream.poses_world_sensor[:, 0, 3], ROBOT_SPEED * camera_stamps + CAMERA_TRANSLATION[0])
    assert np.allclose(camera_stream.poses_world_sensor[:, 1, 3], CAMERA_TRANSLATION[1])
    assert np.allclose(camera_stream.poses_world_sensor[:, 2, 3], CAMERA_TRANSLATION[2])
    assert np.allclose(lidar_stream.poses_world_sensor[:, 0, 3], ROBOT_SPEED * lidar_stamps + LIDAR_TRANSLATION[0])
    # Camera rotation is the file's optical-frame quaternion (-.5,.5,-.5,.5): columns are the
    # optical axes (x right -> -y, y down -> -z, z forward -> +x) in the base frame.
    expected_rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    assert np.allclose(camera_stream.poses_world_sensor[0, :3, :3], expected_rotation, atol=1e-6)
    # Intrinsics: alpha=0 undistorted K from OpenCV directly, cropped to the valid ROI.
    camera_matrix = np.array([[CAMERA_FOCAL, 0, IMAGE_WIDTH / 2], [0, CAMERA_FOCAL, IMAGE_HEIGHT / 2], [0, 0, 1]])
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, np.array([CAMERA_K1, 0, 0, 0, 0.0]), (IMAGE_WIDTH, IMAGE_HEIGHT), 0, (IMAGE_WIDTH, IMAGE_HEIGHT))
    intrinsics = dataset.cameras["ids_camera"]
    assert (intrinsics.width, intrinsics.height) == (roi[2], roi[3])
    assert intrinsics.fx == pytest.approx(new_matrix[0, 0]) and intrinsics.cx == pytest.approx(new_matrix[0, 2] - roi[0])
    assert intrinsics.distortion is None
    assert camera_stream.frames[0].shape[:2] == (roi[3], roi[2])
    # Lidar content: ring ids are channel keys 0..31, deskewed t is the column time offset >= 0.
    sweep = lidar_stream.frames[0]
    assert sweep.shape[1] == 6 and sweep[:, 4].min() >= 0.0
    assert set(np.unique(sweep[:, 5]).astype(int)) <= set(range(RING_COUNT))
    assert 0.0 <= sweep[:, 3].min() and sweep[:, 3].max() <= 1.0
    # Ego box: body returns lie at range 0.4 m, so each half-extent is <= 0.4 + margin and > 0.
    ego_box = np.array(json.loads((output_dir / DERIVED_FILE_NAME).read_text())["min_lidar_dist"])
    assert np.all(ego_box > 0) and np.all(ego_box <= 0.4 + config["lidar"]["ego_margin_m"] + 1e-6)
    # Radar sidecar: NaN-doppler row dropped -> 4 rows x 5 requested fields.
    assert all(array.shape == (RADAR_POINT_COUNT - 1, 5) for array in dataset.radar_sidecar.values())
    # Cross-check against /tf: all three extrinsics agree to print precision.
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert all(entry["within_tolerance"] for entry in manifest["pipeline"]["extrinsic_check"].values())
    assert len(manifest["pipeline"]["extrinsic_check"]) == 3


def test_incremental_second_chunk_is_consistent_and_idempotent(synthetic_session):
    """A later chunk adds frames (incl. the recovered tail of chunk 0); derived.json is untouched."""
    config = make_config(synthetic_session)
    output_dir = config["session"]["output_dir"]
    run_pipeline(config)
    derived_before = (output_dir / DERIVED_FILE_NAME).read_bytes()
    write_chunk(synthetic_session / "bag" / "short_grass__ros2_1.mcap", 1)
    second = run_pipeline(config)
    assert second.processed_chunks == ["short_grass__ros2_1.mcap"]
    assert second.skipped_chunks == ["short_grass__ros2_0.mcap"]
    assert (output_dir / DERIVED_FILE_NAME).read_bytes() == derived_before
    assert validate_dataset(output_dir).passed
    # With both chunks every stamp up to the last odometry sample has a pose (chunk-0 tail included).
    last_pose_time = 2 * CHUNK_DURATION - ODOMETRY_PERIOD
    camera_stamps = expected_stamps(CAMERA_PERIOD, CAMERA_OFFSET, last_pose_time, 2)
    lidar_stamps = expected_stamps(LIDAR_PERIOD, LIDAR_OFFSET, last_pose_time, 2)
    radar_stamps = expected_stamps(RADAR_PERIOD, 0.0, np.inf, 2)
    assert second.frames_total == {"camera": len(camera_stamps), "lidar": len(lidar_stamps), "radar": len(radar_stamps)}
    dataset = read_dataset(output_dir)
    assert np.allclose(dataset.streams["ids_camera"].times, camera_stamps, atol=LAYOUT_TOLERANCE)
    assert np.allclose(dataset.streams["ouster"].poses_world_sensor[:, 0, 3],
                       ROBOT_SPEED * lidar_stamps + LIDAR_TRANSLATION[0])
    # The incremental result equals a from-scratch run over both chunks.
    fresh_config = make_config(synthetic_session, output_name="fresh")
    run_pipeline(fresh_config)
    fresh = read_dataset(fresh_config["session"]["output_dir"])
    assert np.array_equal(dataset.streams["ouster"].times, fresh.streams["ouster"].times)
    assert np.allclose(dataset.streams["ouster"].frames[-1], fresh.streams["ouster"].frames[-1])
    # Idempotent: a third run changes nothing on disk.
    fingerprint = directory_fingerprint(output_dir)
    third = run_pipeline(config)
    assert third.processed_chunks == [] and directory_fingerprint(output_dir) == fingerprint


def test_partial_run_then_full_run_equals_full_run(synthetic_session):
    """--max-seconds writes a prefix, does not mark the chunk done, and a full run completes it."""
    write_chunk(synthetic_session / "bag" / "short_grass__ros2_1.mcap", 1)
    config = make_config(synthetic_session)
    partial = run_pipeline(config, max_seconds=1.0)
    assert partial.frames_total["camera"] == len(expected_stamps(CAMERA_PERIOD, CAMERA_OFFSET, 1.0, 2))
    state = json.loads((config["session"]["output_dir"] / "extract_state.json").read_text())
    assert state["extracted_chunks"] == []
    run_pipeline(config)
    resumed = read_dataset(config["session"]["output_dir"])
    fresh_config = make_config(synthetic_session, output_name="fresh")
    run_pipeline(fresh_config)
    fresh = read_dataset(fresh_config["session"]["output_dir"])
    for sensor_name in ("ids_camera", "ouster"):
        assert np.array_equal(resumed.streams[sensor_name].times, fresh.streams[sensor_name].times)
    assert validate_dataset(config["session"]["output_dir"]).passed


def test_raw_own_deskew_matches_deskewed_topic(synthetic_session):
    """Own deskew (T13) of the raw topic reproduces the deskewed topic's points for a known motion."""
    deskewed_config = make_config(synthetic_session, "deskewed", "from_deskewed")
    raw_config = make_config(synthetic_session, "raw_own_deskew", "from_raw")
    run_pipeline(deskewed_config)
    run_pipeline(raw_config)
    from_deskewed = read_dataset(deskewed_config["session"]["output_dir"]).streams["ouster"]
    from_raw = read_dataset(raw_config["session"]["output_dir"]).streams["ouster"]
    # Raw runs need poses for the whole sweep, so compare the common prefix of sweeps.
    common = min(len(from_deskewed.frames), len(from_raw.frames))
    assert common > 0 and validate_dataset(raw_config["session"]["output_dir"]).passed
    for index in range(common):
        assert from_deskewed.frames[index].shape == from_raw.frames[index].shape
        # float32 storage of coordinates ~ 1e-6 relative; the slack covers mm range quantisation.
        assert np.allclose(from_deskewed.frames[index][:, :3], from_raw.frames[index][:, :3], atol=2e-3)
        assert np.allclose(from_deskewed.frames[index][:, 4], from_raw.frames[index][:, 4], atol=1e-6)


def test_sessions_never_mix(synthetic_session):
    """A derived.json from another session is refused instead of reused."""
    config = make_config(synthetic_session)
    run_pipeline(config)
    other_config = make_config(synthetic_session)
    other_config["session"]["name"] = "another_session"
    with pytest.raises(ValueError, match="another_session|synthetic_session"):
        run_pipeline(other_config)


def test_force_rebuilds_from_scratch(synthetic_session):
    """--force removes old outputs and re-extracts the same frames."""
    config = make_config(synthetic_session)
    first = run_pipeline(config)
    forced = run_pipeline(config, force=True)
    assert forced.processed_chunks == ["short_grass__ros2_0.mcap"]
    assert forced.frames_total == first.frames_total
