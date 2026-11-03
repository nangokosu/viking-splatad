"""Content-level checks of the synthetic bag -> dataset pipeline (audit area 6).

``test_pipeline_synthetic.py`` asserts counts, stamps, x-translation of poses and shapes. This file
compares the written CONTENT with the analytic scene that file defines:

* lidar sweeps: per-row xyz, intensity, per-point time and channel against
  ``lidar_scene_start_frame`` (row order = ring-major, column-minor, no-return and ego rows dropped);
* radar sidecar: values and column order, with the NaN-doppler row dropped;
* camera image: pixel content against an independent OpenCV undistortion of the synthetic image;
* lidar pose: identity rotation and the y / z of the fixed lidar extrinsic;
* the own-deskew path (raw topic) reproduces the analytic sweep-start points.

Every expected value is derived from the scene definitions, never by calling the pipeline code.
"""

import json
import shutil

import cv2
import numpy as np
import pytest

from tests.integration.test_pipeline_synthetic import (
    CAMERA_K1,
    CAMERA_FOCAL,
    CHUNK_DURATION,
    COLUMN_COUNT,
    EPOCH_BASE_SECONDS,
    FIXTURE_EXTRINSICS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    LIDAR_OFFSET,
    LIDAR_PERIOD,
    LIDAR_TRANSLATION,
    NANOSECONDS_PER_SECOND,
    ODOMETRY_PERIOD,
    RADAR_PERIOD,
    RADAR_POINT_COUNT,
    RING_COUNT,
    expected_stamps,
    lidar_scene_start_frame,
    make_config,
    stamp_dict,
    synthetic_image,
    write_chunk,
)
from viking_splatad.pipeline.bag_to_dataset import run_pipeline
from viking_splatad.pipeline.derived import DERIVED_FILE_NAME
from viking_splatad.dataset.writer import read_dataset

# Radar field order of the synthetic message (see build_radar_message), restated here so the
# expected sidecar is selected by NAME from the source layout, independently of the pipeline.
RADAR_SOURCE_FIELDS = ["x", "y", "z", "range", "elevation", "azimuth", "power", "doppler"]
# float32 storage of coordinates up to 20 m: half-ulp is 20 * 2**-24 ~ 1.2e-6, so 1e-5 is safe.
FLOAT32_COORDINATE_TOLERANCE = 1e-5
# Own-deskew interpolates poses linearly (exact for constant speed) but goes through float32 raw
# coordinates and a 4x4 round trip; same float32 budget plus slack for the speed * time product.
DESKEW_COORDINATE_TOLERANCE = 1e-4
# Mean absolute 8-bit difference allowed between the pipeline image (JPEG -> undistort -> JPEG)
# and the independent single undistortion: two lossy JPEG passes at quality 95 on a grid image.
IMAGE_MEAN_ABS_TOLERANCE = 6.0


def build_session(session_root):
    """Fill ``session_root`` with calibration plus chunk 0 of the synthetic bag.

    Args:
        session_root: Empty directory.

    Returns:
        ``session_root``.
    """
    (session_root / "bag").mkdir()
    (session_root / "calibration" / "extrinsics").mkdir(parents=True)
    shutil.copy(FIXTURE_EXTRINSICS, session_root / "calibration" / "extrinsics" / "extrinsics.txt")
    write_chunk(session_root / "bag" / "short_grass__ros2_0.mcap", 0)
    return session_root


@pytest.fixture(scope="module")
def deskewed_run(tmp_path_factory):
    """Run the pipeline once (deskewed source, default radar fields) and return its artefacts.

    Returns:
        dict with config, output_dir and the dataset read back from disk.
    """
    session_root = build_session(tmp_path_factory.mktemp("content_session"))
    config = make_config(session_root)
    run_pipeline(config)
    output_dir = config["session"]["output_dir"]
    return {"config": config, "output_dir": output_dir, "dataset": read_dataset(output_dir)}


def analytic_lidar_rows(ego_half_extents, intensity_scale):
    """Expected [n, 6] rows (x, y, z, intensity, t, channel) of one deskewed sweep.

    Rows follow the cloud order (ring-major, column-minor). A cell is dropped when it is the
    no-return placeholder (zero range) or lies inside the ego box (all |axis| < half-extent).

    Args:
        ego_half_extents: (3,) ego box half extents in metres.
        intensity_scale: Divisor of the raw reflectivity.

    Returns:
        float64 array of expected rows.
    """
    start_frame = lidar_scene_start_frame()
    rows = []
    # Per-point time is column * (period_ns // columns) ns, converted to seconds.
    column_step_seconds = (round(LIDAR_PERIOD * NANOSECONDS_PER_SECOND) // COLUMN_COUNT) * 1e-9
    for ring_index in range(RING_COUNT):
        for column_index in range(COLUMN_COUNT):
            point = start_frame[ring_index, column_index]
            if not np.linalg.norm(point) > 0:
                continue  # no-return placeholder
            if not np.any(np.abs(point) >= np.asarray(ego_half_extents)):
                continue  # inside the ego box
            reflectivity = (ring_index * COLUMN_COUNT + column_index) % 1000
            rows.append([*point, min(reflectivity / intensity_scale, 1.0),
                         column_index * column_step_seconds, ring_index])
    return np.array(rows)


def independent_intensity_scale():
    """99th percentile of the valid reflectivities of one sweep (all sweeps carry the same values).

    Returns:
        The intensity divisor the pipeline must derive.
    """
    start_frame = lidar_scene_start_frame()
    ring_ids, column_ids = np.meshgrid(np.arange(RING_COUNT), np.arange(COLUMN_COUNT), indexing="ij")
    reflectivity = (ring_ids * COLUMN_COUNT + column_ids) % 1000
    valid = np.linalg.norm(start_frame, axis=-1) > 0
    return float(np.percentile(reflectivity[valid], 99.0))


def test_lidar_sweep_content_matches_analytic_scene(deskewed_run):
    """Every sweep equals the analytic scene row by row: xyz, intensity, time, channel."""
    output_dir, dataset = deskewed_run["output_dir"], deskewed_run["dataset"]
    derived = json.loads((output_dir / DERIVED_FILE_NAME).read_text())
    # The derived scale must be the independent 99th percentile; this pins T14's percentile.
    assert derived["intensity_scale"] == pytest.approx(independent_intensity_scale())
    expected = analytic_lidar_rows(derived["min_lidar_dist"], independent_intensity_scale())
    lidar_stream = dataset.streams["ouster"]
    assert len(lidar_stream.frames) > 1
    for sweep in lidar_stream.frames:
        assert sweep.shape == expected.shape
        assert np.allclose(sweep[:, :3], expected[:, :3], atol=FLOAT32_COORDINATE_TOLERANCE, rtol=0)
        assert np.allclose(sweep[:, 3], expected[:, 3], atol=1e-6, rtol=0)
        # Time column is exact in float32 up to its own half-ulp (values < 0.1 s).
        assert np.allclose(sweep[:, 4], expected[:, 4], atol=1e-8, rtol=0)
        assert np.array_equal(sweep[:, 5], expected[:, 5])


def test_ego_box_removes_exactly_the_body_returns_it_covers(deskewed_run):
    """Scene returns (>= 5 m) are never removed; the sweep has no zero-range row."""
    dataset = deskewed_run["dataset"]
    start_frame = lidar_scene_start_frame()
    scene_cell_count = int(np.count_nonzero(np.linalg.norm(start_frame, axis=-1) >= 5.0))
    body_cell_count = int(np.count_nonzero(np.isclose(np.linalg.norm(start_frame, axis=-1), 0.4)))
    for sweep in dataset.streams["ouster"].frames:
        ranges = np.linalg.norm(sweep[:, :3], axis=1)
        assert np.count_nonzero(ranges >= 5.0) == scene_cell_count
        assert np.all(ranges > 0)
        # Body returns kept can only be a subset of the body cells.
        assert np.count_nonzero(np.isclose(ranges, 0.4, atol=1e-4)) <= body_cell_count


def test_lidar_frame_timestamps_and_pose_rotation_and_offsets(deskewed_run):
    """Frame stamps are header stamps in ns; lidar pose = identity rotation, y = 0, z = extrinsic z."""
    dataset = deskewed_run["dataset"]
    lidar_stream = dataset.streams["ouster"]
    last_pose_time = CHUNK_DURATION - ODOMETRY_PERIOD
    stamps = expected_stamps(LIDAR_PERIOD, LIDAR_OFFSET, last_pose_time, 1)
    expected_ns = [EPOCH_BASE_SECONDS * NANOSECONDS_PER_SECOND + round(stamp * NANOSECONDS_PER_SECOND)
                   for stamp in stamps]
    assert lidar_stream.frame_timestamps_ns.tolist() == expected_ns
    poses = lidar_stream.poses_world_sensor
    # Robot orientation is identity and the lidar extrinsic quaternion is (0, 0, 0, 1).
    assert np.allclose(poses[:, :3, :3], np.eye(3), atol=1e-9)
    assert np.allclose(poses[:, 1, 3], LIDAR_TRANSLATION[1], atol=1e-9)
    assert np.allclose(poses[:, 2, 3], LIDAR_TRANSLATION[2], atol=1e-9)
    assert np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0])
    # stamp_dict consistency: the helper used to write the bag round-trips the same nanoseconds.
    first = stamp_dict(stamps[0])
    assert first["sec"] * NANOSECONDS_PER_SECOND + first["nanosec"] == expected_ns[0]


def test_radar_sidecar_values_and_default_column_order(deskewed_run):
    """Each scan keeps the 4 finite rows with columns (x, y, z, power, doppler) in config order."""
    dataset = deskewed_run["dataset"]
    config = deskewed_run["config"]
    source = np.arange(RADAR_POINT_COUNT * len(RADAR_SOURCE_FIELDS), dtype=np.float32).reshape(RADAR_POINT_COUNT, -1)
    column_indices = [RADAR_SOURCE_FIELDS.index(name) for name in config["radar"]["fields"]]
    expected = source[:-1][:, column_indices]  # the last row has NaN doppler and is dropped
    radar_stamps = expected_stamps(RADAR_PERIOD, 0.0, np.inf, 1)
    assert len(dataset.radar_sidecar) == len(radar_stamps)
    for scan in dataset.radar_sidecar.values():
        assert scan.dtype == np.float32
        assert np.array_equal(scan, expected)


def test_radar_sidecar_follows_a_reordered_field_list(tmp_path):
    """Requesting ("doppler", "power", "x") yields exactly those columns in that order."""
    session_root = build_session(tmp_path)
    config = make_config(session_root)
    config["radar"]["fields"] = ["doppler", "power", "x"]
    run_pipeline(config)
    sidecar = read_dataset(config["session"]["output_dir"]).radar_sidecar
    source = np.arange(RADAR_POINT_COUNT * len(RADAR_SOURCE_FIELDS), dtype=np.float32).reshape(RADAR_POINT_COUNT, -1)
    expected = source[:-1][:, [RADAR_SOURCE_FIELDS.index(name) for name in ("doppler", "power", "x")]]
    assert len(sidecar) > 0
    assert all(np.array_equal(scan, expected) for scan in sidecar.values())


def test_camera_image_content_matches_independent_undistortion(deskewed_run):
    """Pixels equal the synthetic image undistorted with OpenCV (alpha 0, cropped to the ROI)."""
    dataset = deskewed_run["dataset"]
    # Independent expectation: JPEG round trip of the source (as in the bag), then one undistortion.
    encode_ok, encoded = cv2.imencode(".jpg", synthetic_image())
    assert encode_ok
    source_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    camera_matrix = np.array([[CAMERA_FOCAL, 0, IMAGE_WIDTH / 2], [0, CAMERA_FOCAL, IMAGE_HEIGHT / 2], [0, 0, 1]])
    distortion = np.array([CAMERA_K1, 0, 0, 0, 0.0])
    new_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, (IMAGE_WIDTH, IMAGE_HEIGHT), 0, (IMAGE_WIDTH, IMAGE_HEIGHT))
    undistorted = cv2.undistort(source_bgr, camera_matrix, distortion, None, new_matrix)
    roi_x, roi_y, roi_width, roi_height = roi
    expected = undistorted[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width].astype(np.float64)
    frames = dataset.streams["ids_camera"].frames
    assert len(frames) > 0
    for frame in frames:
        assert frame.shape == expected.shape and frame.dtype == np.uint8
        assert np.abs(frame.astype(np.float64) - expected).mean() < IMAGE_MEAN_ABS_TOLERANCE
        # Per-channel means separate a B/R swap (gradient vs reversed gradient vs constant 90).
        for channel_index in range(3):
            assert abs(frame[..., channel_index].mean() - expected[..., channel_index].mean()) < IMAGE_MEAN_ABS_TOLERANCE


def test_raw_own_deskew_reproduces_analytic_sweep_start_points(tmp_path):
    """Raw topic + own deskew recovers the analytic sweep-start points and column times."""
    session_root = build_session(tmp_path)
    config = make_config(session_root, "raw_own_deskew", "from_raw")
    run_pipeline(config)
    output_dir = config["session"]["output_dir"]
    derived = json.loads((output_dir / DERIVED_FILE_NAME).read_text())
    expected = analytic_lidar_rows(derived["min_lidar_dist"], independent_intensity_scale())
    frames = read_dataset(output_dir).streams["ouster"].frames
    assert len(frames) > 0
    for sweep in frames:
        assert sweep.shape == expected.shape
        assert np.allclose(sweep[:, :3], expected[:, :3], atol=DESKEW_COORDINATE_TOLERANCE, rtol=0)
        assert np.allclose(sweep[:, 4], expected[:, 4], atol=1e-8, rtol=0)
        assert np.array_equal(sweep[:, 5], expected[:, 5])
