"""Synthetic known-answer tests for painting radar seed colours from camera frames."""

import cv2
import numpy as np
import pytest

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.dataset.writer import write_dataset
from viking_splatad.radar_init.seeds import NEUTRAL_GREY_RGB, paint_visible_seed_colors
from viking_splatad.splatad.viking_conversions import open_dataset_view


def test_camera_painting_uses_grey_for_lidar_occlusion():
    """A closer LiDAR point inside the radar footprint blocks camera color leakage.

    Moved unchanged from tests/unit/test_radar_initialization.py (whose other tests covered the
    deleted seed builder). Its old marker T36 now names the anchor covariance transform, so the
    marker is dropped here; the T37 painting driver gets its own covering tests.
    """
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2] = [12, 34, 56]
    intrinsics = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    colors, painted = paint_visible_seed_colors(
        np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
        np.eye(4), intrinsics, image, np.array([[0.0, 0.0, 3.0]]), 0.25, 0.1,
    )
    np.testing.assert_array_equal(colors[0], NEUTRAL_GREY_RGB)
    np.testing.assert_array_equal(colors[1], image[2, 2])
    np.testing.assert_array_equal(painted, [False, True])


# --- T37: paint_anchors_from_train_cameras on a tiny on-disk synthetic dataset -------------------

CAMERA_NAME = "cam_front"
LIDAR_NAME = "lidar_top"
FRAME_COUNT = 6  # frames per sensor; times are frame_number * 0.1 s (from build_synthetic_dataset)
ANCHOR_DEPTH_M = 10.0  # anchor depth along the camera axis
OCCLUDER_DEPTH_M = 5.0  # closer than the anchor by more than the margin
OCCLUSION_MARGIN_M = 0.5
CELL_RAD = 0.01  # azimuth and elevation cell; footprint ~0.3 px, so only the exact pixel counts
CHOSEN_FRAME = 3
FAR_BEHIND_POINT = [0.0, 0.0, -30.0, 0.5, 0.0, 0.0]  # sweep row that projects nowhere
IMAGE_SEED = 7


def _pixel_anchor_map(view, frame_index, pixel_column, pixel_row, depth_m):
    """Map-frame point that the pinhole model projects to an exact pixel in a given camera frame.

    Inverts u = fx*X/Z + cx, v = fy*Y/Z + cy with Z = depth, then applies T_map_camera.

    Args:
        view: Dataset view of the synthetic dataset.
        frame_index: Camera frame whose pose is used.
        pixel_column: Target pixel column u.
        pixel_row: Target pixel row v.
        depth_m: Depth Z along the optical axis.

    Returns:
        float64 [3] map-frame point.
    """
    intrinsics = view.manifest["cameras"][CAMERA_NAME]
    point_camera = np.array([
        (pixel_column - intrinsics["cx"]) * depth_m / intrinsics["fx"],
        (pixel_row - intrinsics["cy"]) * depth_m / intrinsics["fy"],
        depth_m,
    ])
    pose = view.sensor_poses[CAMERA_NAME][frame_index]
    return pose[:3, :3] @ point_camera + pose[:3, 3]


def _build_painting_fixture(tmp_path, occluder_map_points_by_frame=None):
    """Write a 6-frame dataset, random PNG-exact camera images and controlled lidar sweeps.

    Args:
        tmp_path: Output directory.
        occluder_map_points_by_frame: frame index -> [n,3] map points added to that lidar sweep.

    Returns:
        ``(view, images_rgb)`` with images_rgb[frame] the RGB array stored for each camera frame.
    """
    occluder_map_points_by_frame = occluder_map_points_by_frame or {}
    dataset = build_synthetic_dataset(num_frames=FRAME_COUNT)
    lidar_stream = dataset.streams[LIDAR_NAME]
    for frame_index in range(FRAME_COUNT):
        # Background: one far-behind row; occluders are converted map -> lidar sensor frame.
        rows = [np.array(FAR_BEHIND_POINT, dtype=np.float32)]
        pose = lidar_stream.poses_world_sensor[frame_index]
        for point_map in occluder_map_points_by_frame.get(frame_index, []):
            point_sensor = pose[:3, :3].T @ (np.asarray(point_map) - pose[:3, 3])
            rows.append(np.concatenate([point_sensor, [0.5, 0.0, 0.0]]).astype(np.float32))
        lidar_stream.frames[frame_index] = np.stack(rows)
    write_dataset(tmp_path, dataset)
    view = open_dataset_view(tmp_path)
    rng = np.random.default_rng(IMAGE_SEED)
    images_rgb = []
    for frame_path in view.sensor_frame_paths[CAMERA_NAME]:
        shape = cv2.imread(str(frame_path)).shape
        image_rgb = rng.integers(0, 256, size=shape, dtype=np.uint8)
        # PNG bytes under the .jpg name: lossless, and cv2.imread sniffs content, not extension.
        encoded_ok, png_bytes = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        assert encoded_ok
        frame_path.write_bytes(png_bytes.tobytes())
        images_rgb.append(image_rgb)
    return view, images_rgb


def _relative_names(view, sensor_name, frame_indices):
    """Dataset-relative posix names (the train_files format) of selected frames."""
    return {
        view.sensor_frame_paths[sensor_name][frame_index].relative_to(view.root).as_posix()
        for frame_index in frame_indices
    }


def _spans_around(times_s, half_width_s):
    """[S,2] time spans centred on the given midpoints."""
    return np.column_stack([times_s - half_width_s, times_s + half_width_s])


def _paint(view, anchors_map, midpoints_s, train_files):
    """Call the function under test (imported lazily so a missing name fails per test)."""
    from viking_splatad.radar_init.seeds import paint_anchors_from_train_cameras

    return paint_anchors_from_train_cameras(
        np.asarray(anchors_map, dtype=np.float64),
        _spans_around(np.asarray(midpoints_s, dtype=np.float64), 0.01),
        view, train_files, CAMERA_NAME, LIDAR_NAME, CELL_RAD, CELL_RAD, OCCLUSION_MARGIN_M,
    )


def _all_train_files(view, camera_frames=None):
    """Train set: all lidar sweeps plus the given camera frames (default all)."""
    camera_frames = range(FRAME_COUNT) if camera_frames is None else camera_frames
    return _relative_names(view, LIDAR_NAME, range(FRAME_COUNT)) | _relative_names(
        view, CAMERA_NAME, camera_frames
    )


@pytest.mark.covers("T37")
def test_visible_anchor_takes_projected_pixel_colour_and_frame_stamp(tmp_path):
    """Anchor at a chosen pixel gets that pixel's colour and the nearest frame's timestamp."""
    view, images_rgb = _build_painting_fixture(tmp_path)
    pixel_column, pixel_row = 20, 7
    anchor = _pixel_anchor_map(view, CHOSEN_FRAME, pixel_column, pixel_row, ANCHOR_DEPTH_M)
    midpoint = view.sensor_times[CAMERA_NAME][CHOSEN_FRAME]
    rgb, painted, stamps = _paint(view, [anchor], [midpoint], _all_train_files(view))
    np.testing.assert_array_equal(rgb[0], images_rgb[CHOSEN_FRAME][pixel_row, pixel_column])
    assert painted.tolist() == [True]
    chosen_name = view.sensor_frame_paths[CAMERA_NAME][CHOSEN_FRAME].name
    expected_stamp = int(chosen_name.rsplit(".", 1)[0].split("_")[1])  # <idx>_<timestamp_ns>
    assert stamps.tolist() == [expected_stamp]


@pytest.mark.covers("T37")
def test_anchor_behind_closer_lidar_return_stays_grey(tmp_path):
    """A lidar return at the same pixel but nearer than depth - margin blocks the colour."""
    pixel_column, pixel_row = 12, 15
    probe_view, _ = _build_painting_fixture(tmp_path / "probe")
    occluder = _pixel_anchor_map(probe_view, CHOSEN_FRAME, pixel_column, pixel_row, OCCLUDER_DEPTH_M)
    view, images_rgb = _build_painting_fixture(
        tmp_path / "occluded", {CHOSEN_FRAME: [occluder]}
    )
    visible_pixel = (20, 7)
    anchors = [
        _pixel_anchor_map(view, CHOSEN_FRAME, pixel_column, pixel_row, ANCHOR_DEPTH_M),
        _pixel_anchor_map(view, CHOSEN_FRAME, *visible_pixel, ANCHOR_DEPTH_M),
    ]
    midpoint = view.sensor_times[CAMERA_NAME][CHOSEN_FRAME]
    rgb, painted, stamps = _paint(view, anchors, [midpoint] * 2, _all_train_files(view))
    np.testing.assert_array_equal(rgb[0], NEUTRAL_GREY_RGB)
    assert painted.tolist() == [False, True]
    assert stamps[0] == -1 and stamps[1] != -1
    np.testing.assert_array_equal(rgb[1], images_rgb[CHOSEN_FRAME][visible_pixel[1], visible_pixel[0]])


@pytest.mark.covers("T37")
def test_anchor_behind_camera_or_outside_image_stays_grey(tmp_path):
    """Negative depth and a pixel beyond the image width are both unpainted."""
    view, _ = _build_painting_fixture(tmp_path)
    width = view.manifest["cameras"][CAMERA_NAME]["width"]
    anchors = [
        _pixel_anchor_map(view, CHOSEN_FRAME, 16, 12, -ANCHOR_DEPTH_M),  # behind the camera
        _pixel_anchor_map(view, CHOSEN_FRAME, width + 5, 12, ANCHOR_DEPTH_M),  # right of image
    ]
    midpoint = view.sensor_times[CAMERA_NAME][CHOSEN_FRAME]
    rgb, painted, stamps = _paint(view, anchors, [midpoint] * 2, _all_train_files(view))
    np.testing.assert_array_equal(rgb, np.tile(NEUTRAL_GREY_RGB, (2, 1)))
    assert painted.tolist() == [False, False]
    assert stamps.tolist() == [-1, -1]


@pytest.mark.covers("T37")
def test_only_train_camera_frames_are_used(tmp_path):
    """With the nearest frame held out, the next-nearest train frame supplies the colour."""
    view, images_rgb = _build_painting_fixture(tmp_path)
    times = view.sensor_times[CAMERA_NAME]
    # Midpoint 0.3 of a frame period after the held-out frame: order is held-out, +1, then -1.
    midpoint = times[CHOSEN_FRAME] + 0.3 * (times[1] - times[0])
    next_frame = CHOSEN_FRAME + 1
    pixel_column, pixel_row = 20, 7
    anchor = _pixel_anchor_map(view, next_frame, pixel_column, pixel_row, ANCHOR_DEPTH_M)
    train_files = _all_train_files(
        view, [frame for frame in range(FRAME_COUNT) if frame != CHOSEN_FRAME]
    )
    rgb, painted, stamps = _paint(view, [anchor], [midpoint], train_files)
    expected = images_rgb[next_frame][pixel_row, pixel_column]
    # The test can tell the frames apart only if they disagree at this pixel.
    assert not np.array_equal(expected, images_rgb[CHOSEN_FRAME][pixel_row, pixel_column])
    np.testing.assert_array_equal(rgb[0], expected)
    assert painted.tolist() == [True]
    next_name = view.sensor_frame_paths[CAMERA_NAME][next_frame].name
    assert stamps.tolist() == [int(next_name.rsplit(".", 1)[0].split("_")[1])]


@pytest.mark.covers("T37")
def test_output_dtypes_and_shapes(tmp_path):
    """uint8 [S,3] colours, bool [S] painted flags, int64 [S] stamps for a mixed batch."""
    view, _ = _build_painting_fixture(tmp_path)
    anchors = [
        _pixel_anchor_map(view, CHOSEN_FRAME, 20, 7, ANCHOR_DEPTH_M),
        _pixel_anchor_map(view, CHOSEN_FRAME, 16, 12, -ANCHOR_DEPTH_M),
        _pixel_anchor_map(view, CHOSEN_FRAME, 10, 5, ANCHOR_DEPTH_M),
    ]
    midpoint = view.sensor_times[CAMERA_NAME][CHOSEN_FRAME]
    rgb, painted, stamps = _paint(view, anchors, [midpoint] * len(anchors), _all_train_files(view))
    assert rgb.dtype == np.uint8 and rgb.shape == (len(anchors), 3)
    assert painted.dtype == np.bool_ and painted.shape == (len(anchors),)
    assert stamps.dtype == np.int64 and stamps.shape == (len(anchors),)
