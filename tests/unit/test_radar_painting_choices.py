"""Painting choices decided 2026-11-14: K nearest train frames, one-cell footprint, review fixes.

Synthetic known-answer tests. The dataset is the 6-frame fixture of ``test_radar_anchor_painting``:
identity camera rotation, camera x translation of 0.5 m per frame, fx = 30, fy = 31, cx = 16,
cy = 12, 32 x 24 pixels. A static map point at camera-frame lateral offset X and depth Z projects to
column u = fx * X / Z + cx; one frame later (camera 0.5 m further along +x) X shrinks by 0.5, so u
drops by fx * 0.5 / Z pixels. At Z = ANCHOR_DEPTH_M that is 1.5 px per frame.
"""

import numpy as np
import pytest

from tests.unit.test_loss_only_artifact import synthetic_artifact
from tests.unit.test_radar_anchor_painting import (
    ANCHOR_DEPTH_M, CAMERA_NAME, LIDAR_NAME, OCCLUDER_DEPTH_M, OCCLUSION_MARGIN_M,
    _all_train_files, _build_painting_fixture, _pixel_anchor_map, _spans_around,
)
from tests.unit.test_radar_painted_build import painted_fixture
from viking_splatad.dataset.run_split import SplitFrame
from viking_splatad.radar_init.artifact import validate_painting_record
from viking_splatad.splatad.viking_conversions import open_dataset_view
from viking_splatad.radar_init.seeds import NEUTRAL_GREY_RGB, paint_anchors_from_train_cameras, project_map_points

# Frame the anchors' time midpoint sits just after, so the time order is 3, 4, 2, 5, 1, 0.
NEAREST_FRAME = 3
# Column that the second-nearest frame sees at the right image edge (inside), the nearest frame outside.
EDGE_COLUMN = 31
PIXEL_ROW = 7
# Fraction of a frame period past the nearest frame (breaks the 3/4/2 time ties, as in the existing test).
MIDPOINT_OFFSET_FRACTION = 0.3
# Cell angle (rad) large enough for the footprint radius to span several pixels.
WIDE_CELL_RAD = 0.2


def _midpoint(view):
    """Anchor time midpoint: just after NEAREST_FRAME, so the order is 3, 4, 2, 5, 1, 0."""
    times = view.sensor_times[CAMERA_NAME]
    return times[NEAREST_FRAME] + MIDPOINT_OFFSET_FRACTION * (times[1] - times[0])


def _stamp_of(view, frame_index):
    """Timestamp_ns encoded in a camera frame's file name ``<idx>_<timestamp_ns>.<ext>``."""
    name = view.sensor_frame_paths[CAMERA_NAME][frame_index].name
    return int(name.rsplit(".", 1)[0].split("_")[1])


def _paint(view, anchors_map, cell_rad=0.01, frames_per_camera=None):
    """Call the painter with all frames in train and the shared midpoint."""
    keyword = {} if frames_per_camera is None else {"paint_frames_per_camera": frames_per_camera}
    return paint_anchors_from_train_cameras(
        np.asarray(anchors_map, dtype=np.float64),
        _spans_around(np.full(len(anchors_map), _midpoint(view)), 0.01),
        view, _all_train_files(view), CAMERA_NAME, LIDAR_NAME, cell_rad, cell_rad,
        OCCLUSION_MARGIN_M, **keyword,
    )


def _pixel_in_frame(view, anchor_map, frame_index):
    """Independent projection of a map point into a camera frame: (u, v)."""
    intrinsics = view.manifest["cameras"][CAMERA_NAME]
    camera_matrix = np.array([[intrinsics["fx"], 0, intrinsics["cx"]],
                              [0, intrinsics["fy"], intrinsics["cy"]], [0, 0, 1.0]])
    pixels, _ = project_map_points(anchor_map[None, :], view.sensor_poses[CAMERA_NAME][frame_index], camera_matrix)
    return pixels[0]


@pytest.mark.covers("T37")
def test_anchor_out_of_view_in_nearest_frame_takes_second_nearest_frame(tmp_path):
    """Anchor outside the image in frame 3 but inside in frame 4 gets frame 4's pixel and stamp."""
    view, images_rgb = _build_painting_fixture(tmp_path)
    width = view.manifest["cameras"][CAMERA_NAME]["width"]
    anchor = _pixel_anchor_map(view, NEAREST_FRAME + 1, EDGE_COLUMN, PIXEL_ROW, ANCHOR_DEPTH_M)
    # Premise from first principles: column in frame 3 is beyond the right edge.
    assert _pixel_in_frame(view, anchor, NEAREST_FRAME)[0] >= width
    rgb, painted, stamps = _paint(view, [anchor], frames_per_camera=2)
    np.testing.assert_array_equal(rgb[0], images_rgb[NEAREST_FRAME + 1][PIXEL_ROW, EDGE_COLUMN])
    assert painted.tolist() == [True]
    assert stamps.tolist() == [_stamp_of(view, NEAREST_FRAME + 1)]


@pytest.mark.covers("T37")
def test_anchor_visible_in_no_candidate_frame_stays_grey(tmp_path):
    """Anchor first visible in frame 5 (4th in time order) stays grey when K = 2."""
    view, _ = _build_painting_fixture(tmp_path)
    width = view.manifest["cameras"][CAMERA_NAME]["width"]
    anchor = _pixel_anchor_map(view, NEAREST_FRAME + 2, EDGE_COLUMN, PIXEL_ROW, ANCHOR_DEPTH_M)
    for frame_index in (NEAREST_FRAME, NEAREST_FRAME + 1):
        assert _pixel_in_frame(view, anchor, frame_index)[0] >= width
    rgb, painted, stamps = _paint(view, [anchor], frames_per_camera=2)
    np.testing.assert_array_equal(rgb[0], NEUTRAL_GREY_RGB)
    assert painted.tolist() == [False] and stamps.tolist() == [-1]


@pytest.mark.covers("T37")
def test_candidate_count_is_the_configured_k(tmp_path):
    """An anchor needing the 4th frame (order 3, 4, 2, 5) is grey for K = 3 and painted for K = 4."""
    view, images_rgb = _build_painting_fixture(tmp_path)
    anchor = _pixel_anchor_map(view, NEAREST_FRAME + 2, EDGE_COLUMN, PIXEL_ROW, ANCHOR_DEPTH_M)
    _, painted_k3, _ = _paint(view, [anchor], frames_per_camera=3)
    rgb_k4, painted_k4, stamps_k4 = _paint(view, [anchor], frames_per_camera=4)
    assert painted_k3.tolist() == [False]
    assert painted_k4.tolist() == [True]
    np.testing.assert_array_equal(rgb_k4[0], images_rgb[NEAREST_FRAME + 2][PIXEL_ROW, EDGE_COLUMN])
    assert stamps_k4.tolist() == [_stamp_of(view, NEAREST_FRAME + 2)]


@pytest.mark.covers("T37")
def test_occluded_in_nearest_frame_is_painted_by_next_frame(tmp_path):
    """A lidar occluder present only in frame 3's sweep sends the anchor to frame 4."""
    probe_view, _ = _build_painting_fixture(tmp_path / "probe")
    # Frame 4 sees column 14, frame 3 sees 14 + 1.5 = 15.5: both inside the image.
    anchor_pixel = (14, 15)
    anchor_map = _pixel_anchor_map(probe_view, NEAREST_FRAME + 1, *anchor_pixel, ANCHOR_DEPTH_M)
    pixel_in_nearest = _pixel_in_frame(probe_view, anchor_map, NEAREST_FRAME)
    occluder = _pixel_anchor_map(probe_view, NEAREST_FRAME, *pixel_in_nearest, OCCLUDER_DEPTH_M)
    view, images_rgb = _build_painting_fixture(tmp_path / "real", {NEAREST_FRAME: [occluder]})
    rgb, painted, stamps = _paint(view, [anchor_map], frames_per_camera=2)
    assert painted.tolist() == [True]
    assert stamps.tolist() == [_stamp_of(view, NEAREST_FRAME + 1)]
    np.testing.assert_array_equal(rgb[0], images_rgb[NEAREST_FRAME + 1][anchor_pixel[1], anchor_pixel[0]])


@pytest.mark.covers("T37")
def test_footprint_is_one_cell_wide_and_radius_is_returned(tmp_path):
    """Radius = max(fx tan(az/2), fy tan(el/2)); occluders between half and one cell no longer block."""
    probe_view, _ = _build_painting_fixture(tmp_path / "probe")
    intrinsics = probe_view.manifest["cameras"][CAMERA_NAME]
    # Half-angle radius (one cell wide) and the old full-angle radius, from the K matrix and cell angle.
    half_cell_radius = max(intrinsics["fx"], intrinsics["fy"]) * np.tan(WIDE_CELL_RAD / 2)
    full_cell_radius = max(intrinsics["fx"], intrinsics["fy"]) * np.tan(WIDE_CELL_RAD)
    assert half_cell_radius < full_cell_radius
    inside_offset = 0.5 * half_cell_radius  # inside half a cell: still occludes
    between_offset = 0.5 * (half_cell_radius + full_cell_radius)  # old footprint blocked it, the new one not
    near_anchor = _pixel_anchor_map(probe_view, NEAREST_FRAME, 10, 5, ANCHOR_DEPTH_M)
    far_anchor = _pixel_anchor_map(probe_view, NEAREST_FRAME, 10, 18, ANCHOR_DEPTH_M)
    near_occluder = _pixel_anchor_map(probe_view, NEAREST_FRAME, 10 + inside_offset, 5, OCCLUDER_DEPTH_M)
    far_occluder = _pixel_anchor_map(probe_view, NEAREST_FRAME, 10 + between_offset, 18, OCCLUDER_DEPTH_M)
    view, images_rgb = _build_painting_fixture(tmp_path / "real", {NEAREST_FRAME: [near_occluder, far_occluder]})
    result = _paint(view, [near_anchor, far_anchor], cell_rad=WIDE_CELL_RAD, frames_per_camera=1)
    rgb, painted, _ = result
    assert painted.tolist() == [False, True]
    np.testing.assert_array_equal(rgb[1], images_rgb[NEAREST_FRAME][18, 10])
    expected_radius = max(intrinsics["fx"] * np.tan(WIDE_CELL_RAD / 2), intrinsics["fy"] * np.tan(WIDE_CELL_RAD / 2))
    assert result.footprint_radius_pixels == pytest.approx(expected_radius)


def test_build_records_the_painters_radius_and_passes_k(tmp_path, monkeypatch):
    """build.py records the painter's returned radius and forwards the configured frame count."""
    from viking_splatad.radar_init import build as build_module

    recorded = {}
    real_painter = build_module.paint_anchors_from_train_cameras

    def spying_painter(*arguments, **keywords):
        """Delegate to the real painter, remembering its keywords and returned radius."""
        recorded["keywords"] = keywords
        result = real_painter(*arguments, **keywords)
        recorded["radius"] = result.footprint_radius_pixels
        return result

    monkeypatch.setattr(build_module, "paint_anchors_from_train_cameras", spying_painter)
    root, settings, config, calibration, options, _ = painted_fixture(tmp_path)
    options = dict(options, paint_frames_per_camera=3)
    metadata = build_module.build_radar_anchor_artifact(
        root, config, settings, options, tmp_path / "out", calibration, paint_seed_colours=True)
    assert recorded["keywords"]["paint_frames_per_camera"] == 3
    assert metadata["painting"]["footprint_radius_pixels"] == pytest.approx(recorded["radius"])
    # The recorded radius is also f*tan(cell/2) from the manifest intrinsics and the configured cells.
    intrinsics = open_dataset_view(root).manifest["cameras"][CAMERA_NAME]
    expected_radius = max(intrinsics["fx"] * np.tan(np.deg2rad(options["azimuth_cell_degrees"]) / 2),
                          intrinsics["fy"] * np.tan(np.deg2rad(options["elevation_cell_degrees"]) / 2))
    assert metadata["painting"]["footprint_radius_pixels"] == pytest.approx(expected_radius)


def test_zero_persistent_anchors_raise_a_clear_error(tmp_path, monkeypatch):
    """No anchors after the persistence filter stops the build with a named error."""
    from viking_splatad.radar_init import build as build_module

    monkeypatch.setattr(build_module, "persistent_anchors", lambda *arguments: {"means": np.empty((0, 3))})
    root, settings, config, calibration, options, _ = synthetic_artifact(tmp_path)
    with pytest.raises(ValueError, match="no persistent radar anchors"):
        build_module.build_loss_only_artifact(root, config, settings, options, tmp_path / "out", calibration)


# --- validate_painting_record ---------------------------------------------------------------------

CAMERA_A_STAMP_NS = 100
CAMERA_B_STAMP_NS = 200


def _two_camera_frames():
    """Two cameras, one train frame each, with distinct stamps."""
    return [
        SplitFrame(sensor="cam_a", kind="camera", split="train", frame_index=0,
                   timestamp_ns=CAMERA_A_STAMP_NS, time_s=0.0, file="a.jpg"),
        SplitFrame(sensor="cam_b", kind="camera", split="train", frame_index=0,
                   timestamp_ns=CAMERA_B_STAMP_NS, time_s=0.0, file="b.jpg"),
    ]


def _record(camera, stamps):
    """Painting record carrying only the keys the validator reads."""
    return {"camera": camera, "camera_stamps_ns": list(stamps)}


def test_stamp_of_another_cameras_train_frame_is_refused():
    """A train frame of cam_b cannot be the source of a painting recorded for cam_a."""
    frames = _two_camera_frames()
    rgb = np.zeros((1, 3), dtype=np.uint8)
    validate_painting_record(_record("cam_a", [CAMERA_A_STAMP_NS]), 1, frames, rgb)
    with pytest.raises(ValueError, match="train frames"):
        validate_painting_record(_record("cam_a", [CAMERA_B_STAMP_NS]), 1, frames, rgb)


def test_unpainted_stamp_must_hold_exactly_neutral_grey():
    """Rows with stamp -1 must be NEUTRAL_GREY_RGB; any other colour there is refused."""
    frames = _two_camera_frames()
    grey_rows = np.tile(NEUTRAL_GREY_RGB, (1, 1))
    validate_painting_record(_record("cam_a", [-1]), 1, frames, grey_rows)
    off_grey_rows = grey_rows.copy()
    off_grey_rows[0, 0] += 1
    with pytest.raises(ValueError, match="grey"):
        validate_painting_record(_record("cam_a", [-1]), 1, frames, off_grey_rows)
    # A painted row may hold any colour, grey included.
    validate_painting_record(_record("cam_a", [CAMERA_A_STAMP_NS]), 1, frames, off_grey_rows)
