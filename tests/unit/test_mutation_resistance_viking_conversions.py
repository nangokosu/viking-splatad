"""Mutation-resistance tests for splatad/viking_conversions.py (T27).

The existing tests use one camera and one lidar, so per-sensor intrinsics, sensor indices and the
image-path/pose row alignment are never distinguished from a single-sensor mix-up. These tests build
an in-memory ``DatasetView`` with two cameras and two lidars of different sizes, intrinsics and
frame counts, and assert every output array against values written down independently.
"""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.contract.splatad_contract import AZIMUTH_INTEGER_RTOL
from viking_splatad.splatad.viking_conversions import (
    DatasetView,
    build_sensor_arrays,
    load_lidar_sweep,
    neurad_lidar_model_from_manifest,
)

# Arbitrary fixed seed: only reproducibility of the random poses.
RANDOM_SEED = 90210
# Frame counts differ per sensor so a concatenation-order or per-sensor-length mix-up shows up.
FRAME_COUNTS = {"cam_left": 3, "cam_right": 2, "lidar_front": 4, "lidar_rear": 2}
# Distinct intrinsics per camera (non-square pixels, off-centre principal points, different sizes).
CAMERA_INTRINSICS = {
    "cam_left": dict(fx=1000.5, fy=1010.25, cx=640.5, cy=480.25, width=1280, height=960),
    "cam_right": dict(fx=720.75, fy=715.5, cx=300.25, cy=250.5, width=640, height=512),
}
# Readouts differ so "first camera defines the rolling shutter" is observable.
ROLLING_SHUTTER_SECONDS = {"cam_left": 0.03, "cam_right": 0.05}


def random_poses(rng, frame_count, translation_offset):
    """Random rigid poses [n,4,4]; frame k has translation x = translation_offset + k (a row tag)."""
    poses = np.tile(np.eye(4), (frame_count, 1, 1))
    for frame_index in range(frame_count):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        poses[frame_index, :3, :3] = Rotation.from_rotvec(axis * rng.uniform(0.3, 3.0)).as_matrix()
        poses[frame_index, :3, 3] = [translation_offset + frame_index, rng.uniform(-9, 9), rng.uniform(-9, 9)]
    return poses


def build_view(rng):
    """In-memory two-camera two-lidar view. Row tag: pose translation x = 1000*sensor_position + frame."""
    sensor_names = list(FRAME_COUNTS)
    view_poses, view_times, view_paths = {}, {}, {}
    for sensor_position, name in enumerate(sensor_names):
        count = FRAME_COUNTS[name]
        view_poses[name] = random_poses(rng, count, 1000.0 * sensor_position)
        view_times[name] = 50.0 * sensor_position + np.arange(count) * 0.1
        view_paths[name] = [Path(f"/dataset/{name}/frame_{frame:06d}.bin") for frame in range(count)]
    manifest = {
        "cameras": CAMERA_INTRINSICS,
        "rolling_shutter": {name: {"readout_seconds": value} for name, value in ROLLING_SHUTTER_SECONDS.items()},
        "lidar": {
            "lidar_type": "OUSTER_OS0_32",
            "elevation_mapping": {"0": 0.1, "1": -0.1},
            "azimuth_resolution": 2.0 * math.pi / 1024,
            "min_lidar_dist": 0.8,
            "sweep_period": 0.1,
        },
    }
    return DatasetView(Path("/dataset"), manifest, view_times, view_poses, view_paths)


@pytest.mark.covers("T27")
def test_build_sensor_arrays_two_cameras_two_lidars_every_field():
    """Intrinsics, sizes, sensor indices, times and image/sweep paths line up row by row."""
    rng = np.random.default_rng(RANDOM_SEED)
    view = build_view(rng)
    time_fraction = 0.25  # not the 0.5 or 0 used elsewhere: a hard-coded fraction would fail
    cameras, lidars = build_sensor_arrays(view, ("cam_left", "cam_right"), ("lidar_front", "lidar_rear"), time_fraction)
    # Per-row expectations, built camera by camera in configuration order.
    expected = {key: [] for key in ("fx", "fy", "cx", "cy", "width", "height", "index", "times", "paths", "c2w")}
    for camera_index, name in enumerate(("cam_left", "cam_right")):
        for frame in range(FRAME_COUNTS[name]):
            for key in ("fx", "fy", "cx", "cy", "width", "height"):
                expected[key].append(CAMERA_INTRINSICS[name][key])
            expected["index"].append(camera_index)
            expected["times"].append(view.sensor_times[name][frame])
            expected["paths"].append(view.sensor_frame_paths[name][frame])
            pose = view.sensor_poses[name][frame]
            # nerfstudio axes: x column kept, y and z columns negated; translation untouched.
            rotation = pose[:3, :3] * np.array([1.0, -1.0, -1.0])[None, :]
            expected["c2w"].append(np.hstack([rotation, pose[:3, 3:4]]))
    np.testing.assert_allclose(cameras.focal_x, expected["fx"], rtol=1e-6)
    np.testing.assert_allclose(cameras.focal_y, expected["fy"], rtol=1e-6)
    np.testing.assert_allclose(cameras.principal_x, expected["cx"], rtol=1e-6)
    np.testing.assert_allclose(cameras.principal_y, expected["cy"], rtol=1e-6)
    np.testing.assert_array_equal(cameras.widths, expected["width"])
    np.testing.assert_array_equal(cameras.heights, expected["height"])
    np.testing.assert_array_equal(cameras.sensor_indices, np.array(expected["index"])[:, None])
    assert cameras.sensor_indices.shape == (sum(FRAME_COUNTS[n] for n in ("cam_left", "cam_right")), 1)
    np.testing.assert_allclose(cameras.times, expected["times"], atol=0)
    assert cameras.image_paths == expected["paths"]
    np.testing.assert_allclose(cameras.camera_to_worlds, np.stack(expected["c2w"]), atol=1e-4)
    # Image-path/pose alignment: the tag in each row's pose translation names the same frame as the path.
    for row_index, path in enumerate(cameras.image_paths):
        camera_position = 0 if "cam_left" in str(path) else 1
        frame = int(path.stem.split("_")[-1])
        assert cameras.camera_to_worlds[row_index, 0, 3] == pytest.approx(1000.0 * camera_position + frame)
    # Rolling shutter: the FIRST camera defines it; centre time uses the caller's fraction.
    assert cameras.rolling_shutter_seconds == ROLLING_SHUTTER_SECONDS["cam_left"]
    assert cameras.time_to_center_pixel_seconds == pytest.approx(time_fraction * ROLLING_SHUTTER_SECONDS["cam_left"])

    # Lidars: concatenated in order, indices are positions inside the lidar tuple, poses unflipped.
    lidar_expected_index = [0] * FRAME_COUNTS["lidar_front"] + [1] * FRAME_COUNTS["lidar_rear"]
    np.testing.assert_array_equal(lidars.sensor_indices, np.array(lidar_expected_index)[:, None])
    assert lidars.sweep_paths == view.sensor_frame_paths["lidar_front"] + view.sensor_frame_paths["lidar_rear"]
    np.testing.assert_allclose(
        lidars.times, np.concatenate([view.sensor_times["lidar_front"], view.sensor_times["lidar_rear"]]), atol=0
    )
    np.testing.assert_allclose(
        lidars.lidar_to_worlds,
        np.concatenate([view.sensor_poses["lidar_front"][:, :3], view.sensor_poses["lidar_rear"][:, :3]]),
        atol=1e-4,
    )


@pytest.mark.covers("T27")
def test_build_sensor_arrays_camera_order_follows_the_requested_tuple():
    """Swapping the camera tuple swaps the blocks and the per-camera intrinsics, indices are positions."""
    view = build_view(np.random.default_rng(RANDOM_SEED + 1))
    cameras, _ = build_sensor_arrays(view, ("cam_right", "cam_left"), (), 0.0)
    right_count = FRAME_COUNTS["cam_right"]
    assert list(cameras.sensor_indices[:, 0]) == [0] * right_count + [1] * FRAME_COUNTS["cam_left"]
    assert cameras.focal_x[0] == np.float32(CAMERA_INTRINSICS["cam_right"]["fx"])
    assert cameras.focal_x[-1] == np.float32(CAMERA_INTRINSICS["cam_left"]["fx"])
    # The first camera in the tuple (cam_right) defines the rolling shutter now.
    assert cameras.rolling_shutter_seconds == ROLLING_SHUTTER_SECONDS["cam_right"]
    assert cameras.time_to_center_pixel_seconds == 0.0


# --------------------------------------------------------------------------- load_lidar_sweep


@pytest.mark.covers("T27")
def test_load_lidar_sweep_box_is_any_axis_and_nan_intensity_drops_the_row(tmp_path):
    """Keep iff outside the box on ANY axis (|p_i| >= half-extent); any non-finite column drops the row."""
    box = (1.0, 2.0, 3.0)  # asymmetric: swapped axes give different answers
    rows = np.array([
        [1.5, 0.0, 0.0, 0.5, 0.0, 1.0],     # outside on x only -> keep (AND rule would drop)
        [0.0, 2.5, 0.0, 0.5, 0.0, 2.0],     # outside on y only -> keep
        [0.0, 0.0, -3.5, 0.5, 0.0, 3.0],    # outside on z only, negative side -> keep (abs)
        [0.9, 1.9, 2.9, 0.5, 0.0, 4.0],     # inside on every axis -> drop
        [1.5, 0.0, 0.0, np.nan, 0.0, 5.0],  # NaN intensity, finite xyz -> drop
        [1.5, 0.0, 0.0, 0.5, np.inf, 6.0],  # inf time -> drop
        [1.0, 0.0, 0.0, 0.5, 0.0, 7.0],     # exactly on the x half-extent -> keep (>=)
        [2.0, 0.0, 0.0, 0.5, 0.0, 8.0],     # keep
    ], dtype=np.float32)
    path = tmp_path / "sweep.npy"
    np.save(path, rows)
    kept = load_lidar_sweep(path, min_lidar_radius=0.0, ego_box_half_extents=box)
    assert list(kept[:, 5]) == [1.0, 2.0, 3.0, 7.0, 8.0]  # channel column doubles as a row id
    # Sphere test is >=: radius exactly 5 keeps the (3,4,0) point, 5.0001 drops it.
    sphere_rows = np.array([[3.0, 4.0, 0.0, 0.5, 0.0, 1.0]], dtype=np.float32)
    np.save(path, sphere_rows)
    assert load_lidar_sweep(path, min_lidar_radius=5.0).shape[0] == 1
    assert load_lidar_sweep(path, min_lidar_radius=5.0001).shape[0] == 0


# --------------------------------------------------------------------------- azimuth snap


def lidar_manifest_block(azimuth_resolution_radians, min_lidar_dist=0.8):
    """Minimal manifest lidar block for ``neurad_lidar_model_from_manifest``."""
    return {
        "lidar_type": "OUSTER_OS0_32", "elevation_mapping": {"0": 0.2, "1": -0.2},
        "azimuth_resolution": azimuth_resolution_radians, "min_lidar_dist": min_lidar_dist, "sweep_period": 0.1,
    }


@pytest.mark.covers("T27")
def test_azimuth_snap_rebuilds_exact_degrees_where_the_unsnapped_value_differs():
    """A resolution a hair off 2*pi/1024 snaps to exactly 360/1024; unsnapped it gives tile width 33."""
    columns_per_turn = 1024
    tile_columns = 32  # azimuth columns per tile (datamanager formula, as in the existing test)
    # Relative perturbation 100x below the snap tolerance: inside the window, so a snap is required.
    perturbation = AZIMUTH_INTEGER_RTOL / 100.0
    resolution_radians = 2.0 * math.pi / columns_per_turn * (1.0 - perturbation)
    unsnapped_degrees = float(np.rad2deg(resolution_radians))
    assert unsnapped_degrees != 360.0 / columns_per_turn  # precondition: the snap changes the value
    assert math.ceil(360.0 / (unsnapped_degrees * tile_columns)) == columns_per_turn // tile_columns + 1
    model = neurad_lidar_model_from_manifest(lidar_manifest_block(resolution_radians))
    assert model.azimuth_resolution_degrees == 360.0 / columns_per_turn
    assert math.ceil(360.0 / (model.azimuth_resolution_degrees * tile_columns)) == columns_per_turn // tile_columns

    # A non-integer column count must NOT snap: 1000.5 columns -> plain radian->degree conversion.
    odd_resolution = 2.0 * math.pi / 1000.5
    model_odd = neurad_lidar_model_from_manifest(lidar_manifest_block(odd_resolution))
    assert model_odd.azimuth_resolution_degrees == pytest.approx(360.0 / 1000.5, rel=1e-12)
    # A column count just outside the tolerance must not snap either (relative offset 100x the tolerance).
    outside = 2.0 * math.pi / columns_per_turn * (1.0 - AZIMUTH_INTEGER_RTOL * 100.0)
    model_outside = neurad_lidar_model_from_manifest(lidar_manifest_block(outside))
    assert model_outside.azimuth_resolution_degrees == pytest.approx(float(np.rad2deg(outside)), rel=1e-15)
    assert model_outside.azimuth_resolution_degrees != 360.0 / columns_per_turn


@pytest.mark.covers("T27")
def test_lidar_model_ego_filter_scalar_vs_box_and_elevation_degrees():
    """Scalar min_lidar_dist is the sphere radius; a 3-vector is the box; elevations go radians->degrees."""
    scalar_model = neurad_lidar_model_from_manifest(lidar_manifest_block(2 * math.pi / 1024, 0.8))
    assert scalar_model.min_lidar_radius == 0.8 and scalar_model.ego_box_half_extents == (0.0, 0.0, 0.0)
    box_model = neurad_lidar_model_from_manifest(lidar_manifest_block(2 * math.pi / 1024, [0.5, 1.5, 2.5]))
    assert box_model.min_lidar_radius == 0.0 and box_model.ego_box_half_extents == (0.5, 1.5, 2.5)
    assert box_model.elevation_mapping_degrees == {0: pytest.approx(math.degrees(0.2)), 1: pytest.approx(math.degrees(-0.2))}
