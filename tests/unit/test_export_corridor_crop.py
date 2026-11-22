"""Tests for the driven-path corridor crop of the SuperSplat export (T87) and its sweep generator."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.dataset.schema import SCHEMA_VERSION
from viking_splatad.dataset.writer import POSE_KEY_SEPARATOR
from viking_splatad.dataset.run_split import file_sha256
from viking_splatad.export.crop import (
    corridor_crop_mask,
    iter_train_sweeps_map,
    train_lidar_bounds_map,
    train_lidar_corridor_map,
    train_sensor_positions_map,
)

TEST_MIN_OPACITY = 0.005
SENSOR_NAME = "ouster"
EGO_RADIUS = 0.1
CSV_HEADER = ["sensor", "kind", "split", "frame_index", "timestamp_ns", "time_s", "file"]
# Corridor used by the mask tests.
PATH_XY = np.array([[0.0, 0.0], [10.0, 0.0]])
WIDTH_M, Z_LOWER, Z_UPPER = 5.0, -1.0, 20.0
# Distance slack for "just inside / just outside" points.
EPSILON = 1e-6


def build_dataset(tmp_path: Path, sweeps_sensor_frame, poses: np.ndarray):
    """Write a one-sensor dataset (every sweep a train row) and its run folder.

    Args:
        tmp_path: pytest temporary directory.
        sweeps_sensor_frame: Per sweep, a list of sensor-frame (x, y, z) points.
        poses: (N, 4, 4) T_world_sensor, one per sweep.

    Returns:
        (dataset_dir, run_dir).
    """
    dataset_dir, run_dir = tmp_path / "dataset", tmp_path / "run"
    lidar_dir = dataset_dir / "lidar" / SENSOR_NAME
    lidar_dir.mkdir(parents=True)
    run_dir.mkdir()
    for sweep_index, points in enumerate(sweeps_sensor_frame):
        # 5-wide layout: x, y, z, intensity, time.
        columns = np.hstack([np.asarray(points, dtype=np.float32), np.ones((len(points), 1), np.float32),
                             np.zeros((len(points), 1), np.float32)])
        np.save(lidar_dir / f"{sweep_index:06d}_{sweep_index}.npy", columns)
    # Manifest and poses (time_s = sweep index), then the split CSV and the split record hashing them.
    manifest = {"schema_version": SCHEMA_VERSION, "sensors": [{"name": SENSOR_NAME, "kind": "lidar"}],
                "lidar": {"min_lidar_dist": EGO_RADIUS}}
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(dataset_dir / "poses.npz",
             **{f"{SENSOR_NAME}{POSE_KEY_SEPARATOR}times": np.arange(len(poses), dtype=np.float64),
                f"{SENSOR_NAME}{POSE_KEY_SEPARATOR}poses_world_sensor": poses})
    with open(run_dir / "split_frames.csv", "w", newline="") as split_file:
        writer = csv.writer(split_file)
        writer.writerow(CSV_HEADER)
        for sweep_index in range(len(poses)):
            writer.writerow([SENSOR_NAME, "lidar", "train", sweep_index, sweep_index, float(sweep_index),
                             f"lidar/{SENSOR_NAME}/{sweep_index:06d}_{sweep_index}.npy"])
    record = {"dataset": {"manifest_sha256": file_sha256(dataset_dir / "manifest.json"),
                          "poses_sha256": file_sha256(dataset_dir / "poses.npz")}}
    (run_dir / "split.json").write_text(json.dumps(record))
    return dataset_dir, run_dir


def translation_poses(origins: np.ndarray) -> np.ndarray:
    """Identity-rotation poses T_world_sensor with the given translations.

    Args:
        origins: (N, 3) map-frame sensor origins.

    Returns:
        (N, 4, 4) float64 poses.
    """
    poses = np.tile(np.eye(4), (len(origins), 1, 1))
    poses[:, :3, 3] = origins
    return poses


@pytest.fixture
def two_sweep_dataset(tmp_path: Path):
    """Sweeps at origins (0,0,0) and (100,0,0); sensor-frame points chosen so ranges are exact integers.

    Sweep 0 points: (3,4,2) range 5, (6,8,30) range 10 (z=30), (0,1,-3) range 1.
    Sweep 1 points: (2,0,7) range 2, (0,20,50) range 20 (z=50, far from the path in map frame).

    Returns:
        (dataset_dir, run_dir, sweeps (sensor-frame points per sweep), origins (2, 3)).
    """
    sweeps = [[(3.0, 4.0, 2.0), (6.0, 8.0, 30.0), (0.0, 1.0, -3.0)],
              [(2.0, 0.0, 7.0), (0.0, 20.0, 50.0)]]
    origins = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    dataset_dir, run_dir = build_dataset(tmp_path, sweeps, translation_poses(origins))
    return dataset_dir, run_dir, sweeps, origins


@pytest.mark.covers("T87")
def test_corridor_mask_known_answer():
    """Path points (0,0), (10,0), width 5, z in [-1, 20]: hand-picked inside/outside points, inclusive boundaries."""
    # Distance is to the nearest path POINT (the sensor samples), not to the segment between them.
    # Each row: (point_x, point_y, point_z, opacity, expected keep, reason).
    cases = [
        (0.0, WIDTH_M, 0.0, 0.9, True, "5 m beside path point (0,0): boundary"),
        (0.0, WIDTH_M + EPSILON, 0.0, 0.9, False, "just outside beside a path point"),
        (0.0, WIDTH_M - EPSILON, 0.0, 0.9, True, "just inside beside a path point"),
        (5.0, 0.0, 0.0, 0.9, True, "midpoint of the two path points: 5 m from each, boundary"),
        (5.0, 1.0, 0.0, 0.9, False, "above the midpoint: sqrt(25+1) > 5 (point distance, not segment distance)"),
        (13.0, 4.0, 0.0, 0.9, True, "beyond the end: Euclid to (10,0) = sqrt(9+16) = 5, boundary"),
        (13.0, 4.0 + EPSILON, 0.0, 0.9, False, "beyond the end, just outside"),
        (-3.0, -4.0, 0.0, 0.9, True, "before the start: Euclid to (0,0) = 5, boundary"),
        (10.0, 0.0, Z_UPPER, 0.9, True, "z at the upper bound"),
        (10.0, 0.0, Z_UPPER + EPSILON, 0.9, False, "z just above"),
        (10.0, 0.0, Z_LOWER, 0.9, True, "z at the lower bound"),
        (10.0, 0.0, Z_LOWER - EPSILON, 0.9, False, "z just below"),
        (10.0, 0.0, 0.0, TEST_MIN_OPACITY, True, "opacity exactly at the floor"),
        (10.0, 0.0, 0.0, TEST_MIN_OPACITY - EPSILON, False, "opacity just below the floor"),
        (1e7, 1e7, 1e7, 0.9, False, "init junk"),
    ]
    means = np.array([[point_x, point_y, point_z] for point_x, point_y, point_z, *_ in cases])
    opacities = np.array([opacity for _x, _y, _z, opacity, *_ in cases])
    # Independent expectation: explicit minimum over the Euclidean distances to each path point.
    expected = []
    for point_x, point_y, point_z, opacity, _keep, _reason in cases:
        distance = min(float(np.hypot(point_x - path_x, point_y - path_y)) for path_x, path_y in PATH_XY)
        expected.append(distance <= WIDTH_M and Z_LOWER <= point_z <= Z_UPPER and opacity >= TEST_MIN_OPACITY)
    assert expected == [case[4] for case in cases]
    mask = corridor_crop_mask(means, opacities, PATH_XY, WIDTH_M, Z_LOWER, Z_UPPER, TEST_MIN_OPACITY)
    assert mask.tolist() == expected


@pytest.mark.covers("T87")
def test_corridor_mask_rejects_bad_min_opacity():
    """A NaN, infinite, negative or above-one opacity floor raises instead of silently keeping nothing or everything."""
    means, opacities = np.zeros((2, 3)), np.full(2, 0.5)
    for bad_floor in (np.nan, np.inf, -0.1, 1.5):
        with pytest.raises(ValueError, match="min_opacity"):
            corridor_crop_mask(means, opacities, PATH_XY, WIDTH_M, Z_LOWER, Z_UPPER, bad_floor)


@pytest.mark.covers("T87")
def test_wrong_dataset_raises_identity_error_before_any_loader_error(two_sweep_dataset):
    """A dataset whose manifest is not the run's raises the identity ValueError, even if the manifest is unloadable."""
    dataset_dir, run_dir, _sweeps, _origins = two_sweep_dataset
    # An unparseable manifest would make the loader fail first if the identity check ran after opening.
    (dataset_dir / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="not the dataset the run trained on"):
        train_lidar_corridor_map(run_dir, dataset_dir)
    with pytest.raises(ValueError, match="not the dataset the run trained on"):
        train_lidar_bounds_map(run_dir, dataset_dir)


@pytest.mark.covers("T87")
def test_corridor_mask_rejects_bad_inputs():
    """Bad shapes, NaN, logits, width <= 0, swapped z and an empty path all raise."""
    means, opacities = np.zeros((2, 3)), np.full(2, 0.5)

    def call(**overrides):
        """Call corridor_crop_mask with valid defaults replaced by ``overrides``."""
        arguments = dict(means=means, opacities=opacities, path_xy=PATH_XY, width_m=WIDTH_M,
                         z_lower=Z_LOWER, z_upper=Z_UPPER, min_opacity=TEST_MIN_OPACITY)
        arguments.update(overrides)
        return corridor_crop_mask(**arguments)

    assert call().shape == (2,)
    for overrides in (
        dict(means=np.zeros((2, 2))),
        dict(opacities=np.full((2, 1), 0.5)),
        dict(opacities=np.array([-3.0, 0.5])),
        dict(opacities=np.array([0.5, 4.0])),
        dict(means=np.full((2, 3), np.nan)),
        dict(path_xy=np.array([[0.0, np.nan]])),
        dict(width_m=0.0),
        dict(width_m=-1.0),
        dict(z_lower=Z_UPPER + 1.0),
        dict(path_xy=np.zeros((0, 2))),
        dict(path_xy=np.zeros((3, 3))),
    ):
        with pytest.raises(ValueError):
            call(**overrides)


@pytest.mark.covers("T87")
def test_sensor_positions_are_pose_translations(two_sweep_dataset):
    """Origins are the pose translations, in sweep order."""
    dataset_dir, run_dir, _sweeps, origins = two_sweep_dataset
    np.testing.assert_allclose(train_sensor_positions_map(run_dir, dataset_dir), origins)


@pytest.mark.covers("T87")
def test_sensor_positions_validate_dataset(two_sweep_dataset):
    """A dataset other than the run's is refused (shared validation)."""
    dataset_dir, run_dir, _sweeps, origins = two_sweep_dataset
    np.savez(dataset_dir / "poses.npz",
             **{f"{SENSOR_NAME}{POSE_KEY_SEPARATOR}times": np.arange(2, dtype=np.float64),
                f"{SENSOR_NAME}{POSE_KEY_SEPARATOR}poses_world_sensor": translation_poses(origins + 1.0)})
    with pytest.raises(ValueError, match="not the dataset"):
        train_sensor_positions_map(run_dir, dataset_dir)


@pytest.mark.covers("T87")
def test_corridor_from_dataset_known_answer(two_sweep_dataset):
    """Width is the percentile of own-origin ranges; z extremes come only from returns inside the corridor.

    Ranges (horizontal, from own origin): 5, 10, 1, 2, 20. Map points: sweep 0 as given; sweep 1 shifted by
    x = 100. Path = (0,0), (100,0).
    """
    dataset_dir, run_dir, sweeps, origins = two_sweep_dataset
    range_percentile = 80.0
    # Independent expectation from first principles with plain numpy / loops.
    map_points = [np.array(points) + origin for points, origin in zip(sweeps, origins)]
    ranges = np.array([np.hypot(*(point[:2] - origin[:2])) for points, origin in zip(map_points, origins)
                       for point in points])
    np.testing.assert_allclose(sorted(ranges), [1, 2, 5, 10, 20])
    expected_width = float(np.percentile(ranges, range_percentile))
    inside_z = [point[2] for points in map_points for point in points
                if min(np.hypot(*(point[:2] - origin[:2])) for origin in origins) <= expected_width]
    corridor = train_lidar_corridor_map(run_dir, dataset_dir, width_sweep_stride=1, range_percentile=range_percentile,
                                        z_tail_percent=0.0)
    np.testing.assert_allclose(corridor.path_xy, origins[:, :2])
    assert corridor.width_m == pytest.approx(expected_width)
    assert corridor.z_lower == pytest.approx(min(inside_z))
    assert corridor.z_upper == pytest.approx(max(inside_z))
    # The (0,20,50) return is 20 m from its path point, beyond the width of 12 m: z = 50 must be excluded.
    assert corridor.z_upper == pytest.approx(30.0)


@pytest.mark.covers("T87")
def test_corridor_width_stride_subsamples_pass_one_only(two_sweep_dataset):
    """Stride 2 reads only sweep 0 for the width (ranges 5, 10, 1) but every sweep for z."""
    dataset_dir, run_dir, _sweeps, _origins = two_sweep_dataset
    corridor = train_lidar_corridor_map(run_dir, dataset_dir, width_sweep_stride=2, range_percentile=100.0,
                                        z_tail_percent=0.0)
    assert corridor.width_m == pytest.approx(10.0)
    # Width 10: sweep 1 return (2,0,7) at x=102 is 2 m from path point (100,0) -> z 7 inside; (0,20,50) is 20 away.
    assert corridor.z_lower == pytest.approx(-3.0)
    assert corridor.z_upper == pytest.approx(30.0)


@pytest.mark.covers("T87")
def test_corridor_raises_on_bad_percentile(two_sweep_dataset):
    """A percentile outside [0, 100] is refused."""
    dataset_dir, run_dir, _sweeps, _origins = two_sweep_dataset
    with pytest.raises(ValueError):
        train_lidar_corridor_map(run_dir, dataset_dir, range_percentile=101.0)


@pytest.mark.covers("T87")
@pytest.mark.covers("T86")
def test_bounds_equal_extremes_of_generator(two_sweep_dataset):
    """Refactor check: T86's box equals min/max over the points the generator yields; origins match poses."""
    dataset_dir, run_dir, sweeps, origins = two_sweep_dataset
    yielded = list(iter_train_sweeps_map(run_dir, dataset_dir, 1))
    assert len(yielded) == len(sweeps)
    for (origin_map, points_map), expected_origin, points in zip(yielded, origins, sweeps):
        np.testing.assert_allclose(origin_map, expected_origin)
        np.testing.assert_allclose(points_map, np.array(points) + expected_origin)
    all_points = np.vstack([points_map for _origin, points_map in yielded])
    lower, upper = train_lidar_bounds_map(run_dir, dataset_dir)
    np.testing.assert_allclose(lower, all_points.min(axis=0))
    np.testing.assert_allclose(upper, all_points.max(axis=0))


@pytest.mark.covers("T87")
def test_generator_rejects_bad_stride(two_sweep_dataset):
    """Stride 0 raises as soon as the generator is consumed."""
    dataset_dir, run_dir, _sweeps, _origins = two_sweep_dataset
    with pytest.raises(ValueError, match="sweep_stride"):
        list(iter_train_sweeps_map(run_dir, dataset_dir, 0))


@pytest.mark.covers("T87")
def test_corridor_exposes_and_reuses_the_sensor_positions(two_sweep_dataset, monkeypatch):
    """path_positions_map holds the (P, 3) origins; passing them in gives the same corridor with no pose read."""
    dataset_dir, run_dir, _sweeps, origins = two_sweep_dataset
    fresh = train_lidar_corridor_map(run_dir, dataset_dir)
    np.testing.assert_allclose(fresh.path_positions_map, origins)
    np.testing.assert_allclose(fresh.path_xy, fresh.path_positions_map[:, :2])

    def forbidden(*_arguments, **_keyword_arguments):
        """The positions were supplied, so the poses must not be read again."""
        raise AssertionError("train_sensor_positions_map called although positions were passed")

    monkeypatch.setattr("viking_splatad.export.crop.train_sensor_positions_map", forbidden)
    reused = train_lidar_corridor_map(run_dir, dataset_dir, sensor_positions_map=fresh.path_positions_map)
    np.testing.assert_array_equal(reused.path_xy, fresh.path_xy)
    assert (reused.width_m, reused.z_lower, reused.z_upper) == (fresh.width_m, fresh.z_lower, fresh.z_upper)


@pytest.mark.covers("T87")
def test_corridor_z_tail_excludes_a_stray_return(tmp_path: Path):
    """One far-off z among many returns is cut by a nonzero tail; the bounds equal np.percentile of the z values."""
    # One sweep at the origin: 1000 returns 1 m from the sensor with z = 0..999 / 100, plus one stray at z = 500.
    ground_z_values = np.arange(1000, dtype=np.float64) / 100.0
    stray_z = 500.0
    points = [(1.0, 0.0, z_value) for z_value in ground_z_values] + [(1.0, 0.0, stray_z)]
    dataset_dir, run_dir = build_dataset(tmp_path, [points], translation_poses(np.zeros((1, 3))))
    z_tail_percent = 1.0
    all_z = np.append(ground_z_values, stray_z)
    expected_lower = float(np.percentile(all_z, z_tail_percent))
    expected_upper = float(np.percentile(all_z, 100.0 - z_tail_percent))
    corridor = train_lidar_corridor_map(run_dir, dataset_dir, width_sweep_stride=1, range_percentile=100.0,
                                        z_tail_percent=z_tail_percent)
    # Float32 storage of z: agree to float32 resolution.
    assert corridor.z_lower == pytest.approx(expected_lower, abs=1e-4)
    assert corridor.z_upper == pytest.approx(expected_upper, abs=1e-4)
    # The stray is excluded by the tail, while tail 0 keeps it (old min/max behaviour).
    assert corridor.z_upper < stray_z
    raw = train_lidar_corridor_map(run_dir, dataset_dir, width_sweep_stride=1, range_percentile=100.0,
                                   z_tail_percent=0.0)
    assert (raw.z_lower, raw.z_upper) == (float(all_z.min()), float(all_z.max()))


@pytest.mark.covers("T87")
def test_corridor_raises_on_bad_z_tail(two_sweep_dataset):
    """A z tail below 0, at 50 or above, or NaN is refused."""
    dataset_dir, run_dir, _sweeps, _origins = two_sweep_dataset
    for bad_tail in (-0.1, 50.0, 75.0, np.nan):
        with pytest.raises(ValueError, match="z_tail_percent"):
            train_lidar_corridor_map(run_dir, dataset_dir, z_tail_percent=bad_tail)


@pytest.mark.covers("T87")
def test_corridor_rejects_bad_supplied_sensor_positions(two_sweep_dataset):
    """A caller-supplied path that is empty, not (P, 3) or non-finite is refused."""
    dataset_dir, run_dir, _sweeps, _origins = two_sweep_dataset
    bad_positions = (np.zeros((0, 3)), np.zeros((2, 2)), np.zeros(3), np.array([[0.0, 0.0, np.nan]]),
                     np.array([[np.inf, 0.0, 0.0]]))
    for positions in bad_positions:
        with pytest.raises(ValueError, match="sensor_positions_map"):
            train_lidar_corridor_map(run_dir, dataset_dir, sensor_positions_map=positions)
