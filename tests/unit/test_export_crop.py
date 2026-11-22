"""Tests for the observed-scene crop of the SuperSplat export (T86)."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.dataset.schema import SCHEMA_VERSION
from viking_splatad.dataset.writer import POSE_KEY_SEPARATOR
from viking_splatad.dataset.run_split import file_sha256
from viking_splatad.export.crop import crop_mask, observed_bounds_from_lidar, train_lidar_bounds_map

# The opacity floor the tests pass in (the module never owns one).
TEST_MIN_OPACITY = 0.005
# Sensor-frame point of every synthetic sweep, and the intensity/time columns of the 5-wide layout.
SYNTHETIC_POINT = (1.0, 0.0, 0.0)
SYNTHETIC_INTENSITY, SYNTHETIC_POINT_TIME = 1.0, 0.0
# Ego radius below the synthetic point's range (1 m).
SYNTHETIC_EGO_RADIUS = 0.1
SYNTHETIC_SENSOR = "ouster"
CSV_HEADER = ["sensor", "kind", "split", "frame_index", "timestamp_ns", "time_s", "file"]


def write_split_record(run_dir: Path, dataset_dir: Path) -> None:
    """Write run_dir/split.json holding the real hashes of the dataset's manifest and poses."""
    record = {"dataset": {"manifest_sha256": file_sha256(dataset_dir / "manifest.json"),
                          "poses_sha256": file_sha256(dataset_dir / "poses.npz")}}
    (run_dir / "split.json").write_text(json.dumps(record))


def build_synthetic_run(tmp_path: Path, poses: np.ndarray, file_indices=None, csv_rows=None,
                        extra_sensors=()):
    """Dataset with one lidar sensor (one point per sweep) plus a run folder with CSV and split record.

    Args:
        tmp_path: Scratch root.
        poses: (N, 4, 4) T_world_sensor, one per pose row.
        file_indices: Frame numbers used in the file names (default 0..N-1); a gap simulates a hole.
        csv_rows: Rows after the header (default: every file as a train row with matching stamps).
        extra_sensors: Camera sensor names that get CSV rows only, to prove they are ignored.

    Returns:
        (dataset_dir, run_dir).
    """
    dataset_dir, run_dir = tmp_path / "dataset", tmp_path / "run"
    lidar_dir = dataset_dir / "lidar" / SYNTHETIC_SENSOR
    lidar_dir.mkdir(parents=True)
    run_dir.mkdir()
    file_indices = list(range(len(poses))) if file_indices is None else list(file_indices)
    for file_index in file_indices:
        np.save(lidar_dir / f"{file_index:06d}_{file_index}.npy",
                np.array([[*SYNTHETIC_POINT, SYNTHETIC_INTENSITY, SYNTHETIC_POINT_TIME]], dtype=np.float32))
    manifest = {"schema_version": SCHEMA_VERSION,
                "sensors": [{"name": SYNTHETIC_SENSOR, "kind": "lidar"}],
                "lidar": {"min_lidar_dist": SYNTHETIC_EGO_RADIUS}}
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(dataset_dir / "poses.npz",
             **{f"{SYNTHETIC_SENSOR}{POSE_KEY_SEPARATOR}times": np.arange(len(poses), dtype=np.float64),
                f"{SYNTHETIC_SENSOR}{POSE_KEY_SEPARATOR}poses_world_sensor": poses})
    if csv_rows is None:
        csv_rows = [[SYNTHETIC_SENSOR, "lidar", "train", index, index, float(index),
                     f"lidar/{SYNTHETIC_SENSOR}/{index:06d}_{index}.npy"] for index in range(len(poses))]
    camera_rows = [[name, "camera", "train", 0, 0, 0.0, f"camera/{name}/000000_0.jpg"] for name in extra_sensors]
    with open(run_dir / "split_frames.csv", "w", newline="") as split_file:
        writer = csv.writer(split_file)
        writer.writerow(CSV_HEADER)
        writer.writerows(camera_rows + csv_rows)
    write_split_record(run_dir, dataset_dir)
    return dataset_dir, run_dir


@pytest.mark.covers("T86")
def test_bounds_are_per_axis_extremes():
    """Known cloud: per-axis min/max read off by hand."""
    cloud = np.array([[1.0, -2.0, 3.0], [-4.0, 5.0, 0.0], [2.0, 0.0, -6.0]])
    lower, upper = observed_bounds_from_lidar(cloud)
    assert lower.tolist() == [-4.0, -2.0, -6.0]
    assert upper.tolist() == [2.0, 5.0, 3.0]


@pytest.mark.covers("T86")
def test_bounds_reject_empty_and_nonfinite():
    """No points or a NaN must raise rather than give a bogus box."""
    with pytest.raises(ValueError):
        observed_bounds_from_lidar(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        observed_bounds_from_lidar(np.array([[0.0, np.nan, 0.0]]))


@pytest.mark.covers("T86")
def test_crop_mask_inside_outside_face_and_opacity():
    """Box [0,10]^3: inside, on a face (kept), just outside, far junk, and too transparent."""
    lower, upper = np.zeros(3), np.full(3, 10.0)
    epsilon = 1e-6
    means = np.array([
        [5.0, 5.0, 5.0],               # inside, opaque
        [10.0, 0.0, 5.0],              # on faces/edge: inclusive
        [10.0 + epsilon, 5.0, 5.0],    # just outside on x
        [5.0, -epsilon, 5.0],          # just outside on y
        [1e7, 1e7, 1e7],               # init junk
        [5.0, 5.0, 5.0],               # inside but transparent
        [5.0, 5.0, 5.0],               # inside, exactly at threshold: kept
    ])
    opacities = np.array([0.9, 0.9, 0.9, 0.9, 0.9, TEST_MIN_OPACITY - epsilon, TEST_MIN_OPACITY])
    mask = crop_mask(means, opacities, lower, upper, TEST_MIN_OPACITY)
    assert mask.tolist() == [True, True, False, False, False, False, True]


@pytest.mark.covers("T86")
def test_load_train_points_use_train_frames_stride_and_pose(tmp_path: Path):
    """Tiny dataset: 4 lidar sweeps, frames 0..2 train; each sweep translated by its own pose.

    Sweep k holds the sensor-frame point (1, 0, 0) and the pose translates by (10 k, 0, 0), so its
    map point is (1 + 10 k, 0, 0). Frame 3 is test and must not appear; stride 2 keeps frames 0, 2.
    """
    # Step 1: dataset layout (lidar/<sensor>/<idx>_<ts>.npy) and an empty run folder.
    dataset_dir, run_dir = tmp_path / "dataset", tmp_path / "run"
    lidar_dir = dataset_dir / "lidar" / "ouster"
    lidar_dir.mkdir(parents=True)
    run_dir.mkdir()
    sweep_count = 4
    # Step 2: pure-translation poses (identity rotation) and one point per sweep.
    poses = np.tile(np.eye(4), (sweep_count, 1, 1))
    for sweep_index in range(sweep_count):
        poses[sweep_index, 0, 3] = 10.0 * sweep_index
        # Columns x, y, z, intensity, t (5-wide sweep layout).
        np.save(lidar_dir / f"{sweep_index:06d}_{sweep_index}.npy", np.array([[1.0, 0.0, 0.0, 1.0, 0.0]], dtype=np.float32))
    # Step 3: minimal manifest and poses.npz that open_dataset_view accepts.
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sensors": [{"name": "ouster", "kind": "lidar"}],
        "lidar": {"min_lidar_dist": 0.1},  # scalar ego radius, below the point's range of 1
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    np.savez(
        dataset_dir / "poses.npz",
        **{f"ouster{POSE_KEY_SEPARATOR}times": np.arange(sweep_count, dtype=np.float64),
           f"ouster{POSE_KEY_SEPARATOR}poses_world_sensor": poses},
    )
    # Step 4: split_frames.csv with the last sweep as test, the rest train.
    with open(run_dir / "split_frames.csv", "w", newline="") as split_file:
        writer = csv.writer(split_file)
        writer.writerow(["sensor", "kind", "split", "frame_index", "timestamp_ns", "time_s", "file"])
        for sweep_index in range(sweep_count):
            split_name = "train" if sweep_index < sweep_count - 1 else "test"
            writer.writerow(["ouster", "lidar", split_name, sweep_index, sweep_index, float(sweep_index),
                             f"lidar/ouster/{sweep_index:06d}_{sweep_index}.npy"])
    write_split_record(run_dir, dataset_dir)
    # Step 5: stride 1 gives every train sweep, stride 2 every other one, stride 0 is refused.
    expected_all = np.array([[1.0 + 10.0 * k, 0.0, 0.0] for k in range(sweep_count - 1)])
    lower_all, upper_all = train_lidar_bounds_map(run_dir, dataset_dir, sweep_stride=1)
    np.testing.assert_allclose(lower_all, expected_all.min(axis=0))
    np.testing.assert_allclose(upper_all, expected_all.max(axis=0))
    lower_strided, upper_strided = train_lidar_bounds_map(run_dir, dataset_dir, sweep_stride=2)
    np.testing.assert_allclose(lower_strided, expected_all[::2].min(axis=0))
    np.testing.assert_allclose(upper_strided, expected_all[::2].max(axis=0))
    with pytest.raises(ValueError):
        train_lidar_bounds_map(run_dir, dataset_dir, sweep_stride=0)


@pytest.mark.covers("T86")
def test_rotated_pose_maps_points(tmp_path: Path):
    """Pose = 90 deg about z plus translation (5, 7, 9): sensor point (1,0,0) -> R p + t = (5, 8, 9)."""
    quarter_turn = np.pi / 2.0
    pose = np.eye(4)
    pose[:3, :3] = [[np.cos(quarter_turn), -np.sin(quarter_turn), 0.0],
                    [np.sin(quarter_turn), np.cos(quarter_turn), 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = [5.0, 7.0, 9.0]
    dataset_dir, run_dir = build_synthetic_run(tmp_path, pose[None])
    lower, upper = train_lidar_bounds_map(run_dir, dataset_dir)
    np.testing.assert_allclose(lower, [5.0, 8.0, 9.0], atol=1e-9)
    np.testing.assert_allclose(upper, [5.0, 8.0, 9.0], atol=1e-9)


@pytest.mark.covers("T86")
def test_camera_rows_are_ignored(tmp_path: Path):
    """Camera rows (even for a sensor absent from the dataset) must not be read."""
    dataset_dir, run_dir = build_synthetic_run(tmp_path, np.tile(np.eye(4), (2, 1, 1)), extra_sensors=("cam_x",))
    lower, upper = train_lidar_bounds_map(run_dir, dataset_dir)
    np.testing.assert_allclose(lower, SYNTHETIC_POINT)
    np.testing.assert_allclose(upper, SYNTHETIC_POINT)


@pytest.mark.covers("T86")
def test_gapped_numbering_and_pose_count_mismatch_raise(tmp_path: Path):
    """Files numbered 0,1,3 (gap) and a pose count different from the file count are both refused."""
    gap_root, count_root = tmp_path / "gap", tmp_path / "count"
    gap_root.mkdir()
    count_root.mkdir()
    dataset_dir, run_dir = build_synthetic_run(gap_root, np.tile(np.eye(4), (3, 1, 1)), file_indices=[0, 1, 3])
    with pytest.raises(ValueError, match="gaps"):
        train_lidar_bounds_map(run_dir, dataset_dir)
    dataset_dir, run_dir = build_synthetic_run(count_root, np.tile(np.eye(4), (3, 1, 1)), file_indices=[0, 1])
    with pytest.raises(ValueError, match="poses"):
        train_lidar_bounds_map(run_dir, dataset_dir)


@pytest.mark.covers("T86")
def test_timestamp_and_file_name_mismatch_raise(tmp_path: Path):
    """A CSV timestamp_ns, time_s or file name that disagrees with the dataset is refused."""
    poses = np.tile(np.eye(4), (1, 1, 1))
    good_file = f"lidar/{SYNTHETIC_SENSOR}/000000_0.npy"
    bad_rows = {
        "timestamp_ns": [SYNTHETIC_SENSOR, "lidar", "train", 0, 999, 0.0, good_file],
        "time_s": [SYNTHETIC_SENSOR, "lidar", "train", 0, 0, 5.0, good_file],
        "file": [SYNTHETIC_SENSOR, "lidar", "train", 0, 0, 0.0, f"lidar/{SYNTHETIC_SENSOR}/000000_1.npy"],
    }
    for case_name, bad_row in bad_rows.items():
        case_root = tmp_path / case_name
        case_root.mkdir()
        dataset_dir, run_dir = build_synthetic_run(case_root, poses, csv_rows=[bad_row])
        with pytest.raises(ValueError):
            train_lidar_bounds_map(run_dir, dataset_dir)


@pytest.mark.covers("T86")
def test_dataset_hash_mismatch_raises(tmp_path: Path):
    """Altering poses.npz after the split record was written must be refused."""
    poses = np.tile(np.eye(4), (2, 1, 1))
    dataset_dir, run_dir = build_synthetic_run(tmp_path, poses)
    shifted = poses.copy()
    shifted[:, 0, 3] += 1.0
    np.savez(dataset_dir / "poses.npz",
             **{f"{SYNTHETIC_SENSOR}{POSE_KEY_SEPARATOR}times": np.arange(2, dtype=np.float64),
                f"{SYNTHETIC_SENSOR}{POSE_KEY_SEPARATOR}poses_world_sensor": shifted})
    with pytest.raises(ValueError, match="not the dataset"):
        train_lidar_bounds_map(run_dir, dataset_dir)


@pytest.mark.covers("T86")
def test_run_without_train_lidar_frames_raises(tmp_path: Path):
    """Only test-split lidar rows: nothing to bound, so raise."""
    rows = [[SYNTHETIC_SENSOR, "lidar", "test", 0, 0, 0.0, f"lidar/{SYNTHETIC_SENSOR}/000000_0.npy"]]
    dataset_dir, run_dir = build_synthetic_run(tmp_path, np.eye(4)[None], csv_rows=rows)
    with pytest.raises(ValueError, match="no train lidar"):
        train_lidar_bounds_map(run_dir, dataset_dir)


@pytest.mark.covers("T86")
def test_crop_mask_rejects_bad_inputs():
    """(N,1) opacities, logit-range opacities, swapped corners and NaN means all raise."""
    means = np.zeros((2, 3))
    lower, upper = np.full(3, -1.0), np.full(3, 1.0)
    with pytest.raises(ValueError):
        crop_mask(means, np.full((2, 1), 0.5), lower, upper, TEST_MIN_OPACITY)
    with pytest.raises(ValueError):
        crop_mask(means, np.array([-3.0, 0.5]), lower, upper, TEST_MIN_OPACITY)
    with pytest.raises(ValueError):
        crop_mask(means, np.array([0.5, 4.0]), lower, upper, TEST_MIN_OPACITY)
    with pytest.raises(ValueError):
        crop_mask(means, np.full(2, 0.5), upper, lower, TEST_MIN_OPACITY)
    with pytest.raises(ValueError):
        crop_mask(np.full((2, 3), np.nan), np.full(2, 0.5), lower, upper, TEST_MIN_OPACITY)


@pytest.mark.covers("T86")
def test_crop_mask_rejects_bad_min_opacity():
    """A NaN, infinite, negative or above-one opacity floor raises."""
    means = np.zeros((2, 3))
    lower, upper = np.full(3, -1.0), np.full(3, 1.0)
    for bad_floor in (np.nan, np.inf, -0.1, 1.5):
        with pytest.raises(ValueError, match="min_opacity"):
            crop_mask(means, np.full(2, 0.5), lower, upper, bad_floor)
