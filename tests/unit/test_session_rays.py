"""Tests for map-frame ray extraction of recorded sweeps (T93)."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.dataset.schema import SCHEMA_VERSION
from viking_splatad.dataset.writer import POSE_KEY_SEPARATOR
from viking_splatad.eval.session_rays import (
    NO_RETURN_DEPTH_PLACEHOLDER_M,
    count_session_sweeps,
    iter_session_sweeps,
    sweep_rays_in_map,
)

ABSOLUTE_TOLERANCE = 1e-12
# Pose: rotation of 90 degrees about z, translation (10, 20, 30).
ROTATION_ANGLE_RADIANS = math.pi / 2.0
TRANSLATION = np.array([10.0, 20.0, 30.0])
# Dataset sizes of the synthetic session.
SWEEP_COUNT = 4
EGO_BOX_HALF_EXTENTS = [1.0, 1.0, 1.0]
EGO_POINT = [0.1, 0.1, 0.1]
OUTSIDE_POINT = [3.0, 0.0, 4.0]
SENSOR_NAME = "ouster"
SWEEP_PERIOD_S = 0.1
TRAIN_FRACTION = 0.5


def make_pose() -> np.ndarray:
    """4x4 pose: Rz(90 deg) and the translation above."""
    cosine, sine = math.cos(ROTATION_ANGLE_RADIANS), math.sin(ROTATION_ANGLE_RADIANS)
    pose = np.eye(4)
    pose[:3, :3] = [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    pose[:3, 3] = TRANSLATION
    return pose


@pytest.mark.covers("T93")
def test_returns_are_rotated_with_range_depth_and_origin():
    """Rz(90): x -> y, y -> -x. Return (3,0,4): range 5, direction R(3,0,4)/5 = (0,3,4)/5."""
    returns = np.array([[3.0, 0.0, 4.0], [0.0, 2.0, 0.0]])
    origin, directions, depths, did_return = sweep_rays_in_map(returns, np.zeros((0, 3)), make_pose())
    np.testing.assert_allclose(origin, TRANSLATION, atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(directions[0], np.array([0.0, 3.0, 4.0]) / 5.0, atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(directions[1], [-1.0, 0.0, 0.0], atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(depths, [5.0, 2.0], atol=ABSOLUTE_TOLERANCE)
    assert did_return.all()
    # The 3x4 form gives the same rays.
    three_by_four = sweep_rays_in_map(returns, np.zeros((0, 3)), make_pose()[:3])
    np.testing.assert_allclose(three_by_four[1], directions, atol=ABSOLUTE_TOLERANCE)


@pytest.mark.covers("T93")
def test_no_return_gets_placeholder_depth_flag_and_rotated_direction():
    """No-return (1,0,0) -> (0,1,0); depth placeholder (finite, positive); returns come first."""
    returns = np.array([[0.0, 0.0, 7.0]])
    no_returns = np.array([[1.0, 0.0, 0.0]])
    _, directions, depths, did_return = sweep_rays_in_map(returns, no_returns, make_pose())
    assert did_return.tolist() == [True, False]
    np.testing.assert_allclose(directions[1], [0.0, 1.0, 0.0], atol=ABSOLUTE_TOLERANCE)
    assert depths[1] == NO_RETURN_DEPTH_PLACEHOLDER_M
    assert np.isfinite(depths[1]) and depths[1] > 0.0
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=ABSOLUTE_TOLERANCE)


@pytest.mark.covers("T93")
def test_zero_and_non_finite_range_returns_are_dropped():
    """A return at the origin and one with NaN have no direction and disappear."""
    returns = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [0.0, 0.0, 2.0]])
    _, directions, depths, did_return = sweep_rays_in_map(returns, np.zeros((0, 3)), make_pose())
    assert directions.shape == (1, 3)
    np.testing.assert_allclose(depths, [2.0], atol=ABSOLUTE_TOLERANCE)
    assert did_return.tolist() == [True]


def write_session(dataset_dir: Path, write_no_return_files: bool = True) -> list:
    """Synthetic dataset: one lidar, SWEEP_COUNT sweeps each with one ego and one outside point.

    Returns:
        The frame time stamps in ns.
    """
    times = np.arange(SWEEP_COUNT, dtype=np.float64)
    timestamps_ns = [round(frame_time * 1e9) + 1 for frame_time in times]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sensors": [{"name": SENSOR_NAME, "kind": "lidar"}],
        "session": {"name": dataset_dir.name, "train_fraction": TRAIN_FRACTION},
        "lidar": {"min_lidar_dist": EGO_BOX_HALF_EXTENTS},
    }
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    poses = np.tile(np.eye(4), (SWEEP_COUNT, 1, 1))
    poses[:, 0, 3] = times  # pose translation x = frame index
    np.savez(dataset_dir / "poses.npz", **{
        f"{SENSOR_NAME}{POSE_KEY_SEPARATOR}times": times,
        f"{SENSOR_NAME}{POSE_KEY_SEPARATOR}poses_world_sensor": poses})
    (dataset_dir / "lidar" / SENSOR_NAME).mkdir(parents=True)
    (dataset_dir / "lidar_no_return" / SENSOR_NAME).mkdir(parents=True)
    for frame_index, timestamp_ns in enumerate(timestamps_ns):
        stem = f"{frame_index:06d}_{timestamp_ns}"
        sweep_rows = np.array([EGO_POINT + [1.0, 0.0, 0.0], OUTSIDE_POINT + [1.0, 0.0, 0.0]], dtype=np.float32)
        np.save(dataset_dir / "lidar" / SENSOR_NAME / f"{stem}.npy", sweep_rows)
        if write_no_return_files:
            no_return_rows = np.array([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
            np.save(dataset_dir / "lidar_no_return" / SENSOR_NAME / f"{stem}.npy", no_return_rows)
    return timestamps_ns


SPLIT_SETTINGS = {
    "dataset_start_fraction": 0.0, "dataset_end_fraction": 1.0, "train_fraction": TRAIN_FRACTION,
    "test_split_method": None, "test_render_fraction": 1.0}


@pytest.mark.covers("T93")
def test_iter_session_sweeps_yields_every_sweep_filtered_and_labelled(tmp_path):
    """Every sweep appears in time order; the ego point is removed; labels are train/test."""
    dataset_dir = tmp_path / "session"
    timestamps_ns = write_session(dataset_dir)
    assert count_session_sweeps(dataset_dir) == SWEEP_COUNT
    sweeps = list(iter_session_sweeps(dataset_dir, SPLIT_SETTINGS))
    assert [sweep.timestamp_ns for sweep in sweeps] == timestamps_ns
    for frame_index, sweep in enumerate(sweeps):
        np.testing.assert_allclose(sweep.returns_xyz, [OUTSIDE_POINT])
        assert sweep.returns_xyz.dtype == np.float64
        np.testing.assert_allclose(sweep.no_return_directions, [[0.0, 1.0, 0.0]])
        assert sweep.lidar_to_map.shape == (4, 4)
        assert sweep.lidar_to_map[0, 3] == float(frame_index)
    labels = {sweep.split for sweep in sweeps}
    assert labels <= {"train", "test", "guard"} and "test" in labels and "train" in labels


def test_missing_no_return_file_raises(tmp_path):
    """Like the dataparser with add_missing_points: a missing sidecar is an error."""
    dataset_dir = tmp_path / "session"
    write_session(dataset_dir, write_no_return_files=False)
    with pytest.raises(FileNotFoundError):
        list(iter_session_sweeps(dataset_dir, SPLIT_SETTINGS))


def test_sweep_outside_split_window_raises(tmp_path):
    """A window covering only the first frames leaves the last sweeps without a split row."""
    dataset_dir = tmp_path / "session"
    write_session(dataset_dir)
    narrow_settings = dict(SPLIT_SETTINGS, dataset_end_fraction=0.5)
    with pytest.raises(ValueError):
        list(iter_session_sweeps(dataset_dir, narrow_settings))
