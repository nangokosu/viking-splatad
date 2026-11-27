"""Tests for streaming the recorded sweeps of a session."""

import json
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.dataset.schema import SCHEMA_VERSION
from viking_splatad.dataset.writer import POSE_KEY_SEPARATOR
from viking_splatad.eval.session_rays import (
    count_session_sweeps,
    iter_session_sweeps,
)

# Dataset sizes of the synthetic session.
SWEEP_COUNT = 4
EGO_BOX_HALF_EXTENTS = [1.0, 1.0, 1.0]
EGO_POINT = [0.1, 0.1, 0.1]
OUTSIDE_POINT = [3.0, 0.0, 4.0]
SENSOR_NAME = "ouster"
SWEEP_PERIOD_S = 0.1
TRAIN_FRACTION = 0.5


def write_session(dataset_dir: Path) -> list:
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
    for frame_index, timestamp_ns in enumerate(timestamps_ns):
        stem = f"{frame_index:06d}_{timestamp_ns}"
        sweep_rows = np.array([EGO_POINT + [1.0, 0.0, 0.0], OUTSIDE_POINT + [1.0, 0.0, 0.0]], dtype=np.float32)
        np.save(dataset_dir / "lidar" / SENSOR_NAME / f"{stem}.npy", sweep_rows)
    return timestamps_ns


SPLIT_SETTINGS = {
    "dataset_start_fraction": 0.0, "dataset_end_fraction": 1.0, "train_fraction": TRAIN_FRACTION,
    "test_split_method": None, "test_render_fraction": 1.0}


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
        assert sweep.lidar_to_map.shape == (4, 4)
        assert sweep.lidar_to_map[0, 3] == float(frame_index)
    labels = {sweep.split for sweep in sweeps}
    assert labels <= {"train", "test", "guard"} and "test" in labels and "train" in labels


def test_sweep_outside_split_window_raises(tmp_path):
    """A window covering only the first frames leaves the last sweeps without a split row."""
    dataset_dir = tmp_path / "session"
    write_session(dataset_dir)
    narrow_settings = dict(SPLIT_SETTINGS, dataset_end_fraction=0.5)
    with pytest.raises(ValueError):
        list(iter_session_sweeps(dataset_dir, narrow_settings))
