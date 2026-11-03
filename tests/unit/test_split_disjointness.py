"""Train / test / guard disjointness of the frame splits (T21, T59, T60, T62, T66, T71, T72).

Leakage guard: a frame that is trained on must never be scored, and a guard frame (held out but
not rendered by the run) must be in neither split. Every property below is asserted from the
split output alone (frame file names and times), with the expected block geometry recomputed
here from the documented formulas rather than from the builder under test.
"""

import numpy as np
import pytest

from tests.unit.test_run_split import write_synthetic_dataset
from viking_splatad.dataset.run_split import (
    build_run_split,
    restrict_test_to_block_cores,
    select_guard_frames_for_render,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view

# Frame counts of the synthetic recording: 100 frames per sensor at 0.1 s.
FRAMES_PER_SENSOR = 100
FRAME_PERIOD_SECONDS = 0.1
# Offsets keep lidar and camera stamps apart and off the block bounds (no frame on a boundary).
CAMERA_OFFSET_SECONDS = 0.0123
LIDAR_OFFSET_SECONDS = 0.0571
WHOLE_RECORDING = (0.0, 1.0)


def make_view(tmp_path):
    """Synthetic camera+lidar dataset view with irregular offsets; returns (view, times by sensor)."""
    frame_ticks = np.arange(FRAMES_PER_SENSOR, dtype=np.float64) * FRAME_PERIOD_SECONDS
    sensor_times = {("cam", "camera"): frame_ticks + CAMERA_OFFSET_SECONDS,
                    ("lidar", "lidar"): frame_ticks + LIDAR_OFFSET_SECONDS}
    dataset_dir = tmp_path / "synthetic"
    # train_fraction in the manifest is irrelevant here: build_run_split takes it explicitly.
    write_synthetic_dataset(dataset_dir, sensor_times, 0.9)
    return open_dataset_view(dataset_dir), {name: times for (name, _), times in sensor_times.items()}


def frame_files_by_split(frames, sensor_name):
    """{split: set of frame file paths} of one sensor."""
    files_by_split = {}
    for frame in frames:
        if frame.sensor == sensor_name:
            files_by_split.setdefault(frame.split, set()).add(frame.file)
    return files_by_split


def assert_splits_are_a_partition(frames, sensor_times):
    """Per sensor: train, test, guard are pairwise disjoint and together are exactly the sensor's frames."""
    for sensor_name, times in sensor_times.items():
        files_by_split = frame_files_by_split(frames, sensor_name)
        train, test, guard = (files_by_split.get(name, set()) for name in ("train", "test", "guard"))
        assert train.isdisjoint(test) and train.isdisjoint(guard) and test.isdisjoint(guard), sensor_name
        # No frame is listed twice and none is lost: the union has as many members as the sensor has frames.
        listed = [frame.file for frame in frames if frame.sensor == sensor_name]
        assert len(listed) == len(set(listed)) == len(times)
        assert len(train | test | guard) == len(times)


@pytest.mark.parametrize("train_fraction", [0.5, 0.75, 0.9])
def test_linspace_split_is_a_disjoint_partition_with_no_guard(tmp_path, train_fraction):
    """T59: train and test never share a frame and no frame is left out."""
    view, sensor_times = make_view(tmp_path)
    frames, _ = build_run_split(view, *WHOLE_RECORDING, train_fraction)
    assert_splits_are_a_partition(frames, sensor_times)
    assert not [frame for frame in frames if frame.split == "guard"]


@pytest.mark.parametrize(("block_count", "train_fraction", "core_fraction"),
                         [(1, 0.8, 0.5), (2, 0.8, 0.5), (3, 0.9, 0.34), (2, 0.7, 1.0)])
def test_block_split_matches_independent_block_geometry(tmp_path, block_count, train_fraction, core_fraction):
    """T60/T62: with D the shared time span, block length L = D (1 - f) / K centred at t0 + D k / (K + 1).

    A frame is train iff outside every block; test iff inside a block core; guard iff inside a
    block but outside its core. Checked frame by frame from the times, and the three sets are disjoint.
    """
    view, sensor_times = make_view(tmp_path)
    frames, _ = build_run_split(
        view, *WHOLE_RECORDING, train_fraction, test_block_count=block_count, test_block_core_fraction=core_fraction)
    assert_splits_are_a_partition(frames, sensor_times)
    # Independent geometry from the documented formulas.
    all_times = np.concatenate(list(sensor_times.values()))
    span_start, span_duration = all_times.min(), all_times.max() - all_times.min()
    block_length = span_duration * (1.0 - train_fraction) / block_count
    centres = [span_start + span_duration * block_number / (block_count + 1) for block_number in range(1, block_count + 1)]
    for frame in frames:
        distance_to_nearest_centre = min(abs(frame.time_s - centre) for centre in centres)
        in_block = distance_to_nearest_centre < block_length / 2.0
        in_core = distance_to_nearest_centre < core_fraction * block_length / 2.0
        expected_split = "train" if not in_block else ("test" if in_core else "guard")
        assert frame.split == expected_split, (frame.sensor, frame.time_s)
    # Whole cores (c = 1) leave no guard frame.
    if core_fraction == 1.0:
        assert not [frame for frame in frames if frame.split == "guard"]
    else:
        assert [frame for frame in frames if frame.split == "guard"]


@pytest.mark.parametrize("render_fraction", [0.25, 0.5, 1.0])
def test_shared_interleaved_split_is_a_disjoint_partition(tmp_path, render_fraction):
    """T71/T72: lidar train / test / guard partition; a camera frame is never both train and held out;
    the render subset keeps ceil-or-so r of the held-out lidar instants, never more than all of them."""
    view, sensor_times = make_view(tmp_path)
    frames, _ = build_run_split(
        view, *WHOLE_RECORDING, 0.75, test_split_method="shared_interleaved", test_render_fraction=render_fraction)
    assert_splits_are_a_partition(frames, sensor_times)
    lidar_files = frame_files_by_split(frames, "lidar")
    held_out_count = len(lidar_files.get("test", set())) + len(lidar_files.get("guard", set()))
    # f = 0.75 over 100 lidar frames: ceil(75) train, so 25 held out whatever r is.
    assert held_out_count == FRAMES_PER_SENSOR - int(np.ceil(FRAMES_PER_SENSOR * 0.75))
    if render_fraction == 1.0:
        assert not lidar_files.get("guard")
    else:
        assert lidar_files["guard"] and lidar_files["test"]


def test_block_cores_move_frames_only_out_of_test():
    """T62: train is returned untouched; core + guard is exactly the original test set (no frame invented)."""
    times = {"cam": np.arange(20, dtype=np.float64)}
    train_positions = np.array([0, 1, 2, 3, 4, 5, 14, 15, 16, 17, 18, 19], dtype=np.int64)
    test_positions = np.arange(6, 14, dtype=np.int64)
    new_positions, guard_positions, _ = restrict_test_to_block_cores(
        times, {"cam": (train_positions, test_positions)}, [(6.0, 14.0)], 0.5)
    core_positions = new_positions["cam"][1]
    assert np.array_equal(new_positions["cam"][0], train_positions)
    assert set(core_positions).isdisjoint(guard_positions)
    assert set(core_positions) | set(guard_positions["cam"]) == set(test_positions)
    # Core = [10 - 0.5*8/2, 10 + 0.5*8/2) = [8, 12): times 8..11.
    assert list(core_positions) == [8, 9, 10, 11]


def test_render_selection_accepts_only_guard_frames():
    """T66: a train frame or a test-core frame in the request is refused; the result is a subset of guard."""
    frame_indices = np.arange(10, dtype=np.int64)
    guard_positions = np.array([3, 4, 7], dtype=np.int64)
    selected = select_guard_frames_for_render(guard_positions, frame_indices, [7, 3])
    assert list(selected) == [3, 7] and set(selected) <= set(guard_positions)
    for non_guard_index in (0, 5):  # 0 is train, 5 is a test core in this layout
        with pytest.raises(ValueError, match="guard"):
            select_guard_frames_for_render(guard_positions, frame_indices, [3, non_guard_index])
