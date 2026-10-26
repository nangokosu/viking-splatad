"""Tests for rebuilding a run's train/test frames (T58, T59) and scripts/09_extract_run_split.py."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.dataset.run_split import (
    build_run_split,
    compare_with_rendered_stems,
    neurad_train_eval_indices,
    split_digest,
    time_window_masks,
)
from viking_splatad.dataset.schema import SCHEMA_VERSION
from viking_splatad.dataset.writer import POSE_KEY_SEPARATOR
from viking_splatad.splatad.viking_conversions import open_dataset_view

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTRACT_SPLIT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "09_extract_run_split.py"


def load_extract_split_module():
    """Import scripts/09_extract_run_split.py as a module (its name starts with a digit)."""
    module_spec = importlib.util.spec_from_file_location("extract_run_split", EXTRACT_SPLIT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def write_synthetic_dataset(dataset_dir: Path, sensor_times: dict, train_fraction: float) -> None:
    """Minimal dataset ``open_dataset_view`` can read: manifest, poses.npz, empty frame files.

    Frame files are never decoded by the split code, so empty files suffice. Each frame's ns
    stamp is its time in ns (any distinct integer works; this keeps the files self-describing).

    Args:
        dataset_dir: Directory to create.
        sensor_times: {(name, kind): float64 times}.
        train_fraction: Written to ``session.train_fraction``.
    """
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "sensors": [{"name": name, "kind": kind} for name, kind in sensor_times],
        "session": {"name": dataset_dir.name, "train_fraction": train_fraction},
    }
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest))
    pose_arrays = {}
    for (name, kind), times in sensor_times.items():
        pose_arrays[f"{name}{POSE_KEY_SEPARATOR}times"] = np.asarray(times, dtype=np.float64)
        pose_arrays[f"{name}{POSE_KEY_SEPARATOR}poses_world_sensor"] = np.tile(np.eye(4), (len(times), 1, 1))
        frame_dir = dataset_dir / kind / name
        frame_dir.mkdir(parents=True)
        suffix = ".jpg" if kind == "camera" else ".npy"
        for frame_index, frame_time in enumerate(times):
            (frame_dir / f"{frame_index:06d}_{round(frame_time * 1e9)}{suffix}").touch()
    np.savez(dataset_dir / "poses.npz", **pose_arrays)


@pytest.mark.covers("T58")
def test_time_window_is_inclusive_and_spans_all_sensors():
    """Integer times 0..10 (camera) and 0.5..9.5 (lidar), window fractions 0.2..0.5.

    t_min = 0 and t_max = 10 over both sensors, so D = 10 and the window is [2, 5]; frames exactly
    on 2 and 5 are kept (inclusive bounds).
    """
    camera_times = np.arange(11, dtype=np.float64)
    lidar_times = np.arange(10, dtype=np.float64) + 0.5
    masks, bounds = time_window_masks({"cam": camera_times, "lidar": lidar_times}, 0.2, 0.5)
    # First principles: [t_min + D*0.2, t_min + D*0.5] with t_min=0, D=10.
    expected_start, expected_end = 0.0 + 10.0 * 0.2, 0.0 + 10.0 * 0.5
    assert bounds == (expected_start, expected_end)
    assert list(camera_times[masks["cam"]]) == [2.0, 3.0, 4.0, 5.0]
    assert list(lidar_times[masks["lidar"]]) == [2.5, 3.5, 4.5]


@pytest.mark.covers("T58")
def test_time_window_whole_recording_skips_the_filter():
    """Fractions (0, 1) keep every frame and report no bounds, like neurad's early return."""
    masks, bounds = time_window_masks({"cam": np.array([3.0, 1.0]), "lidar": np.array([2.0])}, 0.0, 1.0)
    assert bounds is None
    assert masks["cam"].all() and masks["lidar"].all()


@pytest.mark.covers("T59")
def test_linspace_positions_for_fraction_below_one():
    """n=10, f=0.9: ceil(9.0)=9 train picks at floor(k*9/8), k=0..8; eval is the one left out."""
    train_positions, eval_positions = neurad_train_eval_indices(10, 0.9)
    train_count = math.ceil(10 * 0.9)
    # Integer floor of evenly spaced positions on [0, n-1].
    expected_train = [(step * 9) // (train_count - 1) for step in range(train_count)]
    assert list(train_positions) == expected_train
    assert list(eval_positions) == sorted(set(range(10)) - set(expected_train))


@pytest.mark.covers("T59")
def test_fraction_one_puts_every_frame_in_both_splits_and_empty_sensor_is_empty():
    """neurad (unlike T21) evaluates on all frames at f=1.0; n=0 gives two empty splits."""
    train_positions, eval_positions = neurad_train_eval_indices(4, 1.0)
    assert list(train_positions) == list(eval_positions) == [0, 1, 2, 3]
    empty_train, empty_eval = neurad_train_eval_indices(0, 0.9)
    assert empty_train.size == 0 and empty_eval.size == 0


@pytest.mark.covers("T59")
def test_seeded_eval_subset_is_refused():
    """max_eval_frames draws a torch.randperm subset; it must not be silently ignored."""
    with pytest.raises(ValueError):
        neurad_train_eval_indices(10, 0.9, max_eval_frames=3)


@pytest.mark.covers("T58", "T59")
def test_build_run_split_splits_each_sensor_inside_the_window(tmp_path):
    """Two sensors, window [2, 7] over times 0..10: 6 camera and 5 lidar frames, split per sensor.

    Positions are relative to the windowed frames, so the camera's eval frame (position 4 of 6 at
    f=0.75: train floor(k*5/4) for k=0..4 -> 0,1,2,3,5) is recording frame 2+4 = 6.
    """
    camera_times = np.arange(11, dtype=np.float64)
    lidar_times = np.arange(10, dtype=np.float64) + 0.5
    dataset_dir = tmp_path / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): camera_times, ("lidar", "lidar"): lidar_times}, 0.75)
    frames, bounds = build_run_split(open_dataset_view(dataset_dir), 0.2, 0.7, 0.75)
    assert bounds == (2.0, 7.0)

    camera_frames = {frame.frame_index: frame.split for frame in frames if frame.sensor == "cam"}
    # Window keeps camera frames 2..7 (6 frames); ceil(6*0.75)=5 train.
    assert sorted(camera_frames) == [2, 3, 4, 5, 6, 7]
    assert [index for index, split in sorted(camera_frames.items()) if split == "test"] == [6]
    lidar_frames = {frame.frame_index: frame.split for frame in frames if frame.sensor == "lidar"}
    # Lidar 2.5..6.5 are frames 2..6 (5 frames); ceil(3.75)=4 train at floor(k*4/3) = 0,1,2,4 -> eval position 3.
    assert sorted(lidar_frames) == [2, 3, 4, 5, 6]
    assert [index for index, split in sorted(lidar_frames.items()) if split == "test"] == [5]
    # File paths are dataset-relative and name the same frame.
    first_camera_frame = next(frame for frame in frames if frame.sensor == "cam")
    assert first_camera_frame.file == f"camera/cam/000002_{round(2.0 * 1e9)}.jpg"
    assert first_camera_frame.time_s == 2.0


def test_split_digest_ignores_order_and_detects_membership_changes(tmp_path):
    """Same membership in another order -> same digest; one frame moved to test -> new digest."""
    dataset_dir = tmp_path / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(10, dtype=np.float64)}, 0.8)
    frames, _ = build_run_split(open_dataset_view(dataset_dir), 0.0, 1.0, 0.8)
    assert split_digest(frames) == split_digest(list(reversed(frames)))
    moved = [frame if index else type(frame)(**{**frame.__dict__, "split": "test"}) for index, frame in enumerate(frames)]
    assert split_digest(moved) != split_digest(frames)


def test_render_comparison_reports_missing_and_unexpected_stems(tmp_path):
    """Renders listing one wrong file name are a mismatch naming both sides."""
    dataset_dir = tmp_path / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(4, dtype=np.float64)}, 0.75)
    frames, _ = build_run_split(open_dataset_view(dataset_dir), 0.0, 1.0, 0.75)
    test_stem = next(Path(frame.file).stem for frame in frames if frame.split == "test")
    train_stems = [Path(frame.file).stem for frame in frames if frame.split == "train"]
    report = compare_with_rendered_stems(frames, {("camera", "train"): train_stems, ("camera", "test"): ["bogus"]})
    assert report["camera/train"]["match"]
    assert not report["camera/test"]["match"]
    assert report["camera/test"]["missing_from_renders"] == [test_stem]
    assert report["camera/test"]["unexpected_in_renders"] == ["bogus"]


def make_run_dir(tmp_path: Path, start_fraction: float, end_fraction: float) -> Path:
    """Run folder with a job_info.txt in splatad_run.sbatch's format (incl. an indented block)."""
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    (run_dir / "job_info.txt").write_text(
        "RUN_ID: run-a\nSESSION: synthetic\nGIT_SHA: abc\nTRAIN_SEED: 7\n"
        f"DATASET_START_FRACTION: {start_fraction!r}\nDATASET_END_FRACTION: {end_fraction!r}\n"
        "ADD_MISSING_POINTS: True\nenvironment build:\n  RUN_ID: not-this-one\n"
    )
    return run_dir


def test_script_writes_deterministic_record_and_checks_renders(tmp_path):
    """End to end: record matches build_run_split, re-run is byte-identical, a bad render fails."""
    extract_split = load_extract_split_module()
    dataset_dir = tmp_path / "data" / "synthetic"
    write_synthetic_dataset(
        dataset_dir,
        {("cam", "camera"): np.arange(11, dtype=np.float64), ("lidar", "lidar"): np.arange(10, dtype=np.float64) + 0.5},
        0.75,
    )
    run_dir = make_run_dir(tmp_path, 0.2, 0.7)
    arguments = ["--run-dir", str(run_dir), "--dataset-dir", str(dataset_dir)]

    # No renders yet: success, and the record holds the provenance and the same frames.
    assert extract_split.main(arguments) == 0
    first_bytes = (run_dir / "split.json").read_bytes()
    record = json.loads(first_bytes)
    expected_frames, _ = build_run_split(open_dataset_view(dataset_dir), 0.2, 0.7, 0.75)
    assert record["membership_sha256"] == split_digest(expected_frames)
    assert record["run"]["train_seed"] == "7" and record["run"]["run_id"] == "run-a"
    assert record["selection"]["train_fraction_source"] == "manifest.session.train_fraction"
    assert len((run_dir / "split_frames.csv").read_text().splitlines()) == len(expected_frames) + 1
    assert extract_split.main(arguments) == 0
    assert (run_dir / "split.json").read_bytes() == first_bytes

    # Correct camera renders for both splits -> match; then one extra file -> exit 1.
    for frame in expected_frames:
        if frame.kind == "camera":
            render_folder = run_dir / "renders" / frame.split / "rgb"
            render_folder.mkdir(parents=True, exist_ok=True)
            (render_folder / f"{Path(frame.file).stem}.jpg").touch()
    assert extract_split.main(arguments) == 0
    (run_dir / "renders" / "test" / "rgb" / "000000_0.jpg").touch()
    assert extract_split.main(arguments) == 1


def test_script_uses_the_config_override_not_the_saved_default(tmp_path):
    """config.yml's train_split_fraction (saved before the dataparser runs) is ignored; the override is used."""
    extract_split = load_extract_split_module()
    dataset_dir = tmp_path / "data" / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(10, dtype=np.float64)}, 0.9)
    run_dir = make_run_dir(tmp_path, 0.0, 1.0)
    config_dir = run_dir / "train" / "synthetic" / "splatad" / "run-a"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yml").write_text(
        "  train_split_fraction: 0.5\n  train_split_fraction_override: 0.6\n  max_eval_frames: null\n"
    )
    assert extract_split.main(["--run-dir", str(run_dir), "--dataset-dir", str(dataset_dir)]) == 0
    record = json.loads((run_dir / "split.json").read_text())
    assert record["selection"]["train_fraction"] == 0.6
    assert record["selection"]["train_fraction_source"] == "override"
    # ceil(10 * 0.6) = 6 train frames.
    assert record["counts"]["cam"]["train"] == 6


@pytest.mark.covers("T58")
def test_time_window_rounds_like_neurad_at_real_timestamps():
    """At epoch-scale times (~1.7e9 s) float rounding decides frames on a bound; neurad's order must hold.

    The expected bounds are computed here in neurad's operation order (end from the unshifted start,
    then start += D * start_fraction). A frame exactly on each bound is kept; one ulp outside is not.
    """
    t_min, duration, start_fraction, end_fraction = 1714571311.141694208, 182.3, 0.2852, 0.3675
    expected_end = t_min + duration * end_fraction
    expected_start = t_min + duration * start_fraction
    camera_times = np.array([
        t_min,
        np.nextafter(expected_start, -np.inf), expected_start,
        expected_end, np.nextafter(expected_end, np.inf),
        t_min + duration,
    ])
    masks, bounds = time_window_masks({"cam": camera_times}, start_fraction, end_fraction)
    assert bounds == (expected_start, expected_end)
    assert list(masks["cam"]) == [False, False, True, True, False, False]


def test_radar_frames_never_enter_the_window_or_the_split(tmp_path):
    """A radar stream outside the camera span must not widen t_min/t_max or appear in the split."""
    dataset_dir = tmp_path / "synthetic"
    write_synthetic_dataset(
        dataset_dir,
        {("cam", "camera"): np.arange(11, dtype=np.float64), ("radar", "radar"): np.array([-100.0, 100.0])},
        0.75,
    )
    frames, bounds = build_run_split(open_dataset_view(dataset_dir), 0.2, 0.5, 0.75)
    # Camera alone: t_min = 0, D = 10 -> [2, 5]. With radar it would have been [-60, 0].
    assert bounds == (2.0, 5.0)
    assert {frame.sensor for frame in frames} == {"cam"}
