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


def make_completed_run(tmp_path: Path, start_fraction: float, end_fraction: float, train_fraction: float,
                       config_fields: str = "", record_train_fraction: bool = True) -> tuple:
    """A successful run on a synthetic camera+lidar dataset, as splatad_run.sbatch leaves it.

    Writes the dataset, a job_info.txt in the sbatch's format (indented block included, with the
    lidar-render key appended last), config.yml, and correct renders for both splits (file names
    from build_run_split, which the T58/T59 tests above pin from first principles).

    Args:
        tmp_path: pytest temporary directory.
        start_fraction, end_fraction: Time window of the run.
        train_fraction: The dataset manifest's session.train_fraction.
        config_fields: Extra config.yml lines (e.g. an override).
        record_train_fraction: False builds a legacy run (no TRAIN_SPLIT_FRACTION; config.yml holds 0.5).

    Returns:
        (run_dir, dataset_dir, expected frames).
    """
    dataset_dir = tmp_path / "data" / "synthetic"
    write_synthetic_dataset(
        dataset_dir,
        {("cam", "camera"): np.arange(11, dtype=np.float64), ("lidar", "lidar"): np.arange(10, dtype=np.float64) + 0.5},
        train_fraction,
    )
    run_dir = tmp_path / "runs" / "run-a"
    run_dir.mkdir(parents=True)
    fraction_line = f"TRAIN_SPLIT_FRACTION: {train_fraction!r}\n" if record_train_fraction else ""
    (run_dir / "job_info.txt").write_text(
        "RUN_ID: run-a\nSESSION: synthetic\nGIT_SHA: abc\nTRAIN_SEED: 7\nRENDER_POSE_SOURCE: train+test\n"
        f"DATASET_START_FRACTION: {start_fraction!r}\nDATASET_END_FRACTION: {end_fraction!r}\n"
        f"ADD_MISSING_POINTS: True\n{fraction_line}environment build:\n  RUN_ID: not-this-one\n"
        "lidar_render_elapsed_s: 73\n"
    )
    config_dir = run_dir / "train" / "synthetic" / "splatad" / "run-a"
    config_dir.mkdir(parents=True)
    saved_fraction = train_fraction if record_train_fraction else 0.5
    (config_dir / "config.yml").write_text(
        f"  train_split_fraction: {saved_fraction}\n  max_eval_frames: null\n{config_fields}"
    )
    expected_frames, _ = build_run_split(open_dataset_view(dataset_dir), start_fraction, end_fraction, train_fraction)
    write_renders(run_dir, expected_frames)
    return run_dir, dataset_dir, expected_frames


def write_renders(run_dir: Path, frames) -> None:
    """Empty render files named like ns-render (rgb/*.jpg) and scripts/08 (lidar/*.npz) name them."""
    for frame in frames:
        folder_name, suffix = ("rgb", ".jpg") if frame.kind == "camera" else ("lidar", ".npz")
        render_folder = run_dir / "renders" / frame.split / folder_name
        render_folder.mkdir(parents=True, exist_ok=True)
        (render_folder / f"{Path(frame.file).stem}{suffix}").touch()


def run_script(run_dir: Path, dataset_dir: Path) -> int:
    """Run scripts/09 on a run folder and return its exit code."""
    return load_extract_split_module().main(["--run-dir", str(run_dir), "--dataset-dir", str(dataset_dir)])


def test_script_documents_a_successful_run_deterministically(tmp_path):
    """A complete run with matching renders: record written, same frames, byte-identical re-run.

    Replaces the earlier "no renders -> record" test: by the user's requirement (2026-10-26) only
    runs that returned their outputs are documented.
    """
    run_dir, dataset_dir, expected_frames = make_completed_run(tmp_path, 0.2, 0.7, 0.75)
    assert run_script(run_dir, dataset_dir) == 0
    first_bytes = (run_dir / "split.json").read_bytes()
    record = json.loads(first_bytes)
    assert record["membership_sha256"] == split_digest(expected_frames)
    assert record["run"]["train_seed"] == "7" and record["run"]["run_id"] == "run-a"
    assert record["selection"]["train_fraction"] == 0.75
    assert record["selection"]["train_fraction_source"].startswith("manifest.session.train_fraction (= job_info")
    assert all(check["match"] for check in record["render_check"].values())
    assert set(record["render_check"]) == {"camera/train", "camera/test", "lidar/train", "lidar/test"}
    assert len((run_dir / "split_frames.csv").read_text().splitlines()) == len(expected_frames) + 1
    assert run_script(run_dir, dataset_dir) == 0
    assert (run_dir / "split.json").read_bytes() == first_bytes


@pytest.mark.parametrize("break_run", ["no_lidar_render_key", "no_config", "no_test_lidar_renders",
                                       "no_train_camera_renders", "no_job_info", "no_session_key"])
def test_script_never_documents_an_incomplete_run_and_removes_a_stale_record(tmp_path, break_run):
    """Each way a run can fail before returning its outputs -> exit 2, no split.json / CSV left."""
    run_dir, dataset_dir, _ = make_completed_run(tmp_path, 0.2, 0.7, 0.75)
    assert run_script(run_dir, dataset_dir) == 0  # a record exists, then the run is broken
    job_info_path = run_dir / "job_info.txt"
    if break_run == "no_lidar_render_key":
        job_info_path.write_text(job_info_path.read_text().replace("lidar_render_elapsed_s: 73\n", ""))
    elif break_run == "no_config":
        next(run_dir.glob("train/*/splatad/*/config.yml")).unlink()
    elif break_run == "no_test_lidar_renders":
        for rendered_file in (run_dir / "renders" / "test" / "lidar").iterdir():
            rendered_file.unlink()
    elif break_run == "no_train_camera_renders":
        for rendered_file in (run_dir / "renders" / "train" / "rgb").iterdir():
            rendered_file.unlink()
    elif break_run == "no_session_key":
        job_info_path.write_text(job_info_path.read_text().replace("SESSION: synthetic\n", ""))
    else:
        job_info_path.unlink()
    assert run_script(run_dir, dataset_dir) == 2
    assert not (run_dir / "split.json").exists() and not (run_dir / "split_frames.csv").exists()


def test_script_refuses_renders_that_differ_from_the_split(tmp_path):
    """One rendered file outside the rebuilt split -> exit 1 and no record."""
    run_dir, dataset_dir, _ = make_completed_run(tmp_path, 0.2, 0.7, 0.75)
    (run_dir / "renders" / "test" / "rgb" / "000000_0.jpg").touch()
    assert run_script(run_dir, dataset_dir) == 1
    assert not (run_dir / "split.json").exists()


def test_script_refuses_disagreeing_train_fractions(tmp_path):
    """config.yml saying 0.5 while job_info and the manifest say 0.75 -> exit 1 (inconsistent run)."""
    run_dir, dataset_dir, _ = make_completed_run(tmp_path, 0.2, 0.7, 0.75)
    config_path = next(run_dir.glob("train/*/splatad/*/config.yml"))
    config_path.write_text(config_path.read_text().replace("train_split_fraction: 0.75", "train_split_fraction: 0.5"))
    assert run_script(run_dir, dataset_dir) == 1
    assert not (run_dir / "split.json").exists()


def test_new_style_run_without_a_saved_fraction_is_inconsistent_and_loses_its_stale_record(tmp_path):
    """TRAIN_SPLIT_FRACTION recorded but config.yml lacks train_split_fraction -> exit 1, old record deleted."""
    run_dir, dataset_dir, _ = make_completed_run(tmp_path, 0.2, 0.7, 0.75)
    assert run_script(run_dir, dataset_dir) == 0
    config_path = next(run_dir.glob("train/*/splatad/*/config.yml"))
    config_path.write_text(config_path.read_text().replace("  train_split_fraction: 0.75\n", ""))
    assert run_script(run_dir, dataset_dir) == 1
    assert not (run_dir / "split.json").exists() and not (run_dir / "split_frames.csv").exists()


def test_legacy_run_uses_the_manifest_not_the_saved_default(tmp_path):
    """A run from before TRAIN_SPLIT_FRACTION: config.yml's 0.5 is ignored, the manifest's 0.75 is used.

    ceil(n * 0.75) differs from ceil(n * 0.5), so a wrong fraction would also break the render check.
    """
    run_dir, dataset_dir, _ = make_completed_run(tmp_path, 0.2, 0.7, 0.75, record_train_fraction=False)
    assert run_script(run_dir, dataset_dir) == 0
    record = json.loads((run_dir / "split.json").read_text())
    assert record["selection"]["train_fraction"] == 0.75
    assert "legacy run" in record["selection"]["train_fraction_source"]
    # Window keeps camera frames 2..7 (6 frames): ceil(6 * 0.75) = 5 train.
    assert record["counts"]["cam"]["train"] == 5


def test_override_in_config_is_what_the_dataparser_used(tmp_path):
    """train_split_fraction_override (0.6) wins over the manifest (0.9), as in T27 resolve_train_fraction.

    The renders follow the 0.6 split; without the override the script would rebuild the 0.9 split,
    which the renders contradict.
    """
    run_dir, dataset_dir, _ = make_completed_run(tmp_path, 0.0, 1.0, 0.6, record_train_fraction=False)
    manifest_path = dataset_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["session"]["train_fraction"] = 0.9
    manifest_path.write_text(json.dumps(manifest))
    # Without the override: the manifest's 0.9 split disagrees with the 0.6 renders.
    assert run_script(run_dir, dataset_dir) == 1
    config_path = next(run_dir.glob("train/*/splatad/*/config.yml"))
    config_path.write_text(config_path.read_text() + "  train_split_fraction_override: 0.6\n")
    assert run_script(run_dir, dataset_dir) == 0
    record = json.loads((run_dir / "split.json").read_text())
    assert record["selection"]["train_fraction"] == 0.6
    assert record["selection"]["train_fraction_source"].startswith("override")
    # Camera has 11 frames in the whole recording: ceil(11 * 0.6) = 7 train.
    assert record["counts"]["cam"]["train"] == 7


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
