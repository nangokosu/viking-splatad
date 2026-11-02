"""Tests for the shared interleaved train/test split (T71) and its render subset (T72).

The held-out set is every ~1/(1-f)-th lidar sweep (neurad's LINSPACE on the reference sensor) plus,
for every other sensor, the frames nearest those sweeps; only an evenly spaced fraction r of the
held-out instants is rendered test, the rest is guard. Covered here: the transforms (known answers
derived in the tests), the CPU rebuild (build_run_split, scripts/09), the session config ->
submit_run.py -> splatad_run.sbatch plumbing, scripts/12's guard check, and (when torch + patched
neurad-studio are importable) the real dataparser agreeing with the rebuild.
"""

import json
import math
import types
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.unit.test_block_split import SBATCH_PATH, SESSION_CONFIG_DIRECTORY, _neurad_ready
from tests.unit.test_render_only_job import load_prepare_rerender_module
from tests.unit.test_run_split import run_script, write_renders, write_synthetic_dataset
from tests.unit.test_slurm_submit import load_submit_module
from viking_splatad.dataset.run_split import (
    GUARD_SPLIT_NAME,
    SHARED_INTERLEAVED_SPLIT_METHOD,
    TEST_SPLIT_NAME,
    TRAIN_SPLIT_NAME,
    build_run_split,
    nearest_reference_positions,
    resolve_test_split_method,
    restrict_test_to_render_subset,
    shared_interleaved_train_test_indices,
    split_digest,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view

# Sweep period of the synthetic lidar [s]: the Ouster OS0's 10 Hz, as in the real recordings.
SWEEP_PERIOD_SECONDS = 0.1


def linspace_test_positions(frame_count: int, train_fraction: float) -> np.ndarray:
    """Reference held-out positions from first principles: the integers int(j * (n-1)/(c-1)) skips.

    train count c = ceil(n f); train = {int(j * (n - 1) / (c - 1)) : j = 0..c-1}; test = the rest.

    Args:
        frame_count: n.
        train_fraction: f in (0, 1).

    Returns:
        Sorted int64 test positions.
    """
    train_count = math.ceil(frame_count * train_fraction)
    spacing = (frame_count - 1) / (train_count - 1)
    train_positions = {int(rank * spacing) for rank in range(train_count)}
    return np.array(sorted(set(range(frame_count)) - train_positions), dtype=np.int64)


def brute_force_nearest(reference_times: np.ndarray, query_time: float) -> int:
    """Index of the reference time nearest to ``query_time`` (first one on a tie), by full scan."""
    return int(np.argmin(np.abs(np.asarray(reference_times) - query_time)))


# ---- T71: shared interleaved instants ---------------------------------------------------------------

@pytest.mark.covers("T71")
def test_reference_sensor_holds_out_the_linspace_complement():
    """Lidar test positions are exactly the integers neurad's truncated linspace skips."""
    frame_count, train_fraction = 100, 0.9
    lidar_times = np.arange(frame_count) * SWEEP_PERIOD_SECONDS
    positions = shared_interleaved_train_test_indices({"lidar": lidar_times}, train_fraction, "lidar")
    expected_test = linspace_test_positions(frame_count, train_fraction)
    np.testing.assert_array_equal(positions["lidar"][1], expected_test)
    np.testing.assert_array_equal(positions["lidar"][0], np.setdiff1d(np.arange(frame_count), expected_test))
    assert len(expected_test) == frame_count - math.ceil(frame_count * train_fraction)


@pytest.mark.covers("T71")
@pytest.mark.parametrize("frame_count, train_fraction", [(2412, 0.9), (2162, 0.9)])
def test_full_recording_holds_out_every_tenth_sweep(frame_count, train_fraction):
    """At the real May/June sweep counts and f = 0.9 the held-out sweeps are exactly 1/(1-f) apart."""
    lidar_times = np.arange(frame_count) * SWEEP_PERIOD_SECONDS
    test_positions = shared_interleaved_train_test_indices({"lidar": lidar_times}, train_fraction, "lidar")["lidar"][1]
    expected_stride = round(1.0 / (1.0 - train_fraction))
    assert set(np.diff(test_positions).tolist()) == {expected_stride}
    assert len(test_positions) == frame_count - math.ceil(frame_count * train_fraction)


@pytest.mark.covers("T71")
def test_camera_at_the_same_rate_holds_out_the_same_instants():
    """Camera frames 0.03 s after each sweep: the nearest sweep has the same index, so the splits coincide."""
    frame_count, train_fraction = 150, 0.8
    camera_offset_seconds = 0.3 * SWEEP_PERIOD_SECONDS  # under half a period: nearest sweep = same index
    lidar_times = np.arange(frame_count) * SWEEP_PERIOD_SECONDS
    camera_times = lidar_times + camera_offset_seconds
    positions = shared_interleaved_train_test_indices({"cam": camera_times, "lidar": lidar_times}, train_fraction, "lidar")
    np.testing.assert_array_equal(positions["cam"][1], positions["lidar"][1])
    np.testing.assert_array_equal(positions["cam"][0], positions["lidar"][0])


@pytest.mark.covers("T71")
def test_camera_at_another_rate_follows_its_nearest_sweep():
    """A 20 Hz camera with jitter: a camera frame is test iff its brute-force nearest sweep is test."""
    lidar_count, train_fraction = 60, 0.9
    lidar_times = np.arange(lidar_count) * SWEEP_PERIOD_SECONDS
    camera_period_seconds = SWEEP_PERIOD_SECONDS / 2
    jitter_generator = np.random.default_rng(3)
    camera_count = 2 * lidar_count
    camera_times = (np.arange(camera_count) * camera_period_seconds
                    + jitter_generator.uniform(-0.2, 0.2, camera_count) * camera_period_seconds)
    positions = shared_interleaved_train_test_indices({"cam": camera_times, "lidar": lidar_times}, train_fraction, "lidar")
    lidar_test_set = set(positions["lidar"][1].tolist())
    expected_camera_test = [camera_position for camera_position, camera_time in enumerate(camera_times)
                            if brute_force_nearest(lidar_times, camera_time) in lidar_test_set]
    np.testing.assert_array_equal(positions["cam"][1], expected_camera_test)
    np.testing.assert_array_equal(positions["cam"][0], np.setdiff1d(np.arange(camera_count), expected_camera_test))


@pytest.mark.covers("T71")
def test_nearest_reference_breaks_ties_towards_the_earlier_frame():
    """A query exactly half-way between two sweeps belongs to the earlier one; unsorted input is handled."""
    reference_times = np.array([0.2, 0.0, 0.1])  # unsorted on purpose
    query_times = np.array([0.05, 0.149, 0.151, -1.0, 9.0])
    # Exact half-way 0.05 -> 0.0 (position 1); 0.149 -> 0.1 (2); 0.151 -> 0.2 (0); ends clip to the extremes.
    np.testing.assert_array_equal(nearest_reference_positions(reference_times, query_times), [1, 2, 0, 1, 0])


@pytest.mark.covers("T71")
def test_fraction_one_puts_every_frame_of_every_sensor_in_both_splits():
    """neurad's f == 1 convention: train and test are every frame (test is for viewing)."""
    sensor_times = {"cam": np.arange(5) * 0.1, "lidar": np.arange(4) * 0.1}
    positions = shared_interleaved_train_test_indices(sensor_times, 1.0, "lidar")
    for name, times in sensor_times.items():
        np.testing.assert_array_equal(positions[name][0], np.arange(len(times)))
        np.testing.assert_array_equal(positions[name][1], np.arange(len(times)))


@pytest.mark.covers("T71")
def test_invalid_inputs_are_refused():
    """Unknown reference, fraction outside (0, 1], or camera frames without any sweep -> ValueError."""
    with pytest.raises(ValueError, match="reference sensor"):
        shared_interleaved_train_test_indices({"cam": np.arange(3.0)}, 0.9, "lidar")
    for bad_fraction in (0.0, 1.5):
        with pytest.raises(ValueError, match="train_fraction"):
            shared_interleaved_train_test_indices({"lidar": np.arange(3.0)}, bad_fraction, "lidar")
    with pytest.raises(ValueError, match="has none"):
        shared_interleaved_train_test_indices({"cam": np.arange(3.0), "lidar": np.empty(0)}, 0.9, "lidar")


def test_split_method_resolution():
    """Unset method: blocks iff a block count is set; explicit methods validated; blocks ignored by T71."""
    assert resolve_test_split_method(None, None) is None
    assert resolve_test_split_method(None, 2) == "middle_blocks"
    assert resolve_test_split_method("shared_interleaved", 2) == "shared_interleaved"
    assert resolve_test_split_method("shared_interleaved", None) == "shared_interleaved"
    with pytest.raises(ValueError, match="needs test_block_count"):
        resolve_test_split_method("middle_blocks", None)
    with pytest.raises(ValueError, match="must be one of"):
        resolve_test_split_method("every_tenth", None)


# ---- T72: render subset ------------------------------------------------------------------------------

def smoke_like_positions():
    """The smoke window's shape: 150 sweeps and 150 camera frames 0.03 s later, f = 0.8 (T71)."""
    frame_count, train_fraction = 150, 0.8
    lidar_times = np.arange(frame_count) * SWEEP_PERIOD_SECONDS
    sensor_times = {"cam": lidar_times + 0.3 * SWEEP_PERIOD_SECONDS, "lidar": lidar_times}
    return sensor_times, shared_interleaved_train_test_indices(sensor_times, train_fraction, "lidar")


@pytest.mark.covers("T72")
def test_render_subset_keeps_evenly_spaced_held_out_instants():
    """r = 0.2 of 30 held-out sweeps: ceil(6) ranks int(linspace(0, 29, 6)) render; 24 are guard."""
    sensor_times, positions = smoke_like_positions()
    render_fraction = 0.2
    restricted, guard = restrict_test_to_render_subset(sensor_times, positions, "lidar", render_fraction)
    held_out_lidar = positions["lidar"][1]
    rendered_count = math.ceil(len(held_out_lidar) * render_fraction)
    rank_spacing = (len(held_out_lidar) - 1) / (rendered_count - 1)
    expected_rendered = held_out_lidar[[int(rank * rank_spacing) for rank in range(rendered_count)]]
    np.testing.assert_array_equal(restricted["lidar"][1], expected_rendered)
    assert (len(held_out_lidar), len(expected_rendered), len(guard["lidar"])) == (30, 6, 24)
    for name in sensor_times:
        # Train untouched; rendered and guard partition the held-out frames.
        np.testing.assert_array_equal(restricted[name][0], positions[name][0])
        assert np.intersect1d(restricted[name][1], guard[name]).size == 0
        np.testing.assert_array_equal(np.union1d(restricted[name][1], guard[name]), positions[name][1])
    # Camera renders exactly the instants the lidar renders (same index here: 0.03 s offset).
    np.testing.assert_array_equal(restricted["cam"][1], restricted["lidar"][1])


@pytest.mark.covers("T72")
def test_whole_render_fraction_and_fraction_one_change_nothing():
    """r = 1 keeps every held-out frame; with f = 1 (train = test) nothing is held out, so no guard."""
    sensor_times, positions = smoke_like_positions()
    restricted, guard = restrict_test_to_render_subset(sensor_times, positions, "lidar", 1.0)
    for name in sensor_times:
        np.testing.assert_array_equal(restricted[name][1], positions[name][1])
        assert guard[name].size == 0
    all_positions = shared_interleaved_train_test_indices(sensor_times, 1.0, "lidar")
    restricted, guard = restrict_test_to_render_subset(sensor_times, all_positions, "lidar", 0.2)
    for name in sensor_times:
        np.testing.assert_array_equal(restricted[name][1], all_positions[name][1])
        assert guard[name].size == 0


@pytest.mark.covers("T72")
@pytest.mark.parametrize("bad_fraction", [0.0, -0.1, 1.5])
def test_render_fraction_outside_unit_interval_is_refused(bad_fraction):
    """r must lie in (0, 1]."""
    sensor_times, positions = smoke_like_positions()
    with pytest.raises(ValueError, match="test_render_fraction"):
        restrict_test_to_render_subset(sensor_times, positions, "lidar", bad_fraction)


# ---- CPU rebuild (build_run_split) and scripts/09 ---------------------------------------------------

def write_same_rate_dataset(dataset_dir: Path, frame_count: int, train_fraction: float) -> None:
    """Synthetic camera + lidar at the same rate, camera 0.03 s after each sweep."""
    lidar_times = np.arange(frame_count) * SWEEP_PERIOD_SECONDS + 1.0
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): lidar_times + 0.3 * SWEEP_PERIOD_SECONDS,
                                          ("lidar", "lidar"): lidar_times}, train_fraction)


@pytest.mark.covers("T71")
@pytest.mark.covers("T72")
def test_build_run_split_counts_and_shared_instants(tmp_path):
    """150 frames per sensor, f = 0.8, r = 0.2 -> 120 train, 6 test, 24 guard each, same instants."""
    dataset_dir = tmp_path / "data" / "synthetic"
    write_same_rate_dataset(dataset_dir, 150, 0.8)
    frames, _ = build_run_split(open_dataset_view(dataset_dir), 0.0, 1.0, 0.8, test_block_count=2,
                                test_block_core_fraction=0.1, test_split_method=SHARED_INTERLEAVED_SPLIT_METHOD,
                                test_render_fraction=0.2)
    for sensor_name in ("cam", "lidar"):
        counts = {split_name: sum(frame.sensor == sensor_name and frame.split == split_name for frame in frames)
                  for split_name in (TRAIN_SPLIT_NAME, TEST_SPLIT_NAME, GUARD_SPLIT_NAME)}
        assert counts == {TRAIN_SPLIT_NAME: 120, TEST_SPLIT_NAME: 6, GUARD_SPLIT_NAME: 24}
    for split_name in (TEST_SPLIT_NAME, GUARD_SPLIT_NAME):
        indices_by_sensor = [sorted(frame.frame_index for frame in frames
                                    if frame.sensor == sensor_name and frame.split == split_name)
                             for sensor_name in ("cam", "lidar")]
        assert indices_by_sensor[0] == indices_by_sensor[1]


def test_render_fraction_without_shared_interleaved_is_refused(tmp_path):
    """r < 1 with the block or LINSPACE split would be silently ignored, so it is refused."""
    dataset_dir = tmp_path / "data" / "synthetic"
    write_same_rate_dataset(dataset_dir, 30, 0.8)
    for block_count in (None, 2):
        with pytest.raises(ValueError, match="test_render_fraction"):
            build_run_split(open_dataset_view(dataset_dir), 0.0, 1.0, 0.8, test_block_count=block_count,
                            test_render_fraction=0.5)


def make_shared_run(tmp_path: Path, job_info_method: str) -> tuple:
    """A completed shared interleaved run (f = 0.8, r = 0.2), renders named after the rebuilt split.

    Args:
        tmp_path: pytest temporary directory.
        job_info_method: TEST_SPLIT_METHOD written to job_info.txt (config.yml says shared_interleaved).

    Returns:
        (run_dir, dataset_dir, expected frames).
    """
    train_fraction, render_fraction = 0.8, 0.2
    dataset_dir = tmp_path / "data" / "synthetic"
    write_same_rate_dataset(dataset_dir, 60, train_fraction)
    run_dir = tmp_path / "runs" / "run-s"
    run_dir.mkdir(parents=True)
    (run_dir / "job_info.txt").write_text(
        "RUN_ID: run-s\nSESSION: synthetic\nGIT_SHA: abc\nTRAIN_SEED: 7\nRENDER_POSE_SOURCE: test\n"
        "DATASET_START_FRACTION: 0.0\nDATASET_END_FRACTION: 1.0\nADD_MISSING_POINTS: True\n"
        f"TRAIN_SPLIT_FRACTION: {train_fraction!r}\nTEST_BLOCK_COUNT: 2\nTEST_BLOCK_CORE_FRACTION: 0.1\n"
        f"TEST_SPLIT_METHOD: {job_info_method}\nTEST_RENDER_FRACTION: {render_fraction!r}\n"
        "lidar_render_elapsed_s: 73\n")
    config_dir = run_dir / "train" / "synthetic" / "splatad" / "run-s"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yml").write_text(
        f"  train_split_fraction: {train_fraction}\n  max_eval_frames: null\n  test_block_count: 2\n"
        f"  test_block_core_fraction: 0.1\n  test_split_method: shared_interleaved\n"
        f"  test_render_fraction: {render_fraction}\n  eval_whole_test_blocks: true\n")
    expected_frames, _ = build_run_split(open_dataset_view(dataset_dir), 0.0, 1.0, train_fraction,
                                         test_block_count=2, test_block_core_fraction=0.1,
                                         test_split_method=SHARED_INTERLEAVED_SPLIT_METHOD,
                                         test_render_fraction=render_fraction)
    write_renders(run_dir, [frame for frame in expected_frames if frame.split == TEST_SPLIT_NAME])
    return run_dir, dataset_dir, expected_frames


@pytest.mark.covers("T71")
def test_script_documents_a_shared_interleaved_run(tmp_path):
    """Matching renders -> exit 0; the record names T71, the stride 1/(1-f) = 5, r, and val = test + guard."""
    run_dir, dataset_dir, expected_frames = make_shared_run(tmp_path, "shared_interleaved")
    assert run_script(run_dir, dataset_dir) == 0
    record = json.loads((run_dir / "split.json").read_text())
    selection = record["selection"]
    assert record["membership_sha256"] == split_digest(expected_frames)
    assert "T71" in selection["algorithm"] and selection["test_split_method"] == "shared_interleaved"
    assert selection["test_stride_frames"] == pytest.approx(1.0 / (1.0 - 0.8))
    assert selection["test_reference_sensor_kind"] == "lidar"
    assert selection["test_render_fraction"] == 0.2
    assert selection["test_blocks_s"] == [] and selection["eval_during_training_s"] == []
    # 60 frames: 12 held out (60 - ceil(48)), ceil(12 * 0.2) = 3 rendered, 9 guard; val = all 12.
    for sensor_name in ("cam", "lidar"):
        assert {split_name: record["counts"][sensor_name][split_name] for split_name in ("train", "test", "guard", "val")} \
            == {"train": 48, "test": 3, "guard": 9, "val": 12}


def test_script_refuses_disagreeing_split_methods(tmp_path):
    """job_info.txt says middle_blocks while config.yml says shared_interleaved -> exit 1, no record."""
    run_dir, dataset_dir, _ = make_shared_run(tmp_path, "middle_blocks")
    assert run_script(run_dir, dataset_dir) == 1
    assert not (run_dir / "split.json").exists()


# ---- session config -> submit_run.py -> splatad_run.sbatch -----------------------------------------

def write_session(config_directory: Path, session_name: str, split_settings: dict) -> None:
    """Write a minimal session config holding only the given ``split`` mapping."""
    config_directory.mkdir(parents=True, exist_ok=True)
    (config_directory / f"{session_name}.yaml").write_text(yaml.safe_dump({"split": split_settings}))


def test_real_sessions_use_the_same_shared_interleaved_split():
    """May, June B and June R share the split rule (user's choice 2026-11-01: shared interleaved, r = 0.1)."""
    submit_module = load_submit_module()
    session_names = [session_path.stem for session_path in SESSION_CONFIG_DIRECTORY.glob("*.yaml")]
    assert {"may_short_grass", "june_tall_grass"} <= set(session_names)
    for session_name in session_names:
        assert submit_module.session_test_split_method(
            session_name, Path("/absent"), SESSION_CONFIG_DIRECTORY) == "shared_interleaved"
        assert submit_module.profile_test_render_fraction(
            {}, session_name, Path("/absent"), SESSION_CONFIG_DIRECTORY) == 0.1


def test_smoke_profile_renders_a_fifth_of_the_held_out_instants():
    """The smoke profile overrides r to 0.2 (with its f = 0.8), and its A4500 variant inherits it."""
    profiles = load_submit_module().load_slurm_config()["profiles"]
    assert profiles["smoke"]["test_render_fraction"] == 0.2
    assert profiles["smoke_a4500"]["test_render_fraction"] == 0.2


def test_render_fraction_precedence_and_refusals(tmp_path):
    """Profile r wins over the session's; default 1.0; r < 1 without shared_interleaved and bad values refused."""
    submit_module = load_submit_module()
    sessions_dir, intermediate_dir = tmp_path / "sessions", tmp_path / "intermediate"
    write_session(sessions_dir, "shared", {"method": "shared_interleaved", "test_render_fraction": 0.1})
    write_session(sessions_dir, "shared_default", {"method": "shared_interleaved"})
    write_session(sessions_dir, "blocks", {"test_block_count": 2, "test_render_fraction": 0.5})
    write_session(sessions_dir, "unknown", {"method": "every_tenth"})
    assert submit_module.profile_test_render_fraction({}, "shared", intermediate_dir, sessions_dir) == 0.1
    assert submit_module.profile_test_render_fraction(
        {"test_render_fraction": 0.25}, "shared", intermediate_dir, sessions_dir) == 0.25
    assert submit_module.profile_test_render_fraction({}, "shared_default", intermediate_dir, sessions_dir) == 1.0
    with pytest.raises(ValueError, match="needs split.method"):
        submit_module.profile_test_render_fraction({}, "blocks", intermediate_dir, sessions_dir)
    for bad_value in (0, 1.5, True, "0.1"):
        with pytest.raises(ValueError, match="must be a number"):
            submit_module.profile_test_render_fraction(
                {"test_render_fraction": bad_value}, "shared", intermediate_dir, sessions_dir)
    with pytest.raises(ValueError, match="split.method"):
        submit_module.session_test_split_method("unknown", intermediate_dir, sessions_dir)


def test_profile_train_fraction_skips_the_block_limit_for_shared_interleaved(tmp_path):
    """f = 0.3 overlaps two T60 blocks (needs f > 1/3) but is valid for the shared interleaved split."""
    submit_module = load_submit_module()
    sessions_dir, intermediate_dir = tmp_path / "sessions", tmp_path / "intermediate"
    write_session(sessions_dir, "shared", {"method": "shared_interleaved", "test_block_count": 2,
                                           "test_block_core_fraction": 0.1})
    write_session(sessions_dir, "blocks", {"test_block_count": 2, "test_block_core_fraction": 0.1})
    assert submit_module.profile_split_fractions({"train_fraction": 0.3}, "shared", intermediate_dir,
                                                 sessions_dir) == (0.3, 0.3, 0.1)
    with pytest.raises(ValueError, match="1/\\(K\\+1\\)"):
        submit_module.profile_split_fractions({"train_fraction": 0.3}, "blocks", intermediate_dir, sessions_dir)


def test_job_environment_exports_split_method_and_render_fraction():
    """TEST_SPLIT_METHOD verbatim and TEST_RENDER_FRACTION as the float's repr; both left out when not given."""
    submit_module = load_submit_module()
    env_marker = {"env_prefix": "/w/stage/envs/splatad-x"}
    profile_settings = {"seed": 1, "max_num_iterations": 10, "steps_per_save": 5, "render_pose_source": "test",
                        "render_downscale_factor": 1}
    job_environment = submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", env_marker,
        test_split_method="shared_interleaved", test_render_fraction=0.1)
    assert job_environment["TEST_SPLIT_METHOD"] == "shared_interleaved"
    assert job_environment["TEST_RENDER_FRACTION"] == repr(0.1)
    bare_environment = submit_module.build_job_environment(profile_settings, "s", "run", "a" * 40, "/w/stage",
                                                           env_marker)
    assert "TEST_SPLIT_METHOD" not in bare_environment and "TEST_RENDER_FRACTION" not in bare_environment


def test_sbatch_records_and_passes_split_method_and_render_fraction():
    """splatad_run.sbatch writes both to job_info.txt and passes them to viking-data when set."""
    sbatch_text = SBATCH_PATH.read_text()
    assert 'echo "TEST_SPLIT_METHOD: $TEST_SPLIT_METHOD"' in sbatch_text
    assert 'echo "TEST_RENDER_FRACTION: $TEST_RENDER_FRACTION"' in sbatch_text
    assert '--test-split-method "$TEST_SPLIT_METHOD"' in sbatch_text
    assert '--test-render-fraction "$TEST_RENDER_FRACTION"' in sbatch_text
    train_command = sbatch_text[sbatch_text.index("ns-train splatad"):]
    train_command = train_command[:train_command.index("\nTRAIN_END_SECONDS")]
    assert '${TEST_SPLIT_ARGS[@]+"${TEST_SPLIT_ARGS[@]}"}' in train_command


# ---- scripts/12: guard frames of a shared interleaved run -----------------------------------------

def _shared_trainer_config(render_fraction: float) -> types.SimpleNamespace:
    """Object shaped like a shared interleaved run's config.yml (block fields dormant)."""
    dataparser = types.SimpleNamespace(test_block_count=2, test_block_core_fraction=0.1,
                                       test_split_method="shared_interleaved", test_render_fraction=render_fraction)
    datamanager = types.SimpleNamespace(dataparser=dataparser)
    return types.SimpleNamespace(pipeline=types.SimpleNamespace(datamanager=datamanager))


def test_prepare_rerender_accepts_shared_runs_with_guard_and_refuses_without(tmp_path):
    """r < 1 leaves guard frames to re-render; r = 1 has none, whatever the dormant block fields say."""
    prepare_module = load_prepare_rerender_module()
    selection_path = tmp_path / "selection.json"
    prepared = prepare_module.set_render_frame_selection(_shared_trainer_config(0.1), selection_path)
    assert prepared.pipeline.datamanager.dataparser.render_frame_selection_file == selection_path
    with pytest.raises(ValueError, match="no guard frames"):
        prepare_module.set_render_frame_selection(_shared_trainer_config(1.0), selection_path)


# ---- the real dataparser ---------------------------------------------------------------------------

@pytest.mark.covers("T71")
@pytest.mark.covers("T72")
@pytest.mark.skipif(not _neurad_ready(), reason="needs torch + patched neurad-studio")
def test_real_dataparser_splits_like_the_cpu_rebuild(tmp_path):
    """The real ADDataParser with shared_interleaved gives scripts/09's rebuilt frames; val gets every held-out frame.

    Train and eval file names of the dataparser are compared with build_run_split on the same
    dataset and fraction; the dormant block fields must not change anything.
    """
    from tests.unit.test_viking_dataparser import SYNTHETIC_CAMERA, SYNTHETIC_LIDAR, _write_synthetic
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig

    _, root = _write_synthetic(tmp_path)
    train_fraction, render_fraction = 0.8, 0.5
    config_arguments = dict(data=tmp_path, session_name=root.name, cameras=(SYNTHETIC_CAMERA,),
                            lidars=(SYNTHETIC_LIDAR,), train_split_fraction_override=train_fraction,
                            test_block_count=2, test_block_core_fraction=0.1,
                            test_split_method="shared_interleaved", test_render_fraction=render_fraction,
                            eval_whole_test_blocks=True)
    parsed = {split_name: VikingHillDataParserConfig(**config_arguments).setup().get_dataparser_outputs(split_name)
              for split_name in ("train", "eval", "val")}
    expected_frames, _ = build_run_split(open_dataset_view(root), 0.0, 1.0, train_fraction, test_block_count=2,
                                         test_block_core_fraction=0.1, test_split_method="shared_interleaved",
                                         test_render_fraction=render_fraction)
    # "val" (training's eval, eval_whole_test_blocks) covers test + guard; "eval" (renders) only test.
    record_splits_by_parser_split = {"train": (TRAIN_SPLIT_NAME,), "eval": (TEST_SPLIT_NAME,),
                                     "val": (TEST_SPLIT_NAME, GUARD_SPLIT_NAME)}
    for parser_split, record_splits in record_splits_by_parser_split.items():
        outputs = parsed[parser_split]
        camera_names = sorted(Path(image_path).name for image_path in outputs.image_filenames)
        expected_camera = sorted(Path(frame.file).name for frame in expected_frames
                                 if frame.kind == "camera" and frame.split in record_splits)
        assert camera_names == expected_camera
        expected_lidar_count = sum(frame.kind == "lidar" and frame.split in record_splits for frame in expected_frames)
        assert len(outputs.metadata["lidars"]) == expected_lidar_count
