"""Tests for the test block cores (T62) and their path through the run pipeline.

Only the central fraction c of each T60 test block stays test; the rest of the block is guard
(neither trained on nor rendered). Covered here: the transform (known answers), the CPU rebuild
in scripts/09 (guard frames, core bounds, agreement checks), the session config -> submit_run.py
-> splatad_run.sbatch plumbing, the profiles rendering test frames only, and (when torch +
patched neurad-studio are importable) the real dataparser agreeing with the rebuild.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.unit.test_block_split import SBATCH_PATH, SESSION_CONFIG_DIRECTORY, _neurad_ready
from tests.unit.test_run_split import run_script, write_renders, write_synthetic_dataset
from tests.unit.test_slurm_submit import load_submit_module
from viking_splatad.dataset.run_split import (
    build_run_split,
    build_run_split_with_blocks,
    build_run_split_with_cores,
    middle_block_train_eval_indices,
    restrict_test_to_block_cores,
    split_digest,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view


@pytest.mark.covers("T62")
def test_half_cores_known_answer_with_half_open_edges():
    """Camera at integer times 0..12, f = 0.5, K = 2, c = 0.5.

    T60 (see test_block_split): blocks [2.5, 5.5) and [6.5, 9.5), test camera times 3, 4, 5, 7, 8, 9.
    Cores: centre m, half length c * L / 2 = 0.5 * 3 / 2 = 0.75, so [3.25, 4.75) and [7.25, 8.75).
    Test keeps 4 and 8; guard is 3, 5, 7, 9; train is untouched.
    """
    camera_times = np.arange(13, dtype=np.float64)
    positions, blocks = middle_block_train_eval_indices({"cam": camera_times}, 0.5, 2)
    core_positions, guard_positions, cores = restrict_test_to_block_cores({"cam": camera_times}, positions, blocks, 0.5)
    assert cores == [(3.25, 4.75), (7.25, 8.75)]
    assert camera_times[core_positions["cam"][1]].tolist() == [4.0, 8.0]
    assert camera_times[guard_positions["cam"]].tolist() == [3.0, 5.0, 7.0, 9.0]
    np.testing.assert_array_equal(core_positions["cam"][0], positions["cam"][0])


@pytest.mark.covers("T62")
def test_core_edges_are_half_open():
    """Times on a core's start edge are test, on its end edge guard (same convention as T60 blocks).

    One block [0, 4) (given directly), c = 0.5: core [1, 3). Times 0, 1, 2, 3: 1 and 2 test, 0 and 3 guard.
    """
    times = np.array([0.0, 1.0, 2.0, 3.0])
    positions = {"only": (np.empty(0, dtype=np.int64), np.arange(4, dtype=np.int64))}
    core_positions, guard_positions, cores = restrict_test_to_block_cores({"only": times}, positions, [(0.0, 4.0)], 0.5)
    assert cores == [(1.0, 3.0)]
    assert core_positions["only"][1].tolist() == [1, 2]
    assert guard_positions["only"].tolist() == [0, 3]


@pytest.mark.covers("T62")
def test_whole_block_core_changes_nothing():
    """c = 1: cores are the blocks themselves (not recomputed), every test frame stays, no guard."""
    sensor_times = {"cam": np.arange(150) * 0.1, "lidar": np.arange(150) * 0.1 + 0.03}
    positions, blocks = middle_block_train_eval_indices(sensor_times, 0.9, 2)
    core_positions, guard_positions, cores = restrict_test_to_block_cores(sensor_times, positions, blocks, 1.0)
    assert cores == blocks
    for sensor_name in sensor_times:
        np.testing.assert_array_equal(core_positions[sensor_name][0], positions[sensor_name][0])
        np.testing.assert_array_equal(core_positions[sensor_name][1], positions[sensor_name][1])
        assert guard_positions[sensor_name].size == 0


@pytest.mark.covers("T62")
def test_cores_hold_the_core_fraction_of_each_sensors_test_frames():
    """Two 10 Hz sensors over 241 s (the May recording's shape), f = 0.9, K = 2, c = 0.5.

    Test + guard is exactly T60's test set; every kept test frame lies in a core and every guard
    frame outside all cores; the kept share is c of the test frames up to one frame per core edge.
    """
    frame_period, frame_count = 0.1, 2410
    sensor_times = {"cam": np.arange(frame_count) * frame_period,
                    "lidar": np.arange(frame_count) * frame_period + 0.03}
    core_fraction, block_count = 0.5, 2
    positions, blocks = middle_block_train_eval_indices(sensor_times, 0.9, block_count)
    core_positions, guard_positions, cores = restrict_test_to_block_cores(sensor_times, positions, blocks, core_fraction)
    for sensor_name, times in sensor_times.items():
        kept_test = core_positions[sensor_name][1]
        guard = guard_positions[sensor_name]
        assert sorted(kept_test.tolist() + guard.tolist()) == positions[sensor_name][1].tolist()

        def inside_any_core(frame_time):
            """True if the time lies in a half-open core."""
            return any(core_start <= frame_time < core_end for core_start, core_end in cores)
        assert all(inside_any_core(frame_time) for frame_time in times[kept_test])
        assert not any(inside_any_core(frame_time) for frame_time in times[guard])
        # Two edges per core can each gain or lose one frame.
        assert abs(len(kept_test) - core_fraction * len(positions[sensor_name][1])) <= 2 * block_count


@pytest.mark.covers("T62")
@pytest.mark.parametrize("bad_fraction", [0.0, -0.5, 1.5])
def test_core_fraction_outside_zero_one_is_refused(bad_fraction):
    """c = 0 leaves no test frame; c > 1 would reach past the block into train frames."""
    times = np.arange(13, dtype=np.float64)
    positions, blocks = middle_block_train_eval_indices({"cam": times}, 0.5, 2)
    with pytest.raises(ValueError, match="test_block_core_fraction"):
        restrict_test_to_block_cores({"cam": times}, positions, blocks, bad_fraction)


@pytest.mark.covers("T58", "T60", "T62")
def test_build_run_split_marks_guard_frames(tmp_path):
    """Camera 0..30, lidar 0.5..29.5, whole window, f = 0.8, K = 2, c = 0.5.

    Span [0, 30]: L = 3, blocks [8.5, 11.5) and [18.5, 21.5); cores of half length 0.75:
    [9.25, 10.75) and [19.25, 20.75).
    Camera: blocks hold 9, 10, 11 and 19, 20, 21 -> test 10, 20; guard 9, 11, 19, 21.
    Lidar: blocks hold 8.5, 9.5, 10.5 and 18.5, 19.5, 20.5 (11.5 and 21.5 sit on block ends: train)
    -> test 9.5, 10.5, 19.5, 20.5; guard 8.5, 18.5.
    """
    dataset_dir = tmp_path / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(31, dtype=np.float64),
                                          ("lidar", "lidar"): np.arange(30, dtype=np.float64) + 0.5}, 0.8)
    frames, _, blocks, cores = build_run_split_with_cores(open_dataset_view(dataset_dir), 0.0, 1.0, 0.8,
                                                          test_block_count=2, test_block_core_fraction=0.5)
    np.testing.assert_allclose(blocks, [(8.5, 11.5), (18.5, 21.5)], atol=1e-12)
    np.testing.assert_allclose(cores, [(9.25, 10.75), (19.25, 20.75)], atol=1e-12)

    def times_of(sensor_name, split_name):
        """Sorted times of one sensor's frames in one split."""
        return sorted(frame.time_s for frame in frames if frame.sensor == sensor_name and frame.split == split_name)
    assert times_of("cam", "test") == [10.0, 20.0]
    assert times_of("cam", "guard") == [9.0, 11.0, 19.0, 21.0]
    assert times_of("lidar", "test") == [9.5, 10.5, 19.5, 20.5]
    assert times_of("lidar", "guard") == [8.5, 18.5]
    # Train is exactly the T60 train set: guard frames come only out of the test set.
    whole_block_frames, _, _ = build_run_split_with_blocks(open_dataset_view(dataset_dir), 0.0, 1.0, 0.8,
                                                           test_block_count=2)
    for sensor_name in ("cam", "lidar"):
        assert times_of(sensor_name, "train") == sorted(
            frame.time_s for frame in whole_block_frames if frame.sensor == sensor_name and frame.split == "train")
    # Every frame appears once, in exactly one of the three splits.
    assert len(frames) == 31 + 30


def test_build_run_split_refuses_cores_without_blocks(tmp_path):
    """c < 1 with the LINSPACE split (no blocks) is a config mistake, refused rather than ignored."""
    dataset_dir = tmp_path / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(11, dtype=np.float64)}, 0.8)
    with pytest.raises(ValueError, match="test_block_core_fraction"):
        build_run_split(open_dataset_view(dataset_dir), 0.0, 1.0, 0.8, test_block_core_fraction=0.5)


def make_core_run(tmp_path: Path, config_core_fraction, job_info_core_fraction) -> tuple:
    """A completed run with test-only renders of the T62 cores (RENDER_POSE_SOURCE: test).

    Args:
        tmp_path: pytest temporary directory.
        config_core_fraction: ``test_block_core_fraction`` saved in config.yml.
        job_info_core_fraction: TEST_BLOCK_CORE_FRACTION written to job_info.txt, or None to leave it out.

    Returns:
        (run_dir, dataset_dir, expected frames).
    """
    train_fraction, block_count = 0.8, 2
    dataset_dir = tmp_path / "data" / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(31, dtype=np.float64),
                                          ("lidar", "lidar"): np.arange(30, dtype=np.float64) + 0.5}, train_fraction)
    run_dir = tmp_path / "runs" / "run-c"
    run_dir.mkdir(parents=True)
    core_line = "" if job_info_core_fraction is None else f"TEST_BLOCK_CORE_FRACTION: {job_info_core_fraction!r}\n"
    (run_dir / "job_info.txt").write_text(
        "RUN_ID: run-c\nSESSION: synthetic\nGIT_SHA: abc\nTRAIN_SEED: 7\nRENDER_POSE_SOURCE: test\n"
        "DATASET_START_FRACTION: 0.0\nDATASET_END_FRACTION: 1.0\nADD_MISSING_POINTS: True\n"
        f"TRAIN_SPLIT_FRACTION: {train_fraction!r}\nTEST_BLOCK_COUNT: {block_count}\n{core_line}"
        "lidar_render_elapsed_s: 9\n"
    )
    config_dir = run_dir / "train" / "synthetic" / "splatad" / "run-c"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yml").write_text(
        f"  train_split_fraction: {train_fraction}\n  max_eval_frames: null\n  test_block_count: {block_count}\n"
        f"  test_block_core_fraction: {config_core_fraction}\n")
    expected_frames, _, _, _ = build_run_split_with_cores(
        open_dataset_view(dataset_dir), 0.0, 1.0, train_fraction, test_block_count=block_count,
        test_block_core_fraction=config_core_fraction)
    # Test renders only, as the profiles now ask: no renders/train/ folders at all.
    write_renders(run_dir, [frame for frame in expected_frames if frame.split == "test"])
    return run_dir, dataset_dir, expected_frames


def test_script_documents_a_test_only_core_run(tmp_path):
    """Test-only renders of the cores -> exit 0; the record holds c, the core bounds and guard counts."""
    run_dir, dataset_dir, expected_frames = make_core_run(tmp_path, 0.5, 0.5)
    assert not (run_dir / "renders" / "train").exists()
    assert run_script(run_dir, dataset_dir) == 0
    record = json.loads((run_dir / "split.json").read_text())
    assert record["membership_sha256"] == split_digest(expected_frames)
    assert record["selection"]["test_block_core_fraction"] == 0.5
    np.testing.assert_allclose(record["selection"]["test_cores_s"], [[9.25, 10.75], [19.25, 20.75]], atol=1e-12)
    # Counts from test_build_run_split_marks_guard_frames' known answer.
    assert (record["counts"]["cam"]["test"], record["counts"]["cam"]["guard"]) == (2, 4)
    assert (record["counts"]["lidar"]["test"], record["counts"]["lidar"]["guard"]) == (4, 2)
    assert set(record["render_check"]) == {"camera/test", "lidar/test"}


def test_script_refuses_disagreeing_core_fractions(tmp_path):
    """job_info.txt says 0.25 while config.yml (what the dataparser used) says 0.5 -> exit 1, no record."""
    run_dir, dataset_dir, _ = make_core_run(tmp_path, 0.5, 0.25)
    assert run_script(run_dir, dataset_dir) == 1
    assert not (run_dir / "split.json").exists()


def test_script_refuses_whole_block_renders_for_a_core_run(tmp_path):
    """Renders of whole blocks while config.yml says c = 0.5 -> exit 1 (guard frames were rendered)."""
    run_dir, dataset_dir, _ = make_core_run(tmp_path, 0.5, 0.5)
    whole_block_frames, _, _ = build_run_split_with_blocks(open_dataset_view(dataset_dir), 0.0, 1.0, 0.8,
                                                           test_block_count=2)
    write_renders(run_dir, [frame for frame in whole_block_frames if frame.split == "test"])
    assert run_script(run_dir, dataset_dir) == 1


def write_session_core_fraction(config_directory: Path, session_name: str, core_fraction) -> None:
    """Write a minimal session config holding only split.test_block_core_fraction."""
    config_directory.mkdir(parents=True, exist_ok=True)
    (config_directory / f"{session_name}.yaml").write_text(
        yaml.safe_dump({"split": {"test_block_core_fraction": core_fraction}}))


@pytest.mark.parametrize("bad_fraction", [0, -0.5, 1.5, "0.5", True, None])
def test_bad_core_fraction_in_the_session_config_is_refused(tmp_path, bad_fraction):
    """Only a YAML number in (0, 1] is accepted; nothing is coerced."""
    write_session_core_fraction(tmp_path / "sessions", "s", bad_fraction)
    with pytest.raises(ValueError, match="test_block_core_fraction"):
        load_submit_module().session_test_block_core_fraction("s", tmp_path / "intermediate", tmp_path / "sessions")


def test_may_and_june_sessions_render_the_middle_half_of_each_block():
    """Both sessions set c = 0.5 (user's choice 2026-10-26), so May, June B and R test the same way."""
    submit_module = load_submit_module()
    core_fractions = {session_path.stem: submit_module.session_test_block_core_fraction(
        session_path.stem, Path("/absent"), SESSION_CONFIG_DIRECTORY)
        for session_path in SESSION_CONFIG_DIRECTORY.glob("*.yaml")}
    assert core_fractions["may_short_grass"] == core_fractions["june_tall_grass"] == 0.5


def test_job_environment_exports_the_core_fraction():
    """TEST_BLOCK_CORE_FRACTION is the float's repr (every digit); left out when not given."""
    submit_module = load_submit_module()
    profile_settings = {"seed": 1, "max_num_iterations": 1, "steps_per_save": 1,
                        "render_pose_source": "test", "render_downscale_factor": 1}
    job_environment = submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", {"env_prefix": "/w/stage/envs/x"},
        test_block_core_fraction=0.5)
    assert job_environment["TEST_BLOCK_CORE_FRACTION"] == "0.5"
    assert "TEST_BLOCK_CORE_FRACTION" not in submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", {"env_prefix": "/w/stage/envs/x"})


def test_sbatch_requires_records_and_passes_the_core_fraction():
    """Static check: required, written to job_info.txt, given to viking-data as --test-block-core-fraction."""
    sbatch_text = SBATCH_PATH.read_text()
    required_block = sbatch_text[sbatch_text.index("for required_variable in"):sbatch_text.index("; do")]
    assert "TEST_BLOCK_CORE_FRACTION" in required_block
    job_info_block = sbatch_text[sbatch_text.index("for exported_variable in"):]
    assert "TEST_BLOCK_CORE_FRACTION" in job_info_block[:job_info_block.index("; do")]
    train_command = sbatch_text[sbatch_text.index("ns-train splatad"):sbatch_text.index("TRAIN_END_SECONDS=")]
    assert '--test-block-core-fraction "$TEST_BLOCK_CORE_FRACTION"' in train_command


def test_profiles_render_test_frames_only():
    """Every Slurm profile renders the test split only: training views are never re-rendered."""
    profiles = load_submit_module().load_slurm_config()["profiles"]
    assert {"smoke", "may_scene_a"} <= set(profiles)
    assert {profile_name: settings["render_pose_source"] for profile_name, settings in profiles.items()} == {
        profile_name: "test" for profile_name in profiles}


@pytest.mark.covers("T62")
@pytest.mark.skipif(not _neurad_ready(), reason="needs torch + patched neurad-studio")
def test_real_dataparser_eval_split_is_the_cores(tmp_path):
    """The real ADDataParser with test_block_core_fraction gives exactly scripts/09's rebuilt frames."""
    from tests.unit.test_viking_dataparser import SYNTHETIC_CAMERA, SYNTHETIC_LIDAR, _write_synthetic
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig

    _, root = _write_synthetic(tmp_path)
    train_fraction, block_count, core_fraction = 0.8, 2, 0.5
    config_arguments = dict(data=tmp_path, session_name=root.name, cameras=(SYNTHETIC_CAMERA,),
                            lidars=(SYNTHETIC_LIDAR,), train_split_fraction_override=train_fraction,
                            test_block_count=block_count, test_block_core_fraction=core_fraction)
    parsed = {split_name: VikingHillDataParserConfig(**config_arguments).setup().get_dataparser_outputs(split_name)
              for split_name in ("train", "eval")}
    expected_frames, _, _, _ = build_run_split_with_cores(
        open_dataset_view(root), 0.0, 1.0, train_fraction, test_block_count=block_count,
        test_block_core_fraction=core_fraction)
    for parser_split, record_split in (("train", "train"), ("eval", "test")):
        outputs = parsed[parser_split]
        camera_names = sorted(Path(image_path).name for image_path in outputs.image_filenames)
        expected_camera = sorted(Path(frame.file).name for frame in expected_frames
                                 if frame.kind == "camera" and frame.split == record_split)
        assert camera_names == expected_camera
        expected_lidar_count = sum(frame.kind == "lidar" and frame.split == record_split for frame in expected_frames)
        assert len(outputs.metadata["lidars"]) == expected_lidar_count
