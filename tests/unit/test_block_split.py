"""Tests for the middle-block train/test split (T60) and its path through the run pipeline.

The test set is K contiguous time blocks shared by camera and lidar, centred at k/(K+1) of the
window, each (1 - f)/K of it long. Covered here: the transform itself (known answers), the CPU
rebuild in scripts/09, the session config -> submit_run.py -> splatad_run.sbatch plumbing, and
(when torch + patched neurad-studio are importable) the real dataparser agreeing with the rebuild.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.unit.test_run_split import run_script, write_renders, write_synthetic_dataset
from tests.unit.test_slurm_submit import PROJECT_ROOT, load_submit_module
from viking_splatad.dataset.run_split import (
    build_run_split,
    build_run_split_with_blocks,
    middle_block_train_eval_indices,
    split_digest,
)
from viking_splatad.splatad.viking_conversions import open_dataset_view

SBATCH_PATH = PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch"
SESSION_CONFIG_DIRECTORY = PROJECT_ROOT / "configs" / "sessions"


@pytest.mark.covers("T60")
def test_two_blocks_known_answer_with_half_open_edges():
    """Camera at integer times 0..12, lidar at 0.5..11.5, f = 0.5, K = 2.

    Span over both sensors: t_lo = 0, D = 12. Block length L = D(1-f)/K = 12 * 0.5 / 2 = 3;
    centres D*1/3 = 4 and D*2/3 = 8, so blocks [2.5, 5.5) and [6.5, 9.5). The lidar frames at
    2.5 and 6.5 sit on a start edge (test); those at 5.5 and 9.5 sit on an end edge (train).
    """
    camera_times = np.arange(13, dtype=np.float64)
    lidar_times = np.arange(12, dtype=np.float64) + 0.5
    positions, blocks = middle_block_train_eval_indices({"cam": camera_times, "lidar": lidar_times}, 0.5, 2)
    assert blocks == [(2.5, 5.5), (6.5, 9.5)]
    camera_train, camera_test = positions["cam"]
    lidar_train, lidar_test = positions["lidar"]
    assert camera_times[camera_test].tolist() == [3.0, 4.0, 5.0, 7.0, 8.0, 9.0]
    assert lidar_times[lidar_test].tolist() == [2.5, 3.5, 4.5, 6.5, 7.5, 8.5]
    # Train is the complement, sorted.
    assert camera_train.tolist() == sorted(set(range(13)) - set(camera_test.tolist()))
    assert lidar_train.tolist() == sorted(set(range(12)) - set(lidar_test.tolist()))


@pytest.mark.covers("T60")
def test_blocks_are_shared_by_sensors_and_hold_the_test_fraction():
    """Two 10 Hz sensors offset by 30 ms over 15 s, f = 0.9, K = 2 (the smoke window's shape).

    Every test frame of either sensor lies inside a block, every train frame outside all blocks,
    and each sensor's test share is (1 - f) up to one frame per block edge: |test - n(1-f)| <= K.
    """
    frame_period = 0.1
    camera_times = np.arange(150) * frame_period
    lidar_times = np.arange(150) * frame_period + 0.03
    train_fraction, block_count = 0.9, 2
    positions, blocks = middle_block_train_eval_indices({"cam": camera_times, "lidar": lidar_times},
                                                        train_fraction, block_count)
    assert len(blocks) == block_count
    for sensor_times, (train_positions, test_positions) in ((camera_times, positions["cam"]),
                                                            (lidar_times, positions["lidar"])):
        def inside_any_block(frame_time):
            """True if the time lies in a half-open block."""
            return any(block_start <= frame_time < block_end for block_start, block_end in blocks)
        assert all(inside_any_block(frame_time) for frame_time in sensor_times[test_positions])
        assert not any(inside_any_block(frame_time) for frame_time in sensor_times[train_positions])
        assert abs(len(test_positions) - len(sensor_times) * (1 - train_fraction)) <= block_count


@pytest.mark.covers("T60")
def test_blocks_are_centred_at_thirds_and_separated_by_train_time():
    """Bounds from first principles on an asymmetric span [10, 40]: D = 30, f = 0.8, K = 2.

    L = 30 * 0.2 / 2 = 3; centres 10 + 30/3 = 20 and 10 + 60/3 = 30; so [18.5, 21.5), [28.5, 31.5).
    """
    _, blocks = middle_block_train_eval_indices({"only": np.array([10.0, 40.0])}, 0.8, 2)
    np.testing.assert_allclose(blocks, [(18.5, 21.5), (28.5, 31.5)], rtol=0, atol=1e-12)


@pytest.mark.covers("T60")
@pytest.mark.parametrize("block_count, train_fraction", [(0, 0.9), (True, 0.9), (2.0, 0.9), (2, 1 / 3), (2, 0.2),
                                                         (1, 0.5), (2, 1.5), (2, 0.0)])
def test_invalid_block_count_or_overlapping_blocks_are_refused(block_count, train_fraction):
    """K must be a positive int, and f must exceed 1/(K+1) or adjacent blocks touch or overlap."""
    with pytest.raises(ValueError):
        middle_block_train_eval_indices({"cam": np.arange(10.0)}, train_fraction, block_count)


@pytest.mark.covers("T60")
def test_fraction_one_and_empty_input_follow_neurads_conventions():
    """f = 1.0: train and test are both every frame (neurad's viewing-only case); no frames: all empty."""
    positions, blocks = middle_block_train_eval_indices({"cam": np.arange(4.0), "lidar": np.empty(0)}, 1.0, 2)
    assert blocks == []
    assert positions["cam"][0].tolist() == [0, 1, 2, 3] and positions["cam"][1].tolist() == [0, 1, 2, 3]
    assert positions["lidar"][0].size == 0 and positions["lidar"][1].size == 0
    empty_positions, empty_blocks = middle_block_train_eval_indices({"cam": np.empty(0)}, 0.9, 2)
    assert empty_blocks == [] and empty_positions["cam"][1].size == 0


@pytest.mark.covers("T58", "T60")
def test_build_run_split_places_blocks_inside_the_run_window(tmp_path):
    """Camera 0..20, lidar 0.5..19.5, window fractions 0.5..1.0, f = 0.5, K = 2.

    Window (T58): t_min = 0, D = 20 -> [10, 20]; kept camera 10..20, lidar 10.5..19.5. Over the
    kept times t_lo = 10, D = 10, L = 10 * 0.5 / 2 = 2.5, centres 10 + 10/3 and 10 + 20/3:
    blocks [12.0833.., 14.5833..) and [15.4166.., 17.9166..).
    """
    dataset_dir = tmp_path / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(21, dtype=np.float64),
                                          ("lidar", "lidar"): np.arange(20, dtype=np.float64) + 0.5}, 0.5)
    frames, window_bounds, blocks = build_run_split_with_blocks(
        open_dataset_view(dataset_dir), 0.5, 1.0, 0.5, test_block_count=2)
    assert window_bounds == (10.0, 20.0)
    first_centre, second_centre, half_length = 10 + 10 / 3, 10 + 20 / 3, 2.5 / 2
    np.testing.assert_allclose(blocks, [(first_centre - half_length, first_centre + half_length),
                                        (second_centre - half_length, second_centre + half_length)], atol=1e-12)
    camera_test_times = sorted(frame.time_s for frame in frames if frame.sensor == "cam" and frame.split == "test")
    lidar_test_times = sorted(frame.time_s for frame in frames if frame.sensor == "lidar" and frame.split == "test")
    # Camera: 12 < 12.083 and 15 < 15.417 fall outside; 14 and 17 are inside the block ends.
    assert camera_test_times == [13.0, 14.0, 16.0, 17.0]
    # Lidar: 14.5 < 14.583 and 17.5 < 17.917 are inside; 11.5 and 18.5 are outside.
    assert lidar_test_times == [12.5, 13.5, 14.5, 15.5, 16.5, 17.5]
    # Every frame is in exactly one split, and the block split differs from the LINSPACE one.
    assert len(frames) == 11 + 10
    linspace_frames, _ = build_run_split(open_dataset_view(dataset_dir), 0.5, 1.0, 0.5)
    assert split_digest(frames) != split_digest(linspace_frames)


def make_block_run(tmp_path: Path, train_fraction: float, block_count: int, job_info_block_count) -> tuple:
    """A completed run whose dataparser used the block split (config.yml ``test_block_count``).

    Same layout as test_run_split.make_completed_run (job_info.txt, config.yml, renders named after
    the split), but the renders follow the T60 split.

    Args:
        tmp_path: pytest temporary directory.
        train_fraction: The manifest's session.train_fraction (also recorded as TRAIN_SPLIT_FRACTION).
        block_count: ``test_block_count`` saved in config.yml.
        job_info_block_count: TEST_BLOCK_COUNT written to job_info.txt, or None to leave it out.

    Returns:
        (run_dir, dataset_dir, expected frames).
    """
    dataset_dir = tmp_path / "data" / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(31, dtype=np.float64),
                                          ("lidar", "lidar"): np.arange(30, dtype=np.float64) + 0.5}, train_fraction)
    run_dir = tmp_path / "runs" / "run-b"
    run_dir.mkdir(parents=True)
    block_line = "" if job_info_block_count is None else f"TEST_BLOCK_COUNT: {job_info_block_count}\n"
    (run_dir / "job_info.txt").write_text(
        "RUN_ID: run-b\nSESSION: synthetic\nGIT_SHA: abc\nTRAIN_SEED: 7\nRENDER_POSE_SOURCE: train+test\n"
        "DATASET_START_FRACTION: 0.0\nDATASET_END_FRACTION: 1.0\nADD_MISSING_POINTS: True\n"
        f"TRAIN_SPLIT_FRACTION: {train_fraction!r}\n{block_line}lidar_render_elapsed_s: 73\n"
    )
    config_dir = run_dir / "train" / "synthetic" / "splatad" / "run-b"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yml").write_text(
        f"  train_split_fraction: {train_fraction}\n  max_eval_frames: null\n  test_block_count: {block_count}\n")
    expected_frames, _, _ = build_run_split_with_blocks(open_dataset_view(dataset_dir), 0.0, 1.0, train_fraction,
                                                        test_block_count=block_count)
    write_renders(run_dir, expected_frames)
    return run_dir, dataset_dir, expected_frames


def test_script_documents_a_block_split_run(tmp_path):
    """Matching renders -> exit 0; the record names T60, K and the block bounds, and the frames match."""
    run_dir, dataset_dir, expected_frames = make_block_run(tmp_path, 0.8, 2, 2)
    assert run_script(run_dir, dataset_dir) == 0
    record = json.loads((run_dir / "split.json").read_text())
    assert record["membership_sha256"] == split_digest(expected_frames)
    assert record["selection"]["test_block_count"] == 2
    assert "T60" in record["selection"]["algorithm"]
    # Span [0, 30]: L = 30 * 0.2 / 2 = 3, centres 10 and 20.
    np.testing.assert_allclose(record["selection"]["test_blocks_s"], [[8.5, 11.5], [18.5, 21.5]], atol=1e-12)
    assert record["schema"] == "viking-splatad/run-split/2"


def test_script_refuses_disagreeing_block_counts(tmp_path):
    """job_info.txt says 3 blocks while config.yml (what the dataparser used) says 2 -> exit 1, no record."""
    run_dir, dataset_dir, _ = make_block_run(tmp_path, 0.8, 2, 3)
    assert run_script(run_dir, dataset_dir) == 1
    assert not (run_dir / "split.json").exists()


def test_script_refuses_linspace_renders_for_a_block_run(tmp_path):
    """Renders from the LINSPACE split while config.yml says blocks -> exit 1 (the rebuild uses blocks)."""
    run_dir, dataset_dir, _ = make_block_run(tmp_path, 0.8, 2, 2)
    for rendered_file in (run_dir / "renders").rglob("*.*"):
        rendered_file.unlink()
    linspace_frames, _ = build_run_split(open_dataset_view(dataset_dir), 0.0, 1.0, 0.8)
    write_renders(run_dir, linspace_frames)
    assert run_script(run_dir, dataset_dir) == 1


def write_session_split(config_directory: Path, session_name: str, block_count) -> None:
    """Write a minimal session config holding only split.test_block_count."""
    config_directory.mkdir(parents=True, exist_ok=True)
    (config_directory / f"{session_name}.yaml").write_text(yaml.safe_dump({"split": {"test_block_count": block_count}}))


def test_block_count_comes_from_the_session_named_in_the_manifest(tmp_path):
    """A dataset named differently from its session (the smoke dataset) reads that session's value."""
    submit_module = load_submit_module()
    write_session_split(tmp_path / "sessions", "may_short_grass", 2)
    write_session_split(tmp_path / "sessions", "smoke_dataset", 5)  # same name as the dataset: ignored
    dataset_dir = tmp_path / "intermediate" / "smoke_dataset"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps({"session": {"name": "may_short_grass"}}))
    assert submit_module.session_test_block_count("smoke_dataset", tmp_path / "intermediate", tmp_path / "sessions") == 2


@pytest.mark.parametrize("bad_count", [0, -1, 2.0, "2", True, None])
def test_bad_block_count_in_the_session_config_is_refused(tmp_path, bad_count):
    """Only a positive YAML integer is accepted; nothing is coerced."""
    write_session_split(tmp_path / "sessions", "s", bad_count)
    with pytest.raises(ValueError, match="test_block_count"):
        load_submit_module().session_test_block_count("s", tmp_path / "intermediate", tmp_path / "sessions")


def test_may_and_june_sessions_use_the_same_block_split():
    """May, June B and June R must share the split rule (CLAUDE.md: same split for every run)."""
    submit_module = load_submit_module()
    block_counts = {session_path.stem: submit_module.session_test_block_count(
        session_path.stem, PROJECT_ROOT / "absent", SESSION_CONFIG_DIRECTORY)
        for session_path in SESSION_CONFIG_DIRECTORY.glob("*.yaml")}
    assert {"may_short_grass", "june_tall_grass"} <= set(block_counts)
    assert len(set(block_counts.values())) == 1


def test_job_environment_exports_the_block_count():
    """TEST_BLOCK_COUNT is the integer's decimal string, the spelling tyro parses for an int."""
    submit_module = load_submit_module()
    profile_settings = {"seed": 1, "max_num_iterations": 1, "steps_per_save": 1,
                        "render_pose_source": "test", "render_downscale_factor": 1}
    job_environment = submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", {"env_prefix": "/w/stage/envs/x"}, test_block_count=2)
    assert job_environment["TEST_BLOCK_COUNT"] == "2"
    assert "TEST_BLOCK_COUNT" not in submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", {"env_prefix": "/w/stage/envs/x"})


def test_sbatch_requires_records_and_passes_the_block_count():
    """Static check: required, written to job_info.txt, given to viking-data as --test-block-count."""
    sbatch_text = SBATCH_PATH.read_text()
    required_block = sbatch_text[sbatch_text.index("for required_variable in"):sbatch_text.index("; do")]
    assert "TEST_BLOCK_COUNT" in required_block
    job_info_block = sbatch_text[sbatch_text.index("for exported_variable in"):]
    assert "TEST_BLOCK_COUNT" in job_info_block[:job_info_block.index("; do")]
    train_command = sbatch_text[sbatch_text.index("ns-train splatad"):sbatch_text.index("TRAIN_END_SECONDS=")]
    assert '--test-block-count "$TEST_BLOCK_COUNT"' in train_command


def _neurad_ready() -> bool:
    """True when torch and the patched neurad-studio import (Docker image / cluster env only)."""
    try:
        from nerfstudio.cameras.lidars import LidarType
    except ImportError:
        return False
    return hasattr(LidarType, "OUSTER_OS0_32")


@pytest.mark.covers("T60")
@pytest.mark.skipif(not _neurad_ready(), reason="needs torch + patched neurad-studio")
def test_real_dataparser_splits_like_the_cpu_rebuild(tmp_path):
    """The real ADDataParser with test_block_count gives exactly scripts/09's rebuilt frames.

    Train and eval image/sweep file names of the dataparser are compared with
    build_run_split_with_blocks on the same dataset, window and fraction.
    """
    from tests.unit.test_viking_dataparser import SYNTHETIC_CAMERA, SYNTHETIC_LIDAR, _write_synthetic
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig

    _, root = _write_synthetic(tmp_path)
    train_fraction, block_count = 0.8, 2
    config_arguments = dict(data=tmp_path, session_name=root.name, cameras=(SYNTHETIC_CAMERA,),
                            lidars=(SYNTHETIC_LIDAR,), train_split_fraction_override=train_fraction,
                            test_block_count=block_count)
    parsed = {split_name: VikingHillDataParserConfig(**config_arguments).setup().get_dataparser_outputs(split_name)
              for split_name in ("train", "eval")}
    expected_frames, _, blocks = build_run_split_with_blocks(open_dataset_view(root), 0.0, 1.0, train_fraction,
                                                             test_block_count=block_count)
    assert len(blocks) == block_count
    for parser_split, record_split in (("train", "train"), ("eval", "test")):
        outputs = parsed[parser_split]
        camera_names = sorted(Path(image_path).name for image_path in outputs.image_filenames)
        expected_camera = sorted(Path(frame.file).name for frame in expected_frames
                                 if frame.kind == "camera" and frame.split == record_split)
        assert camera_names == expected_camera
        expected_lidar_count = sum(frame.kind == "lidar" and frame.split == record_split for frame in expected_frames)
        assert len(outputs.metadata["lidars"]) == expected_lidar_count
