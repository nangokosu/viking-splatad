"""A Slurm profile may split its run differently from the session (profile_split_fractions).

The smoke profile's 15 s window with the session's 0.9 train fraction and 0.1 block core keeps a
single test frame per sensor, so the profile sets its own ``train_fraction`` and
``test_block_core_fraction``. The train fraction differs from the manifest, so it reaches the
dataparser as ``--train-split-fraction-override`` (T27); scripts/09 then finds job_info.txt,
config.yml and the dataparser's resolution agreeing. Covered here: the resolver, the export, the
sbatch wiring, the smoke profile's values, and scripts/09 documenting an overridden run.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.unit.test_run_split import run_script, write_renders, write_synthetic_dataset
from tests.unit.test_slurm_submit import load_submit_module
from tests.unit.test_slurm_train_fraction import SBATCH_PATH
from viking_splatad.dataset.run_split import build_run_split_with_cores, split_digest
from viking_splatad.splatad.viking_conversions import open_dataset_view

# Values of the synthetic session written by write_session; distinct from every profile value
# below, so each test can tell which source a result came from.
MANIFEST_TRAIN_FRACTION = 0.9
SESSION_BLOCK_COUNT = 2
SESSION_CORE_FRACTION = 0.1


def write_session(tmp_path: Path) -> tuple:
    """A local dataset manifest plus its session config, as submit_run.py reads them.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        (intermediate root, session config directory); the dataset and session are both named "s".
    """
    intermediate_root, config_directory = tmp_path / "intermediate", tmp_path / "sessions"
    (intermediate_root / "s").mkdir(parents=True)
    (intermediate_root / "s" / "manifest.json").write_text(
        json.dumps({"session": {"name": "s", "train_fraction": MANIFEST_TRAIN_FRACTION}}))
    config_directory.mkdir()
    (config_directory / "s.yaml").write_text(yaml.safe_dump(
        {"split": {"test_block_count": SESSION_BLOCK_COUNT, "test_block_core_fraction": SESSION_CORE_FRACTION}}))
    return intermediate_root, config_directory


def test_profile_without_split_keys_uses_the_dataset_values(tmp_path):
    """No profile keys -> the manifest's fraction, no override, the session's core fraction."""
    intermediate_root, config_directory = write_session(tmp_path)
    assert load_submit_module().profile_split_fractions({}, "s", intermediate_root, config_directory) == (
        MANIFEST_TRAIN_FRACTION, None, SESSION_CORE_FRACTION)


def test_profile_split_keys_override_the_dataset_values(tmp_path):
    """Profile keys win; the profile's train fraction is also returned as the override."""
    intermediate_root, config_directory = write_session(tmp_path)
    profile_settings = {"train_fraction": 0.8, "test_block_core_fraction": 0.2}
    assert load_submit_module().profile_split_fractions(profile_settings, "s", intermediate_root,
                                                        config_directory) == (0.8, 0.8, 0.2)


def test_each_profile_split_key_overrides_on_its_own(tmp_path):
    """Setting only one key leaves the other at the dataset's value."""
    submit_module = load_submit_module()
    intermediate_root, config_directory = write_session(tmp_path)
    assert submit_module.profile_split_fractions({"test_block_core_fraction": 0.2}, "s", intermediate_root,
                                                 config_directory) == (MANIFEST_TRAIN_FRACTION, None, 0.2)
    assert submit_module.profile_split_fractions({"train_fraction": 0.8}, "s", intermediate_root,
                                                 config_directory) == (0.8, 0.8, SESSION_CORE_FRACTION)


@pytest.mark.parametrize("key", ["train_fraction", "test_block_core_fraction"])
@pytest.mark.parametrize("bad_value", [0, -0.2, 1.5, "0.8", True])
def test_bad_profile_split_values_are_refused(tmp_path, key, bad_value):
    """Only a real number in (0, 1] is accepted; strings and bools are not coerced."""
    intermediate_root, config_directory = write_session(tmp_path)
    with pytest.raises(ValueError, match=key):
        load_submit_module().profile_split_fractions({key: bad_value}, "s", intermediate_root, config_directory)


def test_profile_train_fraction_that_overlaps_the_blocks_is_refused(tmp_path):
    """K = 2 blocks stay apart only for f > 1/(K+1) = 1/3 (T60): exactly 1/3 is refused, above it passes."""
    submit_module = load_submit_module()
    intermediate_root, config_directory = write_session(tmp_path)
    overlap_limit = 1 / (SESSION_BLOCK_COUNT + 1)
    with pytest.raises(ValueError, match="1/\\(K\\+1\\)"):
        submit_module.profile_split_fractions({"train_fraction": overlap_limit}, "s", intermediate_root,
                                              config_directory)
    above_limit = (overlap_limit + 1) / 2
    assert submit_module.profile_split_fractions({"train_fraction": above_limit}, "s", intermediate_root,
                                                 config_directory)[0] == above_limit


def test_job_environment_exports_the_override_only_when_given():
    """TRAIN_SPLIT_FRACTION_OVERRIDE is the float's repr (every digit), absent when None."""
    submit_module = load_submit_module()
    profile_settings = {"seed": 1, "max_num_iterations": 1, "steps_per_save": 1,
                        "render_pose_source": "test", "render_downscale_factor": 1}
    # 1/3 has no short decimal form, so any rounding on the way would change the value.
    exact_fraction = 1 / 3
    job_environment = submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", {"env_prefix": "/w/stage/envs/x"},
        train_split_fraction=exact_fraction, train_split_fraction_override=exact_fraction)
    assert float(job_environment["TRAIN_SPLIT_FRACTION_OVERRIDE"]) == exact_fraction
    assert job_environment["TRAIN_SPLIT_FRACTION_OVERRIDE"] == job_environment["TRAIN_SPLIT_FRACTION"]
    assert "TRAIN_SPLIT_FRACTION_OVERRIDE" not in submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", {"env_prefix": "/w/stage/envs/x"},
        train_split_fraction=exact_fraction)


def test_sbatch_passes_the_override_only_when_set():
    """Static check: optional (not required), recorded when set, given to viking-data via an empty-safe array."""
    sbatch_text = SBATCH_PATH.read_text()
    required_block = sbatch_text[sbatch_text.index("for required_variable in"):sbatch_text.index("; do")]
    assert "TRAIN_SPLIT_FRACTION_OVERRIDE" not in required_block
    assert 'echo "TRAIN_SPLIT_FRACTION_OVERRIDE: $TRAIN_SPLIT_FRACTION_OVERRIDE"' in sbatch_text
    assert ('TRAIN_FRACTION_OVERRIDE_ARGS=(--train-split-fraction-override "$TRAIN_SPLIT_FRACTION_OVERRIDE")'
            in sbatch_text)
    train_command = sbatch_text[sbatch_text.index("ns-train splatad"):sbatch_text.index("TRAIN_END_SECONDS=")]
    viking_data_arguments = train_command[train_command.index("viking-data"):]
    assert '${TRAIN_FRACTION_OVERRIDE_ARGS[@]+"${TRAIN_FRACTION_OVERRIDE_ARGS[@]}"}' in viking_data_arguments


def test_smoke_profile_splits_eighty_twenty_and_renders_the_middle_fifth():
    """User's choice 2026-10-27: smoke trains on 80%, and renders the central 20% of each test block."""
    smoke_profile = load_submit_module().load_slurm_config()["profiles"]["smoke"]
    assert (smoke_profile["train_fraction"], smoke_profile["test_block_core_fraction"]) == (0.8, 0.2)


def test_smoke_profile_renders_more_than_one_frame_per_block():
    """Known answer: the smoke window's core holds more than one frame per block at the sensors' rate.

    Each block is W (1 - f) / K long and its core c times that; at the ~10 Hz rate the comment in
    configs/slurm.yaml cites (~150 frames over the window), that is the frame count below.
    """
    smoke_profile = load_submit_module().load_slurm_config()["profiles"]["smoke"]
    window_seconds = smoke_profile["window_seconds"]
    # ~150 frames over the window per the profile's own comment, i.e. a frame every 0.1 s.
    frames_per_second = 150 / window_seconds
    core_seconds = (window_seconds * (1 - smoke_profile["train_fraction"]) / SESSION_BLOCK_COUNT
                    * smoke_profile["test_block_core_fraction"])
    assert core_seconds * frames_per_second > 1


def test_script_documents_a_run_whose_profile_overrode_the_train_fraction(tmp_path):
    """Manifest 0.9, profile 0.8: job_info, config.yml and the override agree -> exit 0, source "override"."""
    profile_train_fraction, block_count, core_fraction = 0.8, 2, 0.5
    dataset_dir = tmp_path / "data" / "synthetic"
    write_synthetic_dataset(dataset_dir, {("cam", "camera"): np.arange(31, dtype=np.float64),
                                          ("lidar", "lidar"): np.arange(30, dtype=np.float64) + 0.5},
                            MANIFEST_TRAIN_FRACTION)
    run_dir = tmp_path / "runs" / "run-o"
    run_dir.mkdir(parents=True)
    # job_info.txt as splatad_run.sbatch writes it for a profile override.
    (run_dir / "job_info.txt").write_text(
        "RUN_ID: run-o\nSESSION: synthetic\nGIT_SHA: abc\nTRAIN_SEED: 7\nRENDER_POSE_SOURCE: test\n"
        "DATASET_START_FRACTION: 0.0\nDATASET_END_FRACTION: 1.0\nADD_MISSING_POINTS: True\n"
        f"TRAIN_SPLIT_FRACTION: {profile_train_fraction!r}\nTEST_BLOCK_COUNT: {block_count}\n"
        f"TEST_BLOCK_CORE_FRACTION: {core_fraction!r}\n"
        f"TRAIN_SPLIT_FRACTION_OVERRIDE: {profile_train_fraction!r}\nlidar_render_elapsed_s: 9\n")
    # config.yml as nerfstudio saves it: both CLI values, before the dataparser runs.
    config_dir = run_dir / "train" / "synthetic" / "splatad" / "run-o"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yml").write_text(
        f"  train_split_fraction: {profile_train_fraction}\n"
        f"  train_split_fraction_override: {profile_train_fraction}\n  max_eval_frames: null\n"
        f"  test_block_count: {block_count}\n  test_block_core_fraction: {core_fraction}\n")
    expected_frames, _, _, _ = build_run_split_with_cores(
        open_dataset_view(dataset_dir), 0.0, 1.0, profile_train_fraction, test_block_count=block_count,
        test_block_core_fraction=core_fraction)
    write_renders(run_dir, [frame for frame in expected_frames if frame.split == "test"])
    assert run_script(run_dir, dataset_dir) == 0
    record = json.loads((run_dir / "split.json").read_text())
    assert record["membership_sha256"] == split_digest(expected_frames)
    assert record["selection"]["train_fraction"] == profile_train_fraction
    assert record["selection"]["train_fraction_source"].startswith("override")
