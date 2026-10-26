"""add_missing_points reaches the Slurm job: session config -> submit_run.py -> splatad_run.sbatch.

The flag lives in configs/sessions/<session>.yaml (splatad.add_missing_points). A dataset can be
named differently from its session (may_chunks0_1_smoke was extracted with may_short_grass), so
the session name is read from the dataset's manifest.
"""

import json
from pathlib import Path

import pytest
import yaml

from tests.unit.test_slurm_submit import PROJECT_ROOT, load_submit_module


def write_session(config_directory: Path, session_name: str, flag) -> None:
    """Write a minimal session config holding only splatad.add_missing_points."""
    config_directory.mkdir(parents=True, exist_ok=True)
    (config_directory / f"{session_name}.yaml").write_text(yaml.safe_dump({"splatad": {"add_missing_points": flag}}))


def write_manifest(intermediate_root: Path, dataset_name: str, session_name: str) -> Path:
    """Write a dataset directory whose manifest names ``session_name``; returns the directory."""
    dataset_dir = intermediate_root / dataset_name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps({"session": {"name": session_name}}))
    return dataset_dir


def test_flag_comes_from_the_session_named_in_the_manifest(tmp_path):
    """A dataset named differently from its session reads that session's flag."""
    submit_module = load_submit_module()
    write_session(tmp_path / "sessions", "may_short_grass", True)
    write_session(tmp_path / "sessions", "smoke_dataset", False)  # same name as the dataset: must be ignored
    write_manifest(tmp_path / "intermediate", "smoke_dataset", "may_short_grass")
    assert submit_module.session_add_missing_points("smoke_dataset", tmp_path / "intermediate", tmp_path / "sessions") is True


def test_flag_without_a_local_dataset_uses_the_dataset_name_as_session(tmp_path):
    """No manifest: the dataset name is the session name."""
    submit_module = load_submit_module()
    write_session(tmp_path / "sessions", "june_tall_grass", False)
    assert submit_module.session_add_missing_points("june_tall_grass", tmp_path / "intermediate", tmp_path / "sessions") is False


@pytest.mark.parametrize("bad_flag", [None, "true", 1])
def test_flag_must_be_a_yaml_boolean(tmp_path, bad_flag):
    """Missing, string or integer values are refused rather than coerced."""
    submit_module = load_submit_module()
    write_session(tmp_path / "sessions", "s", bad_flag)
    with pytest.raises(ValueError, match="add_missing_points"):
        submit_module.session_add_missing_points("s", tmp_path / "intermediate", tmp_path / "sessions")


def test_committed_sessions_enable_missing_points_consistently():
    """May and June (hence May, B and R runs) use the same value, and it is on (Phase B)."""
    submit_module = load_submit_module()
    flags = {
        session_name: submit_module.session_add_missing_points(
            session_name, PROJECT_ROOT / "no-such-dir", PROJECT_ROOT / "configs" / "sessions")
        for session_name in ("may_short_grass", "june_tall_grass")
    }
    assert flags == {"may_short_grass": True, "june_tall_grass": True}


def test_stage_refuses_a_dataset_without_sidecar_only_when_the_flag_is_set(tmp_path):
    """The flag needs lidar_no_return/; without the flag its absence is fine."""
    submit_module = load_submit_module()
    dataset_dir = write_manifest(tmp_path, "d", "s")
    submit_module.require_no_return_sidecar(dataset_dir, add_missing_points=False)
    with pytest.raises(FileNotFoundError, match="lidar_no_return"):
        submit_module.require_no_return_sidecar(dataset_dir, add_missing_points=True)
    (dataset_dir / "lidar_no_return").mkdir()
    submit_module.require_no_return_sidecar(dataset_dir, add_missing_points=True)


@pytest.mark.parametrize("flag", [True, False])
def test_job_environment_exports_the_flag_as_a_tyro_boolean(flag):
    """ADD_MISSING_POINTS is True/False, the spelling ns-train's {True,False} option accepts."""
    submit_module = load_submit_module()
    profile_settings = {"seed": 1, "max_num_iterations": 1, "steps_per_save": 1,
                        "render_pose_source": "test", "render_downscale_factor": 1}
    job_environment = submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", {"env_prefix": "/w/stage/envs/x"},
        add_missing_points=flag)
    assert job_environment["ADD_MISSING_POINTS"] == str(flag)


def test_sbatch_requires_the_flag_and_passes_it_to_the_dataparser():
    """Static check of splatad_run.sbatch: required, recorded in job_info, given to viking-data."""
    sbatch_text = (PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch").read_text()
    required_block = sbatch_text[sbatch_text.index("for required_variable in"):sbatch_text.index("; do")]
    assert "ADD_MISSING_POINTS" in required_block
    train_command = sbatch_text[sbatch_text.index("ns-train splatad"):sbatch_text.index("TRAIN_END_SECONDS=")]
    assert '--add-missing-points "$ADD_MISSING_POINTS"' in train_command
    job_info_block = sbatch_text[sbatch_text.index("for exported_variable in"):]
    assert "ADD_MISSING_POINTS" in job_info_block[:job_info_block.index("; do")]
