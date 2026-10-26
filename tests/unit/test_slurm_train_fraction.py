"""The real train fraction reaches config.yml, and only finished runs get a split record.

nerfstudio saves config.yml before the dataparser runs, so without --train-split-fraction it records
the class default (0.5) while the dataparser splits with the manifest's session.train_fraction.
submit_run.py reads that fraction from the dataset and splatad_run.sbatch passes it to viking-data.
scripts/09_extract_run_split.py runs only as the job's last step, after every render.
"""

import json

import pytest

from tests.unit.test_slurm_submit import PROJECT_ROOT, load_submit_module

SBATCH_PATH = PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch"
SPLIT_SCRIPT_CALL = 'python "$CODE_DIR/scripts/09_extract_run_split.py"'


def write_manifest_fraction(intermediate_root, dataset_name: str, train_fraction) -> None:
    """Write a dataset directory whose manifest holds only session.train_fraction."""
    dataset_dir = intermediate_root / dataset_name
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps({"session": {"train_fraction": train_fraction}}))


def test_train_fraction_comes_from_the_dataset_manifest(tmp_path):
    """The manifest's value is returned as a float, unchanged."""
    write_manifest_fraction(tmp_path, "smoke", 0.9)
    assert load_submit_module().dataset_train_fraction("smoke", tmp_path) == 0.9


@pytest.mark.parametrize("bad_fraction", [0, 1.5, -0.1, "0.9", True, None])
def test_train_fraction_outside_zero_one_or_not_a_number_is_refused(tmp_path, bad_fraction):
    """(0, 1] is the only meaningful range; strings, bools and null are refused rather than coerced."""
    write_manifest_fraction(tmp_path, "smoke", bad_fraction)
    with pytest.raises(ValueError, match="train_fraction"):
        load_submit_module().dataset_train_fraction("smoke", tmp_path)


def test_train_fraction_without_a_local_dataset_is_an_error(tmp_path):
    """submit must not guess the fraction when the dataset is not on this machine."""
    with pytest.raises(FileNotFoundError):
        load_submit_module().dataset_train_fraction("absent", tmp_path)


def test_job_environment_exports_the_fraction_with_every_digit():
    """repr() keeps the float exact, so config.yml, job_info.txt and the manifest hold the same value."""
    submit_module = load_submit_module()
    profile_settings = {"seed": 1, "max_num_iterations": 1, "steps_per_save": 1,
                        "render_pose_source": "test", "render_downscale_factor": 1}
    # 1/3 has no short decimal form, so any rounding on the way would change the value.
    exact_fraction = 1 / 3
    job_environment = submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", {"env_prefix": "/w/stage/envs/x"},
        train_split_fraction=exact_fraction)
    assert float(job_environment["TRAIN_SPLIT_FRACTION"]) == exact_fraction


def test_sbatch_requires_records_and_passes_the_fraction():
    """Static check: required, written to job_info.txt, given to viking-data as --train-split-fraction."""
    sbatch_text = SBATCH_PATH.read_text()
    required_block = sbatch_text[sbatch_text.index("for required_variable in"):sbatch_text.index("; do")]
    assert "TRAIN_SPLIT_FRACTION" in required_block
    job_info_block = sbatch_text[sbatch_text.index("for exported_variable in"):]
    assert "TRAIN_SPLIT_FRACTION" in job_info_block[:job_info_block.index("; do")]
    train_command = sbatch_text[sbatch_text.index("ns-train splatad"):sbatch_text.index("TRAIN_END_SECONDS=")]
    assert '--train-split-fraction "$TRAIN_SPLIT_FRACTION"' in train_command


def test_sbatch_writes_the_split_record_only_after_every_render():
    """Static check: script 09 runs once, after the lidar render and its elapsed time, before end_utc."""
    sbatch_text = SBATCH_PATH.read_text()
    assert sbatch_text.count(SPLIT_SCRIPT_CALL) == 1
    split_position = sbatch_text.index(SPLIT_SCRIPT_CALL)
    assert sbatch_text.index("08_render_lidar_pointcloud.py") < split_position
    assert sbatch_text.index('echo "lidar_render_elapsed_s:') < split_position
    assert split_position < sbatch_text.index('echo "end_utc:')
