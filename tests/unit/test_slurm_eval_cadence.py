"""Eval cadence reaches the Slurm job: profile minimum_eval_points -> submit_run.py -> splatad_run.sbatch.

The monitor needs at least ``minimum_eval_points`` whole-stretch evals per run; the interval is
derived by T63 from the profile's iteration count, exported as STEPS_PER_EVAL_ALL_IMAGES and passed
to ns-train together with the whole-stretch eval flag of the dataparser.
"""

import inspect

import pytest

from tests.unit.test_slurm_submit import PROJECT_ROOT, build_example_job_environment, load_submit_module
from viking_splatad.eval.training_curves import count_eval_points, eval_interval_for_minimum_points

SBATCH_TEXT_PATH = PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch"


def test_every_profile_reaches_minimum_eval_points():
    """Each shipped profile declares an int minimum >= 50 and its derived interval really gives it."""
    submit_module = load_submit_module()
    for profile_name, profile_settings in submit_module.load_slurm_config()["profiles"].items():
        minimum_points = profile_settings["minimum_eval_points"]
        assert isinstance(minimum_points, int) and minimum_points >= 50, profile_name
        eval_interval = eval_interval_for_minimum_points(profile_settings["max_num_iterations"], minimum_points)
        assert count_eval_points(profile_settings["max_num_iterations"], eval_interval) >= minimum_points


def test_job_environment_exports_eval_cadence():
    """N = 30001, P = 50 -> k = 30000 // 50 = 600, exported next to the minimum."""
    submit_module = load_submit_module()
    profile_settings = {"max_num_iterations": 30001, "minimum_eval_points": 50, "steps_per_save": 1,
                        "seed": 1, "render_pose_source": "test", "render_downscale_factor": 1}
    minimum_points, eval_interval = submit_module.profile_eval_cadence(profile_settings)
    assert (minimum_points, eval_interval) == (50, 30000 // 50)
    env_marker = {"env_prefix": "/w/stage/envs/splatad-x"}
    job_environment = submit_module.build_job_environment(
        profile_settings, "s", "run", "a" * 40, "/w/stage", env_marker,
        minimum_eval_points=minimum_points, steps_per_eval_all_images=eval_interval)
    assert job_environment["STEPS_PER_EVAL_ALL_IMAGES"] == "600"
    assert job_environment["MINIMUM_EVAL_POINTS"] == "50"
    # Without the kwargs the variables are left out (sbatch then refuses to start).
    bare_environment = build_example_job_environment(submit_module, profile_settings)
    assert "STEPS_PER_EVAL_ALL_IMAGES" not in bare_environment


@pytest.mark.parametrize("bad_profile_fields", [
    {},                                                       # minimum missing
    {"minimum_eval_points": True},                            # bool refused
    {"minimum_eval_points": "50"},                            # string refused
    {"minimum_eval_points": 50, "max_num_iterations": 40},    # N - 1 < P: T63 refuses
])
def test_profile_eval_cadence_refuses_missing_or_short(bad_profile_fields):
    """Missing, non-int or unreachable minimums raise ValueError instead of defaulting."""
    submit_module = load_submit_module()
    profile_settings = {"max_num_iterations": 30001, **bad_profile_fields}
    with pytest.raises(ValueError):
        submit_module.profile_eval_cadence(profile_settings)


def test_submit_passes_eval_cadence_to_the_job_environment():
    """The submit path derives the cadence and passes both kwargs (calling it needs ssh)."""
    submit_source = inspect.getsource(load_submit_module())
    assert "profile_eval_cadence(profile_settings)" in submit_source
    assert "minimum_eval_points=" in submit_source
    assert "steps_per_eval_all_images=" in submit_source


def test_sbatch_passes_eval_flags():
    """ns-train gets the interval before viking-data and the whole-stretch flag after it."""
    sbatch_lines = SBATCH_TEXT_PATH.read_text().splitlines()
    dataparser_line = next(index for index, line in enumerate(sbatch_lines) if line.lstrip().startswith("viking-data --data"))
    interval_line = next(index for index, line in enumerate(sbatch_lines)
                         if '--steps-per-eval-all-images "$STEPS_PER_EVAL_ALL_IMAGES"' in line)
    flag_line = next(index for index, line in enumerate(sbatch_lines)
                     if "--eval-whole-test-blocks True" in line and not line.lstrip().startswith("#"))
    assert interval_line < dataparser_line < flag_line
    sbatch_text = SBATCH_TEXT_PATH.read_text()
    # Variable names appear in the required loop and the job_info loop (plus the header and flag).
    for variable_name in ("MINIMUM_EVAL_POINTS", "STEPS_PER_EVAL_ALL_IMAGES"):
        loop_hits = [line for line in sbatch_lines if variable_name in line and line.rstrip().endswith("; do")]
        assert len(loop_hits) >= 2, variable_name
        assert sbatch_text.count(variable_name) >= 3
