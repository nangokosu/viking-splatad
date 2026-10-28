"""Renders never paint points, and a run that trained but failed later can be resumed.

Painting (SplatAD's FullImageLidarDatamanager.paint_points, run in its constructor) caches every
training image and sweep in RAM; ns-render builds the data manager twice, so job 222994 held the
training set twice and was OOM-killed. Renders turn it off: scripts/08 in its config callback,
ns-render through a config copy (scripts/12 write-run-config / write-config).

`submit_run.py resume` reruns the steps after training of such a run in its own run dir
(RESUME_AFTER_TRAINING in splatad_run.sbatch). Covered here: the config edits, the job script
round trip, the resume checks and environment, the active-job listing, and the sbatch's control
flow run under bash with stand-ins for ns-train, ns-render, nvidia-smi and python.
"""

import json
import os
import shlex
import stat
import subprocess
import types
from pathlib import Path
from typing import Dict, List

import pytest
import yaml

from tests.unit.test_render_lidar_script import RecordingStreamline, load_render_lidar_module
from tests.unit.test_render_only_job import (
    RUN_ID,
    STAGE_ROOT,
    load_prepare_rerender_module,
    make_fake_render_cluster,
    _prepare_submit_module,
    _sbatch_commands,
)
from tests.unit.test_slurm_submit import load_submit_module
from tests.unit.test_slurm_train_fraction import SBATCH_PATH

HEAD_SHA = "fedcba9876543210fedcba9876543210fedcba98"
TRAIN_SHA = "abcdef1" + "0" * 33
# A run killed in the camera render, as job 222994 left May's job_info.txt.
FAILED_IN_RENDER_JOB_INFO_TEXT = (
    f"hostname: gpunode2\nRUN_ID: {RUN_ID}\nSESSION: may_short_grass\nGIT_SHA: {TRAIN_SHA}\n"
    "train_start_epoch_s: 100\ntrain_end_epoch_s: 400\ntrain_elapsed_s: 300\n"
    "overfitting_plot: written\nrender_start_epoch_s: 450\n")
# 5 iterations: the final checkpoint is step 4 (max_num_iterations - 1).
RUN_ITERATIONS = 5


def run_exports() -> Dict[str, str]:
    """The variables a training job's job.sbatch exported (every one splatad_run.sbatch requires)."""
    return {"RUN_ID": RUN_ID, "SESSION": "may_short_grass", "STAGE_ROOT": STAGE_ROOT, "GIT_SHA": TRAIN_SHA,
            "CODE_DIR": f"{STAGE_ROOT}/code/{TRAIN_SHA}", "ENV_PREFIX": f"{STAGE_ROOT}/envs/splatad-x",
            "MAX_NUM_ITERATIONS": str(RUN_ITERATIONS), "STEPS_PER_SAVE": "1", "TRAIN_SEED": "42",
            "RENDER_POSE_SOURCE": "test", "RENDER_DOWNSCALE_FACTOR": "1", "DATASET_START_FRACTION": "0.0",
            "DATASET_END_FRACTION": "1.0", "ADD_MISSING_POINTS": "True", "TRAIN_SPLIT_FRACTION": "0.9",
            "TEST_BLOCK_COUNT": "2", "TEST_BLOCK_CORE_FRACTION": "0.1", "MINIMUM_EVAL_POINTS": "2",
            "STEPS_PER_EVAL_ALL_IMAGES": "2"}


# ---- point painting off for renders ---------------------------------------------------------------

def _trainer_config_with_painting() -> types.SimpleNamespace:
    """Object shaped like a SplatAD config.yml: painting on, plus fields that must survive."""
    dataparser = types.SimpleNamespace(test_block_count=2, test_block_core_fraction=0.1)
    datamanager = types.SimpleNamespace(paint_points=True, paint_points_topk=2, dataparser=dataparser)
    return types.SimpleNamespace(output_dir=Path("/w/stage/runs/r/train"), timestamp="r",
                                 pipeline=types.SimpleNamespace(datamanager=datamanager))


def test_disable_point_painting_turns_it_off_and_leaves_other_managers_alone():
    """paint_points True -> False; a data manager without the field does not gain one."""
    prepare_module = load_prepare_rerender_module()
    painted_config = prepare_module.disable_point_painting(_trainer_config_with_painting())
    assert painted_config.pipeline.datamanager.paint_points is False
    assert painted_config.pipeline.datamanager.paint_points_topk == 2
    plain_config = types.SimpleNamespace(pipeline=types.SimpleNamespace(datamanager=types.SimpleNamespace()))
    assert not hasattr(prepare_module.disable_point_painting(plain_config).pipeline.datamanager, "paint_points")


def test_write_run_config_turns_painting_off_and_keeps_everything_else(tmp_path):
    """The copy differs from the run's config.yml only in paint_points; the original is never edited."""
    prepare_module = load_prepare_rerender_module()
    train_config_path = tmp_path / "config.yml"
    train_config_path.write_text(yaml.dump(_trainer_config_with_painting()))
    original_text = train_config_path.read_text()
    output_path = tmp_path / "render_config" / "config.yml"
    assert prepare_module.main(["write-run-config", "--train-config", str(train_config_path),
                                "--output", str(output_path)]) == 0
    render_config = yaml.load(output_path.read_text(), Loader=yaml.Loader)
    expected_config = yaml.load(original_text, Loader=yaml.Loader)
    expected_config.pipeline.datamanager.paint_points = False
    assert yaml.dump(render_config) == yaml.dump(expected_config)
    assert train_config_path.read_text() == original_text


def test_render_only_config_also_turns_painting_off(tmp_path):
    """write-config (render-only jobs) sets the selection and turns painting off."""
    prepare_module = load_prepare_rerender_module()
    train_config_path = tmp_path / "config.yml"
    train_config_path.write_text(yaml.dump(_trainer_config_with_painting()))
    selection_path = tmp_path / "selection.json"
    selection_path.write_text("{}")
    output_path = tmp_path / "rerender-config.yml"
    assert prepare_module.main(["write-config", "--train-config", str(train_config_path),
                                "--selection", str(selection_path), "--output", str(output_path)]) == 0
    render_config = yaml.load(output_path.read_text(), Loader=yaml.Loader)
    assert render_config.pipeline.datamanager.paint_points is False
    assert render_config.pipeline.datamanager.dataparser.render_frame_selection_file == selection_path.resolve()


def test_lidar_render_config_turns_painting_off():
    """scripts/08's eval_setup callback turns painting off (and still drops the eval cap)."""
    config = _trainer_config_with_painting()
    config.pipeline.datamanager.dataparser.max_eval_frames = 7
    edited_config = load_render_lidar_module().prepare_render_config(config, RecordingStreamline())
    assert edited_config.pipeline.datamanager.paint_points is False
    assert edited_config.pipeline.datamanager.dataparser.max_eval_frames is None


# ---- resume: job script round trip, checks, environment, active listing ---------------------------

def test_job_script_exports_round_trip():
    """parse_job_script_exports inverts build_job_script, including values that need shell quoting."""
    submit_module = load_submit_module()
    job_environment = {"PLAIN": "abc", "SPACED": "a b", "QUOTED": "it's \"x\" $HOME", "EMPTY": ""}
    script_text = submit_module.build_job_script(job_environment, "/stage/code/sha/run.sbatch")
    assert submit_module.parse_job_script_exports(script_text) == job_environment


def test_job_script_exports_refuse_a_malformed_line():
    """An export line without NAME=value is an error, not a silently dropped variable."""
    with pytest.raises(ValueError, match="malformed export"):
        load_submit_module().parse_job_script_exports("#!/bin/bash\nexport NOT_AN_ASSIGNMENT\n")


@pytest.mark.parametrize("job_info_edit, message", [
    (lambda text: text + "end_utc: 2026-10-27T17:00:00Z\n", "already finished"),
    (lambda text: text.replace("train_elapsed_s: 300\n", ""), "did not finish training"),
    (lambda text: text.replace("overfitting_plot: written", "overfitting_plot: FAILED (exit 1)"), "overfitting_plot"),
])
def test_require_resumable_refuses_runs_that_are_not_resumable(job_info_edit, message):
    """Finished, untrained, or plot-less runs are refused, each with its own reason."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError, match=message):
        submit_module.require_resumable(submit_module.parse_job_info_text(job_info_edit(FAILED_IN_RENDER_JOB_INFO_TEXT)))


def test_require_resumable_accepts_a_run_killed_in_a_render():
    """Trained, plotted, no end_utc (job 222994's state): resumable."""
    submit_module = load_submit_module()
    submit_module.require_resumable(submit_module.parse_job_info_text(FAILED_IN_RENDER_JOB_INFO_TEXT))


def test_resume_environment_keeps_the_run_and_swaps_only_code_and_env():
    """Every run variable unchanged (GIT_SHA = the training commit), HEAD's code and env, plus the flag."""
    resume_environment = load_submit_module().build_resume_job_environment(
        run_exports(), STAGE_ROOT, HEAD_SHA, {"env_prefix": f"{STAGE_ROOT}/envs/splatad-y"})
    expected_environment = dict(run_exports(), CODE_DIR=f"{STAGE_ROOT}/code/{HEAD_SHA}",
                                ENV_PREFIX=f"{STAGE_ROOT}/envs/splatad-y", RESUME_AFTER_TRAINING="True",
                                RESUME_GIT_SHA=HEAD_SHA)
    assert resume_environment == expected_environment


def test_active_resume_job_marks_its_run_active():
    """<runs>/<run_id>/resume/job.sbatch in squeue -> <run_id> is active, so clean keeps it."""
    submit_module = load_submit_module()
    listing_text = f"RUN\t{RUN_ID}\nACTIVE\t{STAGE_ROOT}/runs/{RUN_ID}/resume/job.sbatch\n"
    remote_runs, active_run_ids = submit_module.parse_remote_run_listing(listing_text)
    assert RUN_ID in active_run_ids
    with pytest.raises(ValueError, match="scancel"):
        submit_module.plan_cleanup(remote_runs, active_run_ids, {}, [], [RUN_ID])


def _fake_resume_cluster(job_info_text: str, checkpoint_exists: bool, active_listing: str,
                         issued_commands: List[List[str]], written_files: Dict[str, str]):
    """make_fake_render_cluster plus the resume's own remote calls (job.sbatch, listing, checkpoint test)."""
    render_cluster = make_fake_render_cluster(job_info_text, {}, True, issued_commands, written_files)
    job_script_text = load_submit_module().build_job_script(run_exports(), "/x/splatad_run.sbatch")

    def fake_run_command(command, capture_output=True, input_text=None):
        """Answer the resume-specific commands; everything else goes to the render cluster fake."""
        remote_text = command[-1] if command[0] == "ssh" else ""
        if command[:2] == ["git", "rev-parse"]:
            issued_commands.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout=HEAD_SHA + "\n", stderr="")
        if remote_text.startswith("cat ") and remote_text.endswith("/job.sbatch"):
            issued_commands.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout=job_script_text, stderr="")
        if remote_text.startswith("bash -c") and "squeue" in remote_text:
            issued_commands.append(list(command))
            return subprocess.CompletedProcess(command, 0, stdout=active_listing, stderr="")
        if remote_text.startswith("test -f"):
            issued_commands.append(list(command))
            return subprocess.CompletedProcess(command, 0 if checkpoint_exists else 1, stdout="", stderr="")
        return render_cluster(command, capture_output, input_text)
    return fake_run_command


def test_resume_writes_the_job_script_in_resume_and_submits(monkeypatch, tmp_path):
    """End to end: final checkpoint checked, <run>/resume/job.sbatch runs HEAD's splatad_run.sbatch."""
    issued_commands: List[List[str]] = []
    written_files: Dict[str, str] = {}
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, _fake_resume_cluster(
        FAILED_IN_RENDER_JOB_INFO_TEXT, True, "", issued_commands, written_files))
    config = submit_module.load_slurm_config()
    assert submit_module.resume_run(config, RUN_ID, skip_gates=True) == "5151"
    run_dir = f"{STAGE_ROOT}/runs/{RUN_ID}"
    # Step 1: the checkpoint tested is step max_num_iterations - 1 in nerfstudio's 9-digit name.
    checkpoint_tests = [shlex.split(command[-1]) for command in issued_commands
                        if command[0] == "ssh" and command[-1].startswith("test -f")]
    assert checkpoint_tests == [["test", "-f", f"{run_dir}/train/may_short_grass/splatad/{RUN_ID}/nerfstudio_models/"
                                 f"step-{RUN_ITERATIONS - 1:09d}.ckpt"]]
    # Step 2: the job script holds the resume environment and runs HEAD's run script.
    job_script_text = written_files[f"{run_dir}/resume/job.sbatch"]
    assert submit_module.parse_job_script_exports(job_script_text) == submit_module.build_resume_job_environment(
        run_exports(), STAGE_ROOT, HEAD_SHA, {"env_prefix": f"{STAGE_ROOT}/envs/splatad-x"})
    assert job_script_text.splitlines()[-1] == f"exec bash {STAGE_ROOT}/code/{HEAD_SHA}/scripts/slurm/splatad_run.sbatch"
    # Step 3: one sbatch with render_only's limit, the log in the run dir; the ledger is untouched.
    sbatch_arguments = shlex.split(_sbatch_commands(issued_commands)[0])
    assert ["-t", config["render_only"]["time_limit"]] == sbatch_arguments[
        sbatch_arguments.index("-t"):sbatch_arguments.index("-t") + 2]
    assert f"--output={run_dir}/slurm-%j.out" in sbatch_arguments
    assert sbatch_arguments[-1] == f"{run_dir}/resume/job.sbatch"
    assert not (tmp_path / "submitted.tsv").exists()


def test_resume_refuses_without_the_final_checkpoint(monkeypatch, tmp_path):
    """No step-(N-1) checkpoint: refused before any write or sbatch."""
    issued_commands: List[List[str]] = []
    written_files: Dict[str, str] = {}
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, _fake_resume_cluster(
        FAILED_IN_RENDER_JOB_INFO_TEXT, False, "", issued_commands, written_files))
    with pytest.raises(RuntimeError, match="final checkpoint"):
        submit_module.resume_run(submit_module.load_slurm_config(), RUN_ID, skip_gates=True)
    assert not written_files and not _sbatch_commands(issued_commands)


def test_resume_refuses_a_run_with_an_active_job(monkeypatch, tmp_path):
    """A queued or running job on the run dir: refused (never two jobs writing one run)."""
    issued_commands: List[List[str]] = []
    written_files: Dict[str, str] = {}
    active_listing = f"RUN\t{RUN_ID}\nACTIVE\t{STAGE_ROOT}/runs/{RUN_ID}/job.sbatch\n"
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, _fake_resume_cluster(
        FAILED_IN_RENDER_JOB_INFO_TEXT, True, active_listing, issued_commands, written_files))
    with pytest.raises(ValueError, match="queued or running"):
        submit_module.resume_run(submit_module.load_slurm_config(), RUN_ID, skip_gates=True)
    assert not written_files and not _sbatch_commands(issued_commands)


def test_resume_subcommand_reaches_resume_run(monkeypatch):
    """`resume <run> --skip-gates` calls resume_run with them."""
    submit_module = load_submit_module()
    received_calls = []
    monkeypatch.setattr(submit_module, "resume_run",
                        lambda config, run_id, skip_gates: received_calls.append((run_id, skip_gates)))
    assert submit_module.main(["resume", RUN_ID, "--skip-gates"]) == 0
    assert received_calls == [(RUN_ID, True)]


# ---- splatad_run.sbatch run under bash with stand-in tools -----------------------------------------

# Stand-ins: each logs its name and arguments to $CALL_LOG. The python stand-in answers the GPU
# check, and writes the config copy for scripts/12 write-run-config, as the real tools would.
LOGGING_STAND_IN = '#!/bin/bash\necho "$(basename "$0") $*" >> "$CALL_LOG"\n'
PYTHON_STAND_IN = LOGGING_STAND_IN + """\
if [ "$1" = "-c" ]; then echo "gpu_check: stand-in"; exit 0; fi
if [ "${2:-}" = "write-run-config" ]; then
  mkdir -p "$(dirname "$6")"; echo "stand-in config" > "$6"
fi
exit 0
"""


def _write_executable(path: Path, text: str) -> None:
    """Write a script and make it executable."""
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def run_sbatch(tmp_path: Path, extra_environment: Dict[str, str], job_info_text: str,
               checkpoint_exists: bool) -> tuple:
    """Run splatad_run.sbatch with stand-in tools in a temporary stage root.

    Args:
        tmp_path: pytest temporary directory.
        extra_environment: Variables on top of run_exports() (e.g. the resume flag).
        job_info_text: The run's job_info.txt before the job ("" = none).
        checkpoint_exists: Whether the final checkpoint step-(N-1) is on disk.

    Returns:
        (completed process, list of logged tool calls, job_info.txt text after the job).
    """
    # Step 1: stage root, run dir, the run's config.yml (and checkpoint), environment marker.
    stage_root = tmp_path / "stage"
    environment = dict(run_exports(), STAGE_ROOT=str(stage_root), CODE_DIR=str(stage_root / "code"),
                       ENV_PREFIX=str(stage_root / "env"), **extra_environment)
    run_dir = stage_root / "runs" / RUN_ID
    train_dir = run_dir / "train" / "may_short_grass" / "splatad" / RUN_ID
    (train_dir / "nerfstudio_models").mkdir(parents=True)
    (train_dir / "config.yml").write_text("stand-in")
    if checkpoint_exists:
        (train_dir / "nerfstudio_models" / f"step-{RUN_ITERATIONS - 1:09d}.ckpt").write_text("")
    if job_info_text:
        (run_dir / "job_info.txt").write_text(job_info_text)
    # Step 2: stand-in tools on PATH via the environment's activate script.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool_name in ("ns-train", "ns-render", "nvidia-smi"):
        _write_executable(bin_dir / tool_name, LOGGING_STAND_IN)
    _write_executable(bin_dir / "python", PYTHON_STAND_IN)
    (stage_root / "env").mkdir(parents=True)
    (stage_root / "env" / ".build-complete").write_text("built stand-in\n")
    (stage_root / "env" / "viking-activate.sh").write_text(f'export PATH="{bin_dir}:$PATH"\n')
    # Step 3: run it, with only PATH and HOME inherited; the stand-ins lead PATH from the start, since
    # the job_info.txt header calls nvidia-smi before the environment is activated.
    call_log = tmp_path / "calls.log"
    process_environment = dict(environment, PATH=f"{bin_dir}:{os.environ['PATH']}", HOME=str(tmp_path),
                               CALL_LOG=str(call_log))
    completed = subprocess.run(["bash", str(SBATCH_PATH)], env=process_environment, capture_output=True, text=True)
    calls = call_log.read_text().splitlines() if call_log.exists() else []
    job_info_path = run_dir / "job_info.txt"
    return completed, calls, job_info_path.read_text() if job_info_path.exists() else ""


def test_sbatch_resume_skips_training_appends_and_renders_without_painting(tmp_path):
    """Resume: no ns-train, no plot; job_info.txt kept and appended; ns-render loads the paint-free copy."""
    completed, calls, job_info_text = run_sbatch(
        tmp_path, {"RESUME_AFTER_TRAINING": "True", "RESUME_GIT_SHA": HEAD_SHA}, FAILED_IN_RENDER_JOB_INFO_TEXT, True)
    assert completed.returncode == 0, completed.stderr
    tool_names = [call.split()[0] for call in calls]
    assert "ns-train" not in tool_names
    assert not any("11_plot_overfitting_curves.py" in call for call in calls)
    # The training job's record is intact, and this job's lines follow it.
    assert job_info_text.startswith(FAILED_IN_RENDER_JOB_INFO_TEXT)
    appended_info = load_submit_module().parse_job_info_text(job_info_text[len(FAILED_IN_RENDER_JOB_INFO_TEXT):])
    assert appended_info["RESUME_GIT_SHA"] == HEAD_SHA
    assert {"render_elapsed_s", "lidar_render_elapsed_s", "split_render_check", "end_utc"} <= set(appended_info)
    # ns-render loads <run>/render_config/config.yml, written by scripts/12 before it runs.
    render_config = str(tmp_path / "stage" / "runs" / RUN_ID / "render_config" / "config.yml")
    write_index = next(index for index, call in enumerate(calls) if "write-run-config" in call)
    render_index = tool_names.index("ns-render")
    assert write_index < render_index
    assert shlex.split(calls[render_index])[shlex.split(calls[render_index]).index("--load-config") + 1] == render_config
    assert "train 300 s (training job; resumed)" in completed.stdout


def test_sbatch_resume_refuses_without_the_final_checkpoint(tmp_path):
    """Resume with no step-(N-1) checkpoint: exit 1 before any render, no end_utc."""
    completed, calls, job_info_text = run_sbatch(
        tmp_path, {"RESUME_AFTER_TRAINING": "True", "RESUME_GIT_SHA": HEAD_SHA}, FAILED_IN_RENDER_JOB_INFO_TEXT, False)
    assert completed.returncode == 1
    assert "ns-render" not in [call.split()[0] for call in calls]
    assert "end_utc" not in job_info_text


def test_sbatch_normal_run_trains_with_the_override_and_renders_without_painting(tmp_path):
    """No resume: job_info.txt is written fresh, ns-train gets --train-split-fraction-override, ns-render the copy."""
    completed, calls, job_info_text = run_sbatch(
        tmp_path, {"TRAIN_SPLIT_FRACTION": "0.8", "TRAIN_SPLIT_FRACTION_OVERRIDE": "0.8"}, "stale: old\n", True)
    assert completed.returncode == 0, completed.stderr
    train_calls = [shlex.split(call) for call in calls if call.startswith("ns-train ")]
    assert len(train_calls) == 1
    train_arguments = train_calls[0]
    override_value = train_arguments[train_arguments.index("--train-split-fraction-override") + 1]
    assert override_value == "0.8"
    assert train_arguments.index("viking-data") < train_arguments.index("--train-split-fraction-override")
    assert "stale: old" not in job_info_text
    assert "TRAIN_SPLIT_FRACTION_OVERRIDE: 0.8" in job_info_text
    render_call = next(shlex.split(call) for call in calls if call.startswith("ns-render "))
    assert render_call[render_call.index("--load-config") + 1].endswith("/render_config/config.yml")
    # scripts/08 still loads the run's own config.yml, so dataparser_transforms.json lands where scripts/10 reads it.
    lidar_call = next(shlex.split(call) for call in calls if "08_render_lidar_pointcloud.py" in call)
    assert lidar_call[lidar_call.index("--load-config") + 1].endswith(f"/splatad/{RUN_ID}/config.yml")


def test_sbatch_normal_run_without_override_passes_no_override_flag(tmp_path):
    """No TRAIN_SPLIT_FRACTION_OVERRIDE: the flag is absent, and job_info.txt has no override line."""
    completed, calls, job_info_text = run_sbatch(tmp_path, {}, "", True)
    assert completed.returncode == 0, completed.stderr
    train_call = next(call for call in calls if call.startswith("ns-train "))
    assert "--train-split-fraction-override" not in train_call
    assert "TRAIN_SPLIT_FRACTION_OVERRIDE" not in job_info_text
