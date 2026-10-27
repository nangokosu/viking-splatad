"""Tests for render-only jobs: `submit_run.py render`, scripts/12_prepare_rerender.py and render_only.sbatch.

A render-only job renders guard frames of a finished run from its checkpoint (T66 in the dataparser,
tested in test_render_guard_selection). Here: the frame-list parser, the validation against the run's
split.json, the submission (mocked ssh), `clean` keeping a run whose checkpoint a render job uses,
the config copy and render check of script 12, and the wiring between the job script builder and
render_only.sbatch. No cluster access.
"""

import datetime
import importlib.util
import json
import re
import shlex
import subprocess
import types
from pathlib import Path
from typing import Dict, List, Optional

import pytest
import yaml

from tests.unit.test_slurm_submit import PROJECT_ROOT, SYNTHETIC_SINFO_TEXT, load_submit_module

PREPARE_RERENDER_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "12_prepare_rerender.py"
RENDER_SBATCH_PATH = PROJECT_ROOT / "scripts" / "slurm" / "render_only.sbatch"
STAGE_ROOT = "/w/example/user/viking-splatad"
RUN_ID = "may_short_grass-may_scene_a-20261026-120000-abcdef1"
TRAIN_SHA = "abcdef1" + "0" * 33
RENDER_SHA = "fedcba9876543210fedcba9876543210fedcba98"
SUCCESSFUL_JOB_INFO_TEXT = (
    f"hostname: gpunode2\nRUN_ID: {RUN_ID}\nSESSION: may_short_grass\nGIT_SHA: {TRAIN_SHA}\n"
    "RENDER_DOWNSCALE_FACTOR: 1\nenvironment build:\n  built x\nsplit_render_check: match\n"
    "end_utc: 2026-10-26T14:00:00Z\n")


def load_prepare_rerender_module():
    """Import scripts/12_prepare_rerender.py as a module (its name starts with a digit).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("prepare_rerender", PREPARE_RERENDER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _frame(kind: str, split: str, frame_index: int) -> Dict:
    """One split.json ``frames`` entry with a file name derived from the index (ns stamp = index * 10**8)."""
    sensor, folder, suffix = ("ids_camera", "camera", "jpg") if kind == "camera" else ("ouster", "lidar", "npy")
    timestamp_ns = frame_index * 10 ** 8
    return {"sensor": sensor, "kind": kind, "split": split, "frame_index": frame_index, "timestamp_ns": timestamp_ns,
            "time_s": timestamp_ns / 1e9, "file": f"{folder}/{sensor}/{frame_index:06d}_{timestamp_ns}.{suffix}"}


def synthetic_split_record() -> Dict:
    """A split.json with, per kind: train 0-2, guard 3, test 4, guard 5-6, train 7 (a T62 block 3-6)."""
    splits_by_index = {0: "train", 1: "train", 2: "train", 3: "guard", 4: "test", 5: "guard", 6: "guard", 7: "train"}
    frames = [_frame(kind, split, frame_index) for kind in ("camera", "lidar")
              for frame_index, split in splits_by_index.items()]
    return {"membership_sha256": "f" * 64, "frames": frames}


# ---- frame lists --------------------------------------------------------------------------------

def test_parse_frame_index_list_known_values():
    """Single indices and inclusive ranges, merged, sorted and deduplicated."""
    submit_module = load_submit_module()
    assert submit_module.parse_frame_index_list("795,797-800") == [795, 797, 798, 799, 800]
    assert submit_module.parse_frame_index_list(" 12 , 3-4,4 ") == [3, 4, 12]
    assert submit_module.parse_frame_index_list("7-7") == [7]


@pytest.mark.parametrize("bad_text", ["", " ", "1,,2", "1,", "-3", "a", "1-", "1-b", "5-2", "1.5", "2--3"])
def test_parse_frame_index_list_rejects_malformed_lists(bad_text):
    """Empty, blank items, negatives, non-integers and reversed ranges are refused."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError):
        submit_module.parse_frame_index_list(bad_text)


# ---- validation against split.json ---------------------------------------------------------------

def test_validate_render_selection_returns_the_guard_files_in_index_order():
    """Guard camera 6 and 3 and lidar 5 give their split.json files, sorted by frame index."""
    submit_module = load_submit_module()
    split_record = synthetic_split_record()
    selected_files = submit_module.validate_render_selection(split_record, [6, 3], [5])
    assert selected_files == {"camera": [_frame("camera", "guard", 3)["file"], _frame("camera", "guard", 6)["file"]],
                              "lidar": [_frame("lidar", "guard", 5)["file"]]}


def test_validate_render_selection_names_every_non_guard_frame_with_its_split():
    """A train frame, a test core frame and an absent frame are all listed, with what they are."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError) as raised:
        submit_module.validate_render_selection(synthetic_split_record(), [3, 0, 4], [5, 99])
    message = str(raised.value)
    assert "camera 0 (train)" in message and "camera 4 (test)" in message and "lidar 99 (absent)" in message
    assert "camera 3" not in message and "lidar 5" not in message


def test_validate_render_selection_needs_both_lists_and_two_camera_frames():
    """Empty lists are refused; a single camera frame is refused with the ns-render reason."""
    submit_module = load_submit_module()
    split_record = synthetic_split_record()
    with pytest.raises(ValueError, match="both"):
        submit_module.validate_render_selection(split_record, [3, 5], [])
    with pytest.raises(ValueError, match="both"):
        submit_module.validate_render_selection(split_record, [], [5])
    with pytest.raises(ValueError, match="commonpath|common path"):
        submit_module.validate_render_selection(split_record, [3], [5])
    with pytest.raises(ValueError, match="repeats"):
        submit_module.validate_render_selection(split_record, [3, 3], [5])
    assert submit_module.MINIMUM_CAMERA_RENDER_FRAMES == 2


def test_rerender_label_format():
    """render-<UTC yyyymmdd-HHMMSS>-<first 7 hex of the render SHA>."""
    submit_module = load_submit_module()
    moment = datetime.datetime(2026, 10, 27, 9, 8, 7, tzinfo=datetime.timezone.utc)
    assert submit_module.build_rerender_label(RENDER_SHA, moment) == f"render-20261027-090807-{RENDER_SHA[:7]}"


# ---- submission (mocked ssh) -----------------------------------------------------------------------

def make_fake_render_cluster(job_info_text: str, split_record: Dict, code_is_staged: bool,
                             issued_commands: List[List[str]], written_files: Dict[str, str],
                             environment_marker_text: Optional[str] = f"env_prefix={STAGE_ROOT}/envs/splatad-x\n"):
    """Build a run_command stand-in for `render`: git, stage root, run files, code check, marker, sinfo, sbatch.

    Args:
        job_info_text: Served for ``cat <run>/job_info.txt``.
        split_record: Served (as JSON) for ``cat <run>/split.json``.
        code_is_staged: Exit status of ``test -d <code dir>``.
        issued_commands: Filled with every command.
        written_files: Filled with remote path -> text for every ``cat > "$1"`` write.
        environment_marker_text: Served for the env marker; None = not built.

    Returns:
        The fake ``run_command``.
    """
    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command and answer it like the cluster would."""
        issued_commands.append(list(command))
        if command[0] == "ssh" and input_text is not None:
            written_files[shlex.split(command[-1])[-1]] = input_text
        remote_text = command[-1] if command[0] == "ssh" else ""
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout=RENDER_SHA + "\n", stderr="")
        if remote_text.startswith("sh -c") and input_text is None:
            return subprocess.CompletedProcess(command, 0, stdout=STAGE_ROOT + "\n", stderr="")
        if remote_text.startswith("cat ") and remote_text.endswith("job_info.txt"):
            return subprocess.CompletedProcess(command, 0, stdout=job_info_text, stderr="")
        if remote_text.startswith("cat ") and remote_text.endswith("split.json"):
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(split_record), stderr="")
        if remote_text.startswith("cat ") and remote_text.endswith(".marker"):
            if environment_marker_text is None:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="No such file")
            return subprocess.CompletedProcess(command, 0, stdout=environment_marker_text, stderr="")
        if remote_text.startswith("test -d"):
            return subprocess.CompletedProcess(command, 0 if code_is_staged else 1, stdout="", stderr="")
        if remote_text.startswith("sinfo"):
            return subprocess.CompletedProcess(command, 0, stdout=SYNTHETIC_SINFO_TEXT, stderr="")
        if " sbatch " in f" {remote_text} ":
            return subprocess.CompletedProcess(command, 0, stdout="5151\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    return fake_run_command


def _prepare_submit_module(monkeypatch, tmp_path, fake_run_command):
    """Load submit_run with the fake cluster, fixed recipe bytes and a temporary ledger path."""
    submit_module = load_submit_module()
    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    monkeypatch.setattr(submit_module, "read_committed_file", lambda git_sha, relative_path: relative_path.encode())
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    monkeypatch.delenv("VIKING_SLURM_USER", raising=False)
    return submit_module


def _sbatch_commands(issued_commands):
    """Remote texts of every sbatch submission among the issued commands."""
    return [command[-1] for command in issued_commands if command[0] == "ssh" and " sbatch " in f" {command[-1]} "]


def test_submit_render_writes_selection_and_job_script_and_submits(monkeypatch, tmp_path):
    """End to end: selection.json and job.sbatch in <run>/rerenders/<label>/, sbatch with render_only's limit."""
    issued_commands: List[List[str]] = []
    written_files: Dict[str, str] = {}
    split_record = synthetic_split_record()
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, make_fake_render_cluster(
        SUCCESSFUL_JOB_INFO_TEXT, split_record, True, issued_commands, written_files))
    moment = datetime.datetime(2026, 10, 27, 9, 8, 7, tzinfo=datetime.timezone.utc)
    config = submit_module.load_slurm_config()
    job_id, label = submit_module.submit_render(config, RUN_ID, "3,5-6", "5", skip_gates=True, now_utc=moment)
    assert job_id == "5151"
    assert label == submit_module.build_rerender_label(RENDER_SHA, moment)
    rerender_dir = f"{STAGE_ROOT}/runs/{RUN_ID}/rerenders/{label}"
    # Step 1: the selection names the requested guard frames and the split it was checked against.
    selection = json.loads(written_files[f"{rerender_dir}/selection.json"])
    assert selection["camera"] == [3, 5, 6] and selection["lidar"] == [5]
    assert selection["files"] == submit_module.validate_render_selection(split_record, [3, 5, 6], [5])
    assert selection["source_split_membership_sha256"] == split_record["membership_sha256"]
    assert selection["requested_utc"] == "2026-10-27T09:08:07Z"
    # Step 2: the job script exports the render variables and runs the render code's render_only.sbatch.
    job_script_lines = written_files[f"{rerender_dir}/job.sbatch"].splitlines()
    for expected_line in (f"export RUN_ID={RUN_ID}", f"export RERENDER_LABEL={label}", f"export STAGE_ROOT={STAGE_ROOT}",
                          "export SESSION=may_short_grass", f"export GIT_SHA={RENDER_SHA}",
                          f"export TRAIN_GIT_SHA={TRAIN_SHA}", f"export CODE_DIR={STAGE_ROOT}/code/{RENDER_SHA}",
                          f"export ENV_PREFIX={STAGE_ROOT}/envs/splatad-x", "export RENDER_DOWNSCALE_FACTOR=1"):
        assert expected_line in job_script_lines
    assert job_script_lines[-1] == f"exec bash {STAGE_ROOT}/code/{RENDER_SHA}/scripts/slurm/render_only.sbatch"
    # Step 3: one sbatch, with the configured time limit, a render job name and the log in the rerender dir.
    sbatch_texts = _sbatch_commands(issued_commands)
    assert len(sbatch_texts) == 1
    sbatch_arguments = shlex.split(sbatch_texts[0])
    assert ["-t", config["render_only"]["time_limit"]] == sbatch_arguments[
        sbatch_arguments.index("-t"):sbatch_arguments.index("-t") + 2]
    assert "--job-name=splatad-render-may_short_grass" in sbatch_arguments
    assert f"--output={rerender_dir}/slurm-%j.out" in sbatch_arguments
    assert sbatch_arguments[-1] == f"{rerender_dir}/job.sbatch"
    # Step 4: the training ledger is untouched.
    assert not (tmp_path / "submitted.tsv").exists()


def test_submit_render_refuses_an_unsuccessful_run(monkeypatch, tmp_path):
    """A run without end_utc is refused before split.json is read and before any sbatch."""
    issued_commands: List[List[str]] = []
    unfinished_text = SUCCESSFUL_JOB_INFO_TEXT.replace("end_utc: 2026-10-26T14:00:00Z\n", "")
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, make_fake_render_cluster(
        unfinished_text, synthetic_split_record(), True, issued_commands, {}))
    with pytest.raises(ValueError, match="did not succeed"):
        submit_module.submit_render(submit_module.load_slurm_config(), RUN_ID, "3,5", "5", skip_gates=True)
    assert not any(command[-1].endswith("split.json") for command in issued_commands)
    assert not _sbatch_commands(issued_commands)


def test_submit_render_refuses_non_guard_frames_before_any_remote_write(monkeypatch, tmp_path):
    """A core frame in the request stops the submission: no mkdir, no write, no sbatch."""
    issued_commands: List[List[str]] = []
    written_files: Dict[str, str] = {}
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, make_fake_render_cluster(
        SUCCESSFUL_JOB_INFO_TEXT, synthetic_split_record(), True, issued_commands, written_files))
    with pytest.raises(ValueError, match="camera 4 \\(test\\)"):
        submit_module.submit_render(submit_module.load_slurm_config(), RUN_ID, "3-4", "5", skip_gates=True)
    assert not written_files
    assert not any(command[0] == "ssh" and command[-1].startswith("mkdir") for command in issued_commands)
    assert not _sbatch_commands(issued_commands)


def test_submit_render_needs_the_render_code_staged(monkeypatch, tmp_path):
    """HEAD not under code/<sha>: the error names `stage --session <the run's session>`."""
    issued_commands: List[List[str]] = []
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, make_fake_render_cluster(
        SUCCESSFUL_JOB_INFO_TEXT, synthetic_split_record(), False, issued_commands, {}))
    with pytest.raises(RuntimeError, match="stage --session may_short_grass"):
        submit_module.submit_render(submit_module.load_slurm_config(), RUN_ID, "3,5", "5", skip_gates=True)
    assert not _sbatch_commands(issued_commands)


def test_submit_render_needs_the_environment(monkeypatch, tmp_path):
    """No marker for the render code's recipe: refused with the build-env hint."""
    issued_commands: List[List[str]] = []
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, make_fake_render_cluster(
        SUCCESSFUL_JOB_INFO_TEXT, synthetic_split_record(), True, issued_commands, {}, environment_marker_text=None))
    with pytest.raises(RuntimeError, match="build-env"):
        submit_module.submit_render(submit_module.load_slurm_config(), RUN_ID, "3,5", "5", skip_gates=True)
    assert not _sbatch_commands(issued_commands)


def test_submit_render_rejects_a_malformed_run_id_without_ssh(monkeypatch, tmp_path):
    """A run id that is not one path component is refused before any command runs."""
    issued_commands: List[List[str]] = []
    submit_module = _prepare_submit_module(monkeypatch, tmp_path, make_fake_render_cluster(
        SUCCESSFUL_JOB_INFO_TEXT, synthetic_split_record(), True, issued_commands, {}))
    with pytest.raises(ValueError, match="malformed"):
        submit_module.submit_render(submit_module.load_slurm_config(), "../other", "3,5", "5", skip_gates=True)
    assert not issued_commands


def test_render_subcommand_reaches_submit_render(monkeypatch):
    """`render <run> --camera-frames .. --lidar-sweeps .. --skip-gates` calls submit_render with them."""
    submit_module = load_submit_module()
    received_calls = []
    monkeypatch.setattr(submit_module, "submit_render",
                        lambda config, run_id, camera_text, lidar_text, skip_gates: received_calls.append(
                            (run_id, camera_text, lidar_text, skip_gates)))
    exit_code = submit_module.main(["render", RUN_ID, "--camera-frames", "3,5", "--lidar-sweeps", "5", "--skip-gates"])
    assert exit_code == 0
    assert received_calls == [(RUN_ID, "3,5", "5", True)]


# ---- clean keeps a run whose checkpoint a render job uses ---------------------------------------------

def test_active_render_job_marks_its_run_active():
    """<runs>/<run_id>/rerenders/<label>/job.sbatch in squeue -> <run_id> is active."""
    submit_module = load_submit_module()
    listing_text = (f"RUN\t{RUN_ID}\nINFO_BEGIN\n{SUCCESSFUL_JOB_INFO_TEXT}\nINFO_END\n"
                    f"ACTIVE\t{STAGE_ROOT}/runs/{RUN_ID}/rerenders/render-20261027-090807-fedcba9/job.sbatch\n")
    _, active_run_ids = submit_module.parse_remote_run_listing(listing_text)
    assert RUN_ID in active_run_ids


def test_clean_never_deletes_a_run_with_an_active_render_job():
    """Even when named, a run whose render job is queued or running is refused (scancel first)."""
    submit_module = load_submit_module()
    listing_text = (f"RUN\t{RUN_ID}\n"
                    f"ACTIVE\t{STAGE_ROOT}/runs/{RUN_ID}/rerenders/render-20261027-090807-fedcba9/job.sbatch\n")
    remote_runs, active_run_ids = submit_module.parse_remote_run_listing(listing_text)
    # Unsuccessful (no job_info) would normally be deleted; the active render job keeps it.
    plan = submit_module.plan_cleanup(remote_runs, active_run_ids, {}, [], [])
    assert [(decision["run_id"], decision["delete"]) for decision in plan] == [(RUN_ID, False)]
    with pytest.raises(ValueError, match="scancel"):
        submit_module.plan_cleanup(remote_runs, active_run_ids, {}, [], [RUN_ID])


# ---- scripts/12_prepare_rerender.py ------------------------------------------------------------------

SELECTED_FILES = {"camera": ["camera/ids_camera/000003_300000000.jpg", "camera/ids_camera/000005_500000000.jpg"],
                  "lidar": ["lidar/ouster/000005_500000000.npy"]}


def _matching_stems() -> Dict:
    """Rendered stems exactly equal to SELECTED_FILES, per (kind, folder)."""
    camera_stems = ["000003_300000000", "000005_500000000"]
    return {("camera", "rgb"): camera_stems, ("camera", "gt-rgb"): list(camera_stems),
            ("lidar", "lidar"): ["000005_500000000"]}


def test_compare_rendered_with_selection_exit_codes():
    """Equal stems -> 0; an extra or missing stem -> 1 with the folder named; an absent folder -> 2."""
    prepare_module = load_prepare_rerender_module()
    assert prepare_module.compare_rendered_with_selection(SELECTED_FILES, _matching_stems()) == (0, [])
    extra_stems = _matching_stems()
    extra_stems[("camera", "gt-rgb")].append("000004_400000000")
    exit_code, reasons = prepare_module.compare_rendered_with_selection(SELECTED_FILES, extra_stems)
    assert exit_code == 1 and len(reasons) == 1 and "gt-rgb" in reasons[0] and "000004_400000000" in reasons[0]
    missing_stems = _matching_stems()
    missing_stems[("lidar", "lidar")] = []
    exit_code, reasons = prepare_module.compare_rendered_with_selection(SELECTED_FILES, missing_stems)
    assert exit_code == 1 and "000005_500000000" in reasons[0]
    absent_folder = _matching_stems()
    absent_folder[("lidar", "lidar")] = None
    # A missing folder outranks a mismatch elsewhere: the step did not run.
    absent_folder[("camera", "rgb")] = []
    assert prepare_module.compare_rendered_with_selection(SELECTED_FILES, absent_folder)[0] == 2


def _write_rerender_dir(rerender_dir: Path, stems_by_folder: Dict) -> None:
    """Create selection.json and empty render files named by stems_by_folder (None = no folder)."""
    rerender_dir.mkdir(parents=True)
    (rerender_dir / "selection.json").write_text(json.dumps({"camera": [3, 5], "lidar": [5], "files": SELECTED_FILES}))
    for (sensor_kind, folder_name), stems in stems_by_folder.items():
        if stems is None:
            continue
        folder = rerender_dir / "renders" / "test" / folder_name
        folder.mkdir(parents=True)
        suffix = ".npz" if sensor_kind == "lidar" else ".jpg"
        for stem in stems:
            (folder / f"{stem}{suffix}").write_bytes(b"")


def test_check_subcommand_on_disk(tmp_path):
    """check: 0 for matching renders, 1 for an extra file, 2 without selection.json or a render folder."""
    prepare_module = load_prepare_rerender_module()
    _write_rerender_dir(tmp_path / "match", _matching_stems())
    assert prepare_module.main(["check", "--rerender-dir", str(tmp_path / "match")]) == 0
    mismatch_stems = _matching_stems()
    mismatch_stems[("camera", "rgb")] = mismatch_stems[("camera", "rgb")] + ["000006_600000000"]
    _write_rerender_dir(tmp_path / "mismatch", mismatch_stems)
    assert prepare_module.main(["check", "--rerender-dir", str(tmp_path / "mismatch")]) == 1
    absent_stems = _matching_stems()
    absent_stems[("lidar", "lidar")] = None
    _write_rerender_dir(tmp_path / "absent", absent_stems)
    assert prepare_module.main(["check", "--rerender-dir", str(tmp_path / "absent")]) == 2
    (tmp_path / "no-selection").mkdir()
    assert prepare_module.main(["check", "--rerender-dir", str(tmp_path / "no-selection")]) == 2


def _synthetic_trainer_config(test_block_count, core_fraction) -> types.SimpleNamespace:
    """Object with nerfstudio's attribute path pipeline.datamanager.dataparser (yaml round-trips it)."""
    dataparser = types.SimpleNamespace(test_block_count=test_block_count, test_block_core_fraction=core_fraction)
    return types.SimpleNamespace(output_dir=Path("/w/stage/runs/r/train"), timestamp="r",
                                 pipeline=types.SimpleNamespace(datamanager=types.SimpleNamespace(dataparser=dataparser)))


def test_write_config_sets_only_the_selection_file(tmp_path):
    """The copy loads with yaml.Loader, has the absolute selection path, and keeps every other field."""
    prepare_module = load_prepare_rerender_module()
    train_config_path = tmp_path / "config.yml"
    train_config_path.write_text(yaml.dump(_synthetic_trainer_config(2, 0.1)))
    original_text = train_config_path.read_text()
    selection_path = tmp_path / "rerender" / "selection.json"
    selection_path.parent.mkdir()
    selection_path.write_text("{}")
    output_path = tmp_path / "rerender" / "config.yml"
    assert prepare_module.main(["write-config", "--train-config", str(train_config_path),
                                "--selection", str(selection_path), "--output", str(output_path)]) == 0
    render_config = yaml.load(output_path.read_text(), Loader=yaml.Loader)
    dataparser = render_config.pipeline.datamanager.dataparser
    assert dataparser.render_frame_selection_file == selection_path.resolve()
    assert dataparser.render_frame_selection_file.is_absolute()
    assert (dataparser.test_block_count, dataparser.test_block_core_fraction) == (2, 0.1)
    assert render_config.output_dir == Path("/w/stage/runs/r/train") and render_config.timestamp == "r"
    # The run's own config.yml is never edited.
    assert train_config_path.read_text() == original_text


@pytest.mark.parametrize("test_block_count, core_fraction", [(None, 0.1), (2, 1.0)])
def test_write_config_refuses_a_run_without_guard_frames(tmp_path, test_block_count, core_fraction):
    """No blocks, or c = 1: there are no guard frames, so write-config exits 1 and writes nothing."""
    prepare_module = load_prepare_rerender_module()
    train_config_path = tmp_path / "config.yml"
    train_config_path.write_text(yaml.dump(_synthetic_trainer_config(test_block_count, core_fraction)))
    output_path = tmp_path / "render-config.yml"
    assert prepare_module.main(["write-config", "--train-config", str(train_config_path),
                                "--selection", str(tmp_path / "selection.json"), "--output", str(output_path)]) == 1
    assert not output_path.exists()


# ---- render_only.sbatch wiring ----------------------------------------------------------------------

def _shell_loop_variables(sbatch_text: str, loop_variable: str) -> List[str]:
    """Names listed by `for <loop_variable> in ...; do` in the sbatch (joined across continuations)."""
    match = re.search(rf"for {loop_variable} in (.*?); do", sbatch_text, flags=re.DOTALL)
    assert match, f"no `for {loop_variable} in` loop"
    return match.group(1).replace("\\\n", " ").split()


def test_every_exported_render_variable_is_required_and_recorded():
    """The job script builder's variables = the sbatch's required list = its job_info.txt list."""
    submit_module = load_submit_module()
    job_environment = submit_module.build_render_job_environment(
        RUN_ID, "render-x", STAGE_ROOT, submit_module.parse_job_info_text(SUCCESSFUL_JOB_INFO_TEXT), RENDER_SHA,
        {"env_prefix": f"{STAGE_ROOT}/envs/splatad-x"})
    sbatch_text = RENDER_SBATCH_PATH.read_text()
    assert set(_shell_loop_variables(sbatch_text, "required_variable")) == set(job_environment)
    assert set(_shell_loop_variables(sbatch_text, "exported_variable")) == set(job_environment)


def test_render_script_is_staged_and_renders_the_selection_config():
    """render_only.sbatch ships with code/<sha> (scripts/ is staged) and both renders use the rerender config."""
    submit_module = load_submit_module()
    assert (PROJECT_ROOT / submit_module.RENDER_SCRIPT_RELATIVE_PATH) == RENDER_SBATCH_PATH
    assert RENDER_SBATCH_PATH.is_file()
    assert submit_module.RENDER_SCRIPT_RELATIVE_PATH.split("/")[0] in submit_module.STAGED_REPO_PATHS
    sbatch_text = RENDER_SBATCH_PATH.read_text()
    # Step 1: the config is written before the first render, and both renders load it (never the run's own).
    write_position = sbatch_text.index("12_prepare_rerender.py\" write-config")
    assert write_position < sbatch_text.index("\nns-render dataset") < sbatch_text.index("08_render_lidar_pointcloud.py")
    assert sbatch_text.count('--load-config "$RENDER_CONFIG"') == 2
    assert '--load-config "$TRAIN_CONFIG"' not in sbatch_text
    # Argument lines only (a comment also mentions the flag).
    assert sbatch_text.count("\n  --pose-source test \\\n") == 2
    # Step 2: the check gates end_utc.
    check_position = sbatch_text.index("12_prepare_rerender.py\" check")
    assert sbatch_text.index("lidar_render_elapsed_s") < check_position < sbatch_text.index("render_check: match")
    assert sbatch_text.index("render_check: match") < sbatch_text.index('echo "end_utc:')
    # Step 3: the selection file the job passes is the one submit_run.py writes.
    assert f'/{submit_module.SELECTION_FILE_NAME}"' in sbatch_text


def test_render_only_time_limit_is_valid():
    """render_only.time_limit parses as D-HH:MM:SS and is within the 5-day cluster maximum."""
    submit_module = load_submit_module()
    time_limit = submit_module.load_slurm_config()["render_only"]["time_limit"]
    assert 0 < submit_module.parse_slurm_time_limit_seconds(time_limit) <= submit_module.MAXIMUM_TIME_LIMIT_SECONDS
