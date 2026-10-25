"""Tests of the Slurm submission helpers (scripts/slurm/submit_run.py).

All external commands go through ``run_command``, which these tests replace, so nothing touches
git, ssh or the cluster. Expected values are computed here from the stated unit definitions.
"""

import datetime
import hashlib
import importlib.util
import math
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "slurm" / "submit_run.py"

# Synthetic `sinfo -p gpunodes -O NodeHost,Memory,CPUs,Gres` output (MEMORY in MB): one A2000
# node and two identical A6000 nodes, whitespace-padded as sinfo prints it.
SYNTHETIC_SINFO_TEXT = """\
HOSTNAMES           MEMORY              CPUS                GRES
gpunode1            120000              32                  gpu:rtx_a2000:1
gpunode2            125000              24                  gpu:rtx_a6000:1
gpunode3            125000              24                  gpu:rtx_a6000:1
"""
A6000_NODE_MEMORY_MB = 125000
A6000_NODE_CPUS = 24
# Example --mem value: 90% of an A6000 node's RAM in GiB (MB / 1024), floored.
EXAMPLE_MEMORY_GB = math.floor(0.9 * A6000_NODE_MEMORY_MB / 1024)


def load_submit_module():
    """Import scripts/slurm/submit_run.py (scripts/ is not a package).

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location("submit_run", SUBMIT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_example_sbatch(submit_module, time_limit: str) -> List[str]:
    """Build an sbatch command with fixed example inputs and the given time limit.

    Args:
        submit_module: Loaded submit_run module.
        time_limit: Slurm D-HH:MM:SS string.

    Returns:
        The sbatch argument list.
    """
    return submit_module.build_sbatch_command(
        partition="gpunodes", gres="gpu:rtx_a6000:1", cpus=A6000_NODE_CPUS, memory_gb=EXAMPLE_MEMORY_GB,
        time_limit=time_limit, job_name="splatad-example", output_path="/stage/runs/r/slurm-%j.out",
        job_environment={"RUN_ID": "r", "SESSION": "example"}, script_path="/stage/code/sha/run.sbatch")


def test_derive_node_resources_picks_matching_gpu_nodes():
    """A6000 rows only: cpus = 24, mem = floor(0.9 * 125000 MB / 1024 MB-per-GiB)."""
    submit_module = load_submit_module()
    mem_fraction = 0.9
    cpus, memory_gb = submit_module.derive_node_resources(SYNTHETIC_SINFO_TEXT, "rtx_a6000", mem_fraction)
    assert cpus == A6000_NODE_CPUS
    # Slurm's --mem G suffix is GiB = 1024 MB; floor keeps the request within the fraction.
    assert memory_gb == math.floor(mem_fraction * A6000_NODE_MEMORY_MB / 1024)


def test_derive_node_resources_raises_without_matching_node():
    """No node with the requested GPU type -> ValueError."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError):
        submit_module.derive_node_resources(SYNTHETIC_SINFO_TEXT, "h100", 0.9)


def test_gres_type_from_spec():
    """The GPU type is the middle field of name:type:count; a typeless spec is rejected."""
    submit_module = load_submit_module()
    assert submit_module.gres_type_from_spec("gpu:rtx_a6000:1") == "rtx_a6000"
    with pytest.raises(ValueError):
        submit_module.gres_type_from_spec("gpu:1")


def test_parse_slurm_time_limit_seconds_known_values():
    """D-HH:MM:SS -> days*86400 + hours*3600 + minutes*60 + seconds."""
    submit_module = load_submit_module()
    assert submit_module.parse_slurm_time_limit_seconds("0-06:00:00") == 6 * 3600
    assert submit_module.parse_slurm_time_limit_seconds("5-00:00:00") == 5 * 24 * 3600
    assert submit_module.parse_slurm_time_limit_seconds("1-02:03:04") == 86400 + 2 * 3600 + 3 * 60 + 4
    with pytest.raises(ValueError):
        submit_module.parse_slurm_time_limit_seconds("06:00:00")
    with pytest.raises(ValueError):
        submit_module.parse_slurm_time_limit_seconds("0-24:00:00")


def test_build_sbatch_command_contents():
    """Resources are all on the command line; job variables go through `env`, never --export."""
    submit_module = load_submit_module()
    sbatch_arguments = build_example_sbatch(submit_module, "0-06:00:00")
    sbatch_position = sbatch_arguments.index("sbatch")
    # `env K=V ... sbatch`: variables precede sbatch (the cluster holds --export=ALL,K=V jobs).
    assert sbatch_arguments[:sbatch_position] == ["env", "RUN_ID=r", "SESSION=example"]
    assert "--partition=gpunodes" in sbatch_arguments
    assert "--gres=gpu:rtx_a6000:1" in sbatch_arguments
    assert sbatch_arguments[sbatch_arguments.index("-c") + 1] == str(A6000_NODE_CPUS)
    assert f"--mem={EXAMPLE_MEMORY_GB}G" in sbatch_arguments
    assert sbatch_arguments[sbatch_arguments.index("-t") + 1] == "0-06:00:00"
    assert not any(argument.startswith("--export") for argument in sbatch_arguments)
    assert not any("get-user-env" in argument for argument in sbatch_arguments)
    assert sbatch_arguments[-1] == "/stage/code/sha/run.sbatch"


def test_build_sbatch_command_accepts_exactly_five_days_and_rejects_more():
    """5-00:00:00 is the cluster maximum; one second more is refused."""
    submit_module = load_submit_module()
    build_example_sbatch(submit_module, "5-00:00:00")
    with pytest.raises(ValueError):
        build_example_sbatch(submit_module, "5-00:00:01")


def test_build_run_id_format():
    """<session>-<profile>-<UTC YYYYmmdd-HHMMSS>-<first 7 SHA chars>."""
    submit_module = load_submit_module()
    full_sha = "0123456789abcdef0123456789abcdef01234567"
    moment = datetime.datetime(2026, 10, 25, 13, 5, 9, tzinfo=datetime.timezone.utc)
    assert (submit_module.build_run_id("may_short_grass", "smoke", full_sha, moment)
            == f"may_short_grass-smoke-20261025-130509-{full_sha[:7]}")


def test_resolve_ssh_target_uses_env_user_only_when_set(monkeypatch):
    """user@host only when VIKING_SLURM_USER is set; otherwise the bare host."""
    submit_module = load_submit_module()
    monkeypatch.delenv("VIKING_SLURM_USER", raising=False)
    assert submit_module.resolve_ssh_target("login.example") == "login.example"
    monkeypatch.setenv("VIKING_SLURM_USER", "someone")
    assert submit_module.resolve_ssh_target("login.example") == "someone@login.example"


def test_gates_refuse_dirty_tree(monkeypatch):
    """A tracked modification fails the gates before tests or the contract run."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []

    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command; report one modified tracked file and one untracked file."""
        issued_commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout=" M src/module.py\n?? notes.txt\n", stderr="")

    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    with pytest.raises(submit_module.GateFailure):
        submit_module.run_gates("example_session")
    assert issued_commands == [["git", "status", "--porcelain"]]


def test_gates_allow_untracked_and_run_tests_and_contract(monkeypatch):
    """Only untracked files -> pytest then the contract validator run, and the gates pass."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []

    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command; git reports only an untracked file; everything succeeds."""
        issued_commands.append(list(command))
        status_text = "?? notes.txt\n" if command[:2] == ["git", "status"] else ""
        return subprocess.CompletedProcess(command, 0, stdout=status_text, stderr="")

    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    submit_module.run_gates("example_session")
    assert issued_commands[1] == ["uv", "run", "pytest", "-q"]
    assert "scripts/06_validate_contract.py" in issued_commands[2]
    assert issued_commands[2][-1].endswith("data/intermediate/example_session")


def test_null_stage_root_is_a_clear_error():
    """A missing stage_root is reported by name instead of failing later in a remote command."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError, match="stage_root"):
        submit_module.require_stage_root({"stage_root": None})


def test_compute_recipe_hash_known_value_order_independence_and_sensitivity():
    """Hash = sha256 over sorted (path NUL length NUL content); order-free, content-sensitive."""
    submit_module = load_submit_module()
    recipe_contents = {"b.txt": b"second", "a.txt": b"first"}
    # Expected value built here from the documented encoding.
    expected_digest = hashlib.sha256()
    for relative_path in ("a.txt", "b.txt"):
        file_bytes = recipe_contents[relative_path]
        expected_digest.update(relative_path.encode() + b"\0" + str(len(file_bytes)).encode() + b"\0")
        expected_digest.update(file_bytes)
    expected_hash = expected_digest.hexdigest()[:submit_module.RECIPE_HASH_LENGTH]
    assert submit_module.compute_recipe_hash(recipe_contents) == expected_hash
    # Insertion order does not matter; a one-byte content change does.
    assert submit_module.compute_recipe_hash({"a.txt": b"first", "b.txt": b"second"}) == expected_hash
    assert submit_module.compute_recipe_hash({"a.txt": b"first", "b.txt": b"secone"}) != expected_hash
    # Moving bytes across the file boundary changes the hash (length prefix disambiguates).
    assert (submit_module.compute_recipe_hash({"a.txt": b"firsts", "b.txt": b"econd"})
            != expected_hash)


def make_fake_cluster(full_sha: str, absolute_stage_root: str, environment_marker_text: Optional[str],
                      issued_commands: List[List[str]]):
    """Build a run_command stand-in answering git, stage-root resolution, the env marker, sinfo, sbatch.

    Args:
        full_sha: SHA returned by ``git rev-parse HEAD``.
        absolute_stage_root: Path printed by the remote ``pwd -P``.
        environment_marker_text: Contents served for ``cat <...>.marker``; None = no marker (cat fails).
        issued_commands: List that records every command.

    Returns:
        The fake ``run_command`` function.
    """
    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command and return a canned CompletedProcess."""
        issued_commands.append(list(command))
        remote_text = command[-1] if command[0] == "ssh" else ""
        if command[:2] == ["git", "rev-parse"]:
            return subprocess.CompletedProcess(command, 0, stdout=full_sha + "\n", stderr="")
        if remote_text.startswith("sh -c"):
            return subprocess.CompletedProcess(command, 0, stdout=absolute_stage_root + "\n", stderr="")
        if remote_text.startswith("cat ") and remote_text.endswith(".marker"):
            if environment_marker_text is None:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="No such file")
            return subprocess.CompletedProcess(command, 0, stdout=environment_marker_text, stderr="")
        if remote_text.startswith("sinfo"):
            return subprocess.CompletedProcess(command, 0, stdout=SYNTHETIC_SINFO_TEXT, stderr="")
        if " sbatch " in f" {remote_text} ":
            return subprocess.CompletedProcess(command, 0, stdout="4242\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    return fake_run_command


def test_submit_run_issues_expected_remote_commands(monkeypatch, tmp_path):
    """End-to-end submit with mocked ssh: sinfo-derived resources, env and code paths reach sbatch."""
    submit_module = load_submit_module()
    full_sha = "fedcba9876543210fedcba9876543210fedcba98"
    absolute_stage_root = "/h/example/viking-splatad"
    issued_commands: List[List[str]] = []
    moment = datetime.datetime(2026, 10, 25, 1, 2, 3, tzinfo=datetime.timezone.utc)
    # Env on scratch that expires 10 days after submission: outlives the smoke time limit.
    scratch_env_prefix = "/scratch/scratch-space/expires-2026-Nov-04/viking-splatad-user/envs/splatad-x"
    expires_epoch = int(moment.timestamp()) + 10 * 86400
    marker_text = f"env_prefix={scratch_env_prefix}\nexpires_epoch={expires_epoch}\nbuilt_utc=t\n"
    monkeypatch.setattr(submit_module, "run_command",
                        make_fake_cluster(full_sha, absolute_stage_root, marker_text, issued_commands))
    # Committed recipe files replaced by fixed bytes so the expected env name is computable here.
    monkeypatch.setattr(submit_module, "read_committed_file",
                        lambda git_sha, relative_path: relative_path.encode())
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    monkeypatch.delenv("VIKING_SLURM_USER", raising=False)
    # Synthetic recording span, so the profile's time window does not read a real dataset.
    synthetic_span = (5.0, 205.0)
    monkeypatch.setattr(submit_module, "dataset_time_span", lambda session_dir: synthetic_span)
    config = submit_module.load_slurm_config()
    job_id, run_id = submit_module.submit_run(config, "may_short_grass", "smoke", skip_gates=True,
                                              now_utc=moment)
    assert job_id == "4242"
    assert run_id == f"may_short_grass-smoke-20261025-010203-{full_sha[:7]}"
    sbatch_remote_text = next(command[-1] for command in issued_commands
                              if command[0] == "ssh" and " sbatch " in f" {command[-1]} ")
    expected_memory_gb = math.floor(config["mem_fraction"] * A6000_NODE_MEMORY_MB / 1024)
    assert f"--mem={expected_memory_gb}G" in sbatch_remote_text
    assert f"-c {A6000_NODE_CPUS}" in sbatch_remote_text
    # Window fractions exported exactly as s / D and (s + w) / D of the synthetic span.
    smoke_profile = config["profiles"]["smoke"]
    span_duration = synthetic_span[1] - synthetic_span[0]
    expected_start_fraction = smoke_profile["window_start_seconds"] / span_duration
    expected_end_fraction = (smoke_profile["window_start_seconds"] + smoke_profile["window_seconds"]) / span_duration
    assert f"DATASET_START_FRACTION={expected_start_fraction!r}" in sbatch_remote_text
    assert f"DATASET_END_FRACTION={expected_end_fraction!r}" in sbatch_remote_text
    expected_recipe_hash = submit_module.compute_recipe_hash(
        {relative_path: relative_path.encode() for relative_path in config["environment_recipe_files"]})
    # The marker read is the one named by the recipe hash; the env path and expiry come from it.
    marker_reads = [command[-1] for command in issued_commands
                    if command[0] == "ssh" and command[-1].startswith("cat ")]
    assert marker_reads == [f"cat {absolute_stage_root}/envs/splatad-{expected_recipe_hash}.marker"]
    assert f"ENV_PREFIX={scratch_env_prefix}" in sbatch_remote_text
    assert f"ENV_EXPIRES_EPOCH={expires_epoch}" in sbatch_remote_text
    assert f"COPY_BACK_BWLIMIT_KIB={config['copy_back_bwlimit_kib']}" in sbatch_remote_text
    assert f"CODE_DIR={absolute_stage_root}/code/{full_sha}" in sbatch_remote_text
    assert sbatch_remote_text.endswith(f"{absolute_stage_root}/code/{full_sha}/scripts/slurm/splatad_run.sbatch")
    assert "get-user-env" not in sbatch_remote_text
    ledger_lines = (tmp_path / "submitted.tsv").read_text().splitlines()
    assert ledger_lines[1].split("\t")[:2] == [run_id, "4242"]


def test_submit_run_refuses_when_environment_not_built(monkeypatch, tmp_path):
    """No sbatch is issued when the env for the committed recipe has no completion marker."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    monkeypatch.setattr(submit_module, "run_command",
                        make_fake_cluster("a" * 40, "/h/example/viking-splatad", None, issued_commands))
    monkeypatch.setattr(submit_module, "read_committed_file", lambda git_sha, relative_path: b"x")
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    with pytest.raises(RuntimeError, match="build-env"):
        submit_module.submit_run(submit_module.load_slurm_config(), "may_short_grass", "smoke",
                                 skip_gates=True)
    assert not any(command[0] == "ssh" and " sbatch " in f" {command[-1]} " for command in issued_commands)


def test_window_fractions_known_values_and_inverse():
    """f = s / D and (s + w) / D; mapping back with ADDataParser's formula recovers the window."""
    submit_module = load_submit_module()
    earliest_time, latest_time = 10.0, 210.0
    window_start_seconds, window_seconds = 50.0, 15.0
    start_fraction, end_fraction = submit_module.window_fractions(
        earliest_time, latest_time, window_start_seconds, window_seconds)
    duration = latest_time - earliest_time
    assert start_fraction == pytest.approx(window_start_seconds / duration)
    assert end_fraction == pytest.approx((window_start_seconds + window_seconds) / duration)
    # ADDataParser keeps [t0 + f_start * D, t0 + f_end * D]; that must be the requested window.
    assert earliest_time + start_fraction * duration == pytest.approx(earliest_time + window_start_seconds)
    assert earliest_time + end_fraction * duration == pytest.approx(
        earliest_time + window_start_seconds + window_seconds)


def test_window_fractions_rejects_windows_outside_the_recording():
    """A window past the end, of zero length, or starting before the recording is refused."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError):
        submit_module.window_fractions(0.0, 100.0, 90.0, 20.0)
    with pytest.raises(ValueError):
        submit_module.window_fractions(0.0, 100.0, 10.0, 0.0)
    with pytest.raises(ValueError):
        submit_module.window_fractions(0.0, 100.0, -1.0, 5.0)


def test_dataset_time_span_covers_every_sensor_stream(tmp_path):
    """Span = min/max over all '<sensor>__times' arrays, ignoring pose arrays."""
    submit_module = load_submit_module()
    camera_times = np.array([1.5, 2.5, 3.5])
    lidar_times = np.array([1.0, 2.0, 4.0])
    np.savez(tmp_path / "poses.npz", cam__times=camera_times, lidar__times=lidar_times,
             cam__poses_world_sensor=np.zeros((3, 4, 4)))
    assert submit_module.dataset_time_span(tmp_path) == (
        min(camera_times.min(), lidar_times.min()), max(camera_times.max(), lidar_times.max()))


def test_whole_recording_when_profile_has_no_window():
    """Without window keys the job gets fractions 0 and 1 (no dataset files read)."""
    submit_module = load_submit_module()
    assert submit_module.profile_window_fractions({"max_num_iterations": 1}, "unused") == (0.0, 1.0)


def test_half_specified_window_is_a_clear_error():
    """Only one of the two window keys is a config error, reported by name."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError, match="window_start_seconds"):
        submit_module.profile_window_fractions({"window_seconds": 15}, "unused")


def test_submit_sbatch_puts_parsable_right_after_the_sbatch_token(monkeypatch):
    """--parsable must follow `sbatch`, not land among the `env` assignments."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []

    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command; reply like `sbatch --parsable` (job id;cluster)."""
        issued_commands.append(list(command))
        return subprocess.CompletedProcess(command, 0, stdout="777;cluster\n", stderr="")

    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    sbatch_arguments = build_example_sbatch(submit_module, "0-01:00:00")
    assert submit_module.submit_sbatch("host", sbatch_arguments) == "777"
    remote_arguments = shlex.split(issued_commands[0][-1])
    sbatch_position = remote_arguments.index("sbatch")
    assert remote_arguments[sbatch_position + 1] == "--parsable"
    # Everything else is unchanged and in order.
    assert remote_arguments[:sbatch_position + 1] + remote_arguments[sbatch_position + 2:] == sbatch_arguments


def test_build_sbatch_command_rejects_non_identifier_variable_names():
    """`env` splits at the first '=', so a name like 'A=B' or 'A B' must be refused."""
    submit_module = load_submit_module()
    for bad_name in ("A=B", "A B", "1ABC"):
        with pytest.raises(ValueError, match="identifier"):
            submit_module.build_sbatch_command(
                partition="gpunodes", gres="gpu:rtx_a6000:1", cpus=1, memory_gb=1,
                time_limit="0-00:10:00", job_name="job", output_path="out",
                job_environment={bad_name: "value"}, script_path="script")


def test_parse_environment_marker_reads_key_values_and_rejects_incomplete():
    """key=value lines are parsed; a marker without env_prefix or expiry is an error."""
    submit_module = load_submit_module()
    marker = submit_module.parse_environment_marker("env_prefix=/s/env\nexpires_epoch=1700000000\nbuild_job=7\n")
    assert marker == {"env_prefix": "/s/env", "expires_epoch": "1700000000", "build_job": "7"}
    with pytest.raises(ValueError):
        submit_module.parse_environment_marker("env_prefix=/s/env\n")
    with pytest.raises(ValueError):
        submit_module.parse_environment_marker("env_prefix=/s/env\nexpires_epoch=soon\n")


def test_environment_outlives_is_strict_against_job_end():
    """Usable iff expiry > now + time limit (finishing exactly at expiry is not enough)."""
    submit_module = load_submit_module()
    now_epoch, time_limit_seconds = 1_000_000.0, 7200
    job_end_epoch = int(now_epoch) + time_limit_seconds
    assert submit_module.environment_outlives({"expires_epoch": str(job_end_epoch + 1)}, now_epoch, time_limit_seconds)
    assert not submit_module.environment_outlives({"expires_epoch": str(job_end_epoch)}, now_epoch, time_limit_seconds)


def test_submit_run_refuses_env_on_expiring_scratch(monkeypatch, tmp_path):
    """A built env whose scratch expires before the job could finish is refused; no sbatch."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    moment = datetime.datetime(2026, 10, 25, 1, 2, 3, tzinfo=datetime.timezone.utc)
    # Expires one second after submission: shorter than any profile's time limit.
    marker_text = f"env_prefix=/scratch/x/envs/splatad-y\nexpires_epoch={int(moment.timestamp()) + 1}\n"
    monkeypatch.setattr(submit_module, "run_command",
                        make_fake_cluster("b" * 40, "/h/example/viking-splatad", marker_text, issued_commands))
    monkeypatch.setattr(submit_module, "read_committed_file", lambda git_sha, relative_path: b"x")
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    with pytest.raises(RuntimeError, match="expires"):
        submit_module.submit_run(submit_module.load_slurm_config(), "may_short_grass", "smoke",
                                 skip_gates=True, now_utc=moment)
    assert not any(command[0] == "ssh" and " sbatch " in f" {command[-1]} " for command in issued_commands)


def build_example_job_environment(submit_module, profile_settings):
    """Job environment for a profile without a time window (no dataset read).

    Args:
        submit_module: Loaded submit_run module.
        profile_settings: Profile mapping under test.

    Returns:
        The exported variable mapping.
    """
    env_marker = {"env_prefix": "/scratch/envs/splatad-x", "expires_epoch": "1"}
    return submit_module.build_job_environment(profile_settings, "s", "run", "a" * 40, "/h/stage",
                                               env_marker, 1)


def test_job_environment_exports_the_profile_seed():
    """The profile's seed reaches the job verbatim as TRAIN_SEED (the sbatch's --machine.seed)."""
    submit_module = load_submit_module()
    smoke_profile = dict(submit_module.load_slurm_config()["profiles"]["smoke"])
    # Drop the window so no dataset is read; the seed path does not depend on it.
    smoke_profile.pop("window_start_seconds")
    smoke_profile.pop("window_seconds")
    job_environment = build_example_job_environment(submit_module, smoke_profile)
    assert job_environment["TRAIN_SEED"] == str(smoke_profile["seed"])


@pytest.mark.parametrize("bad_seed", [None, "42", 4.2, True])
def test_job_environment_requires_an_integer_seed(bad_seed):
    """Missing, string, float or bool seeds are refused rather than defaulted."""
    submit_module = load_submit_module()
    profile_settings = {"max_num_iterations": 1, "steps_per_save": 1, "render_pose_source": "test",
                        "render_downscale_factor": 1}
    if bad_seed is not None:
        profile_settings["seed"] = bad_seed
    with pytest.raises(ValueError, match="seed"):
        build_example_job_environment(submit_module, profile_settings)


def test_sbatch_passes_seed_and_renders_lidar_for_the_same_poses():
    """Static check of splatad_run.sbatch: seed to ns-train, lidar render on the camera render's splits."""
    sbatch_text = (PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch").read_text()
    assert '--machine.seed "$TRAIN_SEED"' in sbatch_text
    lidar_step_text = sbatch_text[sbatch_text.index('python "$CODE_DIR/scripts/08_render_lidar_pointcloud.py"'):]
    assert '--pose-source "$RENDER_POSE_SOURCE"' in lidar_step_text.split("\n\n")[0]
    assert '--dataset-dir "$STAGE_ROOT/data/$SESSION"' in lidar_step_text.split("\n\n")[0]
