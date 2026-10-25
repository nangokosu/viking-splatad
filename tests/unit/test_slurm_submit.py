"""Tests of the Slurm submission helpers (scripts/slurm/submit_run.py).

All external commands go through ``run_command``, which these tests replace, so nothing touches
git, ssh or the cluster. Expected values are computed here from the stated unit definitions.
"""

import datetime
import hashlib
import importlib.util
import math
import subprocess
from pathlib import Path
from typing import List

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
    """Resources are all on the command line; --get-user-env never appears."""
    submit_module = load_submit_module()
    sbatch_arguments = build_example_sbatch(submit_module, "0-06:00:00")
    assert sbatch_arguments[0] == "sbatch"
    assert "--partition=gpunodes" in sbatch_arguments
    assert "--gres=gpu:rtx_a6000:1" in sbatch_arguments
    assert sbatch_arguments[sbatch_arguments.index("-c") + 1] == str(A6000_NODE_CPUS)
    assert f"--mem={EXAMPLE_MEMORY_GB}G" in sbatch_arguments
    assert sbatch_arguments[sbatch_arguments.index("-t") + 1] == "0-06:00:00"
    assert "--export=ALL,RUN_ID=r,SESSION=example" in sbatch_arguments
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


def make_fake_cluster(full_sha: str, absolute_stage_root: str, environment_built: bool,
                      issued_commands: List[List[str]]):
    """Build a run_command stand-in answering git, stage-root resolution, test -f, sinfo, sbatch.

    Args:
        full_sha: SHA returned by ``git rev-parse HEAD``.
        absolute_stage_root: Path printed by the remote ``pwd -P``.
        environment_built: Exit status of the remote ``test -f`` for the env marker.
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
        if remote_text.startswith("test -f"):
            return subprocess.CompletedProcess(command, 0 if environment_built else 1, stdout="", stderr="")
        if remote_text.startswith("sinfo"):
            return subprocess.CompletedProcess(command, 0, stdout=SYNTHETIC_SINFO_TEXT, stderr="")
        if remote_text.startswith("sbatch"):
            return subprocess.CompletedProcess(command, 0, stdout="4242\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    return fake_run_command


def test_submit_run_issues_expected_remote_commands(monkeypatch, tmp_path):
    """End-to-end submit with mocked ssh: sinfo-derived resources, env and code paths reach sbatch."""
    submit_module = load_submit_module()
    full_sha = "fedcba9876543210fedcba9876543210fedcba98"
    absolute_stage_root = "/h/example/viking-splatad"
    issued_commands: List[List[str]] = []
    monkeypatch.setattr(submit_module, "run_command",
                        make_fake_cluster(full_sha, absolute_stage_root, True, issued_commands))
    # Committed recipe files replaced by fixed bytes so the expected env name is computable here.
    monkeypatch.setattr(submit_module, "read_committed_file",
                        lambda git_sha, relative_path: relative_path.encode())
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    monkeypatch.delenv("VIKING_SLURM_USER", raising=False)
    config = submit_module.load_slurm_config()
    moment = datetime.datetime(2026, 10, 25, 1, 2, 3, tzinfo=datetime.timezone.utc)
    job_id, run_id = submit_module.submit_run(config, "may_short_grass", "smoke", skip_gates=True,
                                              now_utc=moment)
    assert job_id == "4242"
    assert run_id == f"may_short_grass-smoke-20261025-010203-{full_sha[:7]}"
    sbatch_remote_text = next(command[-1] for command in issued_commands
                              if command[0] == "ssh" and command[-1].startswith("sbatch"))
    expected_memory_gb = math.floor(config["mem_fraction"] * A6000_NODE_MEMORY_MB / 1024)
    assert f"--mem={expected_memory_gb}G" in sbatch_remote_text
    assert f"-c {A6000_NODE_CPUS}" in sbatch_remote_text
    expected_recipe_hash = submit_module.compute_recipe_hash(
        {relative_path: relative_path.encode() for relative_path in config["environment_recipe_files"]})
    assert f"ENV_PREFIX={absolute_stage_root}/envs/splatad-{expected_recipe_hash}" in sbatch_remote_text
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
                        make_fake_cluster("a" * 40, "/h/example/viking-splatad", False, issued_commands))
    monkeypatch.setattr(submit_module, "read_committed_file", lambda git_sha, relative_path: b"x")
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    with pytest.raises(RuntimeError, match="build-env"):
        submit_module.submit_run(submit_module.load_slurm_config(), "may_short_grass", "smoke",
                                 skip_gates=True)
    assert not any(command[0] == "ssh" and command[-1].startswith("sbatch") for command in issued_commands)
