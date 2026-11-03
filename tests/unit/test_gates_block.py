"""Gate behaviour of scripts/slurm/submit_run.py: a failing gate must stop everything before the cluster.

All commands go through ``run_command``, replaced here by fakes, so nothing touches git, docker or ssh.
Expected outcomes follow from the documented rule: a run is submitted only when the tree is clean,
the full suite passes and the contract validates (CLAUDE.md, "GPU training on Slurm", step 1).
"""

import subprocess
from typing import Callable, Dict, List

import pytest

from tests.unit.test_slurm_submit import load_submit_module, make_fake_cluster

# Names used to recognise the three gate commands in the recorded command lists.
CONTRACT_SCRIPT_FRAGMENT = "scripts/06_validate_contract.py"


def build_gate_fake(issued_commands: List[List[str]], status_text: str, test_exit: int,
                    contract_exit: int) -> Callable:
    """Build a run_command stand-in that answers the three gate commands with chosen outcomes.

    Args:
        issued_commands: List that records every command.
        status_text: Porcelain text returned by ``git status --porcelain``.
        test_exit: Exit code of the test-suite wrapper.
        contract_exit: Exit code of the contract validator.

    Returns:
        The fake ``run_command``.
    """
    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command and return the canned result for its kind."""
        issued_commands.append(list(command))
        if command[:2] == ["git", "status"]:
            return subprocess.CompletedProcess(command, 0, stdout=status_text, stderr="")
        if any(CONTRACT_SCRIPT_FRAGMENT in part for part in command):
            return subprocess.CompletedProcess(command, contract_exit, stdout="", stderr="")
        return subprocess.CompletedProcess(command, test_exit, stdout="", stderr="")
    return fake_run_command


def test_failing_test_suite_raises_and_skips_the_contract_gate(monkeypatch):
    """Suite exit 1 -> GateFailure naming the suite, and the contract validator is never run."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    monkeypatch.setattr(submit_module, "run_command", build_gate_fake(issued_commands, "", 1, 0))
    with pytest.raises(submit_module.GateFailure, match="test_in_image"):
        submit_module.run_gates("example_session")
    assert not any(CONTRACT_SCRIPT_FRAGMENT in part for command in issued_commands for part in command)


def test_failing_contract_validation_raises(monkeypatch):
    """Suite passes but contract exit 2 -> GateFailure naming the contract."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    monkeypatch.setattr(submit_module, "run_command", build_gate_fake(issued_commands, "", 0, 2))
    with pytest.raises(submit_module.GateFailure, match="contract"):
        submit_module.run_gates("example_session")
    # All three gate commands were attempted in order: status, suite, contract.
    assert len(issued_commands) == 3


def test_all_gates_passing_does_not_raise(monkeypatch):
    """Control: with every exit code 0 the gates return normally (the failures above are specific)."""
    submit_module = load_submit_module()
    monkeypatch.setattr(submit_module, "run_command", build_gate_fake([], "", 0, 0))
    submit_module.run_gates("example_session")


def test_failing_git_status_is_a_gate_failure(monkeypatch):
    """A git status that itself fails must not be read as 'clean' (suite and contract would pass)."""
    submit_module = load_submit_module()

    def fake_run_command(command, capture_output=True, input_text=None):
        """Only git status fails; every other command succeeds."""
        exit_code = 128 if command[:2] == ["git", "status"] else 0
        return subprocess.CompletedProcess(command, exit_code, stdout="", stderr="not a repo")

    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    with pytest.raises(submit_module.GateFailure, match="git status"):
        submit_module.run_gates("example_session")


@pytest.mark.parametrize("failing_gate", ["suite", "contract", "dirty"])
def test_submit_run_with_a_failing_gate_issues_no_remote_command_and_no_ledger_row(
        monkeypatch, tmp_path, failing_gate):
    """Gates on (skip_gates False): any failing gate -> GateFailure, zero ssh, nothing in the ledger."""
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    ledger_path = tmp_path / "submitted.tsv"
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", ledger_path)
    monkeypatch.setattr(submit_module, "run_command", build_gate_fake(
        issued_commands, " M src/a.py\n" if failing_gate == "dirty" else "",
        1 if failing_gate == "suite" else 0, 1 if failing_gate == "contract" else 0))
    with pytest.raises(submit_module.GateFailure):
        submit_module.submit_run(submit_module.load_slurm_config(), "may_short_grass", "smoke",
                                 skip_gates=False)
    assert not any(command[0] in ("ssh", "scp", "rsync") for command in issued_commands)
    assert not ledger_path.exists()


@pytest.mark.parametrize("skip_gates, expected_gate_calls", [(False, ["may_short_grass"]), (True, [])])
def test_submit_run_runs_the_gates_unless_skipped(monkeypatch, tmp_path, skip_gates, expected_gate_calls):
    """skip_gates False calls run_gates once with the session; True never calls it."""
    submit_module = load_submit_module()
    gate_calls: List[str] = []
    issued_commands: List[List[str]] = []
    monkeypatch.setattr(submit_module, "run_gates", lambda session: gate_calls.append(session))
    monkeypatch.setattr(submit_module, "run_command", make_fake_cluster(
        "a" * 40, "/w/example/user/viking-splatad", None, issued_commands))
    monkeypatch.setattr(submit_module, "read_committed_file", lambda git_sha, relative_path: b"x")
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    monkeypatch.setattr(submit_module, "dataset_time_span", lambda session_dir: (5.0, 205.0))
    # No env marker -> RuntimeError right after the gates; enough to observe the gate call.
    with pytest.raises(RuntimeError, match="build-env"):
        submit_module.submit_run(submit_module.load_slurm_config(), "may_short_grass", "smoke",
                                 skip_gates=skip_gates)
    assert gate_calls == expected_gate_calls


@pytest.mark.parametrize("status_line", [
    "M  src/staged_modified.py",      # staged modification (index column only)
    "A  src/new_staged.py",           # staged addition
    " D src/deleted_in_tree.py",      # deleted in the work tree, not staged
    "D  src/deleted_staged.py",       # deleted and staged
    "R  old.py -> new.py",            # staged rename
    "MM src/both.py",                 # staged and further modified
    " M src/work_tree_modified.py",   # unstaged modification
])
def test_check_clean_tree_refuses_every_tracked_change_kind(monkeypatch, status_line):
    """Each porcelain XY code except '??' is a tracked change and fails, naming the path."""
    submit_module = load_submit_module()
    monkeypatch.setattr(submit_module, "run_command",
                        lambda command, capture_output=True, input_text=None:
                        subprocess.CompletedProcess(command, 0, stdout=status_line + "\n", stderr=""))
    with pytest.raises(submit_module.GateFailure) as failure_info:
        submit_module.check_clean_tree()
    assert status_line in str(failure_info.value)


def test_check_clean_tree_returns_only_the_untracked_paths(monkeypatch):
    """Untracked entries are returned (path after the '?? '), never raised on."""
    submit_module = load_submit_module()
    monkeypatch.setattr(submit_module, "run_command",
                        lambda command, capture_output=True, input_text=None: subprocess.CompletedProcess(
                            command, 0, stdout="?? notes.txt\n?? dir/other.py\n", stderr=""))
    assert submit_module.check_clean_tree() == ["notes.txt", "dir/other.py"]


def test_check_clean_tree_mixed_untracked_and_staged_still_fails(monkeypatch):
    """One staged entry among untracked ones is enough to fail."""
    submit_module = load_submit_module()
    monkeypatch.setattr(submit_module, "run_command",
                        lambda command, capture_output=True, input_text=None: subprocess.CompletedProcess(
                            command, 0, stdout="?? a.txt\nA  b.py\n?? c.txt\n", stderr=""))
    with pytest.raises(submit_module.GateFailure, match="b.py"):
        submit_module.check_clean_tree()
