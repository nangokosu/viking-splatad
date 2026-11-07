"""Tests of the submit-time reference split guard (scripts/slurm/submit_run.py).

The gates' dataset check (scripts/16) rebuilds the split with the session YAML values; the job uses the
profile's values where a profile overrides them. The guard compares the job's effective settings with the
session's committed record: a mismatch is refused unless a reason is given (firm, not locked).

Not a registered transform (no T-number). Expected values come from the session YAMLs and the committed
records, never from the code under test. Nothing touches git, ssh or the cluster (``run_command`` is faked).
"""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from tests.unit.test_slurm_submit import load_submit_module, make_fake_cluster
from viking_splatad.config import load_session_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_CONFIG_DIRECTORY = PROJECT_ROOT / "configs" / "sessions"
COMMITTED_RECORD_PATHS = sorted((PROJECT_ROOT / "configs" / "reference_splits").glob("*.json"))
# Profile keys that change the split a job uses (window, train fraction, render fraction).
SPLIT_OVERRIDE_PROFILE_KEYS = ("window_start_seconds", "window_seconds", "train_fraction", "test_render_fraction")
# The example session of the end-to-end tests: it has a committed record and a smoke-overriding profile.
EXAMPLE_SESSION = "may_short_grass"
# Synthetic recording span (s) so the smoke profile's window does not read a real dataset.
SYNTHETIC_SPAN = (5.0, 205.0)
EXAMPLE_STAGE_ROOT = "/w/example/user/viking-splatad"
# Job id the fake cluster's sbatch prints (make_fake_cluster).
FAKE_JOB_ID = "4242"
# A git SHA-1 is 20 bytes = 40 hex characters; any 40-character string stands in for HEAD.
GIT_SHA_HEX_LENGTH = 40
EXAMPLE_FULL_SHA = "a" * GIT_SHA_HEX_LENGTH
# Arbitrary fixed submission time (only fixes the run id; no assertion depends on its value).
EXAMPLE_SUBMIT_TIME = datetime.datetime(2026, 11, 6, tzinfo=datetime.timezone.utc)


def session_reference_settings(session: str) -> Dict[str, object]:
    """The settings a reference run of ``session`` uses: the session YAML's split, whole recording.

    Args:
        session: Session name (``configs/sessions/<session>.yaml``).

    Returns:
        Every key of ``RUN_REFERENCE_SPLIT_KEYS``.
    """
    split_section = load_session_config(SESSION_CONFIG_DIRECTORY / f"{session}.yaml")["split"]
    # Whole recording: a reference profile sets no window, so the dataset fractions span [0, 1].
    return {"dataset_start_fraction": 0.0, "dataset_end_fraction": 1.0,
            "train_fraction": split_section["train_fraction"], "test_split_method": split_section["method"],
            "test_render_fraction": split_section["test_render_fraction"]}


def write_record(reference_directory: Path, session: str, settings: Dict[str, object]) -> Path:
    """Write a minimal record (only ``settings`` is read by the guard).

    Args:
        reference_directory: Directory to write into (created).
        session: Session name (file stem).
        settings: The record's settings.

    Returns:
        The record path.
    """
    reference_directory.mkdir(parents=True, exist_ok=True)
    record_path = reference_directory / f"{session}.json"
    record_path.write_text(json.dumps({"settings": settings}))
    return record_path


def reference_profile_of_record(record: dict) -> str:
    """The profile of a record's source run, parsed from its run id.

    Args:
        record: A committed reference record.

    Returns:
        The profile: run ids are ``<session>-<profile>-<YYYYmmdd>-<HHMMSS>-<sha7>`` (build_run_id).
    """
    session = record["settings"]["session"]
    run_id = record["source"]["run_id"]
    # Drop "<session>-", then the three trailing dash fields (date, time, sha).
    return run_id[len(session) + 1:].rsplit("-", 3)[0]


def test_setting_differences_empty_when_every_key_agrees():
    """Identical settings -> no differences."""
    submit_module = load_submit_module()
    settings = session_reference_settings(EXAMPLE_SESSION)
    assert submit_module.reference_split_setting_differences(settings, dict(settings)) == []


@pytest.mark.parametrize("changed_key", ["dataset_start_fraction", "dataset_end_fraction", "train_fraction",
                                         "test_split_method", "test_render_fraction"])
def test_setting_differences_name_exactly_the_changed_key(changed_key):
    """Changing one key -> exactly one line, naming that key."""
    submit_module = load_submit_module()
    record_settings = session_reference_settings(EXAMPLE_SESSION)
    run_settings = {**record_settings, changed_key: "changed"}
    differences = submit_module.reference_split_setting_differences(record_settings, run_settings)
    assert len(differences) == 1 and differences[0].startswith(changed_key)


def test_guard_is_silent_for_a_session_without_a_record(tmp_path):
    """No record file -> nothing to compare, whatever the run settings."""
    submit_module = load_submit_module()
    run_settings = {**session_reference_settings(EXAMPLE_SESSION), "train_fraction": 0.5}
    submit_module.check_run_split_against_reference(EXAMPLE_SESSION, run_settings, None, tmp_path)


@pytest.mark.parametrize("reason", [None, "", "   "])
def test_guard_refuses_a_mismatch_without_a_real_reason(tmp_path, reason):
    """Mismatch and no (or a blank) reason -> GateFailure naming the key and the record."""
    submit_module = load_submit_module()
    record_settings = session_reference_settings(EXAMPLE_SESSION)
    record_path = write_record(tmp_path, EXAMPLE_SESSION, record_settings)
    run_settings = {**record_settings, "train_fraction": 0.8}
    with pytest.raises(submit_module.GateFailure) as failure_info:
        submit_module.check_run_split_against_reference(EXAMPLE_SESSION, run_settings, reason, tmp_path)
    assert "train_fraction" in str(failure_info.value) and str(record_path) in str(failure_info.value)


def test_guard_lets_a_mismatch_through_with_a_reason_and_warns(tmp_path, capsys):
    """Mismatch with a reason -> no exception; the warning names the reason and says not comparable."""
    submit_module = load_submit_module()
    record_settings = session_reference_settings(EXAMPLE_SESSION)
    write_record(tmp_path, EXAMPLE_SESSION, record_settings)
    run_settings = {**record_settings, "test_render_fraction": 0.5}
    submit_module.check_run_split_against_reference(EXAMPLE_SESSION, run_settings, "ablation of renders", tmp_path)
    printed_text = capsys.readouterr().out
    assert "ablation of renders" in printed_text and "NOT comparable" in printed_text


@pytest.mark.parametrize("record_path", COMMITTED_RECORD_PATHS, ids=lambda path: path.stem)
def test_committed_reference_profile_and_session_yaml_match_the_record(record_path):
    """Firm settings: each record's source profile overrides no split key, and its session YAML gives the record's values."""
    submit_module = load_submit_module()
    record = json.loads(record_path.read_text())
    profile = reference_profile_of_record(record)
    profile_settings = submit_module.load_slurm_config()["profiles"][profile]
    # The reference profile uses the session's own split (no override), so a collaborator's variant of it does too.
    assert [key for key in SPLIT_OVERRIDE_PROFILE_KEYS if key in profile_settings] == []
    # With no override the job's settings are the session YAML's; they must be the record's.
    expected_settings = session_reference_settings(record["settings"]["session"])
    assert submit_module.reference_split_setting_differences(record["settings"], expected_settings) == []


def test_may_and_june_both_have_committed_records():
    """The two Scene A sessions are both guarded."""
    assert {path.stem for path in COMMITTED_RECORD_PATHS} >= {"may_short_grass", "june_tall_grass"}


def prepare_gated_submit(monkeypatch, tmp_path, record_settings: Dict[str, object]) -> Tuple[Any, List[List[str]]]:
    """Fake cluster with a built environment, gates passing, and a temporary record for EXAMPLE_SESSION.

    Args:
        monkeypatch: pytest fixture.
        tmp_path: pytest temporary directory.
        record_settings: Settings of the temporary record.

    Returns:
        (the loaded submit module with the fakes installed, the list that records every issued command).
    """
    submit_module = load_submit_module()
    issued_commands: List[List[str]] = []
    marker_text = f"env_prefix={EXAMPLE_STAGE_ROOT}/envs/splatad-x\nbuilt_utc=t\n"
    monkeypatch.setattr(submit_module, "run_command",
                        make_fake_cluster(EXAMPLE_FULL_SHA, EXAMPLE_STAGE_ROOT, marker_text, issued_commands))
    monkeypatch.setattr(submit_module, "run_gates", lambda session: None)
    monkeypatch.setattr(submit_module, "read_committed_file", lambda git_sha, relative_path: b"x")
    monkeypatch.setattr(submit_module, "SUBMITTED_LEDGER_PATH", tmp_path / "submitted.tsv")
    monkeypatch.setattr(submit_module, "dataset_time_span", lambda session_dir: SYNTHETIC_SPAN)
    # The dataset's manifest fraction is the session YAML's (scripts/16 checks that on the real data).
    monkeypatch.setattr(submit_module, "dataset_train_fraction",
                        lambda session, intermediate_root=None: record_settings["train_fraction"])
    monkeypatch.setattr(submit_module, "REFERENCE_SPLIT_DIRECTORY", tmp_path / "reference_splits")
    write_record(tmp_path / "reference_splits", EXAMPLE_SESSION, record_settings)
    return submit_module, issued_commands


def sbatch_was_issued(issued_commands: List[List[str]]) -> bool:
    """Whether any recorded command is a remote sbatch.

    Args:
        issued_commands: Commands recorded by the fake cluster.

    Returns:
        True iff an ``ssh ... sbatch ...`` command was issued.
    """
    return any(command[0] == "ssh" and " sbatch " in f" {command[-1]} " for command in issued_commands)


def test_gated_submit_of_an_overriding_profile_is_refused_before_the_run_dir_exists(monkeypatch, tmp_path):
    """smoke overrides window and fractions -> GateFailure, no run dir, no sbatch, no ledger row."""
    submit_module, issued_commands = prepare_gated_submit(
        monkeypatch, tmp_path, session_reference_settings(EXAMPLE_SESSION))
    with pytest.raises(submit_module.GateFailure, match="reference record"):
        submit_module.submit_run(submit_module.load_slurm_config(), EXAMPLE_SESSION, "smoke", skip_gates=False)
    assert not sbatch_was_issued(issued_commands)
    # Resolving the stage root may mkdir the stage root itself (idempotent); the run dir must not be made.
    assert not any(command[0] == "ssh" and f"{EXAMPLE_STAGE_ROOT}/runs/" in command[-1] for command in issued_commands)
    assert not (tmp_path / "submitted.tsv").exists()


def test_gated_submit_of_an_overriding_profile_with_a_reason_is_submitted(monkeypatch, tmp_path):
    """The same run with --allow-split-change reaches sbatch."""
    submit_module, issued_commands = prepare_gated_submit(
        monkeypatch, tmp_path, session_reference_settings(EXAMPLE_SESSION))
    job_id, _ = submit_module.submit_run(
        submit_module.load_slurm_config(), EXAMPLE_SESSION, "smoke", skip_gates=False,
        now_utc=EXAMPLE_SUBMIT_TIME,
        allow_split_change_reason="short debugging window")
    assert job_id == FAKE_JOB_ID and sbatch_was_issued(issued_commands)


def test_gated_submit_of_the_reference_profile_passes_the_guard(monkeypatch, tmp_path):
    """may_scene_a uses the session's split -> no reason needed, the job is submitted."""
    submit_module, issued_commands = prepare_gated_submit(
        monkeypatch, tmp_path, session_reference_settings(EXAMPLE_SESSION))
    job_id, _ = submit_module.submit_run(
        submit_module.load_slurm_config(), EXAMPLE_SESSION, "may_scene_a", skip_gates=False,
        now_utc=EXAMPLE_SUBMIT_TIME)
    assert job_id == FAKE_JOB_ID and sbatch_was_issued(issued_commands)


def test_skipping_the_gates_for_a_guarded_session_warns(monkeypatch, tmp_path, capsys):
    """skip_gates on a session with a record -> submitted, but a warning says the split is unchecked."""
    submit_module, issued_commands = prepare_gated_submit(
        monkeypatch, tmp_path, session_reference_settings(EXAMPLE_SESSION))
    submit_module.submit_run(submit_module.load_slurm_config(), EXAMPLE_SESSION, "smoke", skip_gates=True,
                             now_utc=EXAMPLE_SUBMIT_TIME)
    assert sbatch_was_issued(issued_commands)
    assert "NOT checked" in capsys.readouterr().out


def test_submit_cli_passes_the_reason_through(monkeypatch):
    """`submit ... --allow-split-change R` hands R to submit_run."""
    submit_module = load_submit_module()
    received_arguments: Dict[str, object] = {}
    monkeypatch.setattr(submit_module, "submit_run",
                        lambda config, session, profile, skip_gates, allow_split_change_reason=None:
                        received_arguments.update(reason=allow_split_change_reason))
    assert submit_module.main(["submit", "--session", EXAMPLE_SESSION, "--profile", "smoke",
                               "--allow-split-change", "why"]) == 0
    assert received_arguments == {"reason": "why"}
