"""Tests for ``submit_run.py clean``: which runs are deleted, and that every artifact of them goes.

A run is kept iff its job is queued/running or its job_info.txt proves success (``end_utc`` and
``split_render_check: match``, both written only by a run that finished every step), unless the
user names it. No cluster access: the remote listing and commands are replaced.
"""

import subprocess
from pathlib import Path

import pytest

from tests.unit.test_slurm_submit import load_submit_module

STAGE_ROOT = "/w/disk/someone/viking-splatad"
SUCCESSFUL_JOB_INFO = {"slurm_job_id": "1", "split_render_check": "match", "end_utc": "2026-10-26T00:00:00Z"}
CANCELLED_JOB_INFO = {"slurm_job_id": "2", "train_start_epoch_s": "100"}
FAILED_SPLIT_JOB_INFO = {"slurm_job_id": "3", "split_render_check": "FAILED (exit 1; no split record written)"}


def plan_by_run(plan):
    """Index a cleanup plan by run id."""
    return {decision["run_id"]: decision for decision in plan}


def test_listing_parser_reads_runs_job_info_and_active_job_scripts():
    """RUN / INFO_BEGIN..INFO_END / ACTIVE lines become runs, parsed job_info and active run ids."""
    submit_module = load_submit_module()
    listing_text = (
        "RUN\tdone-run\nINFO_BEGIN\nhostname: gpunode2\nenvironment build:\n  built x\n"
        "split_render_check: match\nend_utc: 2026-10-26T00:00:00Z\n\nINFO_END\n"
        "RUN\tpending-run\n"
        f"ACTIVE\t{STAGE_ROOT}/runs/pending-run/job.sbatch\n"
        "ACTIVE\t/some/other/script.sh\n"
    )
    remote_runs, active_run_ids = submit_module.parse_remote_run_listing(listing_text)
    assert remote_runs["pending-run"] is None
    # Indented and value-less lines ("environment build:", "  built x") are not keys.
    assert remote_runs["done-run"] == {"hostname": "gpunode2", "split_render_check": "match",
                                       "end_utc": "2026-10-26T00:00:00Z"}
    # Only a job whose script is a run's job.sbatch maps to a run.
    assert active_run_ids == {"pending-run"}


def test_unsuccessful_runs_are_deleted_and_successful_ones_kept():
    """Cancelled, failed-split-check, never-started and ledger-only runs go; successful ones stay."""
    submit_module = load_submit_module()
    remote_runs = {"ok": SUCCESSFUL_JOB_INFO, "cancelled": CANCELLED_JOB_INFO,
                   "failed": FAILED_SPLIT_JOB_INFO, "never-started": None}
    ledger_rows = [{"run_id": "ok"}, {"run_id": "gone-everywhere-else"}]
    plan = plan_by_run(submit_module.plan_cleanup(remote_runs, set(), {}, ledger_rows, []))
    assert {run_id for run_id, decision in plan.items() if decision["delete"]} == {
        "cancelled", "failed", "never-started", "gone-everywhere-else"}
    assert plan["ok"]["delete"] is False
    assert plan["gone-everywhere-else"]["in_ledger"] and not plan["gone-everywhere-else"]["on_cluster"]


def test_split_check_match_without_end_utc_is_not_success():
    """end_utc is written last: a run killed after the split check but before it is incomplete."""
    submit_module = load_submit_module()
    plan = plan_by_run(submit_module.plan_cleanup({"almost": {"split_render_check": "match"}}, set(), {}, [], []))
    assert plan["almost"]["delete"] is True


def test_active_runs_are_kept_even_without_job_info():
    """A pending job has no job_info.txt yet; it is recognised by its job script and kept."""
    submit_module = load_submit_module()
    plan = plan_by_run(submit_module.plan_cleanup({"pending": None}, {"pending"}, {}, [], []))
    assert plan["pending"]["delete"] is False


def test_named_runs_are_deleted_even_if_successful():
    """The user can delete a good run by naming it; other successful runs stay."""
    submit_module = load_submit_module()
    remote_runs = {"ok-a": SUCCESSFUL_JOB_INFO, "ok-b": SUCCESSFUL_JOB_INFO}
    plan = plan_by_run(submit_module.plan_cleanup(remote_runs, set(), {}, [], ["ok-a"]))
    assert plan["ok-a"]["delete"] is True and plan["ok-b"]["delete"] is False


@pytest.mark.parametrize("named_run_id, message", [
    ("running", "scancel"), ("nowhere", "not on the cluster"), ("../escape", "malformed"), ("a/b", "malformed")])
def test_bad_named_runs_are_refused(named_run_id, message):
    """Active, unknown or path-like names stop the whole clean before anything is decided."""
    submit_module = load_submit_module()
    with pytest.raises(ValueError, match=message):
        submit_module.plan_cleanup({"running": None}, {"running"}, {}, [], [named_run_id])


def test_cluster_record_wins_over_the_fetched_copy():
    """A run fetched mid-way (local copy incomplete) but finished on the cluster is kept, and vice versa."""
    submit_module = load_submit_module()
    plan = plan_by_run(submit_module.plan_cleanup(
        {"finished-later": SUCCESSFUL_JOB_INFO}, set(), {"finished-later": CANCELLED_JOB_INFO,
                                                         "only-local-ok": SUCCESSFUL_JOB_INFO}, [], []))
    assert plan["finished-later"]["delete"] is False
    assert plan["only-local-ok"]["delete"] is False and plan["only-local-ok"]["on_mac"]


def make_clean_environment(tmp_path: Path, monkeypatch):
    """Local runs/ folder, ledger and a fake cluster with one successful and one cancelled run.

    Returns:
        (submit module, local runs root, ledger path, list of remote commands run).
    """
    submit_module = load_submit_module()
    local_runs_root = tmp_path / "runs"
    for run_id, job_info_text in (("ok-run", "split_render_check: match\nend_utc: x\n"),
                                  ("cancelled-run", "train_start_epoch_s: 1\n")):
        (local_runs_root / run_id).mkdir(parents=True)
        (local_runs_root / run_id / "job_info.txt").write_text(job_info_text)
    (local_runs_root / "RUN_FOLDER_GLOSSARY.md").write_text("kept: not a run folder\n")
    ledger_path = local_runs_root / "submitted.tsv"
    submit_module.write_ledger(ledger_path, [
        {"run_id": "ok-run", "job_id": "1", "git_sha": "a", "session": "s", "profile": "p", "submitted_utc": "t"},
        {"run_id": "cancelled-run", "job_id": "2", "git_sha": "a", "session": "s", "profile": "p", "submitted_utc": "t"}])
    remote_commands = []

    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command and succeed."""
        remote_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    monkeypatch.setattr(submit_module, "resolve_ssh_target", lambda host: host)
    monkeypatch.setattr(submit_module, "resolve_remote_stage_root", lambda config, target: STAGE_ROOT)
    monkeypatch.setattr(submit_module, "list_remote_runs", lambda target, root: (
        {"ok-run": SUCCESSFUL_JOB_INFO, "cancelled-run": CANCELLED_JOB_INFO}, set()))
    return submit_module, local_runs_root, ledger_path, remote_commands


def test_dry_run_deletes_nothing(tmp_path, monkeypatch):
    """Without --yes: the plan is printed, no remote rm, no local change, ledger untouched."""
    submit_module, local_runs_root, ledger_path, remote_commands = make_clean_environment(tmp_path, monkeypatch)
    ledger_before = ledger_path.read_text()
    submit_module.clean_runs({"ssh_host": "host"}, [], False, local_runs_root, ledger_path)
    assert remote_commands == []
    assert (local_runs_root / "cancelled-run").is_dir()
    assert ledger_path.read_text() == ledger_before


def test_confirmed_clean_removes_every_artifact_of_the_cancelled_run_only(tmp_path, monkeypatch):
    """With --yes: cluster dir (rm -rf inside runs/), fetched copy and ledger row of the cancelled run go."""
    submit_module, local_runs_root, ledger_path, remote_commands = make_clean_environment(tmp_path, monkeypatch)
    submit_module.clean_runs({"ssh_host": "host"}, [], True, local_runs_root, ledger_path)
    assert remote_commands == [submit_module.ssh_command(
        "host", ["rm", "-rf", "--", f"{STAGE_ROOT}/runs/cancelled-run"])]
    assert not (local_runs_root / "cancelled-run").exists()
    assert (local_runs_root / "ok-run").is_dir() and (local_runs_root / "RUN_FOLDER_GLOSSARY.md").is_file()
    assert [row["run_id"] for row in submit_module.read_ledger(ledger_path)] == ["ok-run"]
    assert ledger_path.read_text().splitlines()[0] == "\t".join(submit_module.LEDGER_COLUMNS)
