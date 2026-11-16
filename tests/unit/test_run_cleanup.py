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


def test_cluster_only_plan_targets_only_named_fetched_runs():
    """Named, idle, fully fetched runs are planned; unnamed successful and cancelled runs are not."""
    submit_module = load_submit_module()
    remote_runs = {"ok-a": SUCCESSFUL_JOB_INFO, "ok-b": SUCCESSFUL_JOB_INFO, "cancelled": CANCELLED_JOB_INFO}
    local_runs = {"ok-a": SUCCESSFUL_JOB_INFO, "ok-b": SUCCESSFUL_JOB_INFO}
    # Duplicate names collapse to one deletion.
    assert submit_module.plan_cluster_only_cleanup(
        remote_runs, set(), local_runs, {"ok-a": [], "ok-b": []}, ["ok-a", "ok-a"]) == ["ok-a"]


@pytest.mark.parametrize("named_run_ids, message", [
    ([], "at least one run id"),
    (["../escape"], "malformed"),
    (["mac-only"], "not on the cluster"),
    (["running"], "scancel"),
    (["unfetched"], "no successful fetched copy"),
    (["fetched-incomplete"], "no successful fetched copy"),
    (["checkpoint-cut-off"], "not fully fetched"),
    (["never-compared"], "not fully fetched"),
    (["ok", "unfetched"], "no successful fetched copy"),
])
def test_cluster_only_refuses_unsafe_runs(named_run_ids, message):
    """No name, a bad name, an active run, or a Mac copy that is missing, unsuccessful, differs from the
    cluster, or was never compared refuses the whole call."""
    submit_module = load_submit_module()
    remote_runs = {name: SUCCESSFUL_JOB_INFO for name in
                   ("ok", "running", "unfetched", "fetched-incomplete", "checkpoint-cut-off", "never-compared")}
    local_runs = {"ok": SUCCESSFUL_JOB_INFO, "mac-only": SUCCESSFUL_JOB_INFO, "fetched-incomplete": CANCELLED_JOB_INFO,
                  "checkpoint-cut-off": SUCCESSFUL_JOB_INFO, "never-compared": SUCCESSFUL_JOB_INFO}
    unfetched_files_by_run = {"ok": [], "fetched-incomplete": [],
                              "checkpoint-cut-off": ["train/s/m/r/nerfstudio_models/step-000019999.ckpt"]}
    with pytest.raises(ValueError, match=message):
        submit_module.plan_cluster_only_cleanup(remote_runs, {"running"}, local_runs, unfetched_files_by_run,
                                                named_run_ids)


def test_pending_transfers_count_only_received_or_created_files():
    """Received (>f) and created (cf) files are pending; directory timestamp lines and deletions are not."""
    submit_module = load_submit_module()
    itemized_text = "\n".join([
        ">f.st...... train/model/step-000019999.ckpt",  # size+time differ: cut-off download
        ">f+++++++++ rerenders/guard/image 1.png",       # new on the cluster, path with a space
        ".d..t...... ./",                                # run root mtime moved because the Mac added eval/
        "cd+++++++++ rerenders/",                        # new directory: its files are listed separately
        "*deleting   stale.txt",                         # only with --delete, never a copy to the Mac
    ])
    assert submit_module.parse_pending_transfers(itemized_text) == [
        "train/model/step-000019999.ckpt", "rerenders/guard/image 1.png"]
    assert submit_module.parse_pending_transfers("") == []


def test_pending_transfers_read_openrsync_nine_character_codes():
    """macOS openrsync prints 9-character codes (captured from `rsync -an --itemize-changes` on the Mac,
    protocol 29): they must count as pending, not be skipped as an unknown length."""
    submit_module = load_submit_module()
    itemized_text = ">f.s..... cut.ckpt\n>f+++++++ sub/new file.png\n"
    assert submit_module.parse_pending_transfers(itemized_text) == ["cut.ckpt", "sub/new file.png"]


def test_pending_transfers_fail_closed_on_unknown_lines():
    """A line in no known form (e.g. a warning) counts as pending, so deletion is refused, not allowed."""
    submit_module = load_submit_module()
    assert submit_module.parse_pending_transfers("rsync: some unexpected notice\n") == ["some unexpected notice"]


def make_guard_sandbox(tmp_path: Path, squeue_body: str):
    """A fake run dir and a PATH whose `squeue` runs squeue_body, for executing the delete guard locally.

    Returns:
        (run dir, environment dict for subprocess).
    """
    run_dir = tmp_path / "stage" / "runs" / "some-run"
    run_dir.mkdir(parents=True)
    (run_dir / "job_info.txt").write_text("end_utc: x\n")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_squeue = fake_bin / "squeue"
    fake_squeue.write_text(f"#!/bin/bash\n{squeue_body}\n")
    fake_squeue.chmod(0o755)
    environment = {"PATH": f"{fake_bin}:/usr/bin:/bin"}
    return run_dir, environment


@pytest.mark.parametrize("squeue_body, expected_exit_code, run_dir_survives", [
    ("exit 1", 4, True),                                       # squeue failed: abort, never rm
    ('echo "$RUN_DIR/job.sbatch"', 3, True),                    # a job of this run is queued: refuse
    ('echo "$RUN_DIR-other/job.sbatch"', 0, False),             # only a prefix-sharing run: delete
    ("true", 0, False),                                         # empty queue: delete
])
def test_cluster_delete_guard_script_fails_closed(tmp_path, squeue_body, expected_exit_code, run_dir_survives):
    """Run the real guard script with bash and a fake squeue: rm happens only when squeue succeeded and
    lists no job script under the run dir."""
    submit_module = load_submit_module()
    run_dir, environment = make_guard_sandbox(tmp_path, squeue_body)
    environment["RUN_DIR"] = str(run_dir)
    completed = subprocess.run(["bash", "-c", submit_module.CLUSTER_RUN_DELETE_GUARD_SCRIPT, "bash", str(run_dir)],
                               env=environment, capture_output=True, text=True)
    assert completed.returncode == expected_exit_code, completed.stderr
    assert run_dir.is_dir() == run_dir_survives


def test_confirmed_cluster_only_clean_keeps_mac_copy_and_ledger(tmp_path, monkeypatch):
    """With --cluster-only --yes and a complete fetch: only the named run's cluster dir goes, behind a
    squeue guard in the same ssh call; fetched copies and ledger stay, the cancelled run is not auto-cleaned."""
    submit_module, local_runs_root, ledger_path, remote_commands = make_clean_environment(tmp_path, monkeypatch)
    ledger_before = ledger_path.read_text()
    submit_module.clean_runs({"ssh_host": "host"}, ["ok-run"], True, local_runs_root, ledger_path, cluster_only=True)
    # The fake rsync/find print nothing (complete fetch), so the last command is the guarded delete.
    assert remote_commands[-1] == submit_module.ssh_command(
        "host", ["bash", "-c", submit_module.CLUSTER_RUN_DELETE_GUARD_SCRIPT, "bash", f"{STAGE_ROOT}/runs/ok-run"])
    # No other command deletes anything, and nothing local changed.
    assert sum("rm -rf" in " ".join(command) for command in remote_commands) == 1
    assert (local_runs_root / "ok-run").is_dir() and (local_runs_root / "cancelled-run").is_dir()
    assert ledger_path.read_text() == ledger_before


def test_unfetched_files_report_missing_and_differing_latest_checkpoint(tmp_path, monkeypatch):
    """The cluster's latest step-*.ckpt is pending when the Mac lacks it, or when rsync says it differs."""
    submit_module = load_submit_module()
    remote_run_dir = f"{STAGE_ROOT}/runs/r"
    checkpoint_relative = "train/s/m/r/nerfstudio_models/step-000020000.ckpt"
    older_relative = "train/s/m/r/nerfstudio_models/step-000018000.ckpt"
    checkpoint_rsync_output = {"text": ""}

    def fake_run_command(command, capture_output=True, input_text=None):
        """find lists two checkpoints; the checkpoint rsync prints the configured text; others nothing."""
        joined = " ".join(command)
        if "find" in joined:
            stdout = f"{remote_run_dir}/{older_relative}\n{remote_run_dir}/{checkpoint_relative}\n"
        elif command[0] == "rsync" and "--include=step-000020000.ckpt" in command:
            # The checkpoint check is a directory-mode rsync filtered to the latest checkpoint only.
            assert "--exclude=*" in command and joined.endswith("/nerfstudio_models/")
            stdout = checkpoint_rsync_output["text"]
        else:
            stdout = ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    local_run_dir = tmp_path / "r"
    # Step 1: no local checkpoint file -> the latest (step 20000, not 18000) is pending.
    assert submit_module.list_unfetched_files("host", remote_run_dir, local_run_dir) == [checkpoint_relative]
    # Step 2: the file exists; rsync says it differs (openrsync code) -> pending.
    (local_run_dir / checkpoint_relative).parent.mkdir(parents=True)
    (local_run_dir / checkpoint_relative).write_bytes(b"cut")
    checkpoint_rsync_output["text"] = ">f....... step-000020000.ckpt\n"
    assert submit_module.list_unfetched_files("host", remote_run_dir, local_run_dir) == [checkpoint_relative]
    # Step 3: rsync reports nothing -> complete.
    checkpoint_rsync_output["text"] = ""
    assert submit_module.list_unfetched_files("host", remote_run_dir, local_run_dir) == []


def test_cluster_only_refuses_when_rsync_dry_run_finds_unfetched_files(tmp_path, monkeypatch):
    """A file the cluster has and the Mac lacks (e.g. a later render) blocks deletion: no rm is issued."""
    submit_module, local_runs_root, ledger_path, remote_commands = make_clean_environment(tmp_path, monkeypatch)

    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command; rsync dry runs report one file missing locally."""
        remote_commands.append(command)
        stdout = ">f+++++++++ rerenders/guard/x.png\n" if command[0] == "rsync" else ""
        return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")
    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    with pytest.raises(ValueError, match="not fully fetched"):
        submit_module.clean_runs({"ssh_host": "host"}, ["ok-run"], True, local_runs_root, ledger_path,
                                 cluster_only=True)
    assert not any("rm -rf" in " ".join(command) for command in remote_commands)


def test_cluster_only_dry_run_deletes_nothing(tmp_path, monkeypatch):
    """Without --yes: the plan is printed and no delete command is issued."""
    submit_module, local_runs_root, ledger_path, remote_commands = make_clean_environment(tmp_path, monkeypatch)
    assert submit_module.clean_runs({"ssh_host": "host"}, ["ok-run"], False, local_runs_root, ledger_path,
                                    cluster_only=True) == ["ok-run"]
    assert not any("rm -rf" in " ".join(command) for command in remote_commands)
