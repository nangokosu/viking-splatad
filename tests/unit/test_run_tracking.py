"""Tests for ``submit_run.py runs``: the session-start check of pending, running and unfinished runs.

Each run gets one status and next action from the ledger, the cluster run folders, the user's
squeue and the fetched copies. No cluster access: the remote listing and commands are replaced.
"""

import subprocess

from tests.unit.test_slurm_submit import load_submit_module

STAGE_ROOT = "/w/disk/someone/viking-splatad"
SUCCESSFUL_JOB_INFO = {"start_utc": "s", "train_elapsed_s": "10", "overfitting_plot": "written",
                       "split_render_check": "match", "end_utc": "2026-10-29T00:00:00Z"}
RESUMABLE_JOB_INFO = {"start_utc": "s", "train_elapsed_s": "10", "overfitting_plot": "written",
                      "render_start_epoch_s": "11"}
PLOT_FAILED_JOB_INFO = {"start_utc": "s", "train_elapsed_s": "10", "overfitting_plot": "FAILED (exit 1)",
                        "split_render_check": "match"}
KILLED_IN_TRAINING_JOB_INFO = {"start_utc": "s", "train_start_epoch_s": "5"}


def plan_by_run(tracking_plan):
    """Index a tracking plan by run id."""
    return {entry["run_id"]: entry for entry in tracking_plan}


def squeue_line(job_id, state, reason, start, script_path):
    """One squeue line in the SQUEUE_TRACK_FIELDS order (space-separated, path last)."""
    return " ".join((job_id, state, reason, start, script_path))


def test_job_script_owner_maps_every_job_layout_to_its_run():
    """train, render-only and resume job scripts map to their run; other scripts to None."""
    submit_module = load_submit_module()
    assert submit_module.job_script_owner(f"{STAGE_ROOT}/runs/run-a/job.sbatch") == ("run-a", "train")
    assert submit_module.job_script_owner(
        f"{STAGE_ROOT}/runs/run-a/rerenders/render-x/job.sbatch") == ("run-a", "render")
    assert submit_module.job_script_owner(f"{STAGE_ROOT}/runs/run-a/resume/job.sbatch") == ("run-a", "resume")
    assert submit_module.job_script_owner(f"{STAGE_ROOT}/code/sha/scripts/slurm/build_env.sbatch") is None


def test_squeue_parser_keeps_run_jobs_and_paths_with_spaces():
    """Fields split on whitespace except the last (path); non-run and malformed lines are dropped."""
    submit_module = load_submit_module()
    squeue_text = "\n".join((
        squeue_line("101", "PENDING", "Resources", "2026-11-03T09:47:44", f"{STAGE_ROOT}/runs/run-a/job.sbatch"),
        squeue_line("102", "RUNNING", "None", "2026-10-29T17:00:00", "/w/my dir/runs/run b/job.sbatch"),
        squeue_line("103", "PENDING", "Priority", "N/A", f"{STAGE_ROOT}/code/sha/scripts/slurm/build_env.sbatch"),
        "garbage",
    ))
    tracked_jobs = submit_module.parse_tracked_jobs(squeue_text)
    assert [(job["job_id"], job["run_id"], job["kind"]) for job in tracked_jobs] == [
        ("101", "run-a", "train"), ("102", "run b", "train")]
    assert tracked_jobs[0]["reason"] == "Resources" and tracked_jobs[0]["start"] == "2026-11-03T09:47:44"


def test_milestone_is_the_furthest_key_reached():
    """The last RUN_MILESTONE_KEYS key present wins, regardless of dict order."""
    submit_module = load_submit_module()
    assert submit_module.run_milestone(None) == "no job_info.txt"
    assert submit_module.run_milestone({"hostname": "gpunode2"}) == "no milestone"
    assert submit_module.run_milestone({"train_elapsed_s": "10", "start_utc": "s"}) == "train_elapsed_s: 10"
    assert submit_module.run_milestone(SUCCESSFUL_JOB_INFO) == "end_utc: 2026-10-29T00:00:00Z"


def test_every_status_rule():
    """One run per rule: active, done, fetch, plot-recovery, resumable, stale-ledger, failed."""
    submit_module = load_submit_module()
    remote_runs = {"active": None, "done": SUCCESSFUL_JOB_INFO, "unfetched": SUCCESSFUL_JOB_INFO,
                   "plot": PLOT_FAILED_JOB_INFO, "resumable": RESUMABLE_JOB_INFO,
                   "killed": KILLED_IN_TRAINING_JOB_INFO}
    # The fetched copy of "unfetched" was taken mid-run, so it is not yet a successful copy.
    local_runs = {"done": SUCCESSFUL_JOB_INFO, "unfetched": KILLED_IN_TRAINING_JOB_INFO}
    tracked_jobs = [{"job_id": "1", "state": "PENDING", "reason": "Resources", "start": "N/A",
                     "run_id": "active", "kind": "train"}]
    ledger_rows = [{"run_id": "ledger-only"}, {"run_id": "done"}]
    plan = plan_by_run(submit_module.plan_run_tracking(ledger_rows, remote_runs, tracked_jobs, local_runs))
    assert {run_id: entry["status"] for run_id, entry in plan.items()} == {
        "active": "active", "done": "done", "unfetched": "fetch", "plot": "plot-recovery",
        "resumable": "resumable", "ledger-only": "stale-ledger", "killed": "failed"}
    assert plan["active"]["jobs"] == ["train 1 PENDING (Resources, est. start N/A)"]
    assert plan["unfetched"]["action"] == "submit_run.py fetch unfetched"


def test_an_active_job_wins_over_a_finished_record():
    """A render-only job on a successful run keeps it active, so it is not reported as done."""
    submit_module = load_submit_module()
    tracked_jobs = [{"job_id": "7", "state": "RUNNING", "reason": "None", "start": "t0",
                     "run_id": "ok", "kind": "render"}]
    plan = plan_by_run(submit_module.plan_run_tracking(
        [], {"ok": SUCCESSFUL_JOB_INFO}, tracked_jobs, {"ok": SUCCESSFUL_JOB_INFO}))
    assert plan["ok"]["status"] == "active"
    assert plan["ok"]["jobs"] == ["render 7 RUNNING (since t0)"]


def test_a_run_cleaned_from_the_cluster_but_fetched_is_done():
    """With no cluster folder, the fetched record decides (as plan_cleanup does)."""
    submit_module = load_submit_module()
    plan = plan_by_run(submit_module.plan_run_tracking([{"run_id": "ok"}], {}, [], {"ok": SUCCESSFUL_JOB_INFO}))
    assert plan["ok"]["status"] == "done" and not plan["ok"]["on_cluster"] and plan["ok"]["on_mac"]


def test_track_runs_hides_done_runs_unless_asked(tmp_path, monkeypatch, capsys):
    """Default output lists only runs needing attention; --all lists the done ones too. Read-only."""
    submit_module = load_submit_module()
    local_runs_root = tmp_path / "runs"
    (local_runs_root / "ok-run").mkdir(parents=True)
    (local_runs_root / "ok-run" / "job_info.txt").write_text("split_render_check: match\nend_utc: x\n")
    ledger_path = local_runs_root / "submitted.tsv"
    submit_module.write_ledger(ledger_path, [
        {"run_id": run_id, "job_id": job_id, "git_sha": "a", "session": "s", "profile": "p", "submitted_utc": "t"}
        for run_id, job_id in (("ok-run", "1"), ("queued-run", "2"))])
    ledger_before = ledger_path.read_text()
    remote_commands = []
    squeue_text = squeue_line("2", "PENDING", "Priority", "N/A", f"{STAGE_ROOT}/runs/queued-run/job.sbatch") + "\n"

    def fake_run_command(command, capture_output=True, input_text=None):
        """Record the command and answer the squeue call with one pending job."""
        remote_commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout=squeue_text, stderr="")
    monkeypatch.setattr(submit_module, "run_command", fake_run_command)
    monkeypatch.setattr(submit_module, "resolve_ssh_target", lambda host: host)
    monkeypatch.setattr(submit_module, "resolve_remote_stage_root", lambda config, target: STAGE_ROOT)
    monkeypatch.setattr(submit_module, "list_remote_runs", lambda target, root: (
        {"ok-run": {"split_render_check": "match", "end_utc": "x"}, "queued-run": None}, {"queued-run"}))
    # Default: the done run is hidden and counted.
    submit_module.track_runs({"ssh_host": "host"}, False, local_runs_root, ledger_path)
    default_output = capsys.readouterr().out
    assert "queued-run" in default_output and "ok-run" not in default_output
    assert "1 done and fetched hidden" in default_output
    # --all: both runs.
    submit_module.track_runs({"ssh_host": "host"}, True, local_runs_root, ledger_path)
    assert "ok-run" in capsys.readouterr().out
    # Read-only: only squeue ran remotely, and nothing local changed.
    assert all("squeue" in " ".join(command) for command in remote_commands)
    assert ledger_path.read_text() == ledger_before
