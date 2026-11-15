"""Measured walltime arithmetic and fail-closed Slurm release without real scheduler writes."""

import importlib.util
from pathlib import Path

import pytest

from viking_splatad.splatad.training_budget import measured_training_budgets


def test_measured_budgets_include_full_baseline_overhead_and_round_to_minutes():
    """Full baseline overhead is retained even for short smoke; no extrapolated fast setup is assumed."""
    budgets = measured_training_budgets(120., 0.5, 3., {"smoke": 100, "full": 200}, 2., 1000)
    assert budgets["smoke"]["predicted_seconds"] == 120 + 3 + 0.5 * 100
    assert budgets["smoke"]["seconds"] == 360
    assert budgets["full"]["seconds"] == 480
    assert budgets["full"]["time_limit"] == "0-00:08:00"
    with pytest.raises(ValueError, match="cluster maximum"):
        measured_training_budgets(120., 0.5, 3., {"full": 200}, 2., 400)


def test_release_validates_every_job_before_mutating_and_releases_last(tmp_path):
    """Only exact pending identities can receive budgets; both budgets precede smoke release."""
    source = Path(__file__).resolve().parents[2] / "scripts/slurm/release_radar_budget.py"
    specification = importlib.util.spec_from_file_location("release_control", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    jobs = [{"job_id": "101", "run_id": "smoke", "profile": "smoke", "release": True},
            {"job_id": "102", "run_id": "full", "profile": "full", "release": False}]
    for job in jobs:
        (tmp_path / "runs" / job["run_id"]).mkdir(parents=True)
    budgets = measured_training_budgets(120., 0.5, 3., {"smoke": 100, "full": 200}, 2., 1000)
    report = {"git_sha": "revision", "gpu_forward_validated": True, "budgets": budgets,
              "benchmark_job_id": "100", "baseline_wall_seconds": 120., "headroom_factor": 2.}
    calls = []

    def fake_control(arguments):
        """Return only synthetic job metadata and record mutations for ordering assertions."""
        calls.append(arguments)
        if arguments[:2] == ["show", "job"]:
            job = next(job for job in jobs if job["job_id"] == arguments[2])
            reason = "JobHeldUser" if job["release"] else "Dependency"
            return f"JobState=PENDING Reason={reason} Command={tmp_path}/runs/{job['run_id']}/job.sbatch"
        return ""

    records = module.release_budgeted_runs(report, "revision", tmp_path, jobs, fake_control)
    assert len(records) == 2 and calls[-1] == ["release", "101"]
    assert calls[-3][0] == "update" and calls[-2][0] == "update"
    assert (tmp_path / "runs/smoke/allocation.json").is_file()
    calls.clear()
    with pytest.raises(ValueError, match="benchmark"):
        module.release_budgeted_runs(dict(report, gpu_forward_validated=False), "revision", tmp_path, jobs, fake_control)
    assert calls == []
