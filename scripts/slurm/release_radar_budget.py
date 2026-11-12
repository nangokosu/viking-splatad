"""Slurm CPU control job: apply measured budgets, then release an already submitted held smoke."""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path


def control(arguments):
    """Run one owner-authorized Slurm control command; return stdout or fail without releasing."""
    return subprocess.run(["scontrol", *arguments], check=True, capture_output=True, text=True).stdout


def release_budgeted_runs(report: dict, expected_revision: str, stage_root: Path, jobs: list,
                           control_command=control):
    """Validate all pending run identities, apply recorded budgets, then release smoke.

    Args:
        report: Atomic successful GPU benchmark report; expected_revision: committed source.
        stage_root: This user's runtime workdisk root; jobs: smoke/full job/run/profile dicts.
        control_command: Slurm command runner, replaceable by a read-only synthetic test double.

    Returns:
        Applied allocation records. All identities/budgets are validated before mutation.
        Failed GPU/source checks, wrong jobs or non-pending runs never get released.
    """
    if report.get("git_sha") != expected_revision or report.get("gpu_forward_validated") is not True:
        raise ValueError("matching successful GPU benchmark required before releasing training")
    if len(jobs) != 2 or sum(job.get("release", False) for job in jobs) != 1:
        raise ValueError("exactly one held smoke and its dependent full job are required")
    plans = []
    for job in jobs:
        if not re.fullmatch(r"[0-9]+", job["job_id"]) or not re.fullmatch(r"[A-Za-z0-9_-]+", job["run_id"]):
            raise ValueError("invalid job/run identifier")
        budget = report["budgets"][job["profile"]]
        # Same five-day cap as the project/cluster policy, expressed as SI seconds.
        if not 0 < budget["seconds"] <= 5 * 24 * 60 * 60:
            raise ValueError("measured budget exceeds cluster policy")
        time_match = re.fullmatch(r"([0-9]+)-([0-9]{2}):([0-9]{2}):00", budget["time_limit"])
        if time_match is None:
            raise ValueError("invalid measured time limit")
        days, hours, minutes = map(int, time_match.groups())
        if hours >= 24 or minutes >= 60 or ((days * 24 + hours) * 60 + minutes) * 60 != budget["seconds"]:
            raise ValueError("measured time string disagrees with budget seconds")
        state = control_command(["show", "job", job["job_id"], "-o"])
        fields = dict(re.findall(r"(?:^|\s)(\w+)=([^\s]+)", state))
        run_dir = Path(stage_root) / "runs" / job["run_id"]
        if (fields.get("JobState") != "PENDING" or fields.get("Command") != str(run_dir / "job.sbatch")
                or (job.get("release") and fields.get("Reason") != "JobHeldUser")):
            raise ValueError("budget release requires the exact pending owner-held smoke/dependent run")
        record = {"source_revision": expected_revision, "benchmark_job_id": report["benchmark_job_id"],
                  "profile": job["profile"], "job_id": job["job_id"], "time_limit": budget["time_limit"],
                  "seconds": budget["seconds"], "predicted_seconds": budget["predicted_seconds"],
                  "baseline_wall_seconds": report["baseline_wall_seconds"], "headroom_factor": report["headroom_factor"],
                  "policy": "measured full B walltime plus measured radar gate cost, with configured headroom"}
        plans.append((job, run_dir, record))
    # Update the dependent full job before smoke can become schedulable.
    for job, run_dir, record in sorted(plans, key=lambda plan: plan[0].get("release", False)):
        control_command(["update", f"JobId={job['job_id']}", f"TimeLimit={record['time_limit']}"])
        temporary = run_dir / "allocation.json.partial"
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(run_dir / "allocation.json")
    smoke = next(job for job in jobs if job.get("release"))
    control_command(["release", smoke["job_id"]])
    return [record for _, _, record in plans]


def main(argument_list=None):
    """Apply a GPU-success-dependent control specification under a small Slurm CPU allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, required=True)
    arguments = parser.parse_args(argument_list)
    if not os.environ.get("SLURM_JOB_ID"):
        raise ValueError("budget release must run inside its allocated Slurm CPU job")
    specification = json.loads(arguments.specification.read_text())
    report = json.loads(Path(specification["benchmark_report"]).read_text())
    records = release_budgeted_runs(report, specification["git_sha"], Path(specification["stage_root"]), specification["jobs"])
    arguments.specification.with_suffix(".released.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps({"released": records}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
