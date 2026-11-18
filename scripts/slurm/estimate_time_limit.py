"""Derive a Slurm time limit for a profile from the measured wall times of fetched runs.

A run's wall time splits into training (``train_elapsed_s`` in ``job_info.txt``) and everything
else (loading, render, export: job end - job start - training). Training time scales with the
iteration count, so for a target profile with N_target iterations each measured run predicts

    predicted_s = train_elapsed_s * (N_target / N_measured) + (job_elapsed_s - train_elapsed_s)

Scaling the whole training time is an estimate, not a bound: the eval count is fixed (minimum_eval_points),
so eval time does not grow with N. The limit is the slowest prediction times
``radar_runtime_budget.headroom_factor`` (configs/slurm.yaml), rounded up to the whole hour.

Usage:
    uv run python scripts/slurm/estimate_time_limit.py --target-profile may_scene_a_30k \
        --base-profile may_scene_a --base-profile may_scene_a_radar_init ...
"""

import argparse
import math
from datetime import datetime
from pathlib import Path

import yaml

# Repository root: scripts/slurm/<this file> -> two levels up.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SLURM_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "slurm.yaml"
RUNS_DIRECTORY = REPOSITORY_ROOT / "runs"
SECONDS_PER_HOUR = 3600
# Slurm refuses limits over 5 days (CLAUDE.md, "GPU training on Slurm" step 5).
HOURS_PER_DAY = 24
MAXIMUM_LIMIT_HOURS = 5 * HOURS_PER_DAY


def parse_job_info(job_info_path):
    """Read the ``key: value`` lines of a run's job_info.txt.

    Args:
        job_info_path: Path to job_info.txt.

    Returns:
        Dict of key -> string value (the first occurrence of each key wins).
    """
    fields = {}
    for line in job_info_path.read_text().splitlines():
        # Only top-level "key: value" lines; indented lines belong to multi-line blocks.
        if line.startswith(" ") or ": " not in line:
            continue
        key, value = line.split(": ", 1)
        fields.setdefault(key, value.strip())
    return fields


def utc_seconds(timestamp_text):
    """Convert job_info's ``YYYY-MM-DDTHH:MM:SSZ`` stamp to epoch seconds.

    Args:
        timestamp_text: UTC stamp as written by splatad_run.sbatch (``date -u +%Y-%m-%dT%H:%M:%SZ``).

    Returns:
        Seconds since the Unix epoch, as a float (``%z`` reads the trailing ``Z`` as UTC).
    """
    return datetime.strptime(timestamp_text, "%Y-%m-%dT%H:%M:%S%z").timestamp()


def predicted_wall_seconds(fields, target_iterations):
    """Predict one measured run's wall time at the target iteration count.

    Args:
        fields: Parsed job_info.txt of a finished run.
        target_iterations: max_num_iterations of the target profile.

    Returns:
        Predicted wall seconds (formula in the module docstring).
    """
    train_seconds = float(fields["train_elapsed_s"])
    job_seconds = utc_seconds(fields["end_utc"]) - utc_seconds(fields["start_utc"])
    iteration_ratio = target_iterations / int(fields["MAX_NUM_ITERATIONS"])
    return train_seconds * iteration_ratio + (job_seconds - train_seconds)


def main():
    """Print each measured run's prediction and the derived limit for the target profile.

    Reads --target-profile, --base-profile (repeatable) and --git-sha from the command line.

    Raises:
        SystemExit: If no finished base-profile run is found, or the limit exceeds Slurm's 5 days.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-profile", required=True)
    parser.add_argument("--base-profile", action="append", required=True,
                        help="profile whose finished runs in runs/ are measured (repeatable)")
    parser.add_argument("--git-sha", default=None, help="only runs whose id ends with this short SHA")
    arguments = parser.parse_args()

    # Step 1: target iterations and headroom from the committed Slurm config.
    slurm_config = yaml.safe_load(SLURM_CONFIG_PATH.read_text())
    target_iterations = int(slurm_config["profiles"][arguments.target_profile]["max_num_iterations"])
    headroom_factor = float(slurm_config["radar_runtime_budget"]["headroom_factor"])

    # Step 2: predict every finished run of the base profiles (a run is finished when end_utc and
    # train_elapsed_s are both recorded).
    predictions = []
    for run_directory in sorted(RUNS_DIRECTORY.iterdir()):
        job_info_path = run_directory / "job_info.txt"
        if not job_info_path.is_file():
            continue
        if arguments.git_sha and not run_directory.name.endswith(arguments.git_sha):
            continue
        # Run ids are <session>-<profile>-<date>-<time>-<sha>; profile names contain underscores only.
        run_profile = run_directory.name.split("-")[1]
        if run_profile not in arguments.base_profile:
            continue
        fields = parse_job_info(job_info_path)
        if "end_utc" not in fields or "train_elapsed_s" not in fields:
            continue
        # A resumed run keeps the first job's start_utc but the resume job's end_utc, so its wall time
        # includes the queue wait between the jobs: leave it out.
        if "resume_start_utc" in fields:
            continue
        predicted_seconds = predicted_wall_seconds(fields, target_iterations)
        predictions.append(predicted_seconds)
        print(f"{run_directory.name}: train {float(fields['train_elapsed_s']) / SECONDS_PER_HOUR:.2f} h "
              f"at {fields['MAX_NUM_ITERATIONS']} it -> predicted {predicted_seconds / SECONDS_PER_HOUR:.2f} h "
              f"at {target_iterations} it")

    if not predictions:
        raise SystemExit("no finished runs of the base profiles found in runs/")

    # Step 3: slowest prediction x headroom, rounded up to the hour, capped by Slurm's 5 days.
    slowest_hours = max(predictions) / SECONDS_PER_HOUR
    limit_hours = math.ceil(slowest_hours * headroom_factor)
    if limit_hours > MAXIMUM_LIMIT_HOURS:
        raise SystemExit(f"derived limit {limit_hours} h exceeds Slurm's {MAXIMUM_LIMIT_HOURS} h")
    days, hours = divmod(limit_hours, HOURS_PER_DAY)
    print(f"runs measured: {len(predictions)}; slowest predicted {slowest_hours:.2f} h; "
          f"x headroom {headroom_factor} -> time_limit \"{days}-{hours:02d}:00:00\"")


if __name__ == "__main__":
    main()
