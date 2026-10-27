---
name: cleaning-up-splatad-runs
description: Use when SplatAD Slurm runs were cancelled (by the user, by Slurm at a time or memory limit, or by anyone else), failed, or never finished, or when the user asks to delete specific runs - leftover run folders on the cluster workdisk, fetched copies in runs/, or stale rows in runs/submitted.tsv.
---

# Cleaning up SplatAD runs

## Overview
A run leaves artifacts in three places: `<stage root>/runs/<run_id>/` on the cluster workdisk, its fetched copy `runs/<run_id>/` on your machine, and its row in `runs/submitted.tsv`. `submit_run.py clean` removes all three, for every run that did not succeed, plus any runs you name. Never delete them by hand with `rm -rf`: the command decides from each run's own record and never touches a job that is still queued or running.

## What gets deleted
| run | action |
|---|---|
| job queued or running (its `job.sbatch`, or a render-only job's `rerenders/<label>/job.sbatch`, is in `squeue`) | **kept**, even when named: `scancel <job>` first |
| named on the command line | deleted, even if it succeeded |
| no `end_utc` + `split_render_check: match` in `job_info.txt` (cancelled, killed at a limit, failed, never started), or a ledger row with no folder left | deleted |
| successful | kept |

The cluster copy of `job_info.txt` decides; the fetched copy is used only when the cluster folder is gone. Staged code (`code/<sha>/`), data and environments are never touched.

## Steps (your machine, repo root)
1. **Dry run** (the default; deletes nothing). It prints `DELETE` or `keep` for every run, where it lives (`cluster`, `mac`, `ledger`) and why:
   ```bash
   uv run python scripts/slurm/submit_run.py clean                 # cancelled/failed runs only
   uv run python scripts/slurm/submit_run.py clean <run_id> ...    # plus these runs
   ```
2. **Show the user the `DELETE` lines.** If they are exactly the runs the user asked to remove (cancelled/failed ones, or the runs they named), go to step 3. If the list has a run the user did not expect, ask before deleting. Deletion cannot be undone, and a fetched run may be the only copy of its renders.
3. **Delete:** the same command plus `--yes`. Then report what was deleted and what was kept.

Only pass a successful run's id when the user named that run. An "ERROR: ... scancel it first" means the job is still queued or running: ask the user whether to `scancel` it. Never cancel a job on your own.

## After deleting
- `runs/RUN_FOLDER_GLOSSARY.md` and the ledger header stay.
- Nothing needs restaging: code and data folders are untouched.
