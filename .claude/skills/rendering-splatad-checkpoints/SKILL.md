---
name: rendering-splatad-checkpoints
description: Use when submitting render-only jobs for an existing successful SplatAD run's checkpoint - rendering more camera images or lidar sweeps from the run's guard frames without retraining, checking such a job, or fetching its renders.
---

# Rendering more frames from a finished run's checkpoint

## Overview
A run renders only its **test** split, the T62 cores (full May run: 22 camera frames, 24 lidar sweeps). The rest of each T60 test block is **guard**: never trained on, never rendered. `submit_run.py render` submits `scripts/slurm/render_only.sbatch`, which renders chosen guard frames from the run's own `config.yml` and latest checkpoint. There is no training. Outputs go to `<stage root>/runs/<run_id>/rerenders/<label>/`.

How it works:
1. `submit_run.py render` checks every index against the run's `split.json`, then writes `selection.json` and `job.sbatch` into the rerender folder.
2. `scripts/12_prepare_rerender.py write-config` copies the run's `config.yml` into the rerender folder with the dataparser's `render_frame_selection_file` set. The run's own `config.yml` is never edited.
3. `ns-render dataset` and `scripts/08_render_lidar_pointcloud.py` load that copy with `--pose-source test`. With the field set, the dataparser's `test`/`eval` split is exactly the selected guard frames (T66). `train` and `val` are unchanged, so the time origin and `w2m` are the run's.
4. `12_prepare_rerender.py check` compares the rendered file names with `selection.json`.

## What can be rendered
Only **guard** frames of that run. `split.json` (written by `scripts/09_extract_run_split.py`) is the source of truth, never `config.yml` or the manifest. List a run's guard frame indices per kind:
```bash
# Fetched copy:
python -c "import json; frames = json.load(open('runs/<run_id>/split.json'))['frames']; [print(kind, [f['frame_index'] for f in frames if f['kind'] == kind and f['split'] == 'guard']) for kind in ('camera', 'lidar')]"
# On the cluster (no fetch needed):
ssh comps0.cs.toronto.edu "python3 -c \"import json; frames = json.load(open('/w/20262/\$USER/viking-splatad/runs/<run_id>/split.json'))['frames']; [print(kind, [f['frame_index'] for f in frames if f['kind'] == kind and f['split'] == 'guard']) for kind in ('camera', 'lidar')]\""
```
Full May run: camera 217 guard frames, lidar 216. Each block is [guard | core | guard]: the core indices (e.g. camera 795–806) are `test`, not guard.

## Prerequisites
| need | check |
|---|---|
| the run succeeded | its `job_info.txt` has `split_render_check: match` and `end_utc` (`render` refuses otherwise) |
| the run folder still exists on the cluster | not removed by `clean` (its `config.yml`, checkpoint and `split.json` are read) |
| HEAD is committed and staged | `submit_run.py stage --session <SESSION>` after the last commit (`render` says so if `code/<sha>` is missing) |
| HEAD's environment is built | `building-splatad-cluster-env` (`render` says `run build-env first` otherwise) |

## Submit (local machine, repo root)
```bash
git status                                     # commit first
uv run python scripts/slurm/submit_run.py stage  --session <SESSION>   # if HEAD is not staged yet
uv run python scripts/slurm/submit_run.py render <run_id> --camera-frames <guard camera indices> --lidar-sweeps <guard lidar indices>   # syntax 795,797-800; indices from the listing above
```
- It prints `submitted render job <job> run <run_id> label <label>`. The label is `render-<UTC yyyymmdd-HHMMSS>-<short sha of the render code>`.
- It runs the same gates as `submit` (pytest, contract on the run's `SESSION`, clean tree). `--skip-gates` skips them.
- It is **not** logged in `runs/submitted.tsv`: that ledger lists training runs only.
- Resources: the full node from `sinfo`, like a run. Time limit: `render_only.time_limit` in `configs/slurm.yaml` (2 h).

## Constraints
| rule | why |
|---|---|
| `--camera-frames` and `--lidar-sweeps` are both required and non-empty | the job always renders both kinds |
| at least **2** camera frames | `ns-render dataset` names outputs relative to `os.path.commonpath` of the split's images. For one image that is the file itself, so `relative_to` raises (render.py ~998–1013, smoke job 222946) |
| every index is a guard frame of that kind | train frames were fitted, and core frames are already in `renders/test/`. The error lists each offender with its actual split (`train`, `test` or `absent`) |
| list syntax `795,797-800` | inclusive ranges, sorted and deduplicated; no negatives, no reversed ranges |

## Wait, then check
Start a background wait right after `render` (same loop as the running skill's "Fetch"):
```bash
until queue=$(ssh -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 comps0.cs.toronto.edu 'squeue -h -u $USER -o %i') && ! grep -qxE '<job1>|<job2>' <<<"$queue"; do sleep 300; done; echo jobs-left-queue
```
- A failed ssh fails the `queue=$(...)` assignment, so the loop keeps waiting. Never pipe ssh into `grep` instead: a failure then looks like an empty queue.
- `squeue -u`, not `-j`: `-j` errors once Slurm purges the job, and the loop would never end.

**Success:** `render_check: match` and `end_utc` in `<run dir>/rerenders/<label>/job_info.txt`. Milestones: `gpu_check` → `render_config` → `render_elapsed_s` → `lidar_render_elapsed_s` → `render_check: match` → `end_utc`. `render_check: FAILED (exit 1)` means the renders differ from the selection; `exit 2` means a render folder is missing. The reason is in `slurm-<job>.out`.

## Fetch
```bash
uv run python scripts/slurm/submit_run.py fetch <run_id>
```
It rsyncs the whole run folder, `rerenders/` included (checkpoints excluded, as always). On a successful render job, fetch without asking. On a failed one, read the log over ssh and report instead.

## Outputs (`runs/<run_id>/rerenders/<label>/`)
| file | content |
|---|---|
| `selection.json` | `camera`, `lidar` (requested indices), `files` (their dataset paths from `split.json`), `source_split_membership_sha256` (the `split.json` it was checked against), `requested_utc` |
| `job.sbatch`, `job_info.txt`, `slurm-<job>.out` | the job's variables, settings and timings, and its log |
| `config.yml` | the run's `config.yml` plus `render_frame_selection_file` |
| `dataparser_transforms.json` | written by script 08; identical to the run's |
| `renders/test/{rgb,gt-rgb}/<idx>_<ts_ns>.jpg` | camera renders, same format as the run's |
| `renders/test/lidar/<idx>_<ts_ns>.npz`, `renders/test/lidar_metrics.json` | lidar renders and metrics, same keys as the run's (`runs/RUN_FOLDER_GLOSSARY.md`) |

The folder is named `test` because ns-render and script 08 name folders by split; these are guard frames.

## Code provenance
`job_info.txt` records both SHAs:
- `GIT_SHA` is the render code: HEAD at submission, in `code/<sha>`.
- `TRAIN_GIT_SHA` is the code that trained the checkpoint (the run's `GIT_SHA`).

They usually differ, because the selection field is newer than older runs. Note both when you compare renders across jobs.

## Cleanup
- `clean` keeps a run while any of its render jobs is queued or running: their `rerenders/<label>/job.sbatch` marks the run active. `scancel` the job first.
- Deleting a run with `clean` deletes its `rerenders/` too. Nothing cleans a single rerender folder, and a failed render job never makes its run look failed: `clean` reads only the run's own `job_info.txt`.
