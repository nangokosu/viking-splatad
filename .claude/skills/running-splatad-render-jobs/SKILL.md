---
name: running-splatad-render-jobs
description: Use when training SplatAD and rendering camera images on the CS Lab Slurm cluster - submitting a smoke or full run, checking whether a running job is stuck, fetching renders, or judging whether rendered vs real camera images indicate a data/calibration bug or just undertraining.
---

# Running SplatAD train+render jobs on Slurm

## Overview
`scripts/slurm/submit_run.py` drives everything from the Mac; `splatad_run.sbatch` trains (`ns-train splatad ... viking-data`), renders (`ns-render dataset`, rgb + gt-rgb, train+test) and copies results to home. Jobs run the **committed** code at HEAD. Requires a built env (REQUIRED SUB-SKILL: building-splatad-cluster-env if `submit` says it is missing or expiring).

## Submit (Mac, repo root)
```bash
git status                                   # commit first; untracked .DS_Store/reports are allowed
export VIKING_SLURM_USER=<cluster user>      # never write it into repo files
uv run python scripts/slurm/submit_run.py stage  --session <session>     # code/<sha> + data to home
uv run python scripts/slurm/submit_run.py submit --session <session> --profile smoke
```
`submit` re-runs the gates (pytest, contract, clean tree) and prints `submitted job <job> run <run_id>` (also logged in `runs/submitted.tsv`). If sbatch says `Unable to open file .../code/<sha>/...`, you forgot `stage` after committing. A `src/` change follows the CLAUDE.md test workflow (new tests, transform-test-runner) before commit.

Re-run the CPU preflight (`scripts/slurm/preflight_dataparser.py` in the image, `src/` bind-mounted, window fractions from a previous run's `job_info.txt`) whenever the dataparser, extraction or window changes; lidar overlays must land on edges.

**Timing (smoke, 52–67 s, 3000 iters, A6000):** train ~9 min, render 150 frames ~72 s, copy-back ~2 min (1.65 GB checkpoint at 20 MB/s).

## Monitor: stuck vs progressing
Log and `job_info.txt` are written straight to home: `~/viking-splatad/runs/<run_id>/`. Milestones: `gpu_check` → `train_elapsed_s` → `render_elapsed_s` → `copied_back_utc`. `job_info.txt` also gives `scratch_run_dir:` for the check below.

**The progress table often stops appearing in the log while training is healthy** (seen right after `Caching eval lidars`, the step-500 eval, with 5+ silent minutes). A silent log is NOT evidence of a hang. Check the job itself before any `scancel`:
```bash
ssh <user>@comps0.cs.toronto.edu "srun --jobid=<job> --overlap -n1 -c1 bash -c \
 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; \
  ls -la --time-style=+%T <scratch run dir>/train/*/splatad/*/ <...>/nerfstudio_models/'"
```
(`-n1` or it runs once per task.) Progressing = GPU busy and the `events.out.tfevents` mtime advancing / new `step-*.ckpt`. Truly stuck = GPU idle and nothing advancing for >10 min. Held `user_env_retrieval_failed_requeued_held` or a silent ~15 s death → see building-splatad-cluster-env table.

## Fetch and compare
```bash
uv run python scripts/slurm/submit_run.py fetch <run_id>
uv run python scripts/slurm/compare_renders.py runs/<run_id> \
    --raw-camera-dir data/intermediate/<session>/camera/ids_camera
```
Prints per-split PSNR and writes `runs/<run_id>/compare/{test,train}_render_vs_gt.jpg` (start/mid/end; render left, real right). Look at the sheets — PSNR alone does not show alignment.

## Judging renders

| Observation | Meaning |
|---|---|
| gt-rgb vs raw frame ≪ 50 dB (baseline 53.7) | Dataparser feeds a different image → input bug |
| Mirrored/flipped/wrong direction, wrong scale, objects offset from real frame, doubled trunks/cars | Data bug: pose convention, intrinsics, extrinsic, timing, world frame, rosbag topic (e.g. deskewed clouds) |
| Held-out geometry wrong while train is fine | Poses inconsistent → data bug |
| Right geometry, speckle/dots on surfaces, blur, floaters | Undertraining (3k of stock 30k iters); fixes with more iterations |

**Reference:** smoke run 222568 (`b8104e1`): PSNR train 20.05 / test 19.88 dB, geometry aligned, speckled — a PASS.
**Train ≈ test is expected here:** test = every 10th frame, interleaved ~0.1–0.9 s from train frames, and short training memorises little. A train>test gap appears with longer training. This split tests interpolation only, not novel views.

## Known traps
- Never pass `ns-render --downscale-factor` with viking-data (asserts; render.py:929). The sbatch omits it when the factor is 1.
- `sacct` may fail from comps0 (`Connection refused`); use `squeue` and `job_info.txt`.
- Results only reach home via the job's copy-back; anything left on scratch dies with the scratch dir.
