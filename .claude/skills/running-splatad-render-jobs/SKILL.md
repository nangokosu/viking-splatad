---
name: running-splatad-render-jobs
description: Use when training SplatAD and rendering camera images or lidar point clouds on the CS Lab Slurm cluster - submitting a smoke or full run, checking whether a running job is stuck or failed, or fetching its camera/lidar renders back to the Mac for evaluation.
---

# Running SplatAD train+render jobs on Slurm

## Overview
`scripts/slurm/submit_run.py` drives everything from the Mac. `splatad_run.sbatch` does four things, then copies results to home:
1. **Trains** with `ns-train splatad ... viking-data`, passing `--machine.seed` = the profile's `seed`.
2. **Renders camera images** with `ns-render dataset`: rgb + gt-rgb jpg.
3. **Renders lidar** with `scripts/08_render_lidar_pointcloud.py`, which calls SplatAD's `get_lidar_outputs` and then `filter_lidar_pred_and_gt(output_point_cloud=True)`.
4. Both renders cover **every input pose** of the profile's `render_pose_source` (train+test).

Jobs run the **committed** code at HEAD. They need a built env: if `submit` says it is missing or expiring, use REQUIRED SUB-SKILL building-splatad-cluster-env.

## Submit (Mac, repo root)
```bash
git status                                   # commit first; untracked .DS_Store/reports are allowed
export VIKING_SLURM_USER=<cluster user>      # never write it into repo files
uv run python scripts/slurm/submit_run.py stage  --session <session>     # code/<sha> + data to home
uv run python scripts/slurm/submit_run.py submit --session <session> --profile smoke
```
- `submit` re-runs the gates (pytest, contract, clean tree) and prints `submitted job <job> run <run_id>`. The run is also logged in `runs/submitted.tsv`.
- If sbatch says `Unable to open file .../code/<sha>/...`, you forgot `stage` after committing.
- A `src/` change follows the CLAUDE.md test workflow (new tests, transform-test-runner) before you commit.
- Every profile in `configs/slurm.yaml` needs an integer `seed`, or `submit` refuses.

Re-run the CPU preflight whenever the dataparser, extraction or window changes:
- Script: `scripts/slurm/preflight_dataparser.py`, run in the image with `src/` bind-mounted.
- Window fractions come from a previous run's `job_info.txt`.
- The lidar overlays must land on edges.

**Timing (smoke, 52–67 s, 3000 iters, A6000):**
- train ~9 min
- camera render of 150 frames ~72 s
- copy-back ~2 min (a 1.65 GB checkpoint at 20 MB/s)
- lidar render: not measured yet. Read `lidar_render_elapsed_s` from the first run.

## Seed and reproducibility
`seed` seeds Python, NumPy and torch (all CUDA devices) before setup (nerfstudio `train.py`). It covers Gaussian init, sensor sampling order, random background, and MCMC relocation and noise.

It does **not** make runs bit-identical:
- the gsplat/SplatAD backward kernels sum gradients with `gpuAtomicAdd`;
- so does the tcnn lidar decoder;
- nerfstudio sets `cudnn.benchmark=True`.

Same seed means statistically similar runs, not identical ones. Compare B vs R with the same seed, and don't read small metric gaps as signal without repeats.

## Monitor: stuck vs progressing
- The log and `job_info.txt` are written straight to home: `~/viking-splatad/runs/<run_id>/`.
- Milestones, in order: `gpu_check` → `train_elapsed_s` → `render_elapsed_s` (camera) → `lidar_render_elapsed_s` → `copied_back_utc`.
- `job_info.txt` also gives `scratch_run_dir:` for the check below.

**The progress table often stops appearing in the log while training is healthy.** It was seen right after `Caching eval lidars`, the step-500 eval, with 5+ silent minutes. A silent log is NOT evidence of a hang, so check the job itself before any `scancel`:
```bash
ssh <user>@comps0.cs.toronto.edu "srun --jobid=<job> --overlap -n1 -c1 bash -c \
 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; \
  ls -la --time-style=+%T <scratch run dir>/train/*/splatad/*/ <...>/nerfstudio_models/'"
```
- Use `-n1`, or it runs once per task.
- Progressing: the GPU is busy and the `events.out.tfevents` mtime advances or a new `step-*.ckpt` appears.
- During the lidar step (after `render_elapsed_s`), replace the `ls` target with `<scratch run dir>/renders/*/lidar/`: new `.npz` files should keep appearing.
- Truly stuck: the GPU is idle and nothing advances for >10 min.
- Held with `user_env_retrieval_failed_requeued_held`, or a silent ~15 s death: see the building-splatad-cluster-env table.

## Fetch
Wait for `copied_back_utc` in `job_info.txt` (the job's copy-back to home), then:
```bash
uv run python scripts/slurm/submit_run.py fetch <run_id>
```
Result on the Mac in `runs/<run_id>/`. Every file and key of every run folder is described once, in `runs/RUN_FOLDER_GLOSSARY.md` (committed; the rest of `runs/` is gitignored). Update it whenever the job's outputs change; a test checks that it documents every lidar npz key.
- `renders/{train,test}/{rgb,gt-rgb}/*.jpg`: one per camera pose; file names are frame index + timestamp.
- `renders/{train,test}/lidar/<idx>_<ts_ns>.npz`: one per lidar pose, named exactly like the input sweep `data/intermediate/<session>/lidar/<sensor>/<idx>_<ts_ns>.npy`.
- `renders/{train,test}/lidar_metrics.json`: SplatAD's built-in per-scan lidar metrics, with `mean` and `per_scan` (depth_median_l2, intensity_rmse, ray_drop_accuracy, chamfer_distance, ...).
- `train/.../config.yml`, the latest `step-*.ckpt`, `job_info.txt`, and the Slurm log.

A complete smoke run has 135 train + 15 test frames per camera output. Lidar scan counts come from the dataparser's own split, so check them against the `.npz` count.

**Lidar npz keys** (full list in the docstring of `scripts/08_render_lidar_pointcloud.py`):

| key | shape | frame / meaning |
|---|---|---|
| `pred_points`, `pred_points_median` | [M,5] x,y,z,intensity,t | lidar frame; rays with ray-drop prob ≤ 0.5 |
| `gt_points` | [N,5] | the input returns, lidar frame |
| `raster_pts` + `pred_depth`, `pred_median_depth`, `pred_intensity`, `pred_ray_drop_prob` | [C,H,W,·] | per-ray, to recompute metrics |
| `lidar_to_map` | 4x4 | **map frame** (T50): `p_map = lidar_to_map @ [p;1]` |
| `lidar_to_world`, `dataparser_transform` | 3x4 | SplatAD dataparser frame, and w2m |
| `absolute_time_s`, `sweep_index` | scalar | poses.npz time base and row |

To compare with GT cuboids or raw sweeps, take points to the map frame with the stored `lidar_to_map` (the T50 result of `rendered_lidar.lidar_pose_in_map_frame`). Use only the xyz columns: `xyz_map = (lidar_to_map @ np.c_[pts[:, :3], np.ones(len(pts))].T).T[:, :3]`. Never apply `lidar_to_world` alone: it is in the dataparser frame, which depends on the time window.

Read the clouds with these limits in mind:
- **Rays come only from the input returns.** SplatAD renders along the input scan's own rays (`raster_pts`), so `pred_points` has no point where the input lidar had no return. A rendered trunk behind grass shows up only as a different depth on an existing ray.
- `pred_points` uses expected depth and `pred_points_median` uses median depth. SplatAD reports Chamfer for both.
- `<sensor>` is the manifest's only lidar (`ouster` for the current sessions).
- `test` is every ~10th frame interleaved with training frames (open issue in the session summary), so it measures interpolation, not a held-out stretch.

## Known traps
- Never pass `ns-render --downscale-factor` with viking-data: it asserts (render.py:929). The sbatch omits it when the factor is 1.
- Never use `ns-render --render-point-clouds` for SplatAD. It calls `get_outputs_for_lidar`, which fails with "Unknown sensor type", and it writes only PNG plots. Lidar goes through `08_render_lidar_pointcloud.py`.
- In that script, train scans are rasterised with `is_eval=True`. Without it, `_lidar_to_raster_pts` shuffles the points and keeps one per raster cell, and some returns get no ray.
- If the lidar step fails, the camera renders and the checkpoint are still copied back (EXIT trap).
- `sacct` may fail from comps0 with `Connection refused`. Use `squeue` and `job_info.txt` instead.
- Results reach home only through the job's copy-back. Anything left on scratch is lost when the scratch dir expires.
