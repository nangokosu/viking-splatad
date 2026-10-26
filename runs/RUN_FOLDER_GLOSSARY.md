# SplatAD run folder glossary

This document covers every run folder in `runs/`. All runs from `scripts/slurm/splatad_run.sbatch` share one layout. It is the only copy: it lives in the repo at `runs/RUN_FOLDER_GLOSSARY.md` and is versioned with the code, so update it whenever the job's outputs change. Differences between older and newer runs are listed under "Version notes" at the end.

Path on the Mac after `submit_run.py fetch <run_id>`: `runs/<run_id>/`. On the cluster: `~/viking-splatad/runs/<run_id>/`.
`<run_id>` = `<session>-<profile>-<UTC yyyymmdd-HHMMSS>-<short sha>`. The short sha is the code commit; `GIT_SHA` in `job_info.txt` gives it in full.

```
runs/<run_id>/
├── job_info.txt                       settings, environment, GPU, timings
├── slurm-<job id>.out                 Slurm log (stdout + stderr of the job)
├── compare/                           only if you ran scripts/slurm/compare_renders.py on the Mac
├── train/<session>/splatad/<run_id>/
│   ├── config.yml                     full nerfstudio config (loads the model; holds the time window)
│   ├── dataparser_transforms.json     w2m transform (dataparser frame) and time offset
│   ├── events.out.tfevents.*          tensorboard training curves
│   └── nerfstudio_models/step-*.ckpt  latest checkpoint only
└── renders/
    ├── train/                         poses the model was trained on
    │   ├── rgb/<idx>_<ts_ns>.jpg
    │   ├── gt-rgb/<idx>_<ts_ns>.jpg
    │   ├── lidar/<idx>_<ts_ns>.npz
    │   └── lidar_metrics.json
    └── test/                          held-out poses, same layout
```

## Frames and time
| name | meaning |
|---|---|
| **lidar frame** | the sweep's own sensor frame. The raw `.npy` sweeps use it too. Metres. |
| **map frame** | the "world" of `data/intermediate/<session>/poses.npz` (`<sensor>__poses_world_sensor`), z up. GT cuboids live here. This is the frame to compare across runs. |
| **dataparser frame** | SplatAD's internal world = map frame left-multiplied by `w2m` (inverse of the mean lidar pose of the training window). It changes with the time window, so never compare it across runs. Scale is 1. |
| **time base** | seconds, the same base as `poses.npz` `<sensor>__times`. SplatAD subtracts `time_offset` internally; the stored `absolute_time_s` has it added back. |
| **`<idx>_<ts_ns>`** | input frame index (6 digits) and its timestamp in ns. Same name as the input frame in `data/intermediate/<session>/camera/...` or `.../lidar/<sensor>/`. |

## Splits
- `train`: the poses used to fit the model. Its renders show how well the model fits.
- `test`: held-out poses. The dataparser holds out every ~10th frame (train fraction 0.9, linspace), interleaved with training frames. It therefore measures interpolation between nearby training views, not a held-out stretch of the drive.
- Cameras and lidar are split separately, so frame counts can differ between `rgb/` and `lidar/`.

## `job_info.txt`
`key: value` lines, written as the job progresses.

| key | meaning |
|---|---|
| `hostname`, `slurm_job_id`, `slurm_cpus_per_task`, `slurm_mem_per_node_mb` | where and with what resources it ran |
| `start_utc`, `end_utc` | job start, end of rendering |
| `copied_back_utc`, `scratch_run_dir` | runs before 2026-10-26 only: end of the copy back to home, and the scratch folder the outputs were written to first (deleted when that scratch dir expires). Newer runs write straight to the run dir on the workdisk. |
| `RUN_ID`, `SESSION`, `GIT_SHA`, `CODE_DIR` | run id, dataset, and the exact committed code |
| `ENV_PREFIX`, `environment build:` | native SplatAD environment (named by recipe hash) and its build record. Runs before 2026-10-26 also have `ENV_EXPIRES_EPOCH`, the expiry of the scratch dir the env lived on. |
| `MAX_NUM_ITERATIONS`, `STEPS_PER_SAVE` | training length and checkpoint interval |
| `TRAIN_SEED` | `ns-train --machine.seed`. It seeds python/numpy/torch. Runs are **not** bit-identical even with the same seed (float-atomic gradient sums on the GPU). |
| `RENDER_POSE_SOURCE`, `RENDER_DOWNSCALE_FACTOR` | rendered splits (`train+test`) and camera resolution factor (1 = native) |
| `DATASET_START_FRACTION`, `DATASET_END_FRACTION` | time window as fractions of the recording's span (from the profile's `window_*` seconds) |
| `gpu (...)`, `gpu_check`, `viking_splatad_from` | GPU model, compute capability, driver; proof that CUDA and the right code were loaded |
| `train_elapsed_s`, `render_elapsed_s`, `lidar_render_elapsed_s` | wall time of training, camera render and lidar render (also `*_start_epoch_s` and `*_end_epoch_s`) |

## Camera renders (`renders/<split>/rgb`, `gt-rgb`)
- There is one JPEG per camera pose (quality 100, still lossy) at native resolution. They come from `ns-render dataset`.
- `rgb/`: SplatAD's render at the input camera pose.
- `gt-rgb/`: the real image at the same pose, as the model saw it.
- Pair the two by file name. `scripts/slurm/compare_renders.py <run dir>` gives per-split PSNR and comparison sheets.

## Lidar renders (`renders/<split>/lidar/<idx>_<ts_ns>.npz`)
- There is one compressed npz per input lidar scan, made by `scripts/08_render_lidar_pointcloud.py`.
- SplatAD renders along the **rays of the real scan** (each real point's azimuth and elevation). So a rendered point exists only where the real lidar got a return; there are no new points in directions where the input had none.
- Load a scan with `np.load(path)`.

| key | dtype, shape | meaning |
|---|---|---|
| `pred_points` | float32 [M, 5] | rendered point cloud: x, y, z (lidar frame, m), intensity (0–1), t (s within the sweep). Keeps rays with ray-drop probability ≤ 0.5 and a valid real ray. xyz = expected depth × ray direction + velocity × t (rolling shutter). |
| `pred_points_median` | float32 [M, 5] | same rays, using the **median** depth along the ray (sharper surfaces) instead of the expected depth |
| `gt_points` | float32 [N, 5 or 6] | the real points that returned (x, y, z, intensity, t, channel), lidar frame, after the ego-vehicle filter. These are the input sweep's own rows: channel (laser ring id, 0 to beams−1) is there only if the sweep stores it, which the current sessions do, so N × 6. The rows are in input order, not row-aligned with `pred_points` (raster order) even when M = N, so pair them by nearest neighbour or through the raster, never by row index. |
| `raster_pts` | float32 [C, H, W, 5] | the real scan on SplatAD's ray grid: azimuth (deg), elevation (deg), depth (m), t (s), intensity. H = beams, W = azimuth bins, C > 1 only when a cell holds several points. depth 0 = no ray in that cell. Rays with depth above the return threshold are non-returns. |
| `pred_depth` | float32 [C, H, W, 1] | rendered expected depth (m) per ray of `raster_pts` |
| `pred_median_depth` | float32 [C, H, W, 1] | rendered median depth (m) per ray |
| `pred_intensity` | float32 [C, H, W, 1] | rendered intensity per ray (0–1) |
| `pred_ray_drop_prob` | float32 [C, H, W, 1] | predicted probability that the ray returns nothing |
| `lidar_to_map` | float64 [4, 4] | scan pose in the **map frame** (T50). `xyz_map = (lidar_to_map @ np.c_[xyz, ones].T).T[:, :3]`, using only the first 3 columns of the point arrays |
| `lidar_to_world` | float32 [3, 4] | scan pose in the dataparser frame. Don't use it for cross-run analysis. |
| `dataparser_transform` | float32 [3, 4] | `w2m`: map frame → dataparser frame |
| `absolute_time_s` | float64 scalar | scan time in the `poses.npz` time base |
| `sweep_index` | int64 scalar | row of this scan in `poses.npz` `<sensor>__times` / `__poses_world_sensor` |

## Lidar metrics (`renders/<split>/lidar_metrics.json`)
`{"mean": {metric: value}, "per_scan": {"<idx>_<ts_ns>": {metric: value}}}`. These are SplatAD's built-in metrics, computed per scan. `mean` averages each metric over the scans that report it.

"Valid returned rays" = rays where the real lidar got a return with depth > 0.

| metric | definition | unit / better |
|---|---|---|
| `depth_median_l2` | median over valid returned rays of (pred expected depth − real depth)² | m², lower |
| `depth_mean_rel_l2` | mean of ((pred − real) / real)², expected depth | unitless, lower |
| `median_depth_median_l2`, `median_depth_mean_rel_l2` | same two, using the rendered median depth | as above |
| `intensity_rmse` | √mean (pred intensity − real intensity)² over valid returned rays | intensity units (0–1), lower |
| `ray_drop_accuracy` | fraction of valid rays where (ray-drop prob > 0.5) equals (the real ray did not return). **Informative only when the run's `config.yml` has `add_missing_points: true`** (Phase B). That needs a dataset extracted after 2026-10-26: the converter keeps no-returns out of the returns files and writes them to `lidar_no_return/` (T51–T54), and the dataparser adds them as 2000 m rays (T55). Without that, every ray is a return. The score is then 1 whenever the model predicts no drop, and the ray-drop loss only ever sees "returned" labels, so ignore it (all runs up to job 222633). | 0–1, higher |
| `chamfer_distance` | (Σ over `pred_points` of the squared distance to the nearest real point + Σ over real points of the squared distance to the nearest `pred_points` point) / N_real. Note that SplatAD divides **both** sums by the real point count. Missing when either cloud is empty. | m², lower |
| `median_chamfer_distance` | same, using `pred_points_median` | m², lower |

## Version notes
- Runs before commit `806db4f` (e.g. `may_chunks0_1_smoke-smoke-20261025-182824-b8104e1`) have camera renders only. They have no `lidar/`, no `lidar_metrics.json`, no `TRAIN_SEED` (their seed was nerfstudio's default, 42).
