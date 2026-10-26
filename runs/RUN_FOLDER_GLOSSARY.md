# SplatAD run folder glossary

This document covers every run folder in `runs/`. All runs from `scripts/slurm/splatad_run.sbatch` share one layout. It is the only copy: it lives in the repo at `runs/RUN_FOLDER_GLOSSARY.md` and is versioned with the code, so update it whenever the job's outputs change. Differences between older and newer runs are listed under "Version notes" at the end.

Path on the Mac after `submit_run.py fetch <run_id>`: `runs/<run_id>/`. On the cluster: `<stage root>/runs/<run_id>/` on the workdisk (currently `/w/20262/<user>/viking-splatad/runs/<run_id>/`).
`<run_id>` = `<session>-<profile>-<UTC yyyymmdd-HHMMSS>-<short sha>`. The short sha is the code commit; `GIT_SHA` in `job_info.txt` gives it in full.

```
runs/<run_id>/
├── job_info.txt                       settings, environment, GPU, timings
├── split.json                         exact train/test frames + provenance; successful runs only
├── split_frames.csv                   the same frames, one row each; successful runs only
├── slurm-<job id>.out                 Slurm log (stdout + stderr of the job)
├── compare/                           only if you ran scripts/slurm/compare_renders.py on the Mac
├── train/<session>/splatad/<run_id>/
│   ├── config.yml                     full nerfstudio config (loads the model; holds the time window)
│   ├── dataparser_transforms.json     w2m transform (dataparser frame), scale and time offset
│   ├── events.out.tfevents.*          tensorboard training curves
│   └── nerfstudio_models/step-*.ckpt  latest checkpoint only
└── renders/
    ├── gaussians.npz                  every Gaussian of the latest checkpoint, map frame (P2)
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
- **The split is deterministic, with no RNG.** First the time window: every camera and lidar frame with time in [t_min + D·`DATASET_START_FRACTION`, t_min + D·`DATASET_END_FRACTION`] (inclusive; t_min and D over all camera and lidar times). Then, per sensor, ceil(n·f) evenly spaced train frames (linspace); the rest are test. At f = 1.0 every frame is in both. `TRAIN_SEED` does not affect it.
- f is the dataset manifest's `session.train_fraction` (0.9), unless `train_split_fraction_override` is set. The job passes it to `ns-train` as `--train-split-fraction` (`TRAIN_SPLIT_FRACTION`), so `config.yml` records the real value. **Runs without `TRAIN_SPLIT_FRACTION` in `job_info.txt` have `train_split_fraction: 0.5` in `config.yml`, which is wrong**: nerfstudio saved the class default before the dataparser set the real value. `manifest.json` `pipeline.splits` covers the whole recording, not the run's window, so it isn't the run's split either.

### `split.json` and `split_frames.csv`
Written by `scripts/09_extract_run_split.py` (T58 window, T59 split), which rebuilds the split on the CPU from `job_info.txt`, `config.yml` and the dataset. **Only successful runs are documented.** The job runs it once, as its last step, after every render. The script writes nothing, and deletes any earlier record, unless all of these hold:
- every render step finished (`lidar_render_elapsed_s` in `job_info.txt`) and `config.yml` exists;
- every split of `RENDER_POSE_SOURCE` has camera and lidar renders;
- `TRAIN_SPLIT_FRACTION`, `config.yml` and the manifest agree;
- the rendered file names equal the rebuilt split.

Exit codes: 0 written, 1 inconsistent (renders or fractions disagree), 2 run incomplete. A failed or killed run therefore has no `split.json`. To (re)write it after `fetch`: `uv run python scripts/09_extract_run_split.py --run-dir runs/<run_id>`; the dataset defaults to `data/intermediate/<SESSION>`. The output is byte-identical for the same inputs.

| key (`split.json`) | meaning |
|---|---|
| `run` | `run_id`, `session`, `git_sha`, `train_seed`, `add_missing_points` from `job_info.txt` |
| `selection` | `dataset_start_fraction`, `dataset_end_fraction`, `window_bounds_s` (`poses.npz` seconds; `null` = whole recording), `train_fraction` and `train_fraction_source` (`manifest.session.train_fraction` or `override`), `max_eval_frames` (always `null`) |
| `dataset` | dataset folder name, and SHA-256 of its `manifest.json` and `poses.npz` (the same split needs the same dataset) |
| `counts` | per sensor: `kind`, `frames_in_recording`, `frames_in_window`, `train`, `test` |
| `membership_sha256` | fingerprint of which frame is in which split. Two runs or two people with equal values used the same frames. |
| `render_check` | per `<kind>/<split>` rendered: `match` (always true in a written record), `missing_from_renders`, `unexpected_in_renders` |
| `frames` | one entry per frame, same columns as the CSV |
| `seed_note`, `schema` | why the seed is recorded, and the layout version (`viking-splatad/run-split/1`) |

`split_frames.csv` columns: `sensor`, `kind`, `split`, `frame_index`, `timestamp_ns`, `time_s` (`poses.npz` seconds), `file` (path inside the dataset).

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
| `ADD_MISSING_POINTS` | `True`/`False`, the session config's `splatad.add_missing_points` (session named by the dataset's manifest), passed to `ns-train` as `--add-missing-points`. `True` adds the recorded no-returns (`lidar_no_return/`) as 2000 m rays. Absent in runs before 2026-10-26, which all ran with `False`. |
| `gpu (...)`, `gpu_check`, `viking_splatad_from` | GPU model, compute capability, driver; proof that CUDA and the right code were loaded |
| `train_elapsed_s`, `render_elapsed_s`, `lidar_render_elapsed_s`, `gaussian_export_elapsed_s` | wall time of training, camera render, lidar render and Gaussian export (also `*_start_epoch_s` and `*_end_epoch_s`) |
| `TRAIN_SPLIT_FRACTION` | the dataset manifest's `session.train_fraction`, passed to `ns-train` as `--train-split-fraction`, so `config.yml` records it. Absent in older runs, whose `config.yml` says 0.5. |
| `split_render_check` | `match` when `split.json` was written; `FAILED (exit N)` when the job stopped instead (1 = renders or fractions disagree, 2 = outputs missing). Then there is no `end_utc` and no `split.json`. |

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
| `pred_accumulation` | float32 [C, H, W, 1] | accumulated opacity along the ray, Σ wᵢ = 1 − ∏(1 − αᵢ), in [0, 1]. The ungated point rule (PLAN §5 P1) keeps rays where it is above `eval.render_accumulation_threshold`, ignoring `pred_ray_drop_prob`. |
| `lidar_to_map` | float64 [4, 4] | scan pose in the **map frame** (T50). `xyz_map = (lidar_to_map @ np.c_[xyz, ones].T).T[:, :3]`, using only the first 3 columns of the point arrays |
| `lidar_to_world` | float32 [3, 4] | scan pose in the dataparser frame. Don't use it for cross-run analysis. |
| `dataparser_transform` | float32 [3, 4] | `w2m`: map frame → dataparser frame |
| `absolute_time_s` | float64 scalar | scan time in the `poses.npz` time base |
| `sweep_index` | int64 scalar | row of this scan in `poses.npz` `<sensor>__times` / `__poses_world_sensor` |

## Gaussians (`renders/gaussians.npz`)
- Every Gaussian of the model at the latest checkpoint (the one the renders use), made by `scripts/10_export_gaussians.py` (T56) from the checkpoint and `dataparser_transforms.json`. There is no rendering, so a Gaussian counts whether or not a sensor could see it (PLAN §5 P2, `docs/research/evaluation_caveats.md` §5).
- All Gaussians are static (no actors); the export fails if any belong to an actor.
- Gaussian k's 3-D alpha at a map-frame point x is `opacities[k] · exp(−½ (x − μ)ᵀ Σ⁻¹ (x − μ))`, with μ = `means[k]`, Σ = `covariances[k]`.
- To export a fetched run on the Mac, run the script inside the Docker image; it needs torch, not a GPU.

| key | dtype, shape | meaning |
|---|---|---|
| `means` | float32 [N, 3] | centre μ (m), **map frame** |
| `covariances` | float32 [N, 3, 3] | Σ = R·diag(std)²·Rᵀ (m²), map frame |
| `axis_std` | float32 [N, 3] | std along each principal axis (m), exp of the stored log-scale; the 3σ ellipsoid fits in a sphere of radius 3·max(`axis_std`) |
| `opacities` | float32 [N] | peak alpha, sigmoid of the stored logit, in (0, 1) |
| `dataparser_transform` | float32 [3, 4] | the `w2m` used to reach the map frame (same as in the lidar npz files) |
| `checkpoint_step` | int64 scalar | training step of the checkpoint |

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
- Runs before commit `a7202f7` (e.g. `may_chunks0_1_smoke-smoke-20261025-182824-b8104e1`) have camera renders only. They have no `lidar/`, no `lidar_metrics.json`, no `TRAIN_SEED` (their seed was nerfstudio's default, 42).
- Runs submitted before `09_extract_run_split.py` existed have no `split.json`, `split_render_check` or `TRAIN_SPLIT_FRACTION`. If such a run succeeded, write the file afterwards with the script. It uses the manifest's fraction and ignores `config.yml`'s 0.5. Failed runs stay undocumented.
- Runs rendered before 2026-10-26 have no `pred_accumulation` in their lidar npz files and no `renders/gaussians.npz`. Export the Gaussians afterwards with `scripts/10_export_gaussians.py` on the fetched checkpoint.
