# SplatAD run folder glossary

This document covers every run folder in `runs/`. All runs from `scripts/slurm/splatad_run.sbatch` share one layout. It is the only copy: it lives in the repo at `runs/RUN_FOLDER_GLOSSARY.md` and is versioned with the code, so update it whenever the job's outputs change. Differences between older and newer runs are listed under "Version notes" at the end.

Path on the Mac after `submit_run.py fetch <run_id>`: `runs/<run_id>/`. On the cluster: `<stage root>/runs/<run_id>/` on the workdisk (currently `/w/20262/<user>/viking-splatad/runs/<run_id>/`).
`<run_id>` = `<session>-<profile>-<UTC yyyymmdd-HHMMSS>-<short sha>`. The short sha is the code commit; `GIT_SHA` in `job_info.txt` gives it in full.

```
runs/<run_id>/
├── job_info.txt                       settings, environment, GPU, timings
├── split.json                         exact train/test frames + provenance; successful runs only
├── split_frames.csv                   the same frames, one row each; successful runs only
├── overfitting/                       train-vs-test curves from the training log; runs since the overfitting monitor only
│   ├── overfitting_curves.png
│   └── overfitting_curves.csv
├── slurm-<job id>.out                 Slurm log (stdout + stderr of the job)
├── compare/                           only if you ran scripts/slurm/compare_renders.py on the Mac
├── rerenders/<label>/                 one per render-only job on this run's checkpoint (guard frames); see "Render-only jobs"
├── render_config/config.yml           config.yml with paint_points off, loaded by the ns-render step; runs since 2026-10-27
├── resume/job.sbatch                  only for a run resumed after training (submit_run.py resume); see job_info.txt resume_* keys
├── train/<session>/splatad/<run_id>/
│   ├── config.yml                     full nerfstudio config (loads the model; holds the time window)
│   ├── dataparser_transforms.json     w2m transform (dataparser frame), scale and time offset; written by 08 (ns-train skips it for SplatAD)
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
- `test`: held-out poses. **Runs with `TEST_BLOCK_COUNT` in `job_info.txt` (since T60):** two contiguous stretches of the drive, the same for camera and lidar, so a test camera image and the lidar sweep taken with it are both held out. **Older runs:** every ~10th frame (linspace), interleaved with training frames and chosen per sensor; that measures interpolation between views 0.1 s apart. **Runs with `TEST_BLOCK_CORE_FRACTION` < 1 (since T62):** only the middle part of each block is test.
- `guard` (only in `split.json` / `split_frames.csv`, since T62): frames inside a test block but outside its core. Never trained on, never rendered.
- **The split is deterministic, with no RNG.** First the time window: every camera and lidar frame with time in [t_min + D·`DATASET_START_FRACTION`, t_min + D·`DATASET_END_FRACTION`] (inclusive; t_min and D over all camera and lidar times). Then:
  - **block split (T60, `test_block_count` = K in `config.yml`):** over the windowed camera and lidar times together (t_lo, D), K test blocks [c_k − L/2, c_k + L/2) with L = D·(1−f)/K and c_k = t_lo + D·k/(K+1). K = 2 gives blocks centred at 1/3 and 2/3 of the window. Each sensor's test share is ~(1−f); counts can still differ by a frame between `rgb/` and `lidar/`.
  - **block cores (T62, `test_block_core_fraction` = c in `config.yml`, absent = 1):** of each block [a, b) with centre m, only [m − c·(b−a)/2, m + c·(b−a)/2) stays test; the rest of the block is `guard`. Train is unchanged.
  - **linspace (older runs, `test_block_count` absent):** per sensor, ceil(n·f) evenly spaced train frames; the rest are test.
  - At f = 1.0 every frame is in both. `TRAIN_SEED` does not affect either.
- K is the session config's `split.test_block_count`. The job passes it to `ns-train` as `--test-block-count` (`TEST_BLOCK_COUNT`), so `config.yml` records it and `ns-render` / `08` reuse the same split.
- f is the dataset manifest's `session.train_fraction` (0.9), unless `train_split_fraction_override` is set. The job passes it to `ns-train` as `--train-split-fraction` (`TRAIN_SPLIT_FRACTION`), so `config.yml` records the real value. **Runs without `TRAIN_SPLIT_FRACTION` in `job_info.txt` have `train_split_fraction: 0.5` in `config.yml`, which is wrong**: nerfstudio saved the class default before the dataparser set the real value. `manifest.json` `pipeline.splits` covers the whole recording, not the run's window, so it isn't the run's split either.

### `split.json` and `split_frames.csv`
Written by `scripts/09_extract_run_split.py` (T58 window, T59 linspace or T60 block split), which rebuilds the split on the CPU from `job_info.txt`, `config.yml` and the dataset. **Only successful runs are documented.** The job runs it once, as its last step, after every render. The script writes nothing, and deletes any earlier record, unless all of these hold:
- every render step finished (`lidar_render_elapsed_s` in `job_info.txt`) and `config.yml` exists;
- every split of `RENDER_POSE_SOURCE` has camera and lidar renders;
- `TRAIN_SPLIT_FRACTION`, `config.yml` and the manifest agree;
- `TEST_BLOCK_COUNT` (when recorded) equals `config.yml`'s `test_block_count`, and `TEST_BLOCK_CORE_FRACTION` (when recorded) equals its `test_block_core_fraction`;
- the rendered file names equal the rebuilt split.

Exit codes: 0 written, 1 inconsistent (renders or fractions disagree), 2 run incomplete. A failed or killed run therefore has no `split.json`. To (re)write it after `fetch`: `uv run python scripts/09_extract_run_split.py --run-dir runs/<run_id>`; the dataset defaults to `data/intermediate/<SESSION>`. The output is byte-identical for the same inputs.

| key (`split.json`) | meaning |
|---|---|
| `run` | `run_id`, `session`, `git_sha`, `train_seed`, `add_missing_points` from `job_info.txt` |
| `selection` | `dataset_start_fraction`, `dataset_end_fraction`, `window_bounds_s` (`poses.npz` seconds; `null` = whole recording), `train_fraction` and `train_fraction_source` (`manifest.session.train_fraction` or `override`), `algorithm`, `test_block_count` (`null` = linspace) and `test_blocks_s` (block [start, end) in `poses.npz` seconds), `test_block_core_fraction` (1.0 = whole blocks) and `test_cores_s` (the test part of each block, [start, end) seconds), `max_eval_frames` (always `null`), `eval_whole_test_blocks` (true: the eval during training used the whole test blocks) and `eval_during_training_s` (the stretches that eval covered: the blocks when true, the cores in older runs) |
| `dataset` | dataset folder name, and SHA-256 of its `manifest.json` and `poses.npz` (the same split needs the same dataset) |
| `counts` | per sensor: `kind`, `frames_in_recording`, `frames_in_window`, `train`, `test`, `guard` (0 without T62 cores), `val` (frames the eval during training used: whole blocks when `eval_whole_test_blocks`, else the cores; derived by script 09 from the record, test + guard, with `config.yml`'s `eval_whole_test_blocks` as the source of truth) |
| `membership_sha256` | fingerprint of which frame is in which split. Two runs or two people with equal values used the same frames. |
| `render_check` | per `<kind>/<split>` rendered: `match` (always true in a written record), `missing_from_renders`, `unexpected_in_renders` |
| `frames` | one entry per frame, same columns as the CSV |
| `seed_note`, `schema` | why the seed is recorded, and the layout version (`viking-splatad/run-split/2`; `/1` lacked `algorithm`'s T60 case, `test_block_count` and `test_blocks_s`; records written before T62 do not have `test_block_core_fraction`, `test_cores_s` and the `guard` count, an additive change within `/2`) |

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
| `RENDER_POSE_SOURCE`, `RENDER_DOWNSCALE_FACTOR` | rendered splits (`test` since 2026-10-26; `train+test` before) and camera resolution factor (1 = native) |
| `DATASET_START_FRACTION`, `DATASET_END_FRACTION` | time window as fractions of the recording's span (from the profile's `window_*` seconds) |
| `ADD_MISSING_POINTS` | `True`/`False`, the session config's `splatad.add_missing_points` (session named by the dataset's manifest), passed to `ns-train` as `--add-missing-points`. `True` adds the recorded no-returns (`lidar_no_return/`) as 2000 m rays. Absent in runs before 2026-10-26, which all ran with `False`. |
| `gpu (...)`, `gpu_check`, `viking_splatad_from` | GPU model, compute capability, driver; proof that CUDA and the right code were loaded |
| `train_elapsed_s`, `render_elapsed_s`, `lidar_render_elapsed_s`, `gaussian_export_elapsed_s` | wall time of training, camera render, lidar render and Gaussian export (also `*_start_epoch_s` and `*_end_epoch_s`) |
| `TEST_BLOCK_COUNT` | the session config's `split.test_block_count`, passed to `ns-train` as `--test-block-count`: the test set is that many time blocks shared by camera and lidar (T60). Absent in older runs (linspace split). |
| `TEST_BLOCK_CORE_FRACTION` | the session config's `split.test_block_core_fraction`, passed to `ns-train` as `--test-block-core-fraction`: only that central fraction of each block is test (T62), the rest is guard. Absent in older runs (whole blocks). |
| `TRAIN_SPLIT_FRACTION` | the dataset manifest's `session.train_fraction`, passed to `ns-train` as `--train-split-fraction`, so `config.yml` records it. Absent in older runs, whose `config.yml` says 0.5. When the Slurm profile sets its own `train_fraction`, it is that value instead. |
| `TRAIN_SPLIT_FRACTION_OVERRIDE` | present only when the Slurm profile sets its own `train_fraction` (e.g. smoke, 0.8): equal to `TRAIN_SPLIT_FRACTION`, passed to viking-data as `--train-split-fraction-override` so the dataparser splits with it rather than with the manifest's value. `split.json` then gives `train_fraction_source: override`. |
| `MINIMUM_EVAL_POINTS` | the profile's `minimum_eval_points`: the least number of all-images evals per run (50). Absent in older runs. |
| `STEPS_PER_EVAL_ALL_IMAGES` | eval interval k = (`MAX_NUM_ITERATIONS` - 1) // `MINIMUM_EVAL_POINTS` (T63), passed to `ns-train --steps-per-eval-all-images`. Absent in older runs. |
| `overfitting_plot` | `written` when `overfitting/` was produced after training; `FAILED (exit N)` otherwise (the job still renders, then fails before `end_utc`). Absent in older runs. |
| `overfitting_plot_recovered_utc` | Only after a plot-only failure that was repaired by hand: UTC time script 11 was re-run successfully on a compute node. Appended after `overfitting_plot: written`, before `end_utc`. Absent in normal runs. |
| `render_config` | path of `render_config/config.yml`, the copy of `config.yml` with `paint_points: false` that `ns-render` loaded. Painting only seeds the Gaussian colours, which the checkpoint replaces, but it caches the whole training set in RAM, and `ns-render` builds its data manager twice (job 222994 was OOM-killed). Script 08 turns painting off itself and loads `config.yml`. Absent in runs before 2026-10-27. |
| `resume_start_utc`, `resume_hostname`, `resume_slurm_job_id`, `RESUME_GIT_SHA`, `resume_code_dir` | only in a run resumed after training (`submit_run.py resume`): the training job trained and wrote its curves, then failed in a later step. A second job reran the steps after training with the code at `RESUME_GIT_SHA`, in the same run dir. These keys mark where its lines start. `GIT_SHA` stays the training commit. Keys written by both jobs (e.g. `render_start_epoch_s`, `gpu_check`) appear twice: the later one is the resume job's. |
| `split_render_check` | `match` when `split.json` was written; `FAILED (exit N)` when the job stopped instead (1 = renders or fractions disagree, 2 = outputs missing). Then there is no `end_utc` and no `split.json`. |

## Overfitting curves (`overfitting/`)
Made by `scripts/11_plot_overfitting_curves.py` from the TensorBoard log, right after training. The eval during training uses split `val` = the whole T60 test stretches (T65), not the T62 cores the renders use. Cadence is `STEPS_PER_EVAL_ALL_IMAGES` (T63).
- `overfitting_curves.png`: one panel per metric, train (raw line plus rolling median) against test (markers), plus Train Loss. Grey shading = steps at reduced resolution, where the gap is not meaningful.
- `overfitting_curves.csv`: columns `metric,split,step,value`; `split` is `train`, `test` or `train_loss`.
- Reading rule: overfitting is test getting worse over a sustained stretch after the shading while train keeps improving. A constant gap is not overfitting.
- Runs from before the monitor have no `overfitting/` folder, no `MINIMUM_EVAL_POINTS`, `STEPS_PER_EVAL_ALL_IMAGES` or `overfitting_plot` in `job_info.txt`, and no `eval_whole_test_blocks`, `eval_during_training_s` or `counts.*.val` in `split.json` (absent in records written before this change; re-running script 09 on an old run writes `false` and cores).

## Camera renders (`renders/<split>/rgb`, `gt-rgb`)
- There is one JPEG per camera pose (quality 100, still lossy) at native resolution. They come from `ns-render dataset`.
- `rgb/`: SplatAD's render at the input camera pose.
- `gt-rgb/`: the real image at the same pose, as the model saw it.
- Pair the two by file name. `scripts/slurm/compare_renders.py <run dir>` gives per-split PSNR and comparison sheets.

## Lidar renders (`renders/<split>/lidar/<idx>_<ts_ns>.npz`)
- There is one compressed npz per input lidar scan, made by `scripts/08_render_lidar_pointcloud.py`. The file name equals the input sweep's, `data/intermediate/<session>/lidar/<sensor>/<idx>_<ts_ns>.npy`.
- **Each file holds both the prediction and the ground truth for that pose**, so you can evaluate a scan from the file alone:

  | | keys |
  |---|---|
  | prediction (SplatAD's render) | `pred_points`, `pred_points_median` (+ `pred_points_did_return`); per ray: `pred_depth`, `pred_median_depth`, `pred_intensity`, `pred_ray_drop_prob`, `pred_accumulation` |
  | ground truth (the real scan) | `gt_points`, a copy of the input `.npy` sweep's rows; per ray: `raster_pts`, `raster_did_return` |
  | pose and identity | `lidar_to_map`, `lidar_to_world`, `dataparser_transform`, `absolute_time_s`, `sweep_index` |

  In `test/` the model never trained on the ground truth. In `train/` it did.
- SplatAD renders along the **rays of the real scan** (each real point's azimuth and elevation), plus, when the run trained with `add_missing_points: true`, the rays the real lidar fired but got **no return** on (`lidar_no_return/`, stored at the 2000 m placeholder depth). There are no rays in directions the lidar never fired.
- **Including or leaving out no-returns in evaluation:** use the masks. Returned rays are `raster_did_return`; no-return rays are `raster_pts[..., 2] > 0` and not `raster_did_return`. `pred_points[pred_points_did_return]` keeps only the points on rays where the real lidar got a return, which is what runs rendered before this change contained. The rest are points the model predicts on no-return rays (its ray-drop probability there was ≤ 0.5).
- Load a scan with `np.load(path)`.

| key | dtype, shape | meaning |
|---|---|---|
| `pred_points` | float32 [M, 5] | **prediction**: rendered point cloud: x, y, z (lidar frame, m), intensity (0–1), t (s within the sweep). Keeps rays with ray-drop probability ≤ 0.5 and a valid real ray. xyz = expected depth × ray direction + velocity × t (rolling shutter). |
| `pred_points_median` | float32 [M, 5] | same rays, using the **median** depth along the ray (sharper surfaces) instead of the expected depth |
| `pred_points_did_return` | bool [M] | per row of `pred_points` / `pred_points_median`: True if its ray returned in the real scan, False if it is a no-return ray (only with `add_missing_points: true`) |
| `gt_points` | float32 [N, 5 or 6] | **ground truth**: the real points that returned (x, y, z, intensity, t, channel), lidar frame, after the ego-vehicle filter. These are the input sweep's own rows (in job 222869 they are identical to the input `.npy`): channel (laser ring id, 0 to beams−1) is there only if the sweep stores it, which the current sessions do, so N × 6. The rows are in input order, not row-aligned with `pred_points` (raster order) even when M = N, so pair them by nearest neighbour or through the raster, never by row index. |
| `raster_pts` | float32 [C, H, W, 5] | the real scan on SplatAD's ray grid: azimuth (deg), elevation (deg), depth (m), t (s), intensity. H = beams, W = azimuth bins, C > 1 only when a cell holds several points. depth 0 = no ray in that cell. No-return rays (only with `add_missing_points: true`) hold the 2000 m placeholder depth, above the return threshold (1000 m). |
| `raster_did_return` | bool [C, H, W] | True where the cell holds a ray that returned: 0 < depth ≤ the lidar's `valid_lidar_distance_threshold` (1000 m). No-return rays: depth > 0 and False. Empty cells: depth 0 and False. |
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
- Every Gaussian of the model at the latest checkpoint (the one the renders use), made by `scripts/10_export_gaussians.py` (T56) from the checkpoint and `dataparser_transforms.json` (written by `08_render_lidar_pointcloud.py`; ns-train skips it for SplatAD). There is no rendering, so a Gaussian counts whether or not a sensor could see it (PLAN §5 P2, `docs/research/evaluation_caveats.md` §5).
- It runs as a job step after the lidar render and before the split record (`gaussian_export_elapsed_s` in `job_info.txt`, 16 s in job 222869). If it fails, the job stops and there is no `split.json`.
- All Gaussians are static (no actors); the export fails if any belong to an actor.
- **N is the MCMC cap, not a scene measurement.** SplatAD's MCMC strategy keeps the count at `mcmc_cap_max` (`config.yml`, 5,000,000), and relocates low-opacity Gaussians rather than pruning them. So N is the same for every run with the same cap.
  - In job 222869 the median opacity is 0.078 and only 15.7 % exceed 0.5, so threshold or weight by `opacities` before counting occupancy.
  - The file is ~256 MB at 5 M Gaussians.
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
| `ray_drop_accuracy` | fraction of valid rays where (ray-drop prob > 0.5) equals (the real ray did not return). Informative only for runs trained **and rendered** with `add_missing_points: true`. Job 222869 trained with it, but 08 then rendered without the no-return rays (nerfstudio's `streamline_ad_config` turned the flag off; fixed after that run). Its score therefore equals `len(pred_points)` / `len(gt_points)` (train 0.969, test 0.947): the share of real returns the model predicts will return. `add_missing_points: true` needs a dataset extracted after 2026-10-26: the converter keeps no-returns out of the returns files and writes them to `lidar_no_return/` (T51–T54), and the dataparser adds them as 2000 m rays (T55). Without that, every ray is a return. The score is then 1 whenever the model predicts no drop, and the ray-drop loss only ever sees "returned" labels, so ignore it (all runs up to job 222633). | 0–1, higher |
| `chamfer_distance` | (Σ over `pred_points` of the squared distance to the nearest real point + Σ over real points of the squared distance to the nearest `pred_points` point) / N_real. Note that SplatAD divides **both** sums by the real point count. Missing when either cloud is empty. | m², lower |
| `median_chamfer_distance` | same, using `pred_points_median` | m², lower |

## Render-only jobs (`rerenders/<label>/`)
Made by `submit_run.py render` and `scripts/slurm/render_only.sbatch` (skill `rendering-splatad-checkpoints`). Each job renders **guard** frames of this run (split `guard` in `split.json`: never trained on, never rendered by the run) from its `config.yml` and latest checkpoint, without training. `<label>` = `render-<UTC yyyymmdd-HHMMSS>-<short sha>`; the short sha is the render code.

| file | meaning |
|---|---|
| `selection.json` | `camera`, `lidar`: the requested frame indices (sorted); `files`: their dataset paths, from `split.json`; `source_split_membership_sha256`: the `split.json` they were checked against; `requested_utc` |
| `job.sbatch`, `slurm-<job id>.out` | the generated job script (its exported variables) and the Slurm log |
| `job_info.txt` | `RUN_ID`, `RERENDER_LABEL`, `SESSION`, `GIT_SHA` (**render code**), `TRAIN_GIT_SHA` (the code that trained the checkpoint), `CODE_DIR`, `ENV_PREFIX`, `RENDER_DOWNSCALE_FACTOR` (the run's); host, GPU and `gpu_check` as for a run; `render_config`; `render_elapsed_s`, `lidar_render_elapsed_s` (+ epochs); `render_check` (`match`, or `FAILED (exit N)`: 1 = renders differ from the selection, 2 = outputs missing); `end_utc` only after `render_check: match` |
| `config.yml` | the run's `config.yml` with the dataparser's `render_frame_selection_file` = this `selection.json` (T66: split `test`/`eval` = exactly the selected guard frames; `train`, `val`, the window, `w2m` and time origin unchanged). The run's own `config.yml` is never edited. |
| `dataparser_transforms.json` | written by script 08; the same `w2m` as the run's |
| `renders/test/{rgb,gt-rgb}/<idx>_<ts_ns>.jpg`, `renders/test/lidar/<idx>_<ts_ns>.npz`, `renders/test/lidar_metrics.json` | same formats and keys as the run's `renders/test/` (sections above). The folder is called `test` because ns-render and 08 name folders by split; the frames are guard frames. |

A failed render job leaves its folder but never changes the run's own `job_info.txt`, so `clean` still sees the run as successful. Deleting the run deletes its `rerenders/`. While a render job is queued or running, `clean` keeps the run.

## Version notes
- Runs before commit `a7202f7` (e.g. `may_chunks0_1_smoke-smoke-20261025-182824-b8104e1`) have camera renders only. They have no `lidar/`, no `lidar_metrics.json`, no `TRAIN_SEED` (their seed was nerfstudio's default, 42).
- Runs submitted before `09_extract_run_split.py` existed have no `split.json`, `split_render_check` or `TRAIN_SPLIT_FRACTION`. If such a run succeeded, write the file afterwards with the script. It uses the manifest's fraction and ignores `config.yml`'s 0.5. Failed runs stay undocumented.
- Lidar renders made before the no-return render fix (all runs up to and including job 222869) have no `raster_did_return` or `pred_points_did_return`, and their rasters hold no no-return rays even if the run trained with `add_missing_points: true`. Their `chamfer_distance` and `ray_drop_accuracy` are not comparable with later renders.
- Runs rendered before 2026-10-26 have no `pred_accumulation` in their lidar npz files and no `renders/gaussians.npz`. Export the Gaussians afterwards with `scripts/10_export_gaussians.py` on the fetched checkpoint.
