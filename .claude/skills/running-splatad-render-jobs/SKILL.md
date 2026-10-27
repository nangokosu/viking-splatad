---
name: running-splatad-render-jobs
description: Use when training SplatAD and rendering camera images or lidar point clouds on the CS Lab Slurm cluster - setting up your own cluster login, submitting a smoke or full run, checking whether a running job is stuck or failed, or fetching its camera/lidar renders back to your machine for evaluation.
---

# Running SplatAD train+render jobs on Slurm

## Overview
`scripts/slurm/submit_run.py` drives everything from your local machine (the submit host: macOS or Linux, repo root, `uv sync --extra dev` done). `splatad_run.sbatch` does these things, writing everything to `<stage root>/runs/<run_id>/` on the MSCAC workdisk (currently `/w/20262/<user>/viking-splatad`):
1. **Trains** with `ns-train splatad ... viking-data`, passing `--machine.seed` = the profile's `seed`.
2. **Renders camera images** with `ns-render dataset`: rgb + gt-rgb jpg.
3. **Renders lidar** with `scripts/08_render_lidar_pointcloud.py`, which calls SplatAD's `get_lidar_outputs` and then `filter_lidar_pred_and_gt(output_point_cloud=True)`.
4. Both renders cover every input pose of the profile's `render_pose_source`, which is **`test` only** (user's choice 2026-10-26): training views are never re-rendered, and the test split is only the middle 10% of each held-out block (T62, below). That keeps the fetch small: 22 camera frames and 24 lidar sweeps for the full May run instead of ~2400 each.

Jobs run the **committed** code at HEAD. They need a built env: if `submit` says it is missing, follow the `building-splatad-cluster-env` skill.

## One-time setup (each user, on your own machine)
Everyone uses their **own** CS Lab account. Your username and password never go into repo files, commits, env files or scripts. The repo stores only the host (`ssh_host` in `configs/slurm.yaml`).

1. **Tell ssh who you are.** In `~/.ssh/config` (local, never committed):
   ```
   Host comps0.cs.toronto.edu
     User <your CS username>
     ControlMaster auto
     ControlPath ~/.ssh/cm-%r@%h:%p
     ControlPersist 30m
   ```
   `submit_run.py` sshes to the bare host, so ssh takes `User` from this block. Alternatively, `export VIKING_SLURM_USER=<your CS username>` in your shell profile (`submit_run.py` then uses `<user>@host`). In the commands below, `<user>` means your own username.
2. **Avoid password prompts.** `submit_run.py` opens many ssh/rsync connections per command. Either install an ssh key (`ssh-keygen -t ed25519`, then `ssh-copy-id comps0.cs.toronto.edu`, typing your password once), or rely on the ControlMaster block above: run `ssh comps0.cs.toronto.edu` once, type the password, and keep that session open. Never use `sshpass` or store the password in a file.
3. **Check your workdisk directory.** `ssh comps0.cs.toronto.edu 'ls -d /w/*/$USER'`. The code expects `<workdisk_root>/<user>/` with `workdisk_root: /w/20262` in `configs/slurm.yaml`, and it creates `viking-splatad/` inside it. If your directory is on a different disk, tell the team before changing the shared config. If you have none, ask CS Lab support.
4. **Your stage root is your own.** Code, data, the environment (`envs/`) and runs live under *your* workdisk directory, so each user stages data and builds the env once (`building-splatad-cluster-env`). `runs/submitted.tsv` is your local, gitignored ledger.

## Submit (local machine, repo root)
```bash
git status                                   # commit first; untracked .DS_Store/reports are allowed
uv run python scripts/slurm/submit_run.py stage  --session <session>     # code/<sha> + data to the workdisk
uv run python scripts/slurm/submit_run.py submit --session <session> --profile smoke
```
- `submit` re-runs the gates (pytest, contract, clean tree) and prints `submitted job <job> run <run_id>`. The run is also logged in `runs/submitted.tsv`.
- Then wait for the job, and fetch it automatically if it succeeded (see "Fetch").
- If sbatch says `Unable to open file .../code/<sha>/...`, you forgot `stage` after committing.
- A `src/` change follows the test workflow in `CLAUDE.md`/`AGENTS.md` before you commit: new tests only; Claude Code runs the `transform-test-runner` subagent, other agents run `uv run python scripts/affected_tests.py --run`.
- Every profile in `configs/slurm.yaml` needs an integer `seed`, or `submit` refuses.

Re-run the CPU preflight whenever the dataparser, extraction or window changes:
- Script: `scripts/slurm/preflight_dataparser.py`, run in the image with `src/` bind-mounted.
- Window fractions come from a previous run's `job_info.txt`.
- The lidar overlays must land on edges.

**Timing (smoke, 52–67 s, 3000 iters, A6000):**
- train ~9 min
- camera render of 150 frames ~72–86 s
- lidar render of 150 scans ~73 s (job 222633)

## Seed and reproducibility
`seed` seeds Python, NumPy and torch (all CUDA devices) before setup (nerfstudio `train.py`). It covers Gaussian init, sensor sampling order, random background, and MCMC relocation and noise.

It does **not** make runs bit-identical:
- the gsplat/SplatAD backward kernels sum gradients with `gpuAtomicAdd`;
- so does the tcnn lidar decoder;
- nerfstudio sets `cudnn.benchmark=True`.

Same seed means statistically similar runs, not identical ones. Compare B vs R with the same seed, and don't read small metric gaps as signal without repeats.

## Train/test split record (successful runs only)
**The split has no randomness, and the seed does not affect it.** SplatAD keeps the frames inside the time window (`DATASET_START/END_FRACTION`, over camera and lidar times together, inclusive). Then the test set is K contiguous time blocks, the same for camera and lidar, centred at k/(K+1) of the window, each (1−f)/K of it long (T60, `VikingHillDataParser.test_block_count`). K is the session config's `split.test_block_count` (2), exported as `TEST_BLOCK_COUNT`. f is the manifest's `session.train_fraction` (0.9) unless overridden. Runs from before T60 used neurad's per-sensor linspace split instead (every ~10th frame).

**Only each block's core is test (T62).** The session config's `split.test_block_core_fraction` c (0.1), exported as `TEST_BLOCK_CORE_FRACTION` and passed as `--test-block-core-fraction`, keeps as test only the frames in [m − c·L/2, m + c·L/2) around each block's centre m. The rest of each block (45% on each side at c = 0.1) is **guard**: never trained on (train is unchanged), never rendered, in neither split. **The eval during training uses the whole blocks** (cores plus guard; see "Overfitting monitor"); only the renders use the cores. The guard also puts 45% of a block of time between the nearest train frame and any test frame. Whole May recording (241.09 s): camera 2162 train / 22 test / 217 guard, lidar 2172 / 24 / 216; blocks 74.34–86.39 s and 154.70–166.76 s, cores 79.76–80.97 s and 160.13–161.33 s (camera frames 795–806 and 1595–1604, lidar sweeps 798–809 and 1602–1613). The 15 s smoke window keeps only 1 camera and 1 lidar test frame.

`scripts/09_extract_run_split.py` rebuilds it on the CPU, without torch, and writes `<run dir>/split.json` and `split_frames.csv`:
- every frame with its split, index, ns stamp and time;
- the seed, git SHA, window, train fraction and its source, test block count, block bounds, core fraction and core bounds (`test_cores_s`);
- guard frames as split `guard` (in `frames` and in `counts`);
- SHA-256 hashes of the dataset's `manifest.json` and `poses.npz`;
- `membership_sha256`, a fingerprint of the split, and `render_check`.

**Only document runs that successfully returned their outputs. Never write, keep or cite a split record for a failed, killed or incomplete run.** The script enforces this. It writes nothing, and deletes any earlier record, unless:
- every render step finished and `config.yml` exists;
- camera and lidar renders exist for every rendered split;
- the train fractions agree (`job_info.txt`, `config.yml`, manifest), and so do the test block counts and core fractions (`job_info.txt`, `config.yml`);
- the render file names equal the rebuilt split.

Exit codes: 0 written, 1 inconsistent, 2 incomplete. `splatad_run.sbatch` runs it once, as the job's last step after the lidar render. It records `split_render_check: match`, or `FAILED (exit N)` and fails the job, before `end_utc`. So a run with `end_utc` has a verified `split.json`, and a run without one has none.

It was verified against the real dataparser on `may_chunks0_1_smoke`'s smoke window: 135/15 camera and 135/15 lidar, identical frames, both for the linspace split and for the block split (blocks 56.71–57.46 s and 61.69–62.44 s).

- **For a fetched run without the file** (older runs), or to check one:
  ```bash
  uv run python scripts/09_extract_run_split.py --run-dir runs/<run_id>    # dataset: data/intermediate/<SESSION>
  ```
  Use this only for runs that finished; for others it exits 2 and writes nothing. The output is byte-identical for the same inputs.
- **Same split for two runs or two people:** equal `membership_sha256` means the same frames. Equal `dataset.*_sha256` means the same dataset.
- **`config.yml` is still needed.** `ns-render` and `08_render_lidar_pointcloud.py` load the model from it, and it holds the window.
  - Its `train_split_fraction` is correct only for runs with `TRAIN_SPLIT_FRACTION` in `job_info.txt`: the job passes the manifest's value as `--train-split-fraction`.
  - Older runs say 0.5, nerfstudio's default, saved before the dataparser set the real 0.9.
  - Read the split from `split.json`, never from `config.yml`.
  - Don't use `manifest.json` `pipeline.splits` either: it covers the whole recording and ignores the window.
- To list a run's test frames: `python -c "import json; print([f['file'] for f in json.load(open('runs/<run_id>/split.json'))['frames'] if f['split'] == 'test'])"`.

## Overfitting monitor (every run)
Training evaluates held-out data throughout, and the job plots train against test, so a run shows whether the model memorises the training views.
- **What is evaluated:** SplatAD's all-images eval, split `val` = the **whole T60 test stretches** (`--eval-whole-test-blocks`, set by the job; T65). May: 239 camera frames, 240 lidar sweeps; June: 216 / 216. The renders (`ns-render`, script 08) still use split `test` = the T62 cores.
- **Cadence (T63):** the profile key `minimum_eval_points` (50) gives `MINIMUM_EVAL_POINTS`. The eval interval is `k = (max_num_iterations - 1) // minimum_eval_points`, exported as `STEPS_PER_EVAL_ALL_IMAGES`. 30001 steps: every 600 steps, 50 points. Smoke (3000): every 59, 50 points. `submit` refuses a profile that cannot reach the minimum.
- **Figure:** after training the job runs `scripts/11_plot_overfitting_curves.py`. It writes `<run dir>/overfitting/overfitting_curves.png` and `overfitting_curves.csv` (columns `metric,split,step,value`; split is `train`, `test` or `train_loss`).
  - One panel per metric: train (thin raw line plus a rolling median) and test (whole stretches, markers), and a Train Loss panel.
  - Test-only metrics (ssim, lpips, chamfer_distance, median_chamfer_distance) are titled "<metric> (no train value logged)".
  - Grey shading marks the steps where training runs at reduced resolution (6000 for stock SplatAD, read from `config.yml`): "gap not meaningful". The y-limits ignore the start-up spike.
- **How to read it:**
  - Overfitting = the test curve gets worse over a sustained stretch **after the shaded region** while the train curve keeps improving.
  - A constant train/test gap is **not** overfitting: train metrics include the camera optimiser (eval turns it off), and the held-out content differs.
  - Train values are per single image or sweep, so they are noisy; read the rolling median.
  - A smoke run lies entirely inside the shaded region, so it only checks that the figure is produced.
- **Cost:** estimated ~30 s per all-images eval, ~25 min per full run. Not measured yet: record it from `train_elapsed_s` on the first full run.
- **Re-plot any run** (the Mac venv lacks tensorboard and matplotlib, so use the cluster env or the image):
  ```bash
  python scripts/11_plot_overfitting_curves.py --train-dir <run>/train/<session>/splatad/<run_id> --output-dir <run>/overfitting
  ```
- Older runs have no `overfitting/` folder; their `split.json` lacks `selection.eval_whole_test_blocks`, `selection.eval_during_training_s` and `counts.<sensor>.val`.

## Monitor: stuck vs progressing
- The log, `job_info.txt` and all outputs are in the run dir: `<stage root>/runs/<run_id>/`.
- Milestones, in order: `gpu_check` → `train_elapsed_s` → `overfitting_plot: written` → `render_elapsed_s` (camera) → `lidar_render_elapsed_s` → `split_render_check: match` (`split.json` written) → `end_utc`.
- `overfitting_plot: FAILED (exit N)` does not stop the job (the renders still run), but the job then fails before `end_utc`.

**The progress table often stops appearing in the log while training is healthy.** It was seen right after `Caching eval lidars`, the step-500 eval, with 5+ silent minutes. A silent log is NOT evidence of a hang, so check the job itself before any `scancel`:
```bash
ssh <user>@comps0.cs.toronto.edu "srun --jobid=<job> --overlap -n1 -c1 bash -c \
 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; \
  ls -la --time-style=+%T <run dir>/train/*/splatad/*/ <...>/nerfstudio_models/'"
```
- Use `-n1`, or it runs once per task.
- Progressing: the GPU is busy and the `events.out.tfevents` mtime advances or a new `step-*.ckpt` appears.
- During the lidar step (after `render_elapsed_s`), replace the `ls` target with `<run dir>/renders/*/lidar/`: new `.npz` files should keep appearing.
- Truly stuck: the GPU is idle and nothing advances for >10 min.
- Held with `user_env_retrieval_failed_requeued_held`, or a silent death in seconds: see the building-splatad-cluster-env table.

## Cancelled, failed or unwanted runs
Remove every artifact of them (cluster run dir, fetched copy, ledger row) with the `cleaning-up-splatad-runs` skill (`submit_run.py clean`), never by hand.

## Fetch (automatic after every successful run)
**Fetching is part of every run: don't ask first.** Right after `submit`, start a background wait for the job to leave `squeue`. Then read `job_info.txt`:
- **`overfitting_plot: written`, `split_render_check: match` and `end_utc` present (success):** fetch right away, then report the results.
  ```bash
  uv run python scripts/slurm/submit_run.py fetch <run_id>
  ```
  Use a background command, since a smoke run is ~2.7 GB.
- **Any of the three missing (failed run):** don't fetch. Read the Slurm log over ssh, diagnose, and report. Fetch a failed run only if the user asks.
Result on your machine in `runs/<run_id>/`. Every file and key of every run folder is described once, in `runs/RUN_FOLDER_GLOSSARY.md` (committed; the rest of `runs/` is gitignored). Update it whenever the job's outputs change; a test checks that it documents every lidar npz key.
- `renders/test/{rgb,gt-rgb}/*.jpg`: one per test camera pose; file names are frame index + timestamp. (`renders/train/` exists only in runs from before 2026-10-26, which rendered `train+test`.)
- `renders/test/lidar/<idx>_<ts_ns>.npz`: one per test lidar pose, named exactly like the input sweep `data/intermediate/<session>/lidar/<sensor>/<idx>_<ts_ns>.npy`.
- `renders/test/lidar_metrics.json`: SplatAD's built-in per-scan lidar metrics, with `mean` and `per_scan` (depth_median_l2, intensity_rmse, ray_drop_accuracy, chamfer_distance, ...).
- `train/.../config.yml`, the latest `step-*.ckpt`, `job_info.txt`, and the Slurm log.
- `overfitting/overfitting_curves.png` and `.csv`: train-vs-test curves (see "Overfitting monitor").
- `split.json` and `split_frames.csv`: the exact train/test frames. They exist only for successful runs (see "Train/test split record").

A complete smoke run renders only its test-core frames: 1 camera frame and 1 lidar sweep. Check `overfitting_plot: written`, `split_render_check: match` and `end_utc` in `job_info.txt` instead of counting files by hand. A run without them failed: don't document it.

**Lidar npz keys** (full list in the docstring of `scripts/08_render_lidar_pointcloud.py`):

| key | shape | frame / meaning |
|---|---|---|
| `pred_points`, `pred_points_median` | [M,5] x,y,z,intensity,t | lidar frame; rays with ray-drop prob ≤ 0.5 |
| `gt_points` | [N,6] x,y,z,intensity,t,channel | the input returns, lidar frame, in input order (not row-aligned with `pred_points`). Width 5 if a sweep has no channel column |
| `raster_pts` + `pred_depth`, `pred_median_depth`, `pred_intensity`, `pred_ray_drop_prob` | [C,H,W,·] | per-ray, to recompute metrics |
| `lidar_to_map` | 4x4 | **map frame** (T50): `p_map = lidar_to_map @ [p;1]` |
| `lidar_to_world`, `dataparser_transform` | 3x4 | SplatAD dataparser frame, and w2m |
| `absolute_time_s`, `sweep_index` | scalar | poses.npz time base and row |

To compare with GT cuboids or raw sweeps, take points to the map frame with the stored `lidar_to_map` (the T50 result of `rendered_lidar.lidar_pose_in_map_frame`). Use only the xyz columns: `xyz_map = (lidar_to_map @ np.c_[pts[:, :3], np.ones(len(pts))].T).T[:, :3]`. Never apply `lidar_to_world` alone: it is in the dataparser frame, which depends on the time window.

Read the clouds with these limits in mind:
- **Rays come only from the input scan.** SplatAD renders along the input scan's own rays (`raster_pts`): its returns, plus its no-returns when the run trained with `add_missing_points: true` (08 keeps the trained flag; nerfstudio's `streamline_ad_config` would turn it off). `raster_did_return` / `pred_points_did_return` mark which rays returned, so eval can include or leave out no-returns. A rendered trunk behind grass shows up only as a different depth on an existing ray.
- `pred_points` uses expected depth and `pred_points_median` uses median depth. SplatAD reports Chamfer for both.
- **`ray_drop_accuracy` is ~1.0 and meaningless unless the run trained and was rendered with `add_missing_points: true`** (renders up to job 222869 lack the no-return rays even when trained with it) (Phase B, check the run's `config.yml`). That needs a dataset extracted with the no-return sidecar (`lidar_no_return/`, T51–T55). Without it, every ray is a return. Job 222633 shows it: 2,392,058 valid raster rays = 2,392,058 GT points, and the max ray-drop prob is 0.05.
- `<sensor>` is the manifest's only lidar (`ouster` for the current sessions).
- `test` is two held-out stretches of the drive (T60 block split, runs with `TEST_BLOCK_COUNT`); in older runs it is every ~10th frame interleaved with training frames, which measures interpolation only.

## Known traps
- Never pass `ns-render --downscale-factor` with viking-data: it asserts (render.py:929). The sbatch omits it when the factor is 1.
- Never use `ns-render --render-point-clouds` for SplatAD. It calls `get_outputs_for_lidar`, which fails with "Unknown sensor type", and it writes only PNG plots. Lidar goes through `08_render_lidar_pointcloud.py`.
- In that script, train scans are rasterised with `is_eval=True`. Without it, `_lidar_to_raster_pts` shuffles the points and keeps one per raster cell, and some returns get no ray.
- If the lidar step fails, the camera renders and the checkpoints are already in the run dir.
- `sacct` may fail from comps0 with `Connection refused`. Use `squeue` and `job_info.txt` instead.
- Runs from before 2026-10-26 wrote to scratch first; their leftovers there are lost when the scratch dir expires.
