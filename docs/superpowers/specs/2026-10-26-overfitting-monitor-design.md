# Overfitting monitor for SplatAD runs: design

Date: 2026-10-26. Status: draft for the user's approval.

## Why
The user worries that SplatAD's stock 30k iterations overfit Scene A: the model memorises
the training views, so held-out quality stops improving or gets worse. Every run should
answer that from its own logs, with a figure saved on the Mac next to the renders. The May
and June runs (jobs 222941 and 222944) were cancelled and cleaned to add this before the
full runs.

## Goals
1. **Evaluation during training covers the whole test stretches.** Both T60 blocks, every
   frame in them: ~240 camera frames and ~240 lidar sweeps per full run (May: 22 + 217 guard
   camera frames, 24 + 216 lidar sweeps). Today it uses only the T62 cores (~22 per sensor).
2. **Renders still cover only the cores** (the middle 10% of each stretch, the session's
   `split.test_block_core_fraction`): `ns-render` RGB and the `08` lidar renders are
   unchanged in what they produce and what is fetched.
3. **At least 50 evaluation points per run** for the all-test-frames evaluation.
4. **A figure: each test metric drawn with its own train value**, against the step. It is
   saved in the run folder by the job and reaches the Mac with `submit_run.py fetch`.
5. The `running-splatad-render-jobs` skill describes the monitor, how to read the figure, and
   what counts as overfitting.

## Non-goals
- Changing the training frames: guard frames stay out of training, so train is byte-for-byte
  what T60/T62 give today.
- Early stopping or choosing the step count automatically. The figure informs the user's call.
- Off-trajectory novel-view evaluation.
- Rendering the guard frames to disk.

## Anti-goals (a failure even if it works)
- Any change to `pyproject.toml`, `docker/constraints.txt` or another environment recipe
  file: it changes the env hash and forces a multi-hour cluster rebuild. Everything needed
  (tensorboard 2.20.0, matplotlib 3.10.9) is already in the cluster env.
- A copied or edited `config.yml` for the render steps (the user asked for no copy).
- Global state (environment variables read by the dataparser) to switch the split.
- Weakening any existing test.

## Design decided with the user
### Split: `val` = whole stretches, `test` = cores, one config
nerfstudio already asks the dataparser for two different held-out splits:
- During training, `FullImageDatamanager` runs with `test_mode="val"` and requests
  `split="val"` (`full_images_datamanager.py:115, 284`; lidar: `full_images_lidar_datamanager.py:340`).
- `ns-render dataset --pose-source test` and `scripts/08_render_lidar_pointcloud.py` load with
  `test_mode="test"` and request `split="test"` (`render.py:962`; `08`:412).

`ADDataParser` treats both the same (`ad_dataparser.py:205`: anything but `train` is eval).
`VikingHillDataParser` gets one new config field, `eval_whole_test_blocks: bool`, default
`False` (today's behaviour, so old runs' `config.yml` and the existing tests calling
`get_dataparser_outputs("eval")` are unchanged). When `True`, and only for `split="val"`,
the whole T60 blocks are returned (core fraction 1, guard included). Every other non-train
split name (`test`, `eval`, ...) keeps the T62 cores; `split="train"` is unchanged.
`splatad_run.sbatch` passes `--eval-whole-test-blocks True`. `config.yml` thus records both
the core fraction (the rendered cores) and whether training evaluated the whole stretches;
that field, not a `job_info.txt` key, is what `09` and the plot script read to tell new
runs from old ones (absent = `False` = old run, eval set = cores).

### Evaluation cadence: ≥ 50 points
`steps_per_eval_all_images` (SplatAD default 2500, i.e. 12 points at 30001 steps) is derived
per run, not hardcoded. 50 is a hard minimum (the user: "at least 50 points"), set as
`minimum_eval_points: 50` in each profile of `configs/slurm.yaml`.
- Training runs steps 0 … N−1 (`trainer.py:298`) and `step_check` skips step 0
  (`misc.py:144`), so an interval k gives floor((N−1)/k) evals.
- Interval: k = floor((N−1) / P) for P = `minimum_eval_points`. Then (N−1)/k ≥ P, so
  floor((N−1)/k) ≥ P always. Refuse a profile with N−1 < P (k would be 0).
- 30001 steps: k = 600, 50 points. Smoke (3000): k = 59, 50 points (k = 60 would give 49).

`submit_run.py` computes and exports it, `splatad_run.sbatch` passes
`--steps-per-eval-all-images`, and `job_info.txt` records it. The known-answer test checks
the point count by enumerating the steps.

Cost (estimate from the cancelled May log, not measured): an all-images eval of 22 + 24 frames
took ~3 s, so ~30 s for ~480 frames, ~25 min added to a ~2 h run. The first full run measures
it; the skill's timing section is updated from that.

### Figure: each test metric with its own train value
One panel per metric, x = step, two series: train and test.
- **Train** values: `Train Metrics Dict/<metric>` (logged every 10 steps on the training
  batch: one camera image or one lidar sweep, so noisy).
- **Test** values: `Eval Images Metrics Dict (all images)/<metric>` (the whole stretches,
  ≥ 50 points).
- Paired metrics (present in both today): `psnr`, `depth_median_l2`, `depth_mean_rel_l2`,
  `median_depth_median_l2`, `median_depth_mean_rel_l2`, `intensity_rmse`,
  `ray_drop_accuracy`.
- The pairs are found from the tags present in the log (a metric is paired when both
  prefixes have it), not from a hardcoded list, so a future SplatAD metric is picked up.

## Calls made by Claude (the user can override)
1. **Unpaired test metrics** (`ssim`, `lpips`, `chamfer_distance`, `median_chamfer_distance`:
   SplatAD logs no train value for them) get their own panels with the test curve only,
   titled "no train value logged". Speed metrics (`fps`, `num_*_rays_per_sec`) are left out.
2. **Train curve smoothing**: raw train values in a light line, plus a centred rolling median
   whose window is the number of train log points between two test evals (so both curves
   have the same effective resolution). Test values are drawn as points joined by a line.
3. **Total train loss** (`Train Loss`) gets one extra panel, as context.
3a. **Train and test are not like-for-like early on; the figure shows it.** SplatAD trains
   at reduced resolution until `resolution_schedule × num_downscales` steps (1/4 until 3000,
   1/2 until 6000 with the stock values, `splatad.py:805-810`), while eval renders at full
   resolution. Train metrics also include the camera optimiser, which eval turns off
   (`use_camopt_in_eval=False`). The plot reads both schedule values from the run's
   `config.yml` and shades the reduced-resolution steps "train at reduced resolution: gap
   not meaningful". The skill's reading rule: overfitting = the test curve getting worse over a
   sustained stretch after the shaded region while train keeps improving. A constant
   train/test gap is not overfitting (camera optimiser, held-out content). The whole smoke run
   (3000 steps) lies in the shaded region, so smoke checks only that the figure is produced.
3b. **FID and other eval-only metrics** (e.g. `calc_fid_steps` at the end) follow call 1:
   a test-only panel if they appear in the log.
4. **Outputs** in `<run dir>/overfitting/`: `overfitting_curves.png` and
   `overfitting_curves.csv` (every plotted series: metric, split, step, value), so the
   numbers can be re-plotted on the Mac without tensorboard.
5. **Where it runs**: a new `scripts/11_plot_overfitting_curves.py`, run by
   `splatad_run.sbatch` right after training (before the renders: the curves exist even if
   a render fails). A plot failure does **not** stop the job: the sbatch records
   `overfitting_plot: FAILED (exit N)` in `job_info.txt` and continues, so a plotting bug
   never throws away a ~2 h run's renders. At the end, after the split check, the job fails
   (no `end_utc`) unless `overfitting_plot: written` is present, so every successful run
   has its figure. The script needs only the train dir, so it can be re-run on any cluster run.
6. **Testing without changing `pyproject.toml`**: the pure parts (pairing tags, the cadence
   formula, the smoothing window, the CSV rows) live in a numpy-only module under
   `src/viking_splatad/` with Mac unit tests. Reading event files and drawing need
   tensorboard and matplotlib, which exist only in the cluster env and the Docker image, so
   those tests are skipped in the Mac venv (like the existing torch-only tests) and run in
   the image.
7. **Split record** (`scripts/09_extract_run_split.py`): the guard frames stay `guard` in
   `split.json`, and the record adds the eval stretches (`eval_during_training_s` = the
   whole blocks when `config.yml` has `eval_whole_test_blocks: true`, else the cores; with
   per-sensor `val` counts) so it states what the curves measured. The render-name check
   still compares the renders with the cores.
   Expected counts (recomputed from the datasets): May val = 239 camera / 240 lidar
   (22 + 217, 24 + 216); June val = 216 camera / 216 lidar (21 + 195, 22 + 194).
8. **Transforms**: the `val`/`test` split behaviour and the cadence formula are registered
   transforms (next free ids, T63+), each with a covering known-answer test.

## Constraints
- CLAUDE.md project and code rules: registered transforms with tests, the contract, no
  personal info, run on Slurm only via `submit_run.py`, add tests and never weaken existing ones.
- Old runs (no `minimum_eval_points`, `val` = cores) must stay readable by `09` and by the
  plot script.
- `runs/RUN_FOLDER_GLOSSARY.md` documents the new `overfitting/` files.

## Left to the builder
- Exact panel layout, figure size, colours.
- How `VikingHillDataParser` learns the requested split (e.g. remembering it in
  `_generate_dataparser_outputs` before `_filter_based_on_time` runs), provided `train` is
  untouched and nothing global is used.
- How tests that need tensorboard/matplotlib are marked and run in the image.
