# Overfitting Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every SplatAD Slurm run evaluates the whole test stretches during training (≥ 50 points), still renders only the 10% cores, and saves a train-vs-test curve figure that `fetch` brings to the Mac.

**Architecture:** `VikingHillDataParser` returns the whole T60 blocks for `split="val"` (what training's eval asks for) when a new config flag is set, and the T62 cores for every other eval split (what `ns-render`/`08` ask for). `submit_run.py` derives the eval cadence from each profile's `minimum_eval_points`. A new `scripts/11_plot_overfitting_curves.py`, run by the job after training, reads the TensorBoard log and writes `overfitting/overfitting_curves.{png,csv}`; its pure logic lives in a numpy-only module tested on the Mac.

**Tech Stack:** Python 3, numpy, nerfstudio/neurad-studio (cluster env and Docker image only), tensorboard 2.20.0 + matplotlib 3.10.9 (cluster env and image only), bash/Slurm, pytest.

**Spec:** `docs/superpowers/specs/2026-10-26-overfitting-monitor-design.md` (read it first).

## Global Constraints
- No change to `pyproject.toml`, `docker/constraints.txt`, `docker/Dockerfile.splatad`, `scripts/slurm/cluster_env.yaml`, `scripts/slurm/build_env.sbatch` (env hash must stay `splatad-3f478d42f203`).
- No copied/edited `config.yml`; no environment variable read by the dataparser.
- Existing tests are never edited, loosened or deleted; add new tests only (CLAUDE.md "Verification workflow").
- `split="train"` output is identical with the flag on or off; `get_dataparser_outputs("eval")` and `("test")` keep returning the cores.
- `minimum_eval_points: 50` in every profile; interval k = (N − 1) // P; refuse N − 1 < P.
- CLAUDE.md code rules: docstring per function, step comments, descriptive names (no `i`, `x`), every number derived or justified in a comment.
- New conversion/selection functions are `@transform("T63" | "T64" | "T65", ...)` and added to `PLANNED_TRANSFORM_IDS` in `src/viking_splatad/registry.py`; each has a `@pytest.mark.covers` known-answer test.
- Tests needing torch/neurad/tensorboard/matplotlib use `pytest.mark.skipif(<import check>, reason=...)` like `tests/unit/test_block_cores.py:279`; they run in the image, not the Mac venv.
- After each task touching `src/`, `tests/`, `scripts/` or `configs/`: run the `transform-test-runner` agent (main agent) with the changed files.

## Review Focus
1. **An old run** (config.yml without `eval_whole_test_blocks`, job_info without `STEPS_PER_EVAL_ALL_IMAGES`): `09` still writes the same record as before plus `eval_during_training_s` = the cores, and `11` still plots. Test in Task 5 and Task 4.
2. **A log with no all-images eval yet** (run killed before step k): `11` exits 1 with "no test points" and writes nothing. Test in Task 4.
3. **A resumed run with two `events.out.tfevents.*` files**: `11` reads both, steps merged in order. Test in Task 4.
4. **A future lidar-only run** (no `Train Metrics Dict/psnr`): `psnr` becomes a test-only panel, no crash. Test in Task 1.
5. **A profile too short for 50 points** (e.g. `max_num_iterations: 40`): `submit` refuses before queuing. Test in Task 3.

---

### Task 1: Pure curve helpers (cadence, tag pairing, smoothing, shading, CSV rows)

**Files:**
- Create: `src/viking_splatad/eval/training_curves.py`
- Modify: `src/viking_splatad/registry.py` (add "T63", "T64" with a comment line each)
- Test: `tests/unit/test_training_curves.py`

**Interfaces:**
- Produces (all numpy-only, importable in the Mac venv):
  - `TRAIN_METRIC_PREFIX = "Train Metrics Dict/"`, `TEST_METRIC_PREFIX = "Eval Images Metrics Dict (all images)/"`, `TRAIN_LOSS_TAG = "Train Loss"`
  - `NON_QUALITY_METRICS = frozenset({"fps", "num_camera_rays_per_sec", "num_lidar_rays_per_sec", "num_rays"})` (speed counters, not quality; comment justifies the list as nerfstudio's logged throughput names)
  - `@transform("T63", "eval cadence: largest interval giving at least P all-images evals in N training steps") eval_interval_for_minimum_points(max_num_iterations: int, minimum_eval_points: int) -> int`
  - `count_eval_points(max_num_iterations: int, eval_interval: int) -> int` (steps 1..N−1 divisible by k; nerfstudio `step_check` skips step 0)
  - `@dataclass(frozen=True) class MetricPanel: metric: str; train_tag: Optional[str]; test_tag: str`
  - `@transform("T64", "train/test metric pairing from TensorBoard tags") pair_metric_tags(scalar_tags: Iterable[str]) -> List[MetricPanel]` — paired panels first, then test-only, each group sorted by metric name; `NON_QUALITY_METRICS` dropped.
  - `smoothing_window(train_steps: np.ndarray, eval_interval: int) -> int` = max(1, round(eval_interval / median(diff(train_steps)))), forced odd (centred median).
  - `rolling_median(values: np.ndarray, window: int) -> np.ndarray` centred, same length, edges use the available neighbours.
  - `reduced_resolution_end_step(resolution_schedule: int, num_downscales: int) -> int` = resolution_schedule × num_downscales.
  - `curve_rows(series: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]) -> List[Tuple[str, str, int, float]]` — key (metric, "train"|"test"|"train_loss"), value (steps, values); rows sorted by (metric, split, step).

- [ ] **Step 1: Write the failing tests** in `tests/unit/test_training_curves.py`:
  - `test_interval_gives_at_least_minimum_points` (covers T63): for N in (3000, 30001, 51, 1000) and P = 50, k = `eval_interval_for_minimum_points(N, P)`; enumerate `sum(1 for step in range(1, N) if step % k == 0)` and assert it equals `count_eval_points(N, k)` and is ≥ 50; assert `eval_interval_for_minimum_points(30001, 50) == 600` and `(3000, 50) == 59` (first principles: 30000 // 50, 2999 // 50).
  - `test_interval_refuses_too_short_runs` (T63): `pytest.raises(ValueError)` for (50, 50) and (40, 50).
  - `test_pairing_pairs_shared_metrics_and_keeps_test_only` (T64): tags = the real smoke-run tag list subset `["Train Metrics Dict/psnr", "Train Metrics Dict/depth_median_l2", "Train Metrics Dict/camera_opt_vel_max", "Eval Images Metrics Dict (all images)/psnr", "Eval Images Metrics Dict (all images)/depth_median_l2", "Eval Images Metrics Dict (all images)/lpips", "Eval Images Metrics Dict (all images)/fps", "Train Loss"]` → exactly `[MetricPanel("depth_median_l2", train, test), MetricPanel("psnr", train, test), MetricPanel("lpips", None, test)]` (train-only `camera_opt_vel_max` and `fps` absent).
  - `test_pairing_lidar_only_run_has_test_only_psnr` (T64): without `Train Metrics Dict/psnr`, the psnr panel has `train_tag is None`.
  - `test_smoothing_window_and_rolling_median`: train steps every 10, k = 600 → window 61; `rolling_median(np.array([0, 0, 9, 0, 0.]), 3)` == `[0, 0, 0, 0, 0]`; a constant array is unchanged.
  - `test_reduced_resolution_end_step`: (3000, 2) → 6000.
  - `test_curve_rows_sorted_and_typed`: two series → rows sorted, steps `int`, values `float`.

- [ ] **Step 2: Run** `uv run pytest tests/unit/test_training_curves.py -v` → FAIL (module not found).
- [ ] **Step 3: Implement** the module per the Interfaces block; register T63/T64 in `PLANNED_TRANSFORM_IDS`. Docstring of `eval_interval_for_minimum_points` carries the proof: floor((N−1)/k) ≥ P because k ≤ (N−1)/P.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_training_curves.py tests/test_traceability.py -v` → PASS.
- [ ] **Step 5: Commit** `git add src/viking_splatad/eval/training_curves.py src/viking_splatad/registry.py tests/unit/test_training_curves.py && git commit -m "Training-curve helpers: eval cadence (T63), train/test metric pairing (T64)"`

### Task 2: Dataparser `val` = whole stretches (T65)

**Files:**
- Modify: `src/viking_splatad/dataset/run_split.py` (add T65 function)
- Modify: `src/viking_splatad/splatad/viking_dataparser.py:79-82` (new config field), `:113-157` (remember the split), `:159-189` (use T65)
- Modify: `src/viking_splatad/registry.py` (add "T65")
- Test: `tests/unit/test_eval_split_whole_blocks.py`

**Interfaces:**
- Produces:
  - `EVAL_WHOLE_BLOCKS_SPLIT_NAME = "val"` in `run_split.py` (comment: nerfstudio's training-time eval split name, `full_images_datamanager.py:115`).
  - `@transform("T65", "held-out split by requester: training eval (val) gets whole test blocks, renders (test/eval) get the cores") core_fraction_for_split(split_name: str, test_block_core_fraction: float, eval_whole_test_blocks: bool) -> float` → `WHOLE_BLOCK_CORE_FRACTION` iff `eval_whole_test_blocks and split_name == "val"`, else `test_block_core_fraction`.
  - `VikingHillDataParserConfig.eval_whole_test_blocks: bool = False` (docstring: True = training's `val` eval covers the whole T60 blocks, guard included; renders keep the cores; set by `splatad_run.sbatch`).
  - The dataparser stores the requested split as `self._requested_split` at the top of `_generate_dataparser_outputs` and `_filter_based_on_time` passes `core_fraction_for_split(self._requested_split, ...)` to `restrict_test_to_block_cores`. Docstring of `_generate_dataparser_outputs` updated: split is `"train"`, `"val"`, `"test"` or `"eval"`.

- [ ] **Step 1: Write failing tests**:
  - `test_core_fraction_for_split` (covers T65, Mac): `("val", 0.1, True)` → 1.0; `("val", 0.1, False)` → 0.1; `("test", 0.1, True)` → 0.1; `("eval", 0.1, True)` → 0.1; `("train", 0.1, True)` → 0.1.
  - `test_real_dataparser_val_is_whole_blocks` (covers T65, `skipif(not _neurad_ready())`, reuse `_write_synthetic` and the config arguments of `tests/unit/test_block_cores.py:279-297`, plus `eval_whole_test_blocks=True`): camera and lidar file names of `val` == `build_run_split_with_cores(..., test_block_core_fraction=1.0)` test frames; of `test` and `eval` == the c = 0.5 cores; `train` == the train frames of a parser built with the flag off.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_eval_split_whole_blocks.py -v` → first test FAIL (ImportError), second SKIPPED on the Mac.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_eval_split_whole_blocks.py tests/unit/test_block_cores.py tests/unit/test_block_split.py tests/unit/test_viking_dataparser.py tests/test_traceability.py -v` → PASS (neurad tests skipped on Mac; they run in Task 7).
- [ ] **Step 5: Commit** `"Dataparser: training eval (val) covers the whole test stretches, renders keep the cores (T65)"`

### Task 3: Profiles, submit and sbatch flags

**Files:**
- Modify: `configs/slurm.yaml` (add `minimum_eval_points: 50` with a comment to `smoke`, `may_scene_a`, `june_scene_a`)
- Modify: `scripts/slurm/submit_run.py` `build_job_environment` (`:646-710`)
- Modify: `scripts/slurm/splatad_run.sbatch` (header list, required-variable loop, job_info loop, `ns-train` call `:101-114`)
- Test: `tests/unit/test_slurm_eval_cadence.py`

**Interfaces:**
- Consumes: `eval_interval_for_minimum_points` (Task 1).
- Produces: job variables `MINIMUM_EVAL_POINTS` and `STEPS_PER_EVAL_ALL_IMAGES` (always exported; `build_job_environment` raises `ValueError` when the profile lacks an integer `minimum_eval_points`, same style as `seed`, or when T63 refuses). sbatch requires both, records both in `job_info.txt`, passes `--steps-per-eval-all-images "$STEPS_PER_EVAL_ALL_IMAGES"` to `ns-train` (before `viking-data`) and `--eval-whole-test-blocks True` after `viking-data` (comment: constant on purpose, every new run evaluates the whole stretches; old runs lack the field).

- [ ] **Step 1: Write failing tests** (use `load_submit_module()` from `tests/unit/test_slurm_submit.py` and the existing `build_job_environment` call pattern there):
  - `test_every_profile_reaches_minimum_eval_points`: for each profile in `configs/slurm.yaml`, `minimum_eval_points` is an int ≥ 50 and `count_eval_points(max_num_iterations, eval_interval_for_minimum_points(...)) >= 50`.
  - `test_job_environment_exports_eval_cadence`: profile with N = 30001, P = 50 → `STEPS_PER_EVAL_ALL_IMAGES == "600"`, `MINIMUM_EVAL_POINTS == "50"`.
  - `test_job_environment_refuses_missing_or_short_cadence`: no `minimum_eval_points` → `ValueError`; N = 40 → `ValueError`.
  - `test_sbatch_passes_eval_flags`: sbatch text contains `--steps-per-eval-all-images "$STEPS_PER_EVAL_ALL_IMAGES"` before the line with `viking-data`, and `--eval-whole-test-blocks True` after it; both variable names appear in the required-variable loop and the job_info loop.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_slurm_eval_cadence.py -v` → FAIL.
- [ ] **Step 3: Implement** per Interfaces; the profile comment says 50 is the user's minimum (2026-10-26) and gives k for that profile.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_slurm_eval_cadence.py tests/unit/test_slurm_submit.py tests/unit/test_block_cores.py -v` → PASS.
- [ ] **Step 5: Commit** `"Slurm: >= 50 whole-stretch evals per run (minimum_eval_points, --steps-per-eval-all-images, --eval-whole-test-blocks)"`

### Task 4: Plot script `11_plot_overfitting_curves.py`

**Files:**
- Create: `src/viking_splatad/io/nerfstudio_config.py` — move `YAML_NULL_LITERALS` and `read_config_scalar` here from `scripts/09_extract_run_split.py:111-131`; `09` imports them (same names stay available on the 09 module, so its tests are untouched).
- Create: `scripts/11_plot_overfitting_curves.py`
- Test: `tests/unit/test_plot_overfitting_curves.py`

**Interfaces:**
- Consumes: Task 1 helpers; `read_config_scalar`.
- Produces: CLI `python scripts/11_plot_overfitting_curves.py --train-dir <dir with config.yml + events.out.tfevents.*> --output-dir <run dir>/overfitting`. Exit 0: writes `overfitting_curves.png` and `overfitting_curves.csv` (header `metric,split,step,value`). Exit 1: no events file, or no test points (`"no test points"` in stderr), nothing written.
  - `load_scalar_series(train_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]` — tensorboard `EventAccumulator` over every `events.out.tfevents.*` in the dir with `size_guidance={"scalars": 0}` (keep all points); merged by step.
  - Figure: one panel per `MetricPanel` plus a `Train Loss` panel; train raw (thin, alpha) + `rolling_median` (window from `smoothing_window`), test as markers joined by a line; shaded `[0, reduced_resolution_end_step)` labelled "train at reduced resolution: gap not meaningful" (values from config.yml `resolution_schedule`, `num_downscales`); title line with run dir name and `steps_per_eval_all_images`. Import `matplotlib` with the `Agg` backend.
- [ ] **Step 1: Write failing tests**:
  - `test_read_config_scalar_moved` (Mac): `viking_splatad.io.nerfstudio_config.read_config_scalar` on a temp config with `  num_downscales: 2` → `"2"`; `null` → None.
  - `@skipif(no torch or tensorboard or matplotlib)` tests that write synthetic events with `torch.utils.tensorboard.SummaryWriter` into a temp train dir plus a minimal `config.yml` (`resolution_schedule: 30`, `num_downscales: 2`, `steps_per_eval_all_images: 20`):
    - `test_plot_writes_png_and_csv`: train psnr every step 1..100, test psnr at 20, 40, …, 100 → exit 0; CSV has 5 rows `psnr,test,…` and 100 rows `psnr,train,…`; PNG exists, size > 0.
    - `test_plot_fails_without_test_points`: only train tags → exit 1, `"no test points"` in stderr, output dir has no files.
    - `test_plot_merges_two_event_files`: test points 20, 40 in one writer and 60, 80 in a second file in the same dir → CSV test steps == [20, 40, 60, 80].
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_plot_overfitting_curves.py -v` → first FAIL, rest SKIPPED on Mac.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_plot_overfitting_curves.py tests/unit/test_run_split.py tests/unit/test_block_cores.py -v` → PASS.
- [ ] **Step 5: Commit** `"scripts/11: train-vs-test curves from the TensorBoard log (overfitting/overfitting_curves.png + .csv)"`

### Task 5: Job wiring and split record

**Files:**
- Modify: `scripts/slurm/splatad_run.sbatch` (new section `(d2)` after training; final check before `end_utc`)
- Modify: `scripts/09_extract_run_split.py` (`build_split_record` step 5-6)
- Test: `tests/unit/test_overfitting_job_wiring.py`

**Interfaces:**
- sbatch `(d2)`: `python "$CODE_DIR/scripts/11_plot_overfitting_curves.py" --train-dir "$(dirname "$TRAIN_CONFIG")" --output-dir "$RUN_DIR/overfitting"`; on success append `overfitting_plot: written`, on failure `overfitting_plot: FAILED (exit N)` and continue (`set -e` must not stop the job: wrap in `if …; then …; else …; fi` like section `(f)`). `TRAIN_CONFIG` is defined before `(d2)`. After the split check, before `end_utc`: `grep -q '^overfitting_plot: written$' "$JOB_INFO"` or print an error and `exit 1`.
- `09`: reads `eval_whole_test_blocks` via `read_config_scalar` (`"true"` → True; absent → False). Adds `selection.eval_whole_test_blocks` and `selection.eval_during_training_s` (= `test_blocks` if True, else `test_cores`), and per-sensor `counts[...]["val"]` (= test + guard if True, else test). `frames`, `membership_sha256` and `render_check` unchanged.
- [ ] **Step 1: Write failing tests**:
  - `test_sbatch_plots_after_training_and_gates_end_utc`: in the sbatch text, the `11_plot_overfitting_curves.py` call comes after `train_elapsed_s` and before `ns-render`; `overfitting_plot: written` and `overfitting_plot: FAILED` both appear; the `grep -q '^overfitting_plot: written$'` check comes after `split_render_check: match` and before `end_utc`.
  - `test_split_record_val_counts_whole_blocks`: reuse `write_synthetic_dataset`, `write_renders`, `run_script` from `tests/unit/test_run_split.py` (and the T62 run-dir setup in `tests/unit/test_block_cores.py`) with config.yml containing `eval_whole_test_blocks: true` → `selection.eval_during_training_s == selection.test_blocks_s` and `counts[sensor]["val"] == counts[sensor]["test"] + counts[sensor]["guard"]`.
  - `test_split_record_old_run_val_is_cores`: same without the field → `eval_during_training_s == test_cores_s`, `val == test`, and `membership_sha256` equals the value for the same inputs with the field present.
- [ ] **Step 2: Run** `uv run pytest tests/unit/test_overfitting_job_wiring.py -v` → FAIL.
- [ ] **Step 3: Implement** per Interfaces.
- [ ] **Step 4: Run** `uv run pytest tests/unit/test_overfitting_job_wiring.py tests/unit/test_run_split.py tests/unit/test_block_cores.py tests/unit/test_block_split.py -v` → PASS.
- [ ] **Step 5: Commit** `"Job: plot curves after training, require the figure for end_utc; split.json records the eval stretches"`

### Task 6: Documentation (skill, glossary, PLAN)

**Files:**
- Modify: `.claude/skills/running-splatad-render-jobs/SKILL.md`
- Modify: `runs/RUN_FOLDER_GLOSSARY.md`
- Modify: `docs/PLAN.md` §5 (one paragraph: training eval = whole stretches, renders = cores, overfitting figure)
- Modify: `configs/sessions/*.yaml` `split.test_block_core_fraction` comment (cores = rendered set; training eval uses whole blocks)

- [ ] **Step 1:** Skill: new section "Overfitting monitor" — what is evaluated (whole stretches, ≥ 50 points, k = (N−1)//P), where the figure is (`overfitting/`), how to read it (spec call 3a reading rule: shaded region, sustained test worsening after it while train improves; constant gap ≠ overfitting; noisy train), milestones list gains `overfitting_plot: written` after `train_elapsed_s`; "Fetch" success condition adds `overfitting_plot: written`; T62 paragraph says eval during training uses the whole blocks; timing note "eval cost to be measured on the first full run". Keep the skill's existing style (short bullets).
- [ ] **Step 2:** Glossary: `overfitting/overfitting_curves.png`, `overfitting_curves.csv` columns, job_info keys `MINIMUM_EVAL_POINTS`, `STEPS_PER_EVAL_ALL_IMAGES`, `overfitting_plot`, split.json keys `eval_whole_test_blocks`, `eval_during_training_s`, `counts.*.val`; note that older runs lack them.
- [ ] **Step 3:** Run `uv run pytest tests/test_agent_instructions_sync.py tests/unit/test_gaussian_occupancy.py tests/test_no_personal_info.py -q` → PASS.
- [ ] **Step 4: Commit** `"Docs: overfitting monitor in the Slurm skill, run-folder glossary and PLAN"`

### Task 7: Verification

- [ ] **Step 1:** Main agent runs the `transform-test-runner` agent on the full suite (configs/ changed). Expected: all pass, `existing tests modified: none`.
- [ ] **Step 2:** In the reference image (Docker Desktop must be running), with `src/`, `scripts/`, `tests/` bind-mounted as in `scripts/slurm/preflight_dataparser.py:12-15`, run the neurad/tensorboard-only tests: `pytest tests/unit/test_eval_split_whole_blocks.py tests/unit/test_plot_overfitting_curves.py tests/unit/test_block_cores.py -q` → PASS, none skipped.
- [ ] **Step 3:** In the image, run `11` on the fetched smoke run `runs/may_chunks0_1_smoke-smoke-20261027-001902-0b1f633/train/*/splatad/*/` into a scratch output dir → exit 0; view the PNG (real tags render, shading covers the whole 3000-step run).
- [ ] **Step 4:** Ask the user before any cluster run. If approved: `stage` + `submit --profile smoke` on `may_chunks0_1_smoke`; check `job_info.txt` has `STEPS_PER_EVAL_ALL_IMAGES: 59`, `overfitting_plot: written`, `split_render_check: match`, `end_utc`; fetch; the figure shows ≥ 50 test points; `renders/test/` still holds only the core frames; record the measured train time vs job 222933 (656 s) in the skill's timing section.
