#!/bin/bash
# One SplatAD run (train + render) in the native pinned environment, on one GPU node.
#
# The CS Lab GPU nodes have no container runtime, so runs use the environment built by
# scripts/slurm/build_env.sbatch from the Dockerfile's recipe (same pins), not the image.
#
# Submitted only by scripts/slurm/submit_run.py, which passes every resource (partition, gres,
# -c, --mem, -t, --output, --job-name) on the sbatch command line, derived from sinfo and
# configs/slurm.yaml. That is why this file has no #SBATCH resource lines.
#
# Storage: $STAGE_ROOT is on the persistent MSCAC workdisk (configs/slurm.yaml workdisk_root),
# so training and rendering write straight to $STAGE_ROOT/runs/<run_id>, where `submit_run.py
# fetch` reads them. Nothing goes to $HOME, which has a 10 GB quota.
#
# Required environment (exported by the generated <run dir>/job.sbatch that runs this file;
# submit_run.py build_job_script, CS Lab support's recommended way to pass variables):
#   RUN_ID STAGE_ROOT SESSION GIT_SHA CODE_DIR ENV_PREFIX
#   MAX_NUM_ITERATIONS STEPS_PER_SAVE TRAIN_SEED RENDER_POSE_SOURCE RENDER_DOWNSCALE_FACTOR
#   DATASET_START_FRACTION DATASET_END_FRACTION  (time window; 0 and 1 = whole recording)
#   ADD_MISSING_POINTS  (True/False: the session config's splatad.add_missing_points; True appends
#                        the dataset's recorded no-returns, lidar_no_return/, as 2000 m rays)
#   TRAIN_SPLIT_FRACTION  (the dataset manifest's session.train_fraction, passed to viking-data so
#                          config.yml records the fraction the dataparser actually uses; the
#                          profile's train_fraction instead, when it sets one)
#   TEST_BLOCK_COUNT  (the session config's split.test_block_count: the test set is that many
#                      contiguous time blocks shared by camera and lidar, T60; recorded in config.yml;
#                      ignored by the dataparser when TEST_SPLIT_METHOD is shared_interleaved)
#   TEST_BLOCK_CORE_FRACTION  (the session config's split.test_block_core_fraction: only the central
#                              fraction of each block is the test split, T62; the rest is guard,
#                              neither trained on nor rendered; recorded in config.yml; ignored
#                              with shared_interleaved)
#   MINIMUM_EVAL_POINTS  (the profile's minimum number of whole-stretch evals per run)
#   STEPS_PER_EVAL_ALL_IMAGES  (the eval interval k = (N-1)//MINIMUM_EVAL_POINTS, T63, derived by
#                               submit_run.py; ns-train --steps-per-eval-all-images)
#   TRAIN_METHOD RADAR_LIDAR_LOSS_WEIGHT RADAR_LIDAR_GATE_QUANTILE RADAR_LIDAR_GATE_CHUNK_SIZE
#   (the selected method and the radar opacity-loss settings; when TRAIN_METHOD is unset or splatad,
#    all four default below, so older stock jobs still start)
# Optional:
#   RADAR_INPUTS_DIR  (required by every radar method: the anchor artifact; absolute, or relative to
#                      $STAGE_ROOT/data/$SESSION)
#   RADAR_SEED_OPACITY  (optional override for the seed methods; unset = stock init_opacities)
#   RADAR_LIDAR_AUTO_WEIGHT RADAR_DIAGNOSTICS_PATH  (loss methods; default False and
#                                                    <run dir>/radar_diagnostics.jsonl)
#   TRAIN_LIDAR_ONLY  (True = lidar-only training, no photometric loss; default False)
#   TRAIN_SPLIT_FRACTION_OVERRIDE  (set only when the profile has its own train_fraction; equal to
#                                   TRAIN_SPLIT_FRACTION, passed as --train-split-fraction-override so
#                                   the dataparser splits with it instead of the manifest's value, T27)
#   TEST_SPLIT_METHOD  (the session config's split.method; set by submit_run.py for every new run.
#                       shared_interleaved = every ~1/(1-f)-th lidar sweep and the camera frames
#                       nearest it are held out, T71; middle_blocks = the T60 blocks. Unset = the
#                       dataparser decides from TEST_BLOCK_COUNT, as for runs before the variable)
#   TEST_RENDER_FRACTION  (the profile's or session's test_render_fraction; set by submit_run.py for
#                          every new run: an evenly spaced fraction of the held-out instants is the
#                          test split, rendered; the rest is guard, T72. Unset = 1.0)
#   RESUME_AFTER_TRAINING  (True only for `submit_run.py resume`: the run trained and plotted its
#                           curves, but a later step failed; this job reruns the steps after training
#                           in the same run dir with the code at RESUME_GIT_SHA (CODE_DIR), appending
#                           to job_info.txt instead of replacing it. GIT_SHA stays the training commit.)
#   RESUME_GIT_SHA  (required with RESUME_AFTER_TRAINING: the commit CODE_DIR holds)
#
# The ns-render step loads <run dir>/render_config/config.yml, the run's config.yml with point
# painting off (scripts/12_prepare_rerender.py write-run-config): painting cached the whole training
# set twice inside ns-render and OOM-killed job 222994. scripts/08 turns it off itself.
#
# After training, the overfitting plot script (scripts/11) writes <run dir>/overfitting/ (train-vs-test
# curves); its outcome goes to job_info.txt (overfitting_plot:) and end_utc requires it to be written.
# split.json (scripts/09) also records what training evaluated (selection.eval_during_training_s, counts val).
#
# Stage layout ($STAGE_ROOT): data/<session>/, code/<sha>/ (committed repo snapshot),
# envs/splatad-<recipe hash>/ (environment), runs/<run_id>/.

# Fail on any error, unset variable, or failure inside a pipeline.
set -euo pipefail

# Older stock jobs have no radar settings. Default only to the disabled stock
# method; an enabled radar method must receive all its settings explicitly.
TRAIN_METHOD=${TRAIN_METHOD:-splatad}
if [ "$TRAIN_METHOD" = splatad ]; then
  RADAR_LIDAR_LOSS_WEIGHT=${RADAR_LIDAR_LOSS_WEIGHT:-0}
  RADAR_LIDAR_GATE_QUANTILE=${RADAR_LIDAR_GATE_QUANTILE:-unused}
  RADAR_LIDAR_GATE_CHUNK_SIZE=${RADAR_LIDAR_GATE_CHUNK_SIZE:-unused}
fi

# ---- check the required variables (set -u alone would give a less clear message) ----------
for required_variable in RUN_ID STAGE_ROOT SESSION GIT_SHA CODE_DIR ENV_PREFIX TRAIN_METHOD \
    MAX_NUM_ITERATIONS STEPS_PER_SAVE TRAIN_SEED RENDER_POSE_SOURCE RENDER_DOWNSCALE_FACTOR \
    DATASET_START_FRACTION DATASET_END_FRACTION ADD_MISSING_POINTS TRAIN_SPLIT_FRACTION \
    TEST_BLOCK_COUNT TEST_BLOCK_CORE_FRACTION \
    RADAR_LIDAR_LOSS_WEIGHT RADAR_LIDAR_GATE_QUANTILE RADAR_LIDAR_GATE_CHUNK_SIZE \
    MINIMUM_EVAL_POINTS STEPS_PER_EVAL_ALL_IMAGES; do
  if [ -z "${!required_variable:-}" ]; then
    echo "ERROR: $required_variable is not set (submit with scripts/slurm/submit_run.py)" >&2
    exit 1
  fi
done
if [ ! -f "$ENV_PREFIX/.build-complete" ]; then
  echo "ERROR: environment $ENV_PREFIX is not built (submit_run.py build-env)" >&2
  exit 1
fi

# ---- (a) run directory and job_info.txt -----------------------------------------------------
# RUN_DIR holds everything the job writes (the Slurm log is already there: submit_run.py sets
# --output inside it).
RUN_DIR="$STAGE_ROOT/runs/$RUN_ID"
JOB_INFO="$RUN_DIR/job_info.txt"
mkdir -p "$RUN_DIR"
# Resume mode keeps the training job's record and appends this job's own details to it.
RESUMING=False
if [ "${RESUME_AFTER_TRAINING:-}" = True ]; then
  RESUMING=True
  if [ -z "${RESUME_GIT_SHA:-}" ] || [ ! -f "$JOB_INFO" ]; then
    echo "ERROR: resume needs RESUME_GIT_SHA and the run's job_info.txt (submit_run.py resume)" >&2
    exit 1
  fi
  {
    echo "resume_start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "resume_hostname: $(hostname)"
    echo "resume_slurm_job_id: ${SLURM_JOB_ID:-none}"
    echo "RESUME_GIT_SHA: $RESUME_GIT_SHA"
    echo "resume_code_dir: $CODE_DIR"
  } >> "$JOB_INFO"
  cat "$JOB_INFO"
fi
if [ "$RESUMING" = False ]; then
{
  echo "hostname: $(hostname)"
  echo "start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "slurm_cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
  echo "slurm_mem_per_node_mb: ${SLURM_MEM_PER_NODE:-unset}"
  echo "TRAIN_LIDAR_ONLY: ${TRAIN_LIDAR_ONLY:-False}"
  echo "RADAR_LIDAR_AUTO_WEIGHT: ${RADAR_LIDAR_AUTO_WEIGHT:-False}"
  echo "RADAR_DIAGNOSTICS_PATH: ${RADAR_DIAGNOSTICS_PATH:-}"
  for exported_variable in RUN_ID SESSION STAGE_ROOT GIT_SHA CODE_DIR ENV_PREFIX TRAIN_METHOD \
      MAX_NUM_ITERATIONS STEPS_PER_SAVE TRAIN_SEED RENDER_POSE_SOURCE RENDER_DOWNSCALE_FACTOR \
      DATASET_START_FRACTION DATASET_END_FRACTION ADD_MISSING_POINTS TRAIN_SPLIT_FRACTION \
      TEST_BLOCK_COUNT TEST_BLOCK_CORE_FRACTION \
      RADAR_LIDAR_LOSS_WEIGHT RADAR_LIDAR_GATE_QUANTILE RADAR_LIDAR_GATE_CHUNK_SIZE \
      MINIMUM_EVAL_POINTS STEPS_PER_EVAL_ALL_IMAGES; do
    echo "$exported_variable: ${!exported_variable:-}"
  done
  # Recorded only when set, so runs without a profile override keep their old job_info.txt keys.
  if [ -n "${TRAIN_SPLIT_FRACTION_OVERRIDE:-}" ]; then
    echo "TRAIN_SPLIT_FRACTION_OVERRIDE: $TRAIN_SPLIT_FRACTION_OVERRIDE"
  fi
  # Likewise the split method and render fraction (scripts/09 checks them against config.yml).
  if [ -n "${TEST_SPLIT_METHOD:-}" ]; then
    echo "TEST_SPLIT_METHOD: $TEST_SPLIT_METHOD"
  fi
  if [ -n "${TEST_RENDER_FRACTION:-}" ]; then
    echo "TEST_RENDER_FRACTION: $TEST_RENDER_FRACTION"
  fi
  echo "TRAIN_METHOD: $TRAIN_METHOD"
  # Every radar method has an inputs directory; only the seed methods have a seed opacity, so
  # the key is written only when it is set (no method list here: the selection in (d) validates it).
  if [ "$TRAIN_METHOD" != splatad ]; then
    echo "RADAR_INPUTS_DIR: ${RADAR_INPUTS_DIR:-}"
  fi
  if [ -n "${RADAR_SEED_OPACITY:-}" ]; then
    echo "RADAR_SEED_OPACITY: $RADAR_SEED_OPACITY"
  fi
  echo "environment build:"; sed 's/^/  /' "$ENV_PREFIX/.build-complete"
  echo "gpu (name, compute_cap, memory.total, driver_version):"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
} > "$JOB_INFO"
cat "$JOB_INFO"
fi

# ---- (b) environment ---------------------------------------------------------------------
# PATH, CUDA_HOME, LD_LIBRARY_PATH and the viking-data dataparser registration.
# shellcheck disable=SC1091
source "$ENV_PREFIX/viking-activate.sh"
# This run's committed src/ goes first on sys.path, ahead of the env's editable install
# (PYTHONPATH entries precede site-packages .pth paths), so the code is exactly GIT_SHA's.
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# The radar method input checks happen in (d), next to ns-train. The method registration is also
# exported here (this section runs on resume too, where (d) is skipped), so the ns-render and
# scripts of a resumed job find the radar method specs, as render_only.sbatch does. The literal must
# equal RADAR_METHOD_REGISTRATION in (d) (tests/unit/test_radar_review_fixes.py compares them).
if [ "$TRAIN_METHOD" != splatad ]; then
  export NERFSTUDIO_METHOD_CONFIGS="splatad-radar-init=viking_splatad.splatad.radar_init_method:RADAR_INIT_METHOD_SPEC,splatad-radar-lidar=viking_splatad.splatad.radar_lidar_method:RADAR_LIDAR_METHOD_SPEC,splatad-radar-init-lidar=viking_splatad.splatad.radar_init_lidar_method:RADAR_INIT_LIDAR_METHOD_SPEC"
fi
# Caches inside the run dir, so nothing depends on (or writes to) $HOME.
mkdir -p "$RUN_DIR/.cache/torch"
export TORCH_HOME="$RUN_DIR/.cache/torch" XDG_CACHE_HOME="$RUN_DIR/.cache"

# ---- (c) GPU sanity check: CUDA visible, CUDA extensions load, code comes from GIT_SHA ----
python -c "import torch, gsplat.csrc, tinycudann, viking_splatad; \
assert torch.cuda.is_available(), 'CUDA not available'; \
print('gpu_check:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); \
print('viking_splatad_from:', viking_splatad.__file__)" | tee -a "$JOB_INFO"

# ---- (d) train --------------------------------------------------------------------------
# --eval-whole-test-blocks True is a constant on purpose: every new run evaluates every held-out
# frame during training (whole T60 stretches, or every T71 instant incl. guard; old runs lack the
# field, so they keep the core-only eval).
# --machine.seed seeds python/numpy/torch (all CUDA devices) before setup (nerfstudio train.py).
# It does not make training bit-exact: the gsplat/SplatAD and tcnn backward passes accumulate
# gradients with float atomics, so the same seed gives runs that agree closely and then drift.
# The removable sources are switched off (src/viking_splatad/splatad/determinism.py, gated by
# VIKING_DETERMINISTIC=1): the lidar cache's 2-thread RNG race, cudnn.benchmark, and PyTorch's
# non-deterministic kernels. PYTHONHASHSEED (Python hash order) and CUBLAS_WORKSPACE_CONFIG
# (deterministic cuBLAS workspaces; the value is torch's documented one) must be set before the
# interpreter / CUDA start, so they are exported here, for the training step only.
# --experiment-name and --timestamp make the output path deterministic:
# $RUN_DIR/train/$SESSION/$TRAIN_METHOD/$RUN_ID/{config.yml,nerfstudio_models/step-*.ckpt}
# --pipeline.calc-fid-steps 99999999 turns off SplatAD's novel-view FIDs (lane/vertical shift, actor
# edits). The splatad method config computes them at step 30000, the last step of a 30001-iteration
# run, whenever that step is also an all-images eval (it is for stock's 2500 and our 600). The
# actor-edit FID indexes the dynamic actors, and our scenes have none, so it raises IndexError and
# kills the job (222951, 222952). 99999999 is unavoidably hardcoded: it is SplatAD's own "never"
# default (ADPipelineConfig.calc_fid_steps), and no training step reaches it, resumed runs included.
# The flag is a variable-length tuple: tyro reads values until the next flag, so another flag must
# follow it. Directly before viking-data it swallowed the subcommand name (jobs 222980, 222983).
# A resumed run trained already (its train_* lines are in job_info.txt), so it skips (d) and (d2).
# The block body stays unindented: tests locate the command as a line starting "ns-train splatad".
if [ "$RESUMING" = False ]; then
# A profile's own train fraction must also reach the dataparser as the override: it otherwise
# resolves the manifest's fraction (T27), and the split would disagree with config.yml (scripts/09).
TRAIN_FRACTION_OVERRIDE_ARGS=()
if [ -n "${TRAIN_SPLIT_FRACTION_OVERRIDE:-}" ]; then
  TRAIN_FRACTION_OVERRIDE_ARGS=(--train-split-fraction-override "$TRAIN_SPLIT_FRACTION_OVERRIDE")
fi
# The split method (T71) and render subset (T72) reach the dataparser only when submit_run.py set them.
TEST_SPLIT_ARGS=()
if [ -n "${TEST_SPLIT_METHOD:-}" ]; then
  TEST_SPLIT_ARGS+=(--test-split-method "$TEST_SPLIT_METHOD")
fi
if [ -n "${TEST_RENDER_FRACTION:-}" ]; then
  TEST_SPLIT_ARGS+=(--test-render-fraction "$TEST_RENDER_FRACTION")
fi
RADAR_MODEL_ARGS=()
RADAR_DATAPARSER_ARGS=()
LIDAR_ONLY_ARGS=()
if [ "${TRAIN_LIDAR_ONLY:-False}" = True ]; then
  LIDAR_ONLY_ARGS=(--pipeline.datamanager.train-lidar-only True)
fi
# All radar methods belong to the splatad family; only the method suffix changes, while trainer
# flags and their position before viking-data stay shared.
TRAIN_METHOD_SUFFIX="${TRAIN_METHOD#splatad}"
# The registration of every radar method spec ("<method>=<module>:<spec>", comma-joined), defined
# once. Registering all three for any radar method is harmless and keeps ns-render/ns-train alike.
RADAR_METHOD_REGISTRATION="splatad-radar-init=viking_splatad.splatad.radar_init_method:RADAR_INIT_METHOD_SPEC"
RADAR_METHOD_REGISTRATION+=",splatad-radar-lidar=viking_splatad.splatad.radar_lidar_method:RADAR_LIDAR_METHOD_SPEC"
RADAR_METHOD_REGISTRATION+=",splatad-radar-init-lidar=viking_splatad.splatad.radar_init_lidar_method:RADAR_INIT_LIDAR_METHOD_SPEC"
# The two capabilities of a method, computed once and reused below: USES_RADAR_SEEDS=1 = it appends
# radar seed Gaussians (needs the seed opacity); USES_RADAR_LOSS=1 = it adds the radar opacity loss.
# Stock sets neither; an unknown method is refused here.
USES_RADAR_SEEDS=0
USES_RADAR_LOSS=0
case "$TRAIN_METHOD" in
  splatad) ;;
  splatad-radar-init) USES_RADAR_SEEDS=1 ;;
  splatad-radar-lidar) USES_RADAR_LOSS=1 ;;
  splatad-radar-init-lidar) USES_RADAR_SEEDS=1; USES_RADAR_LOSS=1 ;;
  *)
    echo "ERROR: unsupported training method: $TRAIN_METHOD" >&2
    exit 1
    ;;
esac
if [ "$TRAIN_METHOD" = splatad ]; then
  # Stock: nothing registered, no radar arguments.
  export NERFSTUDIO_METHOD_CONFIGS=""
else
  # Every radar method reads the anchors from RADAR_INPUTS_DIR through the dataparser.
  if [ -z "${RADAR_INPUTS_DIR:-}" ]; then
    echo "ERROR: $TRAIN_METHOD requires RADAR_INPUTS_DIR" >&2
    exit 1
  fi
  export NERFSTUDIO_METHOD_CONFIGS="$RADAR_METHOD_REGISTRATION"
  # These arguments follow the viking-data subcommand, so use parser-local names. A relative
  # directory is resolved under the session's dataset directory.
  if [[ "$RADAR_INPUTS_DIR" = /* ]]; then
    RADAR_DATAPARSER_ARGS+=(--radar-inputs-dir "$RADAR_INPUTS_DIR")
  else
    RADAR_DATAPARSER_ARGS+=(--radar-inputs-dir "$STAGE_ROOT/data/$SESSION/$RADAR_INPUTS_DIR")
  fi
fi
if [ "$USES_RADAR_SEEDS" = 1 ]; then
  # Optional seed-opacity override; unset means the seeds follow stock's init_opacities.
  if [ -n "${RADAR_SEED_OPACITY:-}" ]; then
    RADAR_MODEL_ARGS+=(--pipeline.model.radar-seed-opacity "$RADAR_SEED_OPACITY")
  fi
fi
if [ "$USES_RADAR_LOSS" = 1 ]; then
  # Loss methods carry the radar opacity-loss settings (weight, gate, calibration, diagnostics).
  RADAR_MODEL_ARGS+=(
    --pipeline.model.radar-lidar-loss-weight "$RADAR_LIDAR_LOSS_WEIGHT"
    --pipeline.model.radar-lidar-gate-quantile "$RADAR_LIDAR_GATE_QUANTILE"
    --pipeline.model.radar-lidar-gate-chunk-size "$RADAR_LIDAR_GATE_CHUNK_SIZE"
    --pipeline.model.radar-lidar-auto-weight "${RADAR_LIDAR_AUTO_WEIGHT:-False}"
    --pipeline.model.radar-lidar-diagnostics-path "${RADAR_DIAGNOSTICS_PATH:-$RUN_DIR/radar_diagnostics.jsonl}"
  )
fi
TRAIN_START_SECONDS=$(date +%s)
echo "train_start_epoch_s: $TRAIN_START_SECONDS" >> "$JOB_INFO"
export VIKING_DETERMINISTIC=1 PYTHONHASHSEED="$TRAIN_SEED" CUBLAS_WORKSPACE_CONFIG=:4096:8
ns-train splatad"$TRAIN_METHOD_SUFFIX" \
  --vis tensorboard \
  --output-dir "$RUN_DIR/train" \
  --experiment-name "$SESSION" \
  --timestamp "$RUN_ID" \
  --max-num-iterations "$MAX_NUM_ITERATIONS" \
  --steps-per-save "$STEPS_PER_SAVE" \
  --pipeline.calc-fid-steps 99999999 \
  --machine.seed "$TRAIN_SEED" \
  --steps-per-eval-all-images "$STEPS_PER_EVAL_ALL_IMAGES" \
  ${LIDAR_ONLY_ARGS[@]+"${LIDAR_ONLY_ARGS[@]}"} \
  ${RADAR_MODEL_ARGS[@]+"${RADAR_MODEL_ARGS[@]}"} \
  viking-data --data "$STAGE_ROOT/data" --session-name "$SESSION" \
  --dataset-start-fraction "$DATASET_START_FRACTION" --dataset-end-fraction "$DATASET_END_FRACTION" \
  --add-missing-points "$ADD_MISSING_POINTS" \
  --train-split-fraction "$TRAIN_SPLIT_FRACTION" \
  ${TRAIN_FRACTION_OVERRIDE_ARGS[@]+"${TRAIN_FRACTION_OVERRIDE_ARGS[@]}"} \
  --test-block-count "$TEST_BLOCK_COUNT" \
  --test-block-core-fraction "$TEST_BLOCK_CORE_FRACTION" \
  ${TEST_SPLIT_ARGS[@]+"${TEST_SPLIT_ARGS[@]}"} \
  ${RADAR_DATAPARSER_ARGS[@]+"${RADAR_DATAPARSER_ARGS[@]}"} \
  --eval-whole-test-blocks True
TRAIN_END_SECONDS=$(date +%s)
{
  echo "train_end_epoch_s: $TRAIN_END_SECONDS"
  echo "train_elapsed_s: $((TRAIN_END_SECONDS - TRAIN_START_SECONDS))"
} >> "$JOB_INFO"
fi

# ---- (d2) overfitting curves ------------------------------------------------------------------
# Train-vs-test curves from the TensorBoard log: <run dir>/overfitting/overfitting_curves.{png,csv}.
# A failure is recorded and the job continues (the renders are worth more than the figure), but
# end_utc requires the figure (gate before end_utc).
# Nerfstudio writes under config.method_name, including for registered extensions.
TRAIN_CONFIG="$RUN_DIR/train/$SESSION/$TRAIN_METHOD/$RUN_ID/config.yml"
if [ ! -f "$TRAIN_CONFIG" ]; then
  echo "ERROR: training config not found at $TRAIN_CONFIG" >&2
  exit 1
fi
if [ "$RESUMING" = False ]; then
  if python "$CODE_DIR/scripts/11_plot_overfitting_curves.py" \
      --train-dir "$(dirname "$TRAIN_CONFIG")" --output-dir "$RUN_DIR/overfitting"; then
    echo "overfitting_plot: written" >> "$JOB_INFO"
  else
    # $? here is the plot script's exit code.
    PLOT_EXIT_CODE=$?
    echo "overfitting_plot: FAILED (exit $PLOT_EXIT_CODE)" >> "$JOB_INFO"
  fi
else
  # A resume renders the training job's final model: its last checkpoint is step N-1 (ns-train
  # saves step max_num_iterations - 1 at the end; 9 digits is nerfstudio's step-%09d name), and
  # its curves must already be written (submit_run.py resume checks both too, before queueing).
  FINAL_CHECKPOINT="$(dirname "$TRAIN_CONFIG")/nerfstudio_models/step-$(printf '%09d' $((MAX_NUM_ITERATIONS - 1))).ckpt"
  # -Fx: the whole line, literally (same test as the end_utc gate below, written so that gate stays unique).
  if [ ! -f "$FINAL_CHECKPOINT" ] || ! grep -Fxq 'overfitting_plot: written' "$JOB_INFO"; then
    echo "ERROR: resume needs $FINAL_CHECKPOINT and 'overfitting_plot: written' in $JOB_INFO" >&2
    exit 1
  fi
fi

# ---- (e) render ---------------------------------------------------------------------------
# The time window and the split (train fraction, test blocks, block cores) are stored in
# config.yml, so the render covers the same poses and splits as training. With
# RENDER_POSE_SOURCE=test only the T62 block cores are rendered (guard frames are in no split).
RENDER_START_SECONDS=$(date +%s)
echo "render_start_epoch_s: $RENDER_START_SECONDS" >> "$JOB_INFO"
# `ns-render dataset --downscale-factor` asserts the dataparser config has a downscale_factor
# field, which viking-data lacks (render.py:929, job 222565). Omitting the flag renders at native
# resolution, so pass it only when the factor is not 1 (numeric compare, so "1.0" counts as 1).
RENDER_DOWNSCALE_ARGS=()
if awk -v factor="$RENDER_DOWNSCALE_FACTOR" 'BEGIN { exit !(factor != 1) }'; then
  RENDER_DOWNSCALE_ARGS=(--downscale-factor "$RENDER_DOWNSCALE_FACTOR")
fi
# ns-render has no option for it, so it loads a copy of config.yml with point painting off (header).
RENDER_CONFIG="$RUN_DIR/render_config/config.yml"
python "$CODE_DIR/scripts/12_prepare_rerender.py" write-run-config \
  --train-config "$TRAIN_CONFIG" --output "$RENDER_CONFIG"
echo "render_config: $RENDER_CONFIG" >> "$JOB_INFO"
ns-render dataset \
  --load-config "$RENDER_CONFIG" \
  --output-path "$RUN_DIR/renders" \
  --pose-source "$RENDER_POSE_SOURCE" \
  --rendered-output-names rgb gt-rgb \
  --image-format jpeg \
  ${RENDER_DOWNSCALE_ARGS[@]+"${RENDER_DOWNSCALE_ARGS[@]}"}  # empty-array-safe under set -u (bash < 4.4)
RENDER_END_SECONDS=$(date +%s)
{
  echo "render_end_epoch_s: $RENDER_END_SECONDS"
  echo "render_elapsed_s: $((RENDER_END_SECONDS - RENDER_START_SECONDS))"
} >> "$JOB_INFO"

# ---- (e2) render lidar ------------------------------------------------------------------------
# ns-render's --render-point-clouds fails on SplatAD and only plots PNGs, so lidar goes through
# SplatAD's own get_lidar_outputs + filter_lidar_pred_and_gt in our script, at every input lidar
# pose of the same splits: renders/<split>/lidar/<idx>_<ts_ns>.npz + renders/<split>/lidar_metrics.json.
LIDAR_RENDER_START_SECONDS=$(date +%s)
echo "lidar_render_start_epoch_s: $LIDAR_RENDER_START_SECONDS" >> "$JOB_INFO"
python "$CODE_DIR/scripts/08_render_lidar_pointcloud.py" \
  --load-config "$TRAIN_CONFIG" \
  --output-path "$RUN_DIR/renders" \
  --pose-source "$RENDER_POSE_SOURCE" \
  --dataset-dir "$STAGE_ROOT/data/$SESSION"
LIDAR_RENDER_END_SECONDS=$(date +%s)
{
  echo "lidar_render_end_epoch_s: $LIDAR_RENDER_END_SECONDS"
  echo "lidar_render_elapsed_s: $((LIDAR_RENDER_END_SECONDS - LIDAR_RENDER_START_SECONDS))"
} >> "$JOB_INFO"

# ---- (e3) export Gaussians ----------------------------------------------------------------
# Every Gaussian of the latest checkpoint (the one the renders used), in the map frame, for the
# P2 occupancy grid (T56): renders/gaussians.npz. CPU only.
GAUSSIAN_EXPORT_START_SECONDS=$(date +%s)
echo "gaussian_export_start_epoch_s: $GAUSSIAN_EXPORT_START_SECONDS" >> "$JOB_INFO"
python "$CODE_DIR/scripts/10_export_gaussians.py" \
  --train-dir "$(dirname "$TRAIN_CONFIG")" \
  --output-path "$RUN_DIR/renders/gaussians.npz"
GAUSSIAN_EXPORT_END_SECONDS=$(date +%s)
{
  echo "gaussian_export_end_epoch_s: $GAUSSIAN_EXPORT_END_SECONDS"
  echo "gaussian_export_elapsed_s: $((GAUSSIAN_EXPORT_END_SECONDS - GAUSSIAN_EXPORT_START_SECONDS))"
} >> "$JOB_INFO"

# ---- (f) split record, then done ------------------------------------------------------------
# Only a run that produced its renders is documented: split.json + split_frames.csv are written
# here, after every render step succeeded, and only if the rendered file names equal the rebuilt
# split (scripts/09_extract_run_split.py writes nothing otherwise). A mismatch fails the job, so
# end_utc is written only for a complete, consistent run.
if python "$CODE_DIR/scripts/09_extract_run_split.py" --run-dir "$RUN_DIR" --dataset-dir "$STAGE_ROOT/data/$SESSION"; then
  echo "split_render_check: match" >> "$JOB_INFO"
  # ---- (e4) export the SuperSplat PLY ----------------------------------------------------------
  # renders/supersplat.ply: the same checkpoint as a SuperSplat-ready 3DGS PLY (script 22; colour baked
  # per Gaussian from the RGB decoder, cropped to the train-lidar box). The crop reads split.json and
  # split_frames.csv, which 09 has just written, so this step runs only after a matching split check.
  # It is a viewer product, not an evaluation input: a failure or timeout is recorded in job_info.txt
  # (never silent) and the job goes on to the end_utc gate.
  # --overwrite: a resume job (RESUME_AFTER_TRAINING) re-runs this step in a run that may hold a PLY.
  #
  # Time budget (CPU bake; a slow export must not let Slurm kill the job before end_utc):
  #   fixed cap     = measured export time x safety factor. Measured total (script 22, per-step seconds in
  #                   the saved logs, 30k radar init + loss runs):
  #                   docs/research/supersplat_export/export_log_may_short_grass-may_scene_a_radar_init_loss_30k-20261119-132048-d51a4d6.txt  184.16 s (4,801,958 of 5,000,000 kept)
  #                   docs/research/supersplat_export/export_log_june_tall_grass-june_scene_a_radar_init_loss_cam_30k-20261119-133010-d51a4d6.txt  174.46 s
  #                   The larger total (184.16 s, rounded up to 185) is used. Every run has the MCMC cap of
  #                   5,000,000 Gaussians (mcmc_cap_max, docs/research/splatad_extension_points.md 1.4), so
  #                   the export time is bounded by that of a run at the cap, which these two are.
  #                   Both were measured in the amd64 image emulated on the Mac, slower than a native node,
  #                   so the figure is an upper bound; the factor 4 only absorbs a node that is still
  #                   slower, without allowing a runaway.
  #   remaining cap = seconds left of the job's Slurm limit minus a reserve for the steps after this one
  #                   (overfitting gate, end_utc, summary: file reads and echoes, seconds in practice;
  #                   the reserve is generous because a kill here loses end_utc, and it also covers the
  #                   kill grace below).
  #   limit         = the smaller of the two. Slurm's end time comes from squeue (%e, ISO local time);
  #                   when squeue gives nothing or a non-date ("UNLIMITED", "N/A"), only the fixed cap applies.
  #   kill grace    = timeout sends SIGTERM at the limit (script 22 then removes its temp file) and SIGKILL
  #                   this many seconds later. 30 s: unlinking one temp PLY (<= ~1 GiB) takes well under that.
  SUPERSPLAT_MEASURED_EXPORT_SECONDS=185
  SUPERSPLAT_SAFETY_FACTOR=4
  SUPERSPLAT_RESERVE_SECONDS=300
  SUPERSPLAT_KILL_GRACE_SECONDS=30
  SUPERSPLAT_END_TIME_PATTERN='^[0-9]{4}-[0-9]{2}-[0-9]{2}T'
  # BEGIN supersplat_time_budget (the unit test sources this function)
  # Prints the export's timeout in seconds. Uses the globals above; calls squeue and date.
  supersplat_time_budget_seconds() {
    local fixed_cap_seconds=$((SUPERSPLAT_MEASURED_EXPORT_SECONDS * SUPERSPLAT_SAFETY_FACTOR))
    local job_end_text job_end_seconds remaining_budget_seconds
    job_end_text="$(squeue -h -j "${SLURM_JOB_ID:-}" -o %e 2>/dev/null || true)"
    # date -d runs only on an ISO timestamp: date -d "" means midnight, which would give a bogus negative budget.
    if [[ "$job_end_text" =~ $SUPERSPLAT_END_TIME_PATTERN ]]; then
      job_end_seconds="$(date -d "$job_end_text" +%s 2>/dev/null || true)"
      if [ -n "$job_end_seconds" ]; then
        remaining_budget_seconds=$((job_end_seconds - $(date +%s) - SUPERSPLAT_RESERVE_SECONDS))
        if [ "$remaining_budget_seconds" -lt "$fixed_cap_seconds" ]; then
          fixed_cap_seconds="$remaining_budget_seconds"
        fi
      fi
    fi
    echo "$fixed_cap_seconds"
  }
  # END supersplat_time_budget
  SUPERSPLAT_TIMEOUT_SECONDS="$(supersplat_time_budget_seconds)"
  SUPERSPLAT_EXPORT_START_SECONDS=$(date +%s)
  echo "supersplat_export_start_epoch_s: $SUPERSPLAT_EXPORT_START_SECONDS" >> "$JOB_INFO"
  if [ "$SUPERSPLAT_TIMEOUT_SECONDS" -lt 1 ]; then
    # No time left after the reserve: do not start an export that Slurm would kill.
    echo "supersplat_export: skipped (no job time left)" >> "$JOB_INFO"
  elif timeout -k "$SUPERSPLAT_KILL_GRACE_SECONDS" "$SUPERSPLAT_TIMEOUT_SECONDS" python "$CODE_DIR/scripts/22_export_supersplat_ply.py" \
      --run-dir "$RUN_DIR" --dataset-dir "$STAGE_ROOT/data/$SESSION" --overwrite; then
    echo "supersplat_export: written" >> "$JOB_INFO"
  else
    # $? here is the exit code of the condition: the export script's, or 124 when timeout killed it.
    SUPERSPLAT_EXIT_CODE=$?
    if [ "$SUPERSPLAT_EXIT_CODE" -eq 124 ]; then
      echo "supersplat_export: FAILED (exit 124 timeout after ${SUPERSPLAT_TIMEOUT_SECONDS} s; see Slurm log)" >> "$JOB_INFO"
    else
      echo "supersplat_export: FAILED (exit $SUPERSPLAT_EXIT_CODE; see Slurm log)" >> "$JOB_INFO"
    fi
  fi
  SUPERSPLAT_EXPORT_END_SECONDS=$(date +%s)
  {
    echo "supersplat_export_end_epoch_s: $SUPERSPLAT_EXPORT_END_SECONDS"
    echo "supersplat_export_elapsed_s: $((SUPERSPLAT_EXPORT_END_SECONDS - SUPERSPLAT_EXPORT_START_SECONDS))"
  } >> "$JOB_INFO"
else
  # $? here is the script's exit code: 1 = renders or train fractions disagree, 2 = outputs missing.
  SPLIT_EXIT_CODE=$?
  echo "split_render_check: FAILED (exit $SPLIT_EXIT_CODE; no split record written; reason in the Slurm log)" >> "$JOB_INFO"
  # (e4) needs the split record, so it did not run.
  echo "supersplat_export: skipped (no split record)" >> "$JOB_INFO"
  exit 1
fi
# The figure is part of a complete run: no end_utc without it.
if ! grep -q '^overfitting_plot: written$' "$JOB_INFO"; then
  echo "ERROR: overfitting curves missing (see overfitting_plot in $JOB_INFO); run not marked complete" >&2
  exit 1
fi
echo "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOB_INFO"
# A resumed job did not train, so its summary names the training job's time from job_info.txt.
if [ "$RESUMING" = True ]; then
  TRAIN_SUMMARY="$(grep '^train_elapsed_s: ' "$JOB_INFO" | tail -n 1 | cut -d' ' -f2) s (training job; resumed)"
else
  TRAIN_SUMMARY="$((TRAIN_END_SECONDS - TRAIN_START_SECONDS)) s"
fi
echo "done: run $RUN_ID, train $TRAIN_SUMMARY, render $((RENDER_END_SECONDS - RENDER_START_SECONDS)) s, lidar render $((LIDAR_RENDER_END_SECONDS - LIDAR_RENDER_START_SECONDS)) s"
