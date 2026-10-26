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
#                          config.yml records the fraction the dataparser actually uses)
#
# Stage layout ($STAGE_ROOT): data/<session>/, code/<sha>/ (committed repo snapshot),
# envs/splatad-<recipe hash>/ (environment), runs/<run_id>/.

# Fail on any error, unset variable, or failure inside a pipeline.
set -euo pipefail

# ---- check the required variables (set -u alone would give a less clear message) ----------
for required_variable in RUN_ID STAGE_ROOT SESSION GIT_SHA CODE_DIR ENV_PREFIX \
    MAX_NUM_ITERATIONS STEPS_PER_SAVE TRAIN_SEED RENDER_POSE_SOURCE RENDER_DOWNSCALE_FACTOR \
    DATASET_START_FRACTION DATASET_END_FRACTION ADD_MISSING_POINTS TRAIN_SPLIT_FRACTION; do
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
{
  echo "hostname: $(hostname)"
  echo "start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "slurm_cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
  echo "slurm_mem_per_node_mb: ${SLURM_MEM_PER_NODE:-unset}"
  for exported_variable in RUN_ID SESSION STAGE_ROOT GIT_SHA CODE_DIR ENV_PREFIX \
      MAX_NUM_ITERATIONS STEPS_PER_SAVE TRAIN_SEED RENDER_POSE_SOURCE RENDER_DOWNSCALE_FACTOR \
      DATASET_START_FRACTION DATASET_END_FRACTION ADD_MISSING_POINTS TRAIN_SPLIT_FRACTION; do
    echo "$exported_variable: ${!exported_variable:-}"
  done
  echo "environment build:"; sed 's/^/  /' "$ENV_PREFIX/.build-complete"
  echo "gpu (name, compute_cap, memory.total, driver_version):"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
} > "$JOB_INFO"
cat "$JOB_INFO"

# ---- (b) environment ---------------------------------------------------------------------
# PATH, CUDA_HOME, LD_LIBRARY_PATH and the viking-data dataparser registration.
# shellcheck disable=SC1091
source "$ENV_PREFIX/viking-activate.sh"
# This run's committed src/ goes first on sys.path, ahead of the env's editable install
# (PYTHONPATH entries precede site-packages .pth paths), so the code is exactly GIT_SHA's.
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# Caches inside the run dir, so nothing depends on (or writes to) $HOME.
mkdir -p "$RUN_DIR/.cache/torch"
export TORCH_HOME="$RUN_DIR/.cache/torch" XDG_CACHE_HOME="$RUN_DIR/.cache"

# ---- (c) GPU sanity check: CUDA visible, CUDA extensions load, code comes from GIT_SHA ----
python -c "import torch, gsplat.csrc, tinycudann, viking_splatad; \
assert torch.cuda.is_available(), 'CUDA not available'; \
print('gpu_check:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); \
print('viking_splatad_from:', viking_splatad.__file__)" | tee -a "$JOB_INFO"


# ---- (d) train --------------------------------------------------------------------------
# --machine.seed seeds python/numpy/torch (all CUDA devices) before setup (nerfstudio train.py).
# It does not make training bit-exact: the gsplat/SplatAD and tcnn backward passes accumulate
# gradients with float atomics, so the same seed gives statistically similar runs only.
# --experiment-name and --timestamp make the output path deterministic:
# $RUN_DIR/train/$SESSION/splatad/$RUN_ID/{config.yml,nerfstudio_models/step-*.ckpt}
TRAIN_START_SECONDS=$(date +%s)
echo "train_start_epoch_s: $TRAIN_START_SECONDS" >> "$JOB_INFO"
ns-train splatad \
  --vis tensorboard \
  --output-dir "$RUN_DIR/train" \
  --experiment-name "$SESSION" \
  --timestamp "$RUN_ID" \
  --max-num-iterations "$MAX_NUM_ITERATIONS" \
  --steps-per-save "$STEPS_PER_SAVE" \
  --machine.seed "$TRAIN_SEED" \
  viking-data --data "$STAGE_ROOT/data" --session-name "$SESSION" \
  --dataset-start-fraction "$DATASET_START_FRACTION" --dataset-end-fraction "$DATASET_END_FRACTION" \
  --add-missing-points "$ADD_MISSING_POINTS" \
  --train-split-fraction "$TRAIN_SPLIT_FRACTION"
TRAIN_END_SECONDS=$(date +%s)
{
  echo "train_end_epoch_s: $TRAIN_END_SECONDS"
  echo "train_elapsed_s: $((TRAIN_END_SECONDS - TRAIN_START_SECONDS))"
} >> "$JOB_INFO"

# ---- (e) render ---------------------------------------------------------------------------
# The time window is stored in config.yml, so the render covers the same poses as training.
TRAIN_CONFIG="$RUN_DIR/train/$SESSION/splatad/$RUN_ID/config.yml"
if [ ! -f "$TRAIN_CONFIG" ]; then
  echo "ERROR: training config not found at $TRAIN_CONFIG" >&2
  exit 1
fi
RENDER_START_SECONDS=$(date +%s)
echo "render_start_epoch_s: $RENDER_START_SECONDS" >> "$JOB_INFO"
# `ns-render dataset --downscale-factor` asserts the dataparser config has a downscale_factor
# field, which viking-data lacks (render.py:929, job 222565). Omitting the flag renders at native
# resolution, so pass it only when the factor is not 1 (numeric compare, so "1.0" counts as 1).
RENDER_DOWNSCALE_ARGS=()
if awk -v factor="$RENDER_DOWNSCALE_FACTOR" 'BEGIN { exit !(factor != 1) }'; then
  RENDER_DOWNSCALE_ARGS=(--downscale-factor "$RENDER_DOWNSCALE_FACTOR")
fi
ns-render dataset \
  --load-config "$TRAIN_CONFIG" \
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
else
  # $? here is the script's exit code: 1 = renders or train fractions disagree, 2 = outputs missing.
  SPLIT_EXIT_CODE=$?
  echo "split_render_check: FAILED (exit $SPLIT_EXIT_CODE; no split record written; reason in the Slurm log)" >> "$JOB_INFO"
  exit 1
fi
echo "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOB_INFO"
echo "done: run $RUN_ID, train $((TRAIN_END_SECONDS - TRAIN_START_SECONDS)) s, render $((RENDER_END_SECONDS - RENDER_START_SECONDS)) s, lidar render $((LIDAR_RENDER_END_SECONDS - LIDAR_RENDER_START_SECONDS)) s"
