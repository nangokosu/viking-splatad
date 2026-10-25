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
# Required environment (exported by submit_run.py via --export=ALL,...):
#   RUN_ID STAGE_ROOT SESSION GIT_SHA CODE_DIR ENV_PREFIX
#   MAX_NUM_ITERATIONS STEPS_PER_SAVE RENDER_POSE_SOURCE RENDER_DOWNSCALE_FACTOR
#
# Stage layout ($STAGE_ROOT): data/<session>/, code/<sha>/ (committed repo snapshot),
# envs/splatad-<recipe hash>/ (environment), runs/<run_id>/.

# Fail on any error, unset variable, or failure inside a pipeline.
set -euo pipefail

# ---- check the required variables (set -u alone would give a less clear message) ----------
for required_variable in RUN_ID STAGE_ROOT SESSION GIT_SHA CODE_DIR ENV_PREFIX \
    MAX_NUM_ITERATIONS STEPS_PER_SAVE RENDER_POSE_SOURCE RENDER_DOWNSCALE_FACTOR; do
  if [ -z "${!required_variable:-}" ]; then
    echo "ERROR: $required_variable is not set (submit with scripts/slurm/submit_run.py)" >&2
    exit 1
  fi
done
if [ ! -f "$ENV_PREFIX/.build-complete" ]; then
  echo "ERROR: environment $ENV_PREFIX is not built (submit_run.py build-env)" >&2
  exit 1
fi

# ---- (a) run directory and job_info.txt ---------------------------------------------------
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
      MAX_NUM_ITERATIONS STEPS_PER_SAVE RENDER_POSE_SOURCE RENDER_DOWNSCALE_FACTOR; do
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
  viking-data --data "$STAGE_ROOT/data" --session-name "$SESSION"
TRAIN_END_SECONDS=$(date +%s)
{
  echo "train_end_epoch_s: $TRAIN_END_SECONDS"
  echo "train_elapsed_s: $((TRAIN_END_SECONDS - TRAIN_START_SECONDS))"
} >> "$JOB_INFO"

# ---- (e) render ---------------------------------------------------------------------------
TRAIN_CONFIG="$RUN_DIR/train/$SESSION/splatad/$RUN_ID/config.yml"
if [ ! -f "$TRAIN_CONFIG" ]; then
  echo "ERROR: training config not found at $TRAIN_CONFIG" >&2
  exit 1
fi
RENDER_START_SECONDS=$(date +%s)
echo "render_start_epoch_s: $RENDER_START_SECONDS" >> "$JOB_INFO"
ns-render dataset \
  --load-config "$TRAIN_CONFIG" \
  --output-path "$RUN_DIR/renders" \
  --pose-source "$RENDER_POSE_SOURCE" \
  --rendered-output-names rgb gt-rgb \
  --image-format jpeg \
  --downscale-factor "$RENDER_DOWNSCALE_FACTOR"
RENDER_END_SECONDS=$(date +%s)

# ---- (f) done -------------------------------------------------------------------------------
{
  echo "render_end_epoch_s: $RENDER_END_SECONDS"
  echo "render_elapsed_s: $((RENDER_END_SECONDS - RENDER_START_SECONDS))"
  echo "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >> "$JOB_INFO"
echo "done: run $RUN_ID, train $((TRAIN_END_SECONDS - TRAIN_START_SECONDS)) s, render $((RENDER_END_SECONDS - RENDER_START_SECONDS)) s"
