#!/bin/bash
# Render-only job: guard frames of a finished SplatAD run, from its checkpoint, on one GPU node.
#
# A run renders only its T62 test cores; the rest of each T60 test block is guard (split "guard" in
# the run's split.json): never trained on, never rendered. This job renders the guard frames that
# `submit_run.py render` validated against that split.json and wrote to selection.json. No training.
#
# Code: the job runs the code at GIT_SHA (staged code/<sha>, normally newer than the run), with the
# run's own config.yml and latest checkpoint. TRAIN_GIT_SHA is the code that trained the checkpoint.
# The run's config.yml is never edited: scripts/12_prepare_rerender.py write-config copies it into
# the rerender folder with the dataparser's render_frame_selection_file set (T66), so the `test`
# split ns-render and scripts/08 ask for is exactly the selected guard frames.
#
# Submitted only by scripts/slurm/submit_run.py render, which passes every resource (partition, gres,
# -c, --mem, -t, --output, --job-name) on the sbatch command line, derived from sinfo and
# configs/slurm.yaml (render_only.time_limit). That is why this file has no #SBATCH resource lines.
#
# Storage: everything goes to $STAGE_ROOT/runs/<run_id>/rerenders/<label>/ on the persistent MSCAC
# workdisk, where `submit_run.py fetch <run_id>` copies it back with the rest of the run. Nothing goes
# to $HOME, which has a 10 GB quota.
#
# Required environment (exported by the generated <rerender dir>/job.sbatch that runs this file;
# submit_run.py build_job_script, CS Lab support's recommended way to pass variables):
#   RUN_ID RERENDER_LABEL STAGE_ROOT SESSION CODE_DIR ENV_PREFIX
#   GIT_SHA  (the render code, HEAD at submission)
#   TRAIN_GIT_SHA  (the code that trained the checkpoint: the run's job_info.txt GIT_SHA)
#   RENDER_DOWNSCALE_FACTOR  (the run's own camera resolution factor, from its job_info.txt)
#
# Rerender layout: selection.json (written by submit_run.py), job.sbatch, job_info.txt, slurm-<job>.out,
# config.yml (the run's, with the selection), dataparser_transforms.json (written by scripts/08),
# renders/test/{rgb,gt-rgb}/<idx>_<ts_ns>.jpg, renders/test/lidar/<idx>_<ts_ns>.npz, lidar_metrics.json.

# Fail on any error, unset variable, or failure inside a pipeline.
set -euo pipefail

# ---- check the required variables (set -u alone would give a less clear message) ----------
for required_variable in RUN_ID RERENDER_LABEL STAGE_ROOT SESSION GIT_SHA TRAIN_GIT_SHA CODE_DIR \
    ENV_PREFIX RENDER_DOWNSCALE_FACTOR; do
  if [ -z "${!required_variable:-}" ]; then
    echo "ERROR: $required_variable is not set (submit with scripts/slurm/submit_run.py render)" >&2
    exit 1
  fi
done
if [ ! -f "$ENV_PREFIX/.build-complete" ]; then
  echo "ERROR: environment $ENV_PREFIX is not built (submit_run.py build-env)" >&2
  exit 1
fi

# ---- (a) rerender directory and job_info.txt ------------------------------------------------
# RUN_DIR is the finished run (read only); RERENDER_DIR holds everything this job writes (the
# Slurm log and selection.json are already there: submit_run.py created it).
RUN_DIR="$STAGE_ROOT/runs/$RUN_ID"
RERENDER_DIR="$RUN_DIR/rerenders/$RERENDER_LABEL"
JOB_INFO="$RERENDER_DIR/job_info.txt"
SELECTION_FILE="$RERENDER_DIR/selection.json"
mkdir -p "$RERENDER_DIR"
{
  echo "hostname: $(hostname)"
  echo "start_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "slurm_job_id: ${SLURM_JOB_ID:-none}"
  echo "slurm_cpus_per_task: ${SLURM_CPUS_PER_TASK:-unset}"
  echo "slurm_mem_per_node_mb: ${SLURM_MEM_PER_NODE:-unset}"
  for exported_variable in RUN_ID RERENDER_LABEL STAGE_ROOT SESSION GIT_SHA TRAIN_GIT_SHA CODE_DIR \
      ENV_PREFIX RENDER_DOWNSCALE_FACTOR; do
    echo "$exported_variable: ${!exported_variable:-}"
  done
  echo "environment build:"; sed 's/^/  /' "$ENV_PREFIX/.build-complete"
  echo "gpu (name, compute_cap, memory.total, driver_version):"
  nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv,noheader
} > "$JOB_INFO"
cat "$JOB_INFO"
if [ ! -f "$SELECTION_FILE" ]; then
  echo "ERROR: $SELECTION_FILE not found (submit with scripts/slurm/submit_run.py render)" >&2
  exit 1
fi

# ---- (b) environment ---------------------------------------------------------------------
# PATH, CUDA_HOME, LD_LIBRARY_PATH and the viking-data dataparser registration.
# shellcheck disable=SC1091
source "$ENV_PREFIX/viking-activate.sh"
# The render code's src/ goes first on sys.path, ahead of the env's editable install
# (PYTHONPATH entries precede site-packages .pth paths), so the code is exactly GIT_SHA's.
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
# Caches inside the rerender dir, so nothing depends on (or writes to) $HOME.
mkdir -p "$RERENDER_DIR/.cache/torch"
export TORCH_HOME="$RERENDER_DIR/.cache/torch" XDG_CACHE_HOME="$RERENDER_DIR/.cache"

# ---- (c) GPU sanity check: CUDA visible, CUDA extensions load, code comes from GIT_SHA ----
python -c "import torch, gsplat.csrc, tinycudann, viking_splatad; \
assert torch.cuda.is_available(), 'CUDA not available'; \
print('gpu_check:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)); \
print('viking_splatad_from:', viking_splatad.__file__)" | tee -a "$JOB_INFO"

# ---- (d) render config: the run's config.yml with the guard selection (T66) --------------------
# Same path as splatad_run.sbatch's TRAIN_CONFIG (ns-train --experiment-name/--timestamp).
TRAIN_CONFIG="$RUN_DIR/train/$SESSION/splatad/$RUN_ID/config.yml"
RENDER_CONFIG="$RERENDER_DIR/config.yml"
if [ ! -f "$TRAIN_CONFIG" ]; then
  echo "ERROR: training config not found at $TRAIN_CONFIG" >&2
  exit 1
fi
python "$CODE_DIR/scripts/12_prepare_rerender.py" write-config \
  --train-config "$TRAIN_CONFIG" --selection "$SELECTION_FILE" --output "$RENDER_CONFIG"
echo "render_config: $RENDER_CONFIG" >> "$JOB_INFO"

# ---- (e) render camera images ----------------------------------------------------------------
# --pose-source test: the datamanager asks the dataparser for split "test", which the selection
# replaces by the guard frames (T66); train is unchanged, so the time origin and w2m stay the run's.
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
  --load-config "$RENDER_CONFIG" \
  --output-path "$RERENDER_DIR/renders" \
  --pose-source test \
  --rendered-output-names rgb gt-rgb \
  --image-format jpeg \
  ${RENDER_DOWNSCALE_ARGS[@]+"${RENDER_DOWNSCALE_ARGS[@]}"}  # empty-array-safe under set -u (bash < 4.4)
RENDER_END_SECONDS=$(date +%s)
{
  echo "render_end_epoch_s: $RENDER_END_SECONDS"
  echo "render_elapsed_s: $((RENDER_END_SECONDS - RENDER_START_SECONDS))"
} >> "$JOB_INFO"

# ---- (f) render lidar ------------------------------------------------------------------------
# Same script as the run (SplatAD's get_lidar_outputs + filter_lidar_pred_and_gt); its eval_setup
# uses test_mode="test", so it renders the same selected guard sweeps. It writes
# dataparser_transforms.json next to the config it loads, i.e. into the rerender dir.
LIDAR_RENDER_START_SECONDS=$(date +%s)
echo "lidar_render_start_epoch_s: $LIDAR_RENDER_START_SECONDS" >> "$JOB_INFO"
python "$CODE_DIR/scripts/08_render_lidar_pointcloud.py" \
  --load-config "$RENDER_CONFIG" \
  --output-path "$RERENDER_DIR/renders" \
  --pose-source test \
  --dataset-dir "$STAGE_ROOT/data/$SESSION"
LIDAR_RENDER_END_SECONDS=$(date +%s)
{
  echo "lidar_render_end_epoch_s: $LIDAR_RENDER_END_SECONDS"
  echo "lidar_render_elapsed_s: $((LIDAR_RENDER_END_SECONDS - LIDAR_RENDER_START_SECONDS))"
} >> "$JOB_INFO"

# ---- (g) render check, then done ---------------------------------------------------------------
# The rendered file names must equal selection.json's files exactly (rgb, gt-rgb and lidar), so a
# job with end_utc rendered exactly the requested guard frames.
if python "$CODE_DIR/scripts/12_prepare_rerender.py" check --rerender-dir "$RERENDER_DIR"; then
  echo "render_check: match" >> "$JOB_INFO"
else
  # $? here is the script's exit code: 1 = renders differ from the selection, 2 = outputs missing.
  RENDER_CHECK_EXIT_CODE=$?
  echo "render_check: FAILED (exit $RENDER_CHECK_EXIT_CODE; reason in the Slurm log)" >> "$JOB_INFO"
  exit 1
fi
echo "end_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$JOB_INFO"
echo "done: run $RUN_ID rerender $RERENDER_LABEL, render $((RENDER_END_SECONDS - RENDER_START_SECONDS)) s, lidar render $((LIDAR_RENDER_END_SECONDS - LIDAR_RENDER_START_SECONDS)) s"
