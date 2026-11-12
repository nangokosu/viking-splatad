#!/bin/bash
# CPU-only fixed radar anchors; inputs are exported by submit_run.py prepare-radar.
set -euo pipefail
: "${STAGE_ROOT:?}" "${CODE_DIR:?}" "${ENV_PREFIX:?}" "${SESSION:?}" "${PROFILE:?}"
: "${SLURM_JOB_ID:?submit through Slurm}"
test -f "$ENV_PREFIX/.build-complete"
source "$ENV_PREFIX/viking-activate.sh"
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
cd "$CODE_DIR"
exec python scripts/07_build_radar_seeds.py --dataset "$STAGE_ROOT/data/$SESSION" \
  --session "$SESSION" --profile "$PROFILE" \
  --out "$STAGE_ROOT/radar-inputs/$SESSION/$PROFILE/$(basename "$CODE_DIR")"
