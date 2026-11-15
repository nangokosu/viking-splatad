#!/bin/bash
# Finish validation of completely extracted data after validator memory exhaustion.
# Arguments: stage root, code snapshot, verified env prefix, session, existing dataset workspace.
set -euo pipefail
STAGE_ROOT=${1:?}
CODE_DIR=${2:?}
ENV_PREFIX=${3:?}
SESSION=${4:?}
DATASET_DIR=${5:?}
: "${SLURM_JOB_ID:?submit through sbatch}"
test -f "$ENV_PREFIX/.build-complete"
source "$ENV_PREFIX/viking-activate.sh"
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$STAGE_ROOT/data-prep/cache"
exec python "$CODE_DIR/scripts/slurm/publish_prepared_data.py" \
  --stage-root "$STAGE_ROOT" --code-dir "$CODE_DIR" --session "$SESSION" --dataset-dir "$DATASET_DIR"
