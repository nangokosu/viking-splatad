#!/bin/bash
# Validate Python/data changes on CPU using an already GPU-verified recipe env.
# This job never writes or replaces environment readiness markers.
set -euo pipefail
: "${STAGE_ROOT:?}" "${CODE_DIR:?}" "${ENV_NAME:?}" "${GIT_SHA:?}" "${SLURM_JOB_ID:?}"
ENV_PREFIX="$STAGE_ROOT/envs/$ENV_NAME"
test -f "$ENV_PREFIX/.build-complete"
source "$ENV_PREFIX/viking-activate.sh"
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$STAGE_ROOT/test-inputs/cache"
cd "$CODE_DIR"
python "$CODE_DIR/scripts/slurm/prepare_test_inputs.py" \
  --code-dir "$CODE_DIR" --stage-root "$STAGE_ROOT" --env-prefix "$ENV_PREFIX"
python -m pip check
python -m pytest -q -r a -p no:cacheprovider
echo "== native CPU validation complete: $GIT_SHA"
