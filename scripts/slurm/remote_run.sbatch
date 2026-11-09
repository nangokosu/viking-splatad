#!/bin/bash
# Validate or train using workdisk data in the pinned environment; no local Docker/data needed.
set -euo pipefail
: "${STAGE_ROOT:?}" "${CODE_DIR:?}" "${ENV_PREFIX:?}" "${SESSION:?}" "${PROFILE:?}" "${GIT_SHA:?}" "${RUN_ID:?}" "${SLURM_JOB_ID:?}"
test -f "$ENV_PREFIX/.build-complete"
source "$ENV_PREFIX/viking-activate.sh"
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$STAGE_ROOT/test-inputs/cache"
MODE_ARGS=()
if [ "${PREFLIGHT_ONLY:-False}" = True ]; then
  MODE_ARGS=(--preflight-only)
fi
echo "remote job starting: $SLURM_JOB_ID profile=$PROFILE preflight=${PREFLIGHT_ONLY:-False}"
cd "$CODE_DIR"
exec python "$CODE_DIR/scripts/slurm/remote_run.py" \
  --stage-root "$STAGE_ROOT" --code-dir "$CODE_DIR" --env-prefix "$ENV_PREFIX" \
  --session "$SESSION" --profile "$PROFILE" --git-sha "$GIT_SHA" --run-id "$RUN_ID" \
  ${MODE_ARGS[@]+"${MODE_ARGS[@]}"}
