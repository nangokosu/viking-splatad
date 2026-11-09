#!/bin/bash
# Revalidate an already compiled recipe environment; do not reinstall or recompile.
# Inputs are exported inside the generated job script; verification needs an sm_86 GPU.
set -euo pipefail
: "${STAGE_ROOT:?}" "${CODE_DIR:?}" "${ENV_NAME:?}" "${GIT_SHA:?}" "${SLURM_JOB_ID:?}"
ENV_PREFIX="$STAGE_ROOT/envs/$ENV_NAME"
LOCK_DIR="$STAGE_ROOT/envs/$ENV_NAME.verify-lock"
mkdir "$LOCK_DIR"
trap 'rmdir "$LOCK_DIR"' EXIT
source "$ENV_PREFIX/viking-activate.sh"
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$STAGE_ROOT/test-inputs/cache"
export PIP_DISABLE_PIP_VERSION_CHECK=1
cd "$CODE_DIR"
python "$CODE_DIR/scripts/slurm/prepare_test_inputs.py" \
  --code-dir "$CODE_DIR" --stage-root "$STAGE_ROOT" --env-prefix "$ENV_PREFIX"
python -m pip check
ns-train splatad --help > /dev/null
ns-train splatad viking-data --help > /dev/null
python -c "import torch, gsplat.csrc, tinycudann; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability(0) == (8, 6); print('GPU import check OK:', torch.cuda.get_device_name(0))"
# The exclusion audit remains active: missing prerequisites are failures, never allowlisted here.
python -m pytest -q -r a -p no:cacheprovider
# Publish readiness only after all checks above pass; keep lock records next to the marker.
python -m pip freeze --exclude-editable > "$ENV_PREFIX/viking-pip-freeze.txt"
MICROMAMBA="$STAGE_ROOT/bin/micromamba-$MICROMAMBA_VERSION"
"$MICROMAMBA" env export --explicit -p "$ENV_PREFIX" > "$ENV_PREFIX/viking-conda-explicit.txt"
mkdir -p "$STAGE_ROOT/envs/$ENV_NAME-locks"
cp "$ENV_PREFIX/viking-pip-freeze.txt" "$ENV_PREFIX/viking-conda-explicit.txt" "$STAGE_ROOT/envs/$ENV_NAME-locks/"
printf 'verified %s job %s code %s\n' "$(date -u +%FT%TZ)" "$SLURM_JOB_ID" "$GIT_SHA" > "$ENV_PREFIX/.build-complete"
printf 'env_prefix=%s\nbuilt_utc=%s\nbuild_job=%s\nverified_git_sha=%s\n' \
  "$ENV_PREFIX" "$(date -u +%FT%TZ)" "$SLURM_JOB_ID" "$GIT_SHA" > "$STAGE_ROOT/envs/$ENV_NAME.marker.partial"
mv "$STAGE_ROOT/envs/$ENV_NAME.marker.partial" "$STAGE_ROOT/envs/$ENV_NAME.marker"
echo "== environment verification complete"
