#!/usr/bin/env bash
# Install the Viking Hill SplatAD integration into a neurad-studio checkout.
# Usage: install_into_neurad.sh [NEURAD_ROOT]   (default: <repo>/third_party/neurad-studio)
# Idempotent. Steps: (1) patch nerfstudio/cameras/lidars.py with OUSTER_OS0_32 (T28);
# (2) print the env var that registers the `viking-data` dataparser (T27) with ns-train.
# Prerequisite: viking_splatad importable in the neurad environment (pip install -e <repo>).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NEURAD_ROOT="${1:-$REPO_ROOT/third_party/neurad-studio}"
test -f "$NEURAD_ROOT/nerfstudio/cameras/lidars.py" || { echo "not a neurad-studio checkout: $NEURAD_ROOT" >&2; exit 1; }

# Run the patcher from source so no install is needed just to patch.
PYTHONPATH="$REPO_ROOT/src" python -m viking_splatad.splatad.patches.apply_ouster_patch "$NEURAD_ROOT"

echo "Register the dataparser with:"
echo "  export NERFSTUDIO_DATAPARSER_CONFIGS=viking-data=viking_splatad.splatad.viking_dataparser:VIKING_DATAPARSER_SPEC"
