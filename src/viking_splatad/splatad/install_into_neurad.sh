#!/usr/bin/env bash
# Install the Viking Hill SplatAD integration into a neurad-studio checkout.
# Usage: install_into_neurad.sh [NEURAD_ROOT]   (default: <repo>/third_party/neurad-studio)
# Idempotent. Steps: (1) patch nerfstudio/cameras/lidars.py with OUSTER_OS0_32 (T28);
# (2) print the env var that registers the `viking-data` dataparser (T27) with ns-train.
# Prerequisite: viking_splatad importable in the neurad environment (pip install -e <repo>).
set -euo pipefail

# Resolve paths. This script lives at <repo>/src/viking_splatad/splatad/, so the repo root
# is three directories up (the depth is fixed by the file's location in the package).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NEURAD_ROOT="${1:-$REPO_ROOT/third_party/neurad-studio}"

# Refuse to run unless the target holds the file the patch edits.
test -f "$NEURAD_ROOT/nerfstudio/cameras/lidars.py" || { echo "not a neurad-studio checkout: $NEURAD_ROOT" >&2; exit 1; }

# Step 1: apply the Ouster lidar patch, run from source so no install is needed just to patch.
PYTHONPATH="$REPO_ROOT/src" python -m viking_splatad.splatad.patches.apply_ouster_patch "$NEURAD_ROOT"

# Step 2: print how to register the dataparser with ns-train.
echo "Register the dataparser with:"
echo "  export NERFSTUDIO_DATAPARSER_CONFIGS=viking-data=viking_splatad.splatad.viking_dataparser:VIKING_DATAPARSER_SPEC"
