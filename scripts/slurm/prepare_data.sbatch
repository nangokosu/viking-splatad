#!/bin/bash
# Download and extract pinned session data on a scheduled node, not the submit Mac.
# Arguments: stage root, committed code directory, built environment prefix, session.
# Resource requests belong on sbatch's command line, sized from sinfo.
set -euo pipefail
STAGE_ROOT=${1:?stage root required}
CODE_DIR=${2:?committed code directory required}
ENV_PREFIX=${3:?environment prefix required}
SESSION=${4:?session required}
: "${SLURM_JOB_ID:?submit this script with sbatch}"
test -f "$ENV_PREFIX/.build-complete"
source "$ENV_PREFIX/viking-activate.sh"
export STAGE_ROOT CODE_DIR SESSION
export PYTHONPATH="$CODE_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$STAGE_ROOT/data-prep/cache"
export PYTHONDONTWRITEBYTECODE=1
cd "$CODE_DIR"
python - <<'PY'
"""Prepare a fresh dataset on the workdisk and publish only after validation."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import yaml

stage_root = Path(os.environ["STAGE_ROOT"]).resolve()
code_dir = Path(os.environ["CODE_DIR"]).resolve()
session = os.environ["SESSION"]
if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", session) is None:
    raise ValueError("session must be one safe path component")
config_path = code_dir / "configs/sessions" / f"{session}.yaml"
config = yaml.safe_load(config_path.read_text())
if config["session"]["name"] != session:
    raise ValueError("session config name does not match requested session")
chunk_files = config.get("window", {}).get("chunk_files")
if not isinstance(chunk_files, list) or not chunk_files:
    raise ValueError("window.chunk_files must explicitly pin the download")
bag_name = Path(config["session"]["bag_dir"]).name
chunk_indices = []
for chunk_file in chunk_files:
    match = re.fullmatch(re.escape(bag_name) + r"_(\d+)\.mcap", str(chunk_file))
    if match is None:
        raise ValueError("chunk filename differs from the session bag name")
    chunk_indices.append(int(match.group(1)))
if len(set(chunk_indices)) != len(chunk_indices):
    raise ValueError("duplicate pinned chunk")

preparation_root = stage_root / "data-prep"
preparation_root.mkdir(parents=True, exist_ok=True)
lock_dir = preparation_root / f"{session}.lock"
# Atomic mkdir prevents concurrent writers to the shared raw cache and final dataset.
lock_dir.mkdir()
try:
    destination = stage_root / "data" / session
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing dataset: {destination}")
    workspace = Path(tempfile.mkdtemp(prefix=f"{session}-", dir=preparation_root))
    output_dir = workspace / "dataset"
    for path_key in ("bag_dir", "calibration_dir", "labels_json", "gt_dir"):
        relative_path = Path(config["session"][path_key])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("session paths must be relative and stay within the project")
        config["session"][path_key] = str(stage_root / "source-data" / relative_path)
    config["session"]["output_dir"] = str(output_dir)
    runtime_config = workspace / "session.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"preparation workspace: {workspace}", flush=True)
    # Only the downloader accepts chunk indices here. Extraction reads pinned filenames.
    subprocess.run([
        sys.executable, str(code_dir / "scripts/00_fetch_chunk.py"),
        "--config", str(runtime_config), "--fetch-calibration", "--chunks",
        *map(str, chunk_indices),
    ], check=True)
    subprocess.run([
        sys.executable, str(code_dir / "scripts/03_extract.py"), "--config", str(runtime_config),
    ], check=True)
    subprocess.run([
        sys.executable, str(code_dir / "scripts/06_validate_contract.py"),
        "--dataset", str(output_dir),
    ], check=True)
    reference_path = code_dir / "configs/reference_splits" / f"{session}.json"
    if reference_path.is_file():
        subprocess.run([
            sys.executable, str(code_dir / "scripts/16_check_reference_split.py"),
            "--session", session, "--dataset-dir", str(output_dir),
        ], check=True)
    if config["splatad"].get("add_missing_points") and not (output_dir / "lidar_no_return").is_dir():
        raise ValueError("required lidar_no_return sidecar is absent")
    # This receipt records preparation only, not image tests or CPU preflight approval.
    (output_dir / "preparation.json").write_text(json.dumps({
        "session": session, "chunk_files": chunk_files, "contract": "passed",
        "reference_split": "passed" if reference_path.is_file() else "not_configured",
    }, indent=2) + "\n")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_dir.rename(destination)
    print(f"data preparation complete: {destination}", flush=True)
finally:
    # Failed workspaces/raw downloads remain available for diagnosis and a fresh retry.
    lock_dir.rmdir()
PY
