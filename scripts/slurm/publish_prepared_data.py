"""Recover validation of a fully extracted workspace without re-extracting its data."""

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import yaml


def publish_prepared(stage_root: Path, code_dir: Path, session: str, dataset_dir: Path) -> Path:
    """Validate complete pinned output and atomically publish it; return its workdisk destination.

    Only a workspace under data-prep with every pinned chunk recorded complete
    is eligible. Existing published data is never overwritten. This runs no
    conversion and never reuses cached parameters for a different extraction.
    """
    stage_root, code_dir, dataset_dir = stage_root.resolve(), code_dir.resolve(), dataset_dir.resolve()
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", session) is None:
        raise ValueError("session must be one safe path component")
    preparation_root = stage_root / "data-prep"
    if not dataset_dir.is_relative_to(preparation_root):
        raise ValueError("recovery source must be an unpublished data-prep workspace")
    config = yaml.safe_load((code_dir / "configs/sessions" / f"{session}.yaml").read_text())
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    if config["session"]["name"] != session or manifest["session"]["name"] != session:
        raise ValueError("workspace/config session does not match requested session")
    state = json.loads((dataset_dir / "extract_state.json").read_text())
    chunk_files = config["window"]["chunk_files"]
    if (not isinstance(chunk_files, list) or not chunk_files or state.get("session_name") != session
            or state.get("extracted_chunks") != chunk_files):
        raise ValueError("workspace does not record complete extraction of the pinned session chunks")
    radar_files = list((dataset_dir / "sidecar/radar").glob("*.npy"))
    if state.get("radar_frame_count") != len(radar_files):
        raise ValueError("workspace radar sidecar count differs from completed extraction state")
    lock_dir = preparation_root / f"{session}.lock"
    lock_dir.mkdir()
    try:
        destination = stage_root / "data" / session
        if destination.exists():
            raise FileExistsError("published dataset already exists; refusing to overwrite")
        subprocess.run([
            sys.executable, str(code_dir / "scripts/06_validate_contract.py"),
            "--dataset", str(dataset_dir), "--report", str(dataset_dir / "contract_report.md"),
        ], check=True)
        reference_path = code_dir / "configs/reference_splits" / f"{session}.json"
        if reference_path.is_file():
            subprocess.run([
                sys.executable, str(code_dir / "scripts/16_check_reference_split.py"),
                "--session", session, "--dataset-dir", str(dataset_dir),
            ], check=True)
        if config["splatad"].get("add_missing_points") and not (dataset_dir / "lidar_no_return").is_dir():
            raise ValueError("workspace lacks required lidar_no_return sidecar")
        (dataset_dir / "preparation.json").write_text(json.dumps({
            "session": session, "chunk_files": chunk_files, "contract": "passed",
            "reference_split": "passed" if reference_path.is_file() else "not_configured",
            "validation_only_recovery": True, "validator_revision": code_dir.name,
        }, indent=2) + "\n")
        destination.parent.mkdir(parents=True, exist_ok=True)
        dataset_dir.rename(destination)
        print(f"data preparation complete: {destination}", flush=True)
        return destination
    finally:
        lock_dir.rmdir()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("stage-root", "code-dir", "dataset-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--session", required=True)
    arguments = parser.parse_args()
    publish_prepared(arguments.stage_root, arguments.code_dir, arguments.session, arguments.dataset_dir)
