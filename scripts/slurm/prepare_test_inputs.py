"""Supply actual prerequisites to native cluster tests without changing skip policies."""

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


def ensure_symlink(link_path: Path, target_path: Path) -> None:
    """Link a test input to target_path, rejecting conflicting existing inputs."""
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() and link_path.resolve() == target_path.resolve():
        return
    if link_path.exists() or link_path.is_symlink():
        raise FileExistsError(f"conflicting test input: {link_path}")
    link_path.symlink_to(target_path, target_is_directory=True)


def prepare_test_inputs(code_dir: Path, stage_root: Path, env_prefix: Path) -> None:
    """Fetch real May calibration/metadata and expose upstream source and dataset test inputs.

    Params are workdisk directories for the committed snapshot, shared stage,
    and pinned native environment. Returns None; subprocess failures propagate.
    No bags are extracted and no evaluation label enters training.
    """
    source_root = stage_root / "source-data"
    runtime_root = stage_root / "test-inputs"
    runtime_root.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((code_dir / "configs/sessions/may_short_grass.yaml").read_text())
    # Absolute rebasing is only for runtime input files, never committed settings.
    for path_key in ("bag_dir", "calibration_dir", "labels_json", "output_dir", "gt_dir"):
        relative_path = Path(config["session"][path_key])
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("test fixture config paths must be relative")
        config["session"][path_key] = str(source_root / relative_path)
    runtime_config = runtime_root / "may.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False))
    subprocess.run([
        sys.executable, str(code_dir / "scripts/00_fetch_chunk.py"),
        "--config", str(runtime_config), "--fetch-calibration",
    ], check=True)
    ensure_symlink(code_dir / "data", source_root / "data")
    ensure_symlink(code_dir / "third_party/neurad-studio", env_prefix / "viking-sources/neurad-studio")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path, required=True)
    parser.add_argument("--env-prefix", type=Path, required=True)
    arguments = parser.parse_args()
    prepare_test_inputs(arguments.code_dir, arguments.stage_root, arguments.env_prefix)
