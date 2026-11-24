"""Build the session-wide hidden space H of one session (evaluation only; never a SplatAD input).

H = cells of the scored region that no recorded Ouster sweep (train, guard or test) could see:
not seen, blocked by a nearer return, never passed through, with a range margin m measured from
the data (cross-sweep point-to-plane residuals, T92). Writes next to the GT:
  <gt_dir>/hidden_space.npz   per-cell arrays, evidence, H, per-class neighbourhoods, object CSR
  <gt_dir>/hidden_space.json  provenance: settings, margin, counts, input and output hashes, git state

The rebuilt split must equal the committed reference record (scripts/16) or the build is refused.

Usage:
    uv run python scripts/24_build_hidden_space.py --session june_tall_grass
        [--eval-config configs/eval.yaml] [--output-dir DIR] [--max-sweeps N]
"""

import argparse
import importlib.util
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import yaml

from viking_splatad.config import load_session_config
from viking_splatad.eval.detection import vertical_half_fov_from_elevations
from viking_splatad.eval.hidden_settings import parse_hidden_settings
from viking_splatad.eval.run_context import (
    CUBOIDS_FILE_NAME,
    MANIFEST_FILE_NAME,
    OCCUPANCY_FILE_NAME,
    file_sha256,
    load_ground_truth,
)
from viking_splatad.eval.session_hidden_space import build_hidden_space, estimate_range_margin
from viking_splatad.eval.session_rays import count_session_sweeps, iter_session_sweeps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "viking-splatad/hidden-space/1"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
SESSION_CONFIG_RELATIVE_DIR = Path("configs") / "sessions"
REFERENCE_SPLIT_RELATIVE_DIR = Path("configs") / "reference_splits"
POSES_FILE_NAME = "poses.npz"
SPLIT_CHECK_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "16_check_reference_split.py"
# Output stems; the debug stem can never be mistaken for the real file.
OUTPUT_STEM = "hidden_space"
DEBUG_OUTPUT_STEM = "hidden_space_debug"
# Progress is printed roughly this many times per build.
PROGRESS_PRINT_COUNT = 20
# Git invocations (read-only).
GIT_HEAD_COMMAND = ["git", "rev-parse", "HEAD"]
GIT_STATUS_COMMAND = ["git", "status", "--porcelain", "--untracked-files=no"]


def load_script_module(script_path: Path, module_name: str):
    """Import a script whose file name is not a valid module name.

    Args:
        script_path: Path of the script.
        module_name: Name to register.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def sha256_or_none(path: Path) -> Optional[str]:
    """sha256 of a file, or None when it does not exist.

    Args:
        path: File path.

    Returns:
        Hex digest or None.
    """
    return file_sha256(path) if path.is_file() else None


def git_state() -> Dict[str, Any]:
    """HEAD sha and whether tracked files are modified.

    Returns:
        {"head_sha", "tree_dirty"}.
    """
    head_sha = subprocess.run(GIT_HEAD_COMMAND, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    status_text = subprocess.run(GIT_STATUS_COMMAND, cwd=PROJECT_ROOT, capture_output=True, text=True, check=True).stdout
    return {"head_sha": head_sha, "tree_dirty": bool(status_text.strip())}


def relative_to_repo(path: Path) -> str:
    """Repo-relative path text (absolute paths would leak the home directory).

    Args:
        path: A path inside the repo.

    Returns:
        Path relative to the repo root.
    """
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def parse_arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Command line.

    Args:
        argv: Arguments (None = sys.argv).

    Returns:
        Namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True, help="session name (configs/sessions/<session>.yaml)")
    parser.add_argument("--eval-config", type=Path, default=PROJECT_ROOT / DEFAULT_EVAL_CONFIG_RELATIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: the session's gt_dir")
    parser.add_argument("--max-sweeps", type=int, default=None,
                        help="DEBUG: only the first N sweeps; output is named hidden_space_debug.*")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build and write the hidden space.

    Args:
        argv: Command-line arguments.

    Returns:
        0 on success, 1 when the split differs from the reference record.
    """
    arguments = parse_arguments(argv)
    started_at = time.monotonic()
    # Step 1: session, settings, dataset and GT.
    session_config = load_session_config(PROJECT_ROOT / SESSION_CONFIG_RELATIVE_DIR / f"{arguments.session}.yaml")
    eval_section = yaml.safe_load(arguments.eval_config.read_text())["eval"]
    settings = parse_hidden_settings(eval_section)
    dataset_dir = PROJECT_ROOT / session_config["session"]["output_dir"]
    gt_dir = PROJECT_ROOT / session_config["session"]["gt_dir"]
    output_dir = arguments.output_dir or gt_dir
    # Step 2: refuse a dataset whose split differs from the committed record (script 16's own check).
    split_script = load_script_module(SPLIT_CHECK_SCRIPT_PATH, "check_reference_split")
    if split_script.check_session(arguments.session, dataset_dir) != split_script.EXIT_IDENTICAL:
        print("refusing: the split does not match the committed reference record", file=sys.stderr)
        return 1
    split_settings = split_script.session_split_settings(arguments.session)
    manifest = json.loads((dataset_dir / MANIFEST_FILE_NAME).read_text())
    cuboids, grid = load_ground_truth(gt_dir)
    vertical_half_fov = vertical_half_fov_from_elevations(manifest["lidar"]["elevation_mapping"])
    # Step 3: range margin from the data.
    margin = estimate_range_margin(dataset_dir, split_settings, settings)
    print(f"range margin {margin['margin_m']:.4f} m (quantile {margin['quantile']}, "
          f"{margin['residual_count']} residuals, {margin['pair_count']} pairs)", flush=True)
    # Step 4: sweeps (all, or the first N for debugging) and the build.
    total_sweeps = count_session_sweeps(dataset_dir)
    if arguments.max_sweeps is not None:
        total_sweeps = min(total_sweeps, arguments.max_sweeps)

    def sweeps_factory():
        """Fresh lazy iterable of the session's sweeps (limited when debugging).

        Returns:
            Iterator of SessionSweep, the first --max-sweeps only when that flag is set.
        """
        sweeps = iter_session_sweeps(dataset_dir, split_settings)
        return sweeps if arguments.max_sweeps is None else itertools.islice(sweeps, arguments.max_sweeps)

    progress_step = max(1, total_sweeps // PROGRESS_PRINT_COUNT)
    last_printed = {"count": 0}

    def report_progress(finished_sweep_count: int) -> None:
        """Print sweeps done / total about PROGRESS_PRINT_COUNT times.

        Args:
            finished_sweep_count: Sweeps whose evidence has been accumulated so far.
        """
        if finished_sweep_count - last_printed["count"] >= progress_step or finished_sweep_count == total_sweeps:
            last_printed["count"] = finished_sweep_count
            print(f"sweeps {finished_sweep_count}/{total_sweeps}  elapsed {time.monotonic() - started_at:.0f} s", flush=True)

    scored_class_ids = dict(eval_section["scored_class_ids"])
    result = build_hidden_space(
        sweeps_factory, cuboids, scored_class_ids, grid, vertical_half_fov, margin["margin_m"],
        settings.worker_count, settings.scored_range_quantile, settings.near_band_m, report_progress)
    # Step 5: write the npz.
    stem = DEBUG_OUTPUT_STEM if arguments.max_sweeps is not None else OUTPUT_STEM
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = output_dir / f"{stem}.npz", output_dir / f"{stem}.json"
    arrays = {
        "cell_indices": result["cell_indices"], "centres": result["centres"], "evidence": result["evidence"],
        "hidden": result["hidden"], "hidden_near": result["hidden_near"],
        "nearest_sweep_distance_m": result["nearest_sweep_distance_m"], "gt_class_ids": result["gt_class_ids"],
        "object_box_ids": result["object_box_ids"], "object_class_ids": result["object_class_ids"],
        "object_cell_offsets": result["object_cell_offsets"], "object_cells": result["object_cells"],
        "object_inside": result["object_inside"], "voxel_size": np.float64(grid["voxel_size"]),
        "grid_origin": np.asarray(grid["origin"], dtype=np.float64), "range_margin_m": np.float64(margin["margin_m"]),
    }
    for class_name, membership in result["neighbourhood_by_class"].items():
        arrays[f"neighbourhood_{class_name}"] = membership
    np.savez_compressed(npz_path, **arrays)
    # Step 6: the provenance record. Hidden GT cells per class = cells labelled with the class id in H.
    hidden, gt_class_ids = result["hidden"], result["gt_class_ids"]
    record = {
        "schema": SCHEMA, "session": arguments.session, "debug_max_sweeps": arguments.max_sweeps,
        "range_margin": margin,
        "hidden_space_settings": {key: value for key, value in vars(settings).items()
                                  if key not in ("class_groups", "p2_thresholds")},
        "max_range_m": result["scan"].max_range_m, "scored_range_m": result["scored_range_m"],
        "vertical_half_fov_rad": vertical_half_fov,
        "sweep_count_by_split": result["scan"].sweep_count_by_split,
        "scored_cuboid_count_by_class": {name: len(entry["cuboids"]) for name, entry in result["neighbourhoods"].items()},
        "cell_count": int(len(hidden)), "hidden_cell_count": int(np.count_nonzero(hidden)),
        "near_band_m": settings.near_band_m,
        "hidden_near_cell_count": int(np.count_nonzero(result["hidden_near"])),
        "hidden_near_gt_cell_count_by_class": {
            name: int(np.count_nonzero(result["hidden_near"] & (gt_class_ids == class_id)))
            for name, class_id in scored_class_ids.items()},
        "hidden_gt_cell_count_by_class": {
            name: int(np.count_nonzero(hidden & (gt_class_ids == class_id))) for name, class_id in scored_class_ids.items()},
        "gt_cell_count_by_class": {
            name: int(np.count_nonzero(gt_class_ids == class_id)) for name, class_id in scored_class_ids.items()},
        "input_sha256": {
            relative_to_repo(path): sha256_or_none(path) for path in (
                dataset_dir / MANIFEST_FILE_NAME, dataset_dir / POSES_FILE_NAME, gt_dir / CUBOIDS_FILE_NAME,
                gt_dir / OCCUPANCY_FILE_NAME,
                PROJECT_ROOT / REFERENCE_SPLIT_RELATIVE_DIR / f"{arguments.session}.json")},
        "output_npz": {"path": relative_to_repo(npz_path), "sha256": file_sha256(npz_path)},
        "git": git_state(), "elapsed_s": round(time.monotonic() - started_at, 1),
    }
    json_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {relative_to_repo(npz_path)} and {relative_to_repo(json_path)} in {record['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
