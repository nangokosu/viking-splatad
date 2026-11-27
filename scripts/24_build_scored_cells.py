"""Build the scored cells of one session (evaluation only; never a SplatAD input).

A box is scored when its centre is within the scored range and vertical field of view of some
recorded Ouster sweep (train, guard or test); the scored cells are the boxes grown by the class
margin, restricted to the cells visible from the sweep poses. Writes next to the GT:
  <gt_dir>/scored_cells.npz   per-cell arrays, per-class neighbourhoods, object CSR
  <gt_dir>/scored_cells.json  provenance: settings, counts, input and output hashes

The rebuilt split must equal the committed reference record (scripts/16) or the build is refused.

Usage:
    uv run python scripts/24_build_scored_cells.py --session june_tall_grass
        [--eval-config configs/eval.yaml] [--output-dir DIR] [--max-sweeps N]
"""

import argparse
import importlib.util
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

import numpy as np
import yaml

from viking_splatad.config import load_session_config
from viking_splatad.eval.detection import vertical_half_fov_from_elevations
from viking_splatad.eval.occupancy_settings import parse_occupancy_settings
from viking_splatad.eval.run_context import (
    CUBOIDS_FILE_NAME,
    MANIFEST_FILE_NAME,
    OCCUPANCY_FILE_NAME,
    file_sha256,
    load_ground_truth,
)
from viking_splatad.eval.scene_metrics import OBJECT_ARRAY_KEYS
from viking_splatad.eval.scored_cells import build_scored_cells
from viking_splatad.eval.session_rays import count_session_sweeps, iter_session_sweeps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "viking-splatad/scored-cells/1"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
SESSION_CONFIG_RELATIVE_DIR = Path("configs") / "sessions"
REFERENCE_SPLIT_RELATIVE_DIR = Path("configs") / "reference_splits"
SPLIT_CHECK_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "16_check_reference_split.py"
# Output stems; the debug stem can never be mistaken for the real file.
OUTPUT_STEM = "scored_cells"
DEBUG_OUTPUT_STEM = "scored_cells_debug"
# Progress is printed roughly this many times per build.
PROGRESS_PRINT_COUNT = 20


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
                        help="DEBUG: only the first N sweeps; output is named scored_cells_debug.*")
    return parser.parse_args(argv)


def counted_sweeps(sweeps: Iterable[Any], total_sweeps: int) -> Iterator[Any]:
    """Pass sweeps through, printing the count about PROGRESS_PRINT_COUNT times.

    Args:
        sweeps: Lazy sweep iterable.
        total_sweeps: Expected number of sweeps (for the print interval and the final line).

    Yields:
        Every sweep unchanged.
    """
    progress_step = max(1, total_sweeps // PROGRESS_PRINT_COUNT)
    for sweep_count, sweep in enumerate(sweeps, start=1):
        yield sweep
        # Print on every progress_step-th sweep and on the last one.
        if sweep_count % progress_step == 0 or sweep_count == total_sweeps:
            print(f"sweeps {sweep_count}/{total_sweeps}", flush=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build and write the scored cells.

    Args:
        argv: Command-line arguments.

    Returns:
        0 on success, 1 when the split differs from the reference record.
    """
    arguments = parse_arguments(argv)
    # Step 1: session, settings, dataset and GT.
    session_config = load_session_config(PROJECT_ROOT / SESSION_CONFIG_RELATIVE_DIR / f"{arguments.session}.yaml")
    eval_section = yaml.safe_load(arguments.eval_config.read_text())["eval"]
    settings = parse_occupancy_settings(eval_section)
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
    # Step 3: sweeps (all, or the first N for debugging), streamed once into the build.
    total_sweeps = count_session_sweeps(dataset_dir)
    sweeps = iter_session_sweeps(dataset_dir, split_settings)
    if arguments.max_sweeps is not None:
        total_sweeps = min(total_sweeps, arguments.max_sweeps)
        sweeps = itertools.islice(sweeps, arguments.max_sweeps)
    scored_class_ids = dict(eval_section["scored_class_ids"])
    result = build_scored_cells(
        counted_sweeps(sweeps, total_sweeps), cuboids, scored_class_ids, grid, vertical_half_fov,
        settings.scored_range_quantile)
    # Step 4: write the npz.
    stem = DEBUG_OUTPUT_STEM if arguments.max_sweeps is not None else OUTPUT_STEM
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = output_dir / f"{stem}.npz", output_dir / f"{stem}.json"
    arrays = {
        "cell_indices": result["cell_indices"], "centres": result["centres"],
        "gt_class_ids": result["gt_class_ids"], "voxel_size": np.float64(grid["voxel_size"]),
        "grid_origin": np.asarray(grid["origin"], dtype=np.float64),
    }
    arrays.update({array_key: result[array_key] for array_key in OBJECT_ARRAY_KEYS})
    for class_name, membership in result["neighbourhood_by_class"].items():
        arrays[f"neighbourhood_{class_name}"] = membership
    np.savez_compressed(npz_path, **arrays)
    # Step 5: the provenance record. max_range_m is the farthest single return (a stray), not the scoring range.
    record = {
        "schema": SCHEMA, "session": arguments.session, "debug_max_sweeps": arguments.max_sweeps,
        "scored_range_m": result["scored_range_m"], "scored_range_quantile": settings.scored_range_quantile,
        "max_range_m": result["scan"].max_range_m, "vertical_half_fov_rad": vertical_half_fov,
        "sweep_count_by_split": result["scan"].sweep_count_by_split,
        "cell_count": int(len(result["cell_indices"])),
        "input_sha256": {
            relative_to_repo(path): sha256_or_none(path) for path in (
                gt_dir / CUBOIDS_FILE_NAME, gt_dir / OCCUPANCY_FILE_NAME,
                PROJECT_ROOT / REFERENCE_SPLIT_RELATIVE_DIR / f"{arguments.session}.json")},
        "output_npz": {"path": relative_to_repo(npz_path), "sha256": file_sha256(npz_path)},
    }
    json_path.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    print(f"wrote {relative_to_repo(npz_path)} and {relative_to_repo(json_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
