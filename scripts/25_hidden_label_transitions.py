"""Report how GT voxels move from the old test-sweep labels to the session-wide hidden space H.

Compares an old cells archive (runs/<run>/eval/gaussian_occupancy_cells.npz, T78 evidence from the
test sweeps only) with data/gt/<session>/hidden_space.npz, per scored class and the obstacles group:
transition table, cause of (non-)hiddenness, height above the box bottom, seen-above, distance to
the nearest sweep, and object counts. Evaluation diagnostics only; nothing here feeds SplatAD.

Usage:
    uv run python scripts/25_hidden_label_transitions.py --session june_tall_grass --run runs/<run dir>
        [--eval-config configs/eval.yaml] [--output-dir reports/hidden_space/<session>]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from viking_splatad.config import load_session_config
from viking_splatad.eval.hidden_label_transitions import (
    CAUSE_NAMES,
    DISTANCE_BIN_EDGES_M,
    HEIGHT_BIN_NAMES,
    NO_BOX_BIN_NAME,
    OLD_LABEL_NAMES,
    distance_breakdown,
    hidden_cause_breakdown,
    height_layer_breakdown,
    label_transition_table,
    match_cells,
    nearest_horizontal_distance,
    object_bottom_heights,
    object_counts,
    old_gt_cells_not_scored_now,
    old_labels_for_new_cells,
    seen_above_count,
)
from viking_splatad.eval.hidden_geometry import SEEN_COLUMN
from viking_splatad.eval.hidden_settings import parse_hidden_settings
from viking_splatad.eval.run_context import file_sha256, load_ground_truth
from viking_splatad.eval.session_rays import iter_session_sweeps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA = "viking-splatad/hidden-label-transitions/1"
OLD_REPORT_SCHEMA = "viking-splatad/gaussian-occupancy/1"
OLD_CELLS_FILE_NAME = "gaussian_occupancy_cells.npz"
OLD_REPORT_FILE_NAME = "gaussian_occupancy.json"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
SESSION_CONFIG_RELATIVE_DIR = Path("configs") / "sessions"
HIDDEN_SPACE_STEM = "hidden_space"
REPORT_RELATIVE_DIR = Path("reports") / "hidden_space"
OUTPUT_STEM = "label_transitions"
SPLIT_CHECK_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "16_check_reference_split.py"
# The group reported first (a key of configs/eval.yaml class_groups).
HEADLINE_GROUP_NAME = "obstacles"
# Decimal places of shares in the markdown.
SHARE_DECIMALS = 3


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


def relative_to_repo(path: Path) -> str:
    """Repo-relative path text (absolute paths would leak the home directory).

    Args:
        path: A path inside the repo.

    Returns:
        Path relative to the repo root.
    """
    return str(Path(path).resolve().relative_to(PROJECT_ROOT))


def load_archive(path: Path) -> Dict[str, np.ndarray]:
    """Read every array of an npz into memory.

    Args:
        path: npz file.

    Returns:
        {key: array}.
    """
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def check_archives_compatible(old: Dict[str, np.ndarray], new: Dict[str, np.ndarray]) -> None:
    """Refuse two archives on different grids.

    Args:
        old: Old cells archive.
        new: Hidden-space archive.

    Raises:
        ValueError: If voxel_size or grid_origin differ.
    """
    if float(old["voxel_size"]) != float(new["voxel_size"]):
        raise ValueError(f"voxel_size differs: {float(old['voxel_size'])} vs {float(new['voxel_size'])}")
    if not np.array_equal(old["grid_origin"], new["grid_origin"]):
        raise ValueError("grid_origin differs between the old archive and hidden_space.npz")


def sweep_origins(dataset_dir: Path, split_settings: dict) -> np.ndarray:
    """Translations of every recorded sweep's lidar pose (map frame), streamed.

    Args:
        dataset_dir: Extracted dataset directory.
        split_settings: ``session_split_settings`` dict.

    Returns:
        (S,3) float64 lidar origins.
    """
    return np.asarray([sweep.lidar_to_map[:3, 3] for sweep in iter_session_sweeps(dataset_dir, split_settings)])


def class_entries(eval_section: Dict[str, Any]) -> List[Tuple[str, Tuple[int, ...], Tuple[str, ...]]]:
    """Groups first, then scored classes: (name, class ids, member class names).

    Args:
        eval_section: The ``eval`` section of configs/eval.yaml.

    Returns:
        Entries in report order (headline group first).
    """
    settings = parse_hidden_settings(eval_section)
    scored = dict(eval_section["scored_class_ids"])
    name_by_id = {class_id: name for name, class_id in scored.items()}
    entries = []
    # Key False sorts first, so the headline group leads; sorted is stable, so other groups keep config order.
    for group_name, group_ids in sorted(settings.class_groups.items(), key=lambda item: item[0] != HEADLINE_GROUP_NAME):
        entries.append((group_name, tuple(group_ids), tuple(name_by_id[class_id] for class_id in group_ids)))
    for class_name, class_id in scored.items():
        entries.append((class_name, (class_id,), (class_name,)))
    return entries


def analyse_class(
    class_ids: Tuple[int, ...], member_names: Tuple[str, ...], old: Dict[str, np.ndarray], new: Dict[str, np.ndarray],
    match: Tuple[np.ndarray, np.ndarray, np.ndarray], old_labels_for_new: np.ndarray, object_bottom_z: np.ndarray,
    nearest_distance: np.ndarray, seen_new: np.ndarray,
) -> Dict[str, Any]:
    """All breakdowns of one class or group.

    Args:
        class_ids: Class ids of the entry.
        member_names: Scored class names of the entry.
        old: Old archive.
        new: Hidden-space archive.
        match: ``match_cells`` output.
        old_labels_for_new: Today's label index per new cell.
        object_bottom_z: Bottom height per object.
        nearest_distance: Nearest sweep horizontal distance per new cell.
        seen_new: New seen flag per cell.

    Returns:
        JSON-ready dict of the breakdowns.
    """
    old_row_for_new = match[0]
    # Step 1: GT in the new set = grid label in the class ids AND inside the union of the neighbourhoods.
    neighbourhood_union = np.logical_or.reduce([new[f"neighbourhood_{name}"] for name in member_names])
    gt_mask_new = np.isin(new["gt_class_ids"], list(class_ids)) & neighbourhood_union
    hidden_gt_mask = gt_mask_new & new["hidden"]
    # Step 2: old GT = OR of the members' truth.
    old_truth = np.logical_or.reduce([old[f"truth_{name}"] for name in member_names])
    # Step 3: the breakdowns.
    csr = (new["object_cell_offsets"], new["object_cells"], new["object_inside"])
    return {
        "transition": label_transition_table(
            old_labels_for_new, new["hidden"], gt_mask_new, old_gt_cells_not_scored_now(old_truth, old_row_for_new, gt_mask_new)),
        "old_gt_cells": int(np.count_nonzero(old_truth)),
        "cause": hidden_cause_breakdown(new["evidence"], gt_mask_new),
        "height_layer": height_layer_breakdown(
            new["centres"], hidden_gt_mask, float(new["voxel_size"]), *csr, object_bottom_z, new["object_class_ids"], class_ids),
        "seen_above_hidden_gt": seen_above_count(new["cell_indices"], seen_new, hidden_gt_mask),
        "distance": distance_breakdown(nearest_distance, hidden_gt_mask, gt_mask_new),
        "objects": object_counts(*csr, new["object_class_ids"], class_ids, gt_mask_new, hidden_gt_mask),
    }


def format_share(value: Optional[float]) -> str:
    """Share as text ('-' for None).

    Args:
        value: Fraction or None.

    Returns:
        Text.
    """
    return "-" if value is None else f"{value:.{SHARE_DECIMALS}f}"


def render_markdown(report: Dict[str, Any]) -> str:
    """Markdown of the report.

    Args:
        report: The report dict (as written to json).

    Returns:
        Markdown text.
    """
    info = report["hidden_space"]
    # Step 1: session-wide header.
    lines = [
        f"# Hidden-space label transitions: {report['session']}", "",
        f"- range margin: {info['range_margin_m']:.4f} m; scored range: {info['scored_range_m']} m",
        f"- cells: new {info['cell_count']} (hidden {info['hidden_cell_count']}); old {report['old_cell_count']}; "
        f"new only {report['new_only_cell_count']}; old only {report['old_only_cell_count']}",
        f"- GT cells by class (new): {info['gt_cell_count_by_class']}",
        f"- hidden GT cells by class (new): {info['hidden_gt_cell_count_by_class']}",
        f"- old archive: {report['inputs']['old_cells']['path']}", ""]
    # Step 2: one section per class or group: summary line, then one table per breakdown.
    for name, entry in report["classes"].items():
        transition = entry["transition"]
        lines += [f"## {name}", "",
                  f"GT cells in new set {transition['total_gt']}, hidden {transition['total_hidden']} "
                  f"(share {format_share(transition['hidden_share'])}); old GT cells {entry['old_gt_cells']}; "
                  f"old GT cells not GT now: {transition['old_only_gt_cells']}", "",
                  "### Today's label -> H", "", "| old label | hidden | not hidden | total | hidden share |", "|---|---|---|---|---|"]
        for label in OLD_LABEL_NAMES:
            row = transition["rows"][label]
            lines.append(f"| {label} | {row['hidden']} | {row['not_hidden']} | {row['total']} | {format_share(row['hidden_share'])} |")
        lines += ["", "### New evidence of the GT cells (exclusive)", "", "| cause | cells |", "|---|---|"]
        lines += [f"| {cause} | {entry['cause'][cause]} |" for cause in CAUSE_NAMES]
        lines += ["", "### Hidden GT cells by layer above the box bottom", "", "| layer | hidden GT cells |", "|---|---|"]
        lines += [f"| {layer} | {entry['height_layer'][layer]} |" for layer in HEIGHT_BIN_NAMES + (NO_BOX_BIN_NAME,)]
        lines += ["", f"Hidden GT cells with a seen cell above in the same column: {entry['seen_above_hidden_gt']}", "",
                  "### Horizontal distance to the nearest sweep origin", "", "| distance | hidden GT | all GT |", "|---|---|---|"]
        lines += [f"| {bin_name} | {counts['hidden_gt']} | {counts['all_gt']} |" for bin_name, counts in entry["distance"].items()]
        objects = entry["objects"]
        lines += ["", f"Objects: {objects['objects']}; with GT cells inside: {objects['with_gt_inside']}; "
                      f"with hidden GT cells inside: {objects['with_hidden_gt_inside']}", ""]
    return "\n".join(lines)


def parse_arguments(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Command line.

    Args:
        argv: Arguments (None = sys.argv).

    Returns:
        Namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    parser.add_argument("--run", required=True, type=Path, help="run directory with eval/gaussian_occupancy_cells.npz")
    parser.add_argument("--eval-config", type=Path, default=PROJECT_ROOT / DEFAULT_EVAL_CONFIG_RELATIVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=None, help="default: reports/hidden_space/<session>")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Build and write the transition report.

    Args:
        argv: Command-line arguments.

    Returns:
        0 on success.

    Raises:
        ValueError: If the old report schema is wrong or the grids differ.
    """
    arguments = parse_arguments(argv)
    # Step 1: inputs and their checks.
    session_config = load_session_config(PROJECT_ROOT / SESSION_CONFIG_RELATIVE_DIR / f"{arguments.session}.yaml")
    eval_section = yaml.safe_load(arguments.eval_config.read_text())["eval"]
    dataset_dir = PROJECT_ROOT / session_config["session"]["output_dir"]
    gt_dir = PROJECT_ROOT / session_config["session"]["gt_dir"]
    run_dir = arguments.run if arguments.run.is_absolute() else PROJECT_ROOT / arguments.run
    old_cells_path = run_dir / "eval" / OLD_CELLS_FILE_NAME
    old_report = json.loads((run_dir / "eval" / OLD_REPORT_FILE_NAME).read_text())
    if old_report.get("schema") != OLD_REPORT_SCHEMA:
        raise ValueError(f"{old_cells_path.parent}: report schema {old_report.get('schema')!r}, expected {OLD_REPORT_SCHEMA}")
    new_path = gt_dir / f"{HIDDEN_SPACE_STEM}.npz"
    hidden_info = json.loads((gt_dir / f"{HIDDEN_SPACE_STEM}.json").read_text())
    old, new = load_archive(old_cells_path), load_archive(new_path)
    check_archives_compatible(old, new)
    # Step 2: match cells, today's labels, geometry helpers.
    match = match_cells(old["cell_indices"], new["cell_indices"])
    old_labels_for_new = old_labels_for_new_cells(old["evidence"], match[0])
    cuboids, _ = load_ground_truth(gt_dir)
    object_bottom_z = object_bottom_heights(new["object_box_ids"], cuboids)
    split_script = load_script_module(SPLIT_CHECK_SCRIPT_PATH, "check_reference_split")
    origins = sweep_origins(dataset_dir, split_script.session_split_settings(arguments.session))
    nearest_distance = nearest_horizontal_distance(new["centres"], origins)
    seen_new = new["evidence"][:, SEEN_COLUMN].astype(bool)
    # Step 3: per class and group.
    classes = {name: analyse_class(class_ids, members, old, new, match, old_labels_for_new, object_bottom_z, nearest_distance, seen_new)
               for name, class_ids, members in class_entries(eval_section)}
    report = {
        "schema": SCHEMA, "session": arguments.session,
        "hidden_space": {
            "range_margin_m": hidden_info["range_margin"]["margin_m"], "scored_range_m": hidden_info["scored_range_m"],
            "cell_count": hidden_info["cell_count"], "hidden_cell_count": hidden_info["hidden_cell_count"],
            "gt_cell_count_by_class": hidden_info["gt_cell_count_by_class"],
            "hidden_gt_cell_count_by_class": hidden_info["hidden_gt_cell_count_by_class"]},
        "old_cell_count": int(len(old["cell_indices"])), "new_only_cell_count": int(match[1].sum()),
        "old_only_cell_count": int(match[2].sum()), "sweep_count": int(len(origins)),
        "distance_bin_edges_m": [float(edge) if np.isfinite(edge) else "inf" for edge in DISTANCE_BIN_EDGES_M],
        "inputs": {"old_cells": {"path": relative_to_repo(old_cells_path), "sha256": file_sha256(old_cells_path)},
                   "hidden_space": {"path": relative_to_repo(new_path), "sha256": file_sha256(new_path)}},
        "classes": classes}
    # Step 4: write json and markdown.
    output_dir = arguments.output_dir or PROJECT_ROOT / REPORT_RELATIVE_DIR / arguments.session
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{OUTPUT_STEM}.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    (output_dir / f"{OUTPUT_STEM}.md").write_text(render_markdown(report))
    print(f"wrote {relative_to_repo(output_dir / (OUTPUT_STEM + '.md'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
