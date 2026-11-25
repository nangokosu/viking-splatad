"""P2 evaluation of one or more runs: occupancy from the trained Gaussians on the GT grid (PLAN §5).

P1 (scripts/13) asks what a lidar at the eval poses would see; P2 asks what the model contains. A
trunk kept behind grass Gaussians is invisible in P1 and shows up only here, and geometry the
radar invented shows up here as false positives, so precision matters as much as recall.

Per run:
1. the session, split, GT and test sweeps from the run's own records (as scripts/13);
2. the scored cells: the session hidden space's cells (scripts/24: per scored class, the session-wide
   scored region grown by the class's margin, with the hidden flag H and the per-object cell lists);
   the run's own test-sweep evidence (T78) and P1 hits are computed on those cells;
3. the factorized checkpoint Gaussians (T56), exported inside the reference image (torch) into a
   temporary folder that is deleted afterwards;
4. T81 peak-alpha occupancy on every scored cell (each Gaussian's sigma(o) in the voxel holding its
   mean, T57 3-sigma tails elsewhere); occupied = probability >= the P1 threshold. T57's centre sample
   alone misses SplatAD's thin Gaussians at the GT voxel size (docs/research/p2_estimator_diagnostic/);
5. per class (group ``obstacles`` first, then each class; cells on another annotated class's GT are
   removed from the class's neighbourhood): P1 and P2 voxel P/R/IoU with recall per T78 evidence
   stratum, the T79 detection curve over f_min, T80 present-but-hidden, a threshold sweep, and the
   hidden block (T94/T95: recall_H, precision_H, IoU_H pooled and per object, at each P2 threshold,
   beside the whole-neighbourhood baseline and the P1 numbers);
6. ``<run>/eval/gaussian_occupancy.{json,md}`` plus ``gaussian_occupancy_cells.npz`` (per-cell
   arrays for scripts/20 and scripts/15), then the P2 and hidden-P2 pictures with scripts/15
   (``--no-viz`` skips).

Usage:
    uv run python scripts/19_eval_gaussian_occupancy.py runs/<run_id> [runs/<run_id> ...] [--no-viz]
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from viking_splatad.config import find_project_root
from viking_splatad.eval.detection import class_margin_table
from viking_splatad.eval.gaussian_detection import (
    cuboid_occupied_fractions,
    detection_curve,
    f_min_grid,
    present_but_hidden,
    threshold_sweep,
)
from viking_splatad.eval.hidden_geometry import occupied_cell_metrics, peak_alpha_voxel_occupancy
# select_object_lists, threshold_key and hidden_class_report live in src/ so scripts/20 recomputes the same blocks;
# the names stay importable from this module.
from viking_splatad.eval.hidden_metrics import (  # noqa: F401
    class_group_masks,
    OBJECT_ARRAY_KEYS,
    hidden_class_report,
    select_object_lists,
    threshold_key,
)
from viking_splatad.eval.hidden_settings import HiddenEvalSettings, class_group_names_ordered, parse_hidden_settings
from viking_splatad.eval.p2_cells import (
    HIDDEN_BAND_NEAR,
    HIDDEN_KEY,
    HIDDEN_NEAR_KEY,
    P2_THRESHOLDS_KEY,
    accumulate_recorded_evidence,
    factor_export_command,
    load_factors,
    p1_hit_cells,
)
from viking_splatad.eval.rendered_pointcloud import load_test_lidar_sweeps
from viking_splatad.eval.run_context import (
    CUBOIDS_FILE_NAME,
    OCCUPANCY_FILE_NAME,
    file_sha256,
    load_ground_truth,
    resolve_render_target,
    resolve_session,
)
from viking_splatad.viz.pyvista_scene import (
    VIEWER_PROJECT_RELATIVE_DIR,
    VIEWER_SCRIPT_RELATIVE_PATH,
    VIZ_ENVIRONMENT_FIX_COMMAND,
    repo_relative,
    run_viewer_command,
)

REPORT_SCHEMA = "viking-splatad/gaussian-occupancy/2"
# Recorded in every report so scripts/20 refuses to pool runs scored by different estimators.
OCCUPANCY_ESTIMATOR = "T81 peak alpha in voxel"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
EVAL_OUTPUT_DIR_NAME = "eval"
REPORT_JSON_NAME = "gaussian_occupancy.json"
REPORT_MARKDOWN_NAME = "gaussian_occupancy.md"
CELLS_ARCHIVE_NAME = "gaussian_occupancy_cells.npz"
VIZ_OUTPUT_DIR_NAME = "viz_p2"
# scripts/24 outputs next to the GT, and the record fields this script checks (names as scripts/24 writes them).
HIDDEN_SPACE_NPZ_NAME = "hidden_space.npz"
HIDDEN_SPACE_JSON_NAME = "hidden_space.json"
HIDDEN_SPACE_SCHEMA = "viking-splatad/hidden-space/1"
HIDDEN_RECORD_INPUT_SHA_KEY = "input_sha256"
HIDDEN_RECORD_OUTPUT_KEY = "output_npz"
HIDDEN_RECORD_DEBUG_KEY = "debug_max_sweeps"
# Near-band fields scripts/24 writes: the archive's H_near flag and the record's band width (m).
HIDDEN_NEAR_ARRAY_KEY = HIDDEN_NEAR_KEY
HIDDEN_RECORD_NEAR_BAND_KEY = "near_band_m"
# Report keys of the two hidden blocks of a class: H_near (headline) and full H.
REPORT_HIDDEN_NEAR_KEY, REPORT_HIDDEN_KEY = HIDDEN_NEAR_KEY, HIDDEN_KEY
REFERENCE_SPLIT_RELATIVE_DIR = Path("configs") / "reference_splits"
HIDDEN_SPACE_BUILD_COMMAND = "uv run python scripts/24_build_hidden_space.py --session <session>"
# The headline group, drawn by the hidden picture (a key of configs/eval.yaml class_groups).
HEADLINE_GROUP_NAME = "obstacles"
# scripts/15 flags of the two pictures.
P2_VIZ_ARGUMENTS = ("--p2",)
P2_HIDDEN_VIZ_FLAG = "--p2-hidden"
# scripts/15 --hidden-band value of the launched hidden picture: the headline H_near (the full-H picture stays a manual flag).
HIDDEN_VIZ_BAND = HIDDEN_BAND_NEAR
VIZ_KEY_P2 = "p2"
VIZ_KEY_P2_HIDDEN = "p2_hidden"
# Metric-name suffix of the H_near rows and the full-H rows (as scripts/20 flattens them).
HIDDEN_NEAR_METRIC_SUFFIX, HIDDEN_FULL_METRIC_SUFFIX = "_Hnear", "_H"
# Metrics of the markdown tables: (row label, metric name in the T94 pooled dict and the T95 ``mean_<name>``).
HIDDEN_TABLE_METRICS = (("recall", "recall"), ("precision", "precision"), ("IoU", "iou"))
JOB_INFO_NAME = "job_info.txt"
RUN_LEDGER_RELATIVE_PATH = Path("runs") / "submitted.tsv"
FACTORS_ARCHIVE_NAME = "gaussian_factors.npz"
# Same default and override variable as scripts/test_in_image.sh, so both use one image.
IMAGE_NAME_ENVIRONMENT_VARIABLE = "VIKING_IMAGE"
DEFAULT_IMAGE_NAME = "viking-splatad:cu118"
# job_info.txt markers of a complete run (the same ones scripts/17 requires).
COMPLETE_RUN_SPLIT_CHECK = "match"
# The final checkpoint of a run is step MAX_NUM_ITERATIONS - 1 (nerfstudio counts steps from 0).
FINAL_STEP_OFFSET = 1


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI arguments.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dirs", nargs="+", type=Path, help="fetched run folders (runs/<run_id>)")
    parser.add_argument("--eval-config", type=Path, default=None, help="default: configs/eval.yaml")
    parser.add_argument("--no-viz", action="store_true", help="skip the P2 pictures (scripts/15 --p2 and --p2-hidden)")
    return parser


def recorded_path(path: Path, project_root: Path) -> str:
    """A path as reports record it: relative to the repo when inside it, else with home and repo masked.

    Relative paths are the same on every machine (scripts/20 compares them across runs) and carry
    no account name (CLAUDE.md rule 5).

    Args:
        path: Any path.
        project_root: Repo root.

    Returns:
        Text for the report.
    """
    absolute_path, absolute_root = Path(path).absolute(), Path(project_root).absolute()
    if absolute_path.is_relative_to(absolute_root):
        return str(absolute_path.relative_to(absolute_root))
    return repo_relative(str(absolute_path), project_root)


def read_job_info(run_dir: Path) -> Dict[str, str]:
    """The run's ``job_info.txt`` as key -> value (first occurrence of a repeated key wins).

    Args:
        run_dir: Run folder.

    Returns:
        Mapping of the ``key: value`` lines.
    """
    information: Dict[str, str] = {}
    for line in (run_dir / JOB_INFO_NAME).read_text().splitlines():
        if ": " in line:
            key, value = line.split(": ", 1)
            information.setdefault(key.strip(), value.strip())
    return information


def require_complete_run(job_info: Dict[str, str], run_dir: Path) -> None:
    """Refuse a run that did not finish with matching split and renders (P2 needs the final model).

    Args:
        job_info: Parsed job_info.txt.
        run_dir: Run folder (for the message).

    Raises:
        ValueError: If the run is incomplete.
    """
    if not job_info.get("end_utc") or job_info.get("split_render_check") != COMPLETE_RUN_SPLIT_CHECK:
        raise ValueError(f"{run_dir}: not a complete run (end_utc / split_render_check in {JOB_INFO_NAME})")


def run_profile(project_root: Path, run_id: str) -> str:
    """The Slurm profile a run was submitted with, from the ledger ``runs/submitted.tsv``.

    Args:
        project_root: Repo root.
        run_id: Run id.

    Returns:
        Profile name.

    Raises:
        ValueError: If the run is not in the ledger.
    """
    with open(project_root / RUN_LEDGER_RELATIVE_PATH, encoding="utf-8", newline="") as ledger_file:
        for ledger_row in csv.DictReader(ledger_file, delimiter="\t"):
            if ledger_row["run_id"] == run_id:
                return ledger_row["profile"]
    raise ValueError(f"{run_id} is not in {RUN_LEDGER_RELATIVE_PATH}")


def final_train_directory(run_dir: Path) -> Path:
    """The unique trainer directory (``train/<session>/<method>/<run_id>``) of a run.

    Args:
        run_dir: Run folder.

    Returns:
        Folder holding ``dataparser_transforms.json`` and ``nerfstudio_models/``.

    Raises:
        ValueError: If there is not exactly one.
    """
    transform_files = list((run_dir / "train").glob("*/*/*/dataparser_transforms.json"))
    if len(transform_files) != 1:
        raise ValueError(f"{run_dir}: expected one trainer directory, found {len(transform_files)}")
    return transform_files[0].parent


def gaussian_factors(run_dir: Path, project_root: Path) -> Dict[str, Any]:
    """The run's factorized Gaussians (T56), exported now from its own checkpoint inside the reference image.

    The export goes to a temporary folder that is deleted after loading, so no large archive is kept.

    Args:
        run_dir: Run folder.
        project_root: Repo root.

    Returns:
        {"arrays": FACTOR_KEYS -> arrays, "sha256": archive digest}.

    Raises:
        subprocess.CalledProcessError: If the image export fails.
    """
    # Export the final checkpoint inside the reference image (torch lives only there), then load the
    # archive before the temporary folder is removed.
    image_name = os.environ.get(IMAGE_NAME_ENVIRONMENT_VARIABLE, DEFAULT_IMAGE_NAME)
    with tempfile.TemporaryDirectory() as factors_dir:
        command = factor_export_command(project_root, final_train_directory(run_dir), Path(factors_dir), image_name,
                                        FACTORS_ARCHIVE_NAME)
        subprocess.run(command, check=True)
        archive_path = Path(factors_dir) / FACTORS_ARCHIVE_NAME
        return {"arrays": load_factors(archive_path), "sha256": file_sha256(archive_path)}


def load_run_inputs(run_dir: Path, eval_settings: Dict[str, Any], project_root: Path) -> Dict[str, Any]:
    """Everything a run's P2 score needs from its own records, resolved as scripts/13 resolves it.

    The ledger and job records are read first, so a refused run costs no Gaussian export.

    Args:
        run_dir: Fetched run folder.
        eval_settings: ``eval`` section of configs/eval.yaml.
        project_root: Repo root.

    Returns:
        Dict with job_info, profile, eval_root, render_dir, stems, split_record, session_config,
        manifest, gt_dir, cuboids, grid, threshold, sweeps.

    Raises:
        ValueError: Incomplete run, run not in the ledger, mismatched dataset, missing GT or renders.
    """
    job_info = read_job_info(run_dir)
    require_complete_run(job_info, run_dir)
    if "MAX_NUM_ITERATIONS" not in job_info:
        raise ValueError(f"{run_dir}: no MAX_NUM_ITERATIONS in {JOB_INFO_NAME}; the final checkpoint step is unknown")
    eval_root, render_dir, stems, split_record = resolve_render_target(run_dir, None)
    profile = run_profile(project_root, split_record["run"]["run_id"])
    session_config, manifest = resolve_session(split_record, project_root)
    gt_dir = Path(session_config["session"]["gt_dir"])
    cuboids, grid = load_ground_truth(gt_dir)
    threshold = float(eval_settings["render_accumulation_threshold"])
    return {"job_info": job_info, "profile": profile, "eval_root": eval_root, "render_dir": render_dir, "stems": stems,
            "split_record": split_record, "session_config": session_config, "manifest": manifest, "gt_dir": gt_dir,
            "cuboids": cuboids, "grid": grid, "threshold": threshold,
            "sweeps": load_test_lidar_sweeps(render_dir, stems, threshold)}


def check_hidden_space_provenance(record: Dict[str, Any], hidden_dir: Path, expected_input_hashes: Dict[str, str],
                                  npz_hash: str) -> None:
    """Refuse a hidden space that was not built from the files this run scores against.

    Args:
        record: Parsed ``hidden_space.json``.
        hidden_dir: Folder of the hidden space (for messages).
        expected_input_hashes: Recorded-path -> sha256 of the files the run uses now (cuboids, occupancy, reference split).
        npz_hash: sha256 of ``hidden_space.npz`` on disk.

    Raises:
        ValueError: On a wrong schema, a debug build, a recorded hash that differs, or a changed npz.
    """
    rebuild_hint = f"rebuild with {HIDDEN_SPACE_BUILD_COMMAND}"
    # Step 1: schema and full-sweep build.
    if record.get("schema") != HIDDEN_SPACE_SCHEMA:
        raise ValueError(f"{hidden_dir}: hidden-space schema {record.get('schema')!r}, need {HIDDEN_SPACE_SCHEMA!r}; {rebuild_hint}")
    if record.get(HIDDEN_RECORD_DEBUG_KEY) is not None:
        raise ValueError(f"{hidden_dir}: the hidden space is a debug build (first {record[HIDDEN_RECORD_DEBUG_KEY]} sweeps); {rebuild_hint}")
    # Step 2: every input the run uses now must hash as recorded: a changed GT or split changes the cells.
    recorded_hashes = record.get(HIDDEN_RECORD_INPUT_SHA_KEY, {})
    for input_path, expected_hash in expected_input_hashes.items():
        if recorded_hashes.get(input_path) != expected_hash:
            raise ValueError(f"{hidden_dir}: {input_path} differs from the file the hidden space was built from; {rebuild_hint}")
    # Step 3: the npz is the one the record describes.
    if record.get(HIDDEN_RECORD_OUTPUT_KEY, {}).get("sha256") != npz_hash:
        raise ValueError(f"{hidden_dir}: {HIDDEN_SPACE_NPZ_NAME} is not the file {HIDDEN_SPACE_JSON_NAME} describes; {rebuild_hint}")


def load_session_hidden_space(gt_dir: Path, inputs: Dict[str, Any], project_root: Path,
                              scored_class_ids: Dict[str, int]) -> Dict[str, Any]:
    """Read the session hidden space (scripts/24) and check it matches the run's GT, split and grid.

    Args:
        gt_dir: ``data/gt/<session>`` holding ``hidden_space.{npz,json}``.
        inputs: ``load_run_inputs`` output (cuboids, grid, session_config).
        project_root: Repo root (the reference split record lives under it).
        scored_class_ids: Scored class name -> id (names the ``neighbourhood_<name>`` arrays).

    Returns:
        {"cell_indices", "centres", "hidden", "hidden_near", "near_band_m", "gt_class_ids", "neighbourhood_by_class_name", "objects"
        (OBJECT_ARRAY_KEYS -> arrays), "range_margin_m", "max_range_m", "vertical_half_fov_rad",
        "path" (recorded), "sha256" (of the npz)}.

    Raises:
        ValueError: Hidden space missing, built before the near band existed, built from other inputs, or on another grid.
    """
    npz_path, json_path = gt_dir / HIDDEN_SPACE_NPZ_NAME, gt_dir / HIDDEN_SPACE_JSON_NAME
    # Step 1: both files must exist.
    if not (npz_path.is_file() and json_path.is_file()):
        raise ValueError(f"{gt_dir}: no hidden space; build it with {HIDDEN_SPACE_BUILD_COMMAND}")
    record = json.loads(json_path.read_text())
    # Step 2: provenance of the cuboids, occupancy and reference split.
    session_name = inputs["session_config"]["session"]["name"]
    input_files = (gt_dir / CUBOIDS_FILE_NAME, gt_dir / OCCUPANCY_FILE_NAME,
                   project_root / REFERENCE_SPLIT_RELATIVE_DIR / f"{session_name}.json")
    for input_file in input_files:
        if not input_file.is_file():
            raise ValueError(f"{input_file}: missing, so the hidden space cannot be matched to it")
    expected_input_hashes = {recorded_path(input_file, project_root): file_sha256(input_file) for input_file in input_files}
    npz_hash = file_sha256(npz_path)
    check_hidden_space_provenance(record, gt_dir, expected_input_hashes, npz_hash)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    # Step 3: the headline mask H_near must exist (a hidden space from before the near band lacks it).
    if HIDDEN_NEAR_ARRAY_KEY not in arrays or HIDDEN_RECORD_NEAR_BAND_KEY not in record:
        raise ValueError(f"{gt_dir}: the hidden space has no {HIDDEN_NEAR_ARRAY_KEY} / {HIDDEN_RECORD_NEAR_BAND_KEY} "
                         f"(built before the near band); rerun {HIDDEN_SPACE_BUILD_COMMAND}")
    # Step 4: the hidden space lies on the GT grid of this run.
    grid = inputs["grid"]
    if float(arrays["voxel_size"]) != float(grid["voxel_size"]) or not np.array_equal(
            np.asarray(arrays["grid_origin"], dtype=np.float64), np.asarray(grid["origin"], dtype=np.float64)):
        raise ValueError(f"{gt_dir}: hidden space voxel_size/grid_origin differ from the GT grid; rebuild with {HIDDEN_SPACE_BUILD_COMMAND}")
    return {"cell_indices": arrays["cell_indices"], "centres": arrays["centres"], "hidden": arrays["hidden"].astype(bool),
            HIDDEN_NEAR_ARRAY_KEY: arrays[HIDDEN_NEAR_ARRAY_KEY].astype(bool),
            "near_band_m": float(record[HIDDEN_RECORD_NEAR_BAND_KEY]),
            "gt_class_ids": arrays["gt_class_ids"],
            "neighbourhood_by_class_name": {name: arrays[f"neighbourhood_{name}"].astype(bool) for name in scored_class_ids},
            "objects": {key: arrays[key] for key in OBJECT_ARRAY_KEYS},
            "range_margin_m": float(arrays["range_margin_m"]), "max_range_m": float(record["max_range_m"]),
            "vertical_half_fov_rad": float(record["vertical_half_fov_rad"]),
            "path": recorded_path(npz_path, project_root), "sha256": npz_hash}


def class_entries(settings: HiddenEvalSettings, scored_class_ids: Dict[str, int]) -> Dict[str, Tuple[int, ...]]:
    """Report entries in order: class groups first, then each scored class.

    Args:
        settings: Parsed hidden-evaluation settings.
        scored_class_ids: Scored class name -> id.

    Returns:
        Entry name -> its class ids (a group has several, a class one).
    """
    ids_by_name = {**settings.class_groups, **{name: (class_id,) for name, class_id in scored_class_ids.items()}}
    return {name: ids_by_name[name] for name in class_group_names_ordered(settings, scored_class_ids)}


def entry_setup(class_ids: Sequence[int], is_group: bool, all_cuboids: Sequence[Any], objects: Dict[str, np.ndarray],
                margins: Dict[int, Optional[float]]) -> Dict[str, Any]:
    """The scored cuboids and margin of a class or group, from the hidden space's own objects.

    The scored cuboids are those the hidden space made objects of (box id and class id match), so the
    detection curve uses the same boxes as the per-object metrics. A group's cuboids are the union of
    its members'.

    Args:
        class_ids: The entry's class ids.
        is_group: True for a class group (``class_id`` is then None, margin None).
        all_cuboids: Every GT box of the session.
        objects: Hidden-space ``objects`` arrays (needs object_box_ids and object_class_ids).
        margins: ``class_margin_table`` output.

    Returns:
        {"class_id", "class_ids", "cuboids", "margin"}.
    """
    scored_keys = set(zip(np.asarray(objects["object_box_ids"]).tolist(), np.asarray(objects["object_class_ids"]).tolist()))
    cuboids = [cuboid for cuboid in all_cuboids if cuboid.class_id in class_ids and (cuboid.box_id, cuboid.class_id) in scored_keys]
    return {"class_id": None if is_group else int(class_ids[0]), "class_ids": [int(class_id) for class_id in class_ids],
            "cuboids": cuboids, "margin": None if is_group else margins[class_ids[0]]}


def class_report(setup: Dict[str, Any], class_cells: np.ndarray, truth: np.ndarray, centres: np.ndarray,
                 occupancy: np.ndarray, threshold: float, p1_hit: np.ndarray, evidence: np.ndarray,
                 f_min_values: np.ndarray, sweep_thresholds: np.ndarray) -> Dict[str, Any]:
    """P1 and P2 metrics of one class on its own cells.

    Args:
        setup: ``entry_setup`` output of the class or group.
        class_cells: (V,) bool, the entry's neighbourhood N (other-class mask applied).
        truth: (V,) bool, the entry's GT cells G (subset of class_cells).
        centres: (V, 3) cell centres. occupancy: (V,) T81 probabilities. threshold: Occupied cut.
        p1_hit: (V,) bool P1 hits. evidence: (V, 4) T78 flags.
        f_min_values: T79 grid. sweep_thresholds: P2 thresholds of the threshold sweep.

    Returns:
        JSON-ready class report.
    """
    # Restrict every array to the class's cells once.
    class_truth, class_evidence = truth[class_cells], evidence[class_cells]
    class_occupied, class_p1 = occupancy[class_cells] >= threshold, p1_hit[class_cells]
    fractions = cuboid_occupied_fractions(setup["cuboids"], centres[class_cells], class_truth, class_occupied)
    return {
        "class_id": setup["class_id"],
        "class_ids": setup["class_ids"],
        "scored_cuboids": len(setup["cuboids"]),
        "margin_m": setup["margin"],
        "scored_cells": int(np.count_nonzero(class_cells)),
        "ground_truth_cells": int(np.count_nonzero(class_truth)),
        "P1": occupied_cell_metrics(class_p1, class_truth, class_evidence),
        "P2": occupied_cell_metrics(class_occupied, class_truth, class_evidence),
        "present_but_hidden": present_but_hidden(class_occupied, class_p1, class_truth, class_evidence),
        "detection": {
            "box_ids": [cuboid.box_id for cuboid in setup["cuboids"]],
            "gt_cells": np.asarray(fractions["gt_cells"]).tolist(),
            "occupied_cells": np.asarray(fractions["occupied_cells"]).tolist(),
            "curve": detection_curve(fractions["fraction"], f_min_values),
        },
        "threshold_sweep": threshold_sweep(occupancy[class_cells], class_truth, class_evidence, sweep_thresholds),
    }


def evaluate_run(run_dir: Path, eval_settings: Dict[str, Any], eval_config_label: str, project_root: Path) -> Dict[str, Any]:
    """Score one run's Gaussians on the hidden space's cells (P2), beside its own ungated rendered points (P1).

    Args:
        run_dir: Fetched run folder.
        eval_settings: ``eval`` section of the eval config.
        eval_config_label: Repo-relative path of that config (recorded as the threshold's source).
        project_root: Repo root.

    Returns:
        {"report": JSON-ready dict, "cells": arrays for the cells archive, "eval_dir": output folder}.

    Raises:
        ValueError: Refused input (incomplete run, mismatched dataset, missing GT or renders, wrong checkpoint).
        subprocess.CalledProcessError: If the Gaussian export in the image fails.
    """
    # Step 1: inputs, settings and the session hidden space's cells.
    settings = parse_hidden_settings(eval_settings)
    inputs = load_run_inputs(run_dir, eval_settings, project_root)
    grid, threshold, split_record = inputs["grid"], inputs["threshold"], inputs["split_record"]
    grid_origin, voxel_size = np.asarray(grid["origin"], dtype=np.float64), float(grid["voxel_size"])
    scored_class_ids = dict(eval_settings["scored_class_ids"])
    hidden_space = load_session_hidden_space(inputs["gt_dir"], inputs, project_root, scored_class_ids)
    cell_indices, centres = hidden_space["cell_indices"], hidden_space["centres"]

    # Step 2: recorded evidence (T78) and the run's own P1 hits on those cells.
    evidence = accumulate_recorded_evidence(inputs["render_dir"], inputs["stems"], centres, voxel_size)
    p1_hit = p1_hit_cells(inputs["sweeps"], cell_indices, grid_origin, voxel_size, grid["class_ids"].shape)

    # Step 3: T81 occupancy from the final checkpoint's factorized Gaussians.
    factors = gaussian_factors(run_dir, project_root)
    factor_arrays = factors["arrays"]
    expected_step = int(inputs["job_info"]["MAX_NUM_ITERATIONS"]) - FINAL_STEP_OFFSET
    if int(factor_arrays["checkpoint_step"]) != expected_step:
        raise ValueError(f"{run_dir}: checkpoint step {int(factor_arrays['checkpoint_step'])}, final is {expected_step}")
    occupancy = peak_alpha_voxel_occupancy(
        cell_indices, grid_origin, voxel_size, factor_arrays["means"], factor_arrays["opacities"],
        factor_arrays["principal_axes"], factor_arrays["axis_std"], int(eval_settings["gaussian_occupancy_chunk_size"]))

    # Step 4: per class and group, with the other-class mask applied to N (hidden_metrics.class_group_masks).
    f_min_values = f_min_grid(int(eval_settings["p2_f_min_grid_points"]))
    p2_thresholds = np.asarray(settings.p2_thresholds, dtype=np.float64)
    margins = class_margin_table(inputs["cuboids"], list(scored_class_ids.values()))
    neighbourhood_by_class_id = {class_id: hidden_space["neighbourhood_by_class_name"][name]
                                 for name, class_id in scored_class_ids.items()}
    class_reports: Dict[str, Any] = {}
    cell_arrays: Dict[str, np.ndarray] = {}
    for entry_name, entry_class_ids in class_entries(settings, scored_class_ids).items():
        class_cells, truth = class_group_masks(hidden_space["gt_class_ids"], entry_class_ids, neighbourhood_by_class_id)
        cell_arrays[f"neighbourhood_{entry_name}"], cell_arrays[f"truth_{entry_name}"] = class_cells, truth
        setup = entry_setup(entry_class_ids, entry_name in settings.class_groups, inputs["cuboids"],
                            hidden_space["objects"], margins)
        class_reports[entry_name] = class_report(setup, class_cells, truth, centres, occupancy, threshold, p1_hit,
                                                 evidence, f_min_values, p2_thresholds)
        # The headline block uses H_near, the full-H block H; both share the masks, hits and objects.
        for report_key, hidden_mask in ((REPORT_HIDDEN_NEAR_KEY, hidden_space[HIDDEN_NEAR_ARRAY_KEY]),
                                        (REPORT_HIDDEN_KEY, hidden_space["hidden"])):
            class_reports[entry_name][report_key] = hidden_class_report(
                entry_class_ids, class_cells, truth, hidden_mask, p1_hit, occupancy, settings.p2_thresholds,
                hidden_space["objects"])

    # Step 5: the report, with every setting and input provenance; paths are repo-relative (rule 5).
    gt_dir = inputs["gt_dir"]
    report = {
        "schema": REPORT_SCHEMA,
        "run_id": split_record["run"]["run_id"],
        "profile": inputs["profile"],
        # The trainer folder train/<session>/<method>/<run_id> names the method ns-train ran; job_info's
        # TRAIN_METHOD is missing in runs submitted before it was recorded.
        "train_method": final_train_directory(run_dir).parent.name,
        "git_sha": split_record["run"]["git_sha"],
        "session": inputs["session_config"]["session"]["name"],
        "dataset": split_record["dataset"]["name"],
        "split_membership_sha256": split_record["membership_sha256"],
        # Repo-relative, as scripts/13 records it: scripts/20 compares this dict across runs and machines.
        "ground_truth": {"dir": recorded_path(gt_dir, project_root),
                         "cuboids_sha256": file_sha256(gt_dir / CUBOIDS_FILE_NAME),
                         "occupancy_sha256": file_sha256(gt_dir / OCCUPANCY_FILE_NAME), "voxel_size_m": voxel_size},
        "gaussians": {"count": int(len(factor_arrays["opacities"])), "checkpoint_step": int(factor_arrays["checkpoint_step"]),
                      "factors_sha256": factors["sha256"]},
        "sweep_count": len(inputs["sweeps"]),
        "settings": {"occupancy_estimator": OCCUPANCY_ESTIMATOR, "occupancy_threshold": threshold,
                     "occupancy_threshold_source": f"{eval_config_label} render_accumulation_threshold",
                     "max_range_m": hidden_space["max_range_m"], "vertical_half_fov_rad": hidden_space["vertical_half_fov_rad"],
                     "f_min_values": f_min_values.tolist(), "p2_thresholds": list(settings.p2_thresholds)},
        "hidden_space": {"path": hidden_space["path"], "sha256": hidden_space["sha256"],
                         "range_margin_m": hidden_space["range_margin_m"],
                         "near_band_m": hidden_space["near_band_m"]},
        "scored_cells": int(len(cell_indices)),
        "classes": class_reports,
    }
    cells = {"cell_indices": cell_indices, "centres": centres, "occupancy": occupancy, "p1_hit": p1_hit,
             "evidence": evidence, "threshold": np.float64(threshold), "grid_origin": grid_origin,
             "voxel_size": np.float64(voxel_size), HIDDEN_KEY: hidden_space["hidden"],
             HIDDEN_NEAR_KEY: hidden_space[HIDDEN_NEAR_ARRAY_KEY],
             "gt_class_ids": hidden_space["gt_class_ids"], P2_THRESHOLDS_KEY: p2_thresholds,
             **hidden_space["objects"], **cell_arrays}
    return {"report": report, "cells": cells, "eval_dir": inputs["eval_root"] / EVAL_OUTPUT_DIR_NAME}


def format_ratio(value: Optional[float]) -> str:
    """Three-decimal string, or "n/a" for None.

    Args:
        value: Ratio.

    Returns:
        Text.
    """
    return "n/a" if value is None else f"{value:.3f}"


def format_object_mean(per_object: Dict[str, Any], metric_name: str) -> str:
    """``mean (n=defined)`` of a per-object metric, "n/a (n=0)" when no object defines it.

    Args:
        per_object: T95 summary (mean_<metric>, defined_counts).
        metric_name: "recall", "precision" or "iou".

    Returns:
        Text.
    """
    return f"{format_ratio(per_object[f'mean_{metric_name}'])} (n={per_object['defined_counts'][metric_name]})"


def hidden_table_lines(class_name: str, hidden_block: Dict[str, Any], threshold_text: str,
                       metric_suffix: str = HIDDEN_FULL_METRIC_SUFFIX) -> List[str]:
    """Markdown table of one class at one P2 threshold: hidden metrics, whole-neighbourhood baseline, P1.

    Args:
        class_name: Class or group name.
        hidden_block: ``hidden_class_report`` output.
        threshold_text: A key of ``hidden_block["P2"]``.
        metric_suffix: Metric-name suffix of the block's hidden mask (``_Hnear`` or ``_H``).

    Returns:
        Lines (table, then the context line).
    """
    p2_block, p1_block = hidden_block["P2"][threshold_text], hidden_block["P1"]
    lines = [f"### {class_name}, P2 occupied at >= {threshold_text}", "",
             "| row | pooled | mean over objects (n defined) |", "|---|---|---|"]
    # Hidden rows (H = session hidden space), then the baseline rows (H = every scored cell).
    for metric_label, metric_name in HIDDEN_TABLE_METRICS:
        lines.append(f"| P2 {metric_label}{metric_suffix} | {format_ratio(p2_block['pooled'][metric_name])} | "
                     f"{format_object_mean(p2_block['per_object'], metric_name)} |")
    for metric_label, metric_name in HIDDEN_TABLE_METRICS:
        lines.append(f"| P2 {metric_label} (baseline) | {format_ratio(p2_block['baseline_pooled'][metric_name])} | "
                     f"{format_object_mean(p2_block['baseline_per_object'], metric_name)} |")
    # P1: baseline recall and precision, then recall_H (the lidar-view number).
    for metric_label, metric_name in HIDDEN_TABLE_METRICS[:2]:
        lines.append(f"| P1 {metric_label} | {format_ratio(p1_block['baseline_pooled'][metric_name])} | "
                     f"{format_object_mean(p1_block['baseline_per_object'], metric_name)} |")
    lines.append(f"| P1 recall{metric_suffix} | {format_ratio(p1_block['pooled']['recall'])} | {format_object_mean(p1_block['per_object'], 'recall')} |")
    context = hidden_block["context"]
    lines += ["", f"Hidden GT cells {context['hidden_gt_cells']}; objects with hidden GT {context['objects_with_hidden_gt']} "
                  f"of {context['object_count']}.", ""]
    return lines


def hidden_section_lines(report: Dict[str, Any]) -> List[str]:
    """Markdown of the hidden-voxel tables: H_near (headline, titled with the band) then full H.

    Args:
        report: ``evaluate_run`` report (``hidden_space.near_band_m`` and per class ``hidden_near`` / ``hidden`` blocks).

    Returns:
        Lines; the group ``obstacles`` leads (dict order is report order).
    """
    lines: List[str] = []
    near_band_m = report["hidden_space"]["near_band_m"]
    hidden_sections = (
        (f"## Hidden-voxel metrics, hidden within {near_band_m:g} m of the path (H_near, read after the whole scene; baseline = every scored cell)",
         REPORT_HIDDEN_NEAR_KEY, HIDDEN_NEAR_METRIC_SUFFIX),
        ("## Hidden-voxel metrics, whole session hidden space (H; baseline = every scored cell)",
         REPORT_HIDDEN_KEY, HIDDEN_FULL_METRIC_SUFFIX))
    for section_title, block_key, metric_suffix in hidden_sections:
        lines += [section_title, ""]
        for class_name, class_entry in report["classes"].items():
            for threshold_text in class_entry[block_key]["P2"]:
                lines += hidden_table_lines(class_name, class_entry[block_key], threshold_text, metric_suffix)
    return lines


def report_markdown(report: Dict[str, Any]) -> str:
    """Human-readable summary of one run's P2 report.

    Args:
        report: ``evaluate_run`` report.

    Returns:
        Markdown text.
    """
    lines = [f"# P2 Gaussian occupancy: {report['run_id']}", "",
             f"Profile `{report['profile']}`, method `{report['train_method']}`, code `{report['git_sha']}`, "
             f"{report['gaussians']['count']} Gaussians (step {report['gaussians']['checkpoint_step']}), "
             f"occupied at >= {report['settings']['occupancy_threshold']}, {report['scored_cells']} scored cells.", ""]
    # Step 1: P1 and P2 side by side per class over the whole neighbourhood, with recall per evidence stratum (read first).
    lines += ["## P1 and P2 over the whole neighbourhood", "",
              "| class | test | precision | recall | IoU | recall seen | recall blocked | recall no return |",
              "|---|---|---|---|---|---|---|---|"]
    for class_name, class_report in report["classes"].items():
        for test_name in ("P1", "P2"):
            metrics = class_report[test_name]
            strata = metrics["strata"]
            lines.append(f"| {class_name} | {test_name} | {format_ratio(metrics['precision'])} | {format_ratio(metrics['recall'])} | "
                         f"{format_ratio(metrics['iou'])} | {format_ratio(strata['seen']['recall'])} | "
                         f"{format_ratio(strata['blocked']['recall'])} | {format_ratio(strata['no_return']['recall'])} |")
    # Step 2: the hidden-voxel tables, H_near (read second, after the whole scene) then full H.
    lines += hidden_section_lines(report)
    # Step 3: present-but-hidden and the detection curve per class.
    lines += ["", "## Present but hidden (GT cells occupied in P2, no P1 point)", ""]
    for class_name, class_report in report["classes"].items():
        lines.append(f"- {class_name}: {json.dumps(class_report['present_but_hidden'], sort_keys=True)}")
    lines += ["", "## Detection rate over f_min (fraction of cuboids whose occupied-cell fraction >= f_min)", ""]
    for class_name, class_report in report["classes"].items():
        lines.append(f"- {class_name}: {json.dumps(class_report['detection']['curve'], sort_keys=True)}")
    return "\n".join(lines) + "\n"


def p2_viz_command(run_dir: Path, output_dir: Path, eval_config_path: Path,
                   mode_arguments: Sequence[str] = P2_VIZ_ARGUMENTS) -> List[str]:
    """Command that draws the run's P2 cells with scripts/15 in the isolated viz/ environment.

    Args:
        run_dir: Run folder (absolute).
        output_dir: Picture folder (absolute).
        eval_config_path: configs/eval.yaml (absolute).
        mode_arguments: scripts/15 picture flags (default ``--p2``; the hidden picture passes
            ``--p2-hidden --hidden-class NAME --hidden-threshold T``).

    Returns:
        The command, program first.
    """
    return ["uv", "run", "--project", str(VIEWER_PROJECT_RELATIVE_DIR), "python", str(VIEWER_SCRIPT_RELATIVE_PATH),
            "--run", str(run_dir), *mode_arguments, "--output-dir", str(output_dir), "--eval-config", str(eval_config_path)]


def hidden_viz_arguments(class_name: str, threshold: float) -> Tuple[str, ...]:
    """scripts/15 flags of the hidden-P2 picture (H_near band) of one class or group at one threshold.

    Args:
        class_name: Class or group name (e.g. ``obstacles``).
        threshold: P2 occupancy cut (one of the evaluated thresholds).

    Returns:
        The flag tuple for ``p2_viz_command``.
    """
    return (P2_HIDDEN_VIZ_FLAG, "--hidden-class", class_name, "--hidden-threshold", str(threshold),
            "--hidden-band", HIDDEN_VIZ_BAND)


def launch_p2_viz(run_dir: Path, eval_dir: Path, eval_config_path: Path, project_root: Path,
                  mode_arguments: Sequence[str] = P2_VIZ_ARGUMENTS) -> str:
    """Draw a P2 picture with the shared viewer launcher; never raises (the metrics are already written).

    Args:
        run_dir: Run folder.
        eval_dir: ``<run>/eval``.
        eval_config_path: configs/eval.yaml.
        project_root: Repo root.
        mode_arguments: scripts/15 picture flags.

    Returns:
        "written: <folder>" or "failed: <reason>" (paths repo-relative).
    """
    viz_output_dir = (eval_dir / VIZ_OUTPUT_DIR_NAME).resolve()
    command = p2_viz_command(run_dir.resolve(), viz_output_dir, eval_config_path.resolve(), mode_arguments)
    return run_viewer_command(command, viz_output_dir, project_root)


def main(argument_list: Optional[List[str]] = None) -> int:
    """Evaluate each given run and write its P2 report, cells archive and picture.

    Args:
        argument_list: CLI arguments (default: sys.argv).

    Returns:
        Exit code: 0 success, 1 refused input (a picture failure does not change it).
    """
    arguments = build_argument_parser().parse_args(argument_list)
    project_root = find_project_root()
    eval_config_path = arguments.eval_config or project_root / DEFAULT_EVAL_CONFIG_RELATIVE_PATH
    eval_settings = yaml.safe_load(Path(eval_config_path).read_text())["eval"]
    # Per run: score, write the metrics, then draw and record the drawing's outcome.
    try:
        for run_dir in arguments.run_dirs:
            result = evaluate_run(run_dir, eval_settings, recorded_path(Path(eval_config_path), project_root), project_root)
            eval_dir, report = result["eval_dir"], result["report"]
            eval_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(eval_dir / CELLS_ARCHIVE_NAME, **result["cells"])
            report["cells_sha256"] = file_sha256(eval_dir / CELLS_ARCHIVE_NAME)
            viz_status = "skipped: --no-viz" if arguments.no_viz else "pending"
            report["viz"] = {VIZ_KEY_P2: viz_status, VIZ_KEY_P2_HIDDEN: viz_status}
            (eval_dir / REPORT_JSON_NAME).write_text(json.dumps(report, indent=1, sort_keys=True))
            (eval_dir / REPORT_MARKDOWN_NAME).write_text(report_markdown(report))
            print(f"wrote {eval_dir / REPORT_JSON_NAME}")
            if not arguments.no_viz:
                # Two pictures, each outcome recorded; a failure warns but never changes the exit code.
                headline_threshold = report["settings"]["occupancy_threshold"]
                picture_arguments = {VIZ_KEY_P2: P2_VIZ_ARGUMENTS}
                if HEADLINE_GROUP_NAME in report["classes"]:
                    picture_arguments[VIZ_KEY_P2_HIDDEN] = hidden_viz_arguments(HEADLINE_GROUP_NAME, headline_threshold)
                else:
                    report["viz"][VIZ_KEY_P2_HIDDEN] = f"failed: no {HEADLINE_GROUP_NAME} group in the report"
                for picture_name, mode_arguments in picture_arguments.items():
                    report["viz"][picture_name] = launch_p2_viz(run_dir, eval_dir, Path(eval_config_path), project_root, mode_arguments)
                for picture_name, outcome in report["viz"].items():
                    if outcome.startswith("failed"):
                        print(f"WARNING: no {picture_name} picture ({outcome}). Set up its environment with: "
                              f"{VIZ_ENVIRONMENT_FIX_COMMAND}", file=sys.stderr)
                    else:
                        print(outcome)
                (eval_dir / REPORT_JSON_NAME).write_text(json.dumps(report, indent=1, sort_keys=True))
            print(report_markdown(report))
    # A refused run, a missing record (KeyError/OSError) or a failed image export stops the batch with a message.
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"refused: {error!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
