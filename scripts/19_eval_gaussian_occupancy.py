"""P2 evaluation of one or more runs: occupancy from the trained Gaussians on the GT grid (PLAN §5).

P1 (scripts/13) asks what a lidar at the eval poses would see; P2 asks what the model contains. A
trunk kept behind grass Gaussians is invisible in P1 and shows up only here, and geometry the
radar invented shows up here as false positives, so precision matters as much as recall.

Per run:
1. the session, split, GT and test sweeps from the run's own records (as scripts/13);
2. the scored cells: per scored class, the T32 scored region grown by the class's margin;
3. the factorized checkpoint Gaussians (T56), exported inside the reference image (torch) into a
   temporary folder that is deleted afterwards;
4. T81 peak-alpha occupancy on every scored cell (each Gaussian's sigma(o) in the voxel holding its
   mean, T57 3-sigma tails elsewhere); occupied = probability >= the P1 threshold. T57's centre sample
   alone misses SplatAD's thin Gaussians at the GT voxel size (docs/research/p2_estimator_diagnostic/);
5. per class: P1 and P2 voxel P/R/IoU with recall per T78 evidence stratum, the T79 detection
   curve over f_min, T80 present-but-hidden, and a threshold sweep (floater/haze sensitivity);
6. ``<run>/eval/gaussian_occupancy.{json,md}`` plus ``gaussian_occupancy_cells.npz`` (per-cell
   arrays for scripts/20 and scripts/15), then the P2 picture with scripts/15 (``--no-viz`` skips).

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
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from viking_splatad.config import find_project_root
from viking_splatad.eval.detection import class_margin_table, scored_region_cuboids, voxel_centers
from viking_splatad.eval.gaussian_detection import (
    cuboid_occupied_fractions,
    detection_curve,
    f_min_grid,
    present_but_hidden,
    threshold_sweep,
)
from viking_splatad.eval.hidden_geometry import occupied_cell_metrics, peak_alpha_voxel_occupancy
from viking_splatad.eval.p2_cells import accumulate_recorded_evidence, factor_export_command, load_factors, p1_hit_cells
from viking_splatad.eval.rendered_pointcloud import load_test_lidar_sweeps
from viking_splatad.eval.run_context import (
    CUBOIDS_FILE_NAME,
    OCCUPANCY_FILE_NAME,
    build_class_setup,
    file_sha256,
    load_ground_truth,
    pooled_map_points,
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

REPORT_SCHEMA = "viking-splatad/gaussian-occupancy/1"
# Recorded in every report so scripts/20 refuses to pool runs scored by different estimators.
OCCUPANCY_ESTIMATOR = "T81 peak alpha in voxel"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
EVAL_OUTPUT_DIR_NAME = "eval"
REPORT_JSON_NAME = "gaussian_occupancy.json"
REPORT_MARKDOWN_NAME = "gaussian_occupancy.md"
CELLS_ARCHIVE_NAME = "gaussian_occupancy_cells.npz"
VIZ_OUTPUT_DIR_NAME = "viz_p2"
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
    parser.add_argument("--no-viz", action="store_true", help="skip the P2 picture (scripts/15 --p2)")
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


def scored_cells(inputs: Dict[str, Any], eval_settings: Dict[str, Any]) -> Dict[str, Any]:
    """The scored cells: union of the per-class neighbourhoods (T32 region grown by each class's margin).

    Built by the same helpers as P1 (scripts/13), so P1 and P2 score the same cells.

    Args:
        inputs: ``load_run_inputs`` output.
        eval_settings: ``eval`` section of configs/eval.yaml.

    Returns:
        {"class_setup", "cell_indices" (V, 3), "centres" (V, 3), "max_range", "vertical_half_fov"}.
    """
    sweeps, grid = inputs["sweeps"], inputs["grid"]
    sweep_poses = [sweep.lidar_to_map for sweep in sweeps]
    scored_class_ids = dict(eval_settings["scored_class_ids"])
    class_ids = list(scored_class_ids.values())
    # Step 1: scored region, from the test sweeps' real returns and the lidar's elevation range.
    visible_by_class, max_range, vertical_half_fov = scored_region_cuboids(
        inputs["cuboids"], class_ids, sweep_poses, [sweep.real_points for sweep in sweeps],
        inputs["manifest"]["lidar"]["elevation_mapping"])
    # Step 2: per-class neighbourhoods and GT (n_min is computed too but is not used by P2).
    real_points_map = pooled_map_points(sweeps, "real", grid)["points"]
    class_setup = build_class_setup(scored_class_ids, visible_by_class, class_margin_table(inputs["cuboids"], class_ids),
                                    real_points_map, grid, sweep_poses, max_range, vertical_half_fov,
                                    float(eval_settings["n_min_quantile"]))
    # Step 3: the union; every per-class mask is a subset of it.
    union_mask = np.logical_or.reduce([setup["neighbourhood"] for setup in class_setup.values()])
    cell_indices = np.argwhere(union_mask)
    centres = voxel_centers(cell_indices, np.asarray(grid["origin"], dtype=np.float64), float(grid["voxel_size"]))
    return {"class_setup": class_setup, "cell_indices": cell_indices, "centres": centres,
            "max_range": max_range, "vertical_half_fov": vertical_half_fov}


def class_report(setup: Dict[str, Any], class_cells: np.ndarray, truth: np.ndarray, centres: np.ndarray,
                 occupancy: np.ndarray, threshold: float, p1_hit: np.ndarray, evidence: np.ndarray,
                 f_min_values: np.ndarray, sweep_thresholds: np.ndarray) -> Dict[str, Any]:
    """P1 and P2 metrics of one class on its own cells.

    Args:
        setup: ``build_class_setup`` entry of the class.
        class_cells: (V,) bool, the class's cells within the scored union.
        truth: (V,) bool, the class's GT cells (subset of class_cells).
        centres: (V, 3) cell centres. occupancy: (V,) T81 probabilities. threshold: Occupied cut.
        p1_hit: (V,) bool P1 hits. evidence: (V, 4) T78 flags.
        f_min_values: T79 grid. sweep_thresholds: Threshold-sweep grid.

    Returns:
        JSON-ready class report.
    """
    # Restrict every array to the class's cells once.
    class_truth, class_evidence = truth[class_cells], evidence[class_cells]
    class_occupied, class_p1 = occupancy[class_cells] >= threshold, p1_hit[class_cells]
    fractions = cuboid_occupied_fractions(setup["cuboids"], centres[class_cells], class_truth, class_occupied)
    return {
        "class_id": setup["class_id"],
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
    """Score one run's Gaussians on its scored cells (P2), beside its own ungated rendered points (P1).

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
    # Step 1: inputs and scored cells.
    inputs = load_run_inputs(run_dir, eval_settings, project_root)
    grid, threshold, split_record = inputs["grid"], inputs["threshold"], inputs["split_record"]
    grid_origin, voxel_size = np.asarray(grid["origin"], dtype=np.float64), float(grid["voxel_size"])
    cells_setup = scored_cells(inputs, eval_settings)
    cell_indices, centres = cells_setup["cell_indices"], cells_setup["centres"]

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

    # Step 4: per-class metrics on each class's cells.
    f_min_values = f_min_grid(int(eval_settings["p2_f_min_grid_points"]))
    sweep_thresholds = f_min_grid(int(eval_settings["p2_threshold_sweep_points"]))
    class_reports: Dict[str, Any] = {}
    cell_arrays: Dict[str, np.ndarray] = {}
    for class_name, setup in cells_setup["class_setup"].items():
        class_cells = setup["neighbourhood"][tuple(cell_indices.T)]
        truth = setup["ground_truth"][tuple(cell_indices.T)]
        cell_arrays[f"neighbourhood_{class_name}"], cell_arrays[f"truth_{class_name}"] = class_cells, truth
        class_reports[class_name] = class_report(setup, class_cells, truth, centres, occupancy, threshold, p1_hit,
                                                 evidence, f_min_values, sweep_thresholds)

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
                     "max_range_m": cells_setup["max_range"], "vertical_half_fov_rad": cells_setup["vertical_half_fov"],
                     "f_min_values": f_min_values.tolist(), "threshold_sweep_values": sweep_thresholds.tolist()},
        "scored_cells": int(len(cell_indices)),
        "classes": class_reports,
    }
    cells = {"cell_indices": cell_indices, "centres": centres, "occupancy": occupancy, "p1_hit": p1_hit,
             "evidence": evidence, "threshold": np.float64(threshold), "grid_origin": grid_origin,
             "voxel_size": np.float64(voxel_size), **cell_arrays}
    return {"report": report, "cells": cells, "eval_dir": inputs["eval_root"] / EVAL_OUTPUT_DIR_NAME}


def format_ratio(value: Optional[float]) -> str:
    """Three-decimal string, or "n/a" for None.

    Args:
        value: Ratio.

    Returns:
        Text.
    """
    return "n/a" if value is None else f"{value:.3f}"


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
             f"occupied at >= {report['settings']['occupancy_threshold']}, {report['scored_cells']} scored cells.", "",
             "| class | test | precision | recall | IoU | recall seen | recall blocked | recall no return |",
             "|---|---|---|---|---|---|---|---|"]
    # Step 1: P1 and P2 side by side per class, with recall per evidence stratum.
    for class_name, class_report in report["classes"].items():
        for test_name in ("P1", "P2"):
            metrics = class_report[test_name]
            strata = metrics["strata"]
            lines.append(f"| {class_name} | {test_name} | {format_ratio(metrics['precision'])} | {format_ratio(metrics['recall'])} | "
                         f"{format_ratio(metrics['iou'])} | {format_ratio(strata['seen']['recall'])} | "
                         f"{format_ratio(strata['blocked']['recall'])} | {format_ratio(strata['no_return']['recall'])} |")
    # Step 2: present-but-hidden and the detection curve per class.
    lines += ["", "## Present but hidden (GT cells occupied in P2, no P1 point)", ""]
    for class_name, class_report in report["classes"].items():
        lines.append(f"- {class_name}: {json.dumps(class_report['present_but_hidden'], sort_keys=True)}")
    lines += ["", "## Detection rate over f_min (fraction of cuboids whose occupied-cell fraction >= f_min)", ""]
    for class_name, class_report in report["classes"].items():
        lines.append(f"- {class_name}: {json.dumps(class_report['detection']['curve'], sort_keys=True)}")
    return "\n".join(lines) + "\n"


def p2_viz_command(run_dir: Path, output_dir: Path, eval_config_path: Path) -> List[str]:
    """Command that draws the run's P2 cells with scripts/15 in the isolated viz/ environment.

    Args:
        run_dir: Run folder (absolute).
        output_dir: Picture folder (absolute).
        eval_config_path: configs/eval.yaml (absolute).

    Returns:
        The command, program first.
    """
    return ["uv", "run", "--project", str(VIEWER_PROJECT_RELATIVE_DIR), "python", str(VIEWER_SCRIPT_RELATIVE_PATH),
            "--run", str(run_dir), "--p2", "--output-dir", str(output_dir), "--eval-config", str(eval_config_path)]


def launch_p2_viz(run_dir: Path, eval_dir: Path, eval_config_path: Path, project_root: Path) -> str:
    """Draw the P2 picture with the shared viewer launcher; never raises (the metrics are already written).

    Args:
        run_dir: Run folder.
        eval_dir: ``<run>/eval``.
        eval_config_path: configs/eval.yaml.
        project_root: Repo root.

    Returns:
        "written: <folder>" or "failed: <reason>" (paths repo-relative).
    """
    viz_output_dir = (eval_dir / VIZ_OUTPUT_DIR_NAME).resolve()
    command = p2_viz_command(run_dir.resolve(), viz_output_dir, eval_config_path.resolve())
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
            report["viz"] = "skipped: --no-viz" if arguments.no_viz else "pending"
            (eval_dir / REPORT_JSON_NAME).write_text(json.dumps(report, indent=1, sort_keys=True))
            (eval_dir / REPORT_MARKDOWN_NAME).write_text(report_markdown(report))
            print(f"wrote {eval_dir / REPORT_JSON_NAME}")
            if not arguments.no_viz:
                report["viz"] = launch_p2_viz(run_dir, eval_dir, Path(eval_config_path), project_root)
                if report["viz"].startswith("failed"):
                    print(f"WARNING: no picture ({report['viz']}). Set up its environment with: "
                          f"{VIZ_ENVIRONMENT_FIX_COMMAND}", file=sys.stderr)
                else:
                    print(report["viz"])
                (eval_dir / REPORT_JSON_NAME).write_text(json.dumps(report, indent=1, sort_keys=True))
            print(report_markdown(report))
    # A refused run, a missing record (KeyError/OSError) or a failed image export stops the batch with a message.
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        print(f"refused: {error!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
