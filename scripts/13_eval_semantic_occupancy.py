"""Score a run's rendered test lidar against its session's GT cuboids (PLAN §5 P1, T45/T46/T67/T68).

Plug and play: give it any fetched run folder (or one of its render-only jobs with ``--rerender``)
whose lidar renders exist. Everything else is derived from the run's own records:

    1. ``split.json`` (scripts/09) names the dataset and the rendered test frames; for a render-only
       job, ``rerenders/<label>/selection.json`` names them. A render-name mismatch is refused.
    2. Dataset ``manifest.json`` (its sha256 must equal split.json's) gives the session
       (``session.name``, so smoke datasets map to their real session) and the beam elevations;
       ``configs/sessions/<session>.yaml`` gives the GT folder and labels sample. May and June are
       therefore always scored against their own labels (2024_05 / 2024_06), never each other's.
    3. Each sweep's gated ``pred_points`` and ungated points (T67) go to the map frame with the
       sweep's ``lidar_to_map`` (T68) and are pooled over the test sweeps.
    4. Scored set: GT cuboids (and voxels) within range and vertical FOV of at least one test pose.
       Range = farthest real return of the test sweeps; FOV = largest |beam elevation|.
    5. Per scored class (configs/eval.yaml): T45 detection rate (n_min = lower quartile of the real
       sweeps' per-cuboid counts), T46 neighbourhood point precision, voxel IoU/P/R on the GT grid.
    6. Writes ``<eval root>/eval/semantic_occupancy.{json,md}``. With several runs, also prints a
       side-by-side table, refused unless all runs share one session (no May <-> June mixing).
    7. Visual aid: after the report is safely written, launches scripts/15 (PyVista, isolated viz/
       environment) to draw the pooled voxel maps (overlay, recall, precision) into ``<eval root>/eval/viz/``. A
       failure there is reported and recorded as ``viz`` in the JSON, never changes the metrics or
       the exit code. ``--no-viz`` skips it.

Usage:
    uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> [runs/<run_id> ...]
    uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> --rerender render-<...>
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from viking_splatad.config import find_project_root, load_session_config
from viking_splatad.eval.detection import (
    class_margin_table,
    cuboid_detection_rate,
    cuboid_point_counts,
    load_cuboids_json,
    n_min_from_real_counts,
    neighbourhood_point_precision,
    neighbourhood_voxel_metrics,
    scored_region_cuboids,
    summarise_metrics,
    visible_neighbourhood_mask,
)
from viking_splatad.eval.rendered_pointcloud import (
    RenderedSweep,
    load_test_lidar_sweeps,
    occupied_voxel_grid,
    points_to_map_voxels,
)
from viking_splatad.viz.pyvista_scene import viz_command

# Report schema id (bump when keys change).
REPORT_SCHEMA = "viking-splatad/semantic-occupancy/1"
# Run-record files (scripts/09, submit_run.py render) and render folders (runs/RUN_FOLDER_GLOSSARY.md).
SPLIT_RECORD_FILE_NAME = "split.json"
SELECTION_FILE_NAME = "selection.json"
JOB_INFO_FILE_NAME = "job_info.txt"
RERENDERS_DIRECTORY_NAME = "rerenders"
LIDAR_RENDER_SUBDIR = Path("renders") / "test" / "lidar"
# split.json render_check key of the lidar test renders.
LIDAR_TEST_RENDER_CHECK_KEY = "lidar/test"
# job_info.txt line of a render-only job whose renders equal its selection (render_only.sbatch).
RERENDER_CHECK_MATCH_LINE = "render_check: match"
# Dataset and session locations relative to the project root.
DATASETS_RELATIVE_DIR = Path("data") / "intermediate"
SESSION_CONFIGS_RELATIVE_DIR = Path("configs") / "sessions"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
MANIFEST_FILE_NAME = "manifest.json"
# GT files written by scripts/05_extract_cuboids.py.
CUBOIDS_FILE_NAME = "cuboids.json"
OCCUPANCY_FILE_NAME = "occupancy.npz"
# Output location inside the evaluated folder.
EVAL_OUTPUT_DIR_NAME = "eval"
REPORT_JSON_NAME = "semantic_occupancy.json"
REPORT_MARKDOWN_NAME = "semantic_occupancy.md"
# Pictures of the scored poses (scripts/15) next to the report, and the command that installs their environment.
VIZ_OUTPUT_DIR_NAME = "viz"
VIZ_ENVIRONMENT_FIX_COMMAND = "uv sync --project viz --locked"
# Longest the pictures may take before the evaluation gives up on them. A hardcoded safety net, not a
# measured value: a cold `uv run` may first resolve and install the viz environment, which can take
# minutes, so 10 minutes leaves room for that while stopping a hung off-screen render.
VIZ_TIMEOUT_SECONDS = 600
# Point sets scored, binding one first (PLAN §5 P1: ungated is binding, gated reported beside it).
POINT_SET_NAMES = ("ungated", "gated")
# Smallest possible T45 threshold (one point), used only when the real sweeps hit no box of a class.
MINIMUM_POSSIBLE_POINT_COUNT = 1
# Markdown header of the per-class table (columns in the order class_table_rows writes them).
CLASS_TABLE_HEADER = [
    "| class | cuboids | n_min | detection rate | point precision (in/nbhd) | voxel IoU | voxel P | voxel R |",
    "|---|---|---|---|---|---|---|---|",
]


def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dirs", nargs="+", type=Path, help="fetched run folders (runs/<run_id>)")
    parser.add_argument("--rerender", default=None, help="score rerenders/<label> of the (single) run instead")
    parser.add_argument("--eval-config", type=Path, default=None, help="default: configs/eval.yaml")
    parser.add_argument("--accumulation-threshold", type=float, default=None,
                        help="override eval.render_accumulation_threshold (recorded in the report)")
    parser.add_argument("--no-viz", action="store_true", help="do not draw the PyVista pictures (scripts/15)")
    parser.add_argument("--viz-classes", default=None,
                        help="comma-separated scored class names scripts/15 draws (default: eval.viz_classes, else all scored)")
    parser.add_argument("--viz-radius-meters", type=float, default=None,
                        help="horizontal radius around each sweep drawn by scripts/15 (default: eval.viz_radius_meters)")
    return parser


def file_sha256(path: Path) -> str:
    """Hex sha256 of a file's bytes.

    Args:
        path: File.

    Returns:
        Digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lidar_test_stems(split_record: Dict[str, Any], frame_indices: Optional[Sequence[int]] = None) -> List[str]:
    """``<idx>_<ts_ns>`` stems of the run's lidar frames to score.

    Args:
        split_record: Parsed split.json.
        frame_indices: For a render-only job, its selected lidar frame indices; None = split ``test``.

    Returns:
        Stems, sorted.
    """
    lidar_frames = [frame for frame in split_record["frames"] if frame["kind"] == "lidar"]
    if frame_indices is None:
        chosen_frames = [frame for frame in lidar_frames if frame["split"] == "test"]
    else:
        wanted_indices = set(frame_indices)
        chosen_frames = [frame for frame in lidar_frames if frame["frame_index"] in wanted_indices]
    return sorted(Path(frame["file"]).stem for frame in chosen_frames)


def resolve_render_target(run_dir: Path, rerender_label: Optional[str]) -> Tuple[Path, Path, List[str], Dict[str, Any]]:
    """Folder to score, its lidar renders and the frames they must contain.

    Args:
        run_dir: Fetched run folder.
        rerender_label: ``rerenders/<label>`` to score instead of the run's own renders, or None.

    Returns:
        (eval root, lidar render dir, expected stems, split record).

    Raises:
        ValueError: Missing split.json, a failed render check, or missing renders.
    """
    # Step 1: the split record documents every successful run (scripts/09).
    split_path = run_dir / SPLIT_RECORD_FILE_NAME
    if not split_path.is_file():
        raise ValueError(f"{split_path} missing: run scripts/09_extract_run_split.py on a successful run first")
    split_record = json.loads(split_path.read_text())
    if rerender_label is None:
        # Step 2a: the run's own test renders, checked by scripts/09 against the split.
        if not split_record["render_check"].get(LIDAR_TEST_RENDER_CHECK_KEY, {}).get("match"):
            raise ValueError(f"{split_path}: {LIDAR_TEST_RENDER_CHECK_KEY} renders do not match the split")
        eval_root = run_dir
        expected_stems = lidar_test_stems(split_record)
    else:
        # Step 2b: a render-only job; its own check line and selection.json name the frames.
        eval_root = run_dir / RERENDERS_DIRECTORY_NAME / rerender_label
        job_info_lines = (eval_root / JOB_INFO_FILE_NAME).read_text().splitlines()
        if RERENDER_CHECK_MATCH_LINE not in job_info_lines:
            raise ValueError(f"{eval_root}: no '{RERENDER_CHECK_MATCH_LINE}' in {JOB_INFO_FILE_NAME}")
        selection = json.loads((eval_root / SELECTION_FILE_NAME).read_text())
        expected_stems = lidar_test_stems(split_record, selection["lidar"])
    render_dir = eval_root / LIDAR_RENDER_SUBDIR
    if not render_dir.is_dir():
        raise ValueError(f"{render_dir} missing: this run has no lidar renders")
    return eval_root, render_dir, expected_stems, split_record


def resolve_session(split_record: Dict[str, Any], project_root: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Session config and dataset manifest of a run.

    Args:
        split_record: Parsed split.json.
        project_root: Repo root.

    Returns:
        (session config from load_session_config, parsed manifest).

    Raises:
        ValueError: If the local dataset is not the one the run trained on (manifest hash differs).
    """
    dataset_name = split_record["dataset"]["name"]
    manifest_path = project_root / DATASETS_RELATIVE_DIR / dataset_name / MANIFEST_FILE_NAME
    if file_sha256(manifest_path) != split_record["dataset"]["manifest_sha256"]:
        raise ValueError(f"{manifest_path} differs from the run's dataset (split.json manifest_sha256)")
    manifest = json.loads(manifest_path.read_text())
    # The manifest's session (not the dataset name) picks the labels: smoke datasets share their session's.
    session_config = load_session_config(project_root / SESSION_CONFIGS_RELATIVE_DIR / f"{manifest['session']['name']}.yaml")
    return session_config, manifest


def load_ground_truth(gt_dir: Path) -> Tuple[list, Dict[str, np.ndarray]]:
    """GT cuboids and occupancy grid of a session.

    Args:
        gt_dir: ``data/gt/<session>``.

    Returns:
        (cuboids, occupancy arrays: class_ids, origin, voxel_size).

    Raises:
        ValueError: If the GT has not been extracted.
    """
    cuboids_path, occupancy_path = gt_dir / CUBOIDS_FILE_NAME, gt_dir / OCCUPANCY_FILE_NAME
    if not (cuboids_path.is_file() and occupancy_path.is_file()):
        raise ValueError(f"{gt_dir}: GT missing; run scripts/05_extract_cuboids.py --config <session yaml>")
    with np.load(occupancy_path) as occupancy_arrays:
        occupancy = {key: occupancy_arrays[key] for key in occupancy_arrays.files}
    return load_cuboids_json(cuboids_path), occupancy


def pooled_map_points(sweeps: Sequence[RenderedSweep], point_set_name: str, grid: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Pool one point set of every sweep in the map frame, with its GT-grid voxels (T68).

    Args:
        sweeps: Rendered sweeps.
        point_set_name: ``gated``, ``ungated`` or ``real``.
        grid: GT occupancy arrays.

    Returns:
        {"points" (N, 3), "occupied" bool grid, "outside_grid" int}.
    """
    grid_shape = grid["class_ids"].shape
    point_attribute = {"gated": "gated_points", "ungated": "ungated_points", "real": "real_points"}[point_set_name]
    map_points, voxel_indices, inside_masks = [], [], []
    # Step 1: each sweep's points to the map frame (p_map = lidar_to_map @ p) and their GT-grid voxels.
    for sweep in sweeps:
        sweep_map_points, sweep_voxel_indices, sweep_inside_grid = points_to_map_voxels(
            getattr(sweep, point_attribute), sweep.lidar_to_map, grid["origin"], float(grid["voxel_size"]), grid_shape
        )
        map_points.append(sweep_map_points)
        voxel_indices.append(sweep_voxel_indices)
        inside_masks.append(sweep_inside_grid)
    # Step 2: pool over sweeps; voxels outside the grid are counted, not marked occupied.
    all_indices, all_inside = np.concatenate(voxel_indices), np.concatenate(inside_masks)
    return {
        "points": np.concatenate(map_points),
        "occupied": occupied_voxel_grid(all_indices, all_inside, grid_shape),
        "outside_grid": int(np.count_nonzero(~all_inside)),
    }


def build_class_setup(scored_class_ids: Dict[str, int], visible_by_class: Dict[int, list], margins: Dict[int, Optional[float]],
                      real_points_map: np.ndarray, grid: Dict[str, np.ndarray], sweep_poses: Sequence[np.ndarray],
                      max_range: float, vertical_half_fov: float, n_min_quantile: float) -> Dict[str, Dict[str, Any]]:
    """Per-class GT set-up shared by every point set: n_min, margin, neighbourhood and GT voxels.

    Args:
        scored_class_ids: Class name -> id of the scored classes.
        visible_by_class: Class id -> its scored (visible) cuboids.
        margins: Class id -> neighbourhood margin in metres (None if the session has no box of the class).
        real_points_map: Pooled real test-sweep points in the map frame (source of n_min).
        grid: GT occupancy arrays.
        sweep_poses: lidar_to_map of every test sweep.
        max_range: Scored range in metres.
        vertical_half_fov: Scored vertical half field of view in radians.
        n_min_quantile: Quantile of the real per-box point counts that sets n_min.

    Returns:
        Class name -> {class_id, cuboids, margin, n_min, n_min_source, neighbourhood, ground_truth}.
    """
    grid_origin, voxel_size = np.asarray(grid["origin"], dtype=np.float64), float(grid["voxel_size"])
    grid_shape = grid["class_ids"].shape
    class_setup: Dict[str, Dict[str, Any]] = {}
    for class_name, class_id in scored_class_ids.items():
        class_cuboids = visible_by_class[class_id]
        # T45 threshold: a quantile of how many real points each scored box of the class received.
        n_min = n_min_from_real_counts(cuboid_point_counts(real_points_map, class_cuboids), n_min_quantile)
        n_min_source = f"quantile {n_min_quantile} of real test-sweep counts"
        if n_min is None:
            n_min, n_min_source = MINIMUM_POSSIBLE_POINT_COUNT, "no real hit: minimum possible count"
        # No box of the class in the session (margin None): no growth, the neighbourhood is the boxes.
        margin = margins[class_id] or 0.0
        # Voxels within the margin of the class's boxes and within range/FOV of a test pose; GT voxels are
        # the class's own voxels inside that neighbourhood.
        neighbourhood = visible_neighbourhood_mask(
            class_cuboids, margin, grid_origin, voxel_size, grid_shape, sweep_poses, max_range, vertical_half_fov)
        class_setup[class_name] = {
            "class_id": class_id, "cuboids": class_cuboids, "margin": margin, "n_min": n_min,
            "n_min_source": n_min_source, "neighbourhood": neighbourhood,
            "ground_truth": (grid["class_ids"] == class_id) & neighbourhood,
        }
    return class_setup


def score_point_set(sweeps: Sequence[RenderedSweep], point_set_name: str, grid: Dict[str, np.ndarray],
                    class_setup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Score one pooled point set for every class (T45 detection, T46 precision, voxel IoU/P/R).

    Args:
        sweeps: Rendered test sweeps.
        point_set_name: ``gated`` or ``ungated``.
        grid: GT occupancy arrays.
        class_setup: Output of ``build_class_setup``.

    Returns:
        The point set's report entry; ``{"available": False, ...}`` when the ungated set cannot be built.
    """
    # Older renders lack the accumulation array, so the ungated set cannot be built: record that, score the rest.
    if point_set_name == "ungated" and any(sweep.ungated_points is None for sweep in sweeps):
        return {"available": False, "reason": "lidar npz has no pred_accumulation"}
    # Step 1: pool the point set over all sweeps in the map frame (shared by every class).
    pooled = pooled_map_points(sweeps, point_set_name, grid)
    class_reports = {}
    # Step 2: per class, the three metric families against that class's set-up.
    for class_name, setup in class_setup.items():
        # T45: share of scored boxes holding at least n_min points.
        detection = cuboid_detection_rate(cuboid_point_counts(pooled["points"], setup["cuboids"]), setup["n_min"])
        # T46: points inside the boxes over points inside the margin-grown neighbourhood.
        precision = neighbourhood_point_precision(pooled["points"], setup["cuboids"], setup["margin"])
        # Voxel IoU/P/R of the occupied-voxel grid against the class's GT voxels, within the neighbourhood.
        voxel_metrics = neighbourhood_voxel_metrics(pooled["occupied"], setup["ground_truth"], setup["neighbourhood"])
        class_reports[class_name] = {
            "detection": {"cuboid_count": detection.cuboid_count, "n_min": detection.n_min,
                          "detection_rate": detection.detection_rate,
                          "rate_by_min_points": detection.rate_by_min_points},
            "point_precision": {"inside_count": precision.inside_count,
                                "neighbourhood_count": precision.neighbourhood_count,
                                "precision": precision.precision},
            "voxel": summarise_metrics(voxel_metrics),
        }
    return {
        "available": True, "point_count": int(pooled["points"].shape[0]),
        "outside_grid_count": pooled["outside_grid"], "classes": class_reports,
    }


def evaluate_run(run_dir: Path, rerender_label: Optional[str], eval_settings: Dict[str, Any],
                 accumulation_threshold_override: Optional[float], project_root: Path) -> Tuple[Path, Dict[str, Any]]:
    """Score one run (or render-only job) and build its report.

    Args:
        run_dir: Fetched run folder.
        rerender_label: Render-only job label, or None.
        eval_settings: ``eval`` section of configs/eval.yaml.
        accumulation_threshold_override: CLI override of the T67 threshold, or None.
        project_root: Repo root.

    Returns:
        (eval root, report dict).

    Raises:
        ValueError: Refused input (missing or mismatched split record, renders, dataset or GT).
    """
    # Step 1: inputs, all derived from the run's records.
    eval_root, render_dir, expected_stems, split_record = resolve_render_target(run_dir, rerender_label)
    session_config, manifest = resolve_session(split_record, project_root)
    gt_dir = Path(session_config["session"]["gt_dir"])
    cuboids, grid = load_ground_truth(gt_dir)
    if accumulation_threshold_override is None:
        accumulation_threshold, threshold_source = float(eval_settings["render_accumulation_threshold"]), "configs/eval.yaml"
    else:
        accumulation_threshold, threshold_source = accumulation_threshold_override, "--accumulation-threshold"
    sweeps = load_test_lidar_sweeps(render_dir, expected_stems, accumulation_threshold)

    # Step 2: scored region = within range/FOV of a test pose (both limits from the data).
    sweep_poses = [sweep.lidar_to_map for sweep in sweeps]
    scored_class_ids = dict(eval_settings["scored_class_ids"])
    class_ids = list(scored_class_ids.values())
    # The same helper scripts/15 calls, so the drawn boxes are the counted boxes.
    visible_by_class, max_range, vertical_half_fov = scored_region_cuboids(
        cuboids, class_ids, sweep_poses, [sweep.real_points for sweep in sweeps], manifest["lidar"]["elevation_mapping"]
    )

    # Step 3: per-class GT set-up shared by both point sets.
    real_points_map = pooled_map_points(sweeps, "real", grid)["points"]
    n_min_quantile = float(eval_settings["n_min_quantile"])
    class_setup = build_class_setup(
        scored_class_ids, visible_by_class, class_margin_table(cuboids, class_ids), real_points_map, grid,
        sweep_poses, max_range, vertical_half_fov, n_min_quantile)

    # Step 4: score each point set.
    point_set_reports = {name: score_point_set(sweeps, name, grid, class_setup) for name in POINT_SET_NAMES}

    # Step 5: the report, with every derived setting and its source.
    report = {
        "schema": REPORT_SCHEMA,
        "run_id": split_record["run"]["run_id"],
        "rerender_label": rerender_label,
        "session": session_config["session"]["name"],
        "labels_sample_key": session_config["session"]["labels_sample_key"],
        "dataset": split_record["dataset"]["name"],
        "ground_truth": {
            "dir": str(gt_dir.relative_to(project_root)),
            "cuboids_sha256": file_sha256(gt_dir / CUBOIDS_FILE_NAME),
            "occupancy_sha256": file_sha256(gt_dir / OCCUPANCY_FILE_NAME),
            "voxel_size_m": float(grid["voxel_size"]),
        },
        "sweep_count": len(sweeps),
        "settings": {
            "accumulation_threshold": accumulation_threshold, "accumulation_threshold_source": threshold_source,
            "n_min_quantile": n_min_quantile,
            "max_range_m": max_range, "max_range_source": "farthest real return of the scored sweeps",
            "vertical_half_fov_rad": vertical_half_fov, "vertical_half_fov_source": "manifest lidar.elevation_mapping",
        },
        "classes": {
            class_name: {"class_id": setup["class_id"], "scored_cuboids": len(setup["cuboids"]),
                         "margin_m": setup["margin"], "n_min": setup["n_min"], "n_min_source": setup["n_min_source"],
                         "ground_truth_voxels": int(np.count_nonzero(setup["ground_truth"]))}
            for class_name, setup in class_setup.items()
        },
        "point_sets": point_set_reports,
    }
    return eval_root, report


def format_ratio(value: Optional[float]) -> str:
    """Three-decimal string, or "n/a" for None.

    Args:
        value: Ratio.

    Returns:
        Text.
    """
    return "n/a" if value is None else f"{value:.3f}"


def class_table_rows(point_set_report: Dict[str, Any]) -> List[str]:
    """Markdown rows (one per class) of one point set.

    Args:
        point_set_report: ``report["point_sets"][name]`` (available).

    Returns:
        Table rows.
    """
    rows = []
    for class_name, class_report in point_set_report["classes"].items():
        detection, precision, voxel = class_report["detection"], class_report["point_precision"], class_report["voxel"]
        rows.append(
            f"| {class_name} | {detection['cuboid_count']} | {detection['n_min']} | {format_ratio(detection['detection_rate'])} | "
            f"{format_ratio(precision['precision'])} ({precision['inside_count']}/{precision['neighbourhood_count']}) | "
            f"{format_ratio(voxel['iou'])} | {format_ratio(voxel['precision'])} | {format_ratio(voxel['recall'])} |"
        )
    return rows


def report_markdown(report: Dict[str, Any]) -> str:
    """Human-readable report.

    Args:
        report: ``evaluate_run`` report.

    Returns:
        Markdown text.
    """
    settings = report["settings"]
    lines = [
        f"# Semantic occupancy: {report['run_id']}" + (f" / {report['rerender_label']}" if report["rerender_label"] else ""),
        "",
        f"- Session `{report['session']}` (labels sample `{report['labels_sample_key']}`), dataset `{report['dataset']}`, "
        f"{report['sweep_count']} sweeps.",
        f"- GT `{report['ground_truth']['dir']}`, voxel {report['ground_truth']['voxel_size_m']:.4f} m.",
        f"- Accumulation threshold {settings['accumulation_threshold']} ({settings['accumulation_threshold_source']}); "
        f"range {settings['max_range_m']:.1f} m and vertical half FOV {np.degrees(settings['vertical_half_fov_rad']):.1f}° (derived).",
        "- Neighbourhood margins (m): " + ", ".join(
            f"{class_name} {class_info['margin_m']:.2f}" for class_name, class_info in report["classes"].items()),
        "",
    ]
    for point_set_name, point_set_report in report["point_sets"].items():
        binding_note = " (binding)" if point_set_name == POINT_SET_NAMES[0] else ""
        lines.append(f"## {point_set_name}{binding_note}")
        if not point_set_report["available"]:
            lines += [f"Not available: {point_set_report['reason']}.", ""]
            continue
        lines += [f"{point_set_report['point_count']} points, {point_set_report['outside_grid_count']} outside the GT grid.", ""]
        lines += CLASS_TABLE_HEADER + class_table_rows(point_set_report) + [""]
    return "\n".join(lines)


def comparison_markdown(reports: Sequence[Dict[str, Any]]) -> str:
    """Side-by-side binding (first available) point-set table of several runs of one session.

    Args:
        reports: Reports of one session.

    Returns:
        Markdown text.

    Raises:
        ValueError: If the runs differ in session, labels sample or GT cuboids (cuboids.json sha256).
    """
    session_keys = {(report["session"], report["labels_sample_key"], report["ground_truth"]["cuboids_sha256"])
                    for report in reports}
    if len(session_keys) != 1:
        raise ValueError(f"runs from different sessions/GT cannot be compared: {sorted(session_keys)}")
    lines = [f"# Comparison ({reports[0]['session']}, labels {reports[0]['labels_sample_key']})", ""]
    for report in reports:
        point_set_name = next(name for name in POINT_SET_NAMES if report["point_sets"][name]["available"])
        lines.append(f"## {report['run_id']}" + (f" / {report['rerender_label']}" if report["rerender_label"] else "")
                     + f" ({point_set_name})")
        lines += CLASS_TABLE_HEADER + class_table_rows(report["point_sets"][point_set_name]) + [""]
    return "\n".join(lines)


def repo_relative(text: str, project_root: Path) -> str:
    """Replace the absolute repo root by ``<repo>`` and the home directory by ``~`` in a message.

    The report is meant to be shareable, and CLAUDE.md rule 5 keeps account names (which a home
    directory contains) out of recorded output.

    Args:
        text: A path or message.
        project_root: Repo root.

    Returns:
        ``text`` with the root replaced.
    """
    text = text.replace(str(Path(project_root).resolve()), "<repo>").replace(str(project_root), "<repo>")
    # Any other path under the home directory (e.g. the uv cache) would still carry the account name.
    return text.replace(str(Path.home()), "~")


def launch_pose_viz(run_dir: Path, rerender_label: Optional[str], eval_output_dir: Path, eval_config_path: Path,
                    accumulation_threshold: float, project_root: Path, run_command=subprocess.run,
                    viz_radius_meters: Optional[float] = None, viz_class_names: Optional[List[str]] = None) -> str:
    """Draw the scored poses with scripts/15 in the isolated viz/ environment.

    Never raises: the report is already written, so a drawing problem must not touch the metrics.
    The command runs from the repo root because ``viz_command`` uses repo-relative script and
    project paths; every user path in it is made absolute here.

    Args:
        run_dir: Fetched run folder.
        rerender_label: Render-only job label, or None.
        eval_output_dir: ``<eval root>/eval``; the pictures go to its ``viz`` subfolder.
        eval_config_path: configs/eval.yaml the evaluation used.
        accumulation_threshold: T67 threshold the evaluation used.
        project_root: Repo root (working directory of the subprocess).
        run_command: ``subprocess.run`` stand-in, injectable for tests.
        viz_radius_meters: Radius override for the pictures, or None for the config value.
        viz_class_names: Class names to draw, or None for the config value (all scored classes).

    Returns:
        "written: <folder>" or "failed: <reason>".
    """
    viz_output_dir = (eval_output_dir / VIZ_OUTPUT_DIR_NAME).resolve()
    command = viz_command([Path(run_dir).resolve()], viz_output_dir, Path(eval_config_path).resolve(),
                          accumulation_threshold, rerender_label, viz_radius_meters, viz_class_names)
    # Step 1: run the viewer; a missing launcher (uv) is a failure to report, not a crash.
    try:
        # errors="replace": undecodable viewer output must not raise out of this never-raises helper.
        completed = run_command(command, cwd=project_root, capture_output=True, text=True, errors="replace",
                                timeout=VIZ_TIMEOUT_SECONDS)
    except (OSError, ValueError) as error:
        return f"failed: could not start {command[0]}: {repo_relative(str(error), project_root)}"
    except subprocess.TimeoutExpired:
        return f"failed: no result within {VIZ_TIMEOUT_SECONDS} s"
    # Step 2: a non-zero exit carries the viewer's last stderr line as the reason.
    if completed.returncode != 0:
        stderr_lines = (completed.stderr or "").strip().splitlines()
        last_line = repo_relative(stderr_lines[-1], project_root) if stderr_lines else "no error output"
        return f"failed: exit code {completed.returncode}: {last_line}"
    return f"written: {repo_relative(str(viz_output_dir), project_root)}"


def main(argument_list: Optional[List[str]] = None, run_command=subprocess.run) -> int:
    """Evaluate each given run and write its report.

    Args:
        argument_list: CLI arguments (default: sys.argv).
        run_command: ``subprocess.run`` stand-in for the viz launch, injectable for tests.

    Returns:
        Exit code: 0 success, 1 refused input (a viz failure does not change it).
    """
    arguments = build_argument_parser().parse_args(argument_list)
    project_root = find_project_root()
    # Step 1: a rerender label names one run's folder, so it cannot be combined with several runs.
    if arguments.rerender is not None and len(arguments.run_dirs) != 1:
        print("--rerender takes exactly one run folder", file=sys.stderr)
        return 1
    # Step 2: eval settings and the viz class filter.
    eval_config_path = arguments.eval_config or project_root / DEFAULT_EVAL_CONFIG_RELATIVE_PATH
    eval_settings = yaml.safe_load(Path(eval_config_path).read_text())["eval"]
    reports = []
    # A misspelt class name is caught now, before the evaluation runs, not inside the viewer afterwards.
    viz_class_names = None if arguments.viz_classes is None else [name.strip() for name in arguments.viz_classes.split(",")]
    unknown_class_names = [name for name in (viz_class_names or []) if name not in eval_settings["scored_class_ids"]]
    if unknown_class_names:
        print(f"--viz-classes {unknown_class_names} are not scored classes (have {sorted(eval_settings['scored_class_ids'])})", file=sys.stderr)
        return 1
    # Step 3: per run, score and write the report, then draw; any refused input aborts with exit 1.
    try:
        for run_dir in arguments.run_dirs:
            # One run: score, then write json + md next to its renders.
            eval_root, report = evaluate_run(run_dir, arguments.rerender, eval_settings,
                                             arguments.accumulation_threshold, project_root)
            output_dir = eval_root / EVAL_OUTPUT_DIR_NAME
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / REPORT_JSON_NAME).write_text(json.dumps(report, indent=1, sort_keys=True))
            (output_dir / REPORT_MARKDOWN_NAME).write_text(report_markdown(report))
            print(f"wrote {output_dir / REPORT_JSON_NAME}")
            # Visual aid, only after the metrics are safely on disk; its outcome is recorded beside them.
            if arguments.no_viz:
                report["viz"] = "skipped: --no-viz"
            else:
                report["viz"] = launch_pose_viz(run_dir, arguments.rerender, output_dir, eval_config_path,
                                                report["settings"]["accumulation_threshold"], project_root, run_command,
                                                arguments.viz_radius_meters, viz_class_names)
                if report["viz"].startswith("failed"):
                    print(f"WARNING: no pictures ({report['viz']}). Set up their environment with: "
                          f"{VIZ_ENVIRONMENT_FIX_COMMAND}", file=sys.stderr)
                else:
                    print(report["viz"])
            # Rewrite the JSON so it also carries the "viz" outcome (the first write kept the metrics safe).
            (output_dir / REPORT_JSON_NAME).write_text(json.dumps(report, indent=1, sort_keys=True))
            reports.append(report)
        # Step 4: print one report, or the side-by-side table for several runs.
        if len(reports) > 1:
            print(comparison_markdown(reports))
        else:
            print(report_markdown(reports[0]))
    except ValueError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
