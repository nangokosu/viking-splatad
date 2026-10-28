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

Usage:
    uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> [runs/<run_id> ...]
    uv run python scripts/13_eval_semantic_occupancy.py runs/<run_id> --rerender render-<...>
"""

import argparse
import hashlib
import json
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
    cuboid_voxel_mask,
    grow_cuboid_horizontally,
    load_cuboids_json,
    n_min_from_real_counts,
    neighbourhood_point_precision,
    neighbourhood_voxel_metrics,
    range_limit_from_real_points,
    restrict_mask_to_visible,
    scored_cuboids_by_class,
    summarise_metrics,
    vertical_half_fov_from_elevations,
    visible_cuboid_mask,
)
from viking_splatad.eval.rendered_pointcloud import (
    RenderedSweep,
    load_test_lidar_sweeps,
    occupied_voxel_grid,
    points_to_map_voxels,
)

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
# Point sets scored, binding one first (PLAN §5 P1: ungated is binding, gated reported beside it).
POINT_SET_NAMES = ("ungated", "gated")
# Smallest possible T45 threshold (one point), used only when the real sweeps hit no box of a class.
MINIMUM_POSSIBLE_POINT_COUNT = 1


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
    for sweep in sweeps:
        points_map, indices, inside_grid = points_to_map_voxels(
            getattr(sweep, point_attribute), sweep.lidar_to_map, grid["origin"], float(grid["voxel_size"]), grid_shape
        )
        map_points.append(points_map)
        voxel_indices.append(indices)
        inside_masks.append(inside_grid)
    all_indices, all_inside = np.concatenate(voxel_indices), np.concatenate(inside_masks)
    return {
        "points": np.concatenate(map_points),
        "occupied": occupied_voxel_grid(all_indices, all_inside, grid_shape),
        "outside_grid": int(np.count_nonzero(~all_inside)),
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
    """
    # Step 1: inputs, all derived from the run's records.
    eval_root, render_dir, expected_stems, split_record = resolve_render_target(run_dir, rerender_label)
    session_config, manifest = resolve_session(split_record, project_root)
    gt_dir = Path(session_config["session"]["gt_dir"])
    cuboids, grid = load_ground_truth(gt_dir)
    grid_origin, voxel_size = np.asarray(grid["origin"], dtype=np.float64), float(grid["voxel_size"])
    grid_shape = grid["class_ids"].shape
    if accumulation_threshold_override is None:
        accumulation_threshold, threshold_source = float(eval_settings["render_accumulation_threshold"]), "configs/eval.yaml"
    else:
        accumulation_threshold, threshold_source = accumulation_threshold_override, "--accumulation-threshold"
    sweeps = load_test_lidar_sweeps(render_dir, expected_stems, accumulation_threshold)

    # Step 2: scored region = within range/FOV of a test pose (both limits from the data).
    sweep_poses = [sweep.lidar_to_map for sweep in sweeps]
    max_range = range_limit_from_real_points([sweep.real_points for sweep in sweeps])
    vertical_half_fov = vertical_half_fov_from_elevations(manifest["lidar"]["elevation_mapping"])
    scored_class_ids = dict(eval_settings["scored_class_ids"])
    class_ids = list(scored_class_ids.values())
    visible_by_class = scored_cuboids_by_class(
        cuboids, class_ids, visible_cuboid_mask(cuboids, sweep_poses, max_range, vertical_half_fov)
    )
    margins = class_margin_table(cuboids, class_ids)

    # Step 3: per-class GT set-up shared by both point sets (n_min from the real sweeps, voxel region).
    real_points_map = pooled_map_points(sweeps, "real", grid)["points"]
    class_setup: Dict[str, Dict[str, Any]] = {}
    for class_name, class_id in scored_class_ids.items():
        class_cuboids = visible_by_class[class_id]
        n_min = n_min_from_real_counts(cuboid_point_counts(real_points_map, class_cuboids), float(eval_settings["n_min_quantile"]))
        n_min_source = f"quantile {eval_settings['n_min_quantile']} of real test-sweep counts"
        if n_min is None:
            n_min, n_min_source = MINIMUM_POSSIBLE_POINT_COUNT, "no real hit: minimum possible count"
        margin = margins[class_id] or 0.0
        neighbourhood = restrict_mask_to_visible(
            cuboid_voxel_mask([grow_cuboid_horizontally(cuboid, margin) for cuboid in class_cuboids],
                              grid_origin, voxel_size, grid_shape),
            grid_origin, voxel_size, sweep_poses, max_range, vertical_half_fov,
        )
        class_setup[class_name] = {
            "class_id": class_id, "cuboids": class_cuboids, "margin": margin, "n_min": n_min,
            "n_min_source": n_min_source, "neighbourhood": neighbourhood,
            "ground_truth": (grid["class_ids"] == class_id) & neighbourhood,
        }

    # Step 4: score each point set.
    point_set_reports: Dict[str, Any] = {}
    for point_set_name in POINT_SET_NAMES:
        if point_set_name == "ungated" and any(sweep.ungated_points is None for sweep in sweeps):
            point_set_reports[point_set_name] = {"available": False, "reason": "lidar npz has no pred_accumulation"}
            continue
        pooled = pooled_map_points(sweeps, point_set_name, grid)
        class_reports = {}
        for class_name, setup in class_setup.items():
            detection = cuboid_detection_rate(cuboid_point_counts(pooled["points"], setup["cuboids"]), setup["n_min"])
            precision = neighbourhood_point_precision(pooled["points"], setup["cuboids"], setup["margin"])
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
        point_set_reports[point_set_name] = {
            "available": True, "point_count": int(pooled["points"].shape[0]),
            "outside_grid_count": pooled["outside_grid"], "classes": class_reports,
        }

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
            "voxel_size_m": voxel_size,
        },
        "sweep_count": len(sweeps),
        "settings": {
            "accumulation_threshold": accumulation_threshold, "accumulation_threshold_source": threshold_source,
            "n_min_quantile": float(eval_settings["n_min_quantile"]),
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


CLASS_TABLE_HEADER = [
    "| class | cuboids | n_min | detection rate | point precision (in/nbhd) | voxel IoU | voxel P | voxel R |",
    "|---|---|---|---|---|---|---|---|",
]


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
        ValueError: If the runs come from different sessions or labels samples.
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


def main(argument_list: Optional[List[str]] = None) -> int:
    """Evaluate each given run and write its report.

    Args:
        argument_list: CLI arguments (default: sys.argv).

    Returns:
        Exit code: 0 success, 1 refused input.
    """
    arguments = build_argument_parser().parse_args(argument_list)
    project_root = find_project_root()
    if arguments.rerender is not None and len(arguments.run_dirs) != 1:
        print("--rerender takes exactly one run folder", file=sys.stderr)
        return 1
    eval_config_path = arguments.eval_config or project_root / DEFAULT_EVAL_CONFIG_RELATIVE_PATH
    eval_settings = yaml.safe_load(Path(eval_config_path).read_text())["eval"]
    reports = []
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
            reports.append(report)
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
