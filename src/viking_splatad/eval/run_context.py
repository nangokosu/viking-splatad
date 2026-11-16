"""Read a fetched run the way the semantic-occupancy evaluation does (scripts/13 and scripts/15).

Shared by the evaluation (scripts/13_eval_semantic_occupancy.py) and the PyVista viewer
(scripts/15_view_pose_pyvista.py), so both resolve the same renders, session, ground truth, pooled
map-frame points and per-class scoring set-up from a run's own records:

    1. ``resolve_render_target``: the folder to score, its lidar renders and the frames they must hold
       (``split.json``, or ``rerenders/<label>/selection.json`` for a render-only job).
    2. ``resolve_session``: the dataset manifest (hash-checked against split.json) and the session
       config whose GT folder and labels sample apply.
    3. ``load_ground_truth``: the session's GT cuboids and occupancy grid (``data/gt/<session>``).
    4. ``pooled_map_points``: one point set of every sweep in the map frame, with its GT-grid voxels (T68).
    5. ``build_class_setup``: per scored class, the T45 threshold, the neighbourhood and its GT voxels.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.config import load_session_config
from viking_splatad.eval.detection import (
    cuboid_point_counts,
    load_cuboids_json,
    n_min_from_real_counts,
    visible_neighbourhood_mask,
)
from viking_splatad.eval.rendered_pointcloud import (
    RenderedSweep,
    occupied_voxel_grid,
    points_to_map_voxels,
)

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
MANIFEST_FILE_NAME = "manifest.json"
# GT files written by scripts/05_extract_cuboids.py.
CUBOIDS_FILE_NAME = "cuboids.json"
OCCUPANCY_FILE_NAME = "occupancy.npz"
# Smallest possible T45 threshold (one point), used only when the real sweeps hit no box of a class.
MINIMUM_POSSIBLE_POINT_COUNT = 1
# RenderedSweep attribute holding each named point set.
POINT_SET_ATTRIBUTES = {"gated": "gated_points", "ungated": "ungated_points", "real": "real_points"}


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
        # The run's own renders: every lidar frame the split put in ``test``.
        chosen_frames = [frame for frame in lidar_frames if frame["split"] == "test"]
    else:
        # A render-only job: the lidar frames its selection names, whatever their split.
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
    # Step 3: the lidar renders themselves must exist.
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
    # Step 1: the local dataset must be byte-identical to the one the run trained on.
    dataset_name = split_record["dataset"]["name"]
    manifest_path = project_root / DATASETS_RELATIVE_DIR / dataset_name / MANIFEST_FILE_NAME
    if file_sha256(manifest_path) != split_record["dataset"]["manifest_sha256"]:
        raise ValueError(f"{manifest_path} differs from the run's dataset (split.json manifest_sha256)")
    manifest = json.loads(manifest_path.read_text())
    # Step 2: the manifest's session (not the dataset name) picks the labels: smoke datasets share their session's.
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
    # Step 1: both GT files must exist (scripts/05 writes them together).
    cuboids_path, occupancy_path = gt_dir / CUBOIDS_FILE_NAME, gt_dir / OCCUPANCY_FILE_NAME
    if not (cuboids_path.is_file() and occupancy_path.is_file()):
        raise ValueError(f"{gt_dir}: GT missing; run scripts/05_extract_cuboids.py --config <session yaml>")
    # Step 2: read every occupancy array into memory so the npz file can be closed.
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
    point_attribute = POINT_SET_ATTRIBUTES[point_set_name]
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
