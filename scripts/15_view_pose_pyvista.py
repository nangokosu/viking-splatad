"""Draw a run's test sweeps as three occupancy-voxel maps against the GT with PyVista (visual aid of scripts/13).

By default the sweeps are POOLED exactly as the report pools them: one occupied-voxel grid over all
test sweeps, one scoring neighbourhood, so for a single drawn class the recall and precision in the
captions are the report's voxel recall and precision. With several classes the captions count the union of
their cells, not the per-class report values. ``--pose`` draws single sweeps instead.
For each run, three PNGs ``pooled_overlay.png``, ``pooled_recall.png`` and ``pooled_precision.png``
(or ``<stem>_<map>.png`` per ``--pose``) show, in the map frame within ``--radius-meters``
(horizontal) of a test lidar position (of any pooled sweep), voxels of the selected classes:
    - overlay: GT voxels grey and every render voxel light blue (translucent, so the GT always shows);
    - recall: GT voxels green where the render also occupies the voxel (hit), grey if missed; the other
      render voxels are drawn very faint green as context;
    - precision: render voxels solid green on a GT voxel of a selected class, faint red off every one;
      no GT voxels.
All three maps show the whole rendered cloud, including the voxels outside the metric's scoring
neighbourhood of the selected classes (``render_outside``): they are drawn but never counted.
Dark wireframes mark the scored GT cuboids. The render voxels come from the run's own rendered
test-split lidar returns (ungated, the set the metric binds to, unless ``--gated``), pooled over the
drawn sweeps (all of them, or the ``--pose`` ones). The counted ones lie in the metric's scoring neighbourhood: each box grown horizontally by its class
margin, within the range and vertical FOV of the pooled sweeps (``visible_neighbourhood_mask``). The
cells are those of the voxel IoU / precision / recall in scripts/13, unioned over the selected classes;
the caption counts are over the whole scored region, the drawing is cut to the radius. A ``--pose`` map
is the same computation for one sweep and illustrates the report without being its numbers.
``--classes`` filters which scored classes are drawn.
``--show-points`` (with ``--pose``) adds the rendered and real returns to the overlay. Several ``--run`` folders give
side-by-side panels with linked cameras (e.g. B and R of one session).

Everything is derived as scripts/13 derives it, through the same functions: the run's split record,
session and GT, the T67 accumulation threshold, the scored region (``scored_region_cuboids``) and the
neighbourhood (``visible_neighbourhood_mask``). GT cuboids are display only (CLAUDE.md rule 6). The
run folders are read, never written; only ``--output-dir`` receives PNGs.

PyVista lives in the isolated viz/ project (the root environment has none):
    uv sync --project viz --locked
    uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<run_id> --output-dir <dir>
    uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<B> --run runs/<R> --pose 000529 --classes tree_trunk
Without ``--output-dir`` an interactive window opens for each map.
"""

import argparse
import functools
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from viking_splatad.config import find_project_root
from viking_splatad.eval.detection import (
    class_margin_table,
    cuboid_centers,
    cuboid_point_counts,
    grow_cuboid_horizontally,
    points_in_any_cuboid,
    restrict_mask_to_visible,
    scored_region_cuboids,
    visible_neighbourhood_mask,
    voxel_centers,
)
from viking_splatad.eval.rendered_pointcloud import (
    RenderedSweep,
    load_test_lidar_sweeps,
    occupied_voxel_grid,
    points_to_map_voxels,
)
from viking_splatad.labels.cuboids import Cuboid
from viking_splatad.viz.pyvista_scene import (
    CUBOID_WIREFRAME_RGB,
    GROUND_TRUTH_OVERLAY_OPACITY,
    GROUND_TRUTH_VOXEL_RGB,
    HIT_VOXEL_RGB,
    MAP_NAMES,
    NON_HIT_PRECISION_OPACITY,
    NON_HIT_VOXEL_RGB,
    REAL_POINT_RGB,
    RECALL_CONTEXT_OPACITY,
    RENDER_OVERLAY_OPACITY,
    RENDER_VOXEL_RGB,
    SOLID_VOXEL_OPACITY,
    cuboid_wireframe_arrays,
    inside_outside_colors,
)
from viking_splatad.viz.viewer_export import sweep_points_to_map

# File-name stem of the pooled maps (all test sweeps in one grid, as the report pools them).
POOLED_STEM = "pooled"
EVAL_SCRIPT_PATH = Path(__file__).resolve().parent / "13_eval_semantic_occupancy.py"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
# Camera choices: the default oblique 3D view, or straight down (map x-y plane).
VIEW_CHOICES = ("oblique", "top")
# Display sizes in pixels, a presentation choice with no data-derived value: the rendered points
# are drawn larger than the real ones so both stay visible when they overlap; one panel is
# 800 x 800 px so several side-by-side panels still fit a laptop screen.
REAL_POINT_SIZE, RENDERED_POINT_SIZE, WIREFRAME_LINE_WIDTH = 2.0, 4.0, 1.0
PANEL_PIXELS = 800
# Opacity of the real returns, so rendered points over them stay readable (presentation choice).
REAL_POINT_OPACITY = 0.5
# Share of ambient (direction-free) light on voxel faces, a presentation choice.
VOXEL_AMBIENT_LIGHT = 0.7


@dataclass
class RunScene:
    """Everything one run contributes to a picture.

    Attributes:
        run_id: Run id from split.json.
        sweeps: Test sweeps keyed by ``<idx>_<ts_ns>`` stem.
        scored_cuboids: GT boxes of the scored classes inside the scored region (the counted boxes).
        neighbourhood_cuboids: The same boxes grown horizontally by their class margin (T46 neighbourhood).
        membership_sha256: Hash of the run's train/test membership (split.json).
        gt_cuboids_sha256: Hash of the GT cuboids file the boxes came from.
        grid_origin: (3,) minimum corner of the GT occupancy grid (None: no voxel view).
        voxel_size: Grid voxel edge (m).
        ground_truth_class_ids: (nx, ny, nz) GT class id per voxel.
        class_neighbourhoods: Scored class id -> pooled scoring neighbourhood (bool grid), as in scripts/13.
        max_range: Range limit (m) of the scored region.
        vertical_half_fov: Half vertical FOV (rad) of the scored region.
        class_names: Scored class id -> name (for captions).
    """

    run_id: str
    sweeps: Dict[str, RenderedSweep]
    scored_cuboids: List[Cuboid]
    neighbourhood_cuboids: List[Cuboid]
    membership_sha256: str
    gt_cuboids_sha256: str
    grid_origin: Optional[np.ndarray] = None
    voxel_size: float = 0.0
    ground_truth_class_ids: Optional[np.ndarray] = None
    class_neighbourhoods: Dict[int, np.ndarray] = field(default_factory=dict)
    max_range: float = 0.0
    vertical_half_fov: float = 0.0
    class_names: Dict[int, str] = field(default_factory=dict)


def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line interface.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", dest="run_dirs", action="append", required=True, type=Path,
                        help="fetched run folder (runs/<run_id>); repeat for side-by-side panels")
    parser.add_argument("--rerender", default=None, help="draw rerenders/<label> of the (single) run instead")
    parser.add_argument("--pose", action="append", default=None,
                        help="draw this single sweep (stem or frame index, repeatable) instead of the pooled maps")
    parser.add_argument("--radius-meters", type=float, default=None,
                        help="horizontal radius around the test lidar positions to draw (default: eval.viz_radius_meters)")
    parser.add_argument("--classes", default=None,
                        help="comma-separated scored class names to draw (default: eval.viz_classes, else all scored)")
    parser.add_argument("--view", choices=VIEW_CHOICES, default=VIEW_CHOICES[0],
                        help="camera: oblique 3D or straight down (default: oblique)")
    parser.add_argument("--show-points", action="store_true",
                        help="with --pose: also draw the real (grey) and rendered (red inside a cuboid, blue otherwise) returns on the overlay")
    parser.add_argument("--all-points", action="store_true",
                        help="with --show-points: draw every point in the radius, not only the scored neighbourhood")
    parser.add_argument("--gated", action="store_true",
                        help="use the ray-drop-gated points instead of the ungated set the metric binds to")
    parser.add_argument("--output-dir", type=Path, default=None, help="write <pooled|stem>_<map>.png here; default: interactive window")
    parser.add_argument("--eval-config", type=Path, default=None, help="default: configs/eval.yaml")
    parser.add_argument("--accumulation-threshold", type=float, default=None,
                        help="override eval.render_accumulation_threshold (scripts/13 passes the value it used)")
    return parser


@functools.lru_cache(maxsize=None)
def load_eval_script():
    """Import scripts/13 as a module (its file name is not a valid module name).

    Returns:
        The loaded module (cached, so several runs share one import); its run, session and
        ground-truth resolvers are reused so both scripts read a run the same way.
    """
    module_spec = importlib.util.spec_from_file_location("eval_semantic_occupancy", EVAL_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def load_run_scene(run_dir: Path, rerender_label: Optional[str], eval_settings: Dict[str, Any],
                   accumulation_threshold: float, project_root: Path) -> RunScene:
    """Read a run the way scripts/13 does and pick the boxes and voxels the metric scores.

    Args:
        run_dir: Fetched run folder.
        rerender_label: Render-only job label, or None.
        eval_settings: ``eval`` section of configs/eval.yaml.
        accumulation_threshold: T67 threshold.
        project_root: Repo root.

    Returns:
        The run's scene.

    Raises:
        ValueError: Refused input (see scripts/13 ``evaluate_run``).
    """
    eval_script = load_eval_script()
    # Step 1: renders, session and GT, all derived from the run's own records.
    _, render_dir, expected_stems, split_record = eval_script.resolve_render_target(run_dir, rerender_label)
    session_config, manifest = eval_script.resolve_session(split_record, project_root)
    gt_dir = Path(session_config["session"]["gt_dir"])
    cuboids, grid = eval_script.load_ground_truth(gt_dir)
    grid_origin, voxel_size = np.asarray(grid["origin"], dtype=np.float64), float(grid["voxel_size"])
    sweeps = load_test_lidar_sweeps(render_dir, expected_stems, accumulation_threshold)
    sweep_poses = [sweep.lidar_to_map for sweep in sweeps]
    # Step 2: the scored boxes and region limits, from the same helper the metric uses.
    scored_class_ids = list(dict(eval_settings["scored_class_ids"]).values())
    visible_by_class, max_range, vertical_half_fov = scored_region_cuboids(
        cuboids, scored_class_ids, sweep_poses, [sweep.real_points for sweep in sweeps],
        manifest["lidar"]["elevation_mapping"],
    )
    scored_cuboids = [cuboid for class_cuboids in visible_by_class.values() for cuboid in class_cuboids]
    # Step 3: each class's pooled voxel neighbourhood, built as scripts/13 builds it (None margin: no growth).
    class_margins = class_margin_table(cuboids, scored_class_ids)
    class_neighbourhoods = {
        class_id: visible_neighbourhood_mask(
            visible_by_class[class_id], class_margins[class_id] or 0.0, grid_origin, voxel_size,
            grid["class_ids"].shape, sweep_poses, max_range, vertical_half_fov)
        for class_id in scored_class_ids
    }
    # Step 4: the grown boxes (for the optional point crop) with the same margins.
    neighbourhood_cuboids = [grow_cuboid_horizontally(cuboid, class_margins[cuboid.class_id] or 0.0)
                             for cuboid in scored_cuboids]
    return RunScene(
        run_id=split_record["run"]["run_id"], sweeps={sweep.stem: sweep for sweep in sweeps},
        scored_cuboids=scored_cuboids, neighbourhood_cuboids=neighbourhood_cuboids,
        membership_sha256=split_record["membership_sha256"],
        gt_cuboids_sha256=eval_script.file_sha256(gt_dir / eval_script.CUBOIDS_FILE_NAME),
        grid_origin=grid_origin, voxel_size=voxel_size, ground_truth_class_ids=np.asarray(grid["class_ids"]),
        class_neighbourhoods=class_neighbourhoods, max_range=max_range, vertical_half_fov=vertical_half_fov,
        class_names={class_id: class_name for class_name, class_id in dict(eval_settings["scored_class_ids"]).items()},
    )


def check_scenes_comparable(scenes: Sequence[RunScene]) -> None:
    """Refuse panels that do not show the same poses and ground truth.

    Args:
        scenes: One scene per run.

    Raises:
        ValueError: If the runs differ in train/test membership or GT cuboid file.
    """
    # Same membership hash = same test instants, so a stem means the same pose in every run.
    if len({scene.membership_sha256 for scene in scenes}) != 1:
        raise ValueError("runs have different train/test membership (split.json membership_sha256): not comparable")
    if len({scene.gt_cuboids_sha256 for scene in scenes}) != 1:
        raise ValueError("runs use different GT cuboids (cuboids.json sha256): not comparable")


def select_pose_stems(scenes: Sequence[RunScene], requested_poses: Optional[Sequence[str]]) -> List[str]:
    """Sweep stems to draw: those present in every run, narrowed to the requested poses.

    Args:
        scenes: One scene per run.
        requested_poses: Stems or frame indices (``000529`` or ``529``), or None for all.

    Returns:
        Stems: sorted when no poses are requested, otherwise in request order without duplicates.

    Raises:
        ValueError: If no sweep is shared by every run, or a requested pose matches none.
    """
    # Step 1: stems every run has rendered.
    common_stems = sorted(set.intersection(*(set(scene.sweeps) for scene in scenes)))
    if not common_stems:
        raise ValueError("the runs share no rendered sweep")
    if requested_poses is None:
        return common_stems
    # Step 2: a request matches a stem exactly or by its leading frame index (int compare, so zero padding is free).
    chosen_stems = []
    for requested in requested_poses:
        requested_is_index = requested.isascii() and requested.isdigit()
        matches = [stem for stem in common_stems
                   if stem == requested or (requested_is_index and stem.split("_")[0].isdigit()
                                            and int(stem.split("_")[0]) == int(requested))]
        if not matches:
            # Only the first few stems are listed: enough to show the naming, short enough for one line.
            raise ValueError(f"pose {requested!r} matches no shared sweep (have {common_stems[:5]}...)")
        chosen_stems += [stem for stem in matches if stem not in chosen_stems]
    return chosen_stems


def rendered_point_set(sweep: RenderedSweep, use_gated: bool) -> Tuple[np.ndarray, str]:
    """The rendered lidar-frame points to use: the ungated set the metric binds to, unless gated is asked for or missing.

    Args:
        sweep: Rendered test sweep.
        use_gated: Force the gated set.

    Returns:
        ((N, 3) lidar-frame points, "gated" or "ungated").
    """
    if use_gated or sweep.ungated_points is None:
        return sweep.gated_points, "gated"
    return sweep.ungated_points, "ungated"


def within_radius(points_xyz: np.ndarray, centre_xy: np.ndarray, radius_meters: float) -> np.ndarray:
    """Which points lie within a horizontal radius of a centre (height ignored).

    Args:
        points_xyz: (N, 3) map-frame points.
        centre_xy: (2,) map-frame centre, e.g. a sweep's lidar position.
        radius_meters: Radius (m), inclusive.

    Returns:
        (N,) bool.
    """
    # Horizontal distance: sqrt(dx^2 + dy^2) <= radius.
    return np.linalg.norm(np.asarray(points_xyz)[:, :2] - np.asarray(centre_xy), axis=1) <= radius_meters


def within_radius_of_any(points_xyz: np.ndarray, centres_xy: np.ndarray, radius_meters: float) -> np.ndarray:
    """Which points lie within a horizontal radius of at least one centre (height ignored).

    Args:
        points_xyz: (N, 3) map-frame points.
        centres_xy: (K, 2) map-frame centres, e.g. the lidar positions of the pooled sweeps.
        radius_meters: Radius (m), inclusive.

    Returns:
        (N,) bool: the union of the K discs.
    """
    near_any = np.zeros(np.asarray(points_xyz).shape[0], dtype=bool)
    for centre_xy in np.atleast_2d(centres_xy):
        near_any |= within_radius(points_xyz, centre_xy, radius_meters)
    return near_any


def resolve_selected_class_ids(eval_settings: Dict[str, Any], requested_names: Optional[Sequence[str]]) -> Dict[int, str]:
    """The scored classes to draw, by id.

    Args:
        eval_settings: ``eval`` section of configs/eval.yaml (``scored_class_ids`` name -> id, optional
            ``viz_classes`` list of names, null = all).
        requested_names: Class names from ``--classes``, or None to use the config's ``viz_classes``.

    Returns:
        Class id -> class name for the selected classes (all scored classes when nothing is requested).

    Raises:
        ValueError: If a name is not a scored class or the selection is empty.
    """
    scored_ids_by_name = dict(eval_settings["scored_class_ids"])
    # Step 1: command line first, then the config default, then every scored class.
    chosen_names = requested_names if requested_names is not None else eval_settings.get("viz_classes")
    if chosen_names is None:
        chosen_names = list(scored_ids_by_name)
    # Step 2: every name must be a scored class; an unknown one is a typo, not an empty selection.
    unknown_names = [name for name in chosen_names if name not in scored_ids_by_name]
    if unknown_names or not chosen_names:
        raise ValueError(f"classes {unknown_names or list(chosen_names)} are not scored classes (have {sorted(scored_ids_by_name)})")
    return {scored_ids_by_name[name]: name for name in chosen_names}


def pooled_voxel_maps(sweeps: Sequence[RenderedSweep], scene: RunScene, selected_class_ids: Sequence[int],
                      use_gated: bool, radius_meters: float) -> Dict[str, Any]:
    """The GT and render voxels of one or several pooled sweeps for the three maps (map-frame voxel centres).

    Pooling is the report's (scripts/13 ``pooled_map_points``): every sweep's points go to the map frame
    and onto the GT grid, and a voxel is occupied when any sweep put a point in it. Per selected class,
    inside its scoring neighbourhood (for a subset of the run's sweeps, the part those poses see):
    GT voxels = the class's GT, hit when the render occupies it (recall), missed otherwise; render
    voxels = occupied voxels, correct when on a GT voxel of the class (precision), incorrect
    otherwise. These are the cells ``neighbourhood_voxel_metrics`` counts, unioned over the selected
    classes; a voxel correct for one class and incorrect for another (overlapping neighbourhoods) is
    drawn as correct. Occupied voxels outside every selected neighbourhood are not scored; they come back as
    ``render_outside`` so all three maps can show the whole rendered cloud (the report never counts them).

    Args:
        sweeps: Rendered test sweeps of this run (their own returned points only).
        scene: The run's scene (grid, GT classes, class neighbourhoods).
        selected_class_ids: Scored class ids to include.
        use_gated: Use the gated point set.
        radius_meters: Horizontal radius around any of the sweeps' lidar positions to keep.

    Returns:
        {"gt_hit" / "gt_miss" / "render_correct" / "render_incorrect" / "render_outside": (V, 3) voxel
        centres within the radius, "totals": the first four names -> voxel counts over the whole scored
        region (what the report counts), "outside_total": the uncut count of render_outside (never in
        the report), "point_set": name, "point_count": returned points voxelised, "sweep_count": int}.
    """
    # Step 1: one point set for all sweeps (ungated unless gated is asked for or any sweep lacks it, so sweeps never mix sets).
    point_set = "gated" if use_gated or any(sweep.ungated_points is None for sweep in sweeps) else "ungated"
    grid_shape = scene.ground_truth_class_ids.shape
    # Step 2: pooled occupied voxels on the GT grid (T68), outside-grid points dropped as in the report.
    voxel_indices, inside_masks, point_count = [], [], 0
    for sweep in sweeps:
        lidar_points = sweep.gated_points if point_set == "gated" else sweep.ungated_points
        point_count += int(lidar_points.shape[0])
        _, sweep_indices, sweep_inside = points_to_map_voxels(
            lidar_points, sweep.lidar_to_map, scene.grid_origin, scene.voxel_size, grid_shape)
        voxel_indices.append(sweep_indices)
        inside_masks.append(sweep_inside)
    occupied = occupied_voxel_grid(np.concatenate(voxel_indices), np.concatenate(inside_masks), grid_shape)
    # Step 3: per class, GT and rendered voxels inside the neighbourhood. The run's neighbourhood already is the
    # union over all its poses, so it is restricted again only when fewer sweeps are drawn.
    draws_all_sweeps = {id(sweep) for sweep in sweeps} == {id(sweep) for sweep in scene.sweeps.values()}
    ground_truth_mask = np.zeros(grid_shape, dtype=bool)
    correct_mask = np.zeros(grid_shape, dtype=bool)
    incorrect_mask = np.zeros(grid_shape, dtype=bool)
    for class_id in selected_class_ids:
        region = scene.class_neighbourhoods[class_id]
        if not draws_all_sweeps:
            region = restrict_mask_to_visible(region, scene.grid_origin, scene.voxel_size,
                                              [sweep.lidar_to_map for sweep in sweeps], scene.max_range,
                                              scene.vertical_half_fov)
        class_ground_truth = (scene.ground_truth_class_ids == class_id) & region
        ground_truth_mask |= class_ground_truth
        correct_mask |= occupied & class_ground_truth
        incorrect_mask |= occupied & region & ~class_ground_truth
    # Step 4: one colour per voxel: correct beats incorrect; the hit GT voxels are the correct render voxels.
    incorrect_mask &= ~correct_mask
    # Occupied voxels the metric does not score (outside every selected neighbourhood): shown, never counted.
    outside_mask = occupied & ~correct_mask & ~incorrect_mask
    voxel_masks = {"gt_hit": ground_truth_mask & occupied, "gt_miss": ground_truth_mask & ~occupied,
                   "render_correct": correct_mask, "render_incorrect": incorrect_mask, "render_outside": outside_mask}
    # Step 5: voxel centres within the radius of any lidar position; the totals keep the uncut counts.
    lidar_positions_xy = np.array([sweep.lidar_to_map[:2, 3] for sweep in sweeps])
    maps: Dict[str, Any] = {"point_set": point_set, "point_count": point_count, "sweep_count": len(sweeps), "totals": {}}
    for map_name, voxel_mask in voxel_masks.items():
        centres = voxel_centers(np.argwhere(voxel_mask), scene.grid_origin, scene.voxel_size)
        # The outside voxels are not part of the report, so they stay out of "totals".
        if map_name == "render_outside":
            maps["outside_total"] = int(centres.shape[0])
        else:
            maps["totals"][map_name] = int(centres.shape[0])
        maps[map_name] = centres[within_radius_of_any(centres, lidar_positions_xy, radius_meters)]
    return maps


def sweep_voxel_maps(sweep: RenderedSweep, scene: RunScene, selected_class_ids: Sequence[int], use_gated: bool,
                     radius_meters: float) -> Dict[str, Any]:
    """The voxel maps of one sweep (``pooled_voxel_maps`` of a single sweep).

    Args:
        sweep: Rendered test sweep (its own returned points only).
        scene: The run's scene.
        selected_class_ids: Scored class ids to include.
        use_gated: Use the gated point set.
        radius_meters: Horizontal radius around the sweep's lidar to keep.

    Returns:
        See ``pooled_voxel_maps``.
    """
    return pooled_voxel_maps([sweep], scene, selected_class_ids, use_gated, radius_meters)


def panel_arrays(sweep: RenderedSweep, scored_cuboids: Sequence[Cuboid], use_gated: bool,
                 neighbourhood_cuboids: Optional[Sequence[Cuboid]] = None) -> Dict[str, Any]:
    """Map-frame points of one sweep and their inside-cuboid flags.

    Args:
        sweep: Rendered test sweep.
        scored_cuboids: Boxes the metric scores.
        use_gated: Draw the gated set; otherwise the ungated set when the render has one.
        neighbourhood_cuboids: Boxes grown by their class margin; only points inside them are kept
            (rendered and real). None keeps every point.

    Returns:
        {"real" (N, 3), "rendered" (M, 3), "inside" (M,) bool, "per_cuboid_counts" (B,) int,
        "point_set" name, "rendered_total" count before the neighbourhood crop}.
    """
    # Step 1: the rendered set the metric binds to (ungated) unless gated is asked for or missing.
    rendered_lidar_points, point_set = rendered_point_set(sweep, use_gated)
    # Step 2: both clouds to the shared map frame (T70).
    rendered_map = sweep_points_to_map(rendered_lidar_points, sweep.lidar_to_map)
    real_map = sweep_points_to_map(sweep.real_points, sweep.lidar_to_map)
    rendered_total = int(rendered_map.shape[0])
    # Step 3: keep the neighbourhood of the scored boxes, the points the precision metric counts.
    if neighbourhood_cuboids is not None:
        rendered_map = rendered_map[points_in_any_cuboid(rendered_map, neighbourhood_cuboids)]
        real_map = real_map[points_in_any_cuboid(real_map, neighbourhood_cuboids)]
    # Step 4: the metric's own inside-cuboid tests on what is drawn.
    return {
        "real": real_map,
        "rendered": rendered_map,
        "inside": points_in_any_cuboid(rendered_map, scored_cuboids),
        "per_cuboid_counts": cuboid_point_counts(rendered_map, scored_cuboids),
        "point_set": point_set,
        "rendered_total": rendered_total,
    }


def map_layers(map_name: str, maps: Dict[str, Any]) -> List[Tuple[np.ndarray, Tuple[int, int, int], float]]:
    """The voxel layers of one map, back to front, as (centres, RGB, opacity).

    Every map shows the whole rendered cloud (``render_correct`` + ``render_incorrect`` +
    ``render_outside``), so the three pictures are comparable.

    Args:
        map_name: One of ``MAP_NAMES``.
        maps: ``pooled_voxel_maps`` result.

    Returns:
        Overlay: all GT voxels grey (mostly opaque) under all render voxels light blue (translucent).
        Recall: render voxels off a hit GT voxel very faint green, missed GT grey, hit GT solid green.
        Precision: render voxels not on GT (incorrect and outside the neighbourhood) faint red, correct
        ones solid green, no GT voxel.
    """
    all_render = np.vstack([maps["render_correct"], maps["render_incorrect"], maps["render_outside"]])
    if map_name == "overlay":
        ground_truth = np.vstack([maps["gt_hit"], maps["gt_miss"]])
        return [(ground_truth, GROUND_TRUTH_VOXEL_RGB, GROUND_TRUTH_OVERLAY_OPACITY),
                (all_render, RENDER_VOXEL_RGB, RENDER_OVERLAY_OPACITY)]
    if map_name == "recall":
        # The hit GT voxels are the correct render voxels, so the faint context is every other render voxel.
        context = np.vstack([maps["render_incorrect"], maps["render_outside"]])
        return [(context, HIT_VOXEL_RGB, RECALL_CONTEXT_OPACITY), (maps["gt_miss"], GROUND_TRUTH_VOXEL_RGB, SOLID_VOXEL_OPACITY),
                (maps["gt_hit"], HIT_VOXEL_RGB, SOLID_VOXEL_OPACITY)]
    if map_name == "precision":
        not_on_ground_truth = np.vstack([maps["render_incorrect"], maps["render_outside"]])
        return [(not_on_ground_truth, NON_HIT_VOXEL_RGB, NON_HIT_PRECISION_OPACITY), (maps["render_correct"], HIT_VOXEL_RGB, SOLID_VOXEL_OPACITY)]
    raise ValueError(f"unknown map {map_name!r} (have {MAP_NAMES})")


def map_caption(map_name: str, maps: Dict[str, Any]) -> str:
    """Two caption lines with the map's own voxel counts.

    Args:
        map_name: One of ``MAP_NAMES``.
        maps: ``pooled_voxel_maps`` result.

    Returns:
        Overlay: GT and render voxel counts (all rendered voxels). Recall: hit / GT voxels, faint green =
        the other render voxels. Precision: correct / scored render voxels, faint red = not on GT.
        The ratios are the report's; the faint voxels outside the selected neighbourhoods are named, never counted.
    """
    # Whole-region counts (what the report counts), not only the drawn part within the radius.
    totals = maps["totals"]
    gt_hits, gt_misses = totals["gt_hit"], totals["gt_miss"]
    correct, incorrect, outside = totals["render_correct"], totals["render_incorrect"], maps["outside_total"]
    if map_name == "overlay":
        return f"{gt_hits + gt_misses} GT voxels (grey), {correct + incorrect + outside} render voxels (light blue)"
    if map_name == "recall":
        return (f"recall {ratio_text(gt_hits, gt_hits + gt_misses)}: {gt_hits} of {gt_hits + gt_misses} GT voxels hit (green); "
                f"faint green = {incorrect + outside} other render voxels")
    if map_name == "precision":
        return (f"precision {ratio_text(correct, correct + incorrect)}: {correct} of {correct + incorrect} render voxels on GT (green); "
                f"faint red = not on GT, {outside} of them outside the selected classes' neighbourhoods (not counted)")
    raise ValueError(f"unknown map {map_name!r} (have {MAP_NAMES})")


def ratio_text(numerator: int, denominator: int) -> str:
    """A ratio as two decimals, or ``n/a`` for an empty denominator.

    Args:
        numerator: Count of hits.
        denominator: Count of candidates.

    Returns:
        ``numerator / denominator`` formatted, or "n/a".
    """
    return "n/a" if denominator == 0 else f"{numerator / denominator:.2f}"


def draw_voxels(plotter, pyvista_module, centres: np.ndarray, voxel_size: float,
                colour_rgb: Tuple[int, int, int], opacity: float) -> None:
    """Draw voxels of one colour as cubes.

    Args:
        plotter: ``pyvista.Plotter`` with the panel's subplot active.
        pyvista_module: The imported ``pyvista``.
        centres: (V, 3) voxel centres; nothing is drawn when empty.
        voxel_size: Voxel edge (m).
        colour_rgb: 0-255 RGB.
        opacity: 0-1.
    """
    if centres.shape[0] == 0:
        return
    # One cube per voxel centre: glyph copies the cube to every point (no rotation, no scaling).
    cube = pyvista_module.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    cubes = pyvista_module.PolyData(centres).glyph(geom=cube, orient=False, scale=False)
    # High ambient light keeps the side faces near the true colour (with default lighting they darken, which
    # washed green against grey); the remaining diffuse share still separates the faces of neighbouring cubes.
    plotter.add_mesh(cubes, color=np.array(colour_rgb) / 255.0, opacity=opacity,
                     ambient=VOXEL_AMBIENT_LIGHT, diffuse=1.0 - VOXEL_AMBIENT_LIGHT)


def draw_panel(plotter, pyvista_module, column: int, map_name: str, scene: RunScene, selected_class_ids: Sequence[int],
               stem: str, maps: Dict[str, Any], radius_meters: float, lidar_positions_xy: np.ndarray,
               point_arrays: Optional[Dict[str, Any]]) -> None:
    """Draw one run's panel of one map, for the pooled sweeps or for one pose.

    Args:
        plotter: ``pyvista.Plotter`` with one row of panels.
        pyvista_module: The imported ``pyvista`` (imported lazily by ``render_pose``).
        column: Panel column of this run.
        map_name: One of ``MAP_NAMES``.
        scene: The run's scene.
        selected_class_ids: Classes drawn (the wireframes follow the same filter).
        stem: Sweep stem, or ``pooled``.
        maps: ``pooled_voxel_maps`` result.
        radius_meters: Horizontal radius drawn around the lidar positions.
        lidar_positions_xy: (K, 2) map-frame lidar positions of the drawn sweeps.
        point_arrays: ``panel_arrays`` result for the optional point overlay (overlay map only), or None.
    """
    plotter.subplot(0, column)
    # Step 1: the map's voxel layers, back to front.
    for centres, colour_rgb, opacity in map_layers(map_name, maps):
        draw_voxels(plotter, pyvista_module, centres, scene.voxel_size, colour_rgb, opacity)
    # Step 2: the selected classes' scored cuboids within the radius, as thin dark wireframes for orientation.
    selected_cuboids = [cuboid for cuboid in scene.scored_cuboids if cuboid.class_id in selected_class_ids]
    nearby_cuboids = [cuboid for cuboid, nearby in zip(selected_cuboids, within_radius_of_any(cuboid_centers(selected_cuboids), lidar_positions_xy, radius_meters))
                      if nearby] if selected_cuboids else []
    corner_points, line_cells = cuboid_wireframe_arrays(nearby_cuboids)
    if corner_points.shape[0]:
        plotter.add_mesh(pyvista_module.PolyData(corner_points, lines=line_cells),
                         color=np.array(CUBOID_WIREFRAME_RGB) / 255.0, line_width=WIREFRAME_LINE_WIDTH)
    # Step 3: optional returns, kept to the radius so they cannot stretch the camera.
    if point_arrays is not None:
        real_points = point_arrays["real"][within_radius_of_any(point_arrays["real"], lidar_positions_xy, radius_meters)]
        near_rendered = within_radius_of_any(point_arrays["rendered"], lidar_positions_xy, radius_meters)
        if real_points.shape[0]:
            plotter.add_mesh(pyvista_module.PolyData(real_points), color=np.array(REAL_POINT_RGB) / 255.0,
                             point_size=REAL_POINT_SIZE, opacity=REAL_POINT_OPACITY)
        if near_rendered.any():
            rendered_cloud = pyvista_module.PolyData(point_arrays["rendered"][near_rendered])
            rendered_cloud["colors"] = inside_outside_colors(point_arrays["inside"][near_rendered])
            plotter.add_mesh(rendered_cloud, scalars="colors", rgb=True, point_size=RENDERED_POINT_SIZE)
    # Step 4: caption on short lines (8 pt, about 6 px a character) so it fits an 800 px panel.
    class_names = ", ".join(scene.class_names[class_id] for class_id in selected_class_ids)
    plotter.add_text(f"{scene.run_id}\n{stem}: {maps['sweep_count']} test sweep(s), {maps['point_count']} returned points "
                     f"({maps['point_set']}), drawn within {radius_meters:g} m of a lidar\nclasses: {class_names}\n{map_caption(map_name, maps)}", font_size=8)


def render_pose(scenes: Sequence[RunScene], stem: str, selected_class_ids: Sequence[int], use_gated: bool,
                output_dir: Optional[Path], radius_meters: float, view: str = VIEW_CHOICES[0],
                show_points: bool = False, crop_points_to_neighbourhood: bool = True) -> None:
    """Draw the three maps of every run side by side, for one pose or for all pooled sweeps, to PNGs or windows.

    Args:
        scenes: One scene per run.
        stem: Sweep stem (present in every scene), or ``POOLED_STEM`` for all sweeps of each run pooled.
        selected_class_ids: Scored class ids to draw.
        use_gated: Use the gated point set.
        output_dir: Folder for ``<stem>_<map>.png`` (``pooled_<map>.png``), or None for an interactive window per map.
        radius_meters: Horizontal radius around any drawn sweep's lidar position to draw.
        view: ``oblique`` or ``top``.
        show_points: Also draw the real and rendered returns (overlay map, single pose only).
        crop_points_to_neighbourhood: With ``show_points``, only points in the scored neighbourhood.
    """
    # PyVista (VTK) is imported here so the module loads in environments without it (tests).
    import pyvista

    # Step 1: each run's voxels, once; the three maps are different views of the same arrays.
    is_pooled = stem == POOLED_STEM
    run_sweeps = [list(scene.sweeps.values()) if is_pooled else [scene.sweeps[stem]] for scene in scenes]
    run_maps = [pooled_voxel_maps(sweeps, scene, selected_class_ids, use_gated, radius_meters)
                for scene, sweeps in zip(scenes, run_sweeps)]
    for map_name in MAP_NAMES:
        plotter = pyvista.Plotter(shape=(1, len(scenes)), off_screen=output_dir is not None,
                                  window_size=(PANEL_PIXELS * len(scenes), PANEL_PIXELS))
        # The window is closed even when drawing fails, so one bad map cannot leak it.
        try:
            for column, (scene, sweeps, maps) in enumerate(zip(scenes, run_sweeps, run_maps)):
                point_arrays = None
                if show_points and map_name == "overlay" and not is_pooled:
                    # Cropping to an empty neighbourhood would blank the points, so with no scored box keep them all.
                    # The points follow the class filter too: only the selected classes' boxes crop and colour them.
                    selected_scored = [box for box in scene.scored_cuboids if box.class_id in selected_class_ids]
                    selected_grown = [box for box in scene.neighbourhood_cuboids if box.class_id in selected_class_ids]
                    neighbourhood = selected_grown if (crop_points_to_neighbourhood and selected_grown) else None
                    point_arrays = panel_arrays(sweeps[0], selected_scored, use_gated, neighbourhood)
                draw_panel(plotter, pyvista, column, map_name, scene, selected_class_ids, stem, maps, radius_meters,
                           np.array([sweep.lidar_to_map[:2, 3] for sweep in sweeps]), point_arrays)
                if view == "top":
                    plotter.view_xy()
            # Translucent voxels over opaque ones (render over GT, red over green) need depth peeling to composite in the right order.
            # Peeling is a per-renderer setting, so every run's panel gets it, not only the last active one.
            if map_name in MAP_NAMES:  # every map stacks translucent voxels over solid ones
                for column in range(len(scenes)):
                    plotter.subplot(0, column)
                    plotter.enable_depth_peeling()
            # One shared camera, so the runs are compared from the same viewpoint.
            plotter.link_views()
            if output_dir is None:
                plotter.show()
            else:
                output_path = output_dir / f"{stem}_{map_name}.png"
                plotter.show(screenshot=str(output_path))
                print(f"wrote {output_path}")
        finally:
            plotter.close()


def main(argument_list: Optional[List[str]] = None) -> int:
    """Draw the pooled maps (no ``--pose``) or the chosen poses of the given runs.

    Args:
        argument_list: CLI arguments (default: sys.argv).

    Returns:
        Exit code: 0 success, 1 refused input or a pose that failed to draw.
    """
    arguments = build_argument_parser().parse_args(argument_list)
    project_root = find_project_root()
    # Step 1: a rerender label names one run's folder, so it cannot be combined with several runs.
    if arguments.rerender is not None and len(arguments.run_dirs) != 1:
        print("--rerender takes exactly one run folder", file=sys.stderr)
        return 1
    # Steps 2-3 run in one guard so a missing file or key is reported as a refusal (exit 1), not a traceback.
    try:
        # Step 2: eval settings, the T67 threshold and the radius, resolved as scripts/13 resolves its own.
        eval_config_path = arguments.eval_config or project_root / DEFAULT_EVAL_CONFIG_RELATIVE_PATH
        eval_settings = yaml.safe_load(Path(eval_config_path).read_text())["eval"]
        accumulation_threshold = arguments.accumulation_threshold
        if accumulation_threshold is None:
            accumulation_threshold = float(eval_settings["render_accumulation_threshold"])
        radius_meters = arguments.radius_meters
        if radius_meters is None:
            radius_meters = float(eval_settings["viz_radius_meters"])
        if radius_meters <= 0:
            raise ValueError(f"radius must be positive, got {radius_meters}")
        selected_class_ids = resolve_selected_class_ids(
            eval_settings, None if arguments.classes is None else [name.strip() for name in arguments.classes.split(",")])
        # Step 3: load every run, refuse non-comparable ones, pick the poses.
        scenes = [load_run_scene(run_dir, arguments.rerender, eval_settings, accumulation_threshold, project_root)
                  for run_dir in arguments.run_dirs]
        check_scenes_comparable(scenes)
        # No --pose: the pooled maps the report is computed on. The raw points exist per sweep only.
        if arguments.pose is None:
            if arguments.show_points or arguments.all_points:
                raise ValueError("--show-points / --all-points draw one sweep's returns: give --pose")
            stems = [POOLED_STEM]
        else:
            stems = select_pose_stems(scenes, arguments.pose)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"refused: {error!r}" if isinstance(error, KeyError) else f"refused: {error}", file=sys.stderr)
        return 1
    # Step 4: one picture set per stem (the pooled maps or each --pose; PNG in --output-dir, or an interactive window). A failing pose is
    # reported and the others still drawn; the exit code is 1 if any failed (missing pyvista, no OpenGL, ...).
    failed_stems = []
    try:
        if arguments.output_dir is not None:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    for stem in stems:
        try:
            render_pose(scenes, stem, list(selected_class_ids), arguments.gated, arguments.output_dir, radius_meters, arguments.view,
                        arguments.show_points, not arguments.all_points)
        except Exception as error:  # noqa: BLE001  (any drawing failure must not stop the other poses)
            failed_stems.append(stem)
            print(f"failed {stem}: {error!r}", file=sys.stderr)
    return 1 if failed_stems else 0


if __name__ == "__main__":
    sys.exit(main())
