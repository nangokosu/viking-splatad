"""Draw one lidar pose of a run as occupancy voxels against the GT with PyVista (visual aid of scripts/13).

For each chosen test pose and each run, one 3D panel in the map frame shows, within ``--radius-meters``
(horizontal) of that sweep's lidar, the voxels of the metric's own scoring neighbourhood:
    - green: occupied by the render and by the GT (true positive),
    - red: occupied by the render only (false positive),
    - grey, translucent: GT only (missed),
    - dark wireframes: the scored GT cuboids.
The voxel grid, the neighbourhood (each box grown horizontally by its class margin, within the sweep's
range and vertical FOV) and the GT classes are exactly those of the voxel IoU / precision / recall in
scripts/13, evaluated for this one sweep, so the colours are the metric's confusion cells for the pose
(cells are unioned over the scored classes, so a voxel of one class lying in another class's
neighbourhood is green here but a false positive of that other class in the per-class report).
The report pools all test sweeps, so a single pose's counts illustrate it and are not its numbers.
``--show-points`` adds the rendered and real returns. Several ``--run`` folders give side-by-side
panels with linked cameras (e.g. B and R of one session).

Everything is derived as scripts/13 derives it, through the same functions: the run's split record,
session and GT, the T67 accumulation threshold, the scored region (``scored_region_cuboids``) and the
neighbourhood (``visible_neighbourhood_mask``). GT cuboids are display only (CLAUDE.md rule 6). The
run folders are read, never written; only ``--output-dir`` receives PNGs.

PyVista lives in the isolated viz/ project (the root environment has none):
    uv sync --project viz --locked
    uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<run_id> --output-dir <dir>
    uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<B> --run runs/<R> --pose 000529 --radius-meters 15
Without ``--output-dir`` an interactive window opens for each pose.
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
    FALSE_NEGATIVE_VOXEL_RGB,
    FALSE_POSITIVE_VOXEL_RGB,
    REAL_POINT_RGB,
    TRUE_POSITIVE_VOXEL_RGB,
    cuboid_wireframe_arrays,
    inside_outside_colors,
)
from viking_splatad.viz.viewer_export import sweep_points_to_map

EVAL_SCRIPT_PATH = Path(__file__).resolve().parent / "13_eval_semantic_occupancy.py"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
# Camera choices: the default oblique 3D view, or straight down (map x-y plane).
VIEW_CHOICES = ("oblique", "top")
# Display sizes in pixels, a presentation choice with no data-derived value: the rendered points
# are drawn larger than the real ones so both stay visible when they overlap; one panel is
# 800 x 800 px so several side-by-side panels still fit a laptop screen.
REAL_POINT_SIZE, RENDERED_POINT_SIZE, WIREFRAME_LINE_WIDTH = 2.0, 4.0, 1.0
PANEL_PIXELS = 800
# Opacity of the real returns (so rendered points over them stay readable) and of the missed-GT
# voxels (so the voxels in front of them stay visible); both presentation choices.
REAL_POINT_OPACITY = 0.5
MISSED_VOXEL_OPACITY = 0.35
# Names of the three voxel cells, in drawing order (missed GT first so it sits behind the others).
VOXEL_CELL_NAMES = ("false_negative", "false_positive", "true_positive")


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
                        help="sweep stem or frame index to draw (repeatable); default: every scored sweep")
    parser.add_argument("--radius-meters", type=float, default=None,
                        help="horizontal radius around each sweep's lidar to draw (default: eval.viz_radius_meters)")
    parser.add_argument("--view", choices=VIEW_CHOICES, default=VIEW_CHOICES[0],
                        help="camera: oblique 3D or straight down (default: oblique)")
    parser.add_argument("--show-points", action="store_true",
                        help="also draw the real (grey) and rendered (red inside a cuboid, blue otherwise) returns")
    parser.add_argument("--all-points", action="store_true",
                        help="with --show-points: draw every point in the radius, not only the scored neighbourhood")
    parser.add_argument("--gated", action="store_true",
                        help="use the ray-drop-gated points instead of the ungated set the metric binds to")
    parser.add_argument("--output-dir", type=Path, default=None, help="write <stem>.png here; default: interactive window")
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


def sweep_voxel_cells(sweep: RenderedSweep, scene: RunScene, use_gated: bool, radius_meters: float) -> Dict[str, Any]:
    """The metric's voxel confusion cells for one sweep, as map-frame voxel centres within a radius.

    Per scored class, inside its neighbourhood (restricted to what this sweep's pose can see):
    true positive = rendered-occupied and GT of the class, false positive = rendered-occupied but
    not GT of the class, false negative = GT of the class but not rendered-occupied. These are the
    cells of ``neighbourhood_voxel_metrics`` for one sweep, unioned over the scored classes; a voxel
    that is a hit for one class and a false positive for another is drawn as a hit.

    Args:
        sweep: Rendered test sweep.
        scene: The run's scene (grid, GT classes, class neighbourhoods).
        use_gated: Use the gated point set.
        radius_meters: Horizontal radius around the sweep's lidar to keep.

    Returns:
        {"true_positive" / "false_positive" / "false_negative": (V, 3) voxel centres, "point_set": name}.
    """
    # Step 1: this sweep's occupied voxels on the GT grid (T68).
    rendered_lidar_points, point_set = rendered_point_set(sweep, use_gated)
    grid_shape = scene.ground_truth_class_ids.shape
    _, voxel_indices, inside_grid = points_to_map_voxels(
        rendered_lidar_points, sweep.lidar_to_map, scene.grid_origin, scene.voxel_size, grid_shape)
    occupied = occupied_voxel_grid(voxel_indices, inside_grid, grid_shape)
    # Step 2: the confusion cells of each class, as neighbourhood_voxel_metrics counts them, inside the
    # part of that class's neighbourhood this pose can see (GT voxels of the class within it).
    grid_shape = scene.ground_truth_class_ids.shape
    true_positive_mask = np.zeros(grid_shape, dtype=bool)
    false_positive_mask = np.zeros(grid_shape, dtype=bool)
    false_negative_mask = np.zeros(grid_shape, dtype=bool)
    for class_id, class_neighbourhood in scene.class_neighbourhoods.items():
        sweep_region = restrict_mask_to_visible(class_neighbourhood, scene.grid_origin, scene.voxel_size,
                                                [sweep.lidar_to_map], scene.max_range, scene.vertical_half_fov)
        class_ground_truth = (scene.ground_truth_class_ids == class_id) & sweep_region
        true_positive_mask |= occupied & class_ground_truth
        false_positive_mask |= occupied & sweep_region & ~class_ground_truth
        false_negative_mask |= class_ground_truth & ~occupied
    # Step 3: a voxel that is a hit for one class and a false positive for another (overlapping
    # neighbourhoods) is drawn as a hit, so every voxel has one colour.
    false_positive_mask &= ~true_positive_mask
    cell_masks = {"true_positive": true_positive_mask, "false_positive": false_positive_mask,
                  "false_negative": false_negative_mask}
    # Step 4: voxel centres within the radius of the lidar position.
    lidar_position_xy = sweep.lidar_to_map[:2, 3]
    cells: Dict[str, Any] = {"point_set": point_set}
    for cell_name, cell_mask in cell_masks.items():
        centres = voxel_centers(np.argwhere(cell_mask), scene.grid_origin, scene.voxel_size)
        cells[cell_name] = centres[within_radius(centres, lidar_position_xy, radius_meters)]
    return cells


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


def cell_counts_and_iou(cells: Dict[str, Any]) -> Tuple[int, int, int, Optional[float]]:
    """Voxel counts of the three cells and their IoU = TP / (TP + FP + FN).

    Args:
        cells: ``sweep_voxel_cells`` result.

    Returns:
        (true positives, false positives, false negatives, IoU or None when all three are 0).
    """
    true_positives, false_positives, false_negatives = (
        int(cells[cell_name].shape[0]) for cell_name in ("true_positive", "false_positive", "false_negative"))
    union = true_positives + false_positives + false_negatives
    return true_positives, false_positives, false_negatives, (true_positives / union if union else None)


def draw_voxels(plotter, pyvista_module, cells: Dict[str, Any], voxel_size: float) -> None:
    """Draw the three voxel cells as cubes, missed GT translucent and behind the others.

    Args:
        plotter: ``pyvista.Plotter`` with the panel's subplot active.
        pyvista_module: The imported ``pyvista``.
        cells: ``sweep_voxel_cells`` result.
        voxel_size: Voxel edge (m).
    """
    cell_colours = {"false_negative": FALSE_NEGATIVE_VOXEL_RGB, "false_positive": FALSE_POSITIVE_VOXEL_RGB,
                    "true_positive": TRUE_POSITIVE_VOXEL_RGB}
    cube = pyvista_module.Cube(x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
    for cell_name in VOXEL_CELL_NAMES:
        centres = cells[cell_name]
        if centres.shape[0]:
            # One cube per voxel centre: glyph copies the cube to every point (no rotation, no scaling).
            cubes = pyvista_module.PolyData(centres).glyph(geom=cube, orient=False, scale=False)
            plotter.add_mesh(cubes, color=np.array(cell_colours[cell_name]) / 255.0,
                             opacity=MISSED_VOXEL_OPACITY if cell_name == "false_negative" else 1.0)


def draw_panel(plotter, pyvista_module, column: int, scene: RunScene, stem: str, cells: Dict[str, Any],
               radius_meters: float, lidar_position_xy: np.ndarray, point_arrays: Optional[Dict[str, Any]]) -> None:
    """Draw one run's panel of one pose.

    Args:
        plotter: ``pyvista.Plotter`` with one row of panels.
        pyvista_module: The imported ``pyvista`` (imported lazily by ``render_pose``).
        column: Panel column of this run.
        scene: The run's scene.
        stem: Sweep stem.
        cells: ``sweep_voxel_cells`` result.
        radius_meters: Horizontal radius drawn around the lidar.
        lidar_position_xy: (2,) map-frame lidar position of this sweep.
        point_arrays: ``panel_arrays`` result for the optional point overlay, or None.
    """
    plotter.subplot(0, column)
    # Step 1: the voxel confusion cells.
    draw_voxels(plotter, pyvista_module, cells, scene.voxel_size)
    # Step 2: the scored cuboids within the radius, as thin dark wireframes for orientation.
    cuboid_centres = cuboid_centers(scene.scored_cuboids)
    nearby_cuboids = [cuboid for cuboid, nearby in zip(scene.scored_cuboids, within_radius(cuboid_centres, lidar_position_xy, radius_meters))
                      if nearby]
    corner_points, line_cells = cuboid_wireframe_arrays(nearby_cuboids)
    if corner_points.shape[0]:
        plotter.add_mesh(pyvista_module.PolyData(corner_points, lines=line_cells),
                         color=np.array(CUBOID_WIREFRAME_RGB) / 255.0, line_width=WIREFRAME_LINE_WIDTH)
    # Step 3: optional returns, kept to the radius so they cannot stretch the camera.
    if point_arrays is not None:
        real_points = point_arrays["real"][within_radius(point_arrays["real"], lidar_position_xy, radius_meters)]
        near_rendered = within_radius(point_arrays["rendered"], lidar_position_xy, radius_meters)
        if real_points.shape[0]:
            plotter.add_mesh(pyvista_module.PolyData(real_points), color=np.array(REAL_POINT_RGB) / 255.0,
                             point_size=REAL_POINT_SIZE, opacity=REAL_POINT_OPACITY)
        if near_rendered.any():
            rendered_cloud = pyvista_module.PolyData(point_arrays["rendered"][near_rendered])
            rendered_cloud["colors"] = inside_outside_colors(point_arrays["inside"][near_rendered])
            plotter.add_mesh(rendered_cloud, scalars="colors", rgb=True, point_size=RENDERED_POINT_SIZE)
    # Step 4: caption on short lines (8 pt, about 6 px a character) so it fits an 800 px panel.
    true_positives, false_positives, false_negatives, iou = cell_counts_and_iou(cells)
    iou_text = "n/a" if iou is None else f"{iou:.2f}"
    plotter.add_text(f"{scene.run_id}\n{stem} ({cells['point_set']}), within {radius_meters:g} m\n"
                     f"voxels: {true_positives} both (green), {false_positives} rendered only (red)\n"
                     f"{false_negatives} GT only (grey), IoU {iou_text}", font_size=8)


def render_pose(scenes: Sequence[RunScene], stem: str, use_gated: bool, output_path: Optional[Path],
                radius_meters: float, view: str = VIEW_CHOICES[0], show_points: bool = False,
                crop_points_to_neighbourhood: bool = True) -> None:
    """Draw one pose of every run side by side, to a PNG or an interactive window.

    Args:
        scenes: One scene per run.
        stem: Sweep stem (present in every scene).
        use_gated: Use the gated point set.
        output_path: PNG to write, or None for an interactive window.
        radius_meters: Horizontal radius around each sweep's lidar to draw.
        view: ``oblique`` or ``top``.
        show_points: Also draw the real and rendered returns.
        crop_points_to_neighbourhood: With ``show_points``, only points in the scored neighbourhood.
    """
    # PyVista (VTK) is imported here so the module loads in environments without it (tests).
    import pyvista

    plotter = pyvista.Plotter(shape=(1, len(scenes)), off_screen=output_path is not None,
                              window_size=(PANEL_PIXELS * len(scenes), PANEL_PIXELS))
    # The window is closed even when drawing fails, so one bad pose cannot leak it.
    try:
        for column, scene in enumerate(scenes):
            sweep = scene.sweeps[stem]
            cells = sweep_voxel_cells(sweep, scene, use_gated, radius_meters)
            point_arrays = None
            if show_points:
                # Cropping to an empty neighbourhood would blank the points, so with no scored box keep them all.
                neighbourhood = scene.neighbourhood_cuboids if (crop_points_to_neighbourhood and scene.scored_cuboids) else None
                point_arrays = panel_arrays(sweep, scene.scored_cuboids, use_gated, neighbourhood)
            draw_panel(plotter, pyvista, column, scene, stem, cells, radius_meters, sweep.lidar_to_map[:2, 3], point_arrays)
            if view == "top":
                plotter.view_xy()
        # One shared camera, so the runs are compared from the same viewpoint.
        plotter.link_views()
        if output_path is None:
            plotter.show()
        else:
            plotter.show(screenshot=str(output_path))
            print(f"wrote {output_path}")
    finally:
        plotter.close()


def main(argument_list: Optional[List[str]] = None) -> int:
    """Draw the chosen poses of the given runs.

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
        # Step 3: load every run, refuse non-comparable ones, pick the poses.
        scenes = [load_run_scene(run_dir, arguments.rerender, eval_settings, accumulation_threshold, project_root)
                  for run_dir in arguments.run_dirs]
        check_scenes_comparable(scenes)
        stems = select_pose_stems(scenes, arguments.pose)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"refused: {error!r}" if isinstance(error, KeyError) else f"refused: {error}", file=sys.stderr)
        return 1
    # Step 4: one picture per pose (PNG in --output-dir, or an interactive window). A failing pose is
    # reported and the others still drawn; the exit code is 1 if any failed (missing pyvista, no OpenGL, ...).
    failed_stems = []
    try:
        if arguments.output_dir is not None:
            arguments.output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    for stem in stems:
        output_path = None if arguments.output_dir is None else arguments.output_dir / f"{stem}.png"
        try:
            render_pose(scenes, stem, arguments.gated, output_path, radius_meters, arguments.view,
                        arguments.show_points, not arguments.all_points)
        except Exception as error:  # noqa: BLE001  (any drawing failure must not stop the other poses)
            failed_stems.append(stem)
            print(f"failed {stem}: {error!r}", file=sys.stderr)
    return 1 if failed_stems else 0


if __name__ == "__main__":
    sys.exit(main())
