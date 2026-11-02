"""Draw one lidar pose of a run against the GT cuboids with PyVista (visual aid of scripts/13).

For each chosen test pose and each run, one 3D panel in the map frame shows:
    - the real lidar returns (grey),
    - the rendered returns, red inside a scored GT cuboid and blue in its halo,
    - the scored GT cuboids as green wireframes.
Only points in the neighbourhood of the scored cuboids (each box grown horizontally by its class
margin, the region the T46 point-precision metric counts) are drawn, so the view is framed on the
trunks instead of the whole 60 m scan; ``--all-points`` draws everything.
Several ``--run`` folders give side-by-side panels with linked cameras (e.g. B and R of one session).

Everything is derived exactly as scripts/13 derives it, through the same functions: the run's split
record, session and GT, the T67 accumulation threshold, the scored region (``scored_region_cuboids``)
and the inside-cuboid test (``points_in_any_cuboid``). The boxes drawn are the metric's scored boxes
(pooled over the run's test sweeps) and the points are this one sweep's, so the red/blue ratio of
one pose is a per-pose illustration, not the pooled T46 precision the report states. GT cuboids are display only (CLAUDE.md rule 6). The run folders are
read, never written; only ``--output-dir`` receives PNGs.

PyVista lives in the isolated viz/ project (the root environment has none):
    uv sync --project viz --locked
    uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<run_id> --output-dir <dir>
    uv run --project viz python scripts/15_view_pose_pyvista.py --run runs/<B> --run runs/<R> --pose 000529
Without ``--output-dir`` an interactive window opens for each pose.
"""

import argparse
import functools
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import yaml

from viking_splatad.config import find_project_root
from viking_splatad.eval.detection import (
    class_margin_table,
    cuboid_point_counts,
    grow_cuboid_horizontally,
    points_in_any_cuboid,
    scored_region_cuboids,
)
from viking_splatad.eval.rendered_pointcloud import RenderedSweep, load_test_lidar_sweeps
from viking_splatad.labels.cuboids import Cuboid
from viking_splatad.viz.pyvista_scene import (
    CUBOID_WIREFRAME_RGB,
    REAL_POINT_RGB,
    cuboid_wireframe_arrays,
    inside_outside_colors,
)
from viking_splatad.viz.viewer_export import sweep_points_to_map

EVAL_SCRIPT_PATH = Path(__file__).resolve().parent / "13_eval_semantic_occupancy.py"
DEFAULT_EVAL_CONFIG_RELATIVE_PATH = Path("configs") / "eval.yaml"
# Display sizes in pixels, a presentation choice with no data-derived value: the rendered points
# are drawn larger than the real ones so both stay visible when they overlap; one panel is
# 800 x 800 px so several side-by-side panels still fit a laptop screen.
REAL_POINT_SIZE, RENDERED_POINT_SIZE, WIREFRAME_LINE_WIDTH = 2.0, 4.0, 2.0
PANEL_PIXELS = 800
# Opacity of the real returns, so the rendered points drawn over them stay readable.
REAL_POINT_OPACITY = 0.5


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
    """

    run_id: str
    sweeps: Dict[str, RenderedSweep]
    scored_cuboids: List[Cuboid]
    neighbourhood_cuboids: List[Cuboid]
    membership_sha256: str
    gt_cuboids_sha256: str


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
    parser.add_argument("--all-points", action="store_true",
                        help="draw every point, not only those in the scored cuboids' neighbourhood")
    parser.add_argument("--gated", action="store_true",
                        help="draw the ray-drop-gated points instead of the ungated set the metric binds to")
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
    """Read a run the way scripts/13 does and pick the boxes the metric scores.

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
    cuboids, _ = eval_script.load_ground_truth(gt_dir)
    sweeps = load_test_lidar_sweeps(render_dir, expected_stems, accumulation_threshold)
    # Step 2: the scored boxes, from the same helper the metric uses.
    scored_class_ids = list(dict(eval_settings["scored_class_ids"]).values())
    visible_by_class, _, _ = scored_region_cuboids(
        cuboids, scored_class_ids, [sweep.lidar_to_map for sweep in sweeps],
        [sweep.real_points for sweep in sweeps], manifest["lidar"]["elevation_mapping"],
    )
    scored_cuboids = [cuboid for class_cuboids in visible_by_class.values() for cuboid in class_cuboids]
    # Step 3: the T46 neighbourhood, each box grown by its class margin (None margin: no box of the class, no growth).
    class_margins = class_margin_table(cuboids, scored_class_ids)
    neighbourhood_cuboids = [grow_cuboid_horizontally(cuboid, class_margins[cuboid.class_id] or 0.0)
                             for cuboid in scored_cuboids]
    return RunScene(
        run_id=split_record["run"]["run_id"], sweeps={sweep.stem: sweep for sweep in sweeps},
        scored_cuboids=scored_cuboids, neighbourhood_cuboids=neighbourhood_cuboids, membership_sha256=split_record["membership_sha256"],
        gt_cuboids_sha256=eval_script.file_sha256(gt_dir / eval_script.CUBOIDS_FILE_NAME),
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
        matches = [stem for stem in common_stems
                   if stem == requested or (requested.isdigit() and int(stem.split("_")[0]) == int(requested))]
        if not matches:
            # Only the first few stems are listed: enough to show the naming, short enough for one line.
            raise ValueError(f"pose {requested!r} matches no shared sweep (have {common_stems[:5]}...)")
        chosen_stems += [stem for stem in matches if stem not in chosen_stems]
    return chosen_stems


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
    if use_gated or sweep.ungated_points is None:
        rendered_lidar_points, point_set = sweep.gated_points, "gated"
    else:
        rendered_lidar_points, point_set = sweep.ungated_points, "ungated"
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


def draw_panel(plotter, pyvista_module, column: int, scene: RunScene, stem: str, arrays: Dict[str, Any]) -> None:
    """Draw one run's panel of one pose.

    Args:
        plotter: ``pyvista.Plotter`` with one row of panels.
        pyvista_module: The imported ``pyvista`` (imported lazily by ``main``).
        column: Panel column of this run.
        scene: The run's scene.
        stem: Sweep stem.
        arrays: ``panel_arrays`` result.
    """
    plotter.subplot(0, column)
    # Step 1: real returns, grey and translucent.
    if arrays["real"].shape[0]:
        plotter.add_mesh(pyvista_module.PolyData(arrays["real"]), color=np.array(REAL_POINT_RGB) / 255.0,
                         point_size=REAL_POINT_SIZE, opacity=REAL_POINT_OPACITY)
    # Step 2: rendered returns coloured inside/outside a scored cuboid.
    if arrays["rendered"].shape[0]:
        rendered_cloud = pyvista_module.PolyData(arrays["rendered"])
        rendered_cloud["colors"] = inside_outside_colors(arrays["inside"])
        plotter.add_mesh(rendered_cloud, scalars="colors", rgb=True, point_size=RENDERED_POINT_SIZE)
    # Step 3: the scored cuboids as wireframes.
    corner_points, line_cells = cuboid_wireframe_arrays(scene.scored_cuboids)
    if corner_points.shape[0]:
        plotter.add_mesh(pyvista_module.PolyData(corner_points, lines=line_cells),
                         color=np.array(CUBOID_WIREFRAME_RGB) / 255.0, line_width=WIREFRAME_LINE_WIDTH)
    inside_count, rendered_count = int(arrays["inside"].sum()), int(arrays["rendered"].shape[0])
    # Step 4: caption with the counts, on short lines (8 pt, about 6 px a character) so it fits an 800 px panel.
    plotter.add_text(f"{scene.run_id}\n{stem} ({arrays['point_set']})\n"
                     f"{inside_count} of {rendered_count} drawn points inside\nthe run's {len(scene.scored_cuboids)} scored cuboids "
                     f"({arrays['rendered_total']} rendered in this sweep)", font_size=8)


def render_pose(scenes: Sequence[RunScene], stem: str, use_gated: bool, output_path: Optional[Path],
                crop_to_neighbourhood: bool = True) -> None:
    """Draw one pose of every run side by side, to a PNG or an interactive window.

    Args:
        scenes: One scene per run.
        stem: Sweep stem (present in every scene).
        use_gated: Draw the gated point set.
        output_path: PNG to write, or None for an interactive window.
        crop_to_neighbourhood: Draw only the scored cuboids' neighbourhood (False: every point).
    """
    # PyVista (VTK) is imported here so the module loads in environments without it (tests).
    import pyvista

    plotter = pyvista.Plotter(shape=(1, len(scenes)), off_screen=output_path is not None,
                              window_size=(PANEL_PIXELS * len(scenes), PANEL_PIXELS))
    # The window is closed even when drawing fails, so one bad pose cannot leak it.
    try:
        for column, scene in enumerate(scenes):
            # Cropping to an empty neighbourhood would blank the panel, so with no scored box draw everything.
            neighbourhood = scene.neighbourhood_cuboids if (crop_to_neighbourhood and scene.scored_cuboids) else None
            arrays = panel_arrays(scene.sweeps[stem], scene.scored_cuboids, use_gated, neighbourhood)
            draw_panel(plotter, pyvista, column, scene, stem, arrays)
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
        Exit code: 0 success, 1 refused input.
    """
    arguments = build_argument_parser().parse_args(argument_list)
    project_root = find_project_root()
    # Step 1: a rerender label names one run's folder, so it cannot be combined with several runs.
    if arguments.rerender is not None and len(arguments.run_dirs) != 1:
        print("--rerender takes exactly one run folder", file=sys.stderr)
        return 1
    # Steps 2-3 run in one guard so a missing file or key is reported as a refusal (exit 1), not a traceback.
    try:
        # Step 2: eval settings and the T67 threshold, resolved as scripts/13 resolves them.
        eval_config_path = arguments.eval_config or project_root / DEFAULT_EVAL_CONFIG_RELATIVE_PATH
        eval_settings = yaml.safe_load(Path(eval_config_path).read_text())["eval"]
        accumulation_threshold = arguments.accumulation_threshold
        if accumulation_threshold is None:
            accumulation_threshold = float(eval_settings["render_accumulation_threshold"])
        # Step 3: load every run, refuse non-comparable ones, pick the poses.
        scenes = [load_run_scene(run_dir, arguments.rerender, eval_settings, accumulation_threshold, project_root)
                  for run_dir in arguments.run_dirs]
        check_scenes_comparable(scenes)
        stems = select_pose_stems(scenes, arguments.pose)
    except (ValueError, OSError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"refused: {error!r}" if isinstance(error, KeyError) else f"refused: {error}", file=sys.stderr)
        return 1
    # Step 4: one picture per pose (PNG in --output-dir, or an interactive window).
    if arguments.output_dir is not None:
        arguments.output_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        output_path = None if arguments.output_dir is None else arguments.output_dir / f"{stem}.png"
        render_pose(scenes, stem, arguments.gated, output_path, not arguments.all_points)
    return 0


if __name__ == "__main__":
    sys.exit(main())
