"""Gaussian-occupancy figure suite of one session: per-run pictures and cross-arm comparison figures.

Reads each arm's campaign runs (the 5 repeats of the ``eval.figures.campaign_profiles`` profile), draws ONE
run per arm (the median repeat on ``eval.figures.representative_metric`` of ``<run>/eval/gaussian_occupancy.json``,
or the folder given by ``--run arm=<dir>``, which may be a run of another arm or stand in for an arm without
campaign runs) and writes PNGs plus a ``figures_manifest.json`` per output folder.
Everything is read-only on the runs, GT and session data; GT cuboids are display only (CLAUDE.md rule 6).

Per drawn run, into ``<run>/eval/viz_figures/``:
    F1 ``occupancy_bev.png``          column-max occupancy over all scored cells, obstacle GT boxes, driven path, H_near boundary
    F2 ``occupancy_outcomes_3d.png``  hit / false alarm / missed strata of the headline class (whole scene), seen vs hidden
    F3 ``gaussian_density_bev.png``   opacity-weighted Gaussian density on the F1 extent (+ radar seeds for radar arms)
Across arms, into ``reports/gaussian_occupancy/figures/<session>/`` (or ``--output-dir``):
    C1 ``arms_hidden_near_3d.png`` and ``arms_scene_3d.png``   per-arm outcome panels + hit-difference panel
    C2 ``trunk_gallery_pNN.png``    one close-up tile per hidden object and arm, paginated
    C3 ``occlusion_sections.png``   vertical sections through the trunks with the most hidden GT cells
    C4 ``threshold_curves.png`` and ``occupancy_histograms.png``
    C5 ``hidden_near_heat_bev.png`` column-max occupancy over H_near & neighbourhood per arm + difference

The class is the headline group (``obstacles``) at the binding threshold ``p2_thresholds[0]`` read from the archive.
A figure that fails is reported on stderr, recorded in the manifest (status "failed") and the others are still
drawn; the exit code is then 1. A figure with nothing to draw is recorded as "skipped" with its reason (not a failure).
Run loading, consistency checks, hashing, failure bookkeeping and manifests live in
``viking_splatad.viz.figure_suite_io``.

Usage (isolated viz environment, which has PyVista and matplotlib):
    uv run --project viz python scripts/26_occupancy_figures.py --session june_tall_grass
        [--run stock=<dir> --run radar_init=<dir> --run radar_init_loss=<dir>]
        [--arms-only | --per-run-only] [--output-dir <dir>]
"""

import argparse
import importlib.util
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml
from scipy.spatial import ConvexHull

from viking_splatad.eval.detection import load_cuboids_json
from viking_splatad.eval.hidden_geometry import exclusive_evidence_labels
from viking_splatad.eval.p2_cells import HIDDEN_BAND_NEAR, hidden_cell_outcomes, p2_cell_outcomes
from viking_splatad.labels.cuboids import compute_cuboid_geometry
from viking_splatad.viz.figure_grids import (
    OUTCOME_STRATUM_NAMES,
    arm_difference_sets,
    column_max_grid,
    opacity_weighted_density,
    outcome_strata,
)
from viking_splatad.viz.figure_metrics import (
    BAND_ALL,
    class_ids_for_name,
    object_tiles,
    occupancy_samples,
    threshold_curves,
)
from viking_splatad.viz.figure_runs import FigureSettings, load_figure_settings, resolve_arm_runs
from viking_splatad.viz.figure_sections import in_slab, nearest_origin, session_sweep_origins, slice_frame
from viking_splatad.viz.figure_suite_io import (
    PROJECT_ROOT,
    ArmRun,
    FigureRecorder,
    FigureSkipped,
    HashRecorder,
    build_manifest,
    git_state,
    json_number_list,
    ledger_profile_by_run,
    parse_run_overrides,
    select_arm_runs,
    write_manifest,
)
from viking_splatad.viz.figures_mpl import (
    BOOL_CONTOUR_LEVEL,
    HALF,
    THRESHOLD_METRICS,
    arm_heat_figure,
    gaussian_density_figure,
    occlusion_section_figure,
    occupancy_bev_figure,
    occupancy_histogram_edges,
    occupancy_histogram_figure,
    threshold_curves_figure,
)
from viking_splatad.viz.figures_pyvista import (
    LEGEND_KIND_PATCH,
    compose_scene_figure,
    compose_tile_grid,
    format_tile_caption,
    outcome_stratum_styles,
    paginate,
    render_arm_panels,
    render_object_tile,
    render_outcome_strata,
)
from viking_splatad.viz.pyvista_scene import cuboid_wireframe_arrays

EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"
SLURM_CONFIG_PATH = PROJECT_ROOT / "configs" / "slurm.yaml"
LEDGER_PATH = PROJECT_ROOT / "runs" / "submitted.tsv"
RUNS_DIR = PROJECT_ROOT / "runs"
SUBMIT_RUN_PATH = PROJECT_ROOT / "scripts" / "slurm" / "submit_run.py"
GT_DIR = PROJECT_ROOT / "data" / "gt"
INTERMEDIATE_DIR = PROJECT_ROOT / "data" / "intermediate"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "gaussian_occupancy" / "figures"

PER_RUN_FIGURE_DIR = Path("eval") / "viz_figures"
GAUSSIANS_RELATIVE_PATH = Path("renders") / "gaussians.npz"
CUBOIDS_FILE_NAME = "cuboids.json"
SEEDS_RELATIVE_DIR = "radar_inputs"
SEEDS_FILE_NAME = "radar_seeds.npz"
SEEDS_MEANS_KEY = "means"

# The headline class group, the tree-trunk class (sections), the hidden band of the headline figures.
HEADLINE_CLASS_NAME = "obstacles"
TRUNK_CLASS_NAME = "tree_trunk"
HEADLINE_BAND = HIDDEN_BAND_NEAR
# Arm roles of the difference panels: first = the arm whose hits are blue, second = the baseline.
DIFFERENCE_FIRST_ARM = "radar_init_loss"
DIFFERENCE_SECOND_ARM = "stock"
# Threshold-curve bands: figure row title -> band name of figure_metrics.
CURVE_BANDS = {"H_near": HEADLINE_BAND, "whole scene": BAND_ALL}
# Padding (m) around a trunk's projected box in the occlusion sections, on every side of both axes: 2 m shows
# the occluding grass in front of the trunk and the space behind it, and crops z to the box height range
# plus the same margin.
SECTION_WINDOW_PADDING_M = 2.0
# Camera margin of the whole-scene strata figure (F2): its square render holds a wide scene, and both 0.9 (the C1
# margin) and 0.75 cut the drawn cells at the render edge, so 1.0 keeps a small border.
STRATA_FRAMING_MARGIN = 1.0
# Separator of the short figure titles ("<session> · <subject>").
TITLE_SEPARATOR = " \u00b7 "
# Histogram bin count over [0, 1]: a presentation choice (0.05-wide bins, the curve-grid step).
HISTOGRAM_BIN_COUNT = 20
# 3D scene panels are this many tiles (tile_px) on a side; a layout choice (3 x 360 px = 1080 px).
SCENE_PANEL_TILES = 3
# Gallery page file name; pages are numbered from 1.
GALLERY_FILE_TEMPLATE = "trunk_gallery_p{page_number:02d}.png"
FIRST_PAGE_NUMBER = 1
# Columns of the (N, 3) means array that are the horizontal axes, and the Gaussians' array dtype used here.
HORIZONTAL_COLUMNS = (0, 1)
GAUSSIAN_DTYPE = np.float32
# Exit codes.
EXIT_OK, EXIT_FAILED = 0, 1


# ----------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------------------------------
def threshold_grid(point_count: int, p2_thresholds: Sequence[float]) -> np.ndarray:
    """Evenly spaced [0, 1] threshold grid (ends included) that must contain every P2 threshold.

    The step is 1 / (point_count - 1); a P2 threshold t lies on the grid when t * (point_count - 1) is an integer.
    A grid point that matches a P2 threshold (``np.isclose``) is replaced by the archived value itself, as
    ``hidden_cell_outcomes`` does, so a curve evaluated at that point cuts occupancy at exactly the value
    scripts/19 scored with (a linspace point can differ from it by one rounding step and move cells that sit
    on the cut).

    Args:
        point_count: Number of grid points (>= 2).
        p2_thresholds: Binding thresholds that must be grid points.

    Returns:
        (point_count,) float array.

    Raises:
        ValueError: When a P2 threshold is not on the grid.
    """
    grid = np.linspace(0.0, 1.0, point_count)
    for threshold in p2_thresholds:
        # Step 1: every P2 threshold must match a grid point.
        matching_points = np.isclose(grid, float(threshold))
        if not matching_points.any():
            raise ValueError(f"p2 threshold {threshold} is not on the {point_count}-point curve grid; "
                             f"change eval.figures.threshold_curve_points")
        # Step 2: snap the matching point(s) to the archived value.
        grid[matching_points] = float(threshold)
    return grid


def difference_image(first_image: np.ndarray, second_image: np.ndarray) -> np.ndarray:
    """First minus second, NaN only where both are NaN; a one-sided NaN counts as 0 (documented rule).

    A NaN pixel means "no cell of the mask in this column"; when one arm lacks the column and the other has
    a value, the missing arm contributes occupancy 0, so the difference is the other arm's value (signed).

    Args:
        first_image: [y, x] image.
        second_image: [y, x] image of the same shape.

    Returns:
        [y, x] float difference.
    """
    # Step 1: remember where both are undefined, then replace every NaN by 0.
    both_undefined = np.isnan(first_image) & np.isnan(second_image)
    difference = np.nan_to_num(first_image, nan=0.0) - np.nan_to_num(second_image, nan=0.0)
    # Step 2: restore NaN where there was nothing to compare.
    difference[both_undefined] = np.nan
    return difference


def campaign_envelope(curves_per_repeat: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-threshold min and max over repeats, for each plotted metric.

    For each metric (e.g. "recall") the band spans, at every threshold, the lowest and highest value of BOTH
    the pooled and the mean-over-objects curve over all repeats, so it contains every line the figure draws for
    the arm.
    NaN is ignored per threshold (``fmin``/``fmax``); all-NaN thresholds stay NaN.

    Args:
        curves_per_repeat: One ``threshold_curves`` dict per repeat.

    Returns:
        metric key -> (thresholds, lower, upper).
    """
    envelope: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    thresholds = np.asarray(curves_per_repeat[0]["thresholds"], dtype=float)
    for metric_key, _label, pooled_key, mean_key in THRESHOLD_METRICS:
        # Step 1: stack every repeat's pooled and mean curve as rows: (2 * repeats, thresholds).
        stacked = np.vstack([np.asarray(curves[key], dtype=float) for curves in curves_per_repeat
                             for key in (pooled_key, mean_key)])
        # Step 2: NaN-ignoring min / max over the rows.
        envelope[metric_key] = (thresholds, np.fmin.reduce(stacked, axis=0), np.fmax.reduce(stacked, axis=0))
    return envelope


def in_window(coordinates: np.ndarray, window: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Mask of (M, 2) coordinates inside a closed window.

    Args:
        coordinates: (M, 2) section coordinates.
        window: (lower, upper) from ``padded_window``.

    Returns:
        (M,) bool.
    """
    lower, upper = window
    return np.all((coordinates >= lower) & (coordinates <= upper), axis=1)


def padded_window(corner_coordinates: np.ndarray, padding_m: float) -> Tuple[np.ndarray, np.ndarray]:
    """(along, z) window: the box's projected extent padded by ``padding_m`` metres on each side of each axis.

    window = [min(corners) - padding_m, max(corners) + padding_m] per axis.

    Args:
        corner_coordinates: (K, 2) projected box corners.
        padding_m: Padding (m).

    Returns:
        (lower (2,), upper (2,)).
    """
    return corner_coordinates.min(axis=0) - padding_m, corner_coordinates.max(axis=0) + padding_m


def boxes_inside_extent(polygons: Sequence[np.ndarray], extent: Tuple[float, float, float, float]
                        ) -> List[np.ndarray]:
    """Keep the polygons whose centre (mean of the corners) lies inside the extent.

    Args:
        polygons: (K, 2) corner arrays (m).
        extent: (xmin, xmax, ymin, ymax) in metres.

    Returns:
        The kept polygons, in order.
    """
    x_min, x_max, y_min, y_max = extent
    centres = [polygon.mean(axis=0) for polygon in polygons]
    return [polygon for polygon, centre in zip(polygons, centres)
            if x_min <= centre[0] <= x_max and y_min <= centre[1] <= y_max]


def convex_polygon(points: np.ndarray) -> np.ndarray:
    """Counter-clockwise convex hull of 2D points (outline of a projected box).

    Args:
        points: (K, 2) points, K >= 3 and not collinear.

    Returns:
        (H, 2) hull vertices in order.
    """
    return points[ConvexHull(points).vertices]


def short_run_id(run_id: str, session: str) -> str:
    """Run id without its leading ``<session>-`` (shorter titles).

    Args:
        run_id: Run folder name.
        session: Session name.

    Returns:
        The shortened id (unchanged when the prefix is absent).
    """
    prefix = f"{session}-"
    return run_id[len(prefix):] if run_id.startswith(prefix) else run_id


def scene_outcomes(cells: Mapping[str, np.ndarray], class_name: str) -> Dict[str, np.ndarray]:
    """Whole-scene outcome sets of a class from ``p2_cell_outcomes`` (the metric's own rule).

    With O occupied, N the scored neighbourhood and G truth inside it: hit = O & G, false_alarm = O & N & ~G,
    missed = G & ~O, context = N.

    Args:
        cells: Cells archive arrays.
        class_name: Class or group name.

    Returns:
        {"context", "hit", "false_alarm", "missed"} (V,) bool.
    """
    outcomes = p2_cell_outcomes(cells, [class_name])
    occupied, scored, on_truth = outcomes["occupied"], outcomes["scored"], outcomes["on_ground_truth"]
    return {"context": scored, "hit": occupied & scored & on_truth,
            "false_alarm": occupied & scored & ~on_truth, "missed": on_truth & ~occupied}


def repeat_count_text(repeat_count_by_arm: Mapping[str, int], arm_labels: Mapping[str, str]) -> str:
    """Words for the repeats behind the spread bands, for a figure title.

    Args:
        repeat_count_by_arm: arm -> number of campaign runs (0 = the arm has no band).
        arm_labels: arm -> display label.

    Returns:
        ``n = 5 repeats per arm`` when every arm has the same positive count, else the per-arm counts.
    """
    distinct_counts = set(repeat_count_by_arm.values())
    if len(distinct_counts) == 1 and next(iter(distinct_counts)) > 0:
        return f"n = {next(iter(distinct_counts))} repeats per arm"
    return "repeats: " + ", ".join(f"{arm_labels[arm]} {count}" + ("" if count else " (no band)")
                                   for arm, count in repeat_count_by_arm.items())


def finite_summary(image: np.ndarray) -> Dict[str, Any]:
    """Numbers behind a map: how many columns hold a value and their range.

    Args:
        image: [y, x] image with NaN for "no cell".

    Returns:
        {"finite_columns", "min", "max"}; min and max are None when no column is finite.
    """
    finite_mask = np.isfinite(image)
    has_values = bool(finite_mask.any())
    return {"finite_columns": int(finite_mask.sum()),
            "min": float(np.min(image[finite_mask])) if has_values else None,
            "max": float(np.max(image[finite_mask])) if has_values else None}


def difference_colour_names(settings: FigureSettings) -> Tuple[str, str]:
    """Settings colour names of the difference panels' two arms, ``only_<arm>`` of the constants' arms.

    Args:
        settings: Figure settings (colours).

    Returns:
        (first-arm colour name, second-arm colour name).

    Raises:
        ValueError: When ``eval.figures.colours`` lacks either name.
    """
    colour_names = (f"only_{DIFFERENCE_FIRST_ARM}", f"only_{DIFFERENCE_SECOND_ARM}")
    missing_names = [name for name in colour_names if name not in settings.colours]
    if missing_names:
        raise ValueError(f"eval.figures.colours lacks {missing_names} (one 'only_<arm>' colour per compared arm)")
    return colour_names


def seeds_path_for_profile(profile: str, slurm_profiles: Mapping[str, Mapping[str, Any]], session: str,
                           intermediate_dir: Path) -> Optional[Path]:
    """Radar seed file of a profile (None for a profile without radar seeds).

    Same anchor rule as submit_run: the artifact of ``radar_anchor_profile``, else of the profile itself.

    Args:
        profile: Slurm profile name.
        slurm_profiles: ``profiles`` mapping of configs/slurm.yaml.
        session: Session name.
        intermediate_dir: ``data/intermediate``.

    Returns:
        Path of ``radar_seeds.npz``, or None when the profile has no ``radar_variant``.
    """
    profile_config = slurm_profiles[profile]
    if "radar_variant" not in profile_config:
        return None
    anchor_profile = profile_config.get("radar_anchor_profile", profile)
    return intermediate_dir / session / SEEDS_RELATIVE_DIR / anchor_profile / SEEDS_FILE_NAME


# ----------------------------------------------------------------------------------------------
# Data containers and loading
# ----------------------------------------------------------------------------------------------
@dataclass
class FigureContext:
    """Everything the figure functions share for one session.

    Attributes:
        session: Session name.
        settings: ``eval.figures`` settings.
        eval_section: The ``eval`` section of configs/eval.yaml (class groups).
        cuboids: GT cuboids of the session (display only).
        sweep_origins: (N, 3) lidar sweep origins, i.e. the driven path (map frame).
        class_ids: Class ids of the headline group.
        p2_thresholds: Every evaluated threshold, from the archive.
        threshold: Binding threshold ``p2_thresholds[0]``.
        curve_grid: Threshold grid of the curves (contains every P2 threshold).
        drawn: arm -> the run drawn for it.
        repeats: arm -> its campaign runs ([] when an override stands in).
        seeds_paths: arm -> radar seed file of the drawn run's profile, or None.
        git: ``git_state`` result.
        arms_info: arm -> manifest entry on how its run was chosen.
        hashes: sha256 recorder of every input read.
        cuboids_path: GT cuboid file (hashed when used).
        gaussian_cache: Gaussians file -> (means, opacities), each file loaded once.
    """

    session: str
    settings: FigureSettings
    eval_section: Mapping[str, Any]
    cuboids: List[Any]
    sweep_origins: np.ndarray
    class_ids: Tuple[int, ...]
    p2_thresholds: np.ndarray
    threshold: float
    curve_grid: np.ndarray
    drawn: Dict[str, ArmRun]
    repeats: Dict[str, List[ArmRun]]
    seeds_paths: Dict[str, Optional[Path]]
    git: Dict[str, Any]
    arms_info: Dict[str, Any]
    hashes: HashRecorder
    cuboids_path: Path
    gaussian_cache: Dict[Path, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    @property
    def reference_cells(self) -> Dict[str, np.ndarray]:
        """Cells of the first drawn arm (model-independent arrays are shared by all runs)."""
        return self.drawn[self.settings.arm_order[0]].cells

    @property
    def voxel_size(self) -> float:
        """Voxel edge (m)."""
        return float(self.reference_cells["voxel_size"])

    def arm_label(self, arm: str) -> str:
        """Display label of an arm.

        Args:
            arm: Arm key.

        Returns:
            Label text.
        """
        return self.settings.arm_labels[arm]

    def run_summary(self) -> str:
        """``arm: run id`` list of the drawn runs, for titles.

        Returns:
            Text such as ``stock: <id>; radar init: <id>``.
        """
        return "; ".join(f"{self.arm_label(arm)}: {short_run_id(self.drawn[arm].run_id, self.session)}"
                         for arm in self.settings.arm_order)

    def run_note(self) -> str:
        """Run identification printed under cross-arm figures (wrapped by the renderer).

        Returns:
            ``drawn runs: stock = <id>; radar init = <id>; ...`` with the session prefix removed from ids.
        """
        return "drawn runs: " + "; ".join(
            f"{self.arm_label(arm)} = {short_run_id(self.drawn[arm].run_id, self.session)}"
            for arm in self.settings.arm_order)

    def single_run_note(self, run: "ArmRun") -> str:
        """Run identification printed under a single-run figure.

        Args:
            run: The drawn run.

        Returns:
            ``drawn run: <id without session prefix>``.
        """
        return f"drawn run: {short_run_id(run.run_id, self.session)}"

    def short_title(self, subject: str, arm: Optional[str] = None) -> str:
        """Short figure title ``<session> · [<arm label> · ]<subject>``.

        Args:
            subject: What the figure shows.
            arm: Arm key for single-run figures, or None.

        Returns:
            The title.
        """
        parts = [self.session] + ([self.arm_label(arm)] if arm is not None else []) + [subject]
        return TITLE_SEPARATOR.join(parts)

    def scene_window(self, panel_count: int) -> Tuple[int, int]:
        """Pixel window of a 3D scene with ``panel_count`` side-by-side panels.

        Args:
            panel_count: Number of panels.

        Returns:
            (width, height) in pixels.
        """
        side = self.settings.tile_px * SCENE_PANEL_TILES
        return side * panel_count, side


def load_gaussians(context: FigureContext, run: ArmRun) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian means and opacities of a run (float32, cached; only these two keys are read).

    Args:
        context: Shared context (cache and hashes).
        run: The run.

    Returns:
        (means (N, 3), opacities (N,)); opacities are already sigmoid-activated, map frame.
    """
    gaussians_path = run.run_dir / GAUSSIANS_RELATIVE_PATH
    context.hashes.record(gaussians_path)
    if gaussians_path not in context.gaussian_cache:
        with np.load(gaussians_path, allow_pickle=False) as archive:
            context.gaussian_cache[gaussians_path] = (archive["means"].astype(GAUSSIAN_DTYPE, copy=False),
                                                      archive["opacities"].astype(GAUSSIAN_DTYPE, copy=False))
    return context.gaussian_cache[gaussians_path]


def load_seed_means(context: FigureContext, arm: str) -> Optional[np.ndarray]:
    """Radar seed positions of an arm (None for an arm without seeds).

    Args:
        context: Shared context.
        arm: Arm key.

    Returns:
        (S, 3) map-frame means, or None.
    """
    seeds_path = context.seeds_paths[arm]
    if seeds_path is None:
        return None
    context.hashes.record(seeds_path)
    with np.load(seeds_path, allow_pickle=False) as archive:
        return np.asarray(archive[SEEDS_MEANS_KEY], dtype=float)


def import_pyvista() -> Any:
    """Import PyVista lazily (only the viz environment has it) and select off-screen rendering.

    Returns:
        The ``pyvista`` module.
    """
    import pyvista
    pyvista.OFF_SCREEN = True
    return pyvista


def save_matplotlib_figure(figure: Any, output_path: Path, dpi: int) -> None:
    """Save a figure at the settings dpi and close it.

    Args:
        figure: Matplotlib figure.
        output_path: PNG path (parent is created).
        dpi: Resolution.
    """
    import matplotlib.pyplot as plt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Tight bounding box so no title, legend or note is ever clipped at the image edge.
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    finally:
        plt.close(figure)


def save_rgb_image(image: np.ndarray, output_path: Path) -> None:
    """Write an RGB uint8 array as a PNG.

    Args:
        image: (H, W, 3) array.
        output_path: PNG path (parent is created).
    """
    import matplotlib.pyplot as plt
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, image)


# ----------------------------------------------------------------------------------------------
# Per-run figures F1-F3
# ----------------------------------------------------------------------------------------------
def obstacle_footprints(context: FigureContext) -> List[np.ndarray]:
    """Horizontal outlines (convex hull of the projected corners) of the headline-class GT boxes.

    Args:
        context: Shared context.

    Returns:
        List of (H, 2) polygons in metres.
    """
    boxes = [cuboid for cuboid in context.cuboids if cuboid.class_id in context.class_ids]
    return [convex_polygon(compute_cuboid_geometry(cuboid).corners[:, list(HORIZONTAL_COLUMNS)]) for cuboid in boxes]


def bev_extent(cells: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    """Column-max occupancy image over ALL scored cells, and its extent (the union mask makes extents agree).

    Args:
        cells: Cells archive arrays.

    Returns:
        (image [y, x], extent).
    """
    union_mask = np.ones(len(cells["cell_indices"]), dtype=bool)
    return column_max_grid(cells["cell_indices"], cells["occupancy"], union_mask, cells["grid_origin"],
                           float(cells["voxel_size"]))


def draw_occupancy_bev(context: FigureContext, run: ArmRun, output_path: Path) -> None:
    """F1: occupancy bird's-eye map with obstacle boxes, driven path and the H_near boundary.

    Args:
        context: Shared context.
        run: Drawn run.
        output_path: PNG path.
    """
    cells = run.cells
    context.hashes.record(run.cells_path)
    context.hashes.record(context.cuboids_path)
    # Step 1: occupancy and H_near (as float) on the same union mask, hence the same extent.
    image, extent = bev_extent(cells)
    union_mask = np.ones(len(cells["cell_indices"]), dtype=bool)
    near_image, near_extent = column_max_grid(cells["cell_indices"], cells["hidden_near"].astype(float), union_mask,
                                              cells["grid_origin"], context.voxel_size)
    if near_extent != extent:
        raise ValueError(f"occupancy extent {extent} and H_near extent {near_extent} differ; they must share one")
    # Step 2: NaN > level is False, so only columns holding an H_near cell become True.
    near_contour = near_image > BOOL_CONTOUR_LEVEL
    figure = occupancy_bev_figure(
        image, extent, context.settings, boxes_xy=boxes_inside_extent(obstacle_footprints(context), extent),
        path_xy=context.sweep_origins[:, list(HORIZONTAL_COLUMNS)], near_contour=near_contour,
        title=context.short_title("column-max occupancy; GT boxes, path (red), H_near (green)", run.arm),
        run_note=context.single_run_note(run))
    save_matplotlib_figure(figure, output_path, context.settings.dpi)


def draw_occupancy_outcomes_3d(context: FigureContext, run: ArmRun, output_path: Path) -> Dict[str, int]:
    """F2: outcome strata of the headline class on the whole scene (hit/miss split by seen vs hidden).

    Args:
        context: Shared context.
        run: Drawn run.
        output_path: PNG path.

    Returns:
        Cell count per stratum name (also recorded in the manifest by the caller).
    """
    cells = run.cells
    context.hashes.record(run.cells_path)
    # Step 1: the metric's own outcomes, then the strata (hidden = evidence label other than "seen").
    outcomes = scene_outcomes(cells, HEADLINE_CLASS_NAME)
    strata = outcome_strata(outcomes, cells["evidence"])
    window_size = context.scene_window(1)
    image = render_outcome_strata(
        import_pyvista(), cells["centres"], context.voxel_size, strata, context.settings, window_size=window_size,
        title="", draw_text_legend=False, framing_margin=STRATA_FRAMING_MARGIN)
    # Step 2: colour legend with the per-stratum counts, around the screenshot.
    counts = {name: int((strata == index).sum()) for index, name in enumerate(OUTCOME_STRATUM_NAMES)}
    styles = outcome_stratum_styles(context.settings)
    legend_entries = [(f"{name}: {counts[name]}", styles[name][0], LEGEND_KIND_PATCH)
                      for name in OUTCOME_STRATUM_NAMES]
    figure = compose_scene_figure(
        image, context.short_title(f"{HEADLINE_CLASS_NAME} at occupancy >= {context.threshold}, whole scene",
                                   run.arm),
        legend_entries, context.settings, run_note=context.single_run_note(run))
    save_matplotlib_figure(figure, output_path, context.settings.dpi)
    return counts


def draw_gaussian_density_bev(context: FigureContext, run: ArmRun, output_path: Path) -> None:
    """F3: opacity-weighted Gaussian density over the F1 extent, with radar seeds for radar arms.

    Args:
        context: Shared context.
        run: Drawn run.
        output_path: PNG path.
    """
    context.hashes.record(run.cells_path)
    # Step 1: the F1 extent; bin edge = configured bev_bin_m, else the voxel size.
    _, extent = bev_extent(run.cells)
    bin_size = context.settings.bev_bin_m if context.settings.bev_bin_m is not None else context.voxel_size
    means, opacities = load_gaussians(context, run)
    density, edges_x, edges_y = opacity_weighted_density(means[:, list(HORIZONTAL_COLUMNS)], opacities, extent,
                                                         bin_size)
    seeds = load_seed_means(context, run.arm)
    seeds_xy = None if seeds is None else seeds[:, list(HORIZONTAL_COLUMNS)]
    figure = gaussian_density_figure(
        density, edges_x, edges_y, context.settings, seeds_xy=seeds_xy,
        title=context.short_title(f"Gaussian density (bin {bin_size:.3g} m)", run.arm),
        run_note=context.single_run_note(run))
    save_matplotlib_figure(figure, output_path, context.settings.dpi)


def context_manifest(context: FigureContext, scope_label: str, figures: Mapping[str, Any],
                     reported_numbers: Mapping[str, Any]) -> Dict[str, Any]:
    """Manifest of one scope (the current hash scope of the context is the scope's input list).

    Args:
        context: Shared context.
        scope_label: "arms" or "run:<arm>".
        figures: name -> status record.
        reported_numbers: Numbers printed in or behind the scope's figures.

    Returns:
        JSON-serialisable dict.
    """
    return build_manifest(
        session=context.session, scope_label=scope_label, git=context.git, class_name=HEADLINE_CLASS_NAME,
        binding_threshold=context.threshold, p2_thresholds=context.p2_thresholds,
        representative_metric=context.settings.representative_metric, arms_info=context.arms_info,
        figures=figures, hashes=context.hashes.scope, reported_numbers=reported_numbers)


def generate_per_run_figures(context: FigureContext, recorder_by_arm: Optional[Dict[str, FigureRecorder]] = None
                             ) -> Dict[str, FigureRecorder]:
    """Draw F1-F3 for each arm's drawn run and write each run's manifest.

    Args:
        context: Shared context.
        recorder_by_arm: Optional pre-made recorders (tests).

    Returns:
        arm -> recorder with the per-figure status.
    """
    recorders: Dict[str, FigureRecorder] = recorder_by_arm if recorder_by_arm is not None else {}
    for arm in context.settings.arm_order:
        run = context.drawn[arm]
        recorder = recorders.setdefault(arm, FigureRecorder())
        output_dir = run.run_dir / PER_RUN_FIGURE_DIR
        context.hashes.begin_scope()
        reported: Dict[str, Any] = {}
        # Step 1: one attempt per figure; a failure is recorded and the next figure still runs.
        recorder.attempt("occupancy_bev", output_dir / "occupancy_bev.png",
                         lambda: draw_occupancy_bev(context, run, output_dir / "occupancy_bev.png"))
        recorder.attempt("occupancy_outcomes_3d", output_dir / "occupancy_outcomes_3d.png",
                         lambda: reported.__setitem__("strata_counts", draw_occupancy_outcomes_3d(
                             context, run, output_dir / "occupancy_outcomes_3d.png")))
        recorder.attempt("gaussian_density_bev", output_dir / "gaussian_density_bev.png",
                         lambda: draw_gaussian_density_bev(context, run, output_dir / "gaussian_density_bev.png"))
        # Step 2: the run's manifest, written even when figures failed.
        write_manifest(output_dir, context_manifest(context, f"run:{arm}", recorder.figures, reported))
    return recorders


# ----------------------------------------------------------------------------------------------
# Cross-arm figures C1-C5
# ----------------------------------------------------------------------------------------------
def arms_in_order(context: FigureContext) -> List[str]:
    """Arm keys in settings order.

    Args:
        context: Shared context.

    Returns:
        List of arm keys.
    """
    return list(context.settings.arm_order)


def draw_arm_outcome_panels(context: FigureContext, output_path: Path, band: str) -> Dict[str, Any]:
    """C1: per-arm outcome panels plus the hit-difference panel (first = init+loss, second = stock).

    Args:
        context: Shared context.
        output_path: PNG path.
        band: ``HEADLINE_BAND`` (H_near cells) or ``BAND_ALL`` (whole scene).

    Returns:
        Outcome cell counts per arm (for the manifest).
    """
    # Step 1: outcome sets per arm on the band.
    outcomes_by_arm: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in arms_in_order(context):
        context.hashes.record(context.drawn[arm].cells_path)
        cells = context.drawn[arm].cells
        outcomes_by_arm[arm] = (scene_outcomes(cells, HEADLINE_CLASS_NAME) if band == BAND_ALL else
                                hidden_cell_outcomes(cells, HEADLINE_CLASS_NAME, context.threshold, band))
    # Step 2: hits of the two compared arms -> only-first / only-second / both.
    difference_sets = arm_difference_sets(outcomes_by_arm[DIFFERENCE_FIRST_ARM]["hit"],
                                          outcomes_by_arm[DIFFERENCE_SECOND_ARM]["hit"])
    band_text = "H_near" if band == HEADLINE_BAND else "whole scene"
    titles = [context.arm_label(arm) for arm in arms_in_order(context)]
    titles.append(f"hits: {context.arm_label(DIFFERENCE_FIRST_ARM)} vs {context.arm_label(DIFFERENCE_SECOND_ARM)}")
    window_size = context.scene_window(len(titles))
    first_colour_name, second_colour_name = difference_colour_names(context.settings)
    image = render_arm_panels(import_pyvista(), context.reference_cells["centres"], context.voxel_size,
                              outcomes_by_arm, difference_sets, context.settings, window_size=window_size,
                              title_by_panel=titles, first_colour_name=first_colour_name,
                              second_colour_name=second_colour_name)
    # Step 3: legend of the outcome colours and of the difference-panel colours, in a figure with a short title.
    colours = context.settings.colours
    legend_entries = [("hit", colours["hit"], LEGEND_KIND_PATCH),
                      ("false alarm", colours["false_alarm"], LEGEND_KIND_PATCH),
                      ("missed", colours["missed"], LEGEND_KIND_PATCH),
                      ("context", colours["context"], LEGEND_KIND_PATCH),
                      (f"only {context.arm_label(DIFFERENCE_FIRST_ARM)}", colours[first_colour_name],
                       LEGEND_KIND_PATCH),
                      (f"only {context.arm_label(DIFFERENCE_SECOND_ARM)}", colours[second_colour_name],
                       LEGEND_KIND_PATCH),
                      ("both", colours["both"], LEGEND_KIND_PATCH)]
    figure = compose_scene_figure(
        image, context.short_title(f"{band_text} {HEADLINE_CLASS_NAME} @ {context.threshold}"), legend_entries,
        context.settings, panel_count=len(titles), run_note=context.run_note())
    save_matplotlib_figure(figure, output_path, context.settings.dpi)
    counts = {arm: {name: int(mask.sum()) for name, mask in outcomes_by_arm[arm].items()}
              for arm in arms_in_order(context)}
    counts["difference"] = {name: int(mask.sum()) for name, mask in difference_sets.items()}
    return counts


def gallery_rows(context: FigureContext, tiles_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> List[int]:
    """Box ids of the gallery rows: objects with at least one hidden GT cell, sorted by box id.

    The hidden GT cell count does not depend on the run, so the first arm's tiles decide.

    Args:
        context: Shared context.
        tiles_by_arm: arm -> ``object_tiles`` list.

    Returns:
        Sorted box ids.
    """
    first_arm_tiles = tiles_by_arm[context.settings.arm_order[0]]
    return sorted(tile["box_id"] for tile in first_arm_tiles if tile["hidden_gt_cell_count"] >= 1)


@dataclass
class GalleryPlan:
    """What the gallery pages draw, prepared once for all pages.

    Attributes:
        tiles_by_arm: arm -> box id -> ``object_tiles`` entry (headline class, H_near).
        false_alarm_by_arm: arm -> (V,) false-alarm cells of the arm's drawn run on H_near.
        box_ids: Sorted box ids of the gallery rows.
    """

    tiles_by_arm: Dict[str, Dict[int, Mapping[str, Any]]]
    false_alarm_by_arm: Dict[str, np.ndarray]
    box_ids: List[int]


def prepare_gallery(context: FigureContext) -> GalleryPlan:
    """Tiles, false-alarm cells (one ``hidden_cell_outcomes`` per arm) and rows of the trunk gallery.

    Args:
        context: Shared context.

    Returns:
        The plan.

    Raises:
        FigureSkipped: When no object has a hidden GT cell on the headline band.
    """
    tiles_by_arm: Dict[str, Dict[int, Mapping[str, Any]]] = {}
    false_alarm_by_arm: Dict[str, np.ndarray] = {}
    # Step 1: per arm, its object tiles (keyed by box id) and its H_near false-alarm cells.
    for arm in arms_in_order(context):
        context.hashes.record(context.drawn[arm].cells_path)
        cells = context.drawn[arm].cells
        tile_list = object_tiles(cells, HEADLINE_CLASS_NAME, context.class_ids, context.threshold, HEADLINE_BAND)
        tiles_by_arm[arm] = {tile["box_id"]: tile for tile in tile_list}
        false_alarm_by_arm[arm] = hidden_cell_outcomes(cells, HEADLINE_CLASS_NAME, context.threshold,
                                                       HEADLINE_BAND)["false_alarm"]
    # Step 2: rows = objects with hidden GT; none means nothing to draw.
    box_ids = gallery_rows(context, {arm: list(tiles.values()) for arm, tiles in tiles_by_arm.items()})
    if not box_ids:
        raise FigureSkipped(f"no {HEADLINE_CLASS_NAME} object has a hidden GT cell on H_near")
    return GalleryPlan(tiles_by_arm, false_alarm_by_arm, box_ids)


def draw_gallery_page(context: FigureContext, plan: GalleryPlan, page_box_ids: Sequence[int], page_number: int,
                      page_path: Path, reported: Dict[str, Any]) -> None:
    """Render and save one page of tiles (rows = objects, columns = arms).

    Args:
        context: Shared context.
        plan: The gallery plan.
        page_box_ids: Box ids of this page's rows.
        page_number: Page number (1-based), printed in the title.
        page_path: PNG path of this page.
        reported: Receives, per box id and arm, the numbers printed in the captions.
    """
    arms = arms_in_order(context)
    cuboid_by_id = {cuboid.box_id: cuboid for cuboid in context.cuboids}
    centres = context.reference_cells["centres"]
    window_size = (context.settings.tile_px, context.settings.tile_px)
    pyvista_module = import_pyvista()
    images, captions, row_labels = [], [], []
    # Step 1: one row per object; one rendered tile, caption and reported entry per arm.
    for box_id in page_box_ids:
        cuboid = cuboid_by_id[box_id]
        wireframe = cuboid_wireframe_arrays([cuboid])
        row_images, row_captions = [], []
        for arm in arms:
            cells = context.drawn[arm].cells
            tile = plan.tiles_by_arm[arm][box_id]
            row_images.append(render_object_tile(
                pyvista_module, centres, context.voxel_size, tile, cells["occupancy"],
                cells[f"truth_{HEADLINE_CLASS_NAME}"], cells["hidden_near"], plan.false_alarm_by_arm[arm],
                context.settings, window_size=window_size, cuboid_wireframe=wireframe))
            row_captions.append(format_tile_caption(tile["recall_H"], tile["baseline_recall"]))
            reported.setdefault(str(box_id), {})[arm] = {
                "recall_H": tile["recall_H"], "precision_H": tile["precision_H"],
                "baseline_recall": tile["baseline_recall"],
                "hidden_gt_cell_count": int(tile["hidden_gt_cell_count"]), "class_name": cuboid.class_name}
        images.append(row_images)
        captions.append(row_captions)
        row_labels.append(f"box {box_id} {cuboid.class_name}\n"
                          f"{int(plan.tiles_by_arm[arms[0]][box_id]['hidden_gt_cell_count'])} hidden GT cells")
    # Step 2: lay the tiles in a grid (rows = objects, columns = arms) and save the page.
    grid_figure = compose_tile_grid(
        images, row_labels, [context.arm_label(arm) for arm in arms], captions, context.settings,
        title=context.short_title(f"{HEADLINE_CLASS_NAME} in H_near, page {page_number}"),
        run_note=context.run_note(), show_key=True)
    save_matplotlib_figure(grid_figure, page_path, context.settings.dpi)


def draw_trunk_gallery(context: FigureContext, output_dir: Path, recorder: FigureRecorder) -> Dict[str, Any]:
    """C2: one close-up per hidden object and arm, paginated; each page is its own recorded figure.

    A failure while preparing the tiles is recorded once as the figure "trunk_gallery" (a skip likewise).

    Args:
        context: Shared context.
        output_dir: Folder for the pages.
        recorder: Records one entry per page.

    Returns:
        Per-object, per-arm numbers printed in the captions (for the manifest).
    """
    # Step 1: tiles and false alarms per arm, rows = objects with hidden GT.
    prepared, plan = recorder.guard("trunk_gallery", output_dir, lambda: prepare_gallery(context))
    if not prepared:
        return {}
    context.hashes.record(context.cuboids_path)
    reported: Dict[str, Any] = {}
    # Step 2: pages of rows_per_page objects, each attempted on its own.
    for page_index, page_box_ids in enumerate(paginate(plan.box_ids, context.settings.rows_per_page)):
        page_number = FIRST_PAGE_NUMBER + page_index
        page_path = output_dir / GALLERY_FILE_TEMPLATE.format(page_number=page_number)
        recorder.attempt(page_path.stem, page_path,
                         lambda page_box_ids=page_box_ids, page_number=page_number, page_path=page_path:
                         draw_gallery_page(context, plan, page_box_ids, page_number, page_path, reported))
    return reported


def draw_occlusion_sections(context: FigureContext, output_path: Path) -> Dict[str, Any]:
    """C3: vertical sections through the trunks with the most hidden GT cells, one column per arm.

    Per trunk: the section plane through the cuboid centre and the nearest sweep origin; cells in a slab one
    voxel thick (half width voxel_size / 2) and in the (along, z) window around the box padded by
    ``SECTION_WINDOW_PADDING_M``; Gaussians and radar seeds of the arm in the same slab and window.

    Args:
        context: Shared context.
        output_path: PNG path.

    Returns:
        Section geometry numbers for the manifest.

    Raises:
        FigureSkipped: When no tree trunk has a hidden GT cell on H_near (nothing to section).
    """
    cells = context.reference_cells
    context.hashes.record(context.cuboids_path)
    trunk_ids = class_ids_for_name(TRUNK_CLASS_NAME, context.eval_section)
    # Step 1: trunks with hidden GT, ranked by hidden GT cells (descending), ties by box id; keep section_trunk_count.
    tile_list = [tile for tile in object_tiles(cells, TRUNK_CLASS_NAME, trunk_ids, context.threshold, HEADLINE_BAND)
                 if tile["hidden_gt_cell_count"] >= 1]
    if not tile_list:
        raise FigureSkipped(f"no {TRUNK_CLASS_NAME} object has a hidden GT cell on H_near")
    ranked = sorted(tile_list, key=lambda tile: (-tile["hidden_gt_cell_count"], tile["box_id"]))
    chosen = ranked[:context.settings.section_trunk_count]
    cuboid_by_id = {cuboid.box_id: cuboid for cuboid in context.cuboids}
    half_width = context.voxel_size * HALF
    evidence_labels = exclusive_evidence_labels(cells["evidence"])
    sections, reported = [], {}
    for tile in chosen:
        cuboid = cuboid_by_id[tile["box_id"]]
        # Step 2: frame from the nearest sweep origin to the cuboid centre.
        centre = np.asarray(cuboid.center_xyz, dtype=float)
        _, origins = nearest_origin(centre[None, :], context.sweep_origins)
        frame = slice_frame(centre, origins[0])
        # Step 3: the box projected into (along, z): outline = hull, window = padded extent. An infinite
        # half width keeps every corner, so in_slab acts as a pure projection.
        _, corner_coordinates = in_slab(compute_cuboid_geometry(cuboid).corners, frame, np.inf)
        window = padded_window(corner_coordinates, SECTION_WINDOW_PADDING_M)
        # Step 4: cells in the slab and window, with the evidence label of each.
        slab_mask, slab_coordinates = in_slab(cells["centres"], frame, half_width)
        keep = in_window(slab_coordinates, window)
        cell_rows = np.flatnonzero(slab_mask)[keep]
        by_arm = {}
        for arm in arms_in_order(context):
            # Step 5: this arm's occupancy of those cells, its Gaussians and seeds in the slab and window.
            means, opacities = load_gaussians(context, context.drawn[arm])
            gaussian_mask, gaussian_coordinates = in_slab(means, frame, half_width)
            gaussian_keep = in_window(gaussian_coordinates, window)
            seeds = load_seed_means(context, arm)
            seed_coordinates = None
            if seeds is not None:
                _, seed_slab_coordinates = in_slab(seeds, frame, half_width)
                seed_coordinates = seed_slab_coordinates[in_window(seed_slab_coordinates, window)]
            by_arm[arm] = {"occupancy": context.drawn[arm].cells["occupancy"][cell_rows],
                           "gaussian_coords": gaussian_coordinates[gaussian_keep],
                           "gaussian_opacity": opacities[gaussian_mask][gaussian_keep].astype(float),
                           "seed_coords": seed_coordinates}
        sections.append({"label": f"box {tile['box_id']} ({tile['hidden_gt_cell_count']} hidden GT cells)",
                         "cells_coords": slab_coordinates[keep], "evidence_labels": evidence_labels[cell_rows],
                         "box_polygon": convex_polygon(corner_coordinates), "window": window, "by_arm": by_arm})
        reported[str(tile["box_id"])] = {
            "hidden_gt_cell_count": int(tile["hidden_gt_cell_count"]), "sweep_origin": origins[0].tolist(),
            "cells_in_section": int(len(cell_rows)),
            "gaussians_in_section": {arm: int(len(by_arm[arm]["gaussian_opacity"])) for arm in by_arm}}
    for arm in arms_in_order(context):
        context.hashes.record(context.drawn[arm].cells_path)
    figure = occlusion_section_figure(
        sections, context.settings, voxel_size=context.voxel_size,
        title=context.short_title(f"occlusion sections of the {len(sections)} trunks with the most hidden GT cells"),
        run_note=context.run_note())
    save_matplotlib_figure(figure, output_path, context.settings.dpi)
    return reported


def curves_of_run(context: FigureContext, run: ArmRun, band: str) -> Dict[str, np.ndarray]:
    """Threshold curves of one run on one band (always with the headline class ids).

    Args:
        context: Shared context.
        run: The run.
        band: Band name.

    Returns:
        ``threshold_curves`` dict.
    """
    return threshold_curves(run.cells, HEADLINE_CLASS_NAME, context.curve_grid, band, class_ids=context.class_ids)


def draw_threshold_curves(context: FigureContext, output_path: Path) -> Dict[str, Any]:
    """C4a: curves of the drawn run per arm with the min/max band over all repeats.

    An arm without campaign runs (an override stands in) has its curves but no band.

    Args:
        context: Shared context.
        output_path: PNG path.

    Returns:
        The numbers drawn: the grid, the repeat count per arm, each drawn curve and each band envelope
        (NaN as None).
    """
    arms = arms_in_order(context)
    drawn_curves: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    campaign_bands: Dict[str, Dict[str, Any]] = {}
    reported: Dict[str, Any] = {
        "curve_grid": json_number_list(context.curve_grid),
        "repeat_counts": {arm: len(context.repeats[arm]) for arm in arms}, "drawn_curves": {}, "campaign_bands": {}}
    for row_title, band in CURVE_BANDS.items():
        drawn_curves[row_title] = {arm: curves_of_run(context, context.drawn[arm], band) for arm in arms}
        # Step 1: spread over every repeat of the arm (all campaign runs, whichever was drawn).
        campaign_bands[row_title] = {
            arm: campaign_envelope([curves_of_run(context, repeat, band) for repeat in context.repeats[arm]])
            for arm in arms if context.repeats[arm]}
        # Step 2: the same numbers as JSON-ready lists, for the manifest.
        reported["drawn_curves"][row_title] = {
            arm: {key: json_number_list(values) for key, values in curves.items() if key != "thresholds"}
            for arm, curves in drawn_curves[row_title].items()}
        reported["campaign_bands"][row_title] = {
            arm: {metric_key: {"lower": json_number_list(lower), "upper": json_number_list(upper)}
                  for metric_key, (_, lower, upper) in envelope.items()}
            for arm, envelope in campaign_bands[row_title].items()}
    # Step 3: every repeat's archive fed a band, so all of them are inputs of this figure.
    for repeats in context.repeats.values():
        for repeat in repeats:
            context.hashes.record(repeat.cells_path)
    repeat_text = repeat_count_text(reported["repeat_counts"], context.settings.arm_labels)
    figure = threshold_curves_figure(
        drawn_curves, context.settings, p2_thresholds=context.p2_thresholds, campaign_bands=campaign_bands,
        title=context.short_title(f"{HEADLINE_CLASS_NAME} recall and precision vs occupancy threshold "
                                  f"(bands = min/max over the repeats, {repeat_text})"),
        run_note=context.run_note())
    save_matplotlib_figure(figure, output_path, context.settings.dpi)
    return reported


def draw_occupancy_histograms(context: FigureContext, output_path: Path) -> Dict[str, Any]:
    """C4b: occupancy histograms on truth vs off-truth cells of the headline band, per arm.

    Args:
        context: Shared context.
        output_path: PNG path.

    Returns:
        The bin edges and, per arm and series, the sample count and the count in each bin (the same
        edges and finite values the figure bins).
    """
    edges = occupancy_histogram_edges(HISTOGRAM_BIN_COUNT)
    samples = {}
    reported: Dict[str, Any] = {"bin_edges": json_number_list(edges), "counts": {}}
    for arm in arms_in_order(context):
        # Step 1: occupancy of the arm's truth and off-truth cells on the headline band.
        context.hashes.record(context.drawn[arm].cells_path)
        samples[arm] = occupancy_samples(context.drawn[arm].cells, HEADLINE_CLASS_NAME, HEADLINE_BAND)
        reported["counts"][arm] = {}
        # Step 2: bin the finite values on the figure's own edges, for the manifest.
        for series_name, values in samples[arm].items():
            finite_values = values[np.isfinite(values)]
            bin_counts, _ = np.histogram(finite_values, bins=edges)
            reported["counts"][arm][series_name] = {"n": int(finite_values.size),
                                                    "counts": [int(count) for count in bin_counts]}
    figure = occupancy_histogram_figure(
        samples, context.settings, bins=HISTOGRAM_BIN_COUNT,
        title=context.short_title(f"occupancy of {HEADLINE_CLASS_NAME} cells in H_near"),
        run_note=context.run_note())
    save_matplotlib_figure(figure, output_path, context.settings.dpi)
    return reported


def draw_hidden_near_heat(context: FigureContext, output_path: Path) -> Dict[str, Any]:
    """C5: column-max occupancy over H_near & neighbourhood per arm on one extent, plus init+loss minus stock.

    The mask is identical for all arms (model-independent), so the extent is common. The difference uses
    ``difference_image`` (NaN only where both are NaN, a one-sided NaN counts as 0).

    Args:
        context: Shared context.
        output_path: PNG path.

    Returns:
        The extent, the finite-column count and value range of each arm's map, and the same for the difference.
    """
    reference = context.reference_cells
    # Step 1: the shared mask (H_near & headline neighbourhood), then one column-max map per arm on it.
    mask = reference["hidden_near"] & reference[f"neighbourhood_{HEADLINE_CLASS_NAME}"]
    images: Dict[str, np.ndarray] = {}
    extent = None
    for arm in arms_in_order(context):
        context.hashes.record(context.drawn[arm].cells_path)
        cells = context.drawn[arm].cells
        images[arm], extent = column_max_grid(cells["cell_indices"], cells["occupancy"], mask,
                                              cells["grid_origin"], context.voxel_size)
    # Step 2: first-arm minus second-arm map.
    difference_label = f"{context.arm_label(DIFFERENCE_FIRST_ARM)} - {context.arm_label(DIFFERENCE_SECOND_ARM)}"
    difference_map = difference_image(images[DIFFERENCE_FIRST_ARM], images[DIFFERENCE_SECOND_ARM])
    figure = arm_heat_figure(
        images, extent, context.settings, difference=(difference_map, difference_label),
        title=context.short_title(f"column-max occupancy over H_near & {HEADLINE_CLASS_NAME} neighbourhood"),
        run_note=context.run_note())
    save_matplotlib_figure(figure, output_path, context.settings.dpi)
    return {"extent": [float(value) for value in extent],
            "arm_maps": {arm: finite_summary(image) for arm, image in images.items()},
            "difference": {"label": difference_label, **finite_summary(difference_map)}}


def generate_arm_figures(context: FigureContext, output_dir: Path) -> FigureRecorder:
    """Draw C1-C5, write the cross-arm manifest and return the recorder.

    Args:
        context: Shared context.
        output_dir: Folder for the PNGs and the manifest.

    Returns:
        The recorder holding every figure's status.
    """
    recorder = FigureRecorder()
    reported: Dict[str, Any] = {}
    context.hashes.begin_scope()

    def attempt_with_numbers(name: str, key: str, draw: Callable[[Path], Any]) -> None:
        """Run one figure and keep its returned numbers under ``key``.

        Args:
            name: Figure name (also the PNG stem in ``output_dir``).
            key: Key of ``reported`` that receives ``draw``'s return value.
            draw: Callable taking the PNG path, drawing and saving, returning the numbers to record.
        """
        path = output_dir / f"{name}.png"
        recorder.attempt(name, path, lambda: reported.__setitem__(key, draw(path)))

    # Step 1: one attempt per figure (the gallery records one entry per page itself).
    attempt_with_numbers("arms_hidden_near_3d", "arms_hidden_near_outcome_counts",
                         lambda path: draw_arm_outcome_panels(context, path, HEADLINE_BAND))
    attempt_with_numbers("arms_scene_3d", "arms_scene_outcome_counts",
                         lambda path: draw_arm_outcome_panels(context, path, BAND_ALL))
    reported["trunk_gallery_objects"] = draw_trunk_gallery(context, output_dir, recorder)
    attempt_with_numbers("occlusion_sections", "occlusion_sections", lambda path: draw_occlusion_sections(context, path))
    attempt_with_numbers("threshold_curves", "threshold_curves", lambda path: draw_threshold_curves(context, path))
    attempt_with_numbers("occupancy_histograms", "occupancy_histograms",
                         lambda path: draw_occupancy_histograms(context, path))
    attempt_with_numbers("hidden_near_heat_bev", "hidden_near_heat", lambda path: draw_hidden_near_heat(context, path))
    # Step 2: the manifest is written whatever happened.
    write_manifest(output_dir, context_manifest(context, "arms", recorder.figures, reported))
    return recorder


# ----------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------
def build_argument_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="session name, e.g. june_tall_grass")
    parser.add_argument("--run", action="append", default=[], metavar="ARM=DIR",
                        help="draw this run folder for the arm instead of its median repeat (repeatable)")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--arms-only", action="store_true", help="only the cross-arm figures")
    scope.add_argument("--per-run-only", action="store_true", help="only the per-run figures")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="cross-arm output folder (default reports/gaussian_occupancy/figures/<session>)")
    parser.add_argument("--eval-config", type=Path, default=EVAL_CONFIG_PATH, help="default: configs/eval.yaml")
    return parser


def load_submit_run_module() -> Any:
    """Import scripts/slurm/submit_run.py by path (for ``read_ledger``).

    Returns:
        The module.
    """
    module_spec = importlib.util.spec_from_file_location("submit_run_for_occupancy_figures", SUBMIT_RUN_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def build_context(arguments: argparse.Namespace) -> FigureContext:
    """Resolve runs, load archives, check consistency and assemble the shared context.

    Args:
        arguments: Parsed command line.

    Returns:
        The context.

    Raises:
        ValueError: On unknown session/arm, an arm without runs or override, non-comparable runs (any
            model-independent array, the archived threshold or the evidence labels differ), a run whose archived
            threshold differs from ``p2_thresholds[0]``, or a P2 threshold off the curve grid.
    """
    eval_section = yaml.safe_load(Path(arguments.eval_config).read_text())["eval"]
    settings = load_figure_settings(eval_section)
    session = arguments.session
    if session not in settings.campaign_profiles:
        raise ValueError(f"unknown session {session!r}; known: {sorted(settings.campaign_profiles)}")
    overrides = parse_run_overrides(arguments.run, settings.arm_order)
    ledger_rows = load_submit_run_module().read_ledger(LEDGER_PATH)
    slurm_profiles = yaml.safe_load(SLURM_CONFIG_PATH.read_text())["profiles"]
    # Step 1: campaign runs per arm, then the loaded runs, the consistency checks and the drawn run per arm.
    campaign_profiles = settings.campaign_profiles[session]
    runs_by_arm = resolve_arm_runs(session, campaign_profiles, ledger_rows, RUNS_DIR)
    selection = select_arm_runs(settings.arm_order, campaign_profiles, runs_by_arm, overrides,
                                ledger_profile_by_run(ledger_rows), settings.representative_metric)
    # Step 2: thresholds from the archive (checked on every loaded run); binding = first; the curve grid must
    # contain every one.
    p2_thresholds = np.asarray(selection.drawn[settings.arm_order[0]].cells["p2_thresholds"], dtype=float)
    curve_grid = threshold_grid(settings.threshold_curve_points, p2_thresholds)
    # Step 3: session data (GT display only), driven path, class ids, seeds.
    cuboids_path = GT_DIR / session / CUBOIDS_FILE_NAME
    return FigureContext(
        session=session, settings=settings, eval_section=eval_section, cuboids=load_cuboids_json(cuboids_path),
        sweep_origins=session_sweep_origins(INTERMEDIATE_DIR / session),
        class_ids=class_ids_for_name(HEADLINE_CLASS_NAME, eval_section), p2_thresholds=p2_thresholds,
        threshold=selection.binding_threshold, curve_grid=curve_grid, drawn=selection.drawn,
        repeats=selection.repeats,
        seeds_paths={arm: seeds_path_for_profile(selection.drawn[arm].profile, slurm_profiles, session,
                                                 INTERMEDIATE_DIR) for arm in settings.arm_order},
        git=git_state(PROJECT_ROOT), arms_info=selection.arms_info, hashes=HashRecorder(),
        cuboids_path=cuboids_path)


def main(argument_list: Optional[List[str]] = None) -> int:
    """Draw the figure suite of one session.

    Args:
        argument_list: CLI arguments (default: sys.argv).

    Returns:
        0 when every figure was written, 1 when setup failed or any figure failed.
    """
    arguments = build_argument_parser().parse_args(argument_list)
    start_time = time.time()
    # Step 1: setup problems (bad arguments, missing runs, non-comparable runs) refuse with exit 1.
    try:
        context = build_context(arguments)
    except (ValueError, OSError, KeyError) as error:
        print(f"refused: {error!r}" if isinstance(error, KeyError) else f"refused: {error}", file=sys.stderr)
        return EXIT_FAILED
    for arm in arms_in_order(context):
        print(f"{arm}: drawn {context.drawn[arm].run_id} ({context.arms_info[arm]['drawn_by']})")
    any_failed = False
    # Step 2: per-run figures, then cross-arm figures.
    if not arguments.arms_only:
        recorders = generate_per_run_figures(context)
        any_failed |= any(recorder.any_failed() for recorder in recorders.values())
    if not arguments.per_run_only:
        output_dir = arguments.output_dir or DEFAULT_OUTPUT_ROOT / context.session
        any_failed |= generate_arm_figures(context, output_dir).any_failed()
    print(f"done in {time.time() - start_time:.1f} s; {'SOME FIGURES FAILED' if any_failed else 'all figures written'}")
    return EXIT_FAILED if any_failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
