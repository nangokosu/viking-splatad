"""Gaussian-occupancy figure suite of one session: cross-arm comparison figures.

Reads each arm's campaign runs (the arm's repeats of the ``eval.figures.campaign_profiles`` profile), draws ONE
run per arm (the median repeat on ``eval.figures.representative_metric`` of ``<run>/eval/gaussian_occupancy.json``,
or the folder given by ``--run arm=<dir>``, which may be a run of another arm or stand in for an arm without
campaign runs) and writes PNGs plus a ``figures_manifest.json`` into ``reports/gaussian_occupancy/figures/<session>/``
(or ``--output-dir``).
Everything is read-only on the runs, GT and session data.

Figures:
    ``arms_scene_3d.png``         whole-scene per-arm outcome panels (hit / false alarm / missed / context) plus a
                                  hit-difference panel; ``draw_arm_outcome_panels`` also takes the H_near band
    ``threshold_curves_scene.png``  whole-scene recall and precision against the occupancy threshold (the configured curve range),
                                  with the repeat min/max band

The class is the headline group (``obstacles``) at the binding threshold ``p2_thresholds[0]`` read from the archive.
A figure that fails is reported on stderr, recorded in the manifest (status "failed") and the others are still
drawn; the exit code is then 1. A figure with nothing to draw is recorded as "skipped" with its reason (not a failure).
The scene cell and outcome counts (``reported_numbers["scene_counts"]``) are computed before and outside every
figure, so the reports that read them never depend on a figure succeeding; a failure there raises. The GT boxes are
loaded inside the ``arms_scene_3d`` attempt, so a missing cuboid file fails only that figure.
Run loading, consistency checks, hashing, failure bookkeeping and manifests live in
``viking_splatad.viz.figure_suite_io``.

Usage (isolated viz environment, which has PyVista and matplotlib):
    uv run --project viz python scripts/26_occupancy_figures.py --session june_tall_grass
        [--run stock=<dir> --run radar_init=<dir> --run radar_init_loss=<dir>] [--output-dir <dir>]
"""

import argparse
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from viking_splatad.eval.p2_cells import HIDDEN_BAND_NEAR, hidden_cell_outcomes, p2_cell_outcomes
from viking_splatad.eval.detection import load_cuboids_json
from viking_splatad.viz.figure_grids import arm_difference_sets
from viking_splatad.viz.figure_metrics import BAND_ALL, class_ids_for_name, occupancy_samples, threshold_curves
from viking_splatad.viz.figure_runs import FigureSettings, load_figure_settings, resolve_arm_runs
from viking_splatad.viz.figure_suite_io import (
    PROJECT_ROOT,
    REPORT_RELATIVE_PATH,
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
from viking_splatad.viz.figures_mpl import THRESHOLD_METRICS, threshold_curves_figure
from viking_splatad.viz.figures_pyvista import (
    LEGEND_KIND_PATCH,
    SCENE_GRID_COLUMNS,
    SCENE_GRID_ROWS,
    compose_scene_figure,
    render_arm_panels,
    scene_camera_aspect,
    scene_camera_points,
    scene_panel_titles,
)
from viking_splatad.viz.scored_boxes import class_box_counts, select_scored_cuboids

EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"
LEDGER_PATH = PROJECT_ROOT / "runs" / "submitted.tsv"
RUNS_DIR = PROJECT_ROOT / "runs"
SUBMIT_RUN_PATH = PROJECT_ROOT / "scripts" / "slurm" / "submit_run.py"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "reports" / "gaussian_occupancy" / "figures"
GT_DIR = PROJECT_ROOT / "data" / "gt"
CUBOIDS_FILE_NAME = "cuboids.json"

# The headline class group and the hidden band of the headline figures.
HEADLINE_CLASS_NAME = "obstacles"
HEADLINE_BAND = HIDDEN_BAND_NEAR
# Arm roles of the difference panels: first = the arm whose hits are blue, second = the baseline.
DIFFERENCE_FIRST_ARM = "radar_init_loss"
DIFFERENCE_SECOND_ARM = "stock"
# Display text of the bands in titles.
BAND_TEXT = {HEADLINE_BAND: "H_near", BAND_ALL: "whole scene"}
# Bounds of the unit interval: the default ends of ``threshold_grid``.
UNIT_INTERVAL_MIN = 0.0
UNIT_INTERVAL_MAX = 1.0
# Separator of the short figure titles ("<session> · <subject>").
TITLE_SEPARATOR = " \u00b7 "
# 3D scene panels are this many tiles (tile_px) on a side; a layout choice (3 x 360 px = 1080 px).
SCENE_PANEL_TILES = 3
# Settings colour of a GT class is ``class_<name>``; title of the ground-truth panel.
GT_CLASS_COLOUR_PREFIX = "class_"
GT_PANEL_TITLE = "ground truth (scored obstacle boxes)"
# Exit codes.
EXIT_OK, EXIT_FAILED = 0, 1


# ----------------------------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ----------------------------------------------------------------------------------------------
def threshold_grid(point_count: int, p2_thresholds: Sequence[float], minimum: float = UNIT_INTERVAL_MIN,
                   maximum: float = UNIT_INTERVAL_MAX) -> np.ndarray:
    """Evenly spaced threshold grid on [minimum, maximum] (ends included) that must contain every P2 threshold.

    The step is (maximum - minimum) / (point_count - 1); a P2 threshold t lies on the grid when
    (t - minimum) / step is an integer.
    A grid point that matches a P2 threshold (``np.isclose``) is replaced by the archived value itself, as
    ``hidden_cell_outcomes`` does, so a curve evaluated at that point cuts occupancy at exactly the value
    scripts/19 scored with (a linspace point can differ from it by one rounding step and move cells that sit
    on the cut).

    Args:
        point_count: Number of grid points (>= 2).
        p2_thresholds: Binding thresholds that must be grid points.
        minimum: Lowest grid point.
        maximum: Highest grid point.

    Returns:
        (point_count,) float array.

    Raises:
        ValueError: When minimum >= maximum, or a P2 threshold is not on the grid.
    """
    # Step 1: a grid needs a positive extent.
    if minimum >= maximum:
        raise ValueError(f"threshold grid minimum {minimum} must be below maximum {maximum}")
    grid = np.linspace(minimum, maximum, point_count)
    for threshold in p2_thresholds:
        # Step 2: every P2 threshold must match a grid point.
        matching_points = np.isclose(grid, float(threshold))
        if not matching_points.any():
            raise ValueError(f"p2 threshold {threshold} is not on the {point_count}-point curve grid over "
                             f"[{minimum}, {maximum}]; change eval.figures.threshold_curve_points/min/max")
        # Step 3: snap the matching point(s) to the archived value.
        grid[matching_points] = float(threshold)
    return grid


def campaign_envelope(curves_per_repeat: Sequence[Mapping[str, np.ndarray]]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-threshold min and max over repeats, one envelope per plotted curve.

    Every plotted curve key (``pooled_recall``, ``mean_recall``, ``pooled_precision``, ``mean_precision``) gets
    its own band: at every threshold the lowest and highest value of THAT curve over the repeats, so the
    pooled and the mean-over-objects bands do not merge.
    NaN is ignored per threshold (``fmin``/``fmax``); all-NaN thresholds stay NaN.

    Args:
        curves_per_repeat: One ``threshold_curves`` dict per repeat.

    Returns:
        curve key -> (thresholds, lower, upper).
    """
    envelope: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    thresholds = np.asarray(curves_per_repeat[0]["thresholds"], dtype=float)
    for _metric_key, _label, pooled_key, mean_key in THRESHOLD_METRICS:
        for curve_key in (pooled_key, mean_key):
            # Step 1: stack this curve of every repeat as rows: (repeats, thresholds).
            stacked = np.vstack([np.asarray(curves[curve_key], dtype=float) for curves in curves_per_repeat])
            # Step 2: NaN-ignoring min / max over the rows.
            envelope[curve_key] = (thresholds, np.fmin.reduce(stacked, axis=0), np.fmax.reduce(stacked, axis=0))
    return envelope


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
        class_ids: Class ids of the headline group.
        p2_thresholds: Every evaluated threshold, from the archive.
        threshold: Binding threshold ``p2_thresholds[0]``.
        curve_grid: Threshold grid of the curves (contains every P2 threshold).
        drawn: arm -> the run drawn for it.
        repeats: arm -> its campaign runs ([] when an override stands in).
        git: ``git_state`` result.
        arms_info: arm -> manifest entry on how its run was chosen.
        hashes: sha256 recorder of every input read.
        scored_boxes: GT boxes to draw when ``cuboids_path`` is None (tests); otherwise loaded by the figure.
        cuboids_path: GT cuboid file the figure loads its scored boxes from (hashed when used).
    """

    session: str
    settings: FigureSettings
    eval_section: Mapping[str, Any]
    class_ids: Tuple[int, ...]
    p2_thresholds: np.ndarray
    threshold: float
    curve_grid: np.ndarray
    drawn: Dict[str, ArmRun]
    repeats: Dict[str, List[ArmRun]]
    git: Dict[str, Any]
    arms_info: Dict[str, Any]
    hashes: HashRecorder
    scored_boxes: List[Any] = field(default_factory=list)
    cuboids_path: Optional[Path] = None

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

    def scene_window(self, panel_aspect: float) -> Tuple[int, int]:
        """Pixel window of the (``SCENE_GRID_ROWS`` x ``SCENE_GRID_COLUMNS``) 3D scene grid.

        A panel is ``tile_px * SCENE_PANEL_TILES`` pixels wide and ``width / panel_aspect`` pixels high (rounded),
        so the panels have the shape of the scene's projection; the window is the grid of those panels.

        Args:
            panel_aspect: Panel width / height (> 0), e.g. ``scene_camera_aspect``.

        Returns:
            (width, height) in pixels.
        """
        panel_width = self.settings.tile_px * SCENE_PANEL_TILES
        panel_height = max(1, round(panel_width / panel_aspect))
        return panel_width * SCENE_GRID_COLUMNS, panel_height * SCENE_GRID_ROWS


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


# ----------------------------------------------------------------------------------------------
# Cross-arm figures
# ----------------------------------------------------------------------------------------------
def arms_in_order(context: FigureContext) -> List[str]:
    """Arm keys in settings order.

    Args:
        context: Shared context.

    Returns:
        List of arm keys.
    """
    return list(context.settings.arm_order)


def gt_class_colours(context: FigureContext, scored_boxes: Sequence[Any]) -> Dict[int, Tuple[int, int, int]]:
    """Display colour per class id of the scored boxes (settings colour ``class_<name>``).

    Args:
        context: Shared context (colour settings).
        scored_boxes: The GT boxes to colour.

    Returns:
        class id -> 0-255 RGB.

    Raises:
        ValueError: When ``eval.figures.colours`` lacks a ``class_<name>`` entry.
    """
    colour_by_class_id: Dict[int, Tuple[int, int, int]] = {}
    for cuboid in scored_boxes:
        colour_name = f"{GT_CLASS_COLOUR_PREFIX}{cuboid.class_name}"
        if colour_name not in context.settings.colours:
            raise ValueError(f"eval.figures.colours lacks {colour_name!r} (one colour per drawn GT class)")
        colour_by_class_id[cuboid.class_id] = context.settings.colours[colour_name]
    return colour_by_class_id


def headline_class_names(context: FigureContext) -> List[str]:
    """Names of the headline group's classes in ``context.class_ids`` order (the group's member order).

    Args:
        context: Shared context.

    Returns:
        Class names, one per class id of the group.

    Raises:
        ValueError: When a class id has no name in ``eval.scored_class_ids``.
    """
    class_name_by_id = {int(class_id): class_name for class_name, class_id in context.eval_section["scored_class_ids"].items()}
    missing_ids = [class_id for class_id in context.class_ids if int(class_id) not in class_name_by_id]
    if missing_ids:
        raise ValueError(f"class ids {missing_ids} are not in eval.scored_class_ids")
    return [class_name_by_id[int(class_id)] for class_id in context.class_ids]


def ordered_box_counts(context: FigureContext, scored_boxes: Sequence[Any]) -> Dict[str, int]:
    """Scored-box count per class name, ordered like ``context.class_ids`` (not by dict or encounter order).

    Args:
        context: Shared context.
        scored_boxes: The GT boxes drawn.

    Returns:
        class name -> count, for the classes that have at least one box, in the group's member order.
    """
    counts_by_name = class_box_counts(scored_boxes)
    return {class_name: counts_by_name[class_name] for class_name in headline_class_names(context)
            if class_name in counts_by_name}


def check_difference_arms(context: FigureContext) -> None:
    """Refuse a settings arm order that lacks either arm of the difference panel.

    Args:
        context: Shared context.

    Raises:
        ValueError: When ``DIFFERENCE_FIRST_ARM`` or ``DIFFERENCE_SECOND_ARM`` is not in ``arm_order``.
    """
    missing_arms = [arm for arm in (DIFFERENCE_FIRST_ARM, DIFFERENCE_SECOND_ARM) if arm not in context.settings.arm_order]
    if missing_arms:
        raise ValueError(f"the difference panel compares {DIFFERENCE_FIRST_ARM!r} with {DIFFERENCE_SECOND_ARM!r}, but "
                         f"eval.figures.arm_order {list(context.settings.arm_order)} lacks {missing_arms}")


def scene_counts(context: FigureContext) -> Dict[str, Any]:
    """Whole-scene cell and outcome counts of the headline class, independent of any figure.

    Args:
        context: Shared context.

    Returns:
        {"band": "all", "cell_counts": {"on_obstacle", "next_to_obstacle"},
         "outcome_counts_by_arm": {arm: {"hit", "false_alarm", "missed", "context"}}} with int counts, from the drawn
        runs' ``scene_outcomes`` (the archived threshold equals ``context.threshold``, checked at setup).
    """
    # Step 1: per-arm outcome counts of the drawn run on the whole scene.
    outcome_counts_by_arm = {}
    for arm in arms_in_order(context):
        context.hashes.record(context.drawn[arm].cells_path)
        outcomes = scene_outcomes(context.drawn[arm].cells, HEADLINE_CLASS_NAME)
        outcome_counts_by_arm[arm] = {name: int(outcomes[name].sum())
                                      for name in ("hit", "false_alarm", "missed", "context")}
    # Step 2: model-independent cell counts (checked equal across arms).
    return {"band": BAND_ALL, "cell_counts": whole_scene_cell_counts(context),
            "outcome_counts_by_arm": outcome_counts_by_arm}


def draw_arm_outcome_panels(context: FigureContext, output_path: Path, band: str) -> Dict[str, Any]:
    """Ground truth, hit difference (first = init+loss, second = stock) and one outcome panel per arm.

    Layout: a 2 x 3 grid on one linked camera fitted to the drawn cells and boxes; the window has the shape of
    the scene's projection (``FigureContext.scene_window``).

    Args:
        context: Shared context.
        output_path: PNG path.
        band: ``HEADLINE_BAND`` (H_near cells) or ``BAND_ALL`` (whole scene).

    Returns:
        Outcome cell counts per arm and the difference sets (for the manifest), plus ``gt_box_counts``
        (class name -> number of scored boxes drawn, in the headline group's class order).
    """
    # Step 0: the compared arms exist, and the GT boxes load (a failure here fails only this figure).
    check_difference_arms(context)
    scored_boxes = (load_scored_boxes(context.session, context.cuboids_path, context.drawn[context.settings.arm_order[0]],
                                      context.eval_section, context.class_ids)
                    if context.cuboids_path is not None else list(context.scored_boxes))
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
    band_text = BAND_TEXT[band]
    titles = [context.arm_label(arm) for arm in arms_in_order(context)]
    titles.append(f"hits: {context.arm_label(DIFFERENCE_FIRST_ARM)} vs {context.arm_label(DIFFERENCE_SECOND_ARM)}")
    # Step 3: window shaped like the scene's projection (cells and GT box corners the panels draw).
    camera_points = scene_camera_points(context.reference_cells["centres"], outcomes_by_arm, difference_sets,
                                        scored_boxes)
    if camera_points.shape[0] == 0:
        raise FigureSkipped("no cell or GT box to draw")
    window_size = context.scene_window(scene_camera_aspect(camera_points))
    first_colour_name, second_colour_name = difference_colour_names(context.settings)
    if context.cuboids_path is not None:
        # The GT boxes and the run report whose counts they were checked against are inputs of this figure.
        context.hashes.record(context.cuboids_path)
        context.hashes.record(context.drawn[context.settings.arm_order[0]].run_dir / REPORT_RELATIVE_PATH)
    layout_report: Dict[str, Any] = {}
    image = render_arm_panels(import_pyvista(), context.reference_cells["centres"], context.voxel_size,
                              outcomes_by_arm, difference_sets, context.settings, window_size=window_size,
                              first_colour_name=first_colour_name,
                              second_colour_name=second_colour_name, gt_cuboids=scored_boxes,
                              gt_colour_by_class_id=gt_class_colours(context, scored_boxes), layout_report=layout_report)
    # Step 4: legend of the outcome, difference and GT-class colours, in a figure with a short title.
    colours = context.settings.colours
    gt_box_counts = ordered_box_counts(context, scored_boxes)
    legend_entries = [("hit", colours["hit"], LEGEND_KIND_PATCH),
                      ("false alarm", colours["false_alarm"], LEGEND_KIND_PATCH),
                      ("missed", colours["missed"], LEGEND_KIND_PATCH),
                      ("context", colours["context"], LEGEND_KIND_PATCH),
                      (f"only {context.arm_label(DIFFERENCE_FIRST_ARM)}", colours[first_colour_name],
                       LEGEND_KIND_PATCH),
                      (f"only {context.arm_label(DIFFERENCE_SECOND_ARM)}", colours[second_colour_name],
                       LEGEND_KIND_PATCH),
                      ("both", colours["both"], LEGEND_KIND_PATCH)]
    legend_entries += [(f"{class_name} ({box_count})", colours[f"{GT_CLASS_COLOUR_PREFIX}{class_name}"],
                        LEGEND_KIND_PATCH) for class_name, box_count in gt_box_counts.items()]
    figure = compose_scene_figure(
        image, context.short_title(f"{band_text} {HEADLINE_CLASS_NAME} @ {context.threshold}"), legend_entries,
        context.settings, panel_count=SCENE_GRID_COLUMNS, run_note=context.run_note(),
        panel_titles=scene_panel_titles(arms_in_order(context), titles, GT_PANEL_TITLE), grid_rows=SCENE_GRID_ROWS)
    save_matplotlib_figure(figure, output_path, context.settings.dpi)
    counts: Dict[str, Any] = {arm: {name: int(mask.sum()) for name, mask in outcomes_by_arm[arm].items()}
                              for arm in arms_in_order(context)}
    counts["difference"] = {name: int(mask.sum()) for name, mask in difference_sets.items()}
    counts["gt_box_counts"] = gt_box_counts
    counts["panel_crop"] = layout_report
    return counts


def load_scored_boxes(session: str, cuboids_path: Path, reference_run: ArmRun,
                      eval_section: Mapping[str, Any], class_ids: Sequence[int]) -> List[Any]:
    """The headline classes' GT boxes the evaluation scores, checked against the run report.

    Rule: the boxes whose (id, class) the cells archive lists as scored objects (``object_box_ids``); the
    per-class count must equal ``classes[<class>]["scored_cuboids"]`` of the run's gaussian_occupancy.json.

    Args:
        session: Session name (for messages).
        cuboids_path: GT cuboid file of the session.
        reference_run: A drawn run (the object lists are model-independent).
        eval_section: The ``eval`` section (``scored_class_ids``: class name -> id).
        class_ids: Class ids of the headline group.

    Returns:
        The scored boxes of those classes.

    Raises:
        ValueError: On a count mismatch (``select_scored_cuboids``) or a class id without a name.
    """
    # Step 1: names of the headline classes.
    class_name_by_id = {int(class_id): class_name for class_name, class_id in eval_section["scored_class_ids"].items()
                        if int(class_id) in set(int(wanted) for wanted in class_ids)}
    if set(class_name_by_id) != set(int(class_id) for class_id in class_ids):
        raise ValueError(f"{session}: headline class ids {list(class_ids)} are not all in eval.scored_class_ids")
    # Step 2: the run report's scored-box counts.
    report = json.loads((reference_run.run_dir / REPORT_RELATIVE_PATH).read_text())
    reported_counts = {class_name: entry["scored_cuboids"] for class_name, entry in report["classes"].items()}
    # Step 3: select and check.
    return select_scored_cuboids(load_cuboids_json(cuboids_path), reference_run.cells["object_box_ids"],
                                 reference_run.cells["object_class_ids"], class_name_by_id, reported_counts)


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


def whole_scene_cell_counts(context: FigureContext) -> Dict[str, int]:
    """Whole-scene cell counts of the headline class: the obstacle cells G and the cells next to them N \\ G.

    The masks come from the ground truth, not the model, so the counts are the same for every arm: they are the
    sizes of the ``occupancy_samples`` series of band "all", the cells the threshold curves are computed on.

    Args:
        context: Shared context.

    Returns:
        {"on_obstacle": |G|, "next_to_obstacle": |N \\ G|}.

    Raises:
        ValueError: When the drawn runs of two arms disagree on a count.
    """
    # Step 1: count each arm's drawn-run samples (every G and N \ G cell; no value is filtered out).
    counts_by_arm = {}
    for arm in arms_in_order(context):
        samples = occupancy_samples(context.drawn[arm].cells, HEADLINE_CLASS_NAME, BAND_ALL)
        counts_by_arm[arm] = {"on_obstacle": int(samples["truth"].size),
                              "next_to_obstacle": int(samples["off_truth"].size)}
    # Step 2: model-independent, so every arm must agree with the first.
    reference_arm = next(iter(counts_by_arm))
    for arm, counts in counts_by_arm.items():
        if counts != counts_by_arm[reference_arm]:
            raise ValueError(f"cell counts differ between arms: {reference_arm} {counts_by_arm[reference_arm]} "
                             f"vs {arm} {counts}")
    return counts_by_arm[reference_arm]


def draw_threshold_curves(context: FigureContext, output_path: Path, band: str) -> Dict[str, Any]:
    """Threshold curves of one band: curves of the drawn run per arm, each with its min/max band over all repeats.

    An arm without campaign runs (an override stands in) has its curves but no band.

    Args:
        context: Shared context.
        output_path: PNG path.
        band: ``BAND_ALL`` (whole scene) or ``HEADLINE_BAND`` (H_near).

    Returns:
        The numbers drawn: the grid, the repeat count per arm, each drawn curve and each band envelope
        (NaN as None).
    """
    arms = arms_in_order(context)
    band_text = BAND_TEXT[band]
    # Step 1: the drawn run's curves, and the spread over every repeat of the arm (all campaign runs, whichever
    # was drawn).
    drawn_curves = {arm: curves_of_run(context, context.drawn[arm], band) for arm in arms}
    campaign_bands = {arm: campaign_envelope([curves_of_run(context, repeat, band) for repeat in context.repeats[arm]])
                      for arm in arms if context.repeats[arm]}
    # Step 2: the same numbers as JSON-ready lists, for the manifest.
    reported: Dict[str, Any] = {
        "band": band, "curve_grid": json_number_list(context.curve_grid),
        "repeat_counts": {arm: len(context.repeats[arm]) for arm in arms},
        "drawn_curves": {arm: {key: json_number_list(values) for key, values in curves.items() if key != "thresholds"}
                         for arm, curves in drawn_curves.items()},
        "campaign_bands": {arm: {curve_key: {"lower": json_number_list(lower), "upper": json_number_list(upper)}
                                 for curve_key, (_, lower, upper) in envelope.items()}
                           for arm, envelope in campaign_bands.items()}}
    if band == BAND_ALL:
        # The cell counts are the whole scene's, so they belong to this band only (``scene_counts`` also has them).
        reported["cell_counts"] = whole_scene_cell_counts(context)
    # Step 3: every repeat's archive fed a band, so all of them are inputs of this figure.
    for repeats in context.repeats.values():
        for repeat in repeats:
            context.hashes.record(repeat.cells_path)
    figure = threshold_curves_figure(
        drawn_curves, context.settings, p2_thresholds=context.p2_thresholds, campaign_bands=campaign_bands,
        title=context.short_title(f"{HEADLINE_CLASS_NAME} recall and precision vs occupancy threshold ({band_text})"),
        run_note=context.run_note())
    save_matplotlib_figure(figure, output_path, context.settings.dpi)
    return reported


def generate_arm_figures(context: FigureContext, output_dir: Path) -> FigureRecorder:
    """Draw the kept figures, write the cross-arm manifest and return the recorder.

    Args:
        context: Shared context.
        output_dir: Folder for the PNGs and the manifest.

    Returns:
        The recorder holding every figure's status.
    """
    recorder = FigureRecorder()
    reported: Dict[str, Any] = {}
    context.hashes.begin_scope()
    # Step 0: the counts the reports read, before and outside every figure (a failure here is a real error).
    reported["scene_counts"] = scene_counts(context)

    def attempt_with_numbers(name: str, key: str, draw: Callable[[Path], Any]) -> None:
        """Run one figure and keep its returned numbers under ``key``.

        Args:
            name: Figure name (also the PNG stem in ``output_dir``).
            key: Key of ``reported`` that receives ``draw``'s return value.
            draw: Callable taking the PNG path, drawing and saving, returning the numbers to record.
        """
        path = output_dir / f"{name}.png"
        recorder.attempt(name, path, lambda: reported.__setitem__(key, draw(path)))

    # Step 1: one attempt per figure; a failure is recorded and the next figure still runs.
    attempt_with_numbers("arms_scene_3d", "arms_scene_outcome_counts",
                         lambda path: draw_arm_outcome_panels(context, path, BAND_ALL))
    attempt_with_numbers("threshold_curves_scene", "threshold_curves_scene",
                         lambda path: draw_threshold_curves(context, path, BAND_ALL))
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
    # Step 1: campaign runs per arm, then the loaded runs, the consistency checks and the drawn run per arm.
    campaign_profiles = settings.campaign_profiles[session]
    runs_by_arm = resolve_arm_runs(session, campaign_profiles, ledger_rows, RUNS_DIR)
    selection = select_arm_runs(settings.arm_order, campaign_profiles, runs_by_arm, overrides,
                                ledger_profile_by_run(ledger_rows), settings.representative_metric)
    # Step 2: thresholds from the archive (checked on every loaded run); binding = first; the curve grid must
    # contain every one.
    p2_thresholds = np.asarray(selection.drawn[settings.arm_order[0]].cells["p2_thresholds"], dtype=float)
    curve_grid = threshold_grid(settings.threshold_curve_points, p2_thresholds, settings.threshold_curve_min,
                                settings.threshold_curve_max)
    # Step 3: class ids of the headline group, the GT cuboid path (boxes load inside the figure) and the context.
    class_ids = class_ids_for_name(HEADLINE_CLASS_NAME, eval_section)
    cuboids_path = GT_DIR / session / CUBOIDS_FILE_NAME
    return FigureContext(
        session=session, settings=settings, eval_section=eval_section,
        class_ids=class_ids, p2_thresholds=p2_thresholds,
        threshold=selection.binding_threshold, curve_grid=curve_grid, drawn=selection.drawn,
        repeats=selection.repeats, git=git_state(PROJECT_ROOT), arms_info=selection.arms_info,
        hashes=HashRecorder(),
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
    # Step 2: the cross-arm figures.
    output_dir = arguments.output_dir or DEFAULT_OUTPUT_ROOT / context.session
    any_failed |= generate_arm_figures(context, output_dir).any_failed()
    print(f"done in {time.time() - start_time:.1f} s; {'SOME FIGURES FAILED' if any_failed else 'all figures written'}")
    return EXIT_FAILED if any_failed else EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
