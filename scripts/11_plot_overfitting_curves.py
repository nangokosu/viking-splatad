"""Plot train-vs-test curves of one SplatAD run from its TensorBoard log (overfitting monitor).

Each test metric (nerfstudio's all-images eval) gets a panel with its own train value: the raw
per-step train metric (thin), its rolling median over about one eval interval, and the test
points joined by a line. A final panel shows the train loss. The region before the training
resolution reaches full size (``resolution_schedule`` x ``num_downscales`` steps) is shaded:
train is rendered at reduced resolution there, so the train/test gap is not meaningful.
Reading rule: overfitting = the test curve getting worse over a sustained stretch after the
shaded region while train keeps improving.

Outputs, in ``--output-dir``:
    overfitting_curves.png  the figure
    overfitting_curves.csv  header ``metric,split,step,value``; split in train, test, train_loss.
                            Values are the raw logged ones: nan and inf are written as Python's
                            float text (``nan``, ``inf``); the figure's y-limits ignore them.

Usage:
    python scripts/11_plot_overfitting_curves.py --train-dir <run dir>/train/<session>/splatad/<run id> \
        --output-dir <run dir>/overfitting

Exit code 0 when written; 1 when there is no events file or no test points (nothing is written).
Needs tensorboard + matplotlib (the cluster environment / the Docker image).
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend, chosen before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

from viking_splatad.eval.training_curves import (  # noqa: E402
    TRAIN_LOSS_TAG,
    MetricPanel,
    curve_rows,
    pair_metric_tags,
    reduced_resolution_end_step,
    rolling_median,
    smoothing_window,
)
from viking_splatad.io.nerfstudio_config import read_config_scalar  # noqa: E402

# Output file names inside --output-dir.
PNG_NAME = "overfitting_curves.png"
CSV_NAME = "overfitting_curves.csv"
CSV_HEADER = ("metric", "split", "step", "value")
# Split labels used in the CSV; the loss series has no test counterpart.
TRAIN_SPLIT, TEST_SPLIT, TRAIN_LOSS_SPLIT = "train", "test", "train_loss"
LOSS_METRIC_NAME = "train_loss"
EXIT_WRITTEN, EXIT_NOTHING_TO_PLOT = 0, 1
SHADING_LABEL = "train at reduced resolution: gap not meaningful"
# Figure geometry: panel size in inches and panels per row (a fixed layout choice, not data-derived).
PANEL_WIDTH_INCHES, PANEL_HEIGHT_INCHES, PANELS_PER_ROW = 5.0, 3.2, 3
# Opacity of the raw train trace and of the shading (visual styling only).
RAW_TRAIN_ALPHA, SHADING_ALPHA = 0.3, 0.15
# More visual styling only: PNG resolution and font sizes of the legend and the suptitle.
FIGURE_DPI, LEGEND_FONT_SIZE, TITLE_FONT_SIZE = 120, 7, 10
# Display-only reference magnitude for the y-margin of data that is all zero (or has no finite
# value): the margin is 5% of it, i.e. 1.0 metric unit. Any positive constant would do, because
# nothing is measured here; it only has to make the lower and upper limit differ.
ZERO_RANGE_REFERENCE_MAGNITUDE = 20.0


def load_scalar_series(train_dir: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load every scalar tag from all ``events.out.tfevents.*`` files of a directory.

    Files are read one by one with ``size_guidance={"scalars": 0}`` (0 keeps all points; the
    default would subsample) and merged per tag: sorted by step, a later file's value wins on a
    repeated step.

    Args:
        train_dir: Directory holding the events files.

    Returns:
        tag -> (steps, values), steps strictly increasing; empty when there is no events file.
    """
    merged_points: Dict[str, Dict[int, float]] = {}
    # Step 1: read each events file independently (a resumed run writes a second file).
    for events_path in sorted(train_dir.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(events_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        # Step 2: collect points per tag, keyed by step so duplicates collapse.
        for tag in accumulator.Tags()["scalars"]:
            tag_points = merged_points.setdefault(tag, {})
            for scalar_event in accumulator.Scalars(tag):
                tag_points[int(scalar_event.step)] = float(scalar_event.value)
    # Step 3: sort each tag by step into arrays.
    series: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for tag, tag_points in merged_points.items():
        sorted_steps = sorted(tag_points)
        series[tag] = (np.array(sorted_steps, dtype=int), np.array([tag_points[step] for step in sorted_steps]))
    return series


def collect_curves(
    series: Dict[str, Tuple[np.ndarray, np.ndarray]], panels: Sequence[MetricPanel]
) -> Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]]:
    """Map the panels' tags and the loss tag to (metric, split) curves for the CSV.

    Args:
        series: Output of ``load_scalar_series``.
        panels: Metric panels from ``pair_metric_tags``.

    Returns:
        (metric, split) -> (steps, values).
    """
    curves = {}
    for panel in panels:
        curves[(panel.metric, TEST_SPLIT)] = series[panel.test_tag]
        if panel.train_tag is not None:
            curves[(panel.metric, TRAIN_SPLIT)] = series[panel.train_tag]
    if TRAIN_LOSS_TAG in series:
        curves[(LOSS_METRIC_NAME, TRAIN_LOSS_SPLIT)] = series[TRAIN_LOSS_TAG]
    return curves


def read_int_field(config_path: Path, field_name: str) -> Optional[int]:
    """Read an integer field of config.yml, None when absent.

    Args:
        config_path: The run's config.yml.
        field_name: Field name.

    Returns:
        The integer, or None.
    """
    field_text = read_config_scalar(config_path, field_name)
    return None if field_text is None else int(field_text)


def run_step_range(series: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> Tuple[int, int]:
    """Shared x-range of every panel: from step 0 to the largest step of any series.

    Args:
        series: Output of ``load_scalar_series``.

    Returns:
        (0, max step over all series).
    """
    return 0, int(max(steps.max() for steps, _ in series.values() if len(steps) > 0))


def warm_up_step_for_panel(
    train_steps: Optional[np.ndarray], test_steps: np.ndarray, shading_end_step: Optional[int]
) -> int:
    """First step whose values set a panel's y-limits (earlier values are start-up transients).

    It is the reduced-resolution end step when any train sample lies beyond it, else the first
    test-eval step (the smallest test step).

    Args:
        train_steps: Steps of the panel's train samples, None when no train value is logged.
        test_steps: Steps of the panel's test points (may be empty for the loss panel).
        shading_end_step: End of the reduced-resolution region, or None when unknown.

    Returns:
        The warm-up step.
    """
    if shading_end_step is not None and train_steps is not None and bool((train_steps > shading_end_step).any()):
        return shading_end_step
    return int(test_steps.min()) if len(test_steps) > 0 else 0


def panel_y_limits(
    warm_curves: Sequence[Tuple[np.ndarray, np.ndarray]], warm_up_step: int,
    test_values: np.ndarray = np.empty(0),
) -> Tuple[float, float]:
    """y-limits from the plotted values at steps >= warm_up_step, plus all test values.

    limits = [min, max] widened by 5% of (max - min) on each side (display margin only). A zero
    range (constant data) is widened by 5% of |value| (or 1.0 at zero) so the limits differ.
    Non-finite values (nan, inf) are dropped first, since they would make the limits NaN and
    ``set_ylim`` raise; when nothing finite remains the limits are those of a lone zero.

    Args:
        warm_curves: (steps, values) of the train raw and train median curves.
        warm_up_step: Values at earlier steps are ignored.
        test_values: Test values, always inside the limits.

    Returns:
        (lower, upper) with lower < upper.
    """
    # Step 1: collect every value that must be visible.
    kept_values = [test_values] + [values[steps >= warm_up_step] for steps, values in warm_curves]
    all_values = np.concatenate(kept_values)
    finite_values = all_values[np.isfinite(all_values)]
    # Step 1b: with no finite value, fall back to the zero-range path (deterministic limits).
    if len(finite_values) == 0:
        finite_values = np.zeros(1)
    lowest, highest = float(finite_values.min()), float(finite_values.max())
    # Step 2: display margin, with a guard for constant data.
    value_range = highest - lowest
    zero_range_scale = abs(lowest) if lowest != 0 else ZERO_RANGE_REFERENCE_MAGNITUDE
    margin = 0.05 * value_range if value_range > 0 else 0.05 * zero_range_scale
    return lowest - margin, highest + margin


def draw_figure(
    run_name: str,
    series: Dict[str, Tuple[np.ndarray, np.ndarray]],
    panels: Sequence[MetricPanel],
    eval_interval: Optional[int],
    shading_end_step: Optional[int],
):
    """Draw one panel per metric plus a train-loss panel.

    Args:
        run_name: Name of the run folder, for the title.
        series: Output of ``load_scalar_series``.
        panels: Metric panels from ``pair_metric_tags``.
        eval_interval: ``steps_per_eval_all_images``, or None when config.yml lacks it.
        shading_end_step: End of the reduced-resolution region, or None to skip the shading.

    Returns:
        The matplotlib Figure.
    """
    # Step 1: grid with one cell per metric panel, plus the loss panel when logged.
    has_loss = TRAIN_LOSS_TAG in series
    panel_count = len(panels) + int(has_loss)
    column_count = min(PANELS_PER_ROW, panel_count)
    row_count = -(-panel_count // column_count)  # ceiling division
    figure, axes_grid = plt.subplots(
        row_count, column_count, figsize=(PANEL_WIDTH_INCHES * column_count, PANEL_HEIGHT_INCHES * row_count),
        squeeze=False)
    axes_list = list(axes_grid.ravel())

    run_x_limits = run_step_range(series)
    # Step 2: metric panels: raw train, rolling-median train, test markers.
    for panel, axes in zip(panels, axes_list):
        test_steps, test_values = series[panel.test_tag]
        warm_curves = []
        train_steps = None
        if panel.train_tag is not None:
            train_steps, train_values = series[panel.train_tag]
            axes.plot(train_steps, train_values, color="tab:blue", linewidth=0.6, alpha=RAW_TRAIN_ALPHA,
                      label="train (raw)")
            warm_curves.append((train_steps, train_values))
            if eval_interval is not None and len(train_steps) > 1:
                # Median window spans about one eval interval, so it is comparable to a test point.
                window = smoothing_window(train_steps, eval_interval)
                median_values = rolling_median(train_values, window)
                axes.plot(train_steps, median_values, color="tab:blue", linewidth=1.5,
                          label=f"train (median, {window} samples)")
                warm_curves.append((train_steps, median_values))
        axes.plot(test_steps, test_values, color="tab:red", marker="o", markersize=3, linewidth=1.2, label="test")
        axes.set_title(panel.metric if panel.train_tag is not None else f"{panel.metric} (no train value logged)")
        axes.set_xlabel("step")
        # y-limits skip the start-up spike so the train/test detail stays visible.
        axes.set_ylim(panel_y_limits(
            warm_curves, warm_up_step_for_panel(train_steps, test_steps, shading_end_step), test_values))
    # Step 3: loss panel.
    if has_loss:
        loss_axes = axes_list[len(panels)]
        loss_steps, loss_values = series[TRAIN_LOSS_TAG]
        loss_axes.plot(loss_steps, loss_values, color="tab:blue", linewidth=0.8)
        loss_axes.set_title(TRAIN_LOSS_TAG)
        loss_axes.set_xlabel("step")
        loss_axes.set_ylim(panel_y_limits(
            [(loss_steps, loss_values)], warm_up_step_for_panel(loss_steps, np.empty(0), shading_end_step)))
    # Step 4: shade the reduced-resolution phase (clipped to the run range below) and label it once.
    if shading_end_step is not None:
        for axes in axes_list[:panel_count]:
            axes.axvspan(0, shading_end_step, color="grey", alpha=SHADING_ALPHA,
                         label=SHADING_LABEL if axes is axes_list[0] else None)
    # Step 4b: every panel shares the run's x-range (set after the shading, which would stretch it).
    for axes in axes_list[:panel_count]:
        axes.set_xlim(run_x_limits)
    axes_list[0].legend(fontsize=LEGEND_FONT_SIZE)
    for unused_axes in axes_list[panel_count:]:
        unused_axes.axis("off")
    # Step 5: title states the run, the eval interval and, if skipped, why there is no shading.
    title = f"{run_name}  |  steps_per_eval_all_images = {eval_interval}"
    if shading_end_step is None:
        title += "  |  no shading (resolution_schedule / num_downscales absent from config.yml)"
    figure.suptitle(title, fontsize=TITLE_FONT_SIZE)
    figure.tight_layout()
    return figure


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (default sys.argv[1:]).

    Returns:
        EXIT_WRITTEN, or EXIT_NOTHING_TO_PLOT when there is no events file or no test points.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-dir", type=Path, required=True, help="dir with config.yml and events.out.tfevents.*")
    parser.add_argument("--output-dir", type=Path, required=True, help="where the PNG and CSV are written")
    arguments = parser.parse_args(argv)
    train_dir: Path = arguments.train_dir

    # Step 1: load the log; nothing to plot without events or without a test point.
    series = load_scalar_series(train_dir)
    if not series:
        print(f"no events file in {train_dir}", file=sys.stderr)
        return EXIT_NOTHING_TO_PLOT
    panels = pair_metric_tags(series.keys())
    if not any(len(series[panel.test_tag][0]) > 0 for panel in panels):
        print(f"no test points in {train_dir} (nothing written)", file=sys.stderr)
        return EXIT_NOTHING_TO_PLOT

    # Step 2: run settings from config.yml; shading needs both values, never invented.
    config_path = train_dir / "config.yml"
    eval_interval = read_int_field(config_path, "steps_per_eval_all_images")
    resolution_schedule = read_int_field(config_path, "resolution_schedule")
    num_downscales = read_int_field(config_path, "num_downscales")
    shading_end_step = (None if resolution_schedule is None or num_downscales is None
                        else reduced_resolution_end_step(resolution_schedule, num_downscales))

    # Step 3: figure and CSV.
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    figure = draw_figure(train_dir.name, series, panels, eval_interval, shading_end_step)
    figure.savefig(arguments.output_dir / PNG_NAME, dpi=FIGURE_DPI)
    plt.close(figure)
    with open(arguments.output_dir / CSV_NAME, "w", newline="") as csv_handle:
        csv_writer = csv.writer(csv_handle, lineterminator="\n")
        csv_writer.writerow(CSV_HEADER)
        csv_writer.writerows(curve_rows(collect_curves(series, panels)))
    print(f"wrote {arguments.output_dir / PNG_NAME} and {arguments.output_dir / CSV_NAME}")
    return EXIT_WRITTEN


if __name__ == "__main__":
    sys.exit(main())
