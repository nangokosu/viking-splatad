"""Headline figure: whole-scene then hidden-near-band recall and precision per arm, one panel per session.

Usage (from anywhere; paths resolve from the repo root). matplotlib lives in the isolated viz environment
(viz/pyproject.toml), not in the root project, so the script runs there:
    uv run --project viz python docs/research/hidden_space/plot_headline_near_band.py [--output-dir PATH]

Reads each session's Gaussian-occupancy summary (scripts/20, working tree) and draws, for the headline
class group at the binding P2 threshold, the mean over objects of each arm over its repeats. Rows are read
in order: the whole scene first (recall, precision), then the hidden near band (recall_Hnear, precision_Hnear).
Each arm is a dot at its mean over repeats, with a bar from the lowest to the highest repeat. Nothing is
recomputed: every plotted value is read from the summary's ``metrics`` block (keys
``<group>/P2@<t>/<metric>/mean`` with ``min``, ``mean``, ``max``), and the threshold ``t`` is read from the
summary's ``settings.occupancy_threshold``.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

# Non-interactive backend: the script only writes files, so no display is needed.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)

# Repo root: this file lives in docs/research/hidden_space/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Per-session summaries written by scripts/20, relative to the repo root.
SUMMARY_PATH_TEMPLATE = "reports/gaussian_occupancy/{session}/gaussian_occupancy_summary.json"
# Output folder for the figure, next to the other occupancy figures.
DEFAULT_OUTPUT_DIR = "reports/gaussian_occupancy/figures"
# Output file name of the figure.
OUTPUT_FILE_NAME = "headline_whole_then_hidden.png"
# Sessions drawn, in panel order (CLAUDE.md "Current scope").
SESSIONS = ("may_short_grass", "june_tall_grass")
# Headline class group (configs/eval.yaml class_groups).
HEADLINE_GROUP = "obstacles"
# Settings key holding the binding P2 occupancy threshold (PLAN §5), read from each summary.
THRESHOLD_SETTING_KEY = "occupancy_threshold"
# Figure rows in reading order: (metric key stem, axis label). Whole scene first, then the hidden near band.
FIGURE_ROWS = (("recall", "recall (whole scene)"),
               ("precision", "precision (whole scene)"),
               ("recall_Hnear", "recall (hidden near band)"),
               ("precision_Hnear", "precision (hidden near band)"))
# Averaging used for the figure: mean over objects (each object weighs the same).
AVERAGING_SUFFIX = "mean"
# Arm display order, keyed by the SplatAD method name stored in each arm.
ARM_ORDER_BY_METHOD = ("splatad", "splatad-radar-init", "splatad-radar-init-lidar")
# Arm labels. Kept here rather than imported from viking_splatad, because this script runs in the viz
# environment, which does not install the root package.
ARM_LABEL_BY_METHOD = {
    "splatad": "stock",
    "splatad-radar-init": "radar init",
    "splatad-radar-init-lidar": "init + loss",
}
# Figure geometry in inches per panel.
PANEL_WIDTH_INCHES = 5.0
PANEL_HEIGHT_INCHES = 3.4
# Dot size of an arm's mean, in points.
MEAN_MARKER_SIZE = 9
# Marker colour of an arm's mean.
MEAN_MARKER_COLOUR = "#1f4e79"
# Half-width of the error-bar caps, in points.
ERROR_BAR_CAP_SIZE = 5
# Opacity of the horizontal grid lines (0 = invisible, 1 = solid).
GRID_ALPHA = 0.3
# Offset of each mean's value label from its dot, in points (right, up).
MEAN_LABEL_OFFSET_POINTS = (10, 6)
# Font size of the value labels and of the figure title, in points.
MEAN_LABEL_FONT_SIZE = 9
TITLE_FONT_SIZE = 10
# Headroom above the highest repeat, as a fraction of it, so value labels clear the panel title.
HEADROOM_FRACTION = 0.15
# Output resolution in dots per inch.
FIGURE_DPI = 150


def load_summary(session: str) -> Dict[str, Any]:
    """Read one session's occupancy summary from disk.

    Args:
        session: Session name, e.g. ``june_tall_grass``.

    Returns:
        The parsed summary JSON (keys include ``arms`` and ``settings``).
    """
    # Resolve the summary path from the session name and read it.
    summary_path = PROJECT_ROOT / SUMMARY_PATH_TEMPLATE.format(session=session)
    with summary_path.open() as summary_file:
        return json.load(summary_file)


def arm_points(summary: Dict[str, Any], metric_name: str, threshold_text: str) -> List[Dict[str, Any]]:
    """Collect each arm's statistic for one figure row, in display order.

    Args:
        summary: A session summary as returned by ``load_summary``.
        metric_name: Metric stem, e.g. ``recall`` (whole scene) or ``recall_Hnear`` (hidden near band).
        threshold_text: P2 threshold key as the summary spells it, e.g. ``0.5``.

    Returns:
        One dict per arm with ``label``, ``mean``, ``min`` and ``max`` taken from the summary key
        ``obstacles/P2@<threshold>/<metric_name>/mean``.
    """
    # Full key of the statistic: group / threshold / metric / averaging.
    metric_key = f"{HEADLINE_GROUP}/P2@{threshold_text}/{metric_name}/{AVERAGING_SUFFIX}"
    points = []
    # Walk the arms in the fixed display order.
    for method in ARM_ORDER_BY_METHOD:
        for arm_summary in summary["arms"].values():
            # Match each arm by its SplatAD method name, not by its profile name.
            if arm_summary["train_method"] != method:
                continue
            statistics = arm_summary["metrics"][metric_key]
            points.append({"label": ARM_LABEL_BY_METHOD[method],
                           "mean": statistics["mean"],
                           "min": statistics["min"],
                           "max": statistics["max"]})
    return points


def draw_panel(axis, points: List[Dict[str, Any]], title: str, y_label: str) -> None:
    """Draw one metric's arms as mean dots with min-to-max bars.

    Args:
        axis: Matplotlib Axes to draw on.
        points: Arm entries from ``arm_points``.
        title: Panel title (the session name).
        y_label: Axis label for the metric.
    """
    # Horizontal positions: one column per arm, in display order.
    column_positions = list(range(len(points)))
    for column_position, point in zip(column_positions, points):
        # Error-bar half-widths: distance from the mean down to the lowest and up to the highest repeat.
        # max(0, ...) guards float rounding: a mean can land a hair above the max when repeats are equal.
        lower_error = max(0.0, point["mean"] - point["min"])
        upper_error = max(0.0, point["max"] - point["mean"])
        axis.errorbar(column_position, point["mean"], yerr=[[lower_error], [upper_error]],
                      fmt="o", markersize=MEAN_MARKER_SIZE, capsize=ERROR_BAR_CAP_SIZE,
                      color=MEAN_MARKER_COLOUR)
        # Print the mean beside its dot so the reader does not have to estimate it.
        axis.annotate(f"{point['mean']:.3f}", (column_position, point["mean"]),
                      textcoords="offset points", xytext=MEAN_LABEL_OFFSET_POINTS,
                      fontsize=MEAN_LABEL_FONT_SIZE)
    axis.set_xticks(column_positions)
    axis.set_xticklabels([point["label"] for point in points])
    axis.set_title(title)
    axis.set_ylabel(y_label)
    # Top limit: highest repeat of any arm plus headroom for the value labels. An empty panel is refused
    # by name, so a missing arm is visible rather than an opaque max() error.
    if not points:
        raise ValueError(f"no arm of panel '{title}' is in the summary")
    highest_value = max(point["max"] for point in points)
    axis.set_ylim(bottom=0, top=highest_value * (1 + HEADROOM_FRACTION))
    axis.grid(axis="y", alpha=GRID_ALPHA)


def build_figure(output_dir: Path) -> Path:
    """Build the headline figure: one row per metric, one column per session.

    The threshold is read from the first session's summary; the summaries must agree on it, which is
    checked here so a mismatch cannot mislabel the figure.

    Args:
        output_dir: Folder the PNG is written to.

    Returns:
        Path of the written PNG.

    Raises:
        ValueError: when the sessions' summaries were scored at different thresholds.
    """
    summaries = {session: load_summary(session) for session in SESSIONS}
    # Step 1: the binding threshold, as the summaries spell it, must be the same in every session.
    threshold_texts = {str(summary["settings"][THRESHOLD_SETTING_KEY]) for summary in summaries.values()}
    if len(threshold_texts) != 1:
        raise ValueError(f"sessions scored at different P2 thresholds: {sorted(threshold_texts)}")
    threshold_text = threshold_texts.pop()
    # Step 2: grid, rows are the four metrics, columns are the two sessions.
    figure, axes = plt.subplots(len(FIGURE_ROWS), len(SESSIONS),
                                figsize=(PANEL_WIDTH_INCHES * len(SESSIONS),
                                         PANEL_HEIGHT_INCHES * len(FIGURE_ROWS)),
                                squeeze=False)
    # Step 3: fill each cell with one metric row and one session column.
    for row_index, (metric_name, y_label) in enumerate(FIGURE_ROWS):
        for column_index, session in enumerate(SESSIONS):
            draw_panel(axes[row_index][column_index],
                       arm_points(summaries[session], metric_name, threshold_text),
                       title=session, y_label=y_label)
    # Step 4: title, layout and save.
    figure.suptitle(f"{HEADLINE_GROUP}, P2 occupied at >= {threshold_text}, mean over objects\n"
                    "Top: whole scene. Bottom: hidden near band. Dots = mean over repeats, bars = lowest to highest repeat",
                    fontsize=TITLE_FONT_SIZE)
    figure.tight_layout()
    output_path = output_dir / OUTPUT_FILE_NAME
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)
    return output_path


def main() -> None:
    """Parse the command line, build the figure and print its path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / DEFAULT_OUTPUT_DIR),
                        help=f"folder for the PNG (default: {DEFAULT_OUTPUT_DIR})")
    arguments = parser.parse_args()
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(build_figure(output_dir))


if __name__ == "__main__":
    main()
