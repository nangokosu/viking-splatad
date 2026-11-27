"""Summary report for peers: recall, precision and IoU at both P2 thresholds over the whole scene, plus image and lidar render quality.

Usage (from anywhere; paths resolve from the repo root):
    uv run python docs/research/build_summary_report.py [--output PATH]

Reads each session's Gaussian-occupancy summary (scripts/20, working tree) and writes a Markdown report.
Every number is read from the summaries; nothing is recomputed except the arm-pair gaps and separated
flags, which come from ``p2_repeats.compare_arms`` (the function scripts/20 and scripts/21 use): gap = the
difference of the two arms' means; separated = the two arms' [min, max] ranges over their repeats do not
overlap, undefined (n/a) unless both arms have enough defined repeats.

The whole-scene metrics are pooled over cells (the headline of session_comparison.md). The quality rows
(image psnr, ssim, lpips; lidar render metrics) are mean ± std over each arm's repeats.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from viking_splatad.eval.evaluation_terminology import ARM_LABEL_BY_METHOD
from viking_splatad.eval.p2_repeats import compare_arms
from viking_splatad.eval.run_quality import IMAGE_METRIC_NAMES, IMAGE_RESULT_KEY, LIDAR_RESULT_KEY, REPORTED_LIDAR_METRICS

# Repo root: this file lives in docs/research/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Per-session summaries written by scripts/20, relative to the repo root.
SUMMARY_PATH_TEMPLATE = "reports/gaussian_occupancy/{session}/gaussian_occupancy_summary.json"
# Default output of the report, next to the other occupancy reports.
DEFAULT_OUTPUT_RELATIVE_PATH = "reports/gaussian_occupancy/summary_report.md"
# Sessions in table order (CLAUDE.md "Current scope").
SESSIONS = ("june_tall_grass", "may_short_grass")
# Arm order by SplatAD method; the first is the baseline the gaps are taken against.
METHOD_ORDER = ("splatad", "splatad-radar-init", "splatad-radar-init-lidar")
BASELINE_METHOD = "splatad"
# Arms compared with the baseline, in table order.
COMPARED_METHODS = ("splatad-radar-init", "splatad-radar-init-lidar")
# Headline group of the report.
HEADLINE_GROUP = "obstacles"
# The three metrics of every table, with their row labels.
METRIC_LABELS = (("recall", "recall"), ("precision", "precision"), ("iou", "IoU"))
# Averaging of the whole-scene metrics: pooled over cells.
WHOLE_SCENE_AVERAGING = "pooled"
# First key segment of the per-run quality metrics in the summary (scripts/20 ``flatten_quality_metrics``).
QUALITY_KEY_PREFIX = "quality"
# Decimal places of every printed fraction (fractions in [0, 1]; 3 places resolve 0.1 %).
DECIMALS = 3
# Printed in place of a statistic that is undefined (too few defined repeats).
UNDEFINED_TEXT = "n/a"
# Key of the one metric handed to ``compare_arms`` by ``gap_and_separation``.
COMPARED_METRIC_KEY = "metric"


def load_summary(session: str) -> Dict[str, Any]:
    """Read one session's summary.

    Args:
        session: Session name, e.g. ``june_tall_grass``.

    Returns:
        The parsed summary JSON.
    """
    summary_path = PROJECT_ROOT / SUMMARY_PATH_TEMPLATE.format(session=session)
    return json.loads(summary_path.read_text())


def arm_by_method(summary: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Index a summary's arms by SplatAD method name.

    Args:
        summary: A session summary.

    Returns:
        Map from method name to that arm's entry (its ``metrics`` and ``train_method``).
    """
    return {arm["train_method"]: arm for arm in summary["arms"].values()}


def metric_key(threshold_text: str, metric_name: str) -> str:
    """Flat summary key of one whole-scene metric at one threshold.

    Args:
        threshold_text: P2 threshold key, e.g. ``0.5``.
        metric_name: ``recall``, ``precision`` or ``iou``.

    Returns:
        Key such as ``obstacles/P2@0.5/recall/pooled``.
    """
    return f"{HEADLINE_GROUP}/P2@{threshold_text}/{metric_name}/{WHOLE_SCENE_AVERAGING}"


def quality_key(result_key: str, metric_name: str) -> str:
    """Flat summary key of one image or lidar render quality metric.

    Args:
        result_key: ``IMAGE_RESULT_KEY`` or ``LIDAR_RESULT_KEY``.
        metric_name: Metric name within that group.

    Returns:
        Key such as ``quality/image/psnr``.
    """
    return f"{QUALITY_KEY_PREFIX}/{result_key}/{metric_name}"


def format_mean_std(statistics: Dict[str, Any]) -> str:
    """Mean ± std of one metric, as printed in the tables.

    Args:
        statistics: A summary metric entry with ``mean`` and ``std`` (``describe_values``: mean None when
            no repeat is defined, std None when fewer than two are).

    Returns:
        Text such as ``0.342 ± 0.006``, ``0.342 ± n/a`` or ``n/a``.
    """
    # No defined repeat: nothing to print.
    if statistics["mean"] is None:
        return UNDEFINED_TEXT
    std_text = UNDEFINED_TEXT if statistics["std"] is None else f"{statistics['std']:.{DECIMALS}f}"
    return f"{statistics['mean']:.{DECIMALS}f} ± {std_text}"


def gap_and_separation(arm_stats: Dict[str, Any], base_stats: Dict[str, Any]) -> Tuple[Optional[float], Optional[bool]]:
    """Gap between two arms and whether their repeat ranges are separated, by ``compare_arms``.

    Args:
        arm_stats: The compared arm's statistic (``describe_values`` entry: n, mean, std, min, max).
        base_stats: The baseline arm's statistic.

    Returns:
        ``(gap, separated)``: gap = arm mean − baseline mean (None when an arm has no defined repeat);
        separated = the [min, max] ranges do not overlap (None unless both arms have enough repeats).
    """
    comparison = compare_arms({COMPARED_METRIC_KEY: arm_stats}, {COMPARED_METRIC_KEY: base_stats})[COMPARED_METRIC_KEY]
    return comparison["gap"], comparison["separated"]


def format_gap(gap: Optional[float], separated: Optional[bool]) -> str:
    """Printed gap with its separation label.

    Args:
        gap: Difference of means, or None.
        separated: True when the ranges do not overlap, None when undefined.

    Returns:
        Text such as ``+0.038 (sep)``, ``−0.005 (ovl)``, ``+0.010 (n/a)`` or ``n/a``.
    """
    # No gap without a defined mean in both arms.
    if gap is None:
        return UNDEFINED_TEXT
    sign = "+" if gap >= 0 else "−"
    label = UNDEFINED_TEXT if separated is None else ("sep" if separated else "ovl")
    return f"{sign}{abs(gap):.{DECIMALS}f} ({label})"


def whole_scene_tables(summaries: Dict[str, Dict[str, Any]], threshold_text: str) -> List[str]:
    """Markdown lines of one threshold: arm values, then the gaps against stock.

    Args:
        summaries: Session name → summary.
        threshold_text: P2 threshold key.

    Returns:
        Markdown lines.
    """
    # Step 1: header with one column per session and arm.
    columns = [(session, method) for session in SESSIONS for method in METHOD_ORDER]
    header = "| metric | " + " | ".join(f"{session} {ARM_LABEL_BY_METHOD[method]}" for session, method in columns) + " |"
    lines = [header, "|---|" + "---|" * len(columns)]
    # Step 2: one row per metric with each arm's mean ± std.
    for metric_name, metric_label in METRIC_LABELS:
        key = metric_key(threshold_text, metric_name)
        cells = [format_mean_std(arm_by_method(summaries[session])[method]["metrics"][key]) for session, method in columns]
        lines.append(f"| {metric_label} | " + " | ".join(cells) + " |")
    # Step 3: gaps of each compared arm against stock, per session.
    gap_columns = [(session, method) for session in SESSIONS for method in COMPARED_METHODS]
    lines += ["", "| metric | " + " | ".join(f"{session} {ARM_LABEL_BY_METHOD[method]} − stock" for session, method in gap_columns) + " |",
              "|---|" + "---|" * len(gap_columns)]
    for metric_name, metric_label in METRIC_LABELS:
        key = metric_key(threshold_text, metric_name)
        cells = []
        for session, method in gap_columns:
            arms = arm_by_method(summaries[session])
            gap, separated = gap_and_separation(arms[method]["metrics"][key], arms[BASELINE_METHOD]["metrics"][key])
            cells.append(format_gap(gap, separated))
        lines.append(f"| {metric_label} | " + " | ".join(cells) + " |")
    return lines + [""]


def quality_table(summary: Dict[str, Any], result_key: str, metric_names: Tuple[str, ...]) -> List[str]:
    """Markdown table of one session's quality metrics: one row per metric, mean ± std per arm.

    Args:
        summary: One session's scripts/20 summary.
        result_key: ``IMAGE_RESULT_KEY`` or ``LIDAR_RESULT_KEY``.
        metric_names: The group's metric names, in row order.

    Returns:
        Markdown lines (header, rule, rows, trailing blank).
    """
    arms = arm_by_method(summary)
    lines = ["| metric | " + " | ".join(ARM_LABEL_BY_METHOD[method] for method in METHOD_ORDER) + " |",
             "|---|" + "---|" * len(METHOD_ORDER)]
    for metric_name in metric_names:
        key = quality_key(result_key, metric_name)
        lines.append(f"| {metric_name} | " + " | ".join(format_mean_std(arms[method]["metrics"][key]) for method in METHOD_ORDER) + " |")
    return lines + [""]


def build_report() -> str:
    """Whole report: TL;DR, whole-scene tables, per-session quality tables and definitions.

    Returns:
        Markdown text.
    """
    summaries = {session: load_summary(session) for session in SESSIONS}
    thresholds = summaries[SESSIONS[0]]["p2_thresholds"]
    # Repeats per arm, read from the summaries' run lists (one figure when every arm has the same count).
    repeat_counts = sorted({len(arm["runs"]) for summary in summaries.values() for arm in summary["arms"].values()})
    repeat_text = " or ".join(str(count) for count in repeat_counts)
    lines = ["# Radar arms: recall, precision and IoU over the whole scene", "",
             "## TL;DR", ""]
    # Step 1: TL;DR bullets from the same numbers as the tables, one per threshold.
    for threshold_text in thresholds:
        parts = []
        for session in SESSIONS:
            arms = arm_by_method(summaries[session])
            for metric_name, metric_label in METRIC_LABELS:
                key = metric_key(threshold_text, metric_name)
                gap, separated = gap_and_separation(arms["splatad-radar-init-lidar"]["metrics"][key],
                                                    arms[BASELINE_METHOD]["metrics"][key])
                parts.append(f"{session} {metric_label} {format_gap(gap, separated)}")
        lines.append(f"- Whole scene (pooled), P2 ≥ {threshold_text}, init + loss − stock: " + "; ".join(parts) + ".")
    lines += ["", "Radar init alone is in the tables below. Every gap is read only when it exceeds the spread "
              "within the arms (sep = the two arms' [min, max] ranges do not overlap).", ""]
    # Step 2: whole scene at each threshold.
    lines += ["## Whole scene: obstacles, pooled over every scored cell", ""]
    for threshold_text in thresholds:
        lines += [f"### P2 occupied at ≥ {threshold_text}", ""] + whole_scene_tables(summaries, threshold_text)
    # Step 3: image (guard) and lidar render quality per session, mean ± std over the repeats.
    for session in SESSIONS:
        lines += [f"## Render quality: {session}", "", "Image metrics (test + guard frames):", ""]
        lines += quality_table(summaries[session], IMAGE_RESULT_KEY, IMAGE_METRIC_NAMES)
        lines += ["Lidar render metrics:", ""]
        lines += quality_table(summaries[session], LIDAR_RESULT_KEY, REPORTED_LIDAR_METRICS)
    # Step 4: definitions for peers reading the tables.
    lines += ["## Definitions", "",
              "- **Recall**: share of obstacle cells the model fills. **Precision**: share of filled cells near an obstacle "
              "that are obstacle cells. **IoU**: both errors in one number (overlap ÷ union in the scored area).",
              "- **P2 occupied at ≥ t**: a cell counts as filled when the model's occupancy there is at least t. 0.5 is the "
              "binding threshold; 0.8 is a stricter check.",
              "- **Whole scene**: all scored cells, pooled.",
              "- **Arms**: stock (no radar); radar init (radar seeds at initialisation); init + loss (radar seeds and a "
              f"label-free opacity loss near them). Each arm trained {repeat_text} times per session.",
              f"- Sources: {', '.join(SUMMARY_PATH_TEMPLATE.format(session=session) for session in SESSIONS)} "
              "(scripts/20), built by docs/research/build_summary_report.py.", ""]
    return "\n".join(lines) + "\n"


def main() -> None:
    """Parse arguments, write the report and print its path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_RELATIVE_PATH)
    arguments = parser.parse_args()
    arguments.output.write_text(build_report())
    print(arguments.output)


if __name__ == "__main__":
    main()
