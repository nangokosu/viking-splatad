"""Summary report for peers: recall, precision and IoU at both P2 thresholds, whole scene and hidden near band.

Usage (from anywhere; paths resolve from the repo root):
    uv run python docs/research/build_summary_report.py [--output PATH]

Reads each session's Gaussian-occupancy summary (scripts/20, working tree) and writes a Markdown report.
Every number is read from the summaries; nothing is recomputed except the arm-pair gaps, which are the
difference of the two arms' means, and the separated flag, which is true when the two arms' [min, max]
ranges over their repeats do not overlap (the same definition as scripts/20 and scripts/21).

Averaging per band: the whole scene is pooled over cells (the headline of session_comparison.md); the
hidden near band is the mean over objects (the reading-rule view of PLAN §5).
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from viking_splatad.eval.evaluation_terminology import ARM_LABEL_BY_METHOD

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
# Averaging used for each band: whole scene pooled over cells, hidden near band mean over objects.
WHOLE_SCENE_AVERAGING = "pooled"
HIDDEN_AVERAGING = "mean"
# Suffix of the hidden near band metrics (``recall_Hnear`` etc.).
HIDDEN_NEAR_SUFFIX = "_Hnear"
# Context keys of the hidden near band: GT cells and objects holding one.
HIDDEN_CONTEXT_KEYS = ("hidden_near/gt_cells", "hidden_near/objects_with_hidden_gt")
# Decimal places of every printed fraction (fractions in [0, 1]; 3 places resolve 0.1 %).
DECIMALS = 3


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


def metric_key(threshold_text: str, metric_name: str, band: str) -> str:
    """Flat summary key of one metric at one threshold and band.

    Args:
        threshold_text: P2 threshold key, e.g. ``0.5``.
        metric_name: ``recall``, ``precision`` or ``iou``.
        band: ``whole`` or ``hidden``.

    Returns:
        Key such as ``obstacles/P2@0.5/recall/pooled`` or ``obstacles/P2@0.5/recall_Hnear/mean``.
    """
    if band == "whole":
        return f"{HEADLINE_GROUP}/P2@{threshold_text}/{metric_name}/{WHOLE_SCENE_AVERAGING}"
    return f"{HEADLINE_GROUP}/P2@{threshold_text}/{metric_name}{HIDDEN_NEAR_SUFFIX}/{HIDDEN_AVERAGING}"


def format_mean_std(statistics: Dict[str, Any]) -> str:
    """Mean ± std of one metric, as printed in the tables.

    Args:
        statistics: A summary metric entry with ``mean`` and ``std``.

    Returns:
        Text such as ``0.342 ± 0.006``.
    """
    return f"{statistics['mean']:.{DECIMALS}f} ± {statistics['std']:.{DECIMALS}f}"


def gap_and_separation(arm_stats: Dict[str, Any], base_stats: Dict[str, Any]) -> Tuple[float, bool]:
    """Gap between two arms and whether their repeat ranges are separated.

    Args:
        arm_stats: The compared arm's statistic (``mean``, ``min``, ``max``).
        base_stats: The baseline arm's statistic.

    Returns:
        ``(gap, separated)``: gap = arm mean − baseline mean; separated = the [min, max] ranges do not overlap.
    """
    gap = arm_stats["mean"] - base_stats["mean"]
    separated = arm_stats["min"] > base_stats["max"] or base_stats["min"] > arm_stats["max"]
    return gap, separated


def format_gap(gap: float, separated: bool) -> str:
    """Printed gap with its separation label.

    Args:
        gap: Difference of means.
        separated: True when the ranges do not overlap.

    Returns:
        Text such as ``+0.038 (sep)`` or ``−0.005 (ovl)``.
    """
    sign = "+" if gap >= 0 else "−"
    label = "sep" if separated else "ovl"
    return f"{sign}{abs(gap):.{DECIMALS}f} ({label})"


def band_tables(summaries: Dict[str, Dict[str, Any]], band: str, threshold_text: str) -> List[str]:
    """Markdown lines of one band at one threshold: arm values, then the gaps against stock.

    Args:
        summaries: Session name → summary.
        band: ``whole`` or ``hidden``.
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
        key = metric_key(threshold_text, metric_name, band)
        cells = [format_mean_std(arm_by_method(summaries[session])[method]["metrics"][key]) for session, method in columns]
        lines.append(f"| {metric_label} | " + " | ".join(cells) + " |")
    # Step 3: gaps of each compared arm against stock, per session.
    gap_columns = [(session, method) for session in SESSIONS for method in COMPARED_METHODS]
    lines += ["", "| metric | " + " | ".join(f"{session} {ARM_LABEL_BY_METHOD[method]} − stock" for session, method in gap_columns) + " |",
              "|---|" + "---|" * len(gap_columns)]
    for metric_name, metric_label in METRIC_LABELS:
        key = metric_key(threshold_text, metric_name, band)
        cells = []
        for session, method in gap_columns:
            arms = arm_by_method(summaries[session])
            gap, separated = gap_and_separation(arms[method]["metrics"][key], arms[BASELINE_METHOD]["metrics"][key])
            cells.append(format_gap(gap, separated))
        lines.append(f"| {metric_label} | " + " | ".join(cells) + " |")
    return lines + [""]


def hidden_context_lines(summaries: Dict[str, Dict[str, Any]]) -> List[str]:
    """Markdown lines with the hidden near band's evidence size per session.

    Args:
        summaries: Session name → summary.

    Returns:
        Markdown bullet lines.
    """
    lines = []
    for session in SESSIONS:
        stock_metrics = arm_by_method(summaries[session])[BASELINE_METHOD]["metrics"]
        cells_key, objects_key = (f"{HEADLINE_GROUP}/{name}" for name in HIDDEN_CONTEXT_KEYS)
        lines.append(f"- {session}: {stock_metrics[cells_key]['mean']:.0f} hidden obstacle GT cells in "
                     f"{stock_metrics[objects_key]['mean']:.0f} objects (H_near).")
    return lines


def build_report() -> str:
    """Whole report: TL;DR, whole scene tables, hidden near band tables, context and definitions.

    Returns:
        Markdown text.
    """
    summaries = {session: load_summary(session) for session in SESSIONS}
    thresholds = summaries[SESSIONS[0]]["p2_thresholds"]
    lines = ["# Radar arms: recall, precision and IoU, whole scene and hidden near band", "",
             "## TL;DR", ""]
    # Step 1: TL;DR bullets from the same numbers as the tables, one per band and threshold.
    for band, band_name in (("whole", "Whole scene (pooled)"), ("hidden", "Hidden near band (mean over objects)")):
        for threshold_text in thresholds:
            parts = []
            for session in SESSIONS:
                arms = arm_by_method(summaries[session])
                for metric_name, metric_label in METRIC_LABELS:
                    key = metric_key(threshold_text, metric_name, band)
                    gap, separated = gap_and_separation(arms["splatad-radar-init-lidar"]["metrics"][key],
                                                        arms[BASELINE_METHOD]["metrics"][key])
                    parts.append(f"{session} {metric_label} {format_gap(gap, separated)}")
            lines.append(f"- {band_name}, P2 ≥ {threshold_text}, init + loss − stock: " + "; ".join(parts) + ".")
    lines += ["", "Radar init alone is in the tables below. Every gap is read only when it exceeds the spread "
              "within the arms (sep = the two arms' [min, max] ranges do not overlap).", ""]
    # Step 2: whole scene, then hidden near band, each at both thresholds.
    lines += ["## Whole scene: obstacles, pooled over every scored cell", ""]
    for threshold_text in thresholds:
        lines += [f"### P2 occupied at ≥ {threshold_text}", ""] + band_tables(summaries, "whole", threshold_text)
    lines += ["## Hidden near band (H_near): obstacles, mean over objects", ""]
    lines += hidden_context_lines(summaries) + [""]
    for threshold_text in thresholds:
        lines += [f"### P2 occupied at ≥ {threshold_text}", ""] + band_tables(summaries, "hidden", threshold_text)
    # Step 3: definitions for peers reading the tables.
    lines += ["## Definitions", "",
              "- **Recall**: share of obstacle cells the model fills. **Precision**: share of filled cells near an obstacle "
              "that are obstacle cells. **IoU**: both errors in one number (overlap ÷ union in the scored area).",
              "- **P2 occupied at ≥ t**: a cell counts as filled when the model's occupancy there is at least t. 0.5 is the "
              "binding threshold; 0.8 is a stricter check.",
              "- **Whole scene**: all scored cells, pooled. **Hidden near band (H_near)**: cells no lidar beam reached, "
              "within 20 m of the driven path.",
              "- **Arms**: stock (no radar); radar init (radar seeds at initialisation); init + loss (radar seeds and a "
              "label-free opacity loss near them). Each arm trained 5 times per session.",
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
