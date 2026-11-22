"""Compare the radar arms' effect across sessions (May short grass vs June tall grass), PLAN §5.

Reads the scripts/20 summaries of two or more sessions. Inside each session, the arms are matched by
training method (stock ``splatad``, ``splatad-radar-init``, ``splatad-radar-init-lidar``), because
profile names differ between sessions. Per session and method pair it reports:
- the P1/P2 voxel gaps from scripts/20 (gap = later arm - earlier arm, so a radar gain is positive
  for higher-is-better metrics; the JSON also holds counts such as present_but_hidden, where it is negative),
  beside both arms' ranges and the "separated" flag (the arms' [min, max] do not overlap);
- the scripts/13 P1 cuboid metrics (ungated detection rate T45 and point precision T46), pooled over
  the same runs with the same statistics (eval/p2_repeats.py);
- the difference of the gaps between sessions (last session given - first session given; sessions in
  between are reported per session but do not enter the difference).
  The claim expects a radar gain in June (tall grass occludes trunks), and none in May.

Descriptive only: repeats share one seed, so they are not paired. A gap means something only beside
the spread within both arms.

Usage:
    uv run python scripts/21_compare_sessions.py \
        reports/gaussian_occupancy/may_short_grass/gaussian_occupancy_summary.json \
        reports/gaussian_occupancy/june_tall_grass/gaussian_occupancy_summary.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from viking_splatad.config import find_project_root
from viking_splatad.eval.p2_repeats import arm_pairs, compare_arms, describe_arm

SUMMARY_SCHEMA = "viking-splatad/gaussian-occupancy-summary/1"
SEMANTIC_REPORT_SCHEMA = "viking-splatad/semantic-occupancy/1"
SEMANTIC_REPORT_RELATIVE_PATH = Path("eval") / "semantic_occupancy.json"
RUNS_DIR_NAME = "runs"
# A cross-session difference needs at least two sessions.
MINIMUM_SESSION_COUNT = 2
REPORT_SCHEMA = "viking-splatad/session-comparison/1"
DEFAULT_OUTPUT_RELATIVE_DIR = Path("reports") / "gaussian_occupancy"
OUTPUT_JSON_NAME = "session_comparison.json"
OUTPUT_MARKDOWN_NAME = "session_comparison.md"
# Method order: stock, then the radar variants as they build on each other (same as scripts/20), so
# every gap reads "radar variant - its baseline".
METHOD_ORDER = ("splatad", "splatad-radar-init", "splatad-radar-init-lidar")
METHOD_LABELS = {"splatad": "stock", "splatad-radar-init": "radar init",
                 "splatad-radar-init-lidar": "radar init + loss"}
# P1 is read ungated: the binding P1 form (PLAN §5); the gated set is SplatAD's sensor-faithful view.
P1_POINT_SET = "ungated"
# Metrics shown in the markdown (the JSON holds every scripts/20 metric). Classes: trunks are the
# claim; rocks and bushes are the other scored GT classes, read as controls.
HEADLINE_CLASSES = ("tree_trunk", "rock", "bush_or_small_tree")
HEADLINE_METRICS = ("P1cuboid/detection_rate", "P1cuboid/point_precision", "P1/recall", "P2/recall",
                    "P1/recall_blocked", "P2/recall_blocked", "P1/precision", "P2/precision", "P2/iou")
# Decimal places in the markdown tables (metrics are fractions in [0, 1]; 3 places resolve 0.1 %).
TABLE_DECIMALS = 3


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI arguments.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("summaries", nargs="+", type=Path,
                        help="scripts/20 summary JSONs, one per session, earlier session first")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: reports/gaussian_occupancy")
    return parser


def semantic_run_metrics(run_dir: Path, session: str) -> Dict[str, Optional[float]]:
    """One run's scripts/13 P1 cuboid metrics, flattened as ``class/P1cuboid/metric``.

    Args:
        run_dir: Run folder holding eval/semantic_occupancy.json.
        session: The session the run must belong to.

    Returns:
        ``class/P1cuboid/detection_rate`` and ``class/P1cuboid/point_precision`` -> value.

    Raises:
        ValueError: Missing report, wrong schema or session, or the P1 point set is unavailable.
    """
    report_path = run_dir / SEMANTIC_REPORT_RELATIVE_PATH
    if not report_path.exists():
        raise ValueError(f"{run_dir}: no {SEMANTIC_REPORT_RELATIVE_PATH} (run scripts/13)")
    report = json.loads(report_path.read_text())
    # Boundary checks: the report must be scripts/13's current schema, of the right session.
    if report.get("schema") != SEMANTIC_REPORT_SCHEMA:
        raise ValueError(f"{run_dir}: semantic report schema {report.get('schema')!r}, need {SEMANTIC_REPORT_SCHEMA!r}")
    if report["session"] != session:
        raise ValueError(f"{run_dir}: semantic report session {report['session']!r}, not {session!r}")
    point_set = report["point_sets"][P1_POINT_SET]
    if not point_set["available"]:
        raise ValueError(f"{run_dir}: {P1_POINT_SET} point set unavailable in {SEMANTIC_REPORT_RELATIVE_PATH}")
    # Flatten per class: T45 detection rate at the class's n_min, T46 point precision.
    flat: Dict[str, Optional[float]] = {}
    for class_name, class_metrics in point_set["classes"].items():
        flat[f"{class_name}/P1cuboid/detection_rate"] = class_metrics["detection"]["detection_rate"]
        flat[f"{class_name}/P1cuboid/point_precision"] = class_metrics["point_precision"]["precision"]
    return flat


def session_arms(summary: Dict[str, Any], project_root: Path) -> Dict[str, Dict[str, Any]]:
    """A session's arms keyed by training method, with their scripts/20 and scripts/13 statistics.

    Args:
        summary: One scripts/20 summary.
        project_root: Repo root (run folders are runs/<run_id>).

    Returns:
        method -> {"profile", "run_ids", "metrics"}; metrics merge scripts/20's descriptions with the
        scripts/13 cuboid metrics described over the same runs.

    Raises:
        ValueError: Two arms of one method (ambiguous match across sessions) or a bad scripts/13 report.
    """
    session = summary["session"]
    arms_by_method: Dict[str, Dict[str, Any]] = {}
    for profile, arm in summary["arms"].items():
        method = arm["train_method"]
        # One arm per method per session, or the cross-session match is ambiguous.
        if method in arms_by_method:
            raise ValueError(f"{session}: profiles {arms_by_method[method]['profile']} and {profile} share method "
                             f"{method}; summarize one arm per method")
        run_ids = [run_entry["run_id"] for run_entry in arm["runs"]]
        if not run_ids:
            raise ValueError(f"{session}: arm {profile} has no runs")
        # scripts/13 cuboid metrics over exactly the runs scripts/20 pooled for this arm.
        cuboid_descriptions = describe_arm([semantic_run_metrics(project_root / RUNS_DIR_NAME / run_id, session)
                                            for run_id in run_ids])
        arms_by_method[method] = {"profile": profile, "run_ids": run_ids,
                                  "metrics": {**arm["metrics"], **cuboid_descriptions}}
    return arms_by_method


def method_rank(method: str) -> int:
    """Position of a method in METHOD_ORDER; unknown methods sort last.

    Args:
        method: Training method name.

    Returns:
        Sort rank.
    """
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def compare_sessions(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per method pair and metric: each session's gap and separation, and the gap differences.

    Args:
        sessions: [{"session", "arms": session_arms(...)}] in the order given (earlier session first).

    Returns:
        [{"later", "earlier", "metrics": {metric: {"per_session": {session: compare_arms entry},
        "gap_difference": last session gap - first session gap or None}}}], one per method pair present
        in every session.
    """
    # Method pairs present in every session, ordered so "later" is the radar variant.
    common_methods = sorted(set.intersection(*(set(entry["arms"]) for entry in sessions)),
                            key=lambda method: (method_rank(method), method))
    comparisons: List[Dict[str, Any]] = []
    for earlier_method, later_method in arm_pairs(common_methods):
        per_session = {}
        for entry in sessions:
            # compare_arms(first, second) gives first mean - second mean: later - earlier, so a radar
            # gain is positive for higher-is-better metrics.
            per_session[entry["session"]] = compare_arms(entry["arms"][later_method]["metrics"],
                                                         entry["arms"][earlier_method]["metrics"])
        # Only metrics scored in every session can be differenced.
        metric_keys = sorted(set.intersection(*(set(gaps) for gaps in per_session.values())))
        metrics: Dict[str, Dict[str, Any]] = {}
        for metric_key in metric_keys:
            session_gaps = {session: gaps[metric_key] for session, gaps in per_session.items()}
            first_gap = session_gaps[sessions[0]["session"]]["gap"]
            last_gap = session_gaps[sessions[-1]["session"]]["gap"]
            # Difference of differences: how much larger the radar gap is in the last session.
            gap_difference = None if first_gap is None or last_gap is None else last_gap - first_gap
            metrics[metric_key] = {"per_session": session_gaps, "gap_difference": gap_difference}
        comparisons.append({"later": later_method, "earlier": earlier_method, "metrics": metrics})
    return comparisons


def format_value(value: Optional[float]) -> str:
    """A metric value for the markdown tables.

    Args:
        value: Number or None (undefined).

    Returns:
        Fixed-point text, or "n/a".
    """
    return "n/a" if value is None else f"{value:.{TABLE_DECIMALS}f}"


def format_description(description: Dict[str, Optional[float]]) -> str:
    """An arm's metric as "mean ± std [min, max] (n)".

    Args:
        description: describe_values output.

    Returns:
        Table cell text.
    """
    return (f"{format_value(description['mean'])} ± {format_value(description['std'])} "
            f"[{format_value(description['min'])}, {format_value(description['max'])}] (n={description['n']})")


def separation_label(separated: Optional[bool]) -> str:
    """Table text for compare_arms' "separated" flag.

    Args:
        separated: True (ranges disjoint), False (overlap) or None (an arm has < 2 defined repeats).

    Returns:
        "sep", "-" or "n/a".
    """
    if separated is None:
        return "n/a"
    return "sep" if separated else "-"


def render_markdown(sessions: List[Dict[str, Any]], comparisons: List[Dict[str, Any]]) -> str:
    """The comparison as markdown: per-session arm tables, then per method pair the gaps side by side.

    Args:
        sessions: As passed to compare_sessions.
        comparisons: compare_sessions output.

    Returns:
        Markdown text.
    """
    session_names = [entry["session"] for entry in sessions]
    lines = [f"# Radar arms across sessions: {' vs '.join(session_names)}", "",
             "Descriptive only: repeats share one seed and are not paired. Gap = radar variant - baseline "
             "(gain positive for higher-is-better metrics). `sep` = the two arms' [min, max] do not overlap; `n/a` = an arm has fewer than 2 repeats (no spread). "
             f"Δ = gap in {session_names[-1]} - gap in {session_names[0]}. "
             f"P1cuboid = scripts/13 {P1_POINT_SET} cuboid detection rate (T45) and point precision (T46); "
             "P1/P2 = scripts/19 voxel metrics on the rendered lidar (P1) and the Gaussians (P2).", ""]
    # Arm membership, so every number traces to its runs.
    for entry in sessions:
        lines.append(f"## {entry['session']}: arms")
        lines.append("")
        for method in sorted(entry["arms"], key=lambda method_name: (method_rank(method_name), method_name)):
            arm = entry["arms"][method]
            lines.append(f"- **{METHOD_LABELS.get(method, method)}** (`{arm['profile']}`, n={len(arm['run_ids'])}): "
                         + ", ".join(arm["run_ids"]))
        lines.append("")
    # Per class: each arm's mean ± std per session.
    for class_name in HEADLINE_CLASSES:
        lines += [f"## {class_name}: arm values", ""]
        header = "| metric | session | " + " | ".join(METHOD_LABELS[method] for method in METHOD_ORDER) + " |"
        lines += [header, "|" + "---|" * (len(METHOD_ORDER) + 2)]
        for metric_suffix in HEADLINE_METRICS:
            metric_key = f"{class_name}/{metric_suffix}"
            for entry in sessions:
                cells = [format_description(entry["arms"][method]["metrics"][metric_key])
                         if method in entry["arms"] and metric_key in entry["arms"][method]["metrics"] else "n/a"
                         for method in METHOD_ORDER]
                lines.append(f"| {metric_suffix} | {entry['session']} | " + " | ".join(cells) + " |")
        lines.append("")
    # Per method pair: the gaps of both sessions side by side, and their difference.
    for comparison in comparisons:
        later_label = METHOD_LABELS.get(comparison["later"], comparison["later"])
        earlier_label = METHOD_LABELS.get(comparison["earlier"], comparison["earlier"])
        lines += [f"## Gap: {later_label} - {earlier_label}", ""]
        header = ("| metric | " + " | ".join(f"{session} gap (sep)" for session in session_names) + " | Δ |")
        lines += [header, "|" + "---|" * (len(session_names) + 2)]
        for class_name in HEADLINE_CLASSES:
            for metric_suffix in HEADLINE_METRICS:
                metric_key = f"{class_name}/{metric_suffix}"
                if metric_key not in comparison["metrics"]:
                    continue
                metric_entry = comparison["metrics"][metric_key]
                session_cells = [f"{format_value(metric_entry['per_session'][session]['gap'])} "
                                 f"({separation_label(metric_entry['per_session'][session]['separated'])})"
                                 for session in session_names]
                lines.append(f"| {metric_key} | " + " | ".join(session_cells)
                             + f" | {format_value(metric_entry['gap_difference'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Load the session summaries, compare the arms across sessions, write JSON and markdown.

    Returns:
        Exit code: 0 written, 1 refused input.
    """
    arguments = build_argument_parser().parse_args()
    project_root = find_project_root()
    # Step 1: load every summary and its arms; refusals stop before anything is written.
    sessions: List[Dict[str, Any]] = []
    try:
        for summary_path in arguments.summaries:
            summary = json.loads(summary_path.read_text())
            if summary.get("schema") != SUMMARY_SCHEMA:
                raise ValueError(f"{summary_path}: schema {summary.get('schema')!r}, need {SUMMARY_SCHEMA!r}")
            sessions.append({"session": summary["session"], "summary_path": str(summary_path.resolve()),
                             "arms": session_arms(summary, project_root)})
        session_names = [entry["session"] for entry in sessions]
        if len(session_names) < MINIMUM_SESSION_COUNT or len(set(session_names)) != len(session_names):
            raise ValueError(f"need at least {MINIMUM_SESSION_COUNT} distinct sessions, got {session_names}")
        # Step 2: gaps per session and their differences (compare_arms raises KeyError, hence a refusal,
        # when the earlier arm of a pair lacks a metric the later arm has; the reverse is dropped silently).
        comparisons = compare_sessions(sessions)
    except (OSError, ValueError, KeyError) as error:
        print(f"refused: {error!r}", file=sys.stderr)
        return 1
    # Step 3: write JSON (all metrics) and markdown (headline metrics).
    output_dir = arguments.output_dir or project_root / DEFAULT_OUTPUT_RELATIVE_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema": REPORT_SCHEMA, "p1_point_set": P1_POINT_SET,
              "sessions": [{"session": entry["session"], "summary": entry["summary_path"],
                            "arms": entry["arms"]} for entry in sessions],
              "comparisons": comparisons}
    (output_dir / OUTPUT_JSON_NAME).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / OUTPUT_MARKDOWN_NAME).write_text(render_markdown(sessions, comparisons))
    print(f"wrote {output_dir / OUTPUT_JSON_NAME} and {output_dir / OUTPUT_MARKDOWN_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
