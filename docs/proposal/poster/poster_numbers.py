"""Numbers and strings printed on the redesigned poster, read from the committed whole-scene summaries.

Pure module: no pptx, no plotting, no network. The summary helpers (paths, key names, gap and range
separation via ``compare_arms``) are reused from ``docs/research/build_summary_report.py``.
"""

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root: this file lives in docs/proposal/poster/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# The summary helpers are reused, not rewritten.
REPORT_BUILDER_PATH = PROJECT_ROOT / "docs" / "research" / "build_summary_report.py"
# Fraction -> percent factor (metrics are fractions in [0, 1]).
PERCENT_PER_FRACTION = 100.0
# Fewest repeats for which the min/max range is a spread (matches the report's "at least 2 runs" rule).
MINIMUM_RUN_COUNT = 2
# True minus sign (U+2212), as in build_summary_report's gap style.
MINUS_SIGN = "−"
# Session display names on the poster.
SESSION_LABELS = {"june_tall_grass": "June (tall grass)", "may_short_grass": "May (short grass)"}
# Short session names for the one-line findings.
SESSION_SHORT_LABELS = {"june_tall_grass": "June", "may_short_grass": "May"}
# First header cell of the results table.
TABLE_CORNER_LABEL = "Obstacles (%)"
# Wording of the range comparison in findings.
SEPARATED_TEXT = "ranges separated"
OVERLAP_TEXT = "ranges overlap"
UNAVAILABLE_TEXT = "range comparison unavailable"
# Method whose gaps lead the findings, and the one that gets its own recall sentence.
LOSS_METHOD = "splatad-radar-init-lidar"
INIT_METHOD = "splatad-radar-init"


def _load_report_builder():
    """Import ``build_summary_report.py`` by path (it is not a package).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("build_summary_report_for_poster", REPORT_BUILDER_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


_report = _load_report_builder()
SESSIONS = _report.SESSIONS
METHOD_ORDER = _report.METHOD_ORDER
BASELINE_METHOD = _report.BASELINE_METHOD
COMPARED_METHODS = _report.COMPARED_METHODS
METRIC_LABELS = _report.METRIC_LABELS
ARM_LABEL_BY_METHOD = _report.ARM_LABEL_BY_METHOD


@dataclass(frozen=True)
class MetricRange:
    """Mean and min/max over the repeats of one metric, as fractions in [0, 1]."""

    mean: float
    minimum: float
    maximum: float
    run_count: int


@dataclass(frozen=True)
class GapResult:
    """Gap to the stock arm in percentage points and whether the repeat ranges are separated."""

    gap_points: float
    separated: Optional[bool]


def binding_threshold(summary: Dict[str, Any]) -> str:
    """The binding P2 threshold of a summary: the first of its ``p2_thresholds`` (as scripts/20 orders them).

    Args:
        summary: A session summary.

    Returns:
        Threshold key, e.g. ``0.5``.
    """
    return str(summary["p2_thresholds"][0])


def load_session_metrics(summary: Dict[str, Any], threshold_text: Optional[str] = None) -> Dict[str, Dict[str, MetricRange]]:
    """Read recall, precision and IoU of every arm of one session.

    Args:
        summary: A session summary (scripts/20 layout).
        threshold_text: P2 threshold key, e.g. ``0.5``; None = the summary's binding threshold.

    Returns:
        Map method -> metric name -> ``MetricRange``.

    Raises:
        ValueError: A metric is undefined (mean None) or has fewer than ``MINIMUM_RUN_COUNT`` runs.
    """
    threshold_text = threshold_text or binding_threshold(summary)
    metrics_by_method: Dict[str, Dict[str, MetricRange]] = {}
    for method, arm in _report.arm_by_method(summary).items():
        metrics_by_method[method] = {}
        for metric_name, _ in METRIC_LABELS:
            statistics = arm["metrics"][_report.metric_key(threshold_text, metric_name)]
            # An undefined metric cannot be printed on the poster.
            if statistics["mean"] is None:
                raise ValueError(f"metric {metric_name} of method {method} is undefined (mean is None)")
            # A range needs repeats to mean anything.
            if statistics["n"] < MINIMUM_RUN_COUNT:
                raise ValueError(f"metric {metric_name} of method {method} has {statistics['n']} run(s); "
                                 f"need at least {MINIMUM_RUN_COUNT}")
            metrics_by_method[method][metric_name] = MetricRange(
                mean=float(statistics["mean"]), minimum=float(statistics["min"]),
                maximum=float(statistics["max"]), run_count=int(statistics["n"]))
    return metrics_by_method


def format_percent_range(metric: MetricRange, decimals: int = 1) -> str:
    """Mean and [min, max] in percent.

    Args:
        metric: The metric (fractions).
        decimals: Decimal places.

    Returns:
        Text such as ``38.0 [37.6, 38.3]``.
    """
    # percent = fraction * 100
    mean_text, minimum_text, maximum_text = (f"{value * PERCENT_PER_FRACTION:.{decimals}f}"
                                             for value in (metric.mean, metric.minimum, metric.maximum))
    return f"{mean_text} [{minimum_text}, {maximum_text}]"


def arm_gaps(summary: Dict[str, Any], threshold_text: Optional[str] = None) -> Dict[str, Dict[str, GapResult]]:
    """Gap of each compared arm to stock, per metric.

    Args:
        summary: A session summary.
        threshold_text: P2 threshold key; None = the summary's binding threshold.

    Returns:
        Map compared method -> metric name -> ``GapResult`` (gap = arm mean - stock mean, in points;
        separated = [min, max] ranges do not overlap, via ``compare_arms``).

    Raises:
        ValueError: When a gap is undefined (an arm has no defined repeat), since the poster cannot print it.
    """
    threshold_text = threshold_text or binding_threshold(summary)
    arm_by_method = _report.arm_by_method(summary)
    gaps_by_method: Dict[str, Dict[str, GapResult]] = {}
    for method in COMPARED_METHODS:
        gaps_by_method[method] = {}
        for metric_name, _ in METRIC_LABELS:
            key = _report.metric_key(threshold_text, metric_name)
            gap_fraction, separated = _report.gap_and_separation(
                arm_by_method[method]["metrics"][key], arm_by_method[BASELINE_METHOD]["metrics"][key])
            if gap_fraction is None:
                raise ValueError(f"gap of {method} vs {BASELINE_METHOD} on {key} is undefined")
            # Fractions -> percentage points.
            gaps_by_method[method][metric_name] = GapResult(gap_fraction * PERCENT_PER_FRACTION, separated)
    return gaps_by_method


def format_gap(gap: GapResult, decimals: int = 1) -> str:
    """Signed gap in points.

    Args:
        gap: The gap.
        decimals: Decimal places.

    Returns:
        Text such as ``+3.8 pts`` or ``−0.5 pts`` (true minus sign).
    """
    # Round first, then take the sign of the rounded value, so a tiny negative gap never prints as "−0.0".
    rounded_points = round(gap.gap_points, decimals)
    sign = "+" if rounded_points >= 0 else MINUS_SIGN
    return f"{sign}{abs(rounded_points):.{decimals}f} pts"


def _separation_text(gap: GapResult) -> str:
    """Wording of a gap's range comparison.

    Args:
        gap: The gap.

    Returns:
        ``ranges separated``, ``ranges overlap`` or the unavailable text.
    """
    if gap.separated is None:
        return UNAVAILABLE_TEXT
    return SEPARATED_TEXT if gap.separated else OVERLAP_TEXT


def key_findings(summaries: Dict[str, Dict[str, Any]]) -> List[str]:
    """Callout sentences derived only from the gaps and separation flags.

    Args:
        summaries: Session name -> summary.

    Returns:
        Per session: the init + loss recall/precision sentence, then the radar init recall sentence.
    """
    sentences: List[str] = []
    for session in SESSIONS:
        gaps = arm_gaps(summaries[session])
        session_label = SESSION_SHORT_LABELS[session]
        # Sentence 1: init + loss vs stock, recall and precision with their range wording.
        loss_gaps = gaps[LOSS_METHOD]
        sentences.append(
            f"{session_label} · {ARM_LABEL_BY_METHOD[LOSS_METHOD]} vs {ARM_LABEL_BY_METHOD[BASELINE_METHOD]}: "
            f"recall {format_gap(loss_gaps['recall'])} ({_separation_text(loss_gaps['recall'])}), "
            f"precision {format_gap(loss_gaps['precision'])} ({_separation_text(loss_gaps['precision'])})")
        # Sentence 2: radar init vs stock, recall only.
        init_recall = gaps[INIT_METHOD]["recall"]
        sentences.append(
            f"{session_label} · {ARM_LABEL_BY_METHOD[INIT_METHOD]} vs {ARM_LABEL_BY_METHOD[BASELINE_METHOD]}: "
            f"recall {format_gap(init_recall)} ({_separation_text(init_recall)})")
    return sentences


def table_rows(summaries: Dict[str, Dict[str, Any]]) -> List[List[str]]:
    """The poster's results table: a header row, then recall, precision and IoU.

    Args:
        summaries: Session name -> summary.

    Returns:
        Rows of text; columns are the label, then each session's arms in ``METHOD_ORDER``.
    """
    # Step 1: validated metric ranges per session (raises on undefined or single-run metrics).
    metrics_by_session = {session: load_session_metrics(summaries[session]) for session in SESSIONS}
    # Step 2: header, one column per (session, arm); then one "mean [min, max]" row per metric.
    header_row = [TABLE_CORNER_LABEL] + [f"{SESSION_LABELS[session]} · {ARM_LABEL_BY_METHOD[method]}"
                                         for session in SESSIONS for method in METHOD_ORDER]
    rows = [header_row]
    for metric_name, row_label in METRIC_LABELS:
        rows.append([row_label] + [format_percent_range(metrics_by_session[session][method][metric_name])
                                   for session in SESSIONS for method in METHOD_ORDER])
    return rows


def load_all_summaries() -> Dict[str, Dict[str, Any]]:
    """Read both committed session summaries.

    Returns:
        Session name -> summary, for every session in ``SESSIONS``.
    """
    return {session: _report.load_summary(session) for session in SESSIONS}
