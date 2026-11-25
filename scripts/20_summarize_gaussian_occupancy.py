"""Compare the arms of one session over their repeats: P1 and P2 from scripts/19 (PLAN §5).

Runs are grouped into arms by the Slurm profile they were submitted with (``runs/submitted.tsv``);
every run of a profile is a repeat of that arm. Before anything is pooled:
- every run must be of ``--session`` and its split must equal the committed reference record
  ``configs/reference_splits/<session>.json`` (CLAUDE.md rule 12; ``--allow-split-change`` lets a
  differing run through with a recorded reason);
- every run's per-class P1 and P2 voxel metrics and its hidden block (T94/T95) are recomputed from its
  cells archive and must equal its report exactly (present-but-hidden, detection and threshold-sweep
  values are not re-checked);
- all runs must share the same scored cells and the same hidden-space build, so the per-cell
  consistency counts are meaningful.

Output (``reports/gaussian_occupancy/<session>/`` by default): ``gaussian_occupancy_summary.{json,md}`` with,
per arm, n / mean / sample std / min / max of every P1, P2 and hidden-voxel metric and of the run quality
(test-view image metrics, lidar render metrics), and for every arm pair the gap between means beside
both arms' spreads, plus the GT cells recovered in every repeat of one arm and in no repeat of the
other, per evidence stratum. Reading rule: a variant gap counts only when it exceeds the spread
between repeats without losing precision_H. Descriptive only (eval/p2_repeats.py).
The markdown opens with the plain-language definitions and the whole-scene section
(``eval/report_opening.py``), which need scripts/26's ``figures_manifest.json`` of the session; without it
the script refuses.

Usage:
    uv run python scripts/20_summarize_gaussian_occupancy.py --session june_tall_grass runs/<run_id> ...
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from viking_splatad.config import find_project_root
from viking_splatad.dataset.reference_split import compare_with_reference
from viking_splatad.dataset.run_split import SplitFrame
from viking_splatad.eval.evaluation_terminology import MARGIN_CLASS_NAMES
from viking_splatad.eval.hidden_geometry import occupied_cell_metrics
from viking_splatad.eval.hidden_metrics import OBJECT_ARRAY_KEYS, hidden_class_report, threshold_key
from viking_splatad.eval.p2_cells import HIDDEN_KEY, HIDDEN_NEAR_KEY
from viking_splatad.eval.p2_repeats import (
    arm_pairs,
    compare_arms,
    consistent_cell_counts,
    describe_arm,
    flatten_hidden_metrics,
    flatten_quality_metrics,
    flatten_run_metrics,
)
from viking_splatad.eval.run_quality import (
    IMAGE_METRIC_NAMES,
    IMAGE_RESULT_KEY,
    LIDAR_RESULT_KEY,
    REPORTED_LIDAR_METRICS,
    STEP_KEY,
    run_quality,
)
from viking_splatad.eval.report_opening import FIGURES_RELATIVE_DIR, load_figure_manifest, whole_scene_opening_markdown
from viking_splatad.eval.run_context import file_sha256

REPORT_SCHEMA = "viking-splatad/gaussian-occupancy-summary/2"
RUN_REPORT_SCHEMA = "viking-splatad/gaussian-occupancy/2"
RUN_REPORT_RELATIVE_PATH = Path("eval") / "gaussian_occupancy.json"
RUN_CELLS_RELATIVE_PATH = Path("eval") / "gaussian_occupancy_cells.npz"
SPLIT_JSON_NAME = "split.json"
REFERENCE_SPLIT_RELATIVE_DIR = Path("configs") / "reference_splits"
# Outside runs/: submit_run.py clean treats every folder in runs/ as a run.
DEFAULT_OUTPUT_RELATIVE_DIR = Path("reports") / "gaussian_occupancy"
SUMMARY_JSON_NAME = "gaussian_occupancy_summary.json"
SUMMARY_MARKDOWN_NAME = "gaussian_occupancy_summary.md"
# Abbreviated commit length in tables (git's default short-SHA length).
SHORT_SHA_LENGTH = 7
# Split differences quoted in a refusal; the full list comes from scripts/16_check_reference_split.py.
QUOTED_DIFFERENCE_COUNT = 3
# Report fields that must be identical across pooled runs: one GT, one estimator and the same grids.
SHARED_REPORT_FIELDS = ("ground_truth", "settings", "split_membership_sha256", "hidden_space")
# Cells-archive arrays that depend on the model; every other array (cells, GT masks, evidence) must match.
PER_MODEL_CELL_ARRAYS = ("occupancy", "p1_hit")
# Arm order in tables: stock first, then the radar variants as they build on each other (CLAUDE.md
# scope); unknown methods follow. Gaps read "later arm - earlier arm" so a radar gain is positive.
METHOD_ORDER = ("splatad", "splatad-radar-init", "splatad-radar-init-lidar")
# Headline metrics of the markdown tables, in reading order (the JSON holds all of them). ``{threshold}``
# is a P2 threshold key. 1) H_near (hidden within the near band of the path, ``_Hnear``): P2 recall,
# precision, IoU, P1 recall_Hnear, then the context counts (hidden GT cells, objects having one);
# 2) full H (``_H``): P2 recall, precision, IoU and its context counts; 3) baseline: P2 recall,
# precision, IoU over the whole neighbourhood; 4) lidar view: P1 recall, precision, recall_H.
# P1/P2 recall_blocked is retired from the headline (still in the JSON).
HEADLINE_METRIC_SUFFIXES = ("P2@{threshold}/recall_Hnear", "P2@{threshold}/precision_Hnear", "P2@{threshold}/iou_Hnear",
                            "P1/recall_Hnear", "hidden_near/gt_cells", "hidden_near/objects_with_hidden_gt",
                            "P2@{threshold}/recall_H", "P2@{threshold}/precision_H", "P2@{threshold}/iou_H",
                            "hidden/gt_cells", "hidden/objects_with_hidden_gt",
                            "P2@{threshold}/recall", "P2@{threshold}/precision", "P2@{threshold}/iou",
                            "P1/recall", "P1/precision", "P1/recall_H")
# Context suffixes start with this segment and have no pooled / mean variants.
CONTEXT_SEGMENT_PREFIXES = ("hidden/", "hidden_near/")
# Variants of every other headline metric: pooled over cells, then averaged over objects.
HEADLINE_VARIANTS = ("pooled", "mean")
# Section titles of the quality tables.
GUARD_TABLE_TITLE = "Guard: image quality (test + guard frames)"
LIDAR_TABLE_TITLE = "Report only: lidar render"
QUALITY_KEY_PREFIX = "quality"


def report_class_order(classes: Dict[str, Dict[str, Any]]) -> List[str]:
    """Report order of a run's classes: groups first (the headline), then single classes.

    scripts/19 writes its JSON with sorted keys, so the order of ``classes`` there is alphabetical.
    The settled order is restored from the report itself: entries with ``class_id`` null are groups
    (e.g. ``obstacles``) and come first by name; single classes follow in the order their id appears in
    the groups' ``class_ids`` (obstacles = [1, 3, 4] gives tree_trunk, rock, bush_or_small_tree), and
    any class outside every group comes last by name.

    Args:
        classes: ``report["classes"]`` of a scripts/19 report.

    Returns:
        Class names in report order.
    """
    # Step 1: groups, by name.
    group_names = sorted(name for name, entry in classes.items() if entry.get("class_id") is None)
    # Step 2: id order taken from the groups' member lists, first occurrence wins.
    member_id_order: List[int] = []
    for group_name in group_names:
        for class_id in classes[group_name].get("class_ids", []):
            if class_id not in member_id_order:
                member_id_order.append(class_id)
    # Step 3: single classes by that order; ids outside every group sort after it, by name.
    single_names = [name for name in classes if name not in group_names]
    unknown_position = len(member_id_order)
    single_names.sort(key=lambda name: (member_id_order.index(classes[name]["class_id"])
                                        if classes[name]["class_id"] in member_id_order else unknown_position, name))
    return group_names + single_names


def headline_keys(class_name: str, threshold_text: str) -> List[str]:
    """Flat metric keys of one class's headline table at one P2 threshold, in reading order.

    Args:
        class_name: Class or group name.
        threshold_text: A P2 threshold key (``threshold_key``).

    Returns:
        Keys of ``flatten_hidden_metrics``: pooled then mean for each metric, the context keys once.
    """
    keys: List[str] = []
    for suffix_template in HEADLINE_METRIC_SUFFIXES:
        suffix = suffix_template.format(threshold=threshold_text)
        if suffix.startswith(CONTEXT_SEGMENT_PREFIXES):
            keys.append(f"{class_name}/{suffix}")
        else:
            keys += [f"{class_name}/{suffix}/{variant}" for variant in HEADLINE_VARIANTS]
    return keys


def quality_keys(result_key: str, metric_names: Sequence[str]) -> List[str]:
    """Flat keys of one quality group.

    Args:
        result_key: ``IMAGE_RESULT_KEY`` or ``LIDAR_RESULT_KEY``.
        metric_names: The group's metric names.

    Returns:
        ``quality/<result_key>/<name>`` keys.
    """
    return [f"{QUALITY_KEY_PREFIX}/{result_key}/{metric_name}" for metric_name in metric_names]


def build_argument_parser() -> argparse.ArgumentParser:
    """CLI arguments.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="session every run must belong to")
    parser.add_argument("run_dirs", nargs="+", type=Path, help="run folders already scored by scripts/19")
    parser.add_argument("--allow-split-change", default=None, metavar="REASON",
                        help="pool runs whose split differs from the reference record (recorded with this reason)")
    parser.add_argument("--output-dir", type=Path, default=None, help="default: reports/gaussian_occupancy/<session>")
    return parser


def split_differences(run_dir: Path, record: Dict[str, Any]) -> List[str]:
    """Differences between a run's own split (``split.json``) and the session's reference record.

    Args:
        run_dir: Run folder.
        record: Reference record (configs/reference_splits/<session>.json).

    Returns:
        Difference lines; empty when identical.
    """
    run_split = json.loads((run_dir / SPLIT_JSON_NAME).read_text())
    frames = [SplitFrame(**frame_row) for frame_row in run_split["frames"]]
    return compare_with_reference(frames, record)


def verify_cells_reproduce_report(report: Dict[str, Any], cells: Dict[str, np.ndarray], run_dir: Path) -> None:
    """Recompute every class's P1, P2 and hidden-block metrics from the archive; they must equal the report.

    Args:
        report: scripts/19 report.
        cells: Its cells archive.
        run_dir: Run folder (for messages).

    Raises:
        ValueError: On any mismatch (stale or edited archive or report).
    """
    occupied = cells["occupancy"] >= float(cells["threshold"])
    objects = {key: cells[key] for key in OBJECT_ARRAY_KEYS}
    p2_thresholds = [float(threshold) for threshold in cells["p2_thresholds"]]
    for class_name, class_report in report["classes"].items():
        class_cells = cells[f"neighbourhood_{class_name}"]
        truth, evidence = cells[f"truth_{class_name}"][class_cells], cells["evidence"][class_cells]
        # JSON round trip so the recomputed dict compares with the parsed report value for value.
        for test_name, mask in (("P1", cells["p1_hit"]), ("P2", occupied)):
            recomputed = json.loads(json.dumps(occupied_cell_metrics(mask[class_cells], truth, evidence)))
            if recomputed != class_report[test_name]:
                raise ValueError(f"{run_dir}: {class_name} {test_name} metrics do not recompute from the cells archive")
        # The hidden block uses the full-length masks (hidden flags and object lists index all scored cells).
        # Both bands: H_near (headline) and full H, each from its own archive mask.
        for hidden_key in (HIDDEN_NEAR_KEY, HIDDEN_KEY):
            if hidden_key not in cells or hidden_key not in class_report:
                raise ValueError(f"{run_dir}: no '{hidden_key}' in the cells archive or report (scored before the near band; rerun scripts/19)")
            recomputed_hidden = json.loads(json.dumps(hidden_class_report(
                class_report["class_ids"], class_cells, cells[f"truth_{class_name}"], cells[hidden_key], cells["p1_hit"],
                cells["occupancy"], p2_thresholds, objects)))
            if recomputed_hidden != class_report[hidden_key]:
                raise ValueError(f"{run_dir}: {class_name} {hidden_key} metrics do not recompute from the cells archive")


def load_scored_run(run_dir: Path, session: str, record: Optional[Dict[str, Any]],
                    allow_split_change: Optional[str]) -> Dict[str, Any]:
    """Load, check and flatten one scored run.

    Args:
        run_dir: Run folder.
        session: Required session.
        record: Reference split record, or None if the session has none.
        allow_split_change: Reason to accept a differing split, or None.

    Returns:
        {"run_dir", "report", "cells", "metrics", "image_step", "split_note"}.

    Raises:
        ValueError: Wrong session or schema, split differs without permission, archive mismatch.
    """
    report = json.loads((run_dir / RUN_REPORT_RELATIVE_PATH).read_text())
    if report.get("schema") != RUN_REPORT_SCHEMA:
        raise ValueError(f"{run_dir}: report schema {report.get('schema')!r}, need {RUN_REPORT_SCHEMA!r} (rerun scripts/19)")
    if report["session"] != session:
        raise ValueError(f"{run_dir}: session {report['session']!r}, not {session!r}")
    # Step 1: rule 12, the run trained and tested on the reference frames.
    # Without a record nothing proves the split; that needs the same explicit permission as a differing split.
    differences = ["no reference record"] if record is None else split_differences(run_dir, record)
    if differences and allow_split_change is None:
        raise ValueError(f"{run_dir}: split not shown equal to {REFERENCE_SPLIT_RELATIVE_DIR}/{session}.json: "
                         f"{differences[:QUOTED_DIFFERENCE_COUNT]}")
    split_note = "identical to the reference record"
    if differences:
        split_note = f"NOT VERIFIED ({len(differences)} differences), allowed: {allow_split_change}"
    # Step 2: the archive is the one the report was written with, and reproduces its metrics.
    cells_path = run_dir / RUN_CELLS_RELATIVE_PATH
    if file_sha256(cells_path) != report["cells_sha256"]:
        raise ValueError(f"{run_dir}: cells archive is not the one its report was written with")
    with np.load(cells_path, allow_pickle=False) as archive:
        cells = {key: archive[key] for key in archive.files}
    verify_cells_reproduce_report(report, cells, run_dir)
    # Step 3: every scalar metric of the run: P1/P2, hidden block, and the run's own image/lidar quality.
    quality = run_quality(run_dir)
    metrics = {**flatten_run_metrics(report), **flatten_hidden_metrics(report), **flatten_quality_metrics(quality)}
    return {"run_dir": run_dir, "report": report, "cells": cells, "metrics": metrics, "image_step": quality[IMAGE_RESULT_KEY][STEP_KEY], "split_note": split_note}


def arm_sort_key(profile: str, runs: List[Dict[str, Any]]) -> tuple:
    """Order arms by method (stock, radar init, init + loss, others), then profile name.

    Args:
        profile: Arm name.
        runs: Its runs (one method, checked by ``summarize``).

    Returns:
        Sort key.
    """
    method = runs[0]["report"]["train_method"]
    method_rank = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
    return (method_rank, profile)


def run_provenance(report: Dict[str, Any]) -> Dict[str, Any]:
    """The evaluation settings a run report records, which the definitions text quotes.

    Args:
        report: scripts/19 run report.

    Returns:
        {"voxel_size_m", "near_band_m", "max_range_m", "vertical_half_fov_rad", "margin_m_by_class"}.

    Raises:
        KeyError: The report lacks a field (rerun scripts/19).
    """
    return {"voxel_size_m": report["ground_truth"]["voxel_size_m"],
            "near_band_m": report["hidden_space"]["near_band_m"],
            "max_range_m": report["settings"]["max_range_m"],
            "vertical_half_fov_rad": report["settings"]["vertical_half_fov_rad"],
            "margin_m_by_class": {class_name: report["classes"][class_name]["margin_m"]
                                  for class_name in MARGIN_CLASS_NAMES}}


def summarize(scored_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Per-arm descriptions, pairwise comparisons and consistent-cell counts.

    Args:
        scored_runs: ``load_scored_run`` outputs of one session.

    Returns:
        JSON-ready summary.

    Raises:
        ValueError: If runs of one arm differ in method, or the runs do not share their scored cells.
    """
    # Step 1: arms = profiles; one profile must be one method.
    arms: Dict[str, List[Dict[str, Any]]] = {}
    for scored_run in scored_runs:
        arms.setdefault(scored_run["report"]["profile"], []).append(scored_run)
    for profile, runs in arms.items():
        if len({run["report"]["train_method"] for run in runs}) != 1:
            raise ValueError(f"profile {profile} mixes training methods")
    arm_names = sorted(arms, key=lambda profile: arm_sort_key(profile, arms[profile]))
    # Step 2: one scoring domain. Each run once; same GT, estimator, grids and split; same cells, GT
    # masks and recorded evidence (these depend only on the split's test sweeps and the GT).
    run_ids = [run["report"]["run_id"] for run in scored_runs]
    if len(set(run_ids)) != len(run_ids):
        raise ValueError(f"a run is listed twice: {sorted({run_id for run_id in run_ids if run_ids.count(run_id) > 1})}")
    reference_run = scored_runs[0]
    reference_cells = reference_run["cells"]
    for scored_run in scored_runs[1:]:
        for field_name in SHARED_REPORT_FIELDS:
            if scored_run["report"][field_name] != reference_run["report"][field_name]:
                raise ValueError(f"{scored_run['run_dir']}: {field_name} differs from {reference_run['run_dir']} (rerun scripts/19)")
        model_independent = (set(reference_cells) | set(scored_run["cells"])) - set(PER_MODEL_CELL_ARRAYS)
        for array_name in sorted(model_independent):
            if array_name not in scored_run["cells"] or array_name not in reference_cells or not np.array_equal(
                    scored_run["cells"][array_name], reference_cells[array_name]):
                raise ValueError(f"{scored_run['run_dir']}: cells archive {array_name!r} differs from {reference_run['run_dir']}")
    # The range margin is not inside SHARED_REPORT_FIELDS (it sits in each class entry), so check it here:
    # the definitions text quotes one margin per class for the whole summary.
    provenance = run_provenance(reference_run["report"])
    for scored_run in scored_runs[1:]:
        if run_provenance(scored_run["report"]) != provenance:
            raise ValueError(f"{scored_run['run_dir']}: provenance (cell edge, band, range, field of view or class "
                             f"margins) differs from {reference_run['run_dir']} (rerun scripts/19)")
    descriptions = {name: describe_arm([run["metrics"] for run in arms[name]]) for name in arm_names}
    class_names = report_class_order(scored_runs[0]["report"]["classes"])
    threshold_texts = [threshold_key(threshold) for threshold in scored_runs[0]["report"]["settings"]["p2_thresholds"]]
    # Step 3: every pair, later arm first so "gap" = later - earlier (radar minus stock reads positive for a gain).
    comparisons = []
    for earlier_arm, later_arm in arm_pairs(arm_names):
        consistent = {}
        for class_name in class_names:
            class_cells = reference_cells[f"neighbourhood_{class_name}"]
            occupied_by_arm = {name: [(run["cells"]["occupancy"] >= float(run["cells"]["threshold"]))[class_cells]
                                      for run in arms[name]] for name in (later_arm, earlier_arm)}
            consistent[class_name] = consistent_cell_counts(
                occupied_by_arm[later_arm], occupied_by_arm[earlier_arm],
                reference_cells[f"truth_{class_name}"][class_cells], reference_cells["evidence"][class_cells])
        comparisons.append({"first": later_arm, "second": earlier_arm,
                            "metrics": compare_arms(descriptions[later_arm], descriptions[earlier_arm]),
                            "consistent_gt_cells": consistent})
    return {
        "schema": REPORT_SCHEMA,
        "session": scored_runs[0]["report"]["session"],
        # Report order (obstacles first) and the P2 threshold keys, so the markdown needs no key parsing.
        "class_names": class_names,
        "p2_thresholds": threshold_texts,
        "arms": {name: {"train_method": arms[name][0]["report"]["train_method"],
                        "runs": [{"run_id": run["report"]["run_id"], "git_sha": run["report"]["git_sha"],
                                  "split": run["split_note"], "cells_sha256": run["report"]["cells_sha256"],
                                  "image_step": run["image_step"]}
                                 for run in arms[name]],
                        "metrics": descriptions[name]} for name in arm_names},
        "comparisons": comparisons,
        "settings": scored_runs[0]["report"]["settings"],
        # Additive (schema stays /2): the values the report's definitions quote.
        "provenance": provenance,
    }


def format_description(description: Dict[str, Optional[float]]) -> str:
    """``mean ± std (n=repeats)``, or "n/a (n=0)" when no repeat defines the metric.

    Args:
        description: ``describe_values`` output.

    Returns:
        Text.
    """
    if not description["n"]:
        return "n/a (n=0)"
    std_text = "" if description["std"] is None else f" ± {description['std']:.3f}"
    return f"{description['mean']:.3f}{std_text} (n={description['n']})"


def format_optional(value: Optional[float]) -> str:
    """Three-decimal text, or "n/a" for None.

    Args:
        value: Number or None.

    Returns:
        Text.
    """
    return "n/a" if value is None else f"{value:.3f}"


def arm_table_lines(summary: Dict[str, Any], metric_keys: Sequence[str], row_label_prefix_length: int = 0) -> List[str]:
    """A table with one row per metric key and one ``mean ± std (n)`` column per arm.

    Args:
        summary: ``summarize`` output.
        metric_keys: Flat keys to show, in order.
        row_label_prefix_length: Characters dropped from each key for its row label (the class prefix).

    Returns:
        Markdown lines (header, rule, rows, trailing blank).
    """
    arm_names = list(summary["arms"])
    lines = ["| metric | " + " | ".join(arm_names) + " |", "|---|" + "---|" * len(arm_names)]
    for metric_key in metric_keys:
        lines.append(f"| {metric_key[row_label_prefix_length:]} | " + " | ".join(
            format_description(summary["arms"][name]["metrics"][metric_key]) for name in arm_names) + " |")
    return lines + [""]


def comparison_table_lines(comparison: Dict[str, Any], metric_keys: Sequence[str]) -> List[str]:
    """Gap beside the spread of both arms for each metric key of one arm pair.

    Args:
        comparison: One ``summarize`` comparison.
        metric_keys: Flat keys to show, in order.

    Returns:
        Markdown lines (header, rule, rows).
    """
    lines = ["| metric | gap | first std | second std | first range | second range | separated |", "|---|---|---|---|---|---|---|"]
    for metric_key in metric_keys:
        row = comparison["metrics"][metric_key]
        values = [format_optional(row[name]) for name in ("gap", "first_std", "second_std", "first_range", "second_range")]
        lines.append(f"| {metric_key} | {' | '.join(values)} | {row['separated']} |")
    return lines


def summary_markdown(summary: Dict[str, Any]) -> str:
    """Readable tables: per class and threshold the headline metrics per arm, the pair gaps, then quality.

    Args:
        summary: ``summarize`` output.

    Returns:
        Markdown.
    """
    arm_names = list(summary["arms"])
    class_names, threshold_texts = summary["class_names"], summary["p2_thresholds"]
    lines = [f"# P1/P2 over repeats: {summary['session']}", "",
             "Descriptive only: a variant gap counts only when it exceeds the spread between repeats without "
             "losing precision_H. Repeats share one seed, so they are not paired. Cells are mean ± sample std (n = repeats).", ""]
    # Step 1: runs per arm with their code and split status.
    for name in arm_names:
        arm = summary["arms"][name]
        lines.append(f"- **{name}** (`{arm['train_method']}`): " + ", ".join(
            f"{run['run_id']} [{run['git_sha'][:SHORT_SHA_LENGTH]}; split {run['split']}]" for run in arm["runs"]))
    # Step 2: per class (obstacles first) and P2 threshold (lowest first), headline metrics by arm.
    for class_name in class_names:
        for threshold_text in threshold_texts:
            lines += ["", f"## {class_name}, P2 occupied at >= {threshold_text}", ""]
            lines += arm_table_lines(summary, headline_keys(class_name, threshold_text), len(class_name) + 1)
    # Step 3: pair gaps on the same keys, with the separation flag.
    lines += ["", "## Arm pairs (gap = first - second; separated = the two arms' [min, max] do not overlap)", ""]
    for comparison in summary["comparisons"]:
        lines += [f"### {comparison['first']} - {comparison['second']}", ""]
        for class_name in class_names:
            for threshold_text in threshold_texts:
                lines += [f"#### {class_name}, P2 >= {threshold_text}", ""]
                lines += comparison_table_lines(comparison, headline_keys(class_name, threshold_text)) + [""]
        lines += ["GT cells occupied in every repeat of one arm and none of the other: "
                  + json.dumps(comparison["consistent_gt_cells"], sort_keys=True), ""]
    # Step 4: run quality, guard first, then the report-only lidar render metrics.
    lines += [f"## {GUARD_TABLE_TITLE}", ""]
    lines += arm_table_lines(summary, quality_keys(IMAGE_RESULT_KEY, IMAGE_METRIC_NAMES), len(QUALITY_KEY_PREFIX) + 1)
    lines += [f"## {LIDAR_TABLE_TITLE}", ""]
    lines += arm_table_lines(summary, quality_keys(LIDAR_RESULT_KEY, REPORTED_LIDAR_METRICS), len(QUALITY_KEY_PREFIX) + 1)
    return "\n".join(lines) + "\n"


def main(argument_list: Optional[List[str]] = None) -> int:
    """Check, pool and compare the scored runs of one session.

    Args:
        argument_list: CLI arguments (default: sys.argv).

    Returns:
        Exit code: 0 written, 1 refused input.
    """
    arguments = build_argument_parser().parse_args(argument_list)
    project_root = find_project_root()
    # Step 1: the session's reference split record (rule 12), if it has one.
    record_path = project_root / REFERENCE_SPLIT_RELATIVE_DIR / f"{arguments.session}.json"
    # Step 2: check and load every run, then pool; any refused run aborts before anything is written.
    try:
        record = json.loads(record_path.read_text()) if record_path.is_file() else None
        scored_runs = [load_scored_run(run_dir, arguments.session, record, arguments.allow_split_change)
                       for run_dir in arguments.run_dirs]
        summary = summarize(scored_runs)
    # Missing or malformed inputs are refusals at this CLI boundary, not tracebacks.
    except (ValueError, KeyError, OSError) as error:
        print(f"refused: {error}", file=sys.stderr)
        return 1
    # Step 3: write the JSON first, so a refusal in the opening never loses the pooled numbers.
    output_dir = arguments.output_dir or project_root / DEFAULT_OUTPUT_RELATIVE_DIR / arguments.session
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_JSON_NAME).write_text(json.dumps(summary, indent=1, sort_keys=True))
    # Step 4: the opening (definitions, whole scene, figures) needs scripts/26's manifest of the session.
    figures_dir = project_root / FIGURES_RELATIVE_DIR / arguments.session
    # Figure links are relative to the folder holding the report.
    figures_relative_dir = Path(os.path.relpath(figures_dir, output_dir)).as_posix()
    try:
        manifest = load_figure_manifest(project_root, arguments.session)
        opening = whole_scene_opening_markdown(summary, manifest, figures_relative_dir)
    except (ValueError, KeyError) as error:
        print(f"refused: {error} (the summary JSON was written to {output_dir / SUMMARY_JSON_NAME})", file=sys.stderr)
        return 1
    # Step 5: the markdown; the opening follows the title line.
    title_line, _, remaining_markdown = summary_markdown(summary).partition("\n")
    markdown = f"{title_line}\n\n{opening}\n{remaining_markdown}"
    (output_dir / SUMMARY_MARKDOWN_NAME).write_text(markdown)
    print(markdown)
    print(f"wrote {output_dir / SUMMARY_JSON_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
