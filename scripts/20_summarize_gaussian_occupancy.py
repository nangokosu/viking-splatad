"""Compare the arms of one session over their repeats: P1 and P2 from scripts/19 (PLAN §5).

Runs are grouped into arms by the Slurm profile they were submitted with (``runs/submitted.tsv``);
every run of a profile is a repeat of that arm. Before anything is pooled:
- every run must be of ``--session`` and its split must equal the committed reference record
  ``configs/reference_splits/<session>.json`` (CLAUDE.md rule 12; ``--allow-split-change`` lets a
  differing run through with a recorded reason);
- every run's per-class P1 and P2 voxel metrics are recomputed from its cells archive and must equal
  its report exactly (present-but-hidden, detection and threshold-sweep values are not re-checked);
- all runs must share the same scored cells, so the per-cell consistency counts are meaningful.

Output (``reports/gaussian_occupancy/<session>/`` by default): ``gaussian_occupancy_summary.{json,md}`` with,
per arm, n / mean / sample std / min / max of every P1 and P2 metric, and for every arm pair the gap
between means beside both arms' spreads, plus the GT cells recovered in every repeat of one arm and in
no repeat of the other, per evidence stratum. Descriptive only (eval/p2_repeats.py).

Usage:
    uv run python scripts/20_summarize_gaussian_occupancy.py --session june_tall_grass runs/<run_id> ...
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from viking_splatad.config import find_project_root
from viking_splatad.dataset.reference_split import compare_with_reference
from viking_splatad.dataset.run_split import SplitFrame
from viking_splatad.eval.hidden_geometry import occupied_cell_metrics
from viking_splatad.eval.p2_repeats import (
    arm_pairs,
    compare_arms,
    consistent_cell_counts,
    describe_arm,
    flatten_run_metrics,
)
from viking_splatad.eval.run_context import file_sha256

REPORT_SCHEMA = "viking-splatad/gaussian-occupancy-summary/1"
RUN_REPORT_SCHEMA = "viking-splatad/gaussian-occupancy/1"
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
SHARED_REPORT_FIELDS = ("ground_truth", "settings", "split_membership_sha256")
# Cells-archive arrays that depend on the model; every other array (cells, GT masks, evidence) must match.
PER_MODEL_CELL_ARRAYS = ("occupancy", "p1_hit")
# Arm order in tables: stock first, then the radar variants as they build on each other (CLAUDE.md
# scope); unknown methods follow. Gaps read "later arm - earlier arm" so a radar gain is positive.
METHOD_ORDER = ("splatad", "splatad-radar-init", "splatad-radar-init-lidar")
# Metrics shown in the markdown table (the JSON holds all of them).
HEADLINE_METRIC_SUFFIXES = ("P1/recall", "P2/recall", "P1/recall_blocked", "P2/recall_blocked",
                            "P1/recall_no_return", "P2/recall_no_return", "P1/precision", "P2/precision",
                            "P2/iou", "present_but_hidden/total")


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
    """Recompute every class's P1 and P2 cell metrics from the archive; they must equal the report.

    Args:
        report: scripts/19 report.
        cells: Its cells archive.
        run_dir: Run folder (for messages).

    Raises:
        ValueError: On any mismatch (stale or edited archive or report).
    """
    occupied = cells["occupancy"] >= float(cells["threshold"])
    for class_name, class_report in report["classes"].items():
        class_cells = cells[f"neighbourhood_{class_name}"]
        truth, evidence = cells[f"truth_{class_name}"][class_cells], cells["evidence"][class_cells]
        # JSON round trip so the recomputed dict compares with the parsed report value for value.
        for test_name, mask in (("P1", cells["p1_hit"]), ("P2", occupied)):
            recomputed = json.loads(json.dumps(occupied_cell_metrics(mask[class_cells], truth, evidence)))
            if recomputed != class_report[test_name]:
                raise ValueError(f"{run_dir}: {class_name} {test_name} metrics do not recompute from the cells archive")


def load_scored_run(run_dir: Path, session: str, record: Optional[Dict[str, Any]],
                    allow_split_change: Optional[str]) -> Dict[str, Any]:
    """Load, check and flatten one scored run.

    Args:
        run_dir: Run folder.
        session: Required session.
        record: Reference split record, or None if the session has none.
        allow_split_change: Reason to accept a differing split, or None.

    Returns:
        {"run_dir", "report", "cells", "metrics", "split_note"}.

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
    return {"run_dir": run_dir, "report": report, "cells": cells, "metrics": flatten_run_metrics(report),
            "split_note": split_note}


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
    descriptions = {name: describe_arm([run["metrics"] for run in arms[name]]) for name in arm_names}
    class_names = list(scored_runs[0]["report"]["classes"])
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
        "arms": {name: {"train_method": arms[name][0]["report"]["train_method"],
                        "runs": [{"run_id": run["report"]["run_id"], "git_sha": run["report"]["git_sha"],
                                  "split": run["split_note"], "cells_sha256": run["report"]["cells_sha256"]}
                                 for run in arms[name]],
                        "metrics": descriptions[name]} for name in arm_names},
        "comparisons": comparisons,
        "settings": scored_runs[0]["report"]["settings"],
    }


def format_description(description: Dict[str, Optional[float]]) -> str:
    """``mean ± std [min, max] (n)``, or "n/a".

    Args:
        description: ``describe_values`` output.

    Returns:
        Text.
    """
    if not description["n"]:
        return "n/a"
    std_text = "" if description["std"] is None else f" ± {description['std']:.3f}"
    return f"{description['mean']:.3f}{std_text} [{description['min']:.3f}, {description['max']:.3f}] (n={description['n']})"


def summary_markdown(summary: Dict[str, Any]) -> str:
    """Readable tables of the summary: arms × headline metrics per class, then the pair gaps.

    Args:
        summary: ``summarize`` output.

    Returns:
        Markdown.
    """
    arm_names = list(summary["arms"])
    lines = [f"# P1/P2 over repeats: {summary['session']}", "",
             "Descriptive only: compare each gap with the spread within both arms. Repeats share one seed, so they are not paired.", ""]
    # Step 1: runs per arm with their code and split status.
    for name in arm_names:
        arm = summary["arms"][name]
        lines.append(f"- **{name}** (`{arm['train_method']}`): " + ", ".join(
            f"{run['run_id']} [{run['git_sha'][:SHORT_SHA_LENGTH]}; split {run['split']}]" for run in arm["runs"]))
    class_names = sorted({key.split("/")[0] for key in summary["arms"][arm_names[0]]["metrics"]})
    # Step 2: per class, one row per headline metric, one column per arm.
    for class_name in class_names:
        lines += ["", f"## {class_name}", "", "| metric | " + " | ".join(arm_names) + " |",
                  "|---|" + "---|" * len(arm_names)]
        for suffix in HEADLINE_METRIC_SUFFIXES:
            key = f"{class_name}/{suffix}"
            lines.append(f"| {suffix} | " + " | ".join(format_description(summary["arms"][name]["metrics"][key])
                                                       for name in arm_names) + " |")
    # Step 3: pair gaps on the headline metrics, with the separation flag.
    lines += ["", "## Arm pairs (gap = first - second; separated = the two arms' [min, max] do not overlap)", ""]
    for comparison in summary["comparisons"]:
        lines += [f"### {comparison['first']} - {comparison['second']}", "",
                  "| metric | gap | first range | second range | separated |", "|---|---|---|---|---|"]
        for class_name in class_names:
            for suffix in HEADLINE_METRIC_SUFFIXES:
                row = comparison["metrics"][f"{class_name}/{suffix}"]
                cells = ["n/a" if row[name] is None else f"{row[name]:.3f}" for name in ("gap", "first_range", "second_range")]
                lines.append(f"| {class_name}/{suffix} | {' | '.join(cells)} | {row['separated']} |")
        lines += ["", "GT cells occupied in every repeat of one arm and none of the other: "
                  + json.dumps(comparison["consistent_gt_cells"], sort_keys=True), ""]
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
    # Step 3: write the JSON and markdown summaries.
    output_dir = arguments.output_dir or project_root / DEFAULT_OUTPUT_RELATIVE_DIR / arguments.session
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / SUMMARY_JSON_NAME).write_text(json.dumps(summary, indent=1, sort_keys=True))
    markdown = summary_markdown(summary)
    (output_dir / SUMMARY_MARKDOWN_NAME).write_text(markdown)
    print(markdown)
    print(f"wrote {output_dir / SUMMARY_JSON_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
