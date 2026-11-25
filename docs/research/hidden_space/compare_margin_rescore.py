"""Before/after table for the bounded range margin (run from the repo root).

Usage:
    uv run python docs/research/hidden_space/compare_margin_rescore.py [--old-commit 9b898f7]

Reads each session's Gaussian-occupancy summary (scripts/20) twice: the committed version at
``--old-commit`` (unbounded T92 margin) via ``git show``, and the working-tree version (margin with
the neighbour reach bound). Prints, per session, the hidden-space size on H_near and, for each arm
pair script 20 compared, the headline P2 metrics on H_near with script 20's own gap and
``separated`` flag. Nothing is recomputed: every number is read from the two summaries.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

# Sessions of the campaign (CLAUDE.md "Current scope").
SESSIONS = ("june_tall_grass", "may_short_grass")
# Summary written by scripts/20 per session, relative to the repo root.
# Repo root: this file lives in docs/research/hidden_space/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SUMMARY_PATH_TEMPLATE = "reports/gaussian_occupancy/{session}/gaussian_occupancy_summary.json"
# Commit carrying the summaries scored with the unbounded margin (session summary 2026-11-24).
DEFAULT_OLD_COMMIT = "9b898f7"
# Headline class group (configs/eval.yaml class_groups); the P2 thresholds are read from each summary.
HEADLINE_GROUP = "obstacles"
# Headline metrics on the near band: recall, precision, IoU, each averaged over objects and pooled.
HEADLINE_METRIC_SUFFIXES = ("recall_Hnear", "precision_Hnear", "iou_Hnear")
AVERAGING_NAMES = ("mean", "pooled")
# Size of the scored set on the near band (identical in every run of a session).
SIZE_METRIC_NAMES = ("hidden_near/gt_cells", "hidden_near/objects_with_hidden_gt")


def load_summary(session: str, commit: Optional[str] = None) -> Dict[str, Any]:
    """Load a scripts/20 summary from the working tree or from a git commit.

    Args:
        session: Session name.
        commit: Git revision to read from, or None for the working-tree file.

    Returns:
        The parsed summary JSON.
    """
    relative_path = SUMMARY_PATH_TEMPLATE.format(session=session)
    # Working tree: plain read. Commit: `git show <rev>:<path>` prints the blob as stored.
    if commit is None:
        return json.loads((PROJECT_ROOT / relative_path).read_text())
    blob = subprocess.run(["git", "show", f"{commit}:{relative_path}"], check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)
    return json.loads(blob.stdout)


def headline_metric_keys(p2_thresholds: List[float]) -> List[str]:
    """Flat metric keys of the headline table, e.g. ``obstacles/P2@0.5/recall_Hnear/mean``.

    Args:
        p2_thresholds: P2 thresholds listed in the summary.

    Returns:
        Keys ordered threshold, metric, averaging.
    """
    return [f"{HEADLINE_GROUP}/P2@{threshold}/{suffix}/{averaging}"
            for threshold in p2_thresholds for suffix in HEADLINE_METRIC_SUFFIXES for averaging in AVERAGING_NAMES]


def format_comparison(row: Dict[str, Any]) -> str:
    """One cell: gap (first - second) and the separated flag, as script 20 computed them.

    Args:
        row: One entry of a comparison's ``metrics`` mapping, or None when absent.

    Returns:
        Text such as ``+0.1000 sep`` or ``-0.0200 ovl``; ``n/a`` when the metric is missing.
    """
    if row is None or row.get("gap") is None:
        return "n/a"
    return f"{row['gap']:+.4f} {'sep' if row['separated'] else 'ovl'}"


def format_arm_value(summary: Dict[str, Any], arm: str, key: str) -> str:
    """Mean ± std over an arm's repeats for one metric.

    Args:
        summary: Parsed summary.
        arm: Arm name.
        key: Flat metric key.

    Returns:
        ``mean±std`` with four decimals, or ``n/a``.
    """
    statistics = summary["arms"][arm]["metrics"].get(key)
    if statistics is None or statistics.get("mean") is None:
        return "n/a"
    return f"{statistics['mean']:.4f}±{statistics['std']:.4f}"


def report_session(session: str, old_commit: str) -> None:
    """Print the before/after tables of one session.

    Args:
        session: Session name.
        old_commit: Git revision with the unbounded-margin summary.
    """
    # Step 1: both versions of the summary.
    summaries = {"old": load_summary(session, old_commit), "new": load_summary(session)}
    print(f"\n## {session}\n")
    # Step 2: scored-set size on H_near (the first arm carries it; it is a session property).
    for version, summary in summaries.items():
        first_arm = next(iter(summary["arms"]))
        sizes = {name: summary["arms"][first_arm]["metrics"][f"{HEADLINE_GROUP}/{name}"]["mean"] for name in SIZE_METRIC_NAMES}
        print(f"- {version}: H_near obstacle GT cells {sizes[SIZE_METRIC_NAMES[0]]:.0f}, objects {sizes[SIZE_METRIC_NAMES[1]]:.0f}")
    # Step 3: arm means per metric, old and new.
    keys = headline_metric_keys(summaries["new"]["p2_thresholds"])
    arms = list(summaries["new"]["arms"])
    print("\n| metric | " + " | ".join(f"{arm} old | {arm} new" for arm in arms) + " |")
    print("|---|" + "---|" * (2 * len(arms)))
    for key in keys:
        cells = [format_arm_value(summaries[version], arm, key) for arm in arms for version in ("old", "new")]
        print(f"| {key} | " + " | ".join(cells) + " |")
    # Step 4: script 20's arm-pair gaps and separation, old and new.
    # Old pairs are matched by their (first, second) arm names, never by list position.
    old_by_pair = {(comparison["first"], comparison["second"]): comparison for comparison in summaries["old"]["comparisons"]}
    for new_comparison in summaries["new"]["comparisons"]:
        old_comparison = old_by_pair.get((new_comparison["first"], new_comparison["second"]), {"metrics": {}})
        pair_name = f"{new_comparison['first']} - {new_comparison['second']}"
        print(f"\n### gap {pair_name} (sep = arm ranges do not overlap)\n")
        print("| metric | old | new |\n|---|---|---|")
        for key in keys:
            print(f"| {key} | {format_comparison(old_comparison['metrics'].get(key))} | "
                  f"{format_comparison(new_comparison['metrics'].get(key))} |")


def main() -> None:
    """Parse arguments and print both sessions' tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old-commit", default=DEFAULT_OLD_COMMIT)
    arguments = parser.parse_args()
    for session in SESSIONS:
        report_session(session, arguments.old_commit)


if __name__ == "__main__":
    main()
