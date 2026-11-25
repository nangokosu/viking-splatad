"""Before/after report for the bounded range margin, with the metric definitions.

Usage (from anywhere; paths resolve from the repo root):
    uv run python docs/research/hidden_space/compare_margin_rescore.py [--old-commit 9b898f7] [--output PATH]

Reads each session's Gaussian-occupancy summary (scripts/20) twice: the committed version at
``--old-commit`` (unbounded T92 margin) via ``git show``, and the working-tree version (margin with
the neighbour reach bound). The old range margin comes from the label-transition report committed at
the same revision, the new one from ``data/gt/<session>/hidden_space.json`` (rebuild with script 24).
Writes a Markdown report (default ``reports/gaussian_occupancy/margin_rescore_comparison.md``) and
prints it. Nothing is recomputed: every number is read from those files, and the arm-pair gaps and
``separated`` flags are the ones script 20 computed.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

from viking_splatad.eval.session_hidden_space import NEIGHBOUR_REACH_VOXEL_FRACTION

# Repo root: this file lives in docs/research/hidden_space/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
# Sessions of the campaign (CLAUDE.md "Current scope").
SESSIONS = ("june_tall_grass", "may_short_grass")
# Summary written by scripts/20 per session, relative to the repo root.
SUMMARY_PATH_TEMPLATE = "reports/gaussian_occupancy/{session}/gaussian_occupancy_summary.json"
# Label-transition report (removed after the old commit) that recorded the old range margin.
OLD_MARGIN_PATH_TEMPLATE = "reports/hidden_space/{session}/label_transitions.json"
# Hidden-space provenance written by scripts/24 (gitignored, rebuilt per machine).
NEW_HIDDEN_SPACE_PATH_TEMPLATE = "data/gt/{session}/hidden_space.json"
# Hidden-space arrays written by scripts/24; its voxel_size is the GT grid edge.
NEW_HIDDEN_SPACE_ARRAYS_TEMPLATE = "data/gt/{session}/hidden_space.npz"
# Slurm profiles: an arm is named after its profile, which says whether the camera loss is on.
SLURM_CONFIG_RELATIVE_PATH = "configs/slurm.yaml"
# Commit carrying the summaries scored with the unbounded margin (session summary 2026-11-24).
DEFAULT_OLD_COMMIT = "9b898f7"
# Default report location, next to the summaries it compares.
DEFAULT_OUTPUT_RELATIVE_PATH = "reports/gaussian_occupancy/margin_rescore_comparison.md"
# Headline class group (configs/eval.yaml class_groups); the P2 thresholds are read from the new summary.
HEADLINE_GROUP = "obstacles"
# Headline metrics on the near band: recall, precision, IoU, each averaged over objects and pooled.
HEADLINE_METRIC_SUFFIXES = ("recall_Hnear", "precision_Hnear", "iou_Hnear")
AVERAGING_NAMES = ("mean", "pooled")
# Size of the scored set on the near band (identical in every run of a session).
SIZE_METRIC_NAMES = ("hidden_near/gt_cells", "hidden_near/objects_with_hidden_gt")
# Short arm labels keyed by the arm's SplatAD method (CLAUDE.md "Current scope").
ARM_LABEL_BY_METHOD = {
    "splatad": "stock",
    "splatad-radar-init": "radar init",
    "splatad-radar-init-lidar": "init + loss",
}
# Headline rows (bold in the gap tables): init + loss vs stock, per-object recall.
HEADLINE_PAIR = ("init + loss", "stock")
HEADLINE_METRIC_SUFFIX = "recall_Hnear/mean"
# Quantile (0..1) to percent.
PERCENT_PER_UNIT = 100.0

DEFINITIONS = """\
## How to read this report

### What is being compared
- **Arms.** Three SplatAD variants, each trained {repeat_count} times (repeats) per session:
  - **stock**: `splatad`, no radar. Whether the camera (photometric) loss is on is read from each arm's Slurm profile and listed under each session ("Training").
  - **radar init**: `splatad-radar-init`. Radar anchors are added as extra seed Gaussians at initialisation; no radar loss.
  - **init + loss**: `splatad-radar-init-lidar`. The same seeds, plus a label-free radar opacity loss near the anchors.
- **Sessions.** `june_tall_grass` (tall grass hides the trunks; the test of the radar claim) and `may_short_grass` (short grass; the low-grass reference).
- **Old vs new.** "Old" is the scoring committed at the old commit, with the unbounded range margin. "New" is the same {model_count} trained models re-scored after bounding the margin's neighbour reach. Only the evaluation changed: no model was retrained.

### The grid and the ground truth
- **Cells.** Space is divided into the GT voxel grid (cubes of edge {voxel_edge_m:.2f} m, the `voxel_size` of `data/gt/<session>/`).
- **GT cells (G).** Cells labelled by the hand-annotated cuboids of a class. `obstacles` = tree_trunk + rock + bush_or_small_tree, the objects a robot could hit. GT is used only for evaluation; it never feeds training.
- **Neighbourhood (N).** Each class's cuboids grown horizontally by that class's median half-footprint, with the same height span. Only cells in N are scored, so precision asks "of what the model put near an object, how much is on the object".
- **Other-class mask.** In a class's row, cells on another annotated class's GT count neither for nor against.

### What "occupied" means (P2)
- **P2** reads occupancy straight from the trained Gaussians, not from a rendered sensor, so geometry the model holds behind grass still counts.
- **Occupancy of a cell** (estimator T81): each Gaussian adds its peak opacity to the cell holding its centre, plus its Gaussian tail at nearby cell centres inside its cut-off ellipsoid (T57). Contributions combine by the product rule, occupancy = 1 − ∏(1 − α_k).
- **Occupied cells (O).** Cells with occupancy ≥ the threshold. `P2@0.5` is the binding threshold; `P2@0.8` is a strictness check that can weaken a result but never replace 0.5.

### The hidden space H and the near band H_near
- **Why.** The research question is whether radar helps where the lidar could **not** see. H is that set of cells.
- **How a cell is labelled** (built once per session from all recorded lidar sweeps, `scripts/24_build_hidden_space.py`). Along every recorded beam that crosses a cell, with entry and exit distances and a range margin m:
  - **seen:** a return lands within [entry − m, exit + m];
  - **blocked:** a return lands short of entry − m (something nearer stopped the beam);
  - **passed:** a return lands beyond exit + m (the beam went through).
- **H** = cells never seen, never passed through, and blocked at least once. Beams with no return do not affect H.
- **H_near** = H within {near_band_m:.0f} m horizontal distance of the driven path (`hidden_space.near_band_m`). The headline uses H_near, because most of H lies far from the path, which is distant occlusion rather than grass next to the robot.
- **Range margin m.** The tolerance in the labels above. It is measured from the data: the {margin_quantile_percent:.0f}th percentile of point-to-plane residuals between two lidar sweeps of the same static surfaces (T92). **This is what changed.** Before, the {neighbour_count} points of each plane fit could lie a metre or more apart, so the residual measured surface shape and m came out too large. A large m counts too many cells as seen and makes H too small. Now a residual is kept only when all {neighbour_count} neighbours lie within {reach_fraction:g} of a GT voxel edge ({neighbour_reach_m:.3f} m). m still depends on this chosen bound (it keeps falling as the bound tightens; `docs/research/evaluation_caveats.md` §6).

### The metrics
With O = occupied cells, G = GT cells, N = the neighbourhood and H = H_near:
- **recall_Hnear** = |O ∩ G ∩ H| / |G ∩ H|: of the obstacle cells the lidar could not see, the fraction the model fills. Higher means more hidden structure recovered.
- **precision_Hnear** = |O ∩ G ∩ H| / |O ∩ N ∩ H|: of the cells the model fills near obstacles in the hidden band, the fraction that is really obstacle. Lower means more invented geometry (or real but unannotated objects).
- **iou_Hnear** = |O ∩ G ∩ H| / |(O ∪ G) ∩ N ∩ H|: both errors in one number.
- **pooled** (T94): the ratio over all cells of all objects at once. Large objects dominate it.
- **mean** (T95): the ratio per object k (with its own neighbourhood N_k), averaged over the objects where it is defined (a non-zero denominator). An occupied cell inside several N_k counts against each of them. Every object weighs the same.
- **H_near obstacle GT cells / objects**: |G ∩ H| and the number of objects with at least one such cell. They set how much evidence each ratio rests on.

### Statistics
- **mean±std**: mean and sample standard deviation (n − 1) of the metric over an arm's {repeat_count} repeats.
- **gap** = mean of the first arm − mean of the second, so + means the first arm scores higher.
- **sep** (separated): the two arms' [min, max] ranges over their repeats do not overlap; every repeat of one arm beats every repeat of the other. **ovl**: the ranges overlap. This is a descriptive reading aid, not a significance test. SplatAD runs vary from repeat to repeat, so a gap is only read when it exceeds the spread within the arms.
- **Reading rule** (PLAN §5): the headline is `obstacles`, June, recall_Hnear and precision_Hnear at P2@0.5, pooled and mean side by side. Radar must beat stock on recall **without losing precision**. May is the low-grass reference.
"""


def git_show_json(commit: str, relative_path: str) -> Dict[str, Any]:
    """Parse a JSON file as stored at a git revision.

    Args:
        commit: Git revision.
        relative_path: Path relative to the repo root.

    Returns:
        The parsed JSON.
    """
    blob = subprocess.run(["git", "show", f"{commit}:{relative_path}"], check=True, capture_output=True, text=True,
                          cwd=PROJECT_ROOT)
    return json.loads(blob.stdout)


def load_summary(session: str, commit: Optional[str] = None) -> Dict[str, Any]:
    """Load a scripts/20 summary from the working tree or from a git commit.

    Args:
        session: Session name.
        commit: Git revision to read from, or None for the working-tree file.

    Returns:
        The parsed summary JSON.
    """
    relative_path = SUMMARY_PATH_TEMPLATE.format(session=session)
    if commit is None:
        return json.loads((PROJECT_ROOT / relative_path).read_text())
    return git_show_json(commit, relative_path)


def load_margins(session: str, old_commit: str) -> Dict[str, float]:
    """Old and new range margin and the new neighbour reach of one session.

    Args:
        session: Session name.
        old_commit: Git revision with the old label-transition report.

    Returns:
        {"old_margin_m", "new_margin_m", "neighbour_reach_m", "residual_count"}.
    """
    old_report = git_show_json(old_commit, OLD_MARGIN_PATH_TEMPLATE.format(session=session))
    new_provenance = json.loads((PROJECT_ROOT / NEW_HIDDEN_SPACE_PATH_TEMPLATE.format(session=session)).read_text())
    return {"old_margin_m": old_report["hidden_space"]["range_margin_m"],
            "new_margin_m": new_provenance["range_margin"]["margin_m"],
            "neighbour_reach_m": new_provenance["range_margin"]["neighbour_reach_m"],
            "residual_count": new_provenance["range_margin"]["residual_count"]}


def definition_values() -> Dict[str, Any]:
    """Values quoted in the definitions, read from the data and the summaries rather than typed in.

    The hidden-space settings come from one configs/eval.yaml, so the first session's provenance
    supplies them. The repeat count must be the same for every arm of every session, because the
    definitions quote one number; a different count raises instead of being misreported.

    Returns:
        {"voxel_edge_m", "near_band_m", "margin_quantile_percent", "neighbour_count",
        "neighbour_reach_m", "reach_fraction", "repeat_count", "model_count"}.

    Raises:
        ValueError: If the arms do not all have the same number of runs.
    """
    # Step 1: hidden-space settings, reach and GT voxel edge of the first session.
    first_session = SESSIONS[0]
    provenance = json.loads((PROJECT_ROOT / NEW_HIDDEN_SPACE_PATH_TEMPLATE.format(session=first_session)).read_text())
    settings = provenance["hidden_space_settings"]
    with np.load(PROJECT_ROOT / NEW_HIDDEN_SPACE_ARRAYS_TEMPLATE.format(session=first_session)) as arrays:
        voxel_edge = float(arrays["voxel_size"])
    # Step 2: runs per arm over both sessions; one shared count, and their total.
    run_counts = [len(arm["runs"]) for session in SESSIONS for arm in load_summary(session)["arms"].values()]
    if len(set(run_counts)) != 1:
        raise ValueError(f"arms have different run counts {run_counts}; the definitions quote one repeat count")
    return {"voxel_edge_m": voxel_edge,
            "near_band_m": provenance["near_band_m"],
            "margin_quantile_percent": settings["margin_quantile"] * PERCENT_PER_UNIT,
            "neighbour_count": settings["margin_neighbour_count"],
            "neighbour_reach_m": provenance["range_margin"]["neighbour_reach_m"],
            "reach_fraction": NEIGHBOUR_REACH_VOXEL_FRACTION,
            "repeat_count": run_counts[0],
            "model_count": sum(run_counts)}


def headline_metric_keys(p2_thresholds: List[float]) -> List[str]:
    """Flat metric keys of the headline table, e.g. ``obstacles/P2@0.5/recall_Hnear/mean``.

    Args:
        p2_thresholds: P2 thresholds listed in the summary.

    Returns:
        Keys ordered threshold, metric, averaging.
    """
    return [f"{HEADLINE_GROUP}/P2@{threshold}/{suffix}/{averaging}"
            for threshold in p2_thresholds for suffix in HEADLINE_METRIC_SUFFIXES for averaging in AVERAGING_NAMES]


def metric_label(key: str) -> str:
    """Row label without the group prefix, e.g. ``P2@0.5 recall_Hnear mean``.

    Args:
        key: Flat metric key.

    Returns:
        The label.
    """
    _, threshold_part, metric_part, averaging_part = key.split("/")
    return f"{threshold_part} {metric_part} {averaging_part}"


def format_comparison(row: Optional[Dict[str, Any]]) -> str:
    """One cell: gap (first - second) and the separated flag, as script 20 computed them.

    Args:
        row: One entry of a comparison's ``metrics`` mapping, or None when absent.

    Returns:
        Text such as ``+0.1000 sep`` or ``-0.0200 ovl``; ``n/a`` when the metric is missing.
    """
    if row is None or row.get("gap") is None:
        return "n/a"
    # separated is None when an arm has too few repeats for a spread (p2_repeats.compare_arms).
    if row.get("separated") is None:
        return f"{row['gap']:+.4f} n/a"
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
    statistics = summary["arms"].get(arm, {}).get("metrics", {}).get(key)
    if statistics is None or statistics.get("mean") is None:
        return "n/a"
    # std is None with a single defined repeat (p2_repeats.describe_values).
    if statistics.get("std") is None:
        return f"{statistics['mean']:.4f} (n={statistics['n']})"
    return f"{statistics['mean']:.4f}±{statistics['std']:.4f}"


def session_lines(session: str, old_commit: str) -> List[str]:
    """Report lines of one session: margins, scored-set size, arm table and gap table.

    Args:
        session: Session name.
        old_commit: Git revision with the unbounded-margin summary.

    Returns:
        Markdown lines.
    """
    # Step 1: both versions of the summary, the margins and the short arm labels.
    summaries = {"old": load_summary(session, old_commit), "new": load_summary(session)}
    margins = load_margins(session, old_commit)
    arms = list(summaries["new"]["arms"])
    # Unknown methods keep their own name as the label.
    label_by_arm = {arm: ARM_LABEL_BY_METHOD.get(summaries["new"]["arms"][arm]["train_method"],
                                                 summaries["new"]["arms"][arm]["train_method"]) for arm in arms}
    # Camera loss per arm, from the Slurm profile the arm is named after.
    profiles = yaml.safe_load((PROJECT_ROOT / SLURM_CONFIG_RELATIVE_PATH).read_text())["profiles"]
    camera_by_arm = {arm: "camera loss off (lidar only)" if profiles[arm].get("train_lidar_only") else "camera loss on"
                     for arm in arms}
    lines = [f"## {session}", "",
             f"- Arms: " + "; ".join(f"{label_by_arm[arm]} = `{arm}`" for arm in arms) + ".",
             f"- Training: " + "; ".join(f"{label_by_arm[arm]}: {camera_by_arm[arm]}" for arm in arms) + ".",
             f"- Range margin m: old {margins['old_margin_m']:.4f} m, new {margins['new_margin_m']:.4f} m "
             f"(neighbour reach {margins['neighbour_reach_m']:.4f} m, {margins['residual_count']} residuals)."]
    # Step 2: scored-set size on H_near (a session property; the first arm carries it).
    for version, summary in summaries.items():
        first_arm = next(iter(summary["arms"]))
        sizes = [summary["arms"][first_arm]["metrics"][f"{HEADLINE_GROUP}/{name}"]["mean"] for name in SIZE_METRIC_NAMES]
        lines.append(f"- H_near obstacle GT cells ({version}): {sizes[0]:.0f}, in {sizes[1]:.0f} objects.")
    # Step 3: arm mean ± std per metric, old and new side by side.
    keys = headline_metric_keys(summaries["new"]["p2_thresholds"])
    lines += ["", "### Arm values (mean±std over the repeats)", "",
              "| metric | " + " | ".join(f"{label_by_arm[arm]} old | {label_by_arm[arm]} new" for arm in arms) + " |",
              "|---|" + "---|" * (2 * len(arms))]
    for key in keys:
        cells = [format_arm_value(summaries[version], arm, key) for arm in arms for version in ("old", "new")]
        lines.append(f"| {metric_label(key)} | " + " | ".join(cells) + " |")
    # Step 4: script 20's gaps and separation per arm pair, old -> new. Old pairs are matched by
    # their (first, second) arm names, never by list position.
    old_by_pair = {(comparison["first"], comparison["second"]): comparison for comparison in summaries["old"]["comparisons"]}
    pairs = [(comparison, old_by_pair.get((comparison["first"], comparison["second"]), {"metrics": {}}))
             for comparison in summaries["new"]["comparisons"]]
    pair_labels = [(label_by_arm[new["first"]], label_by_arm[new["second"]]) for new, _ in pairs]
    lines += ["", "### Gaps between arms (first − second), old → new", "",
              "| metric | " + " | ".join(f"{first} − {second}" for first, second in pair_labels) + " |",
              "|---|" + "---|" * len(pairs)]
    binding_threshold = summaries["new"]["settings"]["occupancy_threshold"]
    headline_key = f"{HEADLINE_GROUP}/P2@{binding_threshold}/{HEADLINE_METRIC_SUFFIX}"
    for key in keys:
        cells = []
        for (new_comparison, old_comparison), pair_label in zip(pairs, pair_labels):
            cell = (f"{format_comparison(old_comparison['metrics'].get(key))} → "
                    f"{format_comparison(new_comparison['metrics'].get(key))}")
            # Headline cells in bold: init + loss vs stock, per-object recall at the binding threshold only.
            is_headline = pair_label == HEADLINE_PAIR and key == headline_key
            cells.append(f"**{cell}**" if is_headline else cell)
        lines.append(f"| {metric_label(key)} | " + " | ".join(cells) + " |")
    return lines + [""]


def build_report(old_commit: str) -> str:
    """Whole report: title, provenance, definitions, then each session.

    Args:
        old_commit: Git revision with the unbounded-margin summaries.

    Returns:
        Markdown text.
    """
    lines = ["# Range-margin fix: hidden-voxel results before and after", "",
             f"Generated by `docs/research/hidden_space/compare_margin_rescore.py`. Old = summaries at commit "
             f"`{old_commit}` (unbounded margin); new = the summaries in this tree. Class `obstacles`, band H_near. "
             "Every number is read from those files; nothing is recomputed.", "",
             DEFINITIONS.format(**definition_values())]
    for session in SESSIONS:
        lines += session_lines(session, old_commit)
    return "\n".join(lines)


def main() -> None:
    """Parse arguments, write the report and print it."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--old-commit", default=DEFAULT_OLD_COMMIT)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / DEFAULT_OUTPUT_RELATIVE_PATH)
    arguments = parser.parse_args()
    report = build_report(arguments.old_commit)
    # A relative --output resolves from the repo root, as the default does.
    output_path = arguments.output if arguments.output.is_absolute() else PROJECT_ROOT / arguments.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report)
    print(report)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
