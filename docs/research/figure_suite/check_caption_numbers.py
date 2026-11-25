"""Check that the figure suite's whole-scene threshold curves equal scripts/19's report for every drawn run.

For each session's cross-arm manifest (reports/gaussian_occupancy/figures/<session>/figures_manifest.json,
written by scripts/26) and each arm, at the binding threshold t = p2_thresholds[0]:
    the whole-scene threshold curve of ``reported_numbers.threshold_curves_scene`` at t equals the
    whole-scene block of the drawn run's eval/gaussian_occupancy.json (the ``baseline_pooled`` and
    ``baseline_per_object`` blocks under ``classes.<class>.hidden_near.P2.<t>``): pooled recall / precision
    and the per-object means.
Values must agree to floating-point tolerance (they come from the same T94/T95 functions).

Usage:
    uv run python docs/research/figure_suite/check_caption_numbers.py [--session <s> ...]
Exit code 0 when every comparison agrees, 1 otherwise. Prints one line per comparison.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FIGURES_ROOT = PROJECT_ROOT / "reports" / "gaussian_occupancy" / "figures"
RUNS_ROOT = PROJECT_ROOT / "runs"
MANIFEST_NAME = "figures_manifest.json"
REPORT_RELATIVE_PATH = Path("eval") / "gaussian_occupancy.json"
# Relative and absolute tolerance: both sides are float64 ratios of the same integer counts.
RELATIVE_TOLERANCE = 1e-12
ABSOLUTE_TOLERANCE = 1e-12
# Curve key -> (report block, report key) of the same quantity; the whole-scene blocks of scripts/19 are the
# ``baseline_*`` ones (the same metrics without the hidden-space restriction).
CURVE_TO_REPORT = {"pooled_recall": ("baseline_pooled", "recall"),
                   "pooled_precision": ("baseline_pooled", "precision"),
                   "mean_recall": ("baseline_per_object", "mean_recall"),
                   "mean_precision": ("baseline_per_object", "mean_precision")}


def values_agree(first: Optional[float], second: Optional[float]) -> bool:
    """Equality to tolerance, with None (undefined) equal only to None.

    Args:
        first: A value or None.
        second: A value or None.

    Returns:
        True when both are None or both are close.
    """
    if first is None or second is None:
        return first is None and second is None
    return math.isclose(first, second, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE)


def threshold_report_key(report_block: Dict, threshold: float) -> str:
    """The key of ``report_block`` (scripts/19 P2 thresholds, e.g. "0.5") whose value equals ``threshold``.

    Args:
        report_block: ``hidden_near.P2`` mapping keyed by threshold text.
        threshold: The binding threshold from the manifest.

    Returns:
        The matching key.

    Raises:
        KeyError: When no key parses to the threshold.
    """
    for key in report_block:
        if math.isclose(float(key), threshold, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE):
            return key
    raise KeyError(f"no P2 block at threshold {threshold}: {sorted(report_block)}")


def check_session(session: str) -> List[str]:
    """Compare one session's whole-scene curve numbers with its drawn runs' reports.

    Args:
        session: Session name.

    Returns:
        One line per comparison, each starting with "OK" or "MISMATCH".
    """
    manifest = json.loads((FIGURES_ROOT / session / MANIFEST_NAME).read_text())
    class_name = manifest["class"]
    binding_threshold = float(manifest["binding_threshold"])
    curves = manifest["reported_numbers"]["threshold_curves_scene"]
    curve_grid = curves["curve_grid"]
    # Step 1: the grid index of the binding threshold (scripts/26 snaps it to the archived value).
    grid_index = next(index for index, value in enumerate(curve_grid)
                      if math.isclose(value, binding_threshold, rel_tol=RELATIVE_TOLERANCE, abs_tol=ABSOLUTE_TOLERANCE))
    lines = []
    for arm, arm_record in manifest["arms"].items():
        report = json.loads((RUNS_ROOT / arm_record["drawn_run_id"] / REPORT_RELATIVE_PATH).read_text())
        near_p2 = report["classes"][class_name]["hidden_near"]["P2"]
        report_block = near_p2[threshold_report_key(near_p2, binding_threshold)]
        # Step 2: the whole-scene curve point at the binding threshold against scripts/19's whole-scene block.
        comparisons = []
        for curve_key, (block_name, report_key) in CURVE_TO_REPORT.items():
            curve_value = curves["drawn_curves"][arm][curve_key][grid_index]
            comparisons.append((f"curve {curve_key}", curve_value, report_block[block_name][report_key]))
        for label, figure_value, report_value in comparisons:
            status = "OK" if values_agree(figure_value, report_value) else "MISMATCH"
            lines.append(f"{status} {session} {arm} {label}: figure {figure_value} report {report_value}")
    return lines


def main(argument_list: Optional[List[str]] = None) -> int:
    """Run the checks for the requested (default: every drawn) session.

    Args:
        argument_list: Command-line arguments (default sys.argv[1:]).

    Returns:
        0 when every comparison agrees, else 1.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", action="append", help="session to check (repeatable; default: all)")
    arguments = parser.parse_args(argument_list)
    sessions = arguments.session or sorted(folder.name for folder in FIGURES_ROOT.iterdir()
                                           if (folder / MANIFEST_NAME).is_file())
    lines = [line for session in sessions for line in check_session(session)]
    print("\n".join(lines))
    mismatch_count = sum(line.startswith("MISMATCH") for line in lines)
    print(f"{len(lines) - mismatch_count} OK, {mismatch_count} mismatch")
    return 1 if mismatch_count else 0


if __name__ == "__main__":
    sys.exit(main())
