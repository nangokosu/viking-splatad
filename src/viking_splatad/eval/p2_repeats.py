"""Descriptive comparison of P1/P2 metrics across repeated runs grouped into arms (scripts/20).

SplatAD's run-to-run noise is accepted (CLAUDE.md rule 10), so no comparison between arms is read
without repeats: the gap between arm means is set beside the spread within each arm. Every repeat
of an arm uses the same seed and differs only by GPU nondeterminism, so repeats are not paired
across arms (index i of one arm has nothing in common with index i of another) and no
significance test is made: the output is descriptive (n, mean, sample std, min, max).
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from viking_splatad.eval.hidden_geometry import EVIDENCE_NAMES, exclusive_evidence_labels

# Point-metric names read from occupied_cell_metrics per test, and the evidence strata whose recall
# is reported (the radar claim concerns blocked and no-return cells; seen cells should agree).
POINT_METRIC_NAMES = ("precision", "recall", "iou")
REPORTED_STRATA = ("seen", "blocked", "no_return")
TEST_NAMES = ("P1", "P2")
# Sample standard deviation divides by n - 1 (Bessel), defined only for n >= 2.
SAMPLE_STD_DEGREES_OF_FREEDOM = 1
# An arm needs at least this many defined repeats before its spread means anything (CLAUDE.md rule 10):
# with one value min = max, so any gap would read as "separated".
MINIMUM_REPEATS_FOR_SPREAD = 2


def flatten_run_metrics(report: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """One run's scalar metrics as ``class/test/metric`` -> value (None where undefined).

    Args:
        report: A scripts/19 report (``classes`` -> P1, P2, present_but_hidden, detection).

    Returns:
        Flat mapping; the same keys for every run scored with the same settings.
    """
    flat: Dict[str, Optional[float]] = {}
    for class_name, class_report in report["classes"].items():
        # Step 1: voxel P/R/IoU and per-stratum recall for both tests.
        for test_name in TEST_NAMES:
            metrics = class_report[test_name]
            for metric_name in POINT_METRIC_NAMES:
                flat[f"{class_name}/{test_name}/{metric_name}"] = metrics[metric_name]
            for stratum_name in REPORTED_STRATA:
                flat[f"{class_name}/{test_name}/recall_{stratum_name}"] = metrics["strata"][stratum_name]["recall"]
        # Step 2: present-but-hidden GT cells, in total and in the strata of the claim.
        hidden = class_report["present_but_hidden"]
        flat[f"{class_name}/present_but_hidden/total"] = hidden["totals"]["present_but_hidden"]
        for stratum_name in REPORTED_STRATA:
            flat[f"{class_name}/present_but_hidden/{stratum_name}"] = hidden["strata"][stratum_name]["present_but_hidden"]
        # Step 3: the T79 detection rate at each f_min of the shared grid.
        curve = class_report["detection"]["curve"]
        for f_min, rate in zip(curve["f_min"], curve["detection_rate"]):
            # repr of the float is unique per grid value (a fixed decimal count could merge fine grids).
            flat[f"{class_name}/P2/detection_rate@{f_min!r}"] = rate
    return flat


def describe_values(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    """n, mean, sample std, min and max of the defined values; undefined (None) ones are counted apart.

    Args:
        values: One metric over an arm's repeats.

    Returns:
        {"n", "undefined", "mean", "std", "min", "max"}; statistics None when n == 0, std None when n < 2.
    """
    defined = np.asarray([value for value in values if value is not None], dtype=np.float64)
    count = int(defined.size)
    return {
        "n": count,
        "undefined": len(values) - count,
        "mean": float(defined.mean()) if count else None,
        "std": float(defined.std(ddof=SAMPLE_STD_DEGREES_OF_FREEDOM)) if count > SAMPLE_STD_DEGREES_OF_FREEDOM else None,
        "min": float(defined.min()) if count else None,
        "max": float(defined.max()) if count else None,
    }


def describe_arm(run_metrics: Sequence[Mapping[str, Optional[float]]]) -> Dict[str, Dict[str, Optional[float]]]:
    """``describe_values`` of every metric over an arm's runs.

    Args:
        run_metrics: ``flatten_run_metrics`` of each run of the arm (same keys).

    Returns:
        Metric key -> description.

    Raises:
        ValueError: If the runs do not share one metric set (different settings).
    """
    keys = list(run_metrics[0])
    if any(list(metrics) != keys for metrics in run_metrics):
        raise ValueError("the runs of an arm were scored with different settings (metric keys differ)")
    return {key: describe_values([metrics[key] for metrics in run_metrics]) for key in keys}


def compare_arms(first: Mapping[str, Mapping[str, Optional[float]]],
                 second: Mapping[str, Mapping[str, Optional[float]]]) -> Dict[str, Dict[str, Any]]:
    """Gap between two arms' means beside each arm's spread, per metric.

    ``separated`` is True when the arms' [min, max] ranges do not overlap: every repeat of one arm
    is above every repeat of the other. It is a descriptive reading aid, not a test, and None
    unless both arms have MINIMUM_REPEATS_FOR_SPREAD defined repeats.

    Args:
        first, second: ``describe_arm`` outputs of the two arms.

    Returns:
        Metric key -> {"gap" (first mean - second mean), "first_range", "second_range",
        "first_std", "second_std", "separated"}; values None where an arm has no defined value.
    """
    comparison: Dict[str, Dict[str, Any]] = {}
    for key in first:
        first_description, second_description = first[key], second[key]
        both_defined = first_description["n"] and second_description["n"]
        both_spread = min(first_description["n"], second_description["n"]) >= MINIMUM_REPEATS_FOR_SPREAD
        comparison[key] = {
            "gap": first_description["mean"] - second_description["mean"] if both_defined else None,
            "first_range": (first_description["max"] - first_description["min"]) if first_description["n"] else None,
            "second_range": (second_description["max"] - second_description["min"]) if second_description["n"] else None,
            "first_std": first_description["std"],
            "second_std": second_description["std"],
            # Disjoint closed intervals: one's minimum above the other's maximum.
            "separated": bool(first_description["min"] > second_description["max"]
                              or second_description["min"] > first_description["max"]) if both_spread else None,
        }
    return comparison


def consistent_cell_counts(first_occupied: Sequence[np.ndarray], second_occupied: Sequence[np.ndarray],
                           truth: np.ndarray, evidence: np.ndarray) -> Dict[str, Dict[str, int]]:
    """GT cells occupied in every repeat of one arm and in no repeat of the other, per evidence stratum.

    Args:
        first_occupied, second_occupied: Per-run (V,) bool occupied masks on the shared cells.
        truth: (V,) bool GT cells of the class. evidence: (V, 4) T78 flags of the shared cells.

    Returns:
        {"first_only": {stratum: count}, "second_only": {stratum: count}} with the exclusive T78 strata.
    """
    # all() over the repeats = recovered in every one; any() = recovered in at least one.
    first_always, first_ever = np.logical_and.reduce(first_occupied), np.logical_or.reduce(first_occupied)
    second_always, second_ever = np.logical_and.reduce(second_occupied), np.logical_or.reduce(second_occupied)
    labels = exclusive_evidence_labels(evidence)
    counts: Dict[str, Dict[str, int]] = {"first_only": {}, "second_only": {}}
    for label_index, stratum_name in enumerate(EVIDENCE_NAMES):
        in_stratum = truth & (labels == label_index)
        counts["first_only"][stratum_name] = int(np.count_nonzero(in_stratum & first_always & ~second_ever))
        counts["second_only"][stratum_name] = int(np.count_nonzero(in_stratum & second_always & ~first_ever))
    return counts


def arm_pairs(arm_names: Sequence[str]) -> List[tuple]:
    """Every unordered pair of arms, in the given order (first listed arm is ``first``).

    Args:
        arm_names: Arm names in report order.

    Returns:
        [(first, second), ...], k(k-1)/2 pairs for k arms.
    """
    return [(arm_names[first_index], arm_names[second_index])
            for first_index in range(len(arm_names)) for second_index in range(first_index + 1, len(arm_names))]
