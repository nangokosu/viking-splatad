"""Descriptive comparison of P1/P2 metrics across repeated runs grouped into arms (scripts/20).

SplatAD's run-to-run noise is accepted (CLAUDE.md rule 10), so no comparison between arms is read
without repeats: the gap between arm means is set beside the spread within each arm. Every repeat
of an arm uses the same seed and differs only by GPU nondeterminism, so repeats are not paired
across arms (index i of one arm has nothing in common with index i of another) and no
significance test is made: the output is descriptive (n, mean, sample std, min, max).
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from viking_splatad.eval.run_quality import IMAGE_METRIC_NAMES, IMAGE_RESULT_KEY, LIDAR_RESULT_KEY, REPORTED_LIDAR_METRICS

# Metric names read from each pooled / per-object block of a scripts/19 report.
POINT_METRIC_NAMES = ("precision", "recall", "iou")
# Sample standard deviation divides by n - 1 (Bessel), defined only for n >= 2.
SAMPLE_STD_DEGREES_OF_FREEDOM = 1
# An arm needs at least this many defined repeats before its spread means anything (CLAUDE.md rule 10):
# with one value min = max, so any gap would read as "separated".
MINIMUM_REPEATS_FOR_SPREAD = 2


# Key segments of the flattened voxel metrics: pooled over cells, mean over objects, objects defining the mean.
POOLED_SEGMENT, MEAN_SEGMENT, N_DEFINED_SEGMENT = "pooled", "mean", "n_defined"
P1_TEST_NAME, P2_TEST_NAME = "P1", "P2"
# First key segment of the per-run quality metrics.
QUALITY_KEY_PREFIX = "quality"


def _flatten_test_block(prefix: str, test_block: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """One test's (P1 or one P2 threshold) metrics, pooled and averaged over objects.

    Args:
        prefix: Key prefix, ``<class>/<P1|P2@<threshold>>``.
        test_block: {pooled, per_object} from scripts/19 (``class_metric_report``).

    Returns:
        ``<prefix>/<metric>/<pooled|mean|n_defined>`` -> value (None where undefined).
    """
    pooled, per_object = test_block["pooled"], test_block["per_object"]
    flat: Dict[str, Optional[float]] = {}
    for metric_name in POINT_METRIC_NAMES:
        key_base = f"{prefix}/{metric_name}"
        flat[f"{key_base}/{POOLED_SEGMENT}"] = pooled[metric_name]
        flat[f"{key_base}/{MEAN_SEGMENT}"] = per_object[f"mean_{metric_name}"]
        flat[f"{key_base}/{N_DEFINED_SEGMENT}"] = per_object["defined_counts"][metric_name]
    return flat


def flatten_run_metrics(report: Mapping[str, Any]) -> Dict[str, Optional[float]]:
    """One run's scalar metrics (scripts/19 schema 3) as flat keys.

    Keys: ``<class>/<P1|P2@<t>>/<precision|recall|iou>/<pooled|mean|n_defined>`` (``n_defined`` = objects
    defining the mean) and ``<class>/P2/detection_rate@<f_min>``.

    Args:
        report: A scripts/19 schema-3 report (``classes`` -> P1, P2 by threshold, detection).

    Returns:
        Flat mapping (None where undefined); the same keys for every run scored with the same settings.
    """
    flat: Dict[str, Optional[float]] = {}
    for class_name, class_report in report["classes"].items():
        # Step 1: the lidar view, then each P2 threshold in report order.
        flat.update(_flatten_test_block(f"{class_name}/{P1_TEST_NAME}", class_report[P1_TEST_NAME]))
        for threshold_text, test_block in class_report[P2_TEST_NAME].items():
            flat.update(_flatten_test_block(f"{class_name}/{P2_TEST_NAME}@{threshold_text}", test_block))
        # Step 2: the T79 detection rate at each f_min of the shared grid.
        curve = class_report["detection"]["curve"]
        for f_min, rate in zip(curve["f_min"], curve["detection_rate"]):
            # repr of the float is unique per grid value (a fixed decimal count could merge fine grids).
            flat[f"{class_name}/{P2_TEST_NAME}/detection_rate@{f_min!r}"] = rate
    return flat


def flatten_quality_metrics(quality: Mapping[str, Mapping[str, Any]]) -> Dict[str, Optional[float]]:
    """One run's image-guard and lidar-render metrics as ``quality/<image|lidar_render>/<name>`` keys.

    Args:
        quality: ``run_quality`` output ({"image": {psnr, ssim, lpips, step}, "lidar_render": {...}}).

    Returns:
        Flat mapping of psnr/ssim/lpips and the reported lidar metrics (the image ``step`` is not a metric).
    """
    flat: Dict[str, Optional[float]] = {}
    for image_metric_name in IMAGE_METRIC_NAMES:
        flat[f"{QUALITY_KEY_PREFIX}/{IMAGE_RESULT_KEY}/{image_metric_name}"] = quality[IMAGE_RESULT_KEY][image_metric_name]
    for lidar_metric_name in REPORTED_LIDAR_METRICS:
        flat[f"{QUALITY_KEY_PREFIX}/{LIDAR_RESULT_KEY}/{lidar_metric_name}"] = quality[LIDAR_RESULT_KEY][lidar_metric_name]
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
        Metric key -> {"gap" (first mean - second mean), "first_range", "second_range" (max - min
        of each arm), "first_std", "second_std", "separated"}; values None where an arm has no defined value.
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


def arm_pairs(arm_names: Sequence[str]) -> List[tuple]:
    """Every unordered pair of arms, in the given order (first listed arm is ``first``).

    Args:
        arm_names: Arm names in report order.

    Returns:
        [(first, second), ...], k(k-1)/2 pairs for k arms.
    """
    return [(arm_names[first_index], arm_names[second_index])
            for first_index in range(len(arm_names)) for second_index in range(first_index + 1, len(arm_names))]
