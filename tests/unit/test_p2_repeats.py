"""Known-answer tests for the repeat comparison of P1/P2 metrics (eval/p2_repeats.py)."""

import math

import pytest

from viking_splatad.eval.p2_repeats import (
    arm_pairs,
    compare_arms,
    describe_arm,
    describe_values,
    flatten_run_metrics,
)


def test_describe_values_uses_bessel_std_and_counts_undefined_apart():
    """Values 1, 2, 4 and one None: mean 7/3, sample variance sum((v-mean)^2)/(n-1), min 1, max 4."""
    values = [1.0, 2.0, None, 4.0]
    defined = [1.0, 2.0, 4.0]
    mean = sum(defined) / len(defined)
    sample_std = math.sqrt(sum((value - mean) ** 2 for value in defined) / (len(defined) - 1))
    description = describe_values(values)
    assert description["n"] == 3 and description["undefined"] == 1
    assert description["mean"] == pytest.approx(mean)
    assert description["std"] == pytest.approx(sample_std)
    assert (description["min"], description["max"]) == (1.0, 4.0)


def test_describe_values_single_and_empty():
    """One value has no sample std; no defined value has no statistics at all."""
    assert describe_values([0.5])["std"] is None
    empty = describe_values([None, None])
    assert empty["n"] == 0 and empty["undefined"] == 2 and empty["mean"] is None


def test_compare_arms_gap_ranges_and_separation():
    """Arm A {0.6, 0.7} vs B {0.4, 0.5}: gap 0.2, ranges 0.1 each, separated; overlapping arms are not."""
    first = describe_arm([{"metric": 0.6}, {"metric": 0.7}])
    second = describe_arm([{"metric": 0.4}, {"metric": 0.5}])
    comparison = compare_arms(first, second)["metric"]
    assert comparison["gap"] == pytest.approx(0.2)
    assert comparison["first_range"] == pytest.approx(0.1) and comparison["second_range"] == pytest.approx(0.1)
    assert comparison["separated"] is True
    overlapping = describe_arm([{"metric": 0.45}, {"metric": 0.7}])
    assert compare_arms(overlapping, second)["metric"]["separated"] is False


def test_describe_arm_refuses_runs_with_different_metric_sets():
    """Runs scored with different settings have different keys and cannot be pooled."""
    with pytest.raises(ValueError):
        describe_arm([{"metric": 0.1}, {"other": 0.2}])


def test_arm_pairs_lists_every_unordered_pair_once():
    """k arms give k(k-1)/2 pairs, first-listed arm first."""
    assert arm_pairs(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]


METRIC_NAMES = ("precision", "recall", "iou")


def schema3_block(precision, recall, iou, mean_precision, mean_recall, mean_iou, defined_counts):
    """One schema-3 test block ({pooled, per_object}) with the given metric values.

    Args:
        precision, recall, iou: Pooled values. mean_precision, mean_recall, mean_iou: Means over objects.
        defined_counts: {metric: objects defining it}.
    Returns:
        The block as scripts/19 writes it.
    """
    return {"pooled": {"precision": precision, "recall": recall, "iou": iou},
            "per_object": {"mean_precision": mean_precision, "mean_recall": mean_recall, "mean_iou": mean_iou,
                           "defined_counts": defined_counts}}


# Distinct values per test so a key that read the wrong block would show.
P1_BLOCK = schema3_block(0.5, 0.4, 0.3, 0.55, 0.45, 0.35, {"precision": 4, "recall": 5, "iou": 3})
P2_LOW_BLOCK = schema3_block(0.6, 0.7, 0.8, 0.65, 0.75, None, {"precision": 6, "recall": 7, "iou": 0})
P2_HIGH_BLOCK = schema3_block(None, 0.1, 0.2, None, 0.15, 0.25, {"precision": 0, "recall": 2, "iou": 1})
DETECTION = {"curve": {"f_min": [0.0, 0.5], "detection_rate": [1.0, None]}}


def schema2_baseline_keys(report):
    """Flat baseline keys of a schema-2 report, as the removed hidden flattener spelled them.

    Args:
        report: A schema-2 report (each test block has baseline_pooled / baseline_per_object).
    Returns:
        {key: value} for ``<class>/<P1|P2@t>/<metric>/<pooled|mean|n_defined>``.
    """
    flat = {}
    for class_name, class_report in report["classes"].items():
        hidden_block = class_report["hidden"]
        test_blocks = {"P1": hidden_block["P1"]} | {f"P2@{threshold}": block for threshold, block in hidden_block["P2"].items()}
        for test_name, block in test_blocks.items():
            for metric_name in METRIC_NAMES:
                prefix = f"{class_name}/{test_name}/{metric_name}"
                flat[f"{prefix}/pooled"] = block["baseline_pooled"][metric_name]
                flat[f"{prefix}/mean"] = block["baseline_per_object"][f"mean_{metric_name}"]
                flat[f"{prefix}/n_defined"] = block["baseline_per_object"]["defined_counts"][metric_name]
    return flat


def test_flatten_run_metrics_keys_and_values():
    """A one-class schema-3 report flattens to exactly P1/P2@t P/R/IoU keys (pooled, mean, n_defined) and detection rates."""
    report = {"classes": {"tree_trunk": {
        "P1": P1_BLOCK, "P2": {"0.5": P2_LOW_BLOCK, "0.8": P2_HIGH_BLOCK}, "detection": DETECTION}}}
    flat = flatten_run_metrics(report)
    expected = {}
    for test_name, block in (("P1", P1_BLOCK), ("P2@0.5", P2_LOW_BLOCK), ("P2@0.8", P2_HIGH_BLOCK)):
        for metric_name in METRIC_NAMES:
            prefix = f"tree_trunk/{test_name}/{metric_name}"
            expected[f"{prefix}/pooled"] = block["pooled"][metric_name]
            expected[f"{prefix}/mean"] = block["per_object"][f"mean_{metric_name}"]
            expected[f"{prefix}/n_defined"] = block["per_object"]["defined_counts"][metric_name]
    expected["tree_trunk/P2/detection_rate@0.0"] = 1.0
    expected["tree_trunk/P2/detection_rate@0.5"] = None
    assert flat == expected
    # Spot values by hand: P1 pooled recall, P2@0.8 mean iou, an undefined ratio stays None.
    assert flat["tree_trunk/P1/recall/pooled"] == 0.4
    assert flat["tree_trunk/P2@0.8/iou/mean"] == 0.25
    assert flat["tree_trunk/P2@0.8/precision/pooled"] is None


def test_flatten_run_metrics_keys_match_the_schema2_baseline_keys():
    """Key compatibility: a schema-3 report and the schema-2 report with the same numbers give the same baseline keys and values."""
    def hidden_variant(block):
        """A schema-2 block: made-up hidden numbers beside the baseline pair that equals ``block``."""
        decoy = schema3_block(0.01, 0.02, 0.03, 0.04, 0.05, 0.06, {"precision": 9, "recall": 9, "iou": 9})
        return {"pooled": decoy["pooled"], "per_object": decoy["per_object"],
                "baseline_pooled": block["pooled"], "baseline_per_object": block["per_object"]}
    schema2_report = {"classes": {"tree_trunk": {"hidden": {
        "P1": hidden_variant(P1_BLOCK), "P2": {"0.5": hidden_variant(P2_LOW_BLOCK), "0.8": hidden_variant(P2_HIGH_BLOCK)}}}}}
    schema3_report = {"classes": {"tree_trunk": {
        "P1": P1_BLOCK, "P2": {"0.5": P2_LOW_BLOCK, "0.8": P2_HIGH_BLOCK}, "detection": DETECTION}}}
    flat = flatten_run_metrics(schema3_report)
    baseline = schema2_baseline_keys(schema2_report)
    assert baseline.items() <= flat.items()
    # Nothing but the baseline keys and the detection rates is produced.
    assert set(flat) - set(baseline) == {"tree_trunk/P2/detection_rate@0.0", "tree_trunk/P2/detection_rate@0.5"}


def test_single_repeat_arms_are_never_read_as_separated():
    """With one repeat min = max, so a gap says nothing about spread (rule 10): separated is None."""
    single = describe_arm([{"metric": 0.9}])
    pair = describe_arm([{"metric": 0.1}, {"metric": 0.2}])
    assert compare_arms(single, pair)["metric"]["separated"] is None
    assert compare_arms(single, pair)["metric"]["gap"] == pytest.approx(0.9 - 0.15)
