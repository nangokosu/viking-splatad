"""Known-answer tests for the repeat comparison of P1/P2 metrics (eval/p2_repeats.py)."""

import math

import numpy as np
import pytest

from viking_splatad.eval.hidden_geometry import EVIDENCE_NAMES
from viking_splatad.eval.p2_repeats import (
    arm_pairs,
    compare_arms,
    consistent_cell_counts,
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


def test_consistent_cell_counts_per_stratum():
    """4 GT cells (seen, blocked, no_return, unobserved) and one non-GT cell.

    Arm A recovers cells 0, 1 in both repeats and cell 2 in one; arm B recovers cell 3 in both and cell 1 once.
    Only cell 0 is A-always and B-never; cell 3 is B-always and A-never; the non-GT cell never counts.
    """
    truth = np.array([True, True, True, True, False])
    evidence = np.zeros((5, 4), dtype=bool)
    evidence[0, 0], evidence[1, 1], evidence[2, 2] = True, True, True
    first = [np.array([True, True, True, False, True]), np.array([True, True, False, False, True])]
    second = [np.array([False, True, False, True, False]), np.array([False, False, False, True, False])]
    counts = consistent_cell_counts(first, second, truth, evidence)
    expected_first = {name: 0 for name in EVIDENCE_NAMES} | {"seen": 1}
    expected_second = {name: 0 for name in EVIDENCE_NAMES} | {"unobserved": 1}
    assert counts == {"first_only": expected_first, "second_only": expected_second}


def test_arm_pairs_lists_every_unordered_pair_once():
    """k arms give k(k-1)/2 pairs, first-listed arm first."""
    assert arm_pairs(["a", "b", "c"]) == [("a", "b"), ("a", "c"), ("b", "c")]


def test_flatten_run_metrics_keys_and_values():
    """A one-class report flattens to P1/P2 P/R/IoU, stratum recalls, hidden counts and detection rates."""
    strata = {name: {"recall": None} for name in EVIDENCE_NAMES} | {"blocked": {"recall": 0.25}}
    test_metrics = {"precision": 0.5, "recall": 0.4, "iou": 0.3, "strata": strata}
    hidden_strata = {name: {"present_but_hidden": 0} for name in EVIDENCE_NAMES} | {"no_return": {"present_but_hidden": 3}}
    report = {"classes": {"tree_trunk": {
        "P1": test_metrics, "P2": test_metrics,
        "present_but_hidden": {"strata": hidden_strata, "totals": {"present_but_hidden": 3}},
        "detection": {"curve": {"f_min": [0.0, 0.5], "detection_rate": [1.0, None]}},
    }}}
    flat = flatten_run_metrics(report)
    assert flat["tree_trunk/P2/precision"] == 0.5
    assert flat["tree_trunk/P1/recall_blocked"] == 0.25
    assert flat["tree_trunk/present_but_hidden/no_return"] == 3
    assert flat["tree_trunk/P2/detection_rate@0.0"] == 1.0
    assert flat["tree_trunk/P2/detection_rate@0.5"] is None


def test_single_repeat_arms_are_never_read_as_separated():
    """With one repeat min = max, so a gap says nothing about spread (rule 10): separated is None."""
    single = describe_arm([{"metric": 0.9}])
    pair = describe_arm([{"metric": 0.1}, {"metric": 0.2}])
    assert compare_arms(single, pair)["metric"]["separated"] is None
    assert compare_arms(single, pair)["metric"]["gap"] == pytest.approx(0.9 - 0.15)
