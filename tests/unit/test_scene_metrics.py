"""Tests for whole-scene voxel metrics (T94 pooled, T95 per object, class report); expected values by hand."""

import numpy as np
import pytest

from viking_splatad.eval.scene_metrics import (
    class_group_masks, class_metric_report, csr_from_lists, per_object_metrics, pooled_metrics, threshold_key)

TOLERANCE = 1e-12


def mask(*indices, size=6):
    """Bool mask of length ``size`` that is True at ``indices``."""
    result = np.zeros(size, dtype=bool)
    result[list(indices)] = True
    return result


@pytest.mark.covers("T94")
def test_pooled_whole_neighbourhood():
    """V=6: N={0..4}, G={0,1,2}, O={0,2,3,5}."""
    occupied, truth = mask(0, 2, 3, 5), mask(0, 1, 2)
    neighbourhood = mask(0, 1, 2, 3, 4)
    # G = {0,1,2}; O n N = {0,2,3}; TP = {0,2}; FP = {3}; FN = {1}.
    baseline = pooled_metrics(occupied, truth, neighbourhood)
    assert (baseline["true_positive"], baseline["false_positive"], baseline["false_negative"]) == (2, 1, 1)
    assert baseline["recall"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert baseline["precision"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert baseline["iou"] == pytest.approx(2 / 4, abs=TOLERANCE)


@pytest.mark.covers("T94")
def test_pooled_zero_denominators_are_none():
    """No GT and no occupied cells: every ratio is None, never NaN."""
    result = pooled_metrics(mask(), mask(), mask(0, 1))
    assert result["recall"] is None and result["precision"] is None and result["iou"] is None


@pytest.mark.covers("T94")
def test_truth_outside_neighbourhood_raises():
    """Truth cell 5 is not in N."""
    with pytest.raises(ValueError):
        pooled_metrics(mask(0), mask(5), mask(0, 1))


@pytest.mark.covers("T94")
def test_other_class_mask_excludes_cell():
    """Cell 2 carries class 2; group {1}: it leaves N and is never a false positive."""
    gt_labels = np.array([1, 0, 2, 0, 0, 0])
    neighbourhood, truth = class_group_masks(
        gt_labels, [1], {1: mask(0, 1, 2, 3), 2: mask(2, 3)})
    assert np.array_equal(neighbourhood, mask(0, 1, 3))
    assert np.array_equal(truth, mask(0))
    result = pooled_metrics(mask(0, 2), truth, neighbourhood)
    # O n N = {0}: cell 2 does not count as FP.
    assert result["false_positive"] == 0 and result["true_positive"] == 1
    # Group of two ids: union of both neighbourhoods, no mask for either member.
    group_n, group_truth = class_group_masks(gt_labels, [1, 2], {1: mask(0, 1), 2: mask(2, 3)})
    assert np.array_equal(group_n, mask(0, 1, 2, 3)) and np.array_equal(group_truth, mask(0, 2))
    with pytest.raises(ValueError):
        class_group_masks(gt_labels, [3], {1: mask(0)})


def build_two_objects():
    """Objects 0 and 1 overlap on cell 2 (occupied, no GT). V=6.

    Object 0: cells {0,1,2}, inside {0,1}; object 1: cells {2,3,4}, inside {3,4}.
    """
    return csr_from_lists(
        [np.array([0, 1, 2]), np.array([2, 3, 4])],
        [np.array([True, True, False]), np.array([False, True, True])])


@pytest.mark.covers("T95")
def test_csr_layout():
    """Offsets are cumulative sizes."""
    offsets, cells, inside = build_two_objects()
    assert offsets.tolist() == [0, 3, 6] and cells.tolist() == [0, 1, 2, 2, 3, 4]
    assert inside.tolist() == [True, True, False, False, True, True]


@pytest.mark.covers("T95")
def test_overlap_counts_against_each_object():
    """Occupied cell 2 (off GT) is a false positive of BOTH objects."""
    offsets, cells, inside = build_two_objects()
    truth, neighbourhood = mask(0, 1, 3, 4), mask(0, 1, 2, 3, 4)
    result = per_object_metrics(
        mask(0, 2, 3), truth, neighbourhood, offsets, cells, inside)
    assert [entry["false_positive"] for entry in result["per_object"]] == [1, 1]
    # Object 0: TP={0}, FP={2}, FN={1}; object 1: TP={3}, FP={2}, FN={4}.
    assert result["per_object"][0]["recall"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert result["per_object"][1]["precision"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert result["mean_iou"] == pytest.approx(1 / 3, abs=TOLERANCE)
    assert result["object_count"] == 2 and result["objects_with_gt"] == 2


@pytest.mark.covers("T95")
def test_object_with_gt_but_nothing_occupied():
    """Object 0 has GT, no occupancy: recall 0, precision None. Object 1 is perfect."""
    offsets, cells, inside = build_two_objects()
    truth, neighbourhood = mask(0, 3), mask(0, 1, 2, 3, 4)
    result = per_object_metrics(mask(3), truth, neighbourhood, offsets, cells, inside)
    assert result["per_object"][0]["recall"] == 0.0
    assert result["per_object"][0]["precision"] is None
    assert result["defined_counts"] == {"recall": 2, "precision": 1, "iou": 2}
    assert result["mean_precision"] == pytest.approx(1.0, abs=TOLERANCE)
    assert result["mean_recall"] == pytest.approx((0.0 + 1.0) / 2, abs=TOLERANCE)


@pytest.mark.covers("T95")
def test_per_object_truth_outside_neighbourhood_raises():
    """Truth cell 5 is not in N."""
    offsets, cells, inside = build_two_objects()
    with pytest.raises(ValueError):
        per_object_metrics(mask(0), mask(5), mask(0, 1, 2, 3, 4), offsets, cells, inside)


@pytest.mark.covers("T95")
def test_filled_cell_on_neighbour_object_is_false_positive():
    """Cell 3 is truth of object B (outside A's box) but inside A's grown box: it lowers A's precision.

    V=6. Object A: grown cells {0,1,2}, inside {0,1}; object B: cells {2,3}, inside {3} (B's box).
    G = {0,3}, N = {0,1,2,3}, O = {0,2}. Object A's grown box includes cell 2 (not inside, not truth).
    """
    offsets, cells, inside = csr_from_lists(
        [np.array([0, 1, 2]), np.array([2, 3])],
        [np.array([True, True, False]), np.array([False, True])])
    truth, neighbourhood = mask(0, 3), mask(0, 1, 2, 3)
    # Occupy cell 2, which is on A's grown ring but is not truth of A: FP of A.
    result = per_object_metrics(mask(0, 2), truth, neighbourhood, offsets, cells, inside)
    # A: N_A = {0,1,2}, G_A = {0}; O n N_A = {0,2}: TP=1, FP=1 -> precision 1/2.
    assert result["per_object"][0]["precision"] == pytest.approx(1 / 2, abs=TOLERANCE)
    # Same A with cell 2 empty: precision 1.
    clean = per_object_metrics(mask(0), truth, neighbourhood, offsets, cells, inside)
    assert clean["per_object"][0]["precision"] == pytest.approx(1.0, abs=TOLERANCE)
    # Now put B's truth cell 3 inside A's grown box: A = cells {0,1,3}, inside {0,1}; cell 3 is truth but off A's box.
    offsets_a3, cells_a3, inside_a3 = csr_from_lists(
        [np.array([0, 1, 3])], [np.array([True, True, False])])
    shared = per_object_metrics(mask(0, 3), truth, neighbourhood, offsets_a3, cells_a3, inside_a3)
    # G_A = {0}; O n N_A = {0,3}: TP=1, FP=1 (cell 3 is another object's truth) -> precision 1/2.
    assert shared["per_object"][0]["false_positive"] == 1
    assert shared["per_object"][0]["precision"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert shared["per_object"][0]["gt_cells"] == 1


def test_class_metric_report_matches_direct_calls():
    """Report blocks equal pooled / per-object metrics computed from hand-built counts.

    V=6, one object of class 1 (cells {0,1,2}, inside {0,1}); N={0,1,2,3}, G={0,1}.
    P1 hit = {0,2}: TP=1, FP=1 (cell 2), FN=1 -> recall 1/2, precision 1/2, IoU 1/3.
    occupancy = [0.9, 0.6, 0.1, 0.7, 0, 0]; P2@0.5 = {0,1,3}: TP=2, FP=1, FN=0 -> recall 1, precision 2/3, IoU 2/3.
    P2@0.8 = {0}: TP=1, FP=0, FN=1 -> recall 1/2, precision 1, IoU 1/2.
    """
    offsets, cells, inside = csr_from_lists([np.array([0, 1, 2])], [np.array([True, True, False])])
    objects = {"object_class_ids": np.array([1]), "object_cell_offsets": offsets,
               "object_cells": cells, "object_inside": inside}
    neighbourhood, truth = mask(0, 1, 2, 3), mask(0, 1)
    occupancy = np.array([0.9, 0.6, 0.1, 0.7, 0.0, 0.0])
    report = class_metric_report([1], neighbourhood, truth, mask(0, 2), occupancy, [0.5, 0.8], objects)
    assert set(report) == {"P1", "P2"}
    assert set(report["P2"]) == {threshold_key(0.5), threshold_key(0.8)}
    p1 = report["P1"]["pooled"]
    assert (p1["true_positive"], p1["false_positive"], p1["false_negative"]) == (1, 1, 1)
    assert p1["recall"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert p1["precision"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert p1["iou"] == pytest.approx(1 / 3, abs=TOLERANCE)
    p2_half = report["P2"][threshold_key(0.5)]["pooled"]
    assert (p2_half["true_positive"], p2_half["false_positive"], p2_half["false_negative"]) == (2, 1, 0)
    assert p2_half["precision"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert p2_half["iou"] == pytest.approx(2 / 3, abs=TOLERANCE)
    p2_high = report["P2"][threshold_key(0.8)]["pooled"]
    assert (p2_high["true_positive"], p2_high["false_positive"], p2_high["false_negative"]) == (1, 0, 1)
    assert p2_high["precision"] == pytest.approx(1.0, abs=TOLERANCE)
    # Per-object block: object 0 has N_0={0,1,2}, G_0={0,1}; at P2@0.5 O n N_0={0,1}: TP=2, FP=0, FN=0.
    per_object_half = report["P2"][threshold_key(0.5)]["per_object"]
    assert "per_object" not in per_object_half
    assert per_object_half["mean_recall"] == pytest.approx(1.0, abs=TOLERANCE)
    assert per_object_half["mean_precision"] == pytest.approx(1.0, abs=TOLERANCE)
    assert per_object_half["object_count"] == 1
