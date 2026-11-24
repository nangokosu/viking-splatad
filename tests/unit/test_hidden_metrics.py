"""Tests for hidden-voxel metrics (T94 pooled, T95 per object); expected values by hand."""

import numpy as np
import pytest

from viking_splatad.eval.hidden_metrics import (
    class_group_masks, csr_from_lists, per_object_hidden_metrics, pooled_hidden_metrics)

TOLERANCE = 1e-12


def mask(*indices, size=6):
    """Bool mask of length ``size`` that is True at ``indices``."""
    result = np.zeros(size, dtype=bool)
    result[list(indices)] = True
    return result


@pytest.mark.covers("T94")
def test_pooled_with_and_without_hidden():
    """V=6: N={0..4}, G={0,1,2}, O={0,2,3,5}, H={1,2,3}."""
    occupied, truth = mask(0, 2, 3, 5), mask(0, 1, 2)
    neighbourhood, hidden = mask(0, 1, 2, 3, 4), mask(1, 2, 3)
    result = pooled_hidden_metrics(occupied, truth, neighbourhood, hidden)
    # G n H = {1,2}; O n N n H = {2,3}; TP = {2}; FP = {3}; FN = {1}.
    assert (result["true_positive"], result["false_positive"], result["false_negative"]) == (1, 1, 1)
    assert result["gt_cells"] == 2 and result["occupied_cells"] == 2
    assert result["recall"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert result["precision"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert result["iou"] == pytest.approx(1 / (1 + 1 + 1), abs=TOLERANCE)
    # Baseline: G = {0,1,2}; O n N = {0,2,3}; TP = {0,2}; FP = {3}; FN = {1}.
    baseline = pooled_hidden_metrics(occupied, truth, neighbourhood)
    assert (baseline["true_positive"], baseline["false_positive"], baseline["false_negative"]) == (2, 1, 1)
    assert baseline["recall"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert baseline["precision"] == pytest.approx(2 / 3, abs=TOLERANCE)
    assert baseline["iou"] == pytest.approx(2 / 4, abs=TOLERANCE)


@pytest.mark.covers("T94")
def test_pooled_zero_denominators_are_none():
    """No GT and no occupied cells: every ratio is None, never NaN."""
    result = pooled_hidden_metrics(mask(), mask(), mask(0, 1))
    assert result["recall"] is None and result["precision"] is None and result["iou"] is None


@pytest.mark.covers("T94")
def test_truth_outside_neighbourhood_raises():
    """Truth cell 5 is not in N."""
    with pytest.raises(ValueError):
        pooled_hidden_metrics(mask(0), mask(5), mask(0, 1))


@pytest.mark.covers("T94")
def test_other_class_mask_excludes_cell():
    """Cell 2 carries class 2; group {1}: it leaves N and is never a false positive."""
    gt_labels = np.array([1, 0, 2, 0, 0, 0])
    neighbourhood, truth = class_group_masks(
        gt_labels, [1], {1: mask(0, 1, 2, 3), 2: mask(2, 3)})
    assert np.array_equal(neighbourhood, mask(0, 1, 3))
    assert np.array_equal(truth, mask(0))
    result = pooled_hidden_metrics(mask(0, 2), truth, neighbourhood)
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
    result = per_object_hidden_metrics(
        mask(0, 2, 3), truth, neighbourhood, None, offsets, cells, inside)
    assert [entry["false_positive"] for entry in result["per_object"]] == [1, 1]
    # Object 0: TP={0}, FP={2}, FN={1}; object 1: TP={3}, FP={2}, FN={4}.
    assert result["per_object"][0]["recall"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert result["per_object"][1]["precision"] == pytest.approx(1 / 2, abs=TOLERANCE)
    assert result["mean_iou"] == pytest.approx(1 / 3, abs=TOLERANCE)
    assert result["object_count"] == 2 and result["objects_with_gt"] == 2


@pytest.mark.covers("T95")
def test_object_with_hidden_gt_but_nothing_occupied():
    """Object 0 has hidden GT, no occupancy: recall 0, precision None. Object 1 is perfect."""
    offsets, cells, inside = build_two_objects()
    truth, neighbourhood = mask(0, 3), mask(0, 1, 2, 3, 4)
    result = per_object_hidden_metrics(
        mask(3), truth, neighbourhood, mask(0, 3), offsets, cells, inside)
    assert result["per_object"][0]["recall"] == 0.0
    assert result["per_object"][0]["precision"] is None
    assert result["defined_counts"] == {"recall": 2, "precision": 1, "iou": 2}
    assert result["mean_precision"] == pytest.approx(1.0, abs=TOLERANCE)
    assert result["mean_recall"] == pytest.approx((0.0 + 1.0) / 2, abs=TOLERANCE)


@pytest.mark.covers("T95")
def test_object_without_hidden_gt_has_no_recall():
    """Object 1's GT cell 3 is not hidden: recall None, not counted in objects_with_gt."""
    offsets, cells, inside = build_two_objects()
    truth, neighbourhood = mask(0, 3), mask(0, 1, 2, 3, 4)
    result = per_object_hidden_metrics(
        mask(0), truth, neighbourhood, mask(0, 1), offsets, cells, inside)
    assert result["per_object"][1]["recall"] is None
    assert result["defined_counts"]["recall"] == 1 and result["objects_with_gt"] == 1
    assert result["mean_recall"] == pytest.approx(1.0, abs=TOLERANCE)
    with pytest.raises(ValueError):
        per_object_hidden_metrics(mask(0), mask(5), neighbourhood, None, offsets, cells, inside)
