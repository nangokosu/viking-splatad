"""Unit tests for T33 (occupancy evaluation metrics)."""

import numpy as np
import pytest

from viking_splatad.labels.eval_occupancy import evaluate_occupancy


@pytest.mark.covers("T33")
def test_binary_metrics_exact_values():
    """Hand-built 1x2x4 grids: TP=2, FP=1, FN=2 -> IoU 2/5, P 2/3, R 1/2."""
    predicted = np.array([[[1, 1, 1, 0], [0, 0, 0, 0]]], dtype=bool)
    ground_truth = np.array([[[1, 1, 0, 0], [1, 1, 0, 0]]], dtype=bool)
    evaluation = evaluate_occupancy(predicted, ground_truth)
    assert evaluation.overall.true_positives == 2
    assert evaluation.overall.false_positives == 1
    assert evaluation.overall.false_negatives == 2
    assert evaluation.overall.iou == pytest.approx(2 / 5)
    assert evaluation.overall.precision == pytest.approx(2 / 3)
    assert evaluation.overall.recall == pytest.approx(1 / 2)
    assert evaluation.per_class == {}


@pytest.mark.covers("T33")
def test_zero_over_zero_conventions():
    """Both empty -> all 1.0; GT only -> precision 0 (nothing predicted), recall 0."""
    empty = np.zeros((2, 2, 2), dtype=bool)
    full = np.ones((2, 2, 2), dtype=bool)
    both_empty = evaluate_occupancy(empty, empty).overall
    assert (both_empty.iou, both_empty.precision, both_empty.recall) == (1.0, 1.0, 1.0)
    missed_everything = evaluate_occupancy(empty, full).overall
    assert (missed_everything.iou, missed_everything.precision, missed_everything.recall) == (0.0, 0.0, 0.0)
    hallucinated = evaluate_occupancy(full, empty).overall
    assert (hallucinated.iou, hallucinated.precision, hallucinated.recall) == (0.0, 0.0, 0.0)


@pytest.mark.covers("T33")
def test_per_class_metrics_exact_values():
    """Class grids: class 1 TP=1 FP=1 FN=0; class 2 TP=1 FP=0 FN=1; class 3 absent in both."""
    predicted_classes = np.array([1, 1, 2, 0])
    ground_truth_classes = np.array([1, 0, 2, 2])
    evaluation = evaluate_occupancy(
        predicted_classes > 0, ground_truth_classes > 0,
        predicted_classes=predicted_classes, ground_truth_classes=ground_truth_classes,
        class_ids=(1, 2, 3),
    )
    class_one, class_two, class_three = (evaluation.per_class[class_id] for class_id in (1, 2, 3))
    assert class_one.iou == pytest.approx(1 / 2)
    assert class_one.precision == pytest.approx(1 / 2)
    assert class_one.recall == pytest.approx(1.0)
    assert class_two.iou == pytest.approx(1 / 2)
    assert class_two.precision == pytest.approx(1.0)
    assert class_two.recall == pytest.approx(1 / 2)
    assert class_three.iou == 1.0  # 0/0 convention: absent in both
    # Overall: pred occupied {0,1,2}, GT occupied {0,2,3} -> TP 2, FP 1, FN 1.
    assert evaluation.overall.iou == pytest.approx(2 / 4)


@pytest.mark.covers("T33")
def test_shape_mismatch_raises():
    """Grids of different shape cannot be compared."""
    with pytest.raises(ValueError, match="shape"):
        evaluate_occupancy(np.zeros((2, 2), dtype=bool), np.zeros((3, 2), dtype=bool))
