"""Occupancy evaluation metrics between predicted and ground-truth grids (T33)."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np

from viking_splatad.registry import transform

# Result of a 0/0 metric when the situation is "perfect agreement" (nothing to find
# and nothing found) vs. "total miss". Fixed by definition of the conventions below.
PERFECT_SCORE = 1.0
ZERO_SCORE = 0.0


@dataclass(frozen=True)
class OccupancyMetrics:
    """Confusion counts and derived scores for one boolean comparison.

    Attributes:
        true_positives: Voxels occupied in both.
        false_positives: Predicted occupied, GT empty.
        false_negatives: GT occupied, predicted empty.
        iou: TP / (TP + FP + FN).
        precision: TP / (TP + FP).
        recall: TP / (TP + FN).
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    iou: float
    precision: float
    recall: float


@dataclass(frozen=True)
class OccupancyEvaluation:
    """Overall metrics plus optional per-class metrics (empty if no class grids)."""

    overall: OccupancyMetrics
    per_class: Dict[int, OccupancyMetrics] = field(default_factory=dict)


def compute_binary_metrics(predicted: np.ndarray, ground_truth: np.ndarray) -> OccupancyMetrics:
    """Compute IoU / precision / recall for two boolean grids.

    0/0 conventions (documented, deterministic):
      * IoU with empty union (both empty) = 1.0 (perfect agreement).
      * Precision with no predicted positives = 1.0 if GT is also empty (no
        errors possible) else 0.0.
      * Recall with no GT positives = 1.0 if nothing was predicted else 0.0.

    Args:
        predicted: Boolean grid.
        ground_truth: Boolean grid, same shape.

    Returns:
        OccupancyMetrics.
    """
    # Confusion counts from elementwise boolean algebra.
    true_positives = int(np.count_nonzero(predicted & ground_truth))
    false_positives = int(np.count_nonzero(predicted & ~ground_truth))
    false_negatives = int(np.count_nonzero(~predicted & ground_truth))
    # |P union G| = TP + FP + FN; |P| = TP + FP; |G| = TP + FN.
    union = true_positives + false_positives + false_negatives
    predicted_total = true_positives + false_positives
    ground_truth_total = true_positives + false_negatives
    # IoU = |P and G| / |P union G|; empty union -> perfect agreement.
    iou = true_positives / union if union else PERFECT_SCORE
    if predicted_total:
        precision = true_positives / predicted_total
    else:
        precision = PERFECT_SCORE if ground_truth_total == 0 else ZERO_SCORE
    if ground_truth_total:
        recall = true_positives / ground_truth_total
    else:
        recall = PERFECT_SCORE if predicted_total == 0 else ZERO_SCORE
    return OccupancyMetrics(true_positives, false_positives, false_negatives, iou, precision, recall)


@transform("T33", "Occupancy evaluation: IoU, precision, recall (overall and per class)")
def evaluate_occupancy(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    predicted_classes: Optional[np.ndarray] = None,
    ground_truth_classes: Optional[np.ndarray] = None,
    class_ids: Optional[Sequence[int]] = None,
) -> OccupancyEvaluation:
    """Compare predicted and GT occupancy.

    Per-class metrics treat class c as the boolean grid ``class_grid == c`` on both
    sides (so a voxel predicted with the wrong class counts as FP for one class and
    FN for another). They are computed only when both class grids are given.

    Args:
        predicted: Boolean predicted occupancy.
        ground_truth: Boolean GT occupancy, same shape.
        predicted_classes: Optional integer class grid (0 = empty), same shape.
        ground_truth_classes: Optional integer class grid, same shape.
        class_ids: Classes to report; default all non-zero ids present in either grid.

    Returns:
        OccupancyEvaluation.

    Raises:
        ValueError: If any provided grids differ in shape.
    """
    # Normalise inputs and validate that every supplied grid shares one shape.
    predicted = np.asarray(predicted, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    provided_grids = [predicted, ground_truth]
    if predicted_classes is not None and ground_truth_classes is not None:
        provided_grids += [np.asarray(predicted_classes), np.asarray(ground_truth_classes)]
    if any(grid.shape != predicted.shape for grid in provided_grids):
        raise ValueError(f"Grid shape mismatch: {[grid.shape for grid in provided_grids]}")

    overall = compute_binary_metrics(predicted, ground_truth)
    per_class: Dict[int, OccupancyMetrics] = {}
    if predicted_classes is not None and ground_truth_classes is not None:
        # Default report set: every non-empty class id appearing in either grid.
        if class_ids is None:
            present_ids = np.union1d(np.unique(predicted_classes), np.unique(ground_truth_classes))
            class_ids = [int(class_id) for class_id in present_ids if class_id != 0]
        for class_id in class_ids:
            per_class[int(class_id)] = compute_binary_metrics(
                np.asarray(predicted_classes) == class_id,
                np.asarray(ground_truth_classes) == class_id,
            )
    return OccupancyEvaluation(overall, per_class)
