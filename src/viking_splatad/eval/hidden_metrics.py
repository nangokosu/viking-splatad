"""Hidden-voxel recall, precision and IoU, pooled over cells and averaged over objects.

Notation (all sets are subsets of the V scored cells): O occupied, G ground truth,
N neighbourhood, H hidden. Truth is a subset of N. Pure numpy, no I/O.
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from viking_splatad.registry import transform

# Key order of the metrics averaged over objects (also the defined_counts keys).
OBJECT_MEAN_METRIC_NAMES = ("recall", "precision", "iou")
# Ground-truth label meaning "no annotated object at this cell".
NO_OBJECT_CLASS_ID = 0
# Hidden-space arrays copied into the cells archive: per-object box and class ids, and the CSR cell lists of T95.
OBJECT_ARRAY_KEYS = ("object_box_ids", "object_class_ids", "object_cell_offsets", "object_cells", "object_inside")


def _ratio_or_none(numerator, denominator):
    """Return numerator / denominator, or None when the denominator is zero.

    Args:
        numerator: Non-negative integer count.
        denominator: Non-negative integer count.

    Returns:
        float, or None if denominator == 0 (never NaN).
    """
    # A zero denominator means the metric is undefined, not zero.
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def class_group_masks(gt_class_ids_at_cells, group_class_ids, neighbourhood_by_class_id):
    """Build the neighbourhood and truth masks of a class group.

    neighbourhood = (union_{c in group} N_c) minus {cells whose GT label is nonzero and not in group}
    truth         = {cells with GT label in group} intersect neighbourhood
    The subtraction is the other-class mask: cells on another annotated class's GT
    count neither for nor against this group.

    Args:
        gt_class_ids_at_cells: (V,) int GT label per cell, 0 = no annotated object.
        group_class_ids: Iterable of class ids forming the group (one id, or several).
        neighbourhood_by_class_id: {class_id: (V,) bool neighbourhood of that class}.

    Returns:
        (neighbourhood, truth): two (V,) bool arrays.

    Raises:
        ValueError: If a group id has no entry in neighbourhood_by_class_id.
    """
    gt_labels = np.asarray(gt_class_ids_at_cells)
    group_ids = list(group_class_ids)
    # Every group member must come with its neighbourhood.
    missing_ids = [class_id for class_id in group_ids if class_id not in neighbourhood_by_class_id]
    if missing_ids:
        raise ValueError(f"no neighbourhood for class ids {missing_ids}")
    # Union of the member neighbourhoods.
    union_mask = np.zeros(gt_labels.shape, dtype=bool)
    for class_id in group_ids:
        union_mask |= np.asarray(neighbourhood_by_class_id[class_id], dtype=bool)
    # Other-class mask: annotated cells whose label is outside the group.
    in_group_mask = np.isin(gt_labels, group_ids)
    other_class_mask = (gt_labels != NO_OBJECT_CLASS_ID) & ~in_group_mask
    neighbourhood = union_mask & ~other_class_mask
    truth = in_group_mask & neighbourhood
    return neighbourhood, truth


@transform("T94", "Hidden-voxel recall, precision and IoU pooled over the scored cells")
def pooled_hidden_metrics(occupied, truth, neighbourhood, hidden=None):
    """Pooled recall, precision and IoU restricted to the hidden cells.

    With H the hidden set (all cells when hidden is None):
      TP = |O n G n H|,  FP = |(O n N n H) \\ G|,  FN = |(G n H) \\ O|
      recall = TP / |G n H|,  precision = TP / |O n N n H|,
      IoU = TP / |(O u G) n N n H| = TP / (TP + FP + FN)   (G is a subset of N).

    Args:
        occupied: (V,) bool occupancy O.
        truth: (V,) bool ground truth G, must be a subset of neighbourhood.
        neighbourhood: (V,) bool N.
        hidden: Optional (V,) bool H; None means every cell (whole-neighbourhood baseline).

    Returns:
        dict with true_positive, false_positive, false_negative, precision, recall, iou
        (float or None on a zero denominator), gt_cells = |G n H|, occupied_cells = |O n N n H|.

    Raises:
        ValueError: If the shapes differ or truth is not a subset of neighbourhood.
    """
    occupied = np.asarray(occupied, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    neighbourhood = np.asarray(neighbourhood, dtype=bool)
    # Hidden defaults to all cells, so the same formulas give the baseline.
    hidden = np.ones_like(occupied) if hidden is None else np.asarray(hidden, dtype=bool)
    if not (occupied.shape == truth.shape == neighbourhood.shape == hidden.shape):
        raise ValueError("occupied, truth, neighbourhood and hidden must share one shape")
    if np.any(truth & ~neighbourhood):
        raise ValueError("truth must be a subset of the neighbourhood")
    # Cell sets as defined above.
    hidden_truth = truth & hidden
    hidden_occupied_in_neighbourhood = occupied & neighbourhood & hidden
    true_positive = int(np.count_nonzero(hidden_occupied_in_neighbourhood & truth))
    false_positive = int(np.count_nonzero(hidden_occupied_in_neighbourhood & ~truth))
    false_negative = int(np.count_nonzero(hidden_truth & ~occupied))
    gt_cells = int(np.count_nonzero(hidden_truth))
    occupied_cells = int(np.count_nonzero(hidden_occupied_in_neighbourhood))
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": _ratio_or_none(true_positive, occupied_cells),
        "recall": _ratio_or_none(true_positive, gt_cells),
        "iou": _ratio_or_none(true_positive, true_positive + false_positive + false_negative),
        "gt_cells": gt_cells,
        "occupied_cells": occupied_cells,
    }


def csr_from_lists(list_of_index_arrays, list_of_inside_arrays):
    """Pack per-object cell lists into CSR arrays.

    Args:
        list_of_index_arrays: K int arrays, cell indices of each object's grown box.
        list_of_inside_arrays: K bool arrays aligned with the index arrays.

    Returns:
        (offsets (K+1,) int64, cells (nnz,) int64, inside (nnz,) bool); object k owns
        slice offsets[k]:offsets[k+1].

    Raises:
        ValueError: If the two lists differ in length or an index/inside pair differs in length.
    """
    if len(list_of_index_arrays) != len(list_of_inside_arrays):
        raise ValueError("index and inside lists must have the same length")
    lengths = np.array([len(index_array) for index_array in list_of_index_arrays], dtype=np.int64)
    inside_lengths = np.array([len(inside_array) for inside_array in list_of_inside_arrays], dtype=np.int64)
    if not np.array_equal(lengths, inside_lengths):
        raise ValueError("each inside array must match its index array in length")
    # offsets[k+1] - offsets[k] = size of object k.
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    if len(lengths) == 0:
        return offsets, np.zeros(0, dtype=np.int64), np.zeros(0, dtype=bool)
    cells = np.concatenate([np.asarray(index_array, dtype=np.int64) for index_array in list_of_index_arrays])
    inside = np.concatenate([np.asarray(inside_array, dtype=bool) for inside_array in list_of_inside_arrays])
    return offsets, cells, inside


@transform("T95", "Hidden-voxel recall, precision and IoU per object, averaged over the objects where each is defined")
def per_object_hidden_metrics(occupied, truth, neighbourhood, hidden,
                              object_cell_offsets, object_cells, object_inside):
    """Per-object T94 metrics and their mean over the objects where each is defined.

    For object k: N_k = cells_k n N; G_k = {c in N_k : inside box k and c in G};
    the T94 formulas are applied inside N_k (hidden restricted to N_k as well).
    Overlap decision: a cell in several N_k is scored in EACH of them, so an occupied
    off-GT cell shared by two objects is a false positive of both (double-counted by design).
    mean_m = (1 / |D_m|) * sum_{k in D_m} m_k with D_m = objects where metric m is not None.

    Args:
        occupied, truth, neighbourhood: (V,) bool as in T94.
        hidden: (V,) bool H, or None for the per-object whole-neighbourhood baseline.
        object_cell_offsets: (K+1,) int64 CSR offsets.
        object_cells: (nnz,) int64 cell indices of each object's grown box.
        object_inside: (nnz,) bool, cell centre inside the object's own box.

    Returns:
        dict: mean_recall, mean_precision, mean_iou (float or None), defined_counts
        {"recall","precision","iou"}, object_count K, objects_with_gt (objects with
        |G_k n H| >= 1), per_object (list of T94 dicts).

    Raises:
        ValueError: If shapes disagree or truth is not a subset of neighbourhood.
    """
    occupied = np.asarray(occupied, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    neighbourhood = np.asarray(neighbourhood, dtype=bool)
    if np.any(truth & ~neighbourhood):
        raise ValueError("truth must be a subset of the neighbourhood")
    hidden_all = None if hidden is None else np.asarray(hidden, dtype=bool)
    object_cell_offsets = np.asarray(object_cell_offsets, dtype=np.int64)
    object_cells = np.asarray(object_cells, dtype=np.int64)
    object_inside = np.asarray(object_inside, dtype=bool)
    if object_cells.shape != object_inside.shape:
        raise ValueError("object_cells and object_inside must be aligned")
    # The per-object slices index the full cell arrays, so every cell-level array must share truth's shape,
    # and the CSR offsets must start at 0 and end at the last object cell.
    if occupied.shape != truth.shape or neighbourhood.shape != truth.shape or (
            hidden_all is not None and hidden_all.shape != truth.shape):
        raise ValueError("occupied, truth, neighbourhood and hidden must share one shape")
    if (len(object_cell_offsets) == 0 or object_cell_offsets[0] != 0
            or object_cell_offsets[-1] != len(object_cells)):
        raise ValueError("object_cell_offsets must start at 0 and end at len(object_cells)")
    object_count = len(object_cell_offsets) - 1
    per_object = []
    for object_index in range(object_count):
        # Slice of object k in the CSR arrays.
        start, stop = object_cell_offsets[object_index], object_cell_offsets[object_index + 1]
        cell_indices = object_cells[start:stop]
        inside_box = object_inside[start:stop]
        # N_k: grown-box cells that survive the group neighbourhood (other-class mask).
        in_group_neighbourhood = neighbourhood[cell_indices]
        kept_cells = cell_indices[in_group_neighbourhood]
        kept_inside = inside_box[in_group_neighbourhood]
        # G_k: kept cells inside the box itself and in the truth.
        truth_k = truth[kept_cells] & kept_inside
        hidden_k = None if hidden_all is None else hidden_all[kept_cells]
        # Within N_k every kept cell is a neighbourhood cell.
        per_object.append(pooled_hidden_metrics(
            occupied[kept_cells], truth_k, np.ones(len(kept_cells), dtype=bool), hidden_k))
    means = {}
    defined_counts = {}
    for metric_name in OBJECT_MEAN_METRIC_NAMES:
        defined_values = [entry[metric_name] for entry in per_object if entry[metric_name] is not None]
        defined_counts[metric_name] = len(defined_values)
        means[metric_name] = float(np.mean(defined_values)) if defined_values else None
    return {
        "mean_recall": means["recall"],
        "mean_precision": means["precision"],
        "mean_iou": means["iou"],
        "defined_counts": defined_counts,
        "object_count": object_count,
        "objects_with_gt": sum(1 for entry in per_object if entry["gt_cells"] >= 1),
        "per_object": per_object,
    }


def threshold_key(threshold: float) -> str:
    """The JSON key of a P2 threshold: ``repr(float(threshold))``, e.g. "0.5".

    Args:
        threshold: P2 occupancy cut.

    Returns:
        Key text, identical for the same float on every machine.
    """
    return repr(float(threshold))


def select_object_lists(object_class_ids: np.ndarray, object_cell_offsets: np.ndarray, object_cells: np.ndarray,
                        object_inside: np.ndarray, class_ids: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR cell lists of the objects whose class is in ``class_ids``.

    Args:
        object_class_ids: (K,) class id of each object.
        object_cell_offsets: (K+1,) CSR offsets. object_cells: (nnz,) cell rows. object_inside: (nnz,) bool.
        class_ids: Class ids to keep.

    Returns:
        (offsets, cells, inside) of the kept objects, in their original order.
    """
    kept_objects = np.flatnonzero(np.isin(object_class_ids, list(class_ids)))
    # Slice each kept object's block out of the CSR arrays and repack.
    slices = [slice(object_cell_offsets[object_index], object_cell_offsets[object_index + 1]) for object_index in kept_objects]
    return csr_from_lists([object_cells[block] for block in slices], [object_inside[block] for block in slices])


def without_object_list(per_object_result: Dict[str, Any]) -> Dict[str, Any]:
    """A T95 result minus its long ``per_object`` list (the report keeps the means and counts).

    Args:
        per_object_result: ``per_object_hidden_metrics`` output.

    Returns:
        The same dict without ``per_object``.
    """
    return {key: value for key, value in per_object_result.items() if key != "per_object"}


def hidden_class_report(class_ids: Sequence[int], neighbourhood: np.ndarray, truth: np.ndarray, hidden: np.ndarray,
                        p1_hit: np.ndarray, occupancy: np.ndarray, p2_thresholds: Sequence[float],
                        objects: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """The hidden-voxel block of one class or group (T94 pooled, T95 per object).

    With O the occupied cells, G truth, N neighbourhood, H hidden: hidden metrics use
    recall_H = |O n G n H| / |G n H|, precision_H = |O n G n H| / |O n N n H|,
    IoU_H = |O n G n H| / |(O u G) n N n H|; the baseline is the same with H = every cell. O is
    P1's ``p1_hit`` or P2's ``occupancy >= t`` for each threshold t. Objects are the hidden-space
    objects of the entry's classes.

    Args:
        class_ids: The entry's class ids.
        neighbourhood: (V,) bool N (other-class mask applied). truth: (V,) bool G. hidden: (V,) bool H.
        p1_hit: (V,) bool P1 occupied cells. occupancy: (V,) T81 probabilities.
        p2_thresholds: P2 occupancy cuts.
        objects: Hidden-space ``objects`` arrays (OBJECT_ARRAY_KEYS).

    Returns:
        {"context": {hidden_gt_cells, objects_with_hidden_gt, object_count},
         "P1": block, "P2": {threshold_key(t): block}}, a block being {pooled, per_object,
         baseline_pooled, baseline_per_object}.
    """
    # Step 1: the entry's own objects.
    offsets, cells, inside = select_object_lists(objects["object_class_ids"], objects["object_cell_offsets"],
                                                 objects["object_cells"], objects["object_inside"], class_ids)

    def metric_block(occupied: np.ndarray) -> Dict[str, Any]:
        """Hidden and baseline metrics, pooled and per object, for one occupied set.

        Args:
            occupied: (V,) bool O.

        Returns:
            The four-key block (per-object lists dropped).
        """
        return {"pooled": pooled_hidden_metrics(occupied, truth, neighbourhood, hidden),
                "per_object": without_object_list(per_object_hidden_metrics(occupied, truth, neighbourhood, hidden, offsets, cells, inside)),
                "baseline_pooled": pooled_hidden_metrics(occupied, truth, neighbourhood),
                "baseline_per_object": without_object_list(per_object_hidden_metrics(occupied, truth, neighbourhood, None, offsets, cells, inside))}

    # Step 2: P1 and each P2 threshold.
    p1_block = metric_block(np.asarray(p1_hit, dtype=bool))
    p2_blocks = {threshold_key(threshold): metric_block(np.asarray(occupancy) >= float(threshold)) for threshold in p2_thresholds}
    # Step 3: context = |G n H| and the objects having at least one hidden GT cell (occupancy-independent).
    context = {"hidden_gt_cells": p1_block["pooled"]["gt_cells"],
               "objects_with_hidden_gt": p1_block["per_object"]["objects_with_gt"],
               "object_count": p1_block["per_object"]["object_count"]}
    return {"context": context, "P1": p1_block, "P2": p2_blocks}
