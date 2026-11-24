"""How ground-truth voxels move from the old test-sweep labels to the session-wide hidden space H.

Evaluation diagnostics only (never a SplatAD input). Pure functions over arrays, so every count can
be checked by hand; the I/O lives in scripts/25_hidden_label_transitions.py.

Vocabulary: "old" = the cells archive of an earlier scripts/19 run (T78 evidence from the test
sweeps only, exclusive label by priority); "new" = ``hidden_space.npz`` (evidence from all sweeps
with a range margin; ``hidden`` = never seen, blocked, never passed through).
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.eval.session_hidden_space import nearest_horizontal_distance  # noqa: F401  (re-exported for scripts/25)

from viking_splatad.eval.hidden_geometry import (
    BLOCKED_COLUMN,
    EVIDENCE_NAMES,
    PASSED_THROUGH_COLUMN,
    SEEN_COLUMN,
    exclusive_evidence_labels,
)
from viking_splatad.labels.cuboids import compute_cuboid_geometry

# Label of a new cell that has no counterpart in the old archive.
NOT_IN_OLD_LABEL = "not_in_old_cells"
# Order of the transition-table rows: today's labels, then the absent case.
OLD_LABEL_NAMES = tuple(EVIDENCE_NAMES) + (NOT_IN_OLD_LABEL,)
# Cause names of the new evidence flags that decide H (exclusive, priority seen > passed > blocked).
CAUSE_HIDDEN = "hidden"
CAUSE_SEEN = "seen"
CAUSE_POROUS = "passed_through_not_seen"
CAUSE_UNOBSERVED = "never_blocked_nor_seen_nor_passed"
CAUSE_NAMES = (CAUSE_HIDDEN, CAUSE_SEEN, CAUSE_POROUS, CAUSE_UNOBSERVED)
# Height bins: layers 0, 1, 2 individually; every layer >= this value shares the last bin.
LAST_HEIGHT_BIN_LAYER = 3
HEIGHT_BIN_NAMES = tuple(str(layer) for layer in range(LAST_HEIGHT_BIN_LAYER)) + (f">={LAST_HEIGHT_BIN_LAYER}",)
# Hidden GT cells lying in no object's box (no height reference).
NO_BOX_BIN_NAME = "no_box"
# Horizontal distance bin edges (m) to the nearest sweep origin; the last bin is open-ended.
DISTANCE_BIN_EDGES_M = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0, np.inf)
# Column of the vertical axis in grid indices and centres (map frame: up = +z).
VERTICAL_AXIS = 2


def distance_bin_names() -> Tuple[str, ...]:
    """Readable names of the distance bins from ``DISTANCE_BIN_EDGES_M``.

    Returns:
        One "[lo, hi)" name per bin.
    """
    return tuple(f"[{lower:g}, {upper:g}) m" for lower, upper in zip(DISTANCE_BIN_EDGES_M[:-1], DISTANCE_BIN_EDGES_M[1:]))


def _grid_keys(cell_indices: np.ndarray, index_minimum: np.ndarray, key_shape: Tuple[int, int, int]) -> np.ndarray:
    """Flatten (V,3) grid indices to unique int64 keys.

    Args:
        cell_indices: (V,3) integer grid indices.
        index_minimum: (3,) per-axis offset subtracted first (so keys are non-negative).
        key_shape: Shape that covers every shifted index.

    Returns:
        (V,) int64 keys, equal iff the grid indices are equal.
    """
    return np.ravel_multi_index(tuple((cell_indices - index_minimum).T), key_shape).astype(np.int64)


def match_cells(old_cell_indices: np.ndarray, new_cell_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match two cell lists by exact grid index.

    Args:
        old_cell_indices: (Vold,3) grid indices of the old archive (unique rows).
        new_cell_indices: (Vnew,3) grid indices of the new set (unique rows).

    Returns:
        (old_row_for_new (Vnew,) int64 = row of the same cell in the old list or -1,
        new_only (Vnew,) bool = new cells absent from the old list,
        old_only (Vold,) bool = old cells absent from the new list).

    Raises:
        ValueError: If an index array is not (V,3) or has duplicate rows.
    """
    for name, indices in (("old", old_cell_indices), ("new", new_cell_indices)):
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError(f"{name} cell indices must be (V,3)")
    # Step 1: one shared key space (shift by the joint minimum, shape from the joint maximum).
    joint = np.concatenate([old_cell_indices, new_cell_indices], axis=0)
    index_minimum = joint.min(axis=0)
    key_shape = tuple(int(extent) for extent in (joint.max(axis=0) - index_minimum + 1))
    old_keys = _grid_keys(old_cell_indices, index_minimum, key_shape)
    new_keys = _grid_keys(new_cell_indices, index_minimum, key_shape)
    if len(np.unique(old_keys)) != len(old_keys) or len(np.unique(new_keys)) != len(new_keys):
        raise ValueError("cell index lists must not contain duplicate rows")
    # Step 2: binary search of every new key in the sorted old keys.
    old_order = np.argsort(old_keys, kind="stable")
    sorted_old_keys = old_keys[old_order]
    insert_position = np.searchsorted(sorted_old_keys, new_keys)
    # A key above every old key gets position len(old); clip it to a valid row, the equality test rejects it.
    clipped_position = np.minimum(insert_position, len(sorted_old_keys) - 1)
    found = sorted_old_keys[clipped_position] == new_keys
    old_row_for_new = np.where(found, old_order[clipped_position], -1).astype(np.int64)
    # Step 3: masks of the cells present on one side only.
    old_only = np.ones(len(old_keys), dtype=bool)
    old_only[old_row_for_new[found]] = False
    return old_row_for_new, ~found, old_only


def old_gt_cells_not_scored_now(old_truth: np.ndarray, old_row_for_new: np.ndarray, gt_mask_new: np.ndarray) -> int:
    """Count old GT cells with no GT counterpart in the new class set.

    Args:
        old_truth: (Vold,) bool, old GT membership of the class.
        old_row_for_new: (Vnew,) match from ``match_cells``.
        gt_mask_new: (Vnew,) bool, new GT membership of the class.

    Returns:
        Number of old GT cells that are absent from the new set or not GT-of-class there.
    """
    # Step 1: old rows that are GT in the new set (matched and flagged).
    still_gt_rows = old_row_for_new[gt_mask_new & (old_row_for_new >= 0)]
    still_gt = np.zeros(len(old_truth), dtype=bool)
    still_gt[still_gt_rows] = True
    # Step 2: old GT cells outside that set.
    return int(np.count_nonzero(old_truth & ~still_gt))


def _share(numerator: int, denominator: int) -> Optional[float]:
    """Fraction, or None when the denominator is zero.

    Args:
        numerator: Count in the part.
        denominator: Count in the whole.

    Returns:
        numerator / denominator, or None for an empty whole.
    """
    return None if denominator == 0 else numerator / denominator


def label_transition_table(
    old_labels_for_new: np.ndarray, hidden_new: np.ndarray, gt_mask_new: np.ndarray, old_only_gt_count: int,
) -> Dict[str, Any]:
    """Counts of today's exclusive label x {hidden, not_hidden} over the GT cells of one class.

    Args:
        old_labels_for_new: (Vnew,) index into ``OLD_LABEL_NAMES`` of each new cell (the last index
            = ``NOT_IN_OLD_LABEL``).
        hidden_new: (Vnew,) bool, new H.
        gt_mask_new: (Vnew,) bool, GT cells of the class in the new set.
        old_only_gt_count: Old GT cells not scored now.

    Returns:
        {"rows": {label: {"hidden", "not_hidden", "total", "hidden_share"}}, "total_gt", "total_hidden",
        "hidden_share" (None when no GT), "old_only_gt_cells"}.
    """
    # Step 1: the class's cells only.
    class_labels = old_labels_for_new[gt_mask_new]
    class_hidden = hidden_new[gt_mask_new]
    # Step 2: per label, hidden and not-hidden counts; share = hidden / (hidden + not hidden).
    rows: Dict[str, Dict[str, Any]] = {}
    for label_index, label_name in enumerate(OLD_LABEL_NAMES):
        in_label = class_labels == label_index
        hidden_count = int(np.count_nonzero(in_label & class_hidden))
        not_hidden_count = int(np.count_nonzero(in_label & ~class_hidden))
        rows[label_name] = {
            "hidden": hidden_count, "not_hidden": not_hidden_count, "total": hidden_count + not_hidden_count,
            "hidden_share": _share(hidden_count, hidden_count + not_hidden_count)}
    total_gt = int(np.count_nonzero(gt_mask_new))
    total_hidden = int(np.count_nonzero(class_hidden))
    return {"rows": rows, "total_gt": total_gt, "total_hidden": total_hidden,
            "hidden_share": _share(total_hidden, total_gt), "old_only_gt_cells": int(old_only_gt_count)}


def old_labels_for_new_cells(old_evidence: np.ndarray, old_row_for_new: np.ndarray) -> np.ndarray:
    """Today's exclusive label of every new cell (or the absent label).

    Args:
        old_evidence: (Vold,4) old T78 flags.
        old_row_for_new: (Vnew,) match from ``match_cells`` (-1 = absent).

    Returns:
        (Vnew,) int index into ``OLD_LABEL_NAMES``.
    """
    old_labels = exclusive_evidence_labels(old_evidence)
    absent_index = OLD_LABEL_NAMES.index(NOT_IN_OLD_LABEL)
    # Matched cells take their old label; -1 rows read a dummy row 0 that np.where then discards.
    return np.where(old_row_for_new >= 0, old_labels[np.maximum(old_row_for_new, 0)], absent_index)


def hidden_cause_breakdown(evidence_new: np.ndarray, gt_mask_new: np.ndarray) -> Dict[str, int]:
    """Exclusive flag combination of each GT cell of the class, in the order that decides H.

    H = NOT seen AND blocked AND NOT passed_through (T91), so a cell is: seen (any other flags) ->
    passed through without seen (porous) -> blocked, neither seen nor passed (hidden, exactly H; may
    also carry no_return) -> none of the three (no_return only, or nothing recorded).

    Args:
        evidence_new: (Vnew,4) new T78 flags in seen/blocked/no_return/passed_through order.
        gt_mask_new: (Vnew,) bool, GT cells of the class.

    Returns:
        {cause: count} for ``CAUSE_NAMES``; the counts sum to the number of GT cells.
    """
    # Step 1: flags of the class's GT cells only.
    flags = np.asarray(evidence_new, dtype=bool)[gt_mask_new]
    # Step 2: peel the causes off in priority order, so each cell lands in exactly one.
    seen = flags[:, SEEN_COLUMN]
    passed_through = flags[:, PASSED_THROUGH_COLUMN] & ~seen
    hidden = flags[:, BLOCKED_COLUMN] & ~seen & ~passed_through
    unobserved = ~(seen | passed_through | hidden)
    return {CAUSE_HIDDEN: int(hidden.sum()), CAUSE_SEEN: int(seen.sum()),
            CAUSE_POROUS: int(passed_through.sum()), CAUSE_UNOBSERVED: int(unobserved.sum())}


def object_bottom_heights(object_box_ids: np.ndarray, cuboids: Sequence[Any]) -> np.ndarray:
    """Lowest corner height of each object's box.

    Args:
        object_box_ids: (K,) box ids of the objects.
        cuboids: All GT cuboids (``Cuboid``).

    Returns:
        (K,) float64 minimum corner z (map frame) per object.

    Raises:
        KeyError: If an object's box id is not among the cuboids.
    """
    bottom_by_box_id = {int(cuboid.box_id): float(compute_cuboid_geometry(cuboid).corners[:, VERTICAL_AXIS].min())
                        for cuboid in cuboids}
    return np.asarray([bottom_by_box_id[int(box_id)] for box_id in object_box_ids], dtype=np.float64)


def height_layer_breakdown(
    centres: np.ndarray, target_mask: np.ndarray, voxel_size: float, object_offsets: np.ndarray,
    object_cells: np.ndarray, object_inside: np.ndarray, object_bottom_z: np.ndarray,
    object_class_ids: np.ndarray, class_ids: Sequence[int],
) -> Dict[str, int]:
    """Layer of each target cell above the bottom of the box it lies in.

    layer = max(0, floor((z_centre - z_bottom) / voxel_size)); layer 0 = the centre is within the
    lowest voxel height of the box. A cell in several boxes of the class takes its lowest layer.
    Only boxes whose class id is in ``class_ids`` and cells inside the original box count.

    Args:
        centres: (V,3) cell centres, map frame.
        target_mask: (V,) bool, cells to bin (e.g. hidden GT cells of the class).
        voxel_size: Edge length (m).
        object_offsets: (K+1,) CSR offsets.
        object_cells: (nnz,) cell rows per object.
        object_inside: (nnz,) bool, centre inside the original box.
        object_bottom_z: (K,) bottom height per object.
        object_class_ids: (K,) class id per object.
        class_ids: Class ids of the class or group.

    Returns:
        {bin name: count} over ``HEIGHT_BIN_NAMES`` plus ``NO_BOX_BIN_NAME``; sums to the target count.
    """
    # Step 1: lowest layer per cell over its (inside, class) memberships; sentinel = no box.
    no_box_layer = np.iinfo(np.int64).max
    lowest_layer = np.full(len(centres), no_box_layer, dtype=np.int64)
    object_count = len(object_class_ids)
    # Owning object of every CSR entry (object k repeated offsets[k+1] - offsets[k] times).
    entry_object = np.repeat(np.arange(object_count), np.diff(object_offsets))
    entry_valid = object_inside & np.isin(object_class_ids[entry_object], list(class_ids)) & target_mask[object_cells]
    entry_cells = object_cells[entry_valid]
    entry_height = centres[entry_cells, VERTICAL_AXIS] - object_bottom_z[entry_object[entry_valid]]
    entry_layer = np.maximum(0, np.floor(entry_height / voxel_size)).astype(np.int64)
    np.minimum.at(lowest_layer, entry_cells, entry_layer)
    # Step 2: bins 0 .. LAST-1 individually, everything above in the last bin.
    target_layers = lowest_layer[target_mask]
    counts = {name: 0 for name in HEIGHT_BIN_NAMES}
    for layer in range(LAST_HEIGHT_BIN_LAYER):
        counts[HEIGHT_BIN_NAMES[layer]] = int(np.count_nonzero(target_layers == layer))
    counts[HEIGHT_BIN_NAMES[-1]] = int(np.count_nonzero((target_layers >= LAST_HEIGHT_BIN_LAYER) & (target_layers != no_box_layer)))
    counts[NO_BOX_BIN_NAME] = int(np.count_nonzero(target_layers == no_box_layer))
    return counts


def seen_above_count(cell_indices: np.ndarray, seen_new: np.ndarray, target_mask: np.ndarray) -> int:
    """Target cells with a seen scored cell higher in the same (x, y) grid column.

    A proxy for "under a visible surface" (underground, under a canopy).

    Args:
        cell_indices: (V,3) grid indices.
        seen_new: (V,) bool, new seen flag.
        target_mask: (V,) bool, cells to test (e.g. hidden GT cells).

    Returns:
        Number of target cells for which some seen cell has a larger z index in the same column.
    """
    # Step 1: column id per cell, and the highest seen z index per column (-1 = none; grid indices are >= 0).
    columns, column_of_cell = np.unique(cell_indices[:, :VERTICAL_AXIS], axis=0, return_inverse=True)
    column_of_cell = column_of_cell.reshape(-1)
    highest_seen = np.full(len(columns), -1, dtype=np.int64)
    np.maximum.at(highest_seen, column_of_cell[seen_new], cell_indices[seen_new, VERTICAL_AXIS])
    # Step 2: a target is covered when that maximum is above its own index.
    return int(np.count_nonzero(highest_seen[column_of_cell[target_mask]] > cell_indices[target_mask, VERTICAL_AXIS]))


def distance_breakdown(distances: np.ndarray, hidden_gt_mask: np.ndarray, gt_mask: np.ndarray) -> Dict[str, Dict[str, int]]:
    """Counts of hidden GT cells and of all GT cells per distance bin.

    Args:
        distances: (V,) nearest-sweep horizontal distance per cell.
        hidden_gt_mask: (V,) bool, hidden GT cells of the class.
        gt_mask: (V,) bool, all GT cells of the class.

    Returns:
        {bin name: {"hidden_gt", "all_gt"}} over ``distance_bin_names()``.
    """
    # Step 1: bin index by the edges (right-open; the last edge is infinity).
    bin_index = np.searchsorted(np.asarray(DISTANCE_BIN_EDGES_M), distances, side="right") - 1
    result: Dict[str, Dict[str, int]] = {}
    for position, name in enumerate(distance_bin_names()):
        in_bin = bin_index == position
        result[name] = {"hidden_gt": int(np.count_nonzero(in_bin & hidden_gt_mask)),
                        "all_gt": int(np.count_nonzero(in_bin & gt_mask))}
    return result


def object_counts(
    object_offsets: np.ndarray, object_cells: np.ndarray, object_inside: np.ndarray,
    object_class_ids: np.ndarray, class_ids: Sequence[int], gt_mask: np.ndarray, hidden_gt_mask: np.ndarray,
) -> Dict[str, int]:
    """Objects of the class, with GT cells inside, and with hidden GT cells inside.

    Args:
        object_offsets: (K+1,) CSR offsets.
        object_cells: (nnz,) cell rows.
        object_inside: (nnz,) bool, centre inside the original box.
        object_class_ids: (K,) class id per object.
        class_ids: Class ids of the class or group.
        gt_mask: (V,) bool, GT cells of the class.
        hidden_gt_mask: (V,) bool, hidden GT cells of the class.

    Returns:
        {"objects", "with_gt_inside", "with_hidden_gt_inside"}.
    """
    object_count = len(object_class_ids)
    # Owning object of every CSR entry, and which objects belong to the class.
    entry_object = np.repeat(np.arange(object_count), np.diff(object_offsets))
    in_class = np.isin(object_class_ids, list(class_ids))
    # Step 1: per object, any inside entry whose cell is (hidden) GT.
    with_gt = np.zeros(object_count, dtype=bool)
    with_hidden = np.zeros(object_count, dtype=bool)
    with_gt[entry_object[object_inside & gt_mask[object_cells]]] = True
    with_hidden[entry_object[object_inside & hidden_gt_mask[object_cells]]] = True
    return {"objects": int(in_class.sum()), "with_gt_inside": int(np.count_nonzero(with_gt & in_class)),
            "with_hidden_gt_inside": int(np.count_nonzero(with_hidden & in_class))}
