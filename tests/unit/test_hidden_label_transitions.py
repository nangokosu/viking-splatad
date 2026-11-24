"""Known-answer tests of the old-label -> hidden-space transition diagnostics (hand-built tiny archives)."""

import numpy as np

from viking_splatad.eval.hidden_geometry import exclusive_evidence_labels, hidden_space_mask
from viking_splatad.eval.hidden_label_transitions import (
    CAUSE_HIDDEN,
    CAUSE_POROUS,
    CAUSE_SEEN,
    CAUSE_UNOBSERVED,
    NO_BOX_BIN_NAME,
    OLD_LABEL_NAMES,
    distance_bin_names,
    distance_breakdown,
    height_layer_breakdown,
    hidden_cause_breakdown,
    label_transition_table,
    match_cells,
    nearest_horizontal_distance,
    object_counts,
    old_gt_cells_not_scored_now,
    old_labels_for_new_cells,
    seen_above_count,
)

# Evidence column order: seen, blocked, no_return, passed_through.
SEEN_ONLY = [1, 0, 0, 0]
BLOCKED_ONLY = [0, 1, 0, 0]
NOTHING = [0, 0, 0, 0]
PASSED_ONLY = [0, 0, 0, 1]
BLOCKED_AND_PASSED = [0, 1, 0, 1]
SEEN_AND_BLOCKED = [1, 1, 0, 0]

# Old cells o0..o3 and new cells n0..n4 (n2, n4 are new only; o3 is old only).
OLD_CELL_INDICES = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [5, 5, 5]])
OLD_EVIDENCE = np.array([SEEN_ONLY, BLOCKED_ONLY, NOTHING, PASSED_ONLY])
NEW_CELL_INDICES = np.array([[1, 0, 0], [0, 0, 1], [2, 0, 0], [0, 0, 0], [0, 0, 2]])
NEW_EVIDENCE = np.array([BLOCKED_ONLY, BLOCKED_AND_PASSED, BLOCKED_ONLY, SEEN_AND_BLOCKED, NOTHING])
GT_MASK_NEW = np.array([True, True, True, True, False])
# Hand-derived: old rows of n0..n4 are o2, o1, absent, o0, absent.
EXPECTED_OLD_ROW_FOR_NEW = np.array([2, 1, -1, 0, -1])


def test_match_cells_by_grid_index():
    old_row_for_new, new_only, old_only = match_cells(OLD_CELL_INDICES, NEW_CELL_INDICES)
    assert old_row_for_new.tolist() == EXPECTED_OLD_ROW_FOR_NEW.tolist()
    assert new_only.tolist() == [False, False, True, False, True]
    assert old_only.tolist() == [False, False, False, True]


def test_label_transition_counts_and_shares():
    hidden_new = hidden_space_mask(NEW_EVIDENCE)
    # Hand check of H: n0 blocked only -> hidden; n1 passed -> no; n2 blocked only -> hidden; n3 seen -> no.
    assert hidden_new.tolist() == [True, False, True, False, False]
    old_labels = old_labels_for_new_cells(OLD_EVIDENCE, EXPECTED_OLD_ROW_FOR_NEW)
    # n0 -> o2 unobserved (4); n1 -> o1 blocked (1); n2 absent (5); n3 -> o0 seen (0); n4 absent (5).
    assert old_labels.tolist() == [4, 1, 5, 0, 5]
    old_truth = np.array([True, False, True, True])
    old_only_gt = old_gt_cells_not_scored_now(old_truth, EXPECTED_OLD_ROW_FOR_NEW, GT_MASK_NEW)
    # o0 and o2 are GT now (n3, n0); o3 (old only, truth) is not.
    assert old_only_gt == 1
    table = label_transition_table(old_labels, hidden_new, GT_MASK_NEW, old_only_gt)
    assert table["total_gt"] == 4 and table["total_hidden"] == 2 and table["hidden_share"] == 0.5
    rows = table["rows"]
    assert (rows["unobserved"]["hidden"], rows["unobserved"]["not_hidden"], rows["unobserved"]["hidden_share"]) == (1, 0, 1.0)
    assert (rows["blocked"]["hidden"], rows["blocked"]["not_hidden"], rows["blocked"]["hidden_share"]) == (0, 1, 0.0)
    assert (rows["not_in_old_cells"]["hidden"], rows["not_in_old_cells"]["total"]) == (1, 1)
    assert (rows["seen"]["hidden"], rows["seen"]["not_hidden"]) == (0, 1)
    # Empty label rows give no share.
    assert rows["no_return"]["hidden_share"] is None and rows["passed_through"]["total"] == 0
    assert table["old_only_gt_cells"] == 1
    assert list(rows) == list(OLD_LABEL_NAMES)
    assert exclusive_evidence_labels(OLD_EVIDENCE).tolist() == [0, 1, 4, 3]


def test_cause_breakdown_is_exclusive_and_sums_to_gt():
    causes = hidden_cause_breakdown(NEW_EVIDENCE, GT_MASK_NEW)
    # n0, n2 hidden; n1 passed without seen; n3 seen (blocked too); n4 is not GT.
    assert causes == {CAUSE_HIDDEN: 2, CAUSE_SEEN: 1, CAUSE_POROUS: 1, CAUSE_UNOBSERVED: 0}
    unobserved = hidden_cause_breakdown(np.array([NOTHING]), np.array([True]))
    assert unobserved[CAUSE_UNOBSERVED] == 1


def test_height_layers_bins_lowest_layer_and_no_box():
    # Cell heights (z of centres) with voxel 1 m: layers above bottom 0 are 0, 2, 3, 7; the fifth is in no box.
    centres = np.array([[0, 0, 0.5], [0, 0, 2.5], [0, 0, 3.5], [0, 0, 7.5], [0, 0, 1.5]])
    target = np.ones(5, dtype=bool)
    offsets, cells, inside = np.array([0, 4, 6]), np.array([0, 1, 2, 3, 1, 0]), np.array([True, True, True, True, True, True])
    # Object 0: class 1, bottom 0 over cells 0..3. Object 1: class 2 (ignored for class 1) over cells 1, 0.
    counts = height_layer_breakdown(
        centres, target, 1.0, offsets, cells, inside, np.array([0.0, 2.0]), np.array([1, 2]), [1])
    assert counts == {"0": 1, "1": 0, "2": 1, ">=3": 2, NO_BOX_BIN_NAME: 1}
    # Object 1 now class 1 with bottom 2.0: cell 1 (z 2.5) has layer 0 there, lower than 2 -> takes 0; cell 0 below bottom clips to 0.
    counts = height_layer_breakdown(
        centres, target, 1.0, offsets, cells, inside, np.array([0.0, 2.0]), np.array([1, 1]), [1])
    assert counts == {"0": 2, "1": 0, "2": 0, ">=3": 2, NO_BOX_BIN_NAME: 1}


def test_seen_above_counts_columns():
    # Column (0,0): z 0 hidden target, z 2 seen -> covered. Column (1,0): z 0 target, z -1 seen below -> not covered.
    cells = np.array([[0, 0, 0], [0, 0, 2], [1, 0, 0], [1, 0, -1]])
    seen = np.array([False, True, False, True])
    target = np.array([True, False, True, False])
    assert seen_above_count(cells, seen, target) == 1


def test_distance_and_object_counts():
    centres = np.array([[3.0, 4.0, 9.0], [30.0, 0.0, 0.0]])
    origins = np.array([[0.0, 0.0, 50.0]])
    distances = nearest_horizontal_distance(centres, origins)
    # Horizontal 3-4-5 triangle, then 30 m.
    assert np.allclose(distances, [5.0, 30.0])
    result = distance_breakdown(distances, np.array([True, False]), np.array([True, True]))
    names = distance_bin_names()
    assert result[names[1]] == {"hidden_gt": 1, "all_gt": 1}  # 5.0 is in [5, 10)
    assert result[names[4]] == {"hidden_gt": 0, "all_gt": 1}  # 30.0 is in [30, 40)
    counts = object_counts(
        np.array([0, 2, 3]), np.array([0, 1, 1]), np.array([True, False, True]), np.array([1, 1]), [1],
        np.array([True, True]), np.array([True, False]))
    assert counts == {"objects": 2, "with_gt_inside": 2, "with_hidden_gt_inside": 1}
