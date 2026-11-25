"""Tests for the figure-suite metric preparation; expected values come from set algebra written in the test."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from viking_splatad.eval.hidden_metrics import csr_from_lists, hidden_class_report, threshold_key
from viking_splatad.viz.figure_metrics import class_ids_for_name, occupancy_samples, threshold_curves

TOLERANCE = 1e-12
EVAL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "eval.yaml"
# Synthetic geometry: 12 cells on a line, 1 m voxels (so the half-voxel margin is 0.5 m).
CELL_COUNT = 12
VOXEL_EDGE_M = 1.0
GROUP_NAME = "obstacles"
# Class ids: 1 and 3 are in the obstacles group of configs/eval.yaml, 2 is not.
OBSTACLE_CLASS_IDS = (1, 3)
OTHER_CLASS_ID = 2
# Object table: (box id, class id, grown-box cells, inside-box flags).
OBJECT_TABLE = (
    (10, 1, [0, 1, 2, 3], [True, True, True, False]),
    (11, 3, [4, 5, 6, 7], [True, True, True, True]),
    (12, OTHER_CLASS_ID, [8, 9, 10, 11], [True, True, True, True]),
)
OCCUPANCY_VALUES = [0.9, 0.6, 0.2, 0.7, 0.1, 0.85, 0.6, 0.55, 0.9, 0.3, 0.0, 0.0]
THRESHOLDS = (0.5, 0.8)


def mask(*indices):
    """Bool mask over the synthetic cells that is True at ``indices``."""
    result = np.zeros(CELL_COUNT, dtype=bool)
    result[list(indices)] = True
    return result


def build_cells():
    """Synthetic cells archive: N={0..7}, G={0,1,2,5,6}, H_near={1,2,3,5,6,7}, H={1..9}."""
    offsets, object_cells, object_inside = csr_from_lists(
        [np.array(entry[2]) for entry in OBJECT_TABLE], [np.array(entry[3]) for entry in OBJECT_TABLE])
    return {
        "occupancy": np.array(OCCUPANCY_VALUES), "hidden_near": mask(1, 2, 3, 5, 6, 7), "hidden": mask(*range(1, 10)),
        "centres": np.stack([np.arange(CELL_COUNT, dtype=float), np.zeros(CELL_COUNT), np.zeros(CELL_COUNT)], axis=1),
        "voxel_size": np.float64(VOXEL_EDGE_M), "gt_class_ids": np.zeros(CELL_COUNT, dtype=int),
        f"neighbourhood_{GROUP_NAME}": mask(*range(8)), f"truth_{GROUP_NAME}": mask(0, 1, 2, 5, 6),
        "object_box_ids": np.array([entry[0] for entry in OBJECT_TABLE]),
        "object_class_ids": np.array([entry[1] for entry in OBJECT_TABLE]),
        "object_cell_offsets": offsets, "object_cells": object_cells, "object_inside": object_inside,
        "p2_thresholds": np.array(THRESHOLDS),
    }


def expected_recall_precision(occupied, truth, neighbourhood, hidden, cell_set):
    """First-principles recall and precision over ``cell_set`` (a boolean mask) restricted to the band."""
    scored = cell_set & neighbourhood
    true_positive = np.count_nonzero(occupied & truth & hidden & scored)
    return (true_positive / np.count_nonzero(truth & hidden & scored),
            true_positive / np.count_nonzero(occupied & neighbourhood & hidden & scored))


def test_threshold_curves_known_answer():
    """Pooled and mean values at t=0.5 on H_near, from the set algebra of the synthetic archive."""
    cells = build_cells()
    curves = threshold_curves(cells, GROUP_NAME, THRESHOLDS, "near", class_ids=OBSTACLE_CLASS_IDS)
    occupied = np.array(OCCUPANCY_VALUES) >= THRESHOLDS[0]
    neighbourhood, truth, hidden = cells[f"neighbourhood_{GROUP_NAME}"], cells[f"truth_{GROUP_NAME}"], cells["hidden_near"]
    pooled_recall, pooled_precision = expected_recall_precision(occupied, truth, neighbourhood, hidden, np.ones(CELL_COUNT, dtype=bool))
    assert curves["pooled_recall"][0] == pytest.approx(pooled_recall, abs=TOLERANCE)
    assert curves["pooled_precision"][0] == pytest.approx(pooled_precision, abs=TOLERANCE)
    # Per-object means over the two obstacle objects (their cell masks are the unions of the object cells).
    per_object = [expected_recall_precision(occupied, truth,
                                            neighbourhood, hidden, mask(*entry[2])) for entry in OBJECT_TABLE[:2]]
    # Object 10's G_k excludes the non-inside cell 3, which is outside truth anyway, so the plain masks agree.
    assert curves["mean_recall"][0] == pytest.approx(np.mean([pair[0] for pair in per_object]), abs=TOLERANCE)
    assert curves["mean_precision"][0] == pytest.approx(np.mean([pair[1] for pair in per_object]), abs=TOLERANCE)
    assert curves["thresholds"].tolist() == list(THRESHOLDS)


def test_threshold_curves_undefined_is_nan():
    """A threshold above every occupancy leaves no occupied cell: precision is undefined, so NaN."""
    curves = threshold_curves(build_cells(), GROUP_NAME, [1.5], "near", class_ids=OBSTACLE_CLASS_IDS)
    assert np.isnan(curves["pooled_precision"][0]) and curves["pooled_recall"][0] == 0.0


def test_occupancy_samples_known_answer():
    """truth = occupancy at G n H_near = {1,2,5,6}; off_truth = at (N n H_near) \\ G = {3,7}."""
    samples = occupancy_samples(build_cells(), GROUP_NAME, "near")
    assert samples["truth"].tolist() == [OCCUPANCY_VALUES[index] for index in (1, 2, 5, 6)]
    assert samples["off_truth"].tolist() == [OCCUPANCY_VALUES[index] for index in (3, 7)]
    # Band "all": G = {0,1,2,5,6}, N \ G = {3,4,7}.
    whole = occupancy_samples(build_cells(), GROUP_NAME, "all")
    assert whole["truth"].tolist() == [OCCUPANCY_VALUES[index] for index in (0, 1, 2, 5, 6)]
    assert whole["off_truth"].tolist() == [OCCUPANCY_VALUES[index] for index in (3, 4, 7)]


def test_unknown_band_rejected():
    """A band outside near/full/all is an error."""
    with pytest.raises(ValueError):
        occupancy_samples(build_cells(), GROUP_NAME, "wide")


def test_class_ids_for_name_reads_eval_config():
    """Groups and single classes resolve as scripts/19 resolves them."""
    eval_config = yaml.safe_load(EVAL_CONFIG_PATH.read_text())["eval"]
    assert class_ids_for_name("obstacles", eval_config) == tuple(eval_config["class_groups"]["obstacles"])
    assert class_ids_for_name("tree_trunk", eval_config) == (eval_config["scored_class_ids"]["tree_trunk"],)
    with pytest.raises(KeyError):
        class_ids_for_name("not_a_class", eval_config)


@pytest.mark.parametrize("band,hidden_key", [("near", "hidden_near"), ("full", "hidden")])
def test_consistent_with_hidden_class_report(band, hidden_key):
    """Curves at each p2 threshold equal what scripts/19's hidden_class_report records."""
    cells = build_cells()
    class_ids = OBSTACLE_CLASS_IDS
    objects = {key: cells[key] for key in ("object_box_ids", "object_class_ids", "object_cell_offsets", "object_cells", "object_inside")}
    report = hidden_class_report(class_ids, cells[f"neighbourhood_{GROUP_NAME}"], cells[f"truth_{GROUP_NAME}"], cells[hidden_key],
                                 np.zeros(CELL_COUNT, dtype=bool), cells["occupancy"], THRESHOLDS, objects)
    curves = threshold_curves(cells, GROUP_NAME, THRESHOLDS, band, class_ids=class_ids)
    for threshold_index, threshold in enumerate(THRESHOLDS):
        block = report["P2"][threshold_key(threshold)]
        assert curves["pooled_recall"][threshold_index] == pytest.approx(block["pooled"]["recall"], abs=TOLERANCE)
        assert curves["pooled_precision"][threshold_index] == pytest.approx(block["pooled"]["precision"], abs=TOLERANCE)
        assert curves["mean_recall"][threshold_index] == pytest.approx(block["per_object"]["mean_recall"], abs=TOLERANCE)
        assert curves["mean_precision"][threshold_index] == pytest.approx(block["per_object"]["mean_precision"], abs=TOLERANCE)
