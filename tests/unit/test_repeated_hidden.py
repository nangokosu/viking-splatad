"""Known-answer repeat spread and stable-cell checks, including recall/precision tradeoffs."""

import copy

import cv2
import numpy as np
import pytest

from viking_splatad.eval.hidden_geometry import occupied_cell_metrics
from viking_splatad.eval.repeated_hidden import scalar_repeat_summary, summarize_comparisons
from viking_splatad.viz.repeated_hidden import write_consistency_map
from viking_splatad.viz.hidden_trunks import PANEL_PIXELS


def analytic_repeats():
    """Return two independent synthetic comparisons on seen, blocked, no-return and off-GT cells."""
    truth = np.array([True, True, True, False])
    evidence = np.array([[True, False, False, False], [False, True, False, False],
                         [False, False, True, False], [False, False, False, False]])
    reports, arrays = [], []
    for repeat_index in range(2):
        baseline = np.array([True, bool(repeat_index), False, False])
        radar = np.ones(4, dtype=bool)
        cells = {"truth": truth, "evidence": evidence, "indices": np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0], [3, 1, 0]]),
                 "centres": np.array([[0., 0, 0], [1., 0, 0], [2., 1, 0], [3., 1, 0]]), "threshold": np.array(0.5),
                 "B": baseline.astype(float), "L": radar.astype(float), "B_P1": baseline, "L_P1": radar}
        report = {"schema": "viking-splatad/hidden-trunk-pilot/1", "session": "june", "source_revision": "source",
                  "membership_sha256": "same", "labels_sha256": "same", "threshold": 0.5, "voxel_size_m": 1.,
                  "margin_m": 1., "scored_trunks": 1, "alignment_reviewed": False,
                  "arms": {arm: {"run_id": arm + str(repeat_index), **{
                      representation: occupied_cell_metrics(mask, truth, evidence) for representation in ("P1", "P2")}}
                           for arm, mask in (("B", baseline), ("L", radar))}}
        reports.append(report)
        arrays.append(cells)
    return reports, arrays


def test_repeats_expose_geometry_recovery_and_precision_loss_together():
    """A stable no-return-cell gain and unstable blocked-cell gain cannot hide one off-GT cell."""
    reports, arrays = analytic_repeats()
    result, views = summarize_comparisons(reports, arrays)
    assert result["metrics"]["P2"]["recall"]["B"]["mean"] == 0.5
    assert result["metrics"]["P2"]["no_return_recall"]["paired_L_minus_B"]["mean"] == 1
    assert result["metrics"]["P2"]["precision"]["paired_L_minus_B"]["mean"] == -1 / 4
    assert result["metrics"]["P2"]["blocked_recall"]["absolute_mean_gap_exceeds_both_within_arm_ranges"] is False
    assert result["consistent_cells"]["P2"]["strata"]["no_return"]["consistent_recovery"] == 1
    assert result["consistent_cells"]["P2"]["strata"]["blocked"]["intermittent_B_preserved"] == 1
    assert result["consistent_cells"]["P2"]["consistent_new_off_trunk_cells"] == 1
    np.testing.assert_array_equal(views["P2"]["masks"]["consistent_recovery"], [False, False, True, False])
    assert result["all_alignment_reviews_complete"] is False


def test_repeat_summary_refuses_reused_runs_changed_observations_and_single_pair():
    """Reusing an output or moving the scored observations cannot count as independent repetition."""
    reports, arrays = analytic_repeats()
    with pytest.raises(ValueError, match="at least two"):
        summarize_comparisons(reports[:1], arrays[:1])
    reused = copy.deepcopy(reports)
    reused[1]["arms"]["B"]["run_id"] = reused[0]["arms"]["B"]["run_id"]
    with pytest.raises(ValueError, match="distinct run"):
        summarize_comparisons(reused, arrays)
    changed = copy.deepcopy(arrays)
    changed[1]["evidence"][0, 0] = False
    with pytest.raises(ValueError, match="identical"):
        summarize_comparisons(reports, changed)


def test_undefined_repeat_values_cannot_be_dropped_and_spread_is_sample_spread():
    """Undefined ratios stay undefined; known symmetric samples give first-principles sample variance."""
    assert scalar_repeat_summary([1., None])["mean"] is None
    actual = scalar_repeat_summary([0., 1.])
    assert actual["mean"] == 1 / 2 and actual["range"] == 1
    np.testing.assert_allclose(actual["sample_std"], np.sqrt(((0 - 0.5) ** 2 + (1 - 0.5) ** 2) / (2 - 1)))


def test_consistency_picture_uses_the_same_recovered_and_off_trunk_masks(tmp_path):
    """The matched picture is drawn from the summary's own repeat stacks and names its 3D counts."""
    reports, arrays = analytic_repeats()
    _, views = summarize_comparisons(reports, arrays)
    path = write_consistency_map(tmp_path / "consistent.png", arrays[0]["centres"], arrays[0]["truth"], arrays[0]["evidence"],
                                views["P2"]["B"], views["P2"]["L"])
    assert cv2.imread(str(path)).shape == (2 * (PANEL_PIXELS + PANEL_PIXELS // 8), 3 * PANEL_PIXELS, 3)
    assert "Consistent new off-trunk cells: 1" in path.with_suffix(".txt").read_text()
