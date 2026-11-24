"""Known-answer tests for the T78 range margin and the T91 hidden-space label."""

import numpy as np
import pytest

from viking_splatad.eval.hidden_geometry import (BLOCKED_COLUMN, NO_RETURN_COLUMN, PASSED_THROUGH_COLUMN,
                                                 SEEN_COLUMN, hidden_space_mask, recorded_ray_evidence)

VOXEL_SIZE = 1.0
CELL_CENTRE_X = 10.0
CENTRES = np.array([[CELL_CENTRE_X, 0.0, 0.0]])
# A ray along +x crosses the cube at [centre - half width, centre + half width].
ENTRY_DEPTH = CELL_CENTRE_X - VOXEL_SIZE / 2
EXIT_DEPTH = CELL_CENTRE_X + VOXEL_SIZE / 2
MARGIN = 0.2
PLUS_X = np.array([1.0, 0.0, 0.0])


def evidence_for(depths, returned, margin):
    """Run T78 on beams all along +x with the given depths, return flags and margin; return the [1,4] result."""
    directions = np.tile(PLUS_X, (len(depths), 1))
    return recorded_ray_evidence(CENTRES, VOXEL_SIZE, directions, np.array(depths, dtype=float),
                                 np.array(returned, dtype=bool), range_margin=margin)


@pytest.mark.covers("T78")
@pytest.mark.covers("T91")
def test_return_just_short_of_cube_is_seen_with_margin_and_blocked_without():
    depth = ENTRY_DEPTH - MARGIN / 2
    with_margin = evidence_for([depth], [True], MARGIN)[0]
    assert with_margin[SEEN_COLUMN] and not with_margin[BLOCKED_COLUMN]
    assert not hidden_space_mask(evidence_for([depth], [True], MARGIN))[0]
    without = evidence_for([depth], [True], 0.0)[0]
    assert without[BLOCKED_COLUMN] and not without[SEEN_COLUMN]


@pytest.mark.covers("T78")
@pytest.mark.covers("T91")
def test_return_far_short_is_blocked_and_hidden():
    evidence = evidence_for([ENTRY_DEPTH - 2 * MARGIN], [True], MARGIN)
    assert evidence[0, BLOCKED_COLUMN] and not evidence[0, SEEN_COLUMN]
    assert hidden_space_mask(evidence)[0]


@pytest.mark.covers("T78")
@pytest.mark.covers("T91")
def test_second_beam_passing_beyond_cube_cancels_hidden():
    evidence = evidence_for([ENTRY_DEPTH - 2 * MARGIN, EXIT_DEPTH + 2 * MARGIN], [True, True], MARGIN)
    assert evidence[0, BLOCKED_COLUMN] and evidence[0, PASSED_THROUGH_COLUMN]
    assert not hidden_space_mask(evidence)[0]


@pytest.mark.covers("T78")
def test_return_just_beyond_cube_is_seen_not_passed_with_margin():
    evidence = evidence_for([EXIT_DEPTH + MARGIN / 2], [True], MARGIN)[0]
    assert evidence[SEEN_COLUMN] and not evidence[PASSED_THROUGH_COLUMN]


@pytest.mark.covers("T78")
@pytest.mark.covers("T91")
def test_no_return_only_cell_is_not_hidden():
    evidence = evidence_for([ENTRY_DEPTH], [False], MARGIN)
    assert evidence[0, NO_RETURN_COLUMN]
    assert not hidden_space_mask(evidence)[0]


@pytest.mark.covers("T78")
@pytest.mark.parametrize("bad_margin", [-0.1, float("nan"), float("inf")])
def test_invalid_margin_raises(bad_margin):
    with pytest.raises(ValueError):
        evidence_for([ENTRY_DEPTH], [True], bad_margin)


@pytest.mark.covers("T91")
def test_hidden_space_mask_truth_table():
    # Columns: seen, blocked, no_return, passed_through.
    evidence = np.array([[0, 1, 0, 0],   # blocked only -> hidden
                         [1, 1, 0, 0],   # seen contradicts -> not hidden
                         [0, 1, 1, 0],   # no_return ignored -> hidden
                         [0, 1, 0, 1]],  # passed beyond contradicts -> not hidden
                        dtype=bool)
    assert hidden_space_mask(evidence).tolist() == [True, False, True, False]
    with pytest.raises(ValueError):
        hidden_space_mask(np.zeros((3, 3), dtype=bool))
