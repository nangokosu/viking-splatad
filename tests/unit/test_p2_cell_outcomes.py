"""Known-answer tests for p2_cell_outcomes (eval/p2_cells.py), the cell classification the P2 picture draws."""

import numpy as np
import pytest

from viking_splatad.eval.p2_cells import NEIGHBOURHOOD_KEY_PREFIX, TRUTH_KEY_PREFIX, p2_cell_outcomes

# Binding cut of the synthetic archive (the even-odds value of configs/eval.yaml, chosen here so that one
# cell sits exactly on it and tests the inclusive >=).
SYNTHETIC_THRESHOLD = 0.5


def synthetic_cells():
    """Six cells, two classes ("trunk", "rock"), built so each cell tests one branch of the rule.

    Cell roles:
        0: occupied, trunk neighbourhood, trunk GT            -> trunk TP
        1: occupied (exactly at the threshold), trunk nbhd, no GT -> trunk FP
        2: not occupied, trunk nbhd, trunk GT                 -> trunk FN (not occupied)
        3: occupied, rock neighbourhood only, rock GT         -> rock TP
        4: occupied, in no neighbourhood, truth flag set      -> never scored; truth outside its nbhd ignored
        5: occupied, both neighbourhoods, rock GT only        -> on GT in the union (off GT for trunk)

    Returns:
        Archive-like dict of arrays.
    """
    return {
        "occupancy": np.array([0.9, SYNTHETIC_THRESHOLD, 0.1, 0.7, 0.8, 0.6]),
        "threshold": np.float64(SYNTHETIC_THRESHOLD),
        f"{NEIGHBOURHOOD_KEY_PREFIX}trunk": np.array([True, True, True, False, False, True]),
        f"{TRUTH_KEY_PREFIX}trunk": np.array([True, False, True, False, True, False]),
        f"{NEIGHBOURHOOD_KEY_PREFIX}rock": np.array([False, False, False, True, False, True]),
        f"{TRUTH_KEY_PREFIX}rock": np.array([False, False, False, True, False, True]),
    }


def test_single_class_outcomes_are_the_p2_true_and_false_positives():
    """For one class, occupied & scored & on_ground_truth is the TP set and occupied & scored & ~on_GT the FP set."""
    cells = synthetic_cells()
    outcomes = p2_cell_outcomes(cells, ["trunk"])
    # Expected from the definitions: occupied = occupancy >= threshold (inclusive), scored = trunk nbhd,
    # on GT = trunk truth within the trunk nbhd (cell 4's truth lies outside it).
    expected_occupied = cells["occupancy"] >= SYNTHETIC_THRESHOLD
    expected_scored = cells[f"{NEIGHBOURHOOD_KEY_PREFIX}trunk"]
    expected_on_truth = cells[f"{TRUTH_KEY_PREFIX}trunk"] & expected_scored
    np.testing.assert_array_equal(outcomes["occupied"], expected_occupied)
    np.testing.assert_array_equal(outcomes["scored"], expected_scored)
    np.testing.assert_array_equal(outcomes["on_ground_truth"], expected_on_truth)
    # The metric's counts: TP = cell 0, FP = cells 1 and 5 (cell 1 sits exactly on the cut, so it is occupied).
    true_positive = outcomes["occupied"] & outcomes["scored"] & outcomes["on_ground_truth"]
    false_positive = outcomes["occupied"] & outcomes["scored"] & ~outcomes["on_ground_truth"]
    np.testing.assert_array_equal(np.flatnonzero(true_positive), [0])
    np.testing.assert_array_equal(np.flatnonzero(false_positive), [1, 5])
    # Precision = TP / (TP + FP) = 1 / 3, from the counts above.
    assert np.count_nonzero(true_positive) / np.count_nonzero(true_positive | false_positive) == pytest.approx(1 / 3)


def test_several_classes_take_the_union_of_neighbourhoods_and_truths():
    """With two classes a cell is scored in either neighbourhood and on GT when on either class's GT."""
    outcomes = p2_cell_outcomes(synthetic_cells(), ["trunk", "rock"])
    # Scored: trunk nbhd {0, 1, 2, 5} union rock nbhd {3, 5} = {0, 1, 2, 3, 5}; cell 4 is in neither.
    np.testing.assert_array_equal(np.flatnonzero(outcomes["scored"]), [0, 1, 2, 3, 5])
    # On GT: trunk {0, 2} union rock {3, 5}; cell 5 is off trunk GT but on rock GT, so on GT in the union.
    np.testing.assert_array_equal(np.flatnonzero(outcomes["on_ground_truth"]), [0, 2, 3, 5])


def test_refuses_an_empty_class_list_and_an_unknown_class():
    """No class is a usage error; a class without archive arrays is a missing key, never an empty selection."""
    with pytest.raises(ValueError):
        p2_cell_outcomes(synthetic_cells(), [])
    with pytest.raises(KeyError):
        p2_cell_outcomes(synthetic_cells(), ["bush"])
