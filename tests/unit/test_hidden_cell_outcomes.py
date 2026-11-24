"""Known-answer tests for hidden_cell_outcomes (eval/p2_cells.py), the cell sets the hidden P2 picture draws."""

import numpy as np
import pytest

from viking_splatad.eval.p2_cells import HIDDEN_KEY, HIDDEN_NEAR_KEY, NEIGHBOURHOOD_KEY_PREFIX, TRUTH_KEY_PREFIX, hidden_cell_outcomes

LOW_THRESHOLD, HIGH_THRESHOLD = 0.5, 0.8
# A threshold absent from the synthetic archive's p2_thresholds.
UNEVALUATED_THRESHOLD = 0.7


def synthetic_cells(name):
    """Six cells, each testing one branch of the hidden rule, for class/group ``name``.

    Roles (occupancy, hidden, in N, on GT):
        0: 0.9, hidden, N, GT  -> hit at both thresholds
        1: 0.6, hidden, N, GT  -> hit at 0.5, missed at 0.8
        2: 0.9, hidden, N, no GT -> false alarm
        3: 0.1, hidden, N, GT  -> missed
        4: 0.9, NOT hidden, N, GT -> ignored (visible)
        5: 0.9, hidden, outside N -> ignored (unscored)

    Args:
        name: Class or group name for the per-class keys.

    Returns:
        Archive-like dict.
    """
    return {
        "occupancy": np.array([0.9, 0.6, 0.9, 0.1, 0.9, 0.9]),
        "threshold": np.float64(LOW_THRESHOLD),
        "p2_thresholds": np.array([LOW_THRESHOLD, HIGH_THRESHOLD]),
        HIDDEN_KEY: np.array([True, True, True, True, False, True]),
        HIDDEN_NEAR_KEY: np.array([True, True, True, True, False, True]),
        f"{NEIGHBOURHOOD_KEY_PREFIX}{name}": np.array([True, True, True, True, True, False]),
        f"{TRUTH_KEY_PREFIX}{name}": np.array([True, True, False, True, True, False]),
    }


def indices(mask):
    """Positions of the True entries as a list."""
    return np.flatnonzero(mask).tolist()


@pytest.mark.parametrize("name", ["tree_trunk", "obstacles"])
def test_outcomes_at_low_threshold(name):
    """At 0.5 cells 0 and 1 are hits, 2 a false alarm, 3 missed; visible and unscored cells never count."""
    outcomes = hidden_cell_outcomes(synthetic_cells(name), name, LOW_THRESHOLD)
    assert indices(outcomes["hit"]) == [0, 1]
    assert indices(outcomes["false_alarm"]) == [2]
    assert indices(outcomes["missed"]) == [3]
    assert indices(outcomes["context"]) == [0, 1, 2, 3]


def test_cell_below_high_threshold_becomes_missed():
    """At 0.8 cell 1 (occupancy 0.6) turns from hit into missed."""
    outcomes = hidden_cell_outcomes(synthetic_cells("obstacles"), "obstacles", HIGH_THRESHOLD)
    assert indices(outcomes["hit"]) == [0]
    assert indices(outcomes["missed"]) == [1, 3]
    assert indices(outcomes["false_alarm"]) == [2]


def test_exactly_at_threshold_is_occupied():
    """The cut is inclusive (>=), as in p2_cell_outcomes."""
    cells = synthetic_cells("obstacles")
    cells["occupancy"] = np.full(6, LOW_THRESHOLD)
    assert indices(hidden_cell_outcomes(cells, "obstacles", LOW_THRESHOLD)["hit"]) == [0, 1, 3]


def test_missing_hidden_key_is_a_key_error_naming_it():
    """An archive without ``hidden`` predates the evaluation."""
    cells = synthetic_cells("obstacles")
    del cells[HIDDEN_KEY]
    # The default band is now near; the full band is the one that reads ``hidden``.
    with pytest.raises(KeyError, match="hidden"):
        hidden_cell_outcomes(cells, "obstacles", LOW_THRESHOLD, band="full")


def test_missing_class_arrays_is_a_key_error():
    """A class absent from the archive is reported by key."""
    with pytest.raises(KeyError, match="neighbourhood_rock"):
        hidden_cell_outcomes(synthetic_cells("obstacles"), "rock", LOW_THRESHOLD)


def test_unevaluated_threshold_is_a_value_error():
    """Pictures exist only for evaluated thresholds."""
    with pytest.raises(ValueError, match="not evaluated"):
        hidden_cell_outcomes(synthetic_cells("obstacles"), "obstacles", UNEVALUATED_THRESHOLD)


def synthetic_cells_with_distinct_bands(name):
    """The synthetic archive where H_near excludes hidden cell 3 (a miss), so the two bands give different sets."""
    cells = synthetic_cells(name)
    cells[HIDDEN_NEAR_KEY] = np.array([True, True, True, False, False, True])
    return cells


def test_band_argument_selects_near_or_full_hidden_mask():
    """Default and "near" read hidden_near; "full" reads hidden; the cell beyond the band drops out of near only."""
    cells = synthetic_cells_with_distinct_bands("obstacles")
    near = hidden_cell_outcomes(cells, "obstacles", LOW_THRESHOLD)
    assert indices(near["missed"]) == [] and indices(near["context"]) == [0, 1, 2]
    assert indices(hidden_cell_outcomes(cells, "obstacles", LOW_THRESHOLD, band="near")["context"]) == [0, 1, 2]
    full = hidden_cell_outcomes(cells, "obstacles", LOW_THRESHOLD, band="full")
    assert indices(full["missed"]) == [3] and indices(full["context"]) == [0, 1, 2, 3]
    assert indices(full["hit"]) == indices(near["hit"]) == [0, 1]


def test_unknown_band_is_a_value_error():
    """Only "near" and "full" are bands."""
    with pytest.raises(ValueError, match="band"):
        hidden_cell_outcomes(synthetic_cells("obstacles"), "obstacles", LOW_THRESHOLD, band="medium")


def test_missing_near_key_is_a_key_error_naming_it_but_full_still_works():
    """An archive from before the near band: near names hidden_near; full still draws."""
    cells = synthetic_cells("obstacles")
    del cells[HIDDEN_NEAR_KEY]
    with pytest.raises(KeyError, match="hidden_near"):
        hidden_cell_outcomes(cells, "obstacles", LOW_THRESHOLD)
    assert indices(hidden_cell_outcomes(cells, "obstacles", LOW_THRESHOLD, band="full")["hit"]) == [0, 1]
