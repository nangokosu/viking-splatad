"""Boundary and property tests of the LINSPACE split (T21): rounding direction, tiny N, problem branches.

Expected values are derived by hand or by integer arithmetic, never by calling the split builder.
"""

import numpy as np
import pytest

from viking_splatad.dataset.splits import linspace_train_eval_split, split_problems

# (num_frames, train_fraction, expected train count). The count is ceil(N * fraction) worked out by
# hand; each row is chosen so that ceil, floor and round do not all agree:
#   10 x 0.75 = 7.5   -> ceil 8, floor 7 (round-half-even gives 8)
#    7 x 0.75 = 5.25  -> ceil 6, floor 5, round 5
#    7 x 0.9  = 6.3   -> ceil 7, floor 6, round 6
#   10 x 0.9  = 9.0   -> 9 (exact, all agree: guards an off-by-one in the other direction)
#   20 x 0.75 = 15.0  -> 15 (exact)
#    3 x 0.75 = 2.25  -> ceil 3 (every frame trains, eval is empty)
#    1 x 0.8  = 0.8   -> ceil 1 (the single frame trains)
#    5 x 0.5  = 2.5   -> ceil 3, floor 2
TRAIN_COUNT_CASES = [
    (10, 0.75, 8), (7, 0.75, 6), (7, 0.9, 7), (10, 0.9, 9),
    (20, 0.75, 15), (3, 0.75, 3), (1, 0.8, 1), (5, 0.5, 3),
]


def expected_train_indices(num_frames: int, train_count: int) -> list:
    """Truncated even spacing over [0, N - 1] in integer arithmetic: floor(k (N - 1) / (count - 1)).

    Args:
        num_frames: Frame count N.
        train_count: Number of train frames (1 <= count <= N).

    Returns:
        Sorted train indices. A single train frame is frame 0 (the lone point of linspace).
    """
    if train_count == 1:
        return [0]
    return [(step * (num_frames - 1)) // (train_count - 1) for step in range(train_count)]


@pytest.mark.covers("T21")
@pytest.mark.parametrize("num_frames, train_fraction, train_count", TRAIN_COUNT_CASES)
def test_train_count_rounds_up_and_indices_truncate(num_frames, train_fraction, train_count):
    """The train count is ceil(N f) and the indices are the truncated even spacing."""
    train_indices, eval_indices = linspace_train_eval_split(num_frames, train_fraction)

    # Count and exact indices from first principles.
    assert train_indices.size == train_count
    assert list(train_indices) == expected_train_indices(num_frames, train_count)
    # The complement is eval: sizes add up to N.
    assert eval_indices.size == num_frames - train_count
    assert sorted(set(train_indices) | set(eval_indices)) == list(range(num_frames))


@pytest.mark.covers("T21")
def test_single_frame_has_no_eval_and_is_reported():
    """N = 1: the only frame trains, eval is empty and split_problems says so."""
    train_indices, eval_indices = linspace_train_eval_split(1, 0.75)
    assert list(train_indices) == [0] and eval_indices.size == 0
    assert split_problems(train_indices, eval_indices, 1, 0.75) == ["eval split is empty"]


@pytest.mark.covers("T21")
def test_split_is_disjoint_complete_sorted_for_many_sizes_and_fractions():
    """Leakage property over N = 1..60 and several fractions: no frame is both train and eval."""
    for num_frames in range(1, 61):
        for train_fraction in (0.1, 0.25, 0.5, 0.75, 0.8, 0.9, 1.0):
            train_indices, eval_indices = linspace_train_eval_split(num_frames, train_fraction)
            train_set, eval_set = set(train_indices.tolist()), set(eval_indices.tolist())
            context = f"N={num_frames} f={train_fraction}"
            # Disjoint (no leakage), with no duplicate inside either side.
            assert train_set.isdisjoint(eval_set), context
            assert len(train_set) == train_indices.size and len(eval_set) == eval_indices.size, context
            # Complete partition of the frame range.
            assert train_set | eval_set == set(range(num_frames)), context
            # Sorted ascending, and always at least one train frame.
            assert list(train_indices) == sorted(train_set) and list(eval_indices) == sorted(eval_set), context
            assert train_indices.size >= 1, context


@pytest.mark.covers("T21")
def test_fraction_one_trains_on_everything():
    """Fraction 1.0: every frame trains, none is eval (the caller must handle this separately)."""
    train_indices, eval_indices = linspace_train_eval_split(6, 1.0)
    assert list(train_indices) == list(range(6)) and eval_indices.size == 0


@pytest.mark.covers("T21")
def test_split_problems_overlap_only():
    """Train {0, 2, 4} and eval {2, 3}: sizes sum to N = 5, frame 2 is in both -> only the overlap."""
    problems = split_problems(np.array([0, 2, 4]), np.array([2, 3]), 5, 0.6)
    assert problems == ["train and eval overlap"]


@pytest.mark.covers("T21")
def test_split_problems_coverage_only():
    """Train {0, 4} and eval {2} of N = 5: frames 1 and 3 are in neither -> only the coverage message."""
    problems = split_problems(np.array([0, 4]), np.array([2]), 5, 0.4)
    assert problems == ["train + eval do not cover all frames"]


@pytest.mark.covers("T21")
def test_split_problems_empty_train():
    """No train frame: reported; with eval covering all frames nothing else is wrong."""
    problems = split_problems(np.array([], dtype=int), np.arange(4), 4, 0.5)
    assert problems == ["train split is empty"]


@pytest.mark.covers("T21")
def test_split_problems_interleaving_rule_only_applies_below_fraction_one():
    """A contiguous-tail eval is flagged for f < 1 but not for f = 1.0; an interleaved one never."""
    train_indices, tail_eval = np.arange(3), np.array([3, 4])
    assert split_problems(train_indices, tail_eval, 5, 0.6) == ["eval frames are not interleaved (contiguous tail)"]
    assert split_problems(train_indices, tail_eval, 5, 1.0) == []
    # Eval frame 1 precedes the last train frame 4: interleaved, sound.
    assert split_problems(np.array([0, 2, 4]), np.array([1, 3]), 5, 0.6) == []
