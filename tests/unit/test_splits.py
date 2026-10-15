"""Tests for the LINSPACE train/eval split replication (T21)."""

import math

import numpy as np
import pytest

from viking_splatad.dataset.splits import linspace_train_eval_split, split_problems


@pytest.mark.covers("T21")
def test_split_counts_disjoint_complete_and_interleaved():
    """N=10, fraction 0.8: ceil(8.0)=8 train targets, eval interleaved, no overlap."""
    num_frames, train_fraction = 10, 0.8
    train_indices, eval_indices = linspace_train_eval_split(num_frames, train_fraction)

    # First-principles train indices: floor(k*(N-1)/(count-1)) for k in 0..count-1.
    train_target_count = math.ceil(num_frames * train_fraction)
    expected_train = sorted(
        {(step * (num_frames - 1)) // (train_target_count - 1) for step in range(train_target_count)}
    )
    assert list(train_indices) == expected_train
    # Partition of all frames.
    assert set(train_indices).isdisjoint(eval_indices)
    assert sorted(set(train_indices) | set(eval_indices)) == list(range(num_frames))
    # Eval frames are interleaved, not a contiguous tail.
    assert eval_indices.min() < train_indices.max()
    assert split_problems(train_indices, eval_indices, num_frames, train_fraction) == []


@pytest.mark.covers("T21")
def test_split_problems_flags_contiguous_tail_and_empty_eval():
    """A tail split and a fraction of 1.0 are both reported."""
    num_frames = 10
    tail_problems = split_problems(
        np.arange(8), np.arange(8, 10), num_frames, 0.8
    )
    assert any("interleaved" in message for message in tail_problems)
    full_train, empty_eval = linspace_train_eval_split(num_frames, 1.0)
    assert any("eval" in message for message in split_problems(full_train, empty_eval, num_frames, 1.0))
