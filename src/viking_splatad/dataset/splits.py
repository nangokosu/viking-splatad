"""LINSPACE train/eval split replicating neurad-studio's frame selection (T21)."""

import math
from typing import List, Tuple

import numpy as np

from viking_splatad.registry import transform


@transform("T21", "neurad-studio LINSPACE train/eval split per sensor")
def linspace_train_eval_split(
    num_frames: int, train_fraction: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick train frames evenly spread over the sequence; the rest are eval.

    train_count = ceil(N * fraction); train = unique(int(linspace(0, N-1, train_count))).
    Truncation + uniqueness means eval is "everything not chosen", interleaved with train.

    Args:
        num_frames: Number of frames N of one sensor.
        train_fraction: Fraction in (0, 1] of frames used for training.

    Returns:
        (train_indices, eval_indices): sorted int arrays partitioning range(N).
    """
    train_count = math.ceil(num_frames * train_fraction)
    # Evenly spaced real positions on [0, N-1], truncated to frame indices.
    train_indices = np.unique(np.linspace(0, num_frames - 1, train_count).astype(int))
    eval_indices = np.setdiff1d(np.arange(num_frames), train_indices)
    return train_indices, eval_indices


def split_problems(
    train_indices: np.ndarray,
    eval_indices: np.ndarray,
    num_frames: int,
    train_fraction: float,
) -> List[str]:
    """Sanity-check a split.

    Args:
        train_indices, eval_indices: Output of ``linspace_train_eval_split``.
        num_frames: Total frame count.
        train_fraction: Fraction used to build the split.

    Returns:
        Human-readable problems; empty list when the split is sound.
    """
    problems = []
    if train_indices.size == 0:
        problems.append("train split is empty")
    if eval_indices.size == 0:
        problems.append("eval split is empty")
    if np.intersect1d(train_indices, eval_indices).size > 0:
        problems.append("train and eval overlap")
    if train_indices.size + eval_indices.size != num_frames:
        problems.append("train + eval do not cover all frames")
    # For fraction < 1 an eval frame must precede the last train frame; otherwise
    # eval is a contiguous tail (temporal extrapolation, not neurad's interleaving).
    if train_fraction < 1.0 and eval_indices.size and train_indices.size:
        if eval_indices.min() > train_indices.max():
            problems.append("eval frames are not interleaved (contiguous tail)")
    return problems
