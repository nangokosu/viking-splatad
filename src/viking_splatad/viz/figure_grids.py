"""Pure-numpy set helpers for the Gaussian-occupancy figure suite (no plotting, no I/O)."""

import numpy as np


def arm_difference_sets(hit_first, hit_second):
    """Partition the union of two arms' hit masks into only-first, only-second and both.

    Args:
        hit_first, hit_second: (V,) bool masks.

    Returns:
        Dict ``only_first``, ``only_second``, ``both`` of (V,) bool; pairwise disjoint, union = first | second.
    """
    hit_first = np.asarray(hit_first, dtype=bool)
    hit_second = np.asarray(hit_second, dtype=bool)
    return {"only_first": hit_first & ~hit_second,
            "only_second": hit_second & ~hit_first,
            "both": hit_first & hit_second}
