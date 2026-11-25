"""Known-answer tests for the figure grid helpers (synthetic inputs, expectations derived by hand)."""

import numpy as np

from viking_splatad.viz.figure_grids import arm_difference_sets


def test_arm_difference_sets_partition():
    first = np.array([True, True, False, False])
    second = np.array([True, False, True, False])
    sets = arm_difference_sets(first, second)
    np.testing.assert_array_equal(sets["only_first"], [False, True, False, False])
    np.testing.assert_array_equal(sets["only_second"], [False, False, True, False])
    np.testing.assert_array_equal(sets["both"], [True, False, False, False])
    # Partition: disjoint and union equals first | second.
    total = sets["only_first"].astype(int) + sets["only_second"] + sets["both"]
    np.testing.assert_array_equal(total, (first | second).astype(int))
