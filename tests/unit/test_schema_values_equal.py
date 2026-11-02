"""Tests that ``values_equal`` keeps bool distinct from numbers (bool subclasses int)."""

import pytest

from viking_splatad.dataset.schema import values_equal


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (True, 1, False),  # bool vs int: different types
        (1, True, False),  # symmetric
        (False, 0, False),
        (True, 1.0, False),  # bool vs float
        (True, True, True),
        (False, False, True),
        (True, False, False),
        (1, 1.0, False),  # int vs float stays distinguishable (pre-existing rule)
        (1, 1, True),
        (1.0, 1.0, True),
    ],
)
def test_values_equal_bool_vs_number(left, right, expected):
    """Bools compare by type and value; the int-vs-float distinction is unchanged."""
    assert values_equal(left, right) is expected


def test_values_equal_bool_in_nested_containers():
    """The bool rule applies inside dicts, lists and tuples as well."""
    assert not values_equal({"flag": [True]}, {"flag": [1]})
    assert not values_equal({"flag": (False,)}, {"flag": [0]})
    assert values_equal({"flag": [True, 2]}, {"flag": (True, 2)}) is True
