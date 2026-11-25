"""Known-answer tests of the scored-box selection of the scene figure (viz/scored_boxes.py)."""

from types import SimpleNamespace

import numpy as np
import pytest

from viking_splatad.viz.scored_boxes import class_box_counts, select_scored_cuboids

CLASS_NAME_BY_ID = {1: "tree_trunk", 3: "rock"}


def make_boxes():
    """Five synthetic boxes: three trunks (ids 1-3), one rock (4), one car (5, not drawn)."""
    return [SimpleNamespace(box_id=box_id, class_id=class_id, class_name=class_name)
            for box_id, class_id, class_name in ((1, 1, "tree_trunk"), (2, 1, "tree_trunk"), (3, 1, "tree_trunk"),
                                                 (4, 3, "rock"), (5, 5, "car"))]


def test_selects_only_listed_boxes_of_the_wanted_classes():
    """Box 3 is not a scored object and the car is not a wanted class: 2 trunks and 1 rock remain."""
    selected = select_scored_cuboids(make_boxes(), np.array([1, 2, 4, 5]), np.array([1, 1, 3, 5]), CLASS_NAME_BY_ID,
                                     {"tree_trunk": 2, "rock": 1})
    assert [box.box_id for box in selected] == [1, 2, 4]
    assert class_box_counts(selected) == {"tree_trunk": 2, "rock": 1}


def test_count_mismatch_with_the_report_raises():
    """The report counts 3 trunks but the archive lists 2: refuse."""
    with pytest.raises(ValueError, match="tree_trunk"):
        select_scored_cuboids(make_boxes(), np.array([1, 2, 4]), np.array([1, 1, 3]), CLASS_NAME_BY_ID,
                              {"tree_trunk": 3, "rock": 1})


def test_report_without_the_class_raises():
    """A class missing from the report cannot be checked: refuse."""
    with pytest.raises(ValueError, match="rock"):
        select_scored_cuboids(make_boxes(), np.array([1]), np.array([1]), CLASS_NAME_BY_ID, {"tree_trunk": 1})
