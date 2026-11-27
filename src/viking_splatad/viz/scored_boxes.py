"""The GT boxes the evaluation scores, for the ground-truth panel of the scene figure (display only).

The cells archive of a run lists every scored object (``object_box_ids`` / ``object_class_ids``, one per cuboid
that ``scored_cells.scored_neighbourhoods`` kept), so the boxes drawn are exactly the boxes counted. The count
per class is checked against the run report's ``scored_cuboids``.
"""

from typing import Any, List, Mapping, Sequence

import numpy as np


def select_scored_cuboids(cuboids: Sequence[Any], object_box_ids: np.ndarray, object_class_ids: np.ndarray,
                          class_name_by_id: Mapping[int, str],
                          reported_counts_by_class_name: Mapping[str, int]) -> List[Any]:
    """Boxes of the given classes that the cells archive lists as scored objects.

    A box is selected when its (box id, class id) pair appears in the archive's object lists and its class id is
    in ``class_name_by_id``. Afterwards the number of selected boxes of each class must equal the report's count.

    Args:
        cuboids: All GT boxes of the session (``box_id``, ``class_id``).
        object_box_ids: (K,) box ids of the scored objects (cells archive).
        object_class_ids: (K,) class ids of the scored objects (cells archive).
        class_name_by_id: class id -> class name, for the classes to draw.
        reported_counts_by_class_name: class name -> ``scored_cuboids`` of the run report.

    Returns:
        The selected boxes, in the order of ``cuboids``.

    Raises:
        ValueError: When a class's selected count differs from the report's, or the report lacks a class.
    """
    # Step 1: the scored (box id, class id) pairs, as plain ints for hashing.
    scored_pairs = {(int(box_id), int(class_id)) for box_id, class_id in zip(object_box_ids, object_class_ids)}
    # Step 2: keep the boxes of the wanted classes whose pair is scored.
    selected = [cuboid for cuboid in cuboids if cuboid.class_id in class_name_by_id
                and (int(cuboid.box_id), int(cuboid.class_id)) in scored_pairs]
    # Step 3: per-class counts must equal the evaluation's own count.
    for class_id, class_name in class_name_by_id.items():
        if class_name not in reported_counts_by_class_name:
            raise ValueError(f"the run report has no scored_cuboids for class {class_name!r}")
        selected_count = sum(1 for cuboid in selected if cuboid.class_id == class_id)
        if selected_count != int(reported_counts_by_class_name[class_name]):
            raise ValueError(f"{class_name}: {selected_count} scored boxes selected but the run report counts "
                             f"{reported_counts_by_class_name[class_name]}")
    return selected


def class_box_counts(cuboids: Sequence[Any]) -> dict:
    """Number of boxes per class name.

    Args:
        cuboids: Boxes with ``class_name``.

    Returns:
        class name -> count, in order of first appearance.
    """
    counts: dict = {}
    for cuboid in cuboids:
        counts[cuboid.class_name] = counts.get(cuboid.class_name, 0) + 1
    return counts
