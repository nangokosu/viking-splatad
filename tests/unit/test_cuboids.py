"""Unit tests for T29 (cuboid JSON parsing) and T30 (cuboid pose geometry)."""

import itertools
import math

import numpy as np
import pytest

from viking_splatad.labels.cuboids import (
    Cuboid,
    compute_cuboid_geometry,
    parse_cuboids,
    points_in_cuboid,
)


def build_annotation(annotation_id, class_id, position, dimensions, quaternion):
    """Build one Segments.ai-style cuboid annotation dict.

    Args:
        annotation_id: Annotation id.
        class_id: Integer class id.
        position: (x, y, z) centre.
        dimensions: (x, y, z) full sizes.
        quaternion: (qx, qy, qz, qw).

    Returns:
        Annotation dict as stored in the labels JSON.
    """
    return {
        "id": annotation_id,
        "category_id": class_id,
        "type": "cuboid",
        "position": dict(zip("xyz", position)),
        "dimensions": dict(zip("xyz", dimensions)),
        "rotation": dict(zip(("qx", "qy", "qz", "qw"), quaternion)),
        "yaw": 0.0,
    }


def build_yaw_quaternion(yaw_radians):
    """Return the (x, y, z, w) quaternion of a pure rotation about +z."""
    # Axis-angle -> quaternion: (axis*sin(theta/2), cos(theta/2)) with axis = z.
    return (0.0, 0.0, math.sin(yaw_radians / 2.0), math.cos(yaw_radians / 2.0))


@pytest.mark.covers("T29")
def test_parse_cuboids_returns_typed_boxes_with_correct_fields():
    """Nested Segments-like JSON -> typed Cuboid list with names mapped."""
    yaw_quaternion = build_yaw_quaternion(math.pi / 2.0)
    annotations = [
        build_annotation(7, 1, (1.0, 2.0, 3.0), (0.5, 0.6, 4.0), (0, 0, 0, 1)),
        build_annotation(8, 5, (10.0, 0.0, 1.0), (4.0, 2.0, 1.5), yaw_quaternion),
    ]
    document = {
        "dataset": {
            "samples": [
                {
                    "name": "5cm_map_2024_05_dense.ply",
                    "labels": {
                        "ground-truth": {"attributes": {"annotations": annotations}}
                    },
                }
            ]
        }
    }
    boxes = parse_cuboids(document)
    assert len(boxes) == 2
    assert boxes[0] == Cuboid(7, 1, "tree_trunk", (1.0, 2.0, 3.0), (0.5, 0.6, 4.0), (0.0, 0.0, 0.0, 1.0))
    assert boxes[1].class_name == "car"
    assert boxes[1].dimensions_xyz == (4.0, 2.0, 1.5)
    assert boxes[1].quaternion_xyzw == pytest.approx(yaw_quaternion)


@pytest.mark.covers("T29")
def test_parse_cuboids_custom_class_names_and_top_level_list():
    """A bare list of annotations parses; class-name table is a parameter."""
    annotations = [build_annotation(1, 9, (0, 0, 0), (1, 1, 1), (0, 0, 0, 1))]
    boxes = parse_cuboids(annotations, class_names={9: "custom"})
    assert [box.class_name for box in boxes] == ["custom"]


@pytest.mark.covers("T29")
def test_parse_cuboids_raises_clear_error_when_no_annotations_found():
    """Unrecognised structure -> ValueError naming what was searched."""
    with pytest.raises(ValueError, match="annotations"):
        parse_cuboids({"unrelated": {"a": [1, 2, 3]}})


@pytest.mark.covers("T30")
def test_axis_aligned_box_corners_are_analytic():
    """Identity rotation: corners are centre +/- half full-size."""
    box = Cuboid(1, 3, "rock", (1.0, 2.0, 3.0), (2.0, 4.0, 6.0), (0.0, 0.0, 0.0, 1.0))
    geometry = compute_cuboid_geometry(box)
    assert geometry.half_extents == pytest.approx([1.0, 2.0, 3.0])
    expected_corners = {
        (1.0 + sign_x * 1.0, 2.0 + sign_y * 2.0, 3.0 + sign_z * 3.0)
        for sign_x, sign_y, sign_z in itertools.product((-1.0, 1.0), repeat=3)
    }
    actual_corners = {tuple(np.round(corner, 9)) for corner in geometry.corners}
    assert actual_corners == expected_corners
    assert geometry.rotation_map_box == pytest.approx(np.eye(3))


@pytest.mark.covers("T30")
def test_yaw_90_box_corners_and_point_in_box():
    """90 deg yaw maps box-x to map +y, box-y to map -x; check corners and membership."""
    box = Cuboid(2, 5, "car", (1.0, 2.0, 3.0), (4.0, 2.0, 2.0), build_yaw_quaternion(math.pi / 2.0))
    geometry = compute_cuboid_geometry(box)
    # Local (lx, ly, lz) -> map offset (-ly, lx, lz); half extents (2, 1, 1).
    expected_corners = {
        (1.0 - sign_y * 1.0, 2.0 + sign_x * 2.0, 3.0 + sign_z * 1.0)
        for sign_x, sign_y, sign_z in itertools.product((-1.0, 1.0), repeat=3)
    }
    actual_corners = {tuple(np.round(corner, 9)) for corner in geometry.corners}
    assert actual_corners == expected_corners

    query_points = np.array(
        [
            [1.0, 2.0, 3.0],  # centre -> inside
            [2.0, 2.0, 3.0],  # on the +/-y local face (|local y| = 1) -> inside
            [1.0, 4.0, 3.0],  # on the local +x face (|local x| = 2) -> inside
            [1.0, 5.0, 3.0],  # local x = 3 > 2 -> outside
            [3.5, 2.0, 3.0],  # local y = -2.5 -> outside (would be inside if unrotated)
            [1.0, 2.0, 4.5],  # z offset 1.5 > 1 -> outside
        ]
    )
    assert points_in_cuboid(geometry, query_points).tolist() == [
        True, True, True, False, False, False,
    ]
