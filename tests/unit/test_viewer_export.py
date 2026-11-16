"""Tests of viking_splatad.viz.viewer_export (T70) with synthetic known answers."""

import numpy as np
import pytest

from viking_splatad.viz.viewer_export import sweep_points_to_map


def axis_rotation_matrix(axis_index: int, angle_radians: float) -> np.ndarray:
    """Right-handed 3x3 rotation about one coordinate axis, from the textbook formula.

    Args:
        axis_index: 0 = x, 1 = y, 2 = z.
        angle_radians: Rotation angle.

    Returns:
        (3, 3) rotation matrix.
    """
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    # Rotation in the plane of the two other axes: [[c, -s], [s, c]] placed on those axes.
    first_axis, second_axis = [(1, 2), (2, 0), (0, 1)][axis_index]
    matrix = np.eye(3)
    matrix[first_axis, first_axis], matrix[first_axis, second_axis] = cosine, -sine
    matrix[second_axis, first_axis], matrix[second_axis, second_axis] = sine, cosine
    return matrix


@pytest.mark.covers("T70")
def test_points_follow_rotation_then_translation():
    """+90 deg about z sends (1, 0, 0) to (0, 1, 0); the translation (10, 20, 30) is added after."""
    pose = np.eye(4)
    pose[:3, :3] = axis_rotation_matrix(2, np.pi / 2)
    pose[:3, 3] = [10.0, 20.0, 30.0]
    points_map = sweep_points_to_map(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), pose)
    np.testing.assert_allclose(points_map, [[10.0, 21.0, 30.0], [10.0, 20.0, 30.0]], atol=1e-12)
