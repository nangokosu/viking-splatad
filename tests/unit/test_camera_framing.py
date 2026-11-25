"""Known-answer tests of the tight oblique-camera fit (pure numpy, no PyVista)."""

import numpy as np
import pytest

from viking_splatad.viz.camera_framing import (
    camera_basis,
    tight_oblique_camera,
    tight_projected_aspect,
)

# Test scene constants (synthetic): oblique direction, view-up, vertical field of view.
TEST_DIRECTION = (1.0, -1.0, 0.9)
TEST_VIEW_UP = (0.0, 0.0, 1.0)
TEST_VIEW_ANGLE_DEGREES = 30.0
TEST_VOXEL_SIZE = 0.5
TEST_ASPECT = 1.6
# Tolerance of float comparisons.
TOLERANCE = 1e-9
# Two voxel centres spanning a box of edge 1 m once padded by half a voxel (centres 0.5 m apart, voxel 0.5 m).
CUBE_POINTS = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
# Long flat scene: 100 m x 10 m x 1 m of centres.
FLAT_POINTS = np.array([[0.0, 0.0, 0.0], [100.0, 10.0, 1.0]])


def reference_projection(corner_array, direction, view_up, focal_point):
    """Independent projection of corners: returns (right_offset, up_offset, toward_camera_offset) columns."""
    # Camera looks along -direction; build the frame by hand with explicit cross products.
    toward_camera = np.asarray(direction, float) / np.linalg.norm(direction)
    right_axis = np.cross(view_up, toward_camera)  # = forward x up, since forward = -toward_camera
    right_axis /= np.linalg.norm(right_axis)
    up_axis = np.cross(toward_camera, right_axis)  # = right x forward
    offsets = corner_array - focal_point
    return offsets @ right_axis, offsets @ up_axis, offsets @ toward_camera


def test_basis_orthonormal_right_handed():
    forward, right, true_up = camera_basis(TEST_DIRECTION, TEST_VIEW_UP)
    basis = np.stack([forward, right, true_up])
    assert np.allclose(basis @ basis.T, np.eye(3), atol=TOLERANCE)
    # Forward points from camera to focal point, i.e. against the direction.
    assert np.allclose(forward, -np.asarray(TEST_DIRECTION) / np.linalg.norm(TEST_DIRECTION))
    # right = forward x view_up (normalised); true_up = right x forward.
    assert np.allclose(np.cross(forward, right), -true_up, atol=TOLERANCE)  # f x r = -(r x f)
    assert np.allclose(np.cross(right, forward), true_up, atol=TOLERANCE)
    # true_up keeps a positive component along the requested view-up.
    assert true_up @ np.asarray(TEST_VIEW_UP) > 0


def test_basis_parallel_direction_raises():
    with pytest.raises(ValueError):
        camera_basis((0.0, 0.0, 2.0), TEST_VIEW_UP)
    with pytest.raises(ValueError):
        camera_basis((0.0, 0.0, -1.0), TEST_VIEW_UP)


# ---- tight fit on the points themselves ----
# A scene along the ground diagonal: its axis-aligned box has far more empty corner than the points.
DIAGONAL_POINTS = np.array([[float(step), float(step), 0.1 * float(step)] for step in range(0, 101, 10)])


def test_tight_camera_keeps_every_point_in_view_and_one_on_the_edge():
    """Every point projects inside the frustum; the binding point sits exactly on a frustum edge."""
    camera = tight_oblique_camera(DIAGONAL_POINTS, TEST_ASPECT, 1.0, TEST_DIRECTION, TEST_VIEW_UP,
                                  TEST_VIEW_ANGLE_DEGREES)
    position, focal_point = np.array(camera[0]), np.array(camera[1])
    forward, right, true_up = camera_basis(TEST_DIRECTION, TEST_VIEW_UP)
    offsets = DIAGONAL_POINTS - position
    depth = offsets @ forward
    half_tangent = np.tan(np.radians(TEST_VIEW_ANGLE_DEGREES) / 2)
    vertical_ratio = np.abs(offsets @ true_up) / depth / half_tangent
    horizontal_ratio = np.abs(offsets @ right) / depth / (TEST_ASPECT * half_tangent)
    worst_ratio = np.maximum(vertical_ratio, horizontal_ratio)
    assert (depth > 0).all()
    assert worst_ratio.max() == pytest.approx(1.0, abs=TOLERANCE)
    assert np.allclose(np.cross(forward, focal_point - position), 0.0, atol=1e-6)  # looks at the focal point


def test_tight_projected_aspect_matches_reference_projection():
    """Extents along the hand-built right / up axes."""
    focal = DIAGONAL_POINTS.mean(axis=0)
    right_offsets, up_offsets, _ = reference_projection(DIAGONAL_POINTS, TEST_DIRECTION, TEST_VIEW_UP, focal)
    expected = np.ptp(right_offsets) / np.ptp(up_offsets)
    assert tight_projected_aspect(DIAGONAL_POINTS, TEST_DIRECTION, TEST_VIEW_UP) == pytest.approx(expected)


def test_tight_camera_rejects_bad_input():
    """Empty points and non-positive aspect raise."""
    with pytest.raises(ValueError):
        tight_oblique_camera(np.zeros((0, 3)), 1.0, 1.0, TEST_DIRECTION, TEST_VIEW_UP, TEST_VIEW_ANGLE_DEGREES)
    with pytest.raises(ValueError):
        tight_oblique_camera(DIAGONAL_POINTS, 0.0, 1.0, TEST_DIRECTION, TEST_VIEW_UP, TEST_VIEW_ANGLE_DEGREES)
