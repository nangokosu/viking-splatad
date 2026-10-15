"""Known-answer tests for lidar->camera reprojection (T25)."""

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from viking_splatad.align.spatial_check import (
    edge_alignment_score,
    project_lidar_to_image,
)

# Lidar axes (x fwd, y left, z up) -> camera axes (x right, y down, z fwd).
ROTATION_CAMERA_LIDAR = np.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
)


def make_extrinsic() -> np.ndarray:
    """T_camera_lidar with the axis permutation above and zero translation."""
    extrinsic = np.eye(4)
    extrinsic[:3, :3] = ROTATION_CAMERA_LIDAR
    return extrinsic


def make_intrinsics() -> tuple:
    """Pinhole K plus image size (synthetic values)."""
    focal, width, height = 500.0, 640, 480
    camera_matrix = np.array(
        [[focal, 0.0, width / 2], [0.0, focal, height / 2], [0.0, 0.0, 1.0]]
    )
    return camera_matrix, width, height


@pytest.mark.covers("T25")
def test_known_point_lands_at_analytic_pixel_and_validity_mask():
    """u = fx*X/Z + cx, v = fy*Y/Z + cy in the camera frame; behind/outside flagged."""
    camera_matrix, width, height = make_intrinsics()
    # Lidar point 10 m ahead, 1 m left, 0.5 m up -> camera (X,Y,Z) = (-1, -0.5, 10).
    points = np.array(
        [[10.0, 1.0, 0.5], [-5.0, 0.0, 0.0], [10.0, -1000.0, 0.0]]
    )
    pixels, valid = project_lidar_to_image(
        points, make_extrinsic(), camera_matrix, width, height
    )
    expected_u = 500.0 * (-1.0) / 10.0 + width / 2
    expected_v = 500.0 * (-0.5) / 10.0 + height / 2
    assert pixels[0] == pytest.approx([expected_u, expected_v])
    assert list(valid) == [True, False, False]  # in view, behind camera, off image


def build_scene():
    """Wall of lidar points with 3 vertical step edges plus the matching rendered image."""
    camera_matrix, width, height = make_intrinsics()
    wall_distance = 10.0
    edge_lateral_positions = np.array([-2.0, 0.0, 2.0])  # lidar y of the steps
    grid_y, grid_z = np.meshgrid(np.arange(-3.0, 3.01, 0.25), np.arange(-1.0, 1.01, 0.25))
    points = np.column_stack(
        [np.full(grid_y.size, wall_distance), grid_y.ravel(), grid_z.ravel()]
    )
    # Per-point depth-discontinuity measure: 1 on the edge columns, else 0.
    discontinuity = np.isin(points[:, 1], edge_lateral_positions).astype(float)
    # Render step edges at the projected u of the edge columns (ground-truth extrinsic).
    edge_pixels = (
        camera_matrix[0, 0] * (-edge_lateral_positions) / wall_distance + camera_matrix[0, 2]
    )
    column_index = np.arange(width)
    # Intensity = number of edges to the left of each column, then blurred.
    row = (column_index[:, None] > edge_pixels[None, :]).sum(axis=1).astype(np.float32)
    image = gaussian_filter(np.tile(row, (height, 1)), sigma=3.0)
    return points, discontinuity, image, camera_matrix, width, height


def rotate_about_camera_y(angle_radians: float) -> np.ndarray:
    """4x4 pure rotation about the camera y axis."""
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    rotation = np.eye(4)
    rotation[:3, :3] = [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]]
    return rotation


@pytest.mark.covers("T25")
def test_alignment_score_decreases_monotonically_with_rotation_error():
    """Growing yaw error moves edges off the image gradients -> score strictly falls."""
    points, discontinuity, image, camera_matrix, width, height = build_scene()
    errors_degrees = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    scores = []
    for error_degrees in errors_degrees:
        perturbed = rotate_about_camera_y(np.deg2rad(error_degrees)) @ make_extrinsic()
        pixels, valid = project_lidar_to_image(
            points, perturbed, camera_matrix, width, height
        )
        scores.append(edge_alignment_score(image, pixels, valid, discontinuity))
    assert np.all(np.diff(scores) < 0.0), scores
    assert scores[0] > 0.5  # perfectly aligned scene correlates strongly
