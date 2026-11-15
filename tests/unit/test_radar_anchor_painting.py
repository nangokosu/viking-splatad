"""Synthetic known-answer tests for painting radar seed colours from camera frames."""

import numpy as np

from viking_splatad.radar_init.seeds import NEUTRAL_GREY_RGB, paint_visible_seed_colors


def test_camera_painting_uses_grey_for_lidar_occlusion():
    """A closer LiDAR point inside the radar footprint blocks camera color leakage.

    Moved unchanged from tests/unit/test_radar_initialization.py (whose other tests covered the
    deleted seed builder). Its old marker T36 now names the anchor covariance transform, so the
    marker is dropped here; the T37 painting driver gets its own covering tests.
    """
    image = np.zeros((5, 5, 3), dtype=np.uint8)
    image[2, 2] = [12, 34, 56]
    intrinsics = np.array([[1.0, 0.0, 2.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
    colors, painted = paint_visible_seed_colors(
        np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0]]),
        np.eye(4), intrinsics, image, np.array([[0.0, 0.0, 3.0]]), 0.25, 0.1,
    )
    np.testing.assert_array_equal(colors[0], NEUTRAL_GREY_RGB)
    np.testing.assert_array_equal(colors[1], image[2, 2])
    np.testing.assert_array_equal(painted, [False, True])
