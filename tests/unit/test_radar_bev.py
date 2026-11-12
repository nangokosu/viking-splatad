"""Known pixel geometry for label-free radar/LiDAR BEV diagnostics."""

import cv2
import numpy as np
import pytest

from viking_splatad.viz.radar_bev import bev_pixels, write_radar_lidar_bev


@pytest.mark.covers("T47")
def test_bev_pixels_include_bounds_and_keep_y_up():
    """Minimum, centre and maximum map to exact corners/centre; outside/NaN are excluded."""
    points = np.array([[0., 0.], [1., 1.], [2., 2.], [3., 1.], [np.nan, 0.]])
    pixels, kept = bev_pixels(points, np.array([[0., 0.], [2., 2.]]), (3, 3))
    np.testing.assert_array_equal(pixels, [[2, 0], [1, 1], [0, 2]])
    np.testing.assert_array_equal(kept, [True, True, True, False, False])


def test_bev_picture_uses_the_same_raster_for_both_modalities(tmp_path):
    """A shared XY point is white; modality-only positions use their documented colors."""
    path = write_radar_lidar_bev(np.array([[0., 0., 0.], [1., 1., 0.]]),
                                 np.array([[1., 1., 2.], [2., 2., 2.]]), tmp_path / "view.png", 3)
    image = cv2.imread(str(path))
    np.testing.assert_array_equal(image[2, 0], [160, 160, 160])
    np.testing.assert_array_equal(image[1, 1], [255, 255, 255])
    np.testing.assert_array_equal(image[0, 2], [0, 140, 255])
    assert "not GT agreement" in path.with_suffix(".txt").read_text()
