"""Unit tests for T31 (cuboid voxelisation) and T32 (per-frame GT crop)."""

import math

import numpy as np
import pytest

from viking_splatad.labels.cuboids import Cuboid
from viking_splatad.labels.occupancy import (
    OccupancyGrid,
    crop_occupancy_to_sensor_fov,
    voxelize_cuboids,
)


def make_box(box_id, class_id, center, dimensions, yaw_radians=0.0):
    """Build a Cuboid with a pure-yaw rotation (quaternion = (0, 0, sin(t/2), cos(t/2)))."""
    return Cuboid(
        box_id, class_id, f"class_{class_id}", tuple(center), tuple(dimensions),
        (0.0, 0.0, math.sin(yaw_radians / 2.0), math.cos(yaw_radians / 2.0)),
    )


@pytest.mark.covers("T31")
def test_axis_aligned_grid_aligned_box_volume_is_exact():
    """Box edges on voxel faces: occupied count * voxel volume == analytic volume."""
    voxel_size = 0.5
    dimensions = (3.0, 2.0, 1.0)
    box = make_box(1, 3, (0.0, 0.0, 0.0), dimensions)
    grid = voxelize_cuboids([box], voxel_size, margin=voxel_size)
    occupied_count = int(np.count_nonzero(grid.class_ids))
    assert occupied_count * voxel_size**3 == pytest.approx(np.prod(dimensions))
    assert set(np.unique(grid.class_ids)) == {0, 3}


@pytest.mark.covers("T31")
def test_rotated_box_volume_within_surface_area_bound():
    """Yawed box: |voxel volume - box volume| <= 2*sqrt(3)*voxel_size*surface_area.

    A voxel is misclassified only if its centre and corner-sphere straddle the
    surface, i.e. the voxel lies within sqrt(3)*voxel_size of the surface; the
    shell of that half-thickness on both sides has volume ~ 2*sqrt(3)*v*A.
    """
    voxel_size = 0.1
    length, width, height = 4.0, 2.0, 2.0
    box = make_box(1, 2, (0.3, -0.2, 0.1), (length, width, height), math.radians(30.0))
    grid = voxelize_cuboids([box], voxel_size)
    measured_volume = np.count_nonzero(grid.class_ids) * voxel_size**3
    analytic_volume = length * width * height
    surface_area = 2.0 * (length * width + length * height + width * height)
    error_bound = 2.0 * math.sqrt(3.0) * voxel_size * surface_area
    assert abs(measured_volume - analytic_volume) <= error_bound
    # Sanity: the bound is not vacuous (error well below total volume).
    assert error_bound < analytic_volume


@pytest.mark.covers("T31")
def test_overlap_respects_priority_order():
    """Overlapping voxels take the class that comes first in the priority list."""
    big_canopy = make_box(1, 2, (0.0, 0.0, 0.0), (4.0, 4.0, 4.0))
    small_rock = make_box(2, 3, (0.0, 0.0, 0.0), (2.0, 2.0, 2.0))
    voxel_size = 1.0
    default_grid = voxelize_cuboids([big_canopy, small_rock], voxel_size)
    swapped_grid = voxelize_cuboids(
        [big_canopy, small_rock], voxel_size, class_priority=(2, 3)
    )
    # Default priority (3 before 2): the 2x2x2 core -> 8 rock voxels, rest canopy.
    assert np.count_nonzero(default_grid.class_ids == 3) == 8
    assert np.count_nonzero(default_grid.class_ids == 2) == 64 - 8
    # Priority (2, 3): canopy wins everywhere, rock disappears.
    assert np.count_nonzero(swapped_grid.class_ids == 3) == 0
    assert np.count_nonzero(swapped_grid.class_ids == 2) == 64


@pytest.mark.covers("T32")
def test_fov_crop_returns_analytic_set():
    """Hand-built grid, yawed sensor: only in-range/in-FOV voxels come back."""
    voxel_size = 1.0
    grid_origin = np.array([8.0, -2.0, 0.0])
    class_ids = np.zeros((5, 24, 8), dtype=np.int32)
    # (voxel index) -> class; centres = origin + (index + 0.5).
    voxel_classes = {
        (2, 5, 0): 1,   # A centre (10.5, 3.5, 0.5): straight ahead, inside
        (2, 0, 0): 4,   # B centre (10.5, -1.5, 0.5): behind sensor, outside
        (0, 4, 0): 5,   # C centre (8.5, 2.5, 0.5): azimuth 45 deg > 30 deg, outside
        (2, 22, 0): 6,  # D centre (10.5, 20.5, 0.5): range 20 > 10, outside
        (2, 5, 6): 7,   # E centre (10.5, 3.5, 6.5): elevation 63 deg > 15 deg, outside
        (3, 7, 0): 2,   # F centre (11.5, 5.5, 0.5): azimuth -11.3 deg, inside
    }
    for voxel_index, class_id in voxel_classes.items():
        class_ids[voxel_index] = class_id
    grid = OccupancyGrid(class_ids, grid_origin, voxel_size)

    # Sensor at (10.5, 0.5, 0.5) yawed +90 deg: sensor-x -> map +y, sensor-y -> map -x.
    T_map_sensor = np.eye(4)
    T_map_sensor[:3, :3] = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    T_map_sensor[:3, 3] = [10.5, 0.5, 0.5]

    points_sensor, cropped_classes = crop_occupancy_to_sensor_fov(
        grid, T_map_sensor, max_range=10.0,
        horizontal_fov_radians=math.radians(60.0),
        vertical_fov_radians=math.radians(30.0),
    )
    # A: delta (0, 3, 0) -> (x_s, y_s, z_s) = (3, 0, 0).
    # F: delta (1, 5, 0) -> x_s = 5, y_s = -1, z_s = 0.
    actual = sorted(
        (tuple(np.round(point, 9)), int(class_id))
        for point, class_id in zip(points_sensor, cropped_classes)
    )
    expected = sorted([((3.0, 0.0, 0.0), 1), ((5.0, -1.0, 0.0), 2)])
    assert actual == expected
