"""Ego-filter format: the manifest's ``min_lidar_dist`` as a box (what extraction writes) or a radius.

Regression for the first CPU preflight on real data: every real manifest stores a
(dx, dy, dz) box, while the dataparser reader only accepted a scalar and crashed with
``float() argument must be ... not 'list'``.
"""

import numpy as np
import pytest

from viking_splatad.extract.lidar import assemble_scan_array
from viking_splatad.splatad.viking_conversions import (
    load_lidar_sweep,
    neurad_lidar_model_from_manifest,
    parse_ego_filter,
)

# Box half-extents in the shape and magnitude of the real may_chunks0_1_smoke manifest.
REAL_FORMAT_BOX = [0.4791892945766449, 0.44140546321868895, 0.3919963717460632]


def minimal_lidar_manifest(min_lidar_dist) -> dict:
    """Smallest valid ``manifest["lidar"]`` block: 8 rings, 1024 columns, given ego filter.

    Args:
        min_lidar_dist: value stored under ``min_lidar_dist``.

    Returns:
        Manifest lidar dict.
    """
    ring_count = 8
    return {
        "lidar_type": "OUSTER_OS0_32",
        "elevation_mapping": {str(ring): float(np.deg2rad(10.0 - ring)) for ring in range(ring_count)},
        "azimuth_resolution": 2.0 * np.pi / 1024,
        "azimuth_tile_width": 32,
        "min_lidar_dist": min_lidar_dist,
        "intensity_scale": 1.0,
        "sweep_period": 0.1,
    }


@pytest.mark.covers("T27")
def test_parse_ego_filter_scalar_is_radius_and_vector_is_box():
    """Scalar -> (radius, zero box); 3-vector -> (0, box); any other shape is rejected."""
    assert parse_ego_filter(0.8) == (0.8, (0.0, 0.0, 0.0))
    radius, box = parse_ego_filter(REAL_FORMAT_BOX)
    assert radius == 0.0 and box == tuple(REAL_FORMAT_BOX)
    with pytest.raises(ValueError):
        parse_ego_filter([0.5, 0.5])


@pytest.mark.covers("T27")
def test_lidar_model_accepts_real_box_format():
    """The real manifest format (a list) builds a model with the box set and the radius off."""
    model = neurad_lidar_model_from_manifest(minimal_lidar_manifest(REAL_FORMAT_BOX))
    assert model.min_lidar_radius == 0.0
    assert model.ego_box_half_extents == tuple(REAL_FORMAT_BOX)


@pytest.mark.covers("T27")
def test_load_lidar_sweep_box_keeps_points_outside_on_any_axis(tmp_path):
    """Box test: keep iff |x|>=dx or |y|>=dy or |z|>=dz (boundary counts as outside)."""
    half_extents = (0.5, 0.4, 0.3)
    xyz = np.array([
        [0.1, 0.1, 0.1],     # inside on every axis -> ego
        [0.6, 0.0, 0.0],     # outside on x -> keep
        [0.0, -0.45, 0.0],   # outside on y (|y| = 0.45 >= 0.4) -> keep
        [0.0, 0.0, 0.3],     # exactly on the z boundary -> keep (>=)
        [0.49, 0.39, 0.29],  # just inside on every axis -> ego
        [10.0, 0.0, 0.0],    # far scene point -> keep
    ])
    # Expected mask from first principles, axis by axis.
    expected_keep = np.zeros(len(xyz), dtype=bool)
    for point_index, point in enumerate(xyz):
        expected_keep[point_index] = any(abs(point[axis]) >= half_extents[axis] for axis in range(3))
    sweep = np.column_stack([xyz, np.full(len(xyz), 0.5), np.zeros(len(xyz)), np.arange(len(xyz))]).astype(np.float32)
    path = tmp_path / "sweep.npy"
    np.save(path, sweep)
    kept = load_lidar_sweep(path, min_lidar_radius=0.0, ego_box_half_extents=half_extents)
    # Channel column carries the row id, so the kept set is checked by identity.
    np.testing.assert_array_equal(kept[:, 5], np.flatnonzero(expected_keep).astype(np.float32))


@pytest.mark.covers("T27")
def test_box_filter_is_idempotent_with_extraction(tmp_path):
    """Sweeps written by extraction (already box-filtered) pass the reader's box test unchanged."""
    random_generator = np.random.default_rng(0)
    point_count = 2000
    # Points uniformly in a 4 m cube: a good share fall inside the ego box.
    xyz = random_generator.uniform(-2.0, 2.0, size=(point_count, 3))
    extracted = assemble_scan_array(
        xyz, np.full(point_count, 0.5), np.zeros(point_count), np.zeros(point_count), tuple(REAL_FORMAT_BOX))
    # Extraction must have dropped some points for this test to mean anything.
    assert 0 < len(extracted) < point_count
    path = tmp_path / "sweep.npy"
    np.save(path, extracted)
    _, box = parse_ego_filter(REAL_FORMAT_BOX)
    reloaded = load_lidar_sweep(path, min_lidar_radius=0.0, ego_box_half_extents=box)
    np.testing.assert_array_equal(reloaded, extracted)
