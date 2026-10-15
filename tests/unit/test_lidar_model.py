"""Unit tests for contract/lidar_model.py (T15 ego box, T16 elevations, T17 azimuth tiles)."""

import math

import numpy as np
import pytest

from viking_splatad.contract.lidar_model import (
    derive_azimuth_tiling,
    derive_elevation_mapping,
    derive_min_lidar_dist,
)


@pytest.mark.covers("T15")
def test_derive_min_lidar_dist_encloses_body_and_excludes_scene():
    """Box-shaped body cluster at origin + far scene -> thresholds enclose body only."""
    rng = np.random.default_rng(seed=11)
    half_extents_true = np.array([0.8, 0.5, 0.3])  # body box half sizes (fixture)
    body_count, scene_count = 3000, 6000
    # Body points on the box surface: pick a face axis, pin it to +-half extent.
    body = rng.uniform(-1, 1, size=(body_count, 3)) * half_extents_true
    face_axis = rng.integers(0, 3, size=body_count)
    face_sign = rng.choice([-1.0, 1.0], size=body_count)
    body[np.arange(body_count), face_axis] = face_sign * half_extents_true[face_axis]
    # Scene: uniform directions at ranges 4..40 m (well beyond the body diagonal ~1 m).
    directions = rng.normal(size=(scene_count, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    scene = directions * rng.uniform(4.0, 40.0, size=(scene_count, 1))
    cloud = np.concatenate([body, scene])

    margin = 0.05
    min_lidar_dist = np.array(derive_min_lidar_dist(cloud, margin=margin))
    # Enclosure: thresholds equal the half extents + margin (exact extents are hit by
    # face points up to sampling), so >= true extents and <= true + margin + tiny.
    assert np.all(min_lidar_dist >= half_extents_true + margin - 1e-9)
    assert np.all(min_lidar_dist <= half_extents_true + margin + 1e-9)
    # Keep rule: kept iff |x|>=dx or |y|>=dy or |z|>=dz.
    kept = np.any(np.abs(cloud) >= min_lidar_dist, axis=1)
    assert not kept[:body_count].any()
    assert kept[body_count:].all()


@pytest.mark.covers("T15")
def test_derive_min_lidar_dist_without_gap_raises():
    """A continuous radial distribution has no body gap -> explicit failure."""
    rng = np.random.default_rng(seed=2)
    directions = rng.normal(size=(5000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    cloud = directions * rng.uniform(0.01, 30.0, size=(5000, 1))
    with pytest.raises(ValueError):
        derive_min_lidar_dist(cloud, margin=0.05)


def build_synthetic_scan(beam_elevations_deg, noise_sigma_deg, points_per_beam, seed):
    """Make a scan with known per-beam elevations plus Gaussian elevation noise.

    Returns:
        (xyz, ring): sensor-frame points at 10 m range, and their beam indices.
    """
    rng = np.random.default_rng(seed)
    beam_count = len(beam_elevations_deg)
    ring = np.repeat(np.arange(beam_count), points_per_beam)
    elevation = np.radians(
        beam_elevations_deg[ring] + rng.normal(0, noise_sigma_deg, size=ring.size)
    )
    azimuth = rng.uniform(-np.pi, np.pi, size=ring.size)
    range_metres = 10.0
    xyz = range_metres * np.stack(
        [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)],
        axis=1,
    )
    return xyz, ring


@pytest.mark.covers("T16")
def test_derive_elevation_mapping_with_ring_and_without():
    """32 beams -45..45 deg: recovered within a noise-derived tolerance, both modes."""
    beam_count = 32
    beam_elevations_deg = np.linspace(-45.0, 45.0, beam_count)
    noise_sigma_deg, points_per_beam = 0.05, 400
    xyz, ring = build_synthetic_scan(beam_elevations_deg, noise_sigma_deg, points_per_beam, seed=5)
    # Median of N noisy samples has std ~ 1.2533*sigma/sqrt(N); allow 5 of those.
    tolerance_deg = 5 * 1.2533 * noise_sigma_deg / math.sqrt(points_per_beam)

    with_ring = derive_elevation_mapping(xyz, ring=ring)
    assert list(with_ring) == sorted(with_ring)  # keys listed in ascending elevation order
    assert list(with_ring.values()) == sorted(with_ring.values())
    assert len(with_ring) == beam_count
    np.testing.assert_allclose(list(with_ring.values()), beam_elevations_deg, atol=tolerance_deg)

    without_ring = derive_elevation_mapping(xyz)
    assert list(without_ring) == list(range(beam_count))
    np.testing.assert_allclose(list(without_ring.values()), beam_elevations_deg, atol=tolerance_deg)


@pytest.mark.covers("T16")
def test_derive_elevation_mapping_rejects_non_multiple_of_eight():
    """30 beams is not a multiple of 8 -> ValueError."""
    beam_elevations_deg = np.linspace(-20.0, 20.0, 30)
    xyz, ring = build_synthetic_scan(beam_elevations_deg, 0.02, 100, seed=1)
    with pytest.raises(ValueError):
        derive_elevation_mapping(xyz, ring=ring)


@pytest.mark.covers("T17")
@pytest.mark.parametrize(
    "columns, expected_tile_width",
    [(1024, 32), (512, 16)],
)
def test_azimuth_tiling_valid_columns(columns, expected_tile_width):
    """res = 360/columns; tile_width = ceil(360/(res*32)) = columns/32."""
    resolution_deg, tile_width = derive_azimuth_tiling(columns)
    assert resolution_deg == pytest.approx(360.0 / columns)
    assert tile_width == expected_tile_width
    # Per-ring measured counts (median taken) behave identically.
    resolution_from_rings, tile_from_rings = derive_azimuth_tiling([columns] * 8)
    assert (resolution_from_rings, tile_from_rings) == (resolution_deg, tile_width)


@pytest.mark.covers("T17")
def test_azimuth_tiling_1024_resolution_value():
    """1024 columns -> 0.3515625 deg (the 0.3516 quoted in the plan)."""
    resolution_deg, _ = derive_azimuth_tiling(1024)
    assert resolution_deg == pytest.approx(0.3516, abs=5e-5)


@pytest.mark.covers("T17")
def test_azimuth_tiling_rejects_non_integral():
    """700 columns -> 21.875 tiles, not integral -> rejected."""
    with pytest.raises(ValueError):
        derive_azimuth_tiling(700)
