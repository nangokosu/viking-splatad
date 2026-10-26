"""Unit tests for the Ouster no-return sidecar (Phase B, option C): T51-T55 and the file layout.

Every expected value is built from the definitions: synthetic beams with chosen far-field angles
and parallax, poses with a chosen yaw rate, and hand-made sidecar rows.
"""

from pathlib import Path

import numpy as np
import pytest

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.dataset.schema import NoReturnStream
from viking_splatad.dataset.writer import read_dataset, write_dataset
from viking_splatad.extract.no_return import (
    NO_RETURN_ROW_WIDTH,
    BeamDirectionModel,
    beam_unit_directions,
    derive_beam_direction_model,
    no_return_cells,
    no_return_sweep_rows,
)
from viking_splatad.splatad.viking_conversions import no_return_path_for_sweep, no_return_placeholder_points

# Synthetic beam layout: 8 rings x 64 columns (small, but the same organisation as the OS0-32).
SYNTHETIC_RING_COUNT, SYNTHETIC_COLUMN_COUNT = 8, 64
# Gate and diagnostic percentile passed to T51 in the known-answer tests (exact data: any gate).
TEST_MAX_MEDIAN_RESIDUAL_DEG = 0.1
TEST_RESIDUAL_PERCENTILE = 99.0
# Recovery tolerance for exact synthetic data: least squares in float64 on O(1) angles.
FIT_TOLERANCE_DEG = 1e-8
# Placeholder distance used by neurad's missing points (ad_dataparser.DUMMY_DISTANCE_VALUE).
NEURAD_DUMMY_DISTANCE_M = 2000.0
LIDAR_CONFIG = {"range_field": "range", "ring_field": "ring", "point_time_field": "t", "point_time_unit_seconds": 1e-9}


def synthetic_beam_returns(column_step_sign: float, seed: int = 0):
    """Returns of synthetic beams that follow the T51 model exactly.

    Ring r: far-field elevation -30 + 8r deg, elevation parallax 0.05 + 0.01r deg*m, azimuth of
    column 0 at 175 + r deg (so the +-180 seam is crossed), azimuth parallax 0.02 deg*m. Each
    (ring, column) cell is seen at 3 random ranges in [0.5, 30] m.

    Args:
        column_step_sign: +1 (counter-clockwise columns) or -1 (clockwise, as the Ouster).
        seed: RNG seed for the ranges.

    Returns:
        (points_xyz, ring, column, truth dict of the far-field parameters).
    """
    rng = np.random.default_rng(seed)
    observations_per_cell = 3
    ring_ids = np.arange(SYNTHETIC_RING_COUNT)
    truth = {
        "elevation": {int(ring_id): -30.0 + 8.0 * ring_id for ring_id in ring_ids},
        "elevation_parallax": {int(ring_id): 0.05 + 0.01 * ring_id for ring_id in ring_ids},
        "azimuth_offset": {int(ring_id): 175.0 + ring_id for ring_id in ring_ids},
        "azimuth_parallax": {int(ring_id): 0.02 for ring_id in ring_ids},
        "column_step": column_step_sign * 360.0 / SYNTHETIC_COLUMN_COUNT,
    }
    ring_grid, column_grid, _ = np.meshgrid(
        ring_ids, np.arange(SYNTHETIC_COLUMN_COUNT), np.arange(observations_per_cell), indexing="ij")
    ring = ring_grid.ravel()
    column = column_grid.ravel()
    ranges = rng.uniform(0.5, 30.0, ring.size)
    # Measured angles = far-field angle + parallax slope / range (the T51 model itself).
    elevation = np.array([truth["elevation"][r] + truth["elevation_parallax"][r] / d for r, d in zip(ring, ranges)])
    azimuth = np.array([
        truth["azimuth_offset"][r] + truth["column_step"] * c + truth["azimuth_parallax"][r] / d
        for r, c, d in zip(ring, column, ranges)
    ])
    elevation_rad, azimuth_rad = np.radians(elevation), np.radians(azimuth)
    points = ranges[:, None] * np.column_stack([
        np.cos(elevation_rad) * np.cos(azimuth_rad), np.cos(elevation_rad) * np.sin(azimuth_rad), np.sin(elevation_rad)])
    return points, ring, column, truth


def angle_difference_deg(first: float, second: float) -> float:
    """Absolute difference of two angles on the circle, in degrees."""
    return abs((first - second + 180.0) % 360.0 - 180.0)


@pytest.mark.covers("T51")
@pytest.mark.parametrize("column_step_sign", [1.0, -1.0])
def test_beam_fit_recovers_far_field_angles_and_column_direction(column_step_sign):
    """Exact model data: far-field elevations, azimuth offsets, parallax slopes and step come back."""
    points, ring, column, truth = synthetic_beam_returns(column_step_sign)
    model = derive_beam_direction_model(
        points, ring, column, SYNTHETIC_COLUMN_COUNT, range(SYNTHETIC_RING_COUNT),
        TEST_MAX_MEDIAN_RESIDUAL_DEG, TEST_RESIDUAL_PERCENTILE)
    assert model.column_step_deg == pytest.approx(truth["column_step"], abs=FIT_TOLERANCE_DEG)
    for ring_id in range(SYNTHETIC_RING_COUNT):
        assert model.elevation_deg[ring_id] == pytest.approx(truth["elevation"][ring_id], abs=FIT_TOLERANCE_DEG)
        assert model.elevation_parallax_deg_m[ring_id] == pytest.approx(
            truth["elevation_parallax"][ring_id], abs=FIT_TOLERANCE_DEG)
        assert angle_difference_deg(model.azimuth_offset_deg[ring_id], truth["azimuth_offset"][ring_id]) < FIT_TOLERANCE_DEG
        assert model.azimuth_parallax_deg_m[ring_id] == pytest.approx(
            truth["azimuth_parallax"][ring_id], abs=FIT_TOLERANCE_DEG)
    assert model.median_residual_deg < FIT_TOLERANCE_DEG
    assert model.column_count == SYNTHETIC_COLUMN_COUNT


@pytest.mark.covers("T51")
def test_beam_fit_gate_rejects_a_cloud_the_model_cannot_describe():
    """Directions unrelated to (ring, column) leave a large median residual: the fit refuses."""
    rng = np.random.default_rng(1)
    _, ring, column, _ = synthetic_beam_returns(-1.0)
    # Uniform random directions on the sphere (normalised Gaussian vectors), random ranges.
    random_directions = rng.normal(size=(ring.size, 3))
    random_directions /= np.linalg.norm(random_directions, axis=1, keepdims=True)
    points = random_directions * rng.uniform(1.0, 30.0, ring.size)[:, None]
    with pytest.raises(ValueError, match="median residual"):
        derive_beam_direction_model(points, ring, column, SYNTHETIC_COLUMN_COUNT, range(SYNTHETIC_RING_COUNT),
                                    TEST_MAX_MEDIAN_RESIDUAL_DEG, TEST_RESIDUAL_PERCENTILE)


@pytest.mark.covers("T51")
def test_beam_fit_needs_every_expected_ring_and_two_ranges_per_ring():
    """A ring without returns, or with a single range, cannot be fitted."""
    points, ring, column, _ = synthetic_beam_returns(-1.0)
    missing_ring = SYNTHETIC_RING_COUNT
    with pytest.raises(ValueError, match="no returns"):
        derive_beam_direction_model(points, ring, column, SYNTHETIC_COLUMN_COUNT, range(missing_ring + 1),
                                    TEST_MAX_MEDIAN_RESIDUAL_DEG, TEST_RESIDUAL_PERCENTILE)
    # Put every ring-0 return at the same range: the 1/range column becomes constant.
    ring_zero = ring == 0
    same_range_points = points.copy()
    same_range_points[ring_zero] /= np.linalg.norm(points[ring_zero], axis=1, keepdims=True)
    with pytest.raises(ValueError, match="distinct ranges"):
        derive_beam_direction_model(same_range_points, ring, column, SYNTHETIC_COLUMN_COUNT,
                                    range(SYNTHETIC_RING_COUNT), TEST_MAX_MEDIAN_RESIDUAL_DEG, TEST_RESIDUAL_PERCENTILE)


@pytest.mark.covers("T51")
def test_beam_model_dict_round_trip_is_exact():
    """to_dict -> from_dict gives the same model (int ring keys restored)."""
    points, ring, column, _ = synthetic_beam_returns(-1.0)
    model = derive_beam_direction_model(points, ring, column, SYNTHETIC_COLUMN_COUNT, range(SYNTHETIC_RING_COUNT),
                                        TEST_MAX_MEDIAN_RESIDUAL_DEG, TEST_RESIDUAL_PERCENTILE)
    assert BeamDirectionModel.from_dict(model.to_dict()) == model


def hand_made_model() -> BeamDirectionModel:
    """4 columns of -90 deg; ring 0 horizontal with column 0 at +90 deg; ring 1 straight up."""
    zero_by_ring = {0: 0.0, 1: 0.0}
    return BeamDirectionModel(
        column_count=4, column_step_deg=-90.0, elevation_deg={0: 0.0, 1: 90.0},
        azimuth_offset_deg={0: 90.0, 1: 0.0}, elevation_parallax_deg_m=zero_by_ring,
        azimuth_parallax_deg_m=zero_by_ring, median_residual_deg=0.0, residual_percentile=99.0,
        percentile_residual_deg=0.0)


@pytest.mark.covers("T52")
def test_beam_unit_directions_match_hand_computed_vectors():
    """Ring 0 columns 0,1,2 point along +y, +x, -y; ring 1 points along +z."""
    directions = beam_unit_directions(hand_made_model(), np.array([0, 0, 0, 1]), np.array([0, 1, 2, 3]))
    expected = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
    # cos(pi/2) evaluates to ~6e-17 in float64, hence the absolute tolerance.
    np.testing.assert_allclose(directions, expected, atol=1e-12)
    with pytest.raises(ValueError, match="not in the beam model"):
        beam_unit_directions(hand_made_model(), np.array([7]), np.array([0]))


@pytest.mark.covers("T53")
def test_no_return_cells_finds_range_zero_cells_of_an_organized_cloud():
    """2 rings x 4 columns, range 0 at cells 1, 3, 4, 6: exact ring, column and time."""
    fields = {
        "range": np.array([5, 0, 7, 0, 0, 3, 0, 2], dtype=np.uint32),
        "ring": np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.uint16),
        "t": (np.arange(8) * 1000).astype(np.uint32),
    }
    ring, column, time_seconds = no_return_cells(fields, column_count=4, row_count=2, lidar_config=LIDAR_CONFIG)
    np.testing.assert_array_equal(ring, [0, 0, 1, 1])
    np.testing.assert_array_equal(column, [1, 3, 0, 2])
    np.testing.assert_allclose(time_seconds, np.array([1000, 3000, 4000, 6000]) * 1e-9, rtol=0, atol=0)


@pytest.mark.covers("T53")
def test_no_return_cells_rejects_an_unorganized_cloud():
    """Height 1, or a cell count that is not width*height, has no (ring, column) grid."""
    fields = {"range": np.zeros(8, dtype=np.uint32), "ring": np.zeros(8, dtype=np.uint16),
              "t": np.zeros(8, dtype=np.uint32)}
    with pytest.raises(ValueError, match="not organized"):
        no_return_cells(fields, column_count=8, row_count=1, lidar_config=LIDAR_CONFIG)
    with pytest.raises(ValueError, match="not organized"):
        no_return_cells(fields, column_count=3, row_count=2, lidar_config=LIDAR_CONFIG)


@pytest.mark.covers("T54")
def test_no_return_rows_rotate_directions_by_the_yaw_since_the_reference():
    """Constant yaw rate w and a translation: R_ref^-1 R(t) d = Rz(w (t - t_ref)) d; translation drops out."""
    yaw_rate, speed, reference_time = 0.5, 2.0, 10.0

    def pose_at(query_times):
        """Lidar pose: yaw w*t about z, position (v*t, 0, 0)."""
        poses = np.tile(np.eye(4), (len(query_times), 1, 1))
        angles = yaw_rate * np.asarray(query_times)
        poses[:, 0, 0], poses[:, 0, 1] = np.cos(angles), -np.sin(angles)
        poses[:, 1, 0], poses[:, 1, 1] = np.sin(angles), np.cos(angles)
        poses[:, 0, 3] = speed * np.asarray(query_times)
        return poses

    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    capture_times = np.array([reference_time + 0.04, reference_time - 0.02])
    rows = no_return_sweep_rows(directions, capture_times, pose_at, reference_time,
                                channel=np.array([3, 5]), column=np.array([17, 900]))
    turned_angle = yaw_rate * 0.04
    expected_rows = np.array([
        [np.cos(turned_angle), np.sin(turned_angle), 0.0, 0.04, 3, 17],
        [0.0, 0.0, 1.0, -0.02, 5, 900],
    ], dtype=np.float32)
    assert rows.dtype == np.float32 and rows.shape == (2, NO_RETURN_ROW_WIDTH)
    # Stored as float32: compare at float32 resolution.
    np.testing.assert_allclose(rows, expected_rows, rtol=0, atol=float(np.finfo(np.float32).eps) * 1e3)
    empty_rows = no_return_sweep_rows(np.zeros((0, 3)), np.zeros(0), pose_at, reference_time,
                                      np.zeros(0), np.zeros(0))
    assert empty_rows.shape == (0, NO_RETURN_ROW_WIDTH)


@pytest.mark.covers("T55")
def test_placeholder_points_scale_directions_and_follow_the_sweep_layout():
    """xyz = distance * d, intensity 0, t and channel copied; 5-column sweeps get no channel."""
    rows = np.array([[1.0, 0.0, 0.0, 0.01, 3, 7], [0.0, 0.6, 0.8, -0.02, 5, 9]], dtype=np.float32)
    with_channel = no_return_placeholder_points(rows, NEURAD_DUMMY_DISTANCE_M, sweep_column_count=6)
    expected = np.array([[2000.0, 0.0, 0.0, 0.0, 0.01, 3.0], [0.0, 1200.0, 1600.0, 0.0, -0.02, 5.0]], dtype=np.float32)
    np.testing.assert_array_equal(with_channel, expected)
    np.testing.assert_array_equal(no_return_placeholder_points(rows, NEURAD_DUMMY_DISTANCE_M, 5), expected[:, :5])
    # Every placeholder lies at the dummy distance, beyond SplatAD's 1000 m "returned" threshold.
    np.testing.assert_allclose(np.linalg.norm(with_channel[:, :3], axis=1), NEURAD_DUMMY_DISTANCE_M)
    with pytest.raises(ValueError, match="5 or 6"):
        no_return_placeholder_points(rows, NEURAD_DUMMY_DISTANCE_M, 4)
    with pytest.raises(ValueError, match="rows must be"):
        no_return_placeholder_points(rows[:, :5], NEURAD_DUMMY_DISTANCE_M, 6)


def test_no_return_path_mirrors_the_sweep_path():
    """<root>/lidar/<sensor>/<f> -> <root>/lidar_no_return/<sensor>/<f>."""
    sweep_path = Path("/data/june/lidar/ouster/000012_1714571311020000000.npy")
    assert no_return_path_for_sweep(sweep_path) == Path(
        "/data/june/lidar_no_return/ouster/000012_1714571311020000000.npy")


@pytest.mark.covers("T26")
def test_writer_round_trips_no_return_streams(tmp_path):
    """No-return files are written beside the sweeps and read back exactly; none -> empty dict."""
    dataset = build_synthetic_dataset()
    lidar_stream = dataset.streams["lidar_top"]
    rng = np.random.default_rng(2)
    dataset.no_return_streams = {"lidar_top": NoReturnStream(
        frame_indices=lidar_stream.frame_indices.copy(),
        frame_timestamps_ns=lidar_stream.frame_timestamps_ns.copy(),
        frames=[rng.normal(size=(int(rng.integers(0, 5)), NO_RETURN_ROW_WIDTH)).astype(np.float32)
                for _ in lidar_stream.frames],
    )}
    write_dataset(tmp_path / "with", dataset)
    assert read_dataset(tmp_path / "with") == dataset
    write_dataset(tmp_path / "without", build_synthetic_dataset())
    assert read_dataset(tmp_path / "without").no_return_streams == {}
