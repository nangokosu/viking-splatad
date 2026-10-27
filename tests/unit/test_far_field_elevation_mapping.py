"""Unit tests for T61: the SplatAD elevation mapping from the T51 far-field beam elevations.

Expected values come from the definitions: synthetic beams whose measured elevation is
elevation_inf + k/r (``synthetic_beam_returns``), and a hand-made beam model.
"""

import pytest

from tests.unit.test_no_return import (
    FIT_TOLERANCE_DEG,
    SYNTHETIC_COLUMN_COUNT,
    SYNTHETIC_RING_COUNT,
    TEST_MAX_MEDIAN_RESIDUAL_DEG,
    TEST_RESIDUAL_PERCENTILE,
    synthetic_beam_returns,
)
from viking_splatad.contract.lidar_model import CHANNELS_PER_TILE, derive_elevation_mapping
from viking_splatad.extract.no_return import (
    BeamDirectionModel,
    derive_beam_direction_model,
    elevation_mapping_from_beam_model,
)

# Largest range drawn by synthetic_beam_returns (uniform in [0.5, 30] m).
SYNTHETIC_MAX_RANGE_M = 30.0
# Ouster-like ring layout for the hand-made model: ring 0 highest, one ring every 3 deg downward.
HAND_MADE_TOP_ELEVATION_DEG, HAND_MADE_RING_SPACING_DEG = 20.0, 3.0


def hand_made_model(ring_count: int) -> BeamDirectionModel:
    """Beam model with ring r at elevation 20 - 3r deg (descending with ring id, as the Ouster)."""
    ring_ids = range(ring_count)
    zeros = {ring_id: 0.0 for ring_id in ring_ids}
    return BeamDirectionModel(
        column_count=SYNTHETIC_COLUMN_COUNT, column_step_deg=-360.0 / SYNTHETIC_COLUMN_COUNT,
        elevation_deg={ring_id: HAND_MADE_TOP_ELEVATION_DEG - HAND_MADE_RING_SPACING_DEG * ring_id for ring_id in ring_ids},
        azimuth_offset_deg=dict(zeros), elevation_parallax_deg_m=dict(zeros), azimuth_parallax_deg_m=dict(zeros),
        median_residual_deg=0.0, residual_percentile=TEST_RESIDUAL_PERCENTILE, percentile_residual_deg=0.0,
    )


@pytest.mark.covers("T61")
def test_mapping_is_the_far_field_elevation_while_the_t16_median_carries_parallax():
    """Parallax data: T61 gives elevation_inf exactly; T16's median sits at least k/r_max above it."""
    points, ring, column, truth = synthetic_beam_returns(-1.0)
    model = derive_beam_direction_model(
        points, ring, column, SYNTHETIC_COLUMN_COUNT, range(SYNTHETIC_RING_COUNT),
        TEST_MAX_MEDIAN_RESIDUAL_DEG, TEST_RESIDUAL_PERCENTILE)
    # Channel keys as T16 finds them (the ring ids), passed in T16's order.
    median_mapping = derive_elevation_mapping(points, ring)
    far_field_mapping = elevation_mapping_from_beam_model(model, median_mapping.keys())
    # Far-field angles are the synthetic elevation_inf of every ring.
    assert set(far_field_mapping) == set(range(SYNTHETIC_RING_COUNT))
    for ring_id, elevation in far_field_mapping.items():
        assert elevation == pytest.approx(truth["elevation"][ring_id], abs=FIT_TOLERANCE_DEG)
        # Every return has elevation_inf + k/r with k > 0 and r <= r_max, so even the median is
        # at least k / r_max above the beam's own angle: the bias T61 removes.
        minimum_bias = truth["elevation_parallax"][ring_id] / SYNTHETIC_MAX_RANGE_M
        assert median_mapping[ring_id] - truth["elevation"][ring_id] >= minimum_bias
    # Ascending order, as SplatAD's raster rows expect.
    assert list(far_field_mapping.values()) == sorted(far_field_mapping.values())


@pytest.mark.covers("T61")
def test_mapping_orders_descending_rings_ascending_and_keeps_exact_values():
    """Ring 0 highest: keys come out 7..0 with the model's exact elevations."""
    model = hand_made_model(CHANNELS_PER_TILE)
    mapping = elevation_mapping_from_beam_model(model, range(CHANNELS_PER_TILE))
    assert list(mapping) == list(reversed(range(CHANNELS_PER_TILE)))
    assert mapping == {ring_id: model.elevation_deg[ring_id] for ring_id in range(CHANNELS_PER_TILE)}


@pytest.mark.covers("T61")
def test_mapping_refuses_unfitted_channels_and_partial_tiles():
    """A channel without a fitted beam, half a tile, or no channel at all raises ValueError."""
    model = hand_made_model(CHANNELS_PER_TILE)
    with pytest.raises(ValueError, match="not in the beam model"):
        elevation_mapping_from_beam_model(model, range(CHANNELS_PER_TILE + 1))
    with pytest.raises(ValueError, match="multiple of"):
        elevation_mapping_from_beam_model(model, range(CHANNELS_PER_TILE // 2))
    with pytest.raises(ValueError, match="multiple of"):
        elevation_mapping_from_beam_model(model, [])
