"""Analytic clock calibration tests including training-only search and unidentifiability."""

from types import SimpleNamespace

import numpy as np
import pytest

from viking_splatad.radar_init.clock import doppler_ego_velocity, fit_affine_clock, safe_clock_fit_mask


@pytest.mark.covers("T73")
def test_doppler_velocity_recovers_known_vector_from_axis_rays():
    """Signed stationary Doppler on six axis rays uniquely identifies ego translation."""
    directions = np.concatenate([np.eye(3), -np.eye(3)])
    expected = np.array([1., 2., 3.])
    velocity, radial_sigma, velocity_sigma = doppler_ego_velocity(directions * 10, -directions @ expected)
    np.testing.assert_allclose(velocity, expected)
    assert radial_sigma > 0 and velocity_sigma > 0


@pytest.mark.covers("T75")
def test_clock_fit_mask_rejects_any_candidate_that_can_cross_a_holdout():
    """Even nominally training measurements are excluded if timing uncertainty can reach test."""
    frames = [SimpleNamespace(time_s=float(position), split="test" if position == 2 else "train")
              for position in range(5)]
    kept = safe_clock_fit_mask(np.array([1., 1.45, 2., 3.]), frames, 0.1, 0.01, 2.)
    np.testing.assert_array_equal(kept, [True, False, False, True])


@pytest.mark.covers("T74")
def test_affine_clock_recovers_injected_offset_and_drift():
    """Linear acceleration yields an exact two-parameter timing solution across the stretch."""
    reference_times = np.linspace(0., 10., 101)
    reference_velocity = np.column_stack([reference_times, 2 * reference_times, -reference_times])
    radar_times = np.linspace(0.5, 9.5, 31)
    offset, drift, centre = 0.02, 0.001, 5.
    corrected = radar_times + offset + drift * (radar_times - centre)
    observed = np.column_stack([corrected, 2 * corrected, -corrected])
    report = fit_affine_clock(radar_times, observed, np.full(len(radar_times), 0.05), reference_times,
                               reference_velocity, 0.1, 0.01, centre, 0.95)
    tolerance = np.sqrt(np.finfo(float).eps)
    assert report["offset_seconds"] == pytest.approx(offset, abs=tolerance)
    assert report["drift_rate"] == pytest.approx(drift, abs=tolerance)
    with pytest.raises(ValueError, match="unidentifiable"):
        fit_affine_clock(radar_times, np.ones_like(observed), np.ones(len(radar_times)), reference_times,
                           np.ones_like(reference_velocity), 0.1, 0.01, centre, 0.95)
