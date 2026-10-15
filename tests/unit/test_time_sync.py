"""Known-answer tests for time-offset estimation (T22) and radar drift / extrinsic (T24)."""

import numpy as np
import pytest

from viking_splatad.align.time_sync import (
    estimate_time_offset,
    fit_radar_drift_model,
    radar_extrinsic_residual,
)


def smooth_signal(time_seconds: np.ndarray) -> np.ndarray:
    """Aperiodic test signal: a sum of Gaussian bumps at fixed, irregular centres.

    Args:
        time_seconds: Evaluation times [s].

    Returns:
        Signal values with a unique, sharp cross-correlation peak.
    """
    # Bump centres/widths/heights are arbitrary synthetic shape parameters.
    bump_centres = np.array([2.0, 3.7, 4.1, 6.5, 8.8])
    bump_widths = np.array([0.3, 0.2, 0.25, 0.4, 0.3])
    bump_heights = np.array([1.0, -0.7, 0.5, 1.2, -0.9])
    # Sum_k h_k * exp(-(t - c_k)^2 / (2 w_k^2))
    signal = np.zeros_like(time_seconds)
    for bump_index in range(bump_centres.size):
        signal += bump_heights[bump_index] * np.exp(
            -((time_seconds - bump_centres[bump_index]) ** 2)
            / (2.0 * bump_widths[bump_index] ** 2)
        )
    return signal


@pytest.mark.covers("T22")
def test_recovers_injected_offset_within_fraction_of_sample_spacing():
    """b(t) = a(t - delta) with delta off the sample grid is recovered sub-sample."""
    sample_spacing = 0.05
    injected_offset = 0.0734  # deliberately not a multiple of the spacing
    times_a = np.arange(0.0, 12.0, sample_spacing)
    # Stream b has a different phase/spacing so resampling is exercised.
    times_b = np.arange(0.013, 12.0, sample_spacing * 1.1)
    values_a = smooth_signal(times_a)
    values_b = smooth_signal(times_b - injected_offset)

    offset, confidence = estimate_time_offset(times_a, values_a, times_b, values_b)

    # Tolerance: a fifth of the coarser sample spacing (sub-sample refinement).
    tolerance = 0.2 * sample_spacing
    assert abs(offset - injected_offset) < tolerance
    assert 0.0 < confidence <= 1.0


@pytest.mark.covers("T22")
def test_negative_offset_sign_convention():
    """A negative delta (b leads a) yields a negative estimate."""
    sample_spacing = 0.05
    injected_offset = -0.121
    times = np.arange(0.0, 12.0, sample_spacing)
    offset, _confidence = estimate_time_offset(
        times, smooth_signal(times), times, smooth_signal(times - injected_offset)
    )
    assert abs(offset - injected_offset) < 0.2 * sample_spacing


@pytest.mark.covers("T24")
def test_drift_line_recovered_with_residual_equal_to_orthogonal_noise():
    """offset(t)=a+b*t is recovered; residual RMS equals the injected noise amplitude."""
    window_times = np.array([0.0, 10.0, 20.0, 30.0])
    intercept, slope = 0.25, 2e-4
    noise_amplitude = 1e-3
    # Pattern [+1,-1,-1,+1] is orthogonal to [1,1,1,1] and to the centred times,
    # so least squares recovers (a, b) exactly and RMS residual == amplitude.
    orthogonal_noise = noise_amplitude * np.array([1.0, -1.0, -1.0, 1.0])
    measured = intercept + slope * window_times + orthogonal_noise

    model = fit_radar_drift_model(window_times, measured)

    assert model.offset_at_zero == pytest.approx(intercept, abs=1e-12)
    assert model.drift_rate == pytest.approx(slope, abs=1e-12)
    assert model.residual_rms == pytest.approx(noise_amplitude, rel=1e-9)


@pytest.mark.covers("T24")
def test_extrinsic_residual_identity_and_translation_delta():
    """Coincident clouds give 0 residual; a translation delta is the residual."""
    rng = np.random.default_rng(0)
    lidar_points = rng.uniform(-50.0, 50.0, size=(40, 3))
    radar_points = lidar_points.copy()
    identity = np.eye(4)
    delta = 0.05
    # Residual == delta only if delta < half the minimum point separation.
    separations = np.linalg.norm(
        lidar_points[:, None, :] - lidar_points[None, :, :], axis=2
    )
    np.fill_diagonal(separations, np.inf)
    assert separations.min() > 2.0 * delta

    assert radar_extrinsic_residual(radar_points, lidar_points, identity) == pytest.approx(0.0, abs=1e-12)

    shifted_extrinsic = np.eye(4)
    shifted_extrinsic[:3, 3] = [delta, 0.0, 0.0]
    assert radar_extrinsic_residual(
        radar_points, lidar_points, shifted_extrinsic
    ) == pytest.approx(delta, abs=1e-12)
