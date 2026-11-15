"""Data-derived clock search extent is independent of Doppler values and fitted outcomes."""

from types import SimpleNamespace

import numpy as np
import pytest

from viking_splatad.radar_init.clock import clock_search_domain, safe_clock_fit_mask


@pytest.mark.covers("T77")
def test_clock_domain_comes_from_available_training_corridors():
    """Two known corridors give an analytic median correction budget and retain train-only samples."""
    frames = [SimpleNamespace(time_s=float(position), split="test" if position == 2 else "train")
              for position in range(5)]
    times = np.array([0.75, 1., 1.25, 2.75, 3., 3.25])
    centre, offset_limit, drift_limit = clock_search_domain(times, frames, 0.5)
    numerical_guard = 1 - np.sqrt(np.finfo(float).eps)
    assert centre == 2.
    # Median budget: 2*0.5/(1+1/2) = 2/3; half goes to offset.
    assert offset_limit == pytest.approx((1 / 3) * numerical_guard)
    assert drift_limit == pytest.approx(offset_limit / 2)
    safe = safe_clock_fit_mask(times, frames, offset_limit, drift_limit, centre)
    np.testing.assert_array_equal(safe, [True, True, False, False, True, True])
