"""Zero Doppler residual is not a measurement of zero physical sensor uncertainty."""

import numpy as np
import pytest

from viking_splatad.radar_init.clock import regularize_velocity_noise


@pytest.mark.covers("T76")
def test_measured_doppler_cells_bound_velocity_noise_without_guessing_a_floor():
    """A noiseless quantized scan keeps its geometric factor but gains the measured cell variance."""
    numerical_floor = np.finfo(float).eps
    cell_width = 0.12  # Synthetic observed full bin width, not a deployment constant.
    radial = np.array([numerical_floor, 0.01])
    geometry_factors = np.array([2., 3.])
    sigma, velocities = regularize_velocity_noise(radial, radial * geometry_factors,
                                                  np.array([cell_width, cell_width]))
    expected = cell_width / np.sqrt(12)
    assert sigma == pytest.approx(expected)
    np.testing.assert_allclose(velocities, geometry_factors * expected)
    with pytest.raises(ValueError, match="Doppler spread"):
        regularize_velocity_noise(radial, radial * geometry_factors, np.array([]))
