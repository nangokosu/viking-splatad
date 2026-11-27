"""Known-answer tests for T81 peak-alpha voxel occupancy and T57's excluded-centre option."""

import numpy as np
import pytest

from viking_splatad.eval.voxel_occupancy import gaussian_occupancy_at_centres, peak_alpha_voxel_occupancy

VOXEL_SIZE = 0.5
GRID_ORIGIN = np.zeros(3)
CHUNK_SIZE = 8
# A 3x1x1 row of scored cells along x.
ROW_CELLS = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])


def isotropic_factors(count: int, std: float):
    """Identity principal axes and equal per-axis std for ``count`` Gaussians.

    Args:
        count: Number of Gaussians. std: Per-axis standard deviation (m).
    Returns:
        ([count,3,3] axes, [count,3] std).
    """
    return np.repeat(np.eye(3)[None], count, axis=0), np.full((count, 3), std)


def cell_centre(cell_index) -> np.ndarray:
    """Centre of a GT cell: origin + (index + 0.5) * edge.

    Args:
        cell_index: (3,) integer index.
    Returns:
        (3,) centre.
    """
    return GRID_ORIGIN + (np.asarray(cell_index) + 0.5) * VOXEL_SIZE


@pytest.mark.covers("T81")
def test_thin_off_centre_gaussian_gives_its_opacity_in_its_own_voxel_only():
    """A 1 mm Gaussian 0.2 m off its voxel centre reaches no centre (T57 = 0 everywhere), but T81 gives
    sigma(o) in the voxel holding its mean and 0 in the other cells."""
    opacity = 0.8
    mean = cell_centre([1, 0, 0]) + np.array([0.2, 0.0, 0.0])
    axes, std = isotropic_factors(1, 1e-3)
    centres = GRID_ORIGIN + (ROW_CELLS + 0.5) * VOXEL_SIZE
    centre_sample = gaussian_occupancy_at_centres(centres, mean[None], None, np.array([opacity]), CHUNK_SIZE,
                                                  principal_axes=axes, axis_std=std)
    assert np.array_equal(centre_sample, np.zeros(3))
    occupancy = peak_alpha_voxel_occupancy(ROW_CELLS, GRID_ORIGIN, VOXEL_SIZE, mean[None], np.array([opacity]), axes, std, CHUNK_SIZE)
    assert occupancy == pytest.approx([0.0, opacity, 0.0])


@pytest.mark.covers("T81")
def test_two_gaussians_in_one_voxel_combine_by_the_product_rule():
    """Opacities 0.5 and 0.6 in cell 0: 1 - (1 - 0.5)(1 - 0.6) = 0.8."""
    opacities = np.array([0.5, 0.6])
    means = np.stack([cell_centre([0, 0, 0]) + 0.1, cell_centre([0, 0, 0]) - 0.1])
    axes, std = isotropic_factors(2, 1e-3)
    occupancy = peak_alpha_voxel_occupancy(ROW_CELLS, GRID_ORIGIN, VOXEL_SIZE, means, opacities, axes, std, CHUNK_SIZE)
    assert occupancy == pytest.approx([1 - (1 - 0.5) * (1 - 0.6), 0.0, 0.0])


@pytest.mark.covers("T81")
def test_wide_gaussian_at_a_centre_spreads_its_tail_like_t57():
    """std = one voxel edge, mean at cell 1's centre: cell 1 gets o (peak = centre value), cells 0 and 2,
    one std away, get o * exp(-1/2): identical to T57 when the mean sits on a centre."""
    opacity = 0.9
    mean = cell_centre([1, 0, 0])
    axes, std = isotropic_factors(1, VOXEL_SIZE)
    occupancy = peak_alpha_voxel_occupancy(ROW_CELLS, GRID_ORIGIN, VOXEL_SIZE, mean[None], np.array([opacity]), axes, std, CHUNK_SIZE)
    tail = opacity * np.exp(-0.5)
    assert occupancy == pytest.approx([tail, opacity, tail])


@pytest.mark.covers("T81")
def test_gaussian_outside_the_scored_cells_adds_only_its_tail():
    """Mean in unscored cell 3 (one edge beyond cell 2's centre), std = edge: cell 2 gets o * exp(-1/2),
    cell 1 (2 std) o * exp(-2), cell 0 (3 std, on the inclusive 3-sigma boundary) o * exp(-9/2)."""
    opacity = 0.7
    mean = cell_centre([3, 0, 0])
    axes, std = isotropic_factors(1, VOXEL_SIZE)
    occupancy = peak_alpha_voxel_occupancy(ROW_CELLS, GRID_ORIGIN, VOXEL_SIZE, mean[None], np.array([opacity]), axes, std, CHUNK_SIZE)
    assert occupancy == pytest.approx(opacity * np.exp(-0.5 * np.array([3.0, 2.0, 1.0]) ** 2))


@pytest.mark.covers("T57")
def test_t57_excluded_centre_is_skipped_and_others_are_unchanged():
    """Excluding cell 1 for a wide Gaussian centred on it leaves 0 there and the exact tails elsewhere;
    a negative index excludes nothing."""
    opacity = 0.9
    centres = GRID_ORIGIN + (ROW_CELLS + 0.5) * VOXEL_SIZE
    mean = cell_centre([1, 0, 0])[None]
    axes, std = isotropic_factors(1, VOXEL_SIZE)
    excluded = gaussian_occupancy_at_centres(centres, mean, None, np.array([opacity]), CHUNK_SIZE, principal_axes=axes,
                                             axis_std=std, excluded_centre_per_gaussian=np.array([1]))
    tail = opacity * np.exp(-0.5)
    assert excluded == pytest.approx([tail, 0.0, tail])
    none_excluded = gaussian_occupancy_at_centres(centres, mean, None, np.array([opacity]), CHUNK_SIZE, principal_axes=axes,
                                                  axis_std=std, excluded_centre_per_gaussian=np.array([-1]))
    assert none_excluded == pytest.approx([tail, opacity, tail])


@pytest.mark.covers("T81")
def test_far_away_gaussian_does_not_overflow_and_adds_nothing():
    """A Gaussian 1e9 m away (like a random init point far outside the scene) leaves the scored cells empty
    and the nearby Gaussian's peak unchanged; the home-cell keys must not span the far point."""
    far_distance_m = 1e9
    opacities = np.array([0.6, 0.9])
    means = np.stack([cell_centre([2, 0, 0]) + 0.1, np.full(3, far_distance_m)])
    axes, std = isotropic_factors(2, 1e-3)
    occupancy = peak_alpha_voxel_occupancy(ROW_CELLS, GRID_ORIGIN, VOXEL_SIZE, means, opacities, axes, std, CHUNK_SIZE)
    assert occupancy == pytest.approx([0.0, 0.0, 0.6])
