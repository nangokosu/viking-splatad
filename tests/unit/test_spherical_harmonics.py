"""Tests for the SuperSplat SH basis and fit (T84), with first-principles expected values."""

import numpy as np
import pytest

from viking_splatad.export.spherical_harmonics import (
    DEFAULT_DIRECTION_COUNT,
    MAX_SH_BASIS_SIZE,
    MAX_SH_DEGREE,
    SH_C0,
    SH_C1,
    SH_C2,
    SH_C3,
    SH_COLOUR_OFFSET,
    fibonacci_sphere_directions,
    fit_sh_coefficients,
    real_sh_basis,
)

# Seed of the synthetic coefficients (any fixed value; makes the test deterministic).
RANDOM_SEED = 20260121
# Synthetic Gaussian count and the chunk size forcing several chunks plus a ragged last one.
SYNTHETIC_GAUSSIAN_COUNT = 7
SYNTHETIC_CHUNK_SIZE = 3
# Float64 round-off of a well-conditioned (M >= 8K) pseudo-inverse solve.
EXACT_TOLERANCE = 1e-9
# Two dense Fibonacci sizes for the orthogonality convergence check.
COARSE_DIRECTION_COUNT = 2000
FINE_DIRECTION_COUNT = 20000


@pytest.mark.covers("T84")
def test_constants_match_closed_forms_and_cited_values():
    """Closed forms equal the literals in gsplat sh.cu / INRIA eval_sh."""
    # Literals from /workspace/splatad/gsplat/cuda/csrc/sh.cu (INRIA SH_C0..SH_C3).
    assert SH_C0 == pytest.approx(0.2820947917738781, rel=1e-12)
    assert SH_C1 == pytest.approx(0.48860251190292, rel=1e-12)
    assert SH_C2 == pytest.approx(
        [1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
         -1.0925484305920792, 0.5462742152960396], rel=1e-12)
    assert SH_C3 == pytest.approx(
        [-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
         0.3731763325901154, -0.4570457994644658, 1.445305721320277,
         -0.5900435899266435], rel=1e-12)


@pytest.mark.covers("T84")
def test_basis_matches_sloan_formulation_of_gsplat():
    """real_sh_basis equals an independent numpy port of gsplat's sh_coeffs_to_color_fast."""
    directions = fibonacci_sphere_directions(50)
    # Literals and intermediates copied from sh.cu: tmp_0b/tmp_0c/tmp_1b = fTmp0B/fTmp0C/fTmp1B,
    # c1/s1 = fC1/fS1 = r^2 cos/sin(2 phi), c2/s2 = fC2/fS2 = r^3 cos/sin(3 phi), with r^2 = x^2 + y^2
    # and phi the azimuth (Sloan's recurrence).
    dir_x, dir_y, dir_z = directions.T
    z2 = dir_z**2
    tmp_0b = -1.092548430592079 * dir_z
    c1 = dir_x**2 - dir_y**2
    s1 = 2 * dir_x * dir_y
    tmp_0c = -2.285228997322329 * z2 + 0.4570457994644658
    tmp_1b = 1.445305721320277 * dir_z
    c2 = dir_x * c1 - dir_y * s1
    s2 = dir_x * s1 + dir_y * c1
    expected = np.stack(
        [
            np.full(len(directions), 0.2820947917738781),
            -0.48860251190292 * dir_y, 0.48860251190292 * dir_z, -0.48860251190292 * dir_x,
            0.5462742152960395 * s1, tmp_0b * dir_y,
            0.9461746957575601 * z2 - 0.3153915652525201,
            tmp_0b * dir_x, 0.5462742152960395 * c1,
            -0.5900435899266435 * s2, tmp_1b * s1, tmp_0c * dir_y,
            dir_z * (1.865881662950577 * z2 - 1.119528997770346),
            tmp_0c * dir_x, tmp_1b * c1, -0.5900435899266435 * c2,
        ],
        axis=1,
    )
    assert real_sh_basis(directions, MAX_SH_DEGREE) == pytest.approx(expected, abs=1e-12)


@pytest.mark.covers("T84")
def test_fibonacci_directions_are_unit_and_deterministic():
    """Unit length, repeatable, and the z mean vanishes (equal-area bands symmetric about 0)."""
    first = fibonacci_sphere_directions(DEFAULT_DIRECTION_COUNT)
    second = fibonacci_sphere_directions(DEFAULT_DIRECTION_COUNT)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first, axis=1) == pytest.approx(1.0, abs=1e-12)
    # Heights are symmetric about 0, so the z mean is exactly 0 up to round-off.
    assert first[:, 2].mean() == pytest.approx(0.0, abs=1e-12)
    assert DEFAULT_DIRECTION_COUNT == 8 * MAX_SH_BASIS_SIZE


@pytest.mark.covers("T84")
@pytest.mark.parametrize("degree", range(MAX_SH_DEGREE + 1))
def test_fit_recovers_known_coefficients(degree):
    """Colours synthesised from known coefficients are fitted back to float tolerance."""
    rng = np.random.default_rng(RANDOM_SEED)
    directions = fibonacci_sphere_directions(DEFAULT_DIRECTION_COUNT)
    coefficient_count = (degree + 1) ** 2
    true_coefficients = rng.normal(size=(SYNTHETIC_GAUSSIAN_COUNT, 3, coefficient_count))
    basis = real_sh_basis(directions, degree)
    # colour[n, m, ch] = 0.5 + sum_b coeff[n, ch, b] * Y_b(direction m)
    colours = SH_COLOUR_OFFSET + np.einsum("nck,mk->nmc", true_coefficients, basis)
    sh_dc, sh_rest = fit_sh_coefficients(
        directions, colours, degree, chunk_size=SYNTHETIC_CHUNK_SIZE
    )
    assert sh_dc.shape == (SYNTHETIC_GAUSSIAN_COUNT, 3)
    assert sh_rest.shape == (SYNTHETIC_GAUSSIAN_COUNT, 3, coefficient_count - 1)
    assert sh_dc == pytest.approx(true_coefficients[:, :, 0], abs=EXACT_TOLERANCE)
    assert sh_rest == pytest.approx(true_coefficients[:, :, 1:], abs=EXACT_TOLERANCE)


@pytest.mark.covers("T84")
def test_degree_zero_dc_is_mean_colour_over_c0_on_symmetric_set():
    """Degree-0 fit is dc = (mean colour - 0.5) / C0; holds for any direction set, +-axes used here."""
    axis_directions = np.concatenate([np.eye(3), -np.eye(3)])
    rng = np.random.default_rng(RANDOM_SEED)
    colours = rng.uniform(size=(SYNTHETIC_GAUSSIAN_COUNT, len(axis_directions), 3))
    sh_dc, sh_rest = fit_sh_coefficients(axis_directions, colours, 0)
    # Degree 0: B is a constant column C0, pinv = 1 / (M C0) per entry, so the fit is the mean.
    expected_dc = (colours.mean(axis=1) - SH_COLOUR_OFFSET) / SH_C0
    assert sh_dc == pytest.approx(expected_dc, abs=EXACT_TOLERANCE)
    assert sh_rest.shape == (SYNTHETIC_GAUSSIAN_COUNT, 3, 0)


@pytest.mark.covers("T84")
def test_basis_is_orthonormal_and_error_shrinks_with_more_directions():
    """(4 pi / M) B^T B -> identity (equal-weight quasi-Monte-Carlo quadrature), error decreasing in M."""
    def orthonormality_error(direction_count):
        """Max |(4 pi / M) B^T B - I| over a Fibonacci set of the given size."""
        basis = real_sh_basis(fibonacci_sphere_directions(direction_count), MAX_SH_DEGREE)
        gram = (4.0 * np.pi / direction_count) * basis.T @ basis
        return np.abs(gram - np.eye(MAX_SH_BASIS_SIZE)).max()

    coarse_error = orthonormality_error(COARSE_DIRECTION_COUNT)
    fine_error = orthonormality_error(FINE_DIRECTION_COUNT)
    # Strictly smaller with 10x more points, and already small on the coarse set.
    assert fine_error < coarse_error
    assert coarse_error < 1e-2
    assert fine_error < 1e-3


@pytest.mark.covers("T84")
def test_fit_rejects_too_few_directions():
    """Fewer directions than coefficients is underdetermined and must raise."""
    directions = fibonacci_sphere_directions(MAX_SH_BASIS_SIZE - 1)
    colours = np.zeros((1, len(directions), 3))
    with pytest.raises(ValueError):
        fit_sh_coefficients(directions, colours, MAX_SH_DEGREE)


# --- Added tests: validation, rank check, streaming API, exact quadrature -------------------
from viking_splatad.export.spherical_harmonics import (  # noqa: E402
    DEFAULT_CHUNK_SIZE,
    apply_sh_fit,
    chunk_size_for_budget,
    sh_fit_operator,
)

# Gauss-Legendre nodes in z: n nodes integrate polynomials of degree <= 2n - 1 exactly. A degree-3
# Gram entry is a polynomial of degree <= 6 on the sphere, and uniform phi with more than 6
# equispaced azimuths integrates cos/sin(k phi), k <= 6, exactly.
QUADRATURE_Z_NODE_COUNT = 4          # 2 * 4 - 1 = 7 >= 6
QUADRATURE_AZIMUTH_COUNT = 8         # > 6 equispaced azimuths
# Round-off budget of the Gram sum: a few hundred terms of size <= ~1 -> eps times a safety
# factor covering accumulation and the basis evaluation.
GRAM_ROUNDOFF_SAFETY_FACTOR = 1e3
RANK_DEFICIENT_DIRECTION_COUNT = 40
BAD_COLOUR_SAMPLE_COUNT = 2


@pytest.mark.covers("T84")
def test_rank_deficient_directions_raise():
    """Directions all in the z = 0 plane cannot determine degree-3 coefficients."""
    azimuth = np.linspace(0.0, 2.0 * np.pi, RANK_DEFICIENT_DIRECTION_COUNT, endpoint=False)
    planar = np.stack([np.cos(azimuth), np.sin(azimuth), np.zeros_like(azimuth)], axis=1)
    with pytest.raises(ValueError, match="rank"):
        sh_fit_operator(planar, MAX_SH_DEGREE)


@pytest.mark.covers("T84")
def test_zero_length_and_nonfinite_directions_raise():
    """A zero vector cannot be normalised; NaN directions are rejected."""
    directions = fibonacci_sphere_directions(DEFAULT_DIRECTION_COUNT)
    zero_direction = directions.copy()
    zero_direction[0] = 0.0
    with pytest.raises(ValueError):
        real_sh_basis(zero_direction, MAX_SH_DEGREE)
    nan_direction = directions.copy()
    nan_direction[0, 0] = np.nan
    with pytest.raises(ValueError):
        real_sh_basis(nan_direction, MAX_SH_DEGREE)


@pytest.mark.covers("T84")
@pytest.mark.parametrize("bad_degree", [2.0, "2", None, -1, MAX_SH_DEGREE + 1])
def test_bad_degree_raises(bad_degree):
    """Degree must be an int in 0..3."""
    with pytest.raises(ValueError):
        real_sh_basis(fibonacci_sphere_directions(DEFAULT_DIRECTION_COUNT), bad_degree)


@pytest.mark.covers("T84")
def test_nan_colours_raise():
    """Non-finite colours are rejected instead of spreading through the product."""
    directions = fibonacci_sphere_directions(DEFAULT_DIRECTION_COUNT)
    colours = np.full((BAD_COLOUR_SAMPLE_COUNT, len(directions), 3), SH_COLOUR_OFFSET)
    colours[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_sh_coefficients(directions, colours, MAX_SH_DEGREE)


@pytest.mark.covers("T84")
def test_streaming_operator_and_apply_equal_fit_sh_coefficients():
    """Operator computed once + apply per chunk reproduces the wrapper (float32 input too)."""
    rng = np.random.default_rng(RANDOM_SEED)
    directions = fibonacci_sphere_directions(DEFAULT_DIRECTION_COUNT)
    colours = rng.uniform(size=(SYNTHETIC_GAUSSIAN_COUNT, len(directions), 3)).astype(np.float32)
    expected_dc, expected_rest = fit_sh_coefficients(directions, colours, MAX_SH_DEGREE)
    pseudo_inverse = sh_fit_operator(directions, MAX_SH_DEGREE)
    chunk_results = [
        apply_sh_fit(pseudo_inverse, colours[start:start + SYNTHETIC_CHUNK_SIZE])
        for start in range(0, SYNTHETIC_GAUSSIAN_COUNT, SYNTHETIC_CHUNK_SIZE)
    ]
    streamed_dc = np.concatenate([dc for dc, _ in chunk_results])
    streamed_rest = np.concatenate([rest for _, rest in chunk_results])
    assert streamed_dc == pytest.approx(expected_dc, abs=EXACT_TOLERANCE)
    assert streamed_rest == pytest.approx(expected_rest, abs=EXACT_TOLERANCE)


@pytest.mark.covers("T84")
def test_exact_quadrature_gram_is_identity_to_roundoff():
    """Gauss-Legendre (z) x uniform (phi) quadrature integrates B^T B exactly: Gram = I."""
    z_nodes, z_weights = np.polynomial.legendre.leggauss(QUADRATURE_Z_NODE_COUNT)
    azimuth = 2.0 * np.pi * np.arange(QUADRATURE_AZIMUTH_COUNT) / QUADRATURE_AZIMUTH_COUNT
    # Sphere measure dA = dz dphi; the z weights sum to 2 and phi weights to 2 pi / count each.
    azimuth_weight = 2.0 * np.pi / QUADRATURE_AZIMUTH_COUNT
    ring_radius = np.sqrt(1.0 - z_nodes**2)
    directions = np.stack(
        [
            (ring_radius[:, None] * np.cos(azimuth)[None, :]).ravel(),
            (ring_radius[:, None] * np.sin(azimuth)[None, :]).ravel(),
            np.repeat(z_nodes, QUADRATURE_AZIMUTH_COUNT),
        ],
        axis=1,
    )
    weights = np.repeat(z_weights, QUADRATURE_AZIMUTH_COUNT) * azimuth_weight
    basis = real_sh_basis(directions, MAX_SH_DEGREE)
    gram = basis.T @ (weights[:, None] * basis)
    tolerance = GRAM_ROUNDOFF_SAFETY_FACTOR * np.finfo(np.float64).eps
    assert np.abs(gram - np.eye(MAX_SH_BASIS_SIZE)).max() < tolerance


@pytest.mark.covers("T84")
def test_chunk_size_follows_memory_budget():
    """chunk_size = floor(budget / (M * 3 * 8)), at least 1, and the default derives from it."""
    direction_count = 100
    bytes_per_gaussian = direction_count * 3 * np.dtype(np.float64).itemsize
    budget = 10 * bytes_per_gaussian + 1
    assert chunk_size_for_budget(direction_count, budget) == 10
    assert chunk_size_for_budget(direction_count, 1) == 1
    assert DEFAULT_CHUNK_SIZE == chunk_size_for_budget(DEFAULT_DIRECTION_COUNT)
