"""Real spherical-harmonics (SH) basis and least-squares fit for the SuperSplat export.

Per-Gaussian colour is evaluated for many view directions, then fitted with the 3DGS SH model

    colour(direction) = 0.5 + sum_b coefficient_b * Y_b(direction)      (clamped by the viewer)

The basis is NOT reimplemented here: ``real_sh_basis`` calls ``_eval_sh_bases_fast`` from
the pinned gsplat, ``gsplat/cuda/_torch_impl.py`` (reference image path
``/workspace/splatad/gsplat/cuda/_torch_impl.py``). That function is Sloan's formulation
(Efficient Spherical Harmonic Evaluation, JCGT 2013), equal to INRIA's ``eval_sh`` for unit
directions, and is licensed Apache-2.0 (gsplat, (c) the nerfstudio / gsplat authors). It is
imported, not copied, so the licence and attribution stay with the upstream package. It is a
private (underscore) name, but the gsplat commit is pinned by SHA in ``docker/Dockerfile.splatad``
and the tests check the port against closed forms and an independent port.
The closed-form constants SH_C0..SH_C3 below are kept only as the test oracle and documentation
of the convention.

Basis ordering (index -> function, for a unit direction (x, y, z)):
    0: C0                                   1: -C1 y   2: C1 z   3: -C1 x
    4: C2[0] x y                            5: C2[1] y z
    6: C2[2] (2 z^2 - x^2 - y^2)            7: C2[3] x z      8: C2[4] (x^2 - y^2)
    9: C3[0] y (3 x^2 - y^2)               10: C3[1] x y z
   11: C3[2] y (4 z^2 - x^2 - y^2)         12: C3[3] z (2 z^2 - 3 x^2 - 3 y^2)
   13: C3[4] x (4 z^2 - x^2 - y^2)         14: C3[5] z (x^2 - y^2)
   15: C3[6] x (x^2 - 3 y^2)
"""

import operator

import numpy as np

from viking_splatad.registry import transform

# Highest SH degree 3DGS / SuperSplat store (16 coefficients per channel).
MAX_SH_DEGREE = 3

# Colour offset of the 3DGS SH model: colour = SH_COLOUR_OFFSET + sum(...). This is the
# 3DGS convention (the DC term is stored relative to mid-grey), not a tunable.
SH_COLOUR_OFFSET = 0.5

# Closed-form real SH normalisation constants (orthonormal over the unit sphere).
SH_C0 = 1.0 / (2.0 * np.sqrt(np.pi))
SH_C1 = np.sqrt(3.0 / (4.0 * np.pi))
SH_C2 = np.array(
    [
        0.5 * np.sqrt(15.0 / np.pi),   # xy
        -0.5 * np.sqrt(15.0 / np.pi),  # yz
        0.25 * np.sqrt(5.0 / np.pi),   # 2z^2 - x^2 - y^2
        -0.5 * np.sqrt(15.0 / np.pi),  # xz
        0.25 * np.sqrt(15.0 / np.pi),  # x^2 - y^2
    ]
)
SH_C3 = np.array(
    [
        -0.25 * np.sqrt(35.0 / (2.0 * np.pi)),  # y (3x^2 - y^2)
        0.5 * np.sqrt(105.0 / np.pi),           # xyz
        -0.25 * np.sqrt(21.0 / (2.0 * np.pi)),  # y (4z^2 - x^2 - y^2)
        0.25 * np.sqrt(7.0 / np.pi),            # z (2z^2 - 3x^2 - 3y^2)
        -0.25 * np.sqrt(21.0 / (2.0 * np.pi)),  # x (4z^2 - x^2 - y^2)
        0.25 * np.sqrt(105.0 / np.pi),          # z (x^2 - y^2)
        -0.25 * np.sqrt(35.0 / (2.0 * np.pi)),  # x (x^2 - 3y^2)
    ]
)

# Golden angle (rad): azimuth step of the Fibonacci spiral, pi * (3 - sqrt(5)).
GOLDEN_ANGLE_RAD = np.pi * (3.0 - np.sqrt(5.0))

# Number of SH coefficients per channel at the maximum degree: (3 + 1)^2.
MAX_SH_BASIS_SIZE = (MAX_SH_DEGREE + 1) ** 2

# Directions per basis function. The fit has MAX_SH_BASIS_SIZE unknowns per channel; more
# directions average out the non-uniformity of the sampled colours and the energy above
# degree 3 that a degree-3 model cannot represent (aliasing into the fitted coefficients).
# Measured (random degree<=4 colour functions, degree-3 fit, 200 functions, held-out RMS,
# docs/research/supersplat_export/sh_oversampling_residual.py):
# relative residual 0.297 at 1x, 0.126 at 2x,
# 0.1254 at 4x and 8x, 0.1254 at 256x. The floor is the unrepresentable degree-4 energy; it is
# reached by 4x. 8x is twice that point, a margin for view-dependent colours that are noisier
# than these smooth test functions (noise averages as 1/sqrt(M)), at O(N * M * K) cost.
DIRECTION_OVERSAMPLING_FACTOR = 8

# Default number of view directions per Gaussian (= 8 * 16 = 128), computed from the above.
DEFAULT_DIRECTION_COUNT = DIRECTION_OVERSAMPLING_FACTOR * MAX_SH_BASIS_SIZE

# Memory budget (bytes) for the (chunk, M, 3) float64 temporary of one fit chunk: 256 MiB keeps
# the temporary small next to the full (N, M, 3) input on a laptop, while making each BLAS
# call large enough to be efficient. Not a tunable of the result.
CHUNK_MEMORY_BUDGET_BYTES = 256 * 1024 * 1024

# Bytes of one float64 colour sample component.
FLOAT64_BYTES = np.dtype(np.float64).itemsize

# Number of colour channels (RGB).
COLOUR_CHANNEL_COUNT = 3


def chunk_size_for_budget(direction_count: int, memory_budget_bytes: int = CHUNK_MEMORY_BUDGET_BYTES) -> int:
    """Gaussians per chunk so that a (chunk, M, 3) float64 temporary fits the budget.

    chunk_size = floor(budget / (M * 3 * 8 bytes)), at least 1.

    Args:
        direction_count: M, view directions per Gaussian, >= 1.
        memory_budget_bytes: Byte budget of one chunk's temporary.

    Returns:
        Chunk size >= 1.
    """
    bytes_per_gaussian = direction_count * COLOUR_CHANNEL_COUNT * FLOAT64_BYTES
    return max(1, memory_budget_bytes // bytes_per_gaussian)


# Default number of Gaussians fitted per batch, derived from the budget at the default M.
DEFAULT_CHUNK_SIZE = chunk_size_for_budget(DEFAULT_DIRECTION_COUNT)


def fibonacci_sphere_directions(count: int) -> np.ndarray:
    """Deterministic, near-uniform unit vectors on the sphere (golden-angle spiral).

    Point k (k = 0..count-1) has height z_k = 1 - 2 (k + 0.5) / count, which splits the sphere
    into ``count`` equal-height bands (equal area, by Archimedes' hat-box theorem), and azimuth
    k * golden_angle. The irrational golden angle keeps successive points from lining up on
    meridians, so the points are spread near-uniformly with no random draw.

    Args:
        count: Number of directions, >= 1.

    Returns:
        (count, 3) float64 unit vectors.

    Raises:
        ValueError: If ``count`` < 1.
    """
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    point_indices = np.arange(count, dtype=np.float64)
    # Equal-area band centres: z uniform in (-1, 1) <=> uniform area.
    height_z = 1.0 - 2.0 * (point_indices + 0.5) / count
    # Radius of the circle at that height, so that x^2 + y^2 + z^2 = 1.
    ring_radius = np.sqrt(np.clip(1.0 - height_z**2, 0.0, None))
    azimuth_rad = point_indices * GOLDEN_ANGLE_RAD
    return np.stack(
        [ring_radius * np.cos(azimuth_rad), ring_radius * np.sin(azimuth_rad), height_z],
        axis=1,
    )


def real_sh_basis(directions: np.ndarray, degree: int) -> np.ndarray:
    """Evaluate the real SH basis up to ``degree`` (INRIA / gsplat convention).

    Delegates to gsplat's ``_eval_sh_bases_fast`` (Apache-2.0, see module docstring), run on
    CPU in float64 so the result is exact to round-off.

    Args:
        directions: (M, 3) finite, non-zero direction vectors; normalised here.
        degree: Highest SH degree, an integer 0..3.

    Returns:
        (M, (degree + 1)^2) float64 basis matrix; column b is Y_b of the module docstring's ordering.

    Raises:
        ValueError: If ``degree`` is not an int in 0..MAX_SH_DEGREE, ``directions`` is not (M, 3),
            or a direction is non-finite or has zero length.
    """
    # Step 1: degree must be a true integer (operator.index rejects floats such as 2.0).
    try:
        degree = operator.index(degree)
    except TypeError as error:
        raise ValueError(f"degree must be an int, got {degree!r}") from error
    if not 0 <= degree <= MAX_SH_DEGREE:
        raise ValueError(f"degree must be in 0..{MAX_SH_DEGREE}, got {degree}")
    # Step 2: shape and finiteness.
    directions = np.asarray(directions, dtype=np.float64)
    if directions.ndim != 2 or directions.shape[1] != 3:
        raise ValueError(f"directions must be (M, 3), got {directions.shape}")
    if not np.isfinite(directions).all():
        raise ValueError("directions must be finite")
    # Step 3: the closed forms assume unit length, so a zero vector cannot be normalised.
    direction_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if (direction_norms == 0.0).any():
        raise ValueError("directions must have non-zero length")
    unit_directions = directions / direction_norms
    # Step 4: lazy import keeps torch/gsplat out of module import (they exist only in the image).
    import torch
    from gsplat.cuda._torch_impl import _eval_sh_bases_fast

    basis_dim = (degree + 1) ** 2
    basis_tensor = _eval_sh_bases_fast(basis_dim, torch.from_numpy(unit_directions))
    return basis_tensor.numpy()


def sh_fit_operator(directions: np.ndarray, degree: int) -> np.ndarray:
    """Shared least-squares operator of the SH fit, computed once.

    For each Gaussian and channel the fit solves  min_c || B c - (colour - 0.5) ||^2  with B the
    (M, K) basis (K = (degree + 1)^2). The solution is c = pinv(B) (colour - 0.5). B depends
    only on the directions, so pinv(B) (K, M) is the same for every Gaussian.

    A direction set whose B lacks full column rank (e.g. all directions in one plane) leaves
    some coefficients undetermined; pinv would then silently return the minimum-norm solution,
    which is wrong for the true coefficients, so it is rejected: rank(B) must equal K.

    Args:
        directions: (M, 3) view directions; M >= K.
        degree: SH degree, an integer 0..3.

    Returns:
        (K, M) float64 pseudo-inverse of the basis matrix.

    Raises:
        ValueError: On invalid directions or degree, M < K, or rank(B) < K.
    """
    # Step 1: basis (validates directions and degree).
    basis_matrix = real_sh_basis(directions, degree)
    direction_count, coefficient_count = basis_matrix.shape
    # Step 2: underdetermined fit.
    if direction_count < coefficient_count:
        raise ValueError(
            f"need >= {coefficient_count} directions for degree {degree}, got {direction_count}"
        )
    # Step 3: full column rank (numerical rank from the SVD with numpy's default tolerance).
    basis_rank = np.linalg.matrix_rank(basis_matrix)
    if basis_rank != coefficient_count:
        raise ValueError(
            f"degenerate directions: basis rank {basis_rank} < {coefficient_count} coefficients"
        )
    return np.linalg.pinv(basis_matrix)


def apply_sh_fit(
    pseudo_inverse: np.ndarray, colours_chunk: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit SH coefficients for one chunk of Gaussians with a precomputed operator.

    c[n, ch, k] = sum_m pinv[k, m] * (colour[n, m, ch] - 0.5), evaluated as one BLAS matrix
    product per call: cost O(n * M * K). Only this chunk is cast to float64 and centred.

    Args:
        pseudo_inverse: (K, M) operator from ``sh_fit_operator``.
        colours_chunk: (n, M, 3) finite RGB, colour of Gaussian n seen along direction m.

    Returns:
        (sh_dc, sh_rest): sh_dc (n, 3); sh_rest (n, 3, K - 1), channel-major.

    Raises:
        ValueError: On bad shapes or non-finite colours.
    """
    # Step 1: shape checks against the operator.
    if colours_chunk.ndim != 3 or colours_chunk.shape[2] != COLOUR_CHANNEL_COUNT:
        raise ValueError(f"colours must be (n, M, 3), got {colours_chunk.shape}")
    if colours_chunk.shape[1] != pseudo_inverse.shape[1]:
        raise ValueError(
            f"colours has {colours_chunk.shape[1]} directions but the operator expects "
            f"{pseudo_inverse.shape[1]}"
        )
    # Step 2: cast only this chunk, then reject NaN / inf before they spread through the product.
    centred_colours = np.asarray(colours_chunk, dtype=np.float64) - SH_COLOUR_OFFSET
    if not np.isfinite(centred_colours).all():
        raise ValueError("colours must be finite")
    # Step 3: matmul broadcasts pinv (K, M) over the (n, M, 3) batch axis by transposing
    # colours to (n, 3, M): (n, 3, M) @ (M, K) = (n, 3, K) is a single large BLAS gemm.
    coefficients = np.matmul(centred_colours.transpose(0, 2, 1), pseudo_inverse.T)
    return coefficients[:, :, 0], coefficients[:, :, 1:]


@transform(
    "T84",
    "Per-Gaussian colours at M view directions -> 3DGS SH coefficients "
    "(shared pseudo-inverse least squares, colour = 0.5 + sum coeff * Y)",
)
def fit_sh_coefficients(
    directions: np.ndarray,
    colours: np.ndarray,
    degree: int,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit of SH coefficients to colours sampled at fixed directions.

    Thin wrapper: ``sh_fit_operator`` once, then ``apply_sh_fit`` per chunk of Gaussians.
    Chunking bounds the per-chunk float64 temporary to O(chunk_size * M * 3); the full
    (N, M, 3) ``colours`` input stays in its own dtype (only each chunk is cast).

    Args:
        directions: (M, 3) view directions shared by all Gaussians; M >= K.
        colours: (N, M, 3) RGB in [0, 1], colour of Gaussian n seen along direction m.
        degree: SH degree 0..3.
        chunk_size: Gaussians per batch, >= 1.

    Returns:
        (sh_dc, sh_rest): sh_dc (N, 3) is coefficient 0 per channel; sh_rest (N, 3, K - 1) is
        channel-major: [gaussian, rgb channel, coefficient 1..K-1].

    Raises:
        ValueError: On bad shapes, ``chunk_size`` < 1, non-finite colours, or a degenerate or
            too small direction set.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    colours = np.asarray(colours)
    if colours.ndim != 3 or colours.shape[2] != COLOUR_CHANNEL_COUNT:
        raise ValueError(f"colours must be (N, M, 3), got {colours.shape}")
    if colours.shape[1] != len(directions):
        raise ValueError(
            f"colours has {colours.shape[1]} directions but directions has {len(directions)}"
        )
    pseudo_inverse = sh_fit_operator(directions, degree)
    gaussian_count = colours.shape[0]
    coefficient_count = pseudo_inverse.shape[0]
    coefficients = np.empty(
        (gaussian_count, COLOUR_CHANNEL_COUNT, coefficient_count), dtype=np.float64
    )
    for chunk_start in range(0, gaussian_count, chunk_size):
        chunk_stop = min(chunk_start + chunk_size, gaussian_count)
        chunk_dc, chunk_rest = apply_sh_fit(pseudo_inverse, colours[chunk_start:chunk_stop])
        coefficients[chunk_start:chunk_stop, :, 0] = chunk_dc
        coefficients[chunk_start:chunk_stop, :, 1:] = chunk_rest
    return coefficients[:, :, 0], coefficients[:, :, 1:]
