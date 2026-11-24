"""Lidar range margin measured from the data (T92), numpy + scipy only.

A voxel is labelled "hidden" when recorded beams returned short of it by more than a range
margin ``m``. No sensor range-noise figure is available, so ``m`` is measured from residuals
between two recorded sweeps of the same static scene, both already in the map frame.

Nearest-neighbour point distances would include the beam spacing (the two sweeps sample the
surface at different places). A point-to-plane residual against a local plane fitted to the
other sweep removes that spacing, leaving range noise plus pose error.
"""

from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree

from viking_splatad.registry import transform

# Width of a 3D point row (x, y, z).
POINT_DIMENSION = 3
# A plane fit needs at least this many neighbours (3 non-collinear points define a plane).
MINIMUM_NEIGHBOUR_COUNT = 3
# Valid closed range of the planarity threshold (planarity itself lies in [0, 1]).
PLANARITY_LOWER_BOUND = 0.0
PLANARITY_UPPER_BOUND = 1.0
# Open range of a quantile that leaves the margin a real summary (0 and 1 would be the min / max residual).
QUANTILE_LOWER_BOUND = 0.0
QUANTILE_UPPER_BOUND = 1.0
# np.linalg.eigh returns eigenvalues in ascending order: column 0 is the smallest (normal),
# the last column is the largest.
SMALLEST_EIGENVALUE_INDEX = 0
MIDDLE_EIGENVALUE_INDEX = 1
LARGEST_EIGENVALUE_INDEX = 2
# Axis of the neighbour dimension in the (K, neighbour_count, 3) neighbour array.
NEIGHBOUR_AXIS = 1
# Smallest sweep stride and gap (both count sweeps, so >= 1).
MINIMUM_SWEEP_STEP = 1


def _check_points(points: np.ndarray, name: str) -> None:
    """Raise ValueError unless ``points`` has shape (N, 3).

    Args:
        points: Array to check.
        name: Argument name for the error message.

    Raises:
        ValueError: If the array is not 2D with POINT_DIMENSION columns.
    """
    if points.ndim != 2 or points.shape[1] != POINT_DIMENSION:
        raise ValueError(f"{name} must have shape (N, {POINT_DIMENSION}), got {points.shape}")


@transform("T92", "Cross-sweep point-to-plane residuals (local PCA plane of the other sweep) for the lidar range margin")
def cross_sweep_plane_residuals(
    points_a_map: np.ndarray,
    points_b_map: np.ndarray,
    neighbour_count: int,
    planarity_min: float,
) -> np.ndarray:
    """Point-to-plane residuals of sweep A's points against local planes fitted to sweep B.

    For each point p of A, take its ``neighbour_count`` nearest points in B and run PCA on them:
    centroid c, covariance eigenvalues l1 >= l2 >= l3 with the eigenvector n of l3 as the plane
    normal. Planarity = (l2 - l3) / l1 (Demantke / Weinmann): near 1 for a flat patch, near 0 for
    a line (l2 ~ l3 ~ 0) or an isotropic blob (l1 ~ l2 ~ l3). Only points with l1 > 0 and
    planarity >= ``planarity_min`` are kept. Residual = |(p - c) . n|, the distance of p to the
    plane through c with normal n. Cost O(N_A log N_B) for the query plus O(K k) for the PCAs.

    Args:
        points_a_map: (N_A, 3) points of the first sweep, map frame.
        points_b_map: (N_B, 3) points of the second sweep, map frame.
        neighbour_count: Neighbours per plane fit, >= 3.
        planarity_min: Minimum planarity in [0, 1] for a point to be kept.

    Returns:
        (K,) float64 residuals of the kept points in A order; empty if B has fewer points than
        ``neighbour_count``.

    Raises:
        ValueError: On a bad shape, ``neighbour_count`` < 3 or ``planarity_min`` outside [0, 1].
    """
    # Step 1: validate inputs.
    _check_points(points_a_map, "points_a_map")
    _check_points(points_b_map, "points_b_map")
    if neighbour_count < MINIMUM_NEIGHBOUR_COUNT:
        raise ValueError(f"neighbour_count must be >= {MINIMUM_NEIGHBOUR_COUNT}, got {neighbour_count}")
    if not PLANARITY_LOWER_BOUND <= planarity_min <= PLANARITY_UPPER_BOUND:
        raise ValueError(
            f"planarity_min must be in [{PLANARITY_LOWER_BOUND}, {PLANARITY_UPPER_BOUND}], got {planarity_min}"
        )
    points_a = np.asarray(points_a_map, dtype=np.float64)
    points_b = np.asarray(points_b_map, dtype=np.float64)
    # Step 2: too few reference points (or no query points) -> nothing to measure.
    if points_b.shape[0] < neighbour_count or points_a.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    # Step 3: one vectorised k-nearest-neighbour query; indices have shape (N_A, neighbour_count).
    _, neighbour_indices = cKDTree(points_b).query(points_a, k=neighbour_count)
    neighbours = points_b[neighbour_indices]  # (N_A, neighbour_count, 3)
    # Step 4: batch centroid and covariance, cov = (1/k) sum (q - c)(q - c)^T per point.
    centroids = neighbours.mean(axis=NEIGHBOUR_AXIS)
    centred = neighbours - centroids[:, np.newaxis, :]
    covariances = np.einsum("nki,nkj->nij", centred, centred) / neighbour_count
    # Step 5: eigendecomposition of the (N_A, 3, 3) symmetric matrices, ascending eigenvalues.
    eigenvalues, eigenvectors = np.linalg.eigh(covariances)
    largest = eigenvalues[:, LARGEST_EIGENVALUE_INDEX]
    middle = eigenvalues[:, MIDDLE_EIGENVALUE_INDEX]
    smallest = eigenvalues[:, SMALLEST_EIGENVALUE_INDEX]
    # Step 6: planarity (l2 - l3) / l1, guarded against l1 == 0 (coincident neighbours).
    has_extent = largest > 0.0
    planarity = np.zeros_like(largest)
    planarity[has_extent] = (middle[has_extent] - smallest[has_extent]) / largest[has_extent]
    keep = has_extent & (planarity >= planarity_min)
    # Step 7: residual |(p - c) . n| with n the eigenvector of the smallest eigenvalue.
    normals = eigenvectors[:, :, SMALLEST_EIGENVALUE_INDEX]
    residuals = np.abs(np.einsum("ni,ni->n", points_a - centroids, normals))
    return residuals[keep].astype(np.float64)


def range_margin(residuals: np.ndarray, quantile: float) -> float:
    """Range margin as a quantile of the residuals (numpy linear interpolation).

    Args:
        residuals: (K,) non-negative residuals from ``cross_sweep_plane_residuals``.
        quantile: Quantile in the open interval (0, 1), e.g. a high percentile of the noise.

    Returns:
        The ``quantile`` of ``residuals``.

    Raises:
        ValueError: If ``residuals`` is empty or ``quantile`` is outside (0, 1).
    """
    # Step 1: validate.
    residual_array = np.asarray(residuals, dtype=np.float64)
    if residual_array.size == 0:
        raise ValueError("residuals is empty; cannot measure a range margin")
    if not QUANTILE_LOWER_BOUND < quantile < QUANTILE_UPPER_BOUND:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    # Step 2: linear-interpolated quantile (numpy default method).
    return float(np.quantile(residual_array, quantile))


def sweep_pair_indices(sweep_count: int, stride: int, gap: int) -> List[Tuple[int, int]]:
    """Deterministic sweep index pairs (k, k + gap) for k = 0, stride, 2 stride, ...

    Args:
        sweep_count: Number of available sweeps.
        stride: Step between first indices, >= 1.
        gap: Index distance inside a pair, >= 1.

    Returns:
        List of (first_index, second_index) with second_index < sweep_count.

    Raises:
        ValueError: If ``stride`` or ``gap`` is below 1.
    """
    # Step 1: validate.
    if stride < MINIMUM_SWEEP_STEP or gap < MINIMUM_SWEEP_STEP:
        raise ValueError(f"stride and gap must be >= {MINIMUM_SWEEP_STEP}, got {stride}, {gap}")
    # Step 2: first indices stepping by stride while the partner k + gap still exists.
    return [
        (first_index, first_index + gap)
        for first_index in range(0, sweep_count, stride)
        if first_index + gap < sweep_count
    ]


def points_within_range(points_lidar: np.ndarray, max_range: float) -> np.ndarray:
    """Mask of points whose Euclidean distance from the sensor origin is <= ``max_range``.

    Args:
        points_lidar: (N, 3) points in the lidar frame (origin at the sensor).
        max_range: Maximum range in metres.

    Returns:
        (N,) boolean mask, True where ||xyz|| <= max_range.

    Raises:
        ValueError: If ``points_lidar`` is not (N, 3).
    """
    _check_points(points_lidar, "points_lidar")
    # Range is the Euclidean norm because the lidar frame's origin is the sensor.
    return np.linalg.norm(points_lidar, axis=1) <= max_range
