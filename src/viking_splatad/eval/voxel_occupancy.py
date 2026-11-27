"""Visibility-independent Gaussian occupancy at voxel centres, the P2 occupancy estimators.

No labels or results from this module enter a parser, initialization or training loss.
"""

import numpy as np
from scipy.spatial import cKDTree

from viking_splatad.registry import transform

# PLAN section 5 defines alpha evaluation inside each three-sigma ellipsoid.
SIGMA_EXTENT = 3.0


@transform("T57", "Three-sigma Gaussian product occupancy at shared map-frame voxel centres")
def gaussian_occupancy_at_centres(centres, means, covariances, opacities, chunk_size,
                                 principal_axes=None, axis_std=None, excluded_centre_per_gaussian=None):
    """Evaluate 1-prod(1-alpha) exactly on supplied voxel centres, within 3 sigma.

    Args:
        centres: [V,3] map-frame voxel centres, identical across compared arms.
        means, covariances, opacities: [G,3], [G,3,3], [G] exported static model.
        chunk_size: Positive CPU working-batch size, a memory policy only.
        principal_axes, axis_std: Optional stable checkpoint factors [G,3,3]/[G,3];
        pass covariances=None in this mode. No eigenvalue repair or clipping occurs.
        excluded_centre_per_gaussian: Optional [G] int; Gaussian g skips centre index
        excluded_centre_per_gaussian[g] (negative = none). T81 uses it to replace the centre value in a
        Gaussian's own voxel by its peak; omitted, every centre is evaluated (plain T57).
    Returns:
        [V] probabilities. A conservative sphere finds candidate cells; the final
        predicate is Mahalanobis squared <= 9. Overlap accumulates log survival,
        not summed alpha. Cost is O((G+V)log V + candidate pairs), not G*V.
    """
    # Step 1: validate inputs at the boundary (factor mode xor covariance mode, shapes, finiteness).
    centres, means = np.asarray(centres, dtype=np.float64), np.asarray(means, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64)
    factorized = principal_axes is not None or axis_std is not None
    if factorized:
        principal_axes, axis_std = np.asarray(principal_axes, dtype=np.float64), np.asarray(axis_std, dtype=np.float64)
        if (covariances is not None or principal_axes.shape != (len(means), 3, 3) or axis_std.shape != (len(means), 3)
                or not np.isfinite(principal_axes).all() or not np.isfinite(axis_std).all() or np.any(axis_std <= 0)):
            raise ValueError("finite principal factors and strictly positive scales required; no covariance in factor mode")
    else:
        covariances = np.asarray(covariances, dtype=np.float64)
    if (centres.ndim != 2 or centres.shape[1:] != (3,) or means.ndim != 2 or means.shape[1:] != (3,)
            or (not factorized and covariances.shape != (len(means), 3, 3)) or opacities.shape != (len(means),)
            or isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0):
        raise ValueError("valid voxel/model shapes and positive integer chunk size required")
    if (not all(np.isfinite(array).all() for array in (centres, means, opacities))
            or (not factorized and not np.isfinite(covariances).all())
            or np.any((opacities < 0) | (opacities > 1))):
        raise ValueError("finite geometry and opacities in [0,1] required")
    if excluded_centre_per_gaussian is None:
        excluded_centre_per_gaussian = np.full(len(means), -1, dtype=np.int64)
    excluded_centre_per_gaussian = np.asarray(excluded_centre_per_gaussian)
    if excluded_centre_per_gaussian.shape != (len(means),) or not np.issubdtype(excluded_centre_per_gaussian.dtype, np.integer):
        raise ValueError("one integer excluded centre index per Gaussian required")
    # Step 2: per centre, log survival log prod_g(1 - alpha_g) = sum_g log1p(-alpha_g), accumulated per Gaussian.
    survival_log = np.zeros(len(centres), dtype=np.float64)
    if not len(centres):
        return survival_log
    tree = cKDTree(centres)
    for start_index in range(0, len(means), chunk_size):
        stop_index = min(len(means), start_index + chunk_size)
        if factorized:
            axes = principal_axes[start_index:stop_index]
            scales = axis_std[start_index:stop_index]
            inverse_axes = np.linalg.inv(axes)
            # The 3-sigma ellipsoid of Sigma = A diag(std^2) A^T reaches 3 * sigma_max(A diag(std));
            # the Frobenius norm bounds sigma_max from above, so the tree sphere is conservative.
            radii = SIGMA_EXTENT * np.linalg.norm(axes * scales[:, None, :], axis=(1, 2))
        else:
            covariance_batch = covariances[start_index:stop_index]
            if not np.allclose(covariance_batch, covariance_batch.transpose(0, 2, 1), rtol=0,
                           atol=np.finfo(np.float32).eps * np.max(np.abs(covariance_batch))):
                raise ValueError("symmetric positive-definite covariance required")
            try:
                np.linalg.cholesky(covariance_batch)
            except np.linalg.LinAlgError as error:
                raise ValueError("symmetric positive-definite covariance required") from error
            # trace(Sigma) = sum of eigenvalues >= lambda_max, so 3 sqrt(trace) bounds the 3-sigma extent.
            radii = SIGMA_EXTENT * np.sqrt(np.trace(covariance_batch, axis1=1, axis2=2))
        # One ulp outward so a centre exactly on the bound is not lost to rounding.
        radii = np.nextafter(radii, np.inf)
        # One Gaussian query at a time bounds index-list storage by V; tested
        # candidate arithmetic is further batched, with no neighbor truncation.
        for local_index, radius in enumerate(radii):
            gaussian_index = start_index + local_index
            if opacities[gaussian_index] == 0:
                continue
            candidates = tree.query_ball_point(means[gaussian_index], radius)
            if not candidates:
                continue
            if not factorized:
                inverse = np.linalg.inv(covariance_batch[local_index])
            for candidate_start in range(0, len(candidates), chunk_size):
                indices = np.asarray(candidates[candidate_start:candidate_start + chunk_size], dtype=np.int64)
                differences = centres[indices] - means[gaussian_index]
                if factorized:
                    # d^T Sigma^-1 d = ||diag(1/std) A^-1 d||^2 for Sigma = A diag(std^2) A^T.
                    normalized = (differences @ inverse_axes[local_index].T) / scales[local_index]
                    distance_squared = np.einsum("ni,ni->n", normalized, normalized)
                else:
                    distance_squared = np.einsum("ni,ij,nj->n", differences, inverse, differences)
                # 3-sigma cut, minus the centre this Gaussian is told to skip (negative skips none).
                inside = (distance_squared <= SIGMA_EXTENT ** 2) & (indices != excluded_centre_per_gaussian[gaussian_index])
                indices, distance_squared = indices[inside], distance_squared[inside]
                alpha = opacities[gaussian_index] * np.exp(-0.5 * distance_squared)
                # alpha=1 has zero survival and exactly unit occupancy.
                with np.errstate(divide="ignore"):
                    survival_log[indices] += np.log1p(-alpha)
    return -np.expm1(survival_log)


@transform("T81", "Peak-alpha voxel occupancy: sigma(o) in the voxel holding the mean, T57 3-sigma tails elsewhere")
def peak_alpha_voxel_occupancy(cell_indices, grid_origin, voxel_size, means, opacities, principal_axes, axis_std, chunk_size):
    """Voxel occupancy 1 - prod_k(1 - alpha_kv) with each Gaussian's peak alpha in its own voxel.

    SplatAD's Gaussians are thin discs far smaller than a GT voxel, so T57's centre sample misses
    almost every surface (most 3-sigma ellipsoids reach no voxel centre). With o_k the activated
    opacity sigma(logit_k) in [0, 1], Gaussian k adds:
    - alpha_kv = o_k, its peak alpha (at its mean), to the voxel v that contains its mean, in
      place of its T57 value at that voxel's centre;
    - alpha_kv = o_k exp(-d^2/2) at the centre of every other scored voxel within 3 sigma (T57),
      d the Mahalanobis distance.
    In a Gaussian's own voxel its peak opacity counts, and the peaks of several Gaussians sit at
    different points of that voxel, so these Gaussians count for more than one ray through the voxel
    would collect from them. It reuses SplatAD's product rule 1 - prod(1 - alpha). Source: SplatAD's own alpha-blending
    (arXiv:2411.16816, Eq. 4; its median range is where prod_j(1 - alpha_j) drops below 0.5, section 3.3),
    whose weights alpha_i prod_{j<i}(1 - alpha_j) sum to 1 - prod_i(1 - alpha_i) by telescoping and do not
    depend on Gaussian order; see reports/gaussian_occupancy/terminology.md. Cost: T57's plus
    O((V + G) log V) for the home-cell lookup.

    Args:
        cell_indices: [V,3] integer GT-grid indices of the scored cells (unique).
        grid_origin: [3] grid minimum corner; voxel_size: positive edge (m).
        means, opacities, principal_axes, axis_std: [G,3], [G] activated opacities, [G,3,3], [G,3]
            factorized Gaussians (T56).
        chunk_size: Positive CPU working-batch size (memory policy only).
    Returns:
        [V] probabilities. A Gaussian whose mean lies outside every scored cell adds only tails.
    """
    cell_indices = np.asarray(cell_indices, dtype=np.int64)
    means, opacities = np.asarray(means, dtype=np.float64), np.asarray(opacities, dtype=np.float64)
    grid_origin = np.asarray(grid_origin, dtype=np.float64)
    if (cell_indices.ndim != 2 or cell_indices.shape[1:] != (3,) or len(np.unique(cell_indices, axis=0)) != len(cell_indices)
            or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError("unique [V,3] cell indices and a positive voxel size required")
    centres = grid_origin + (cell_indices + 0.5) * voxel_size
    # Step 1: home cell of each Gaussian. Index = floor((mu - origin) / edge), the inverse of the
    # centre rule; it is looked up among the scored cells by flat key (-1 = not a scored cell).
    home_indices = np.floor((means - grid_origin) / voxel_size).astype(np.int64)
    home_cell = np.full(len(means), -1, dtype=np.int64)
    if len(cell_indices):
        # Keys over the scored cells' bounding box only: a mean outside it cannot be in a scored
        # cell, and far-away Gaussians (e.g. random init points) would overflow a wider key space.
        key_offset = cell_indices.min(axis=0)
        key_extent = cell_indices.max(axis=0) - key_offset + 1
        in_box = np.all((home_indices >= key_offset) & (home_indices < key_offset + key_extent), axis=1)
        cell_keys = np.ravel_multi_index((cell_indices - key_offset).T, key_extent)
        gaussian_keys = np.full(len(means), -1, dtype=np.int64)
        gaussian_keys[in_box] = np.ravel_multi_index((home_indices[in_box] - key_offset).T, key_extent)
        # Binary search in the sorted keys; a hit maps back to the cell's original position.
        key_order = np.argsort(cell_keys)
        sorted_keys = cell_keys[key_order]
        positions = np.minimum(np.searchsorted(sorted_keys, gaussian_keys), len(sorted_keys) - 1)
        home_cell = np.where(sorted_keys[positions] == gaussian_keys, key_order[positions], -1)
    # Step 2: tails at every other centre (T57 with the home centre skipped).
    tail_occupancy = gaussian_occupancy_at_centres(centres, means, None, opacities, chunk_size,
                                                   principal_axes=principal_axes, axis_std=axis_std,
                                                   excluded_centre_per_gaussian=home_cell)
    # Step 3: peaks in the home cells, as log survival sum_k log(1 - o_k) per cell (o_k = 1 gives -inf, i.e. occupancy 1).
    home_survival_log = np.zeros(len(centres), dtype=np.float64)
    in_scored_cell = home_cell >= 0
    with np.errstate(divide="ignore"):
        np.add.at(home_survival_log, home_cell[in_scored_cell], np.log1p(-opacities[in_scored_cell]))
    # Product rule over both parts: 1 - (1 - tail) * prod(1 - peak).
    return 1.0 - (1.0 - tail_occupancy) * np.exp(home_survival_log)
