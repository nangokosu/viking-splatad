"""Visibility-independent Gaussian occupancy and recorded-ray evidence for P2.

No labels or results from this module enter a parser, initialization or training
loss. Occlusion is geometric evidence; a nearer return is not a grass label.
"""

import numpy as np
from scipy.spatial import cKDTree

from viking_splatad.registry import transform

# PLAN section 5 defines alpha evaluation inside each three-sigma ellipsoid.
SIGMA_EXTENT = 3.0
EVIDENCE_NAMES = ("seen", "blocked", "no_return", "passed_through", "unobserved")
# Column positions of the T78 evidence array; they follow the order of EVIDENCE_NAMES.
SEEN_COLUMN, BLOCKED_COLUMN, NO_RETURN_COLUMN, PASSED_THROUGH_COLUMN = 0, 1, 2, 3
# T78 emits exactly the four flags above (the fifth name, "unobserved", is derived, not stored).
EVIDENCE_FLAG_COUNT = 4
# Memory policy for T78 (no effect on results). Far cells are queried FAR_CELL_CHUNK at a time; candidate
# pairs are then evaluated in slices of PAIR_SLICE, each pair costing ~25 float64/bool temporaries of
# 8 B (about 200 B), so a slice holds under ~210 MB. Near cells pair with all R rays, in chunks of
# at most NEAR_PAIR_BUDGET pairs (same bound).
FAR_CELL_CHUNK = 4096
PAIR_SLICE = 1 << 20
NEAR_PAIR_BUDGET = 1 << 20


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
    Combining by the product rule keeps it a blocking probability. Cost: T57's plus
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


def ray_cube_intervals(directions, centre, half_width):
    """Return exact positive ray/AABB entry/exit and hit mask using slab intersection.

    Args:
        directions: [R,3] unit rays from the local origin; centre: [3] cell centre.
        half_width: Positive half voxel edge.
    Returns:
        Entry [R], exit [R], hit [R]; parallel axes outside a slab cannot hit.
    """
    directions = np.asarray(directions, dtype=np.float64)
    lower, upper = np.asarray(centre) - half_width, np.asarray(centre) + half_width
    parallel = directions == 0
    outside_parallel = np.any(parallel & ((lower > 0) | (upper < 0)), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        first, second = lower / directions, upper / directions
    near = np.where(parallel, -np.inf, np.minimum(first, second)).max(axis=1)
    far = np.where(parallel, np.inf, np.maximum(first, second)).min(axis=1)
    near = np.maximum(near, 0)
    return near, far, (~outside_parallel) & (far >= near)


@transform("T78", "Recorded-ray evidence for shared evaluation cells, including unknown and conflicting observations")
def recorded_ray_evidence(centres_relative, voxel_size, directions, depths, did_return, *, range_margin=0.0):
    """Classify evidence from one recorded sweep without nearest-ray substitution.

    Args:
        centres_relative: [V,3] centres relative to ray origin, in voxel-grid axes; voxel_size: edge.
        directions: [R,3] recorded unit directions rotated to the same grid axes; depths: [R] positive depths.
        did_return: [R] boolean; false means an explicitly recorded no-return beam.
        range_margin: Keyword-only, finite and >= 0 (same unit as depths). Range noise makes a return
        within the margin of the cube ambiguous, so it counts as seen and a cell is hidden only when the
        evidence is unambiguous. With cube interval [entry, exit] on a beam and margin m (measured by T92
        in eval/range_margin.py), a returned depth d is: seen if entry - m <= d <= exit + m, blocked if
        d < entry - m, passed through if d > exit + m. m = 0 is the exact-geometry rule.
    Returns:
        [V,4] bool flags in seen/blocked/no_return/passed_through order. All may
        be false (unobserved). Different beams can give multiple evidence flags.
        Every candidate must intersect the cell; no ray means no observation.
        no_return ignores the margin (it concerns beams that gave no range).
    Raises:
        ValueError: on malformed shapes or values, or a non-finite or negative range_margin.
    """
    centres_relative, directions = np.asarray(centres_relative, dtype=np.float64), np.asarray(directions, dtype=np.float64)
    depths, did_return = np.asarray(depths, dtype=np.float64), np.asarray(did_return)
    if (centres_relative.ndim != 2 or centres_relative.shape[1:] != (3,) or directions.ndim != 2
            or directions.shape[1:] != (3,) or depths.shape != (len(directions),)
            or did_return.shape != depths.shape or did_return.dtype != bool or not np.isfinite(voxel_size)
            or voxel_size <= 0 or not all(np.isfinite(array).all() for array in (centres_relative, directions, depths))
            or np.any(depths <= 0) or not np.allclose(np.linalg.norm(directions, axis=1), 1)):
        raise ValueError("finite cells, positive depths, unit recorded directions and boolean return mask required")
    if not np.isfinite(range_margin) or range_margin < 0:
        raise ValueError("range_margin must be finite and >= 0")
    result = np.zeros((len(centres_relative), EVIDENCE_FLAG_COUNT), dtype=bool)
    if not len(directions):
        return result
    tree = cKDTree(directions)
    half_width = voxel_size / 2
    half_diagonal = np.sqrt(3) * half_width
    distances = np.linalg.norm(centres_relative, axis=1)
    # Step 1: near cells (origin within the bounding sphere) test every ray; far cells use the cone query.
    near_cells = np.flatnonzero(distances <= half_diagonal)
    far_cells = np.flatnonzero(~(distances <= half_diagonal))
    # Step 2: near cells in chunks of cells so that chunk * R pairs stays within NEAR_PAIR_BUDGET
    # (at least one cell per chunk, so R pairs when R alone exceeds the budget).
    near_chunk = max(1, NEAR_PAIR_BUDGET // len(directions))
    for chunk_start in range(0, len(near_cells), near_chunk):
        chunk_cells = near_cells[chunk_start:chunk_start + near_chunk]
        pair_cells = np.repeat(chunk_cells, len(directions))
        pair_rays = np.tile(np.arange(len(directions)), len(chunk_cells))
        _accumulate_pair_evidence(result, pair_cells, pair_rays, centres_relative, directions, depths,
                                  did_return, half_width, range_margin)
    # Step 3: far cells in chunks of FAR_CELL_CHUNK cells, one batched cone query per chunk.
    for chunk_start in range(0, len(far_cells), FAR_CELL_CHUNK):
        chunk_cells = far_cells[chunk_start:chunk_start + FAR_CELL_CHUNK]
        chunk_centres, chunk_distances = centres_relative[chunk_cells], distances[chunk_cells]
        # A cell fits inside a sphere of radius half_diagonal. Tangent rays bound a cone;
        # unit-sphere chord distance gives an exact tree query (same formula as the per-cell form).
        ratio = half_diagonal / chunk_distances
        chord = ratio * np.sqrt(2 / (1 + np.sqrt(1 - ratio ** 2)))
        candidate_lists = tree.query_ball_point(chunk_centres / chunk_distances[:, None],
                                                np.nextafter(chord, np.inf), return_sorted=False)
        # Flatten the ragged candidate lists into (cell, ray) pair arrays.
        list_lengths = np.fromiter((len(candidates) for candidates in candidate_lists), dtype=np.int64,
                                   count=len(candidate_lists))
        if not list_lengths.sum():
            continue
        pair_cells = np.repeat(chunk_cells, list_lengths)
        pair_rays = np.concatenate([np.asarray(candidates, dtype=np.int64) for candidates in candidate_lists])
        # Slice the pairs so the per-pair float arrays stay within PAIR_SLICE entries.
        for slice_start in range(0, len(pair_cells), PAIR_SLICE):
            slice_range = slice(slice_start, slice_start + PAIR_SLICE)
            _accumulate_pair_evidence(result, pair_cells[slice_range], pair_rays[slice_range], centres_relative,
                                      directions, depths, did_return, half_width, range_margin)
    return result


def ray_cube_intervals_pairs(directions, centres, half_width):
    """Slab entry/exit and hit mask for P (ray, cube) pairs; ray_cube_intervals with a centre per pair.

    The arithmetic is the same as ray_cube_intervals row by row (lower = centre - half_width, slab
    quotients, parallel-axis handling, near = max(near, 0)), so results are bit-identical.

    Args:
        directions: [P,3] unit rays; centres: [P,3] cube centre of each pair; half_width: half edge.
    Returns:
        Entry [P], exit [P], hit [P].
    """
    directions = np.asarray(directions, dtype=np.float64)
    lower, upper = np.asarray(centres) - half_width, np.asarray(centres) + half_width
    parallel = directions == 0
    outside_parallel = np.any(parallel & ((lower > 0) | (upper < 0)), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        first, second = lower / directions, upper / directions
    near = np.where(parallel, -np.inf, np.minimum(first, second)).max(axis=1)
    far = np.where(parallel, np.inf, np.maximum(first, second)).min(axis=1)
    near = np.maximum(near, 0)
    return near, far, (~outside_parallel) & (far >= near)


def _accumulate_pair_evidence(result, pair_cells, pair_rays, centres_relative, directions, depths,
                              did_return, half_width, range_margin):
    """OR the four T78 flags of (cell, ray) candidate pairs into result, in place.

    Args:
        result: [V,4] bool flags, updated; pair_cells, pair_rays: [P] cell and ray indices.
        centres_relative, directions, depths, did_return: the T78 inputs; half_width, range_margin: as T78.
    Returns:
        None. Cost O(P) time and memory; the per-cell OR is an indexed assignment of True.
    """
    # Exact slab test for every pair at once; keep only intersecting pairs.
    entry, exit_depth, hit = ray_cube_intervals_pairs(directions[pair_rays], centres_relative[pair_cells], half_width)
    pair_cells, pair_rays, entry, exit_depth = pair_cells[hit], pair_rays[hit], entry[hit], exit_depth[hit]
    returned, measured = did_return[pair_rays], depths[pair_rays]
    # Margin-widened interval [entry - m, exit + m]: returns inside it are seen, outside are blocked/passed.
    seen_start, seen_end = entry - range_margin, exit_depth + range_margin
    pair_flags = (returned & (measured >= seen_start) & (measured <= seen_end),
                  returned & (measured < seen_start),
                  ~returned,
                  returned & (measured > seen_end))
    for column, flags in zip((SEEN_COLUMN, BLOCKED_COLUMN, NO_RETURN_COLUMN, PASSED_THROUGH_COLUMN), pair_flags):
        # A cell has a flag when any of its pairs does: set True at each flagged pair's cell (idempotent).
        flagged_cells = pair_cells[flags]
        if len(flagged_cells):
            result[flagged_cells, column] = True


@transform("T91", "Hidden-space label: recorded beams blocked short of the cell, none seen inside or passed beyond it")
def hidden_space_mask(evidence):
    """Label cells hidden behind a nearer return, with no contradicting evidence.

    hidden = NOT seen AND blocked AND NOT passed_through. no_return is ignored by design: a beam that
    did not return gives no range, so it neither proves nor disproves occlusion.

    Args:
        evidence: [V,4] bool-castable T78 flags in seen/blocked/no_return/passed_through order.
    Returns:
        [V] bool mask of hidden cells.
    Raises:
        ValueError: if evidence is not a [V,4] array.
    """
    evidence = np.asarray(evidence)
    if evidence.ndim != 2 or evidence.shape[1] != EVIDENCE_FLAG_COUNT:
        raise ValueError("evidence must be a [V,4] array")
    evidence = evidence.astype(bool)
    return (~evidence[:, SEEN_COLUMN] & evidence[:, BLOCKED_COLUMN] & ~evidence[:, PASSED_THROUGH_COLUMN])


def exclusive_evidence_labels(evidence):
    """Return seen > blocked > no-return > passed-through > unobserved summary labels.

    Args:
        evidence: [V,4] accumulated flags; keep these too so conflicts stay visible.
    Returns:
        [V] indices into EVIDENCE_NAMES. Priority is an explicit reporting policy,
        not a claim that occluded or no-return cells are semantically grass.
    Raises:
        ValueError: if evidence is not a [V,4] array.
    """
    evidence = np.asarray(evidence, dtype=bool)
    if evidence.ndim != 2 or evidence.shape[1] != 4:
        raise ValueError("four evidence columns required")
    # argmax returns the first True column, i.e. the highest-priority flag; no flag -> "unobserved" (last name).
    return np.where(evidence.any(axis=1), evidence.argmax(axis=1), len(EVIDENCE_NAMES) - 1)


def occupied_cell_metrics(occupied, truth, evidence):
    """Score shared cells and report GT recall separately for every observation stratum.

    Args:
        occupied, truth: [V] model and GT bool masks in the frozen neighbourhood.
        evidence: [V,4] measured evidence; used only to stratify GT recall.
    Returns:
        Counts, precision/recall/IoU and per-stratum GT recall. Undefined ratios
        are None. Off-GT occupancy is neighbourhood FP, not an unmatched-tree count.
        "gt_cells_with_conflicting_evidence" counts GT cells with more than one flag set.
    Raises:
        ValueError: if the masks and evidence rows do not match one-dimensionally.
    """
    occupied, truth = np.asarray(occupied, dtype=bool), np.asarray(truth, dtype=bool)
    labels = exclusive_evidence_labels(evidence)
    if occupied.ndim != 1 or occupied.shape != truth.shape or labels.shape != truth.shape:
        raise ValueError("matched one-dimensional cell masks required")
    true_positive = int(np.count_nonzero(occupied & truth))
    false_positive = int(np.count_nonzero(occupied & ~truth))
    false_negative = int(np.count_nonzero(~occupied & truth))
    strata = {}
    for label_index, name in enumerate(EVIDENCE_NAMES):
        selected = truth & (labels == label_index)
        count, recovered = int(selected.sum()), int((selected & occupied).sum())
        strata[name] = {"gt_cells": count, "recovered_cells": recovered,
                        "recall": recovered / count if count else None}
    predicted_count, truth_count = true_positive + false_positive, true_positive + false_negative
    union_count = predicted_count + false_negative
    return {"true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
            "precision": true_positive / predicted_count if predicted_count else None,
            "recall": true_positive / truth_count if truth_count else None,
            "iou": true_positive / union_count if union_count else None, "strata": strata,
            "gt_cells_with_conflicting_evidence": int(np.count_nonzero(truth & (np.asarray(evidence).sum(axis=1) > 1)))}
