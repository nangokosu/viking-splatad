"""P2 cuboid detection and the occupancy-threshold sweep on shared GT-grid cells.

Inputs are arrays over V scored cells: map-frame voxel centres [V,3] and
boolean masks [V] (GT truth, P2 Gaussian occupancy). Nothing here enters a
parser, seed set or training loss; GT cuboids are used only to score
(project rule 6).
"""

import numpy as np

from viking_splatad.eval.scene_metrics import pooled_metrics
from viking_splatad.labels.cuboids import compute_cuboid_geometry, points_in_cuboid
from viking_splatad.registry import transform

# An occupied fraction lies in [0, 1] by construction (occupied cells are a
# subset of GT cells), so the f_min grid spans exactly this closed interval.
F_MIN_LOWER = 0.0
F_MIN_UPPER = 1.0
# A grid must hold both ends of [F_MIN_LOWER, F_MIN_UPPER], hence two points minimum.
MIN_F_MIN_POINT_COUNT = 2


def _validated_cell_mask(mask, cell_count, name):
    """Return a mask as a numpy bool array after checking its dtype and shape.

    Args:
        mask: Array-like [V] that must already have bool dtype (no silent casts
            of probabilities or integer labels).
        cell_count: Expected V.
        name: Argument name used in the error message.
    Returns:
        numpy.ndarray [V] of bool.
    Raises:
        ValueError: on non-bool dtype or a shape other than (V,).
    """
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (cell_count,):
        raise ValueError(f"{name} must be a boolean [V] mask matching the scored cells")
    return mask


@transform("T79", "P2 cuboid detection: per-cuboid occupied-voxel fraction and detection curve over f_min")
def cuboid_occupied_fractions(cuboids, centres, truth, occupied):
    """Count each cuboid's GT cells and the share of them that P2 marks occupied.

    A cell belongs to a cuboid when its centre passes points_in_cuboid (box-frame
    |p| <= half extent + boundary tolerance on every axis). For cuboid k:
        gt_cells[k]       = #{v : inside_k(v) and truth[v]}
        occupied_cells[k] = #{v : inside_k(v) and truth[v] and occupied[v]}
        fraction[k]       = occupied_cells[k] / gt_cells[k]   (NaN if gt_cells[k] == 0)
    Cost O(K * V) for K cuboids.

    Args:
        cuboids: Sequence of K labels.cuboids.Cuboid in the map frame.
        centres: [V,3] finite map-frame voxel centres.
        truth: [V] bool GT occupancy of the scored cells.
        occupied: [V] bool P2 occupancy (thresholded T81 probability).
    Returns:
        dict with "gt_cells" [K] int64, "occupied_cells" [K] int64 and
        "fraction" [K] float64 arrays, in cuboid order.
    Raises:
        ValueError: on malformed centres or masks.
    """
    # Step 1: validate the shared cell arrays once at the boundary.
    centres = np.asarray(centres, dtype=np.float64)
    if centres.ndim != 2 or centres.shape[1:] != (3,) or not np.isfinite(centres).all():
        raise ValueError("finite [V,3] voxel centres required")
    cell_count = len(centres)
    truth = _validated_cell_mask(truth, cell_count, "truth")
    occupied = _validated_cell_mask(occupied, cell_count, "occupied")
    # Step 2: per cuboid, intersect its inside mask with GT, then with occupancy.
    cuboid_count = len(cuboids)
    gt_cells = np.zeros(cuboid_count, dtype=np.int64)
    occupied_cells = np.zeros(cuboid_count, dtype=np.int64)
    for cuboid_index, cuboid in enumerate(cuboids):
        inside_truth = points_in_cuboid(compute_cuboid_geometry(cuboid), centres) & truth
        gt_cells[cuboid_index] = np.count_nonzero(inside_truth)
        occupied_cells[cuboid_index] = np.count_nonzero(inside_truth & occupied)
    # Step 3: ratio only where defined; zero-GT cuboids stay NaN (excluded downstream).
    fraction = np.full(cuboid_count, np.nan, dtype=np.float64)
    defined = gt_cells > 0
    fraction[defined] = occupied_cells[defined] / gt_cells[defined]
    return {"gt_cells": gt_cells, "occupied_cells": occupied_cells, "fraction": fraction}


def detection_curve(fractions, f_min_values):
    """Detection rate of cuboids as a function of the occupied-fraction threshold.

    rate(f_min) = #{k defined : fraction[k] >= f_min} / #{k defined}, where a cuboid
    is defined when its fraction is finite (it has at least one GT cell). The rate
    is non-increasing in f_min. Undefined cuboids are left out of the denominator
    and reported as a count instead.

    Args:
        fractions: [K] float fractions from cuboid_occupied_fractions (NaN = undefined).
        f_min_values: [F] finite thresholds.
    Returns:
        dict with "f_min" (list of F floats), "detection_rate" (list of F floats,
        or of None when no cuboid is defined), "detected_count" (list of F ints),
        "defined_cuboids" and "undefined_cuboids" (ints).
    Raises:
        ValueError: on non-1-D inputs, non-finite thresholds, or defined
            fractions outside [0, 1].
    """
    # Step 1: validate; NaN is the only allowed non-finite fraction value.
    fractions = np.asarray(fractions, dtype=np.float64)
    f_min_values = np.asarray(f_min_values, dtype=np.float64)
    if fractions.ndim != 1 or f_min_values.ndim != 1 or not np.isfinite(f_min_values).all():
        raise ValueError("one-dimensional fractions and finite one-dimensional f_min values required")
    defined = ~np.isnan(fractions)
    defined_fractions = fractions[defined]
    if np.any(~np.isfinite(defined_fractions) | (defined_fractions < F_MIN_LOWER)
              | (defined_fractions > F_MIN_UPPER)):
        raise ValueError("defined fractions must lie in [0, 1]")
    defined_count = int(defined_fractions.size)
    # Step 2: count detections per threshold; the rate is None when nothing is defined.
    detected_counts = [int(np.count_nonzero(defined_fractions >= f_min)) for f_min in f_min_values]
    rates = [detected / defined_count if defined_count else None for detected in detected_counts]
    return {"f_min": [float(f_min) for f_min in f_min_values], "detection_rate": rates,
            "detected_count": detected_counts, "defined_cuboids": defined_count,
            "undefined_cuboids": int(fractions.size - defined_count)}


def f_min_grid(point_count):
    """Evenly spaced f_min thresholds over [0, 1], both ends included.

    Value k is F_MIN_LOWER + (F_MIN_UPPER - F_MIN_LOWER) * k / (point_count - 1), computed as one
    division per value. numpy.linspace accumulates rounding (its 4th of 11 points is
    0.30000000000000004), so a cuboid with exactly 3 of 10 cells occupied (fraction 3/10) would
    miss f_min = 0.3. Division gives the nearest double of each exact ratio, and that rounding is
    monotone, so equal ratios compare equal and the >= test keeps the exact order.

    Args:
        point_count: Integer number of thresholds, at least MIN_F_MIN_POINT_COUNT.
    Returns:
        numpy.ndarray [point_count] of float64 thresholds.
    Raises:
        ValueError: when point_count is not an int (bool rejected) or is below MIN_F_MIN_POINT_COUNT.
    """
    if isinstance(point_count, bool) or not isinstance(point_count, (int, np.integer)) \
            or point_count < MIN_F_MIN_POINT_COUNT:
        raise ValueError(f"integer point_count >= {MIN_F_MIN_POINT_COUNT} required")
    step_indices = np.arange(int(point_count), dtype=np.float64)
    return F_MIN_LOWER + (F_MIN_UPPER - F_MIN_LOWER) * step_indices / (int(point_count) - 1)


def threshold_sweep(probabilities, truth, thresholds):
    """Score P2 occupancy at several probability thresholds (floater/haze sensitivity).

    occupied(t) = probabilities >= t, scored over every cell with scene_metrics.pooled_metrics.
    Raising t shrinks the occupied set, so recall is non-increasing in t.

    Args:
        probabilities: [V] T81 occupancy probabilities in [0, 1].
        truth: [V] bool GT occupancy.
        thresholds: [T] finite thresholds.
    Returns:
        List of T dicts {"threshold": float, "metrics": pooled_metrics output}.
    Raises:
        ValueError: on malformed probabilities, masks or thresholds.
    """
    # Step 1: validate probabilities and thresholds; masks share V.
    probabilities = np.asarray(probabilities, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if (probabilities.ndim != 1 or not np.isfinite(probabilities).all()
            or np.any((probabilities < F_MIN_LOWER) | (probabilities > F_MIN_UPPER))
            or thresholds.ndim != 1 or not np.isfinite(thresholds).all()):
        raise ValueError("[V] probabilities in [0, 1] and finite one-dimensional thresholds required")
    cell_count = len(probabilities)
    truth = _validated_cell_mask(truth, cell_count, "truth")
    # Step 2: every cell is scored, so the neighbourhood is all True.
    whole_scene = np.ones_like(truth, dtype=bool)
    # Step 3: one metric set per threshold.
    return [{"threshold": float(threshold),
             "metrics": pooled_metrics(probabilities >= threshold, truth, whole_scene)}
            for threshold in thresholds]
