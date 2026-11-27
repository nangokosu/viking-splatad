"""Scored cells of a session: the voxel cells where the occupancy evaluation is scored.

A box is scored when its centre lies within the scored range and vertical field of view of some
recorded Ouster sweep (train, guard or test); its class neighbourhood is the box grown by the class
margin, restricted to the cells visible from the sweep poses. Everything here is evaluation-side:
ground-truth boxes only define which cells are scored and which objects own them, they never reach
a parser, seed set or loss (project rule 6).
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from viking_splatad.eval.detection import (
    class_margin_table,
    cuboid_point_masks,
    grow_cuboid_horizontally,
    range_limit_from_real_points,
    scored_cuboids_by_class,
    visible_cuboid_mask,
    visible_neighbourhood_mask,
    voxel_centers,
)
from viking_splatad.eval.scene_metrics import csr_from_lists
from viking_splatad.eval.session_rays import SessionSweep
from viking_splatad.labels.cuboids import compute_cuboid_geometry, points_in_cuboid

# Bin width (m) of the streamed return-range histogram behind the scored range limit. 1 cm is far
# below the cell edge, so binning cannot move the limit by a meaningful amount, and ~2e4 bins cover
# the farthest Ouster return (~200 m) in a few hundred kB.
RANGE_HISTOGRAM_BIN_M = 0.01


@dataclass(frozen=True)
class SweepScan:
    """Cheap per-session facts gathered in one streaming pass over the sweeps.

    Attributes:
        sweep_poses: lidar_to_map (4,4) of every sweep, time order.
        max_range_m: Farthest real return over all sweeps (``range_limit_from_real_points``). Reported
            for provenance only: a single stray return, not the scored range.
        sweep_count_by_split: split name -> number of sweeps.
        range_histogram_counts: Counts of all real return ranges in bins of RANGE_HISTOGRAM_BIN_M
            (bin b holds ranges in [b, b+1) * bin width); source of the robust scored range limit.
    """

    sweep_poses: List[np.ndarray]
    max_range_m: float
    sweep_count_by_split: Dict[str, int]
    range_histogram_counts: np.ndarray


def streaming_max_range(per_sweep_returns: Iterable[np.ndarray]) -> float:
    """Farthest real return over many sweeps, one sweep in memory at a time.

    ``range_limit_from_real_points`` is max over sweeps of (max over points of ||p||), so it
    decomposes: reduce each sweep to its own limit, then take the max of those. max is associative
    and commutative, hence the streamed value equals the all-at-once value exactly.

    Args:
        per_sweep_returns: Iterable of (n, 3) lidar-frame return arrays.

    Returns:
        Max range (m); 0.0 without any return (as ``range_limit_from_real_points``).
    """
    # Step 1: per-sweep reduction with the very function the evaluation uses.
    per_sweep_limits = (range_limit_from_real_points([sweep_returns]) for sweep_returns in per_sweep_returns)
    # Step 2: max of the per-sweep limits; default 0.0 mirrors the empty case.
    return float(max(per_sweep_limits, default=0.0))


def range_quantile_from_histogram(range_histogram_counts: np.ndarray, quantile: float) -> float:
    """Range below which a fraction ``quantile`` of all returns lie, read from the streamed histogram.

    With cumulative counts C_b over bins of width w, the result is the upper edge (b + 1) * w of the
    first bin with C_b >= quantile * C_total: an upper bound on the exact quantile, at most one bin
    (w) above it. Deterministic (pure counting).

    Args:
        range_histogram_counts: (B,) counts per RANGE_HISTOGRAM_BIN_M bin.
        quantile: Fraction in (0, 1].

    Returns:
        Range in metres; 0.0 without any return.
    """
    total_count = int(range_histogram_counts.sum())
    if total_count == 0:
        return 0.0
    # Step 1: first bin whose cumulative count reaches the requested share of all returns.
    cumulative_counts = np.cumsum(range_histogram_counts)
    bin_index = int(np.searchsorted(cumulative_counts, quantile * total_count, side="left"))
    # Step 2: that bin's upper edge.
    return float((bin_index + 1) * RANGE_HISTOGRAM_BIN_M)


def scan_sweeps(sweeps: Iterable[SessionSweep]) -> SweepScan:
    """One streaming pass: poses, farthest return, return-range histogram and sweep counts per split.

    Args:
        sweeps: Session sweeps in time order.

    Returns:
        ``SweepScan``.
    """
    sweep_poses: List[np.ndarray] = []
    sweep_count_by_split: Dict[str, int] = {}
    max_range_m = 0.0
    range_histogram_counts = np.zeros(0, dtype=np.int64)
    for sweep in sweeps:
        # The per-sweep limit is combined with a running max (see streaming_max_range).
        max_range_m = max(max_range_m, streaming_max_range([sweep.returns_xyz]))
        # Range histogram: bin index floor(||p|| / w); the counts array grows to the farthest bin seen.
        sweep_bins = np.bincount(np.floor(np.linalg.norm(sweep.returns_xyz, axis=1) / RANGE_HISTOGRAM_BIN_M)
                                 .astype(np.int64), minlength=len(range_histogram_counts))
        if len(sweep_bins) > len(range_histogram_counts):
            range_histogram_counts = np.pad(range_histogram_counts, (0, len(sweep_bins) - len(range_histogram_counts)))
        range_histogram_counts += sweep_bins
        sweep_poses.append(sweep.lidar_to_map)
        sweep_count_by_split[sweep.split] = sweep_count_by_split.get(sweep.split, 0) + 1
    return SweepScan(sweep_poses=sweep_poses, max_range_m=max_range_m, sweep_count_by_split=sweep_count_by_split,
                     range_histogram_counts=range_histogram_counts)


def scored_neighbourhoods(
    cuboids: Sequence[Any], scored_class_ids: Dict[str, int], sweep_poses: Sequence[np.ndarray],
    max_range: float, vertical_half_fov: float, grid: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, Any]]:
    """Per-class scored neighbourhood, with the rules of ``build_class_setup`` minus n_min.

    A box is scored when its centre is within range/FOV of some pose (here: of ANY recorded sweep);
    its class neighbourhood is the boxes grown by the class margin, restricted to cells visible from
    the poses.

    Args:
        cuboids: All GT boxes.
        scored_class_ids: Class name -> id of the scored classes.
        sweep_poses: lidar_to_map of every sweep that defines the scored region.
        max_range: Scored range (m).
        vertical_half_fov: Scored vertical half FOV (rad).
        grid: GT occupancy arrays (``origin``, ``voxel_size``, ``class_ids``).

    Returns:
        Class name -> {"class_id", "cuboids", "margin", "neighbourhood" (grid-shaped bool)}.
    """
    grid_origin, voxel_size = np.asarray(grid["origin"], dtype=np.float64), float(grid["voxel_size"])
    grid_shape = grid["class_ids"].shape
    class_ids = list(scored_class_ids.values())
    # Step 1: which boxes are scored, and each class's margin from all of its boxes in the session.
    box_visible = visible_cuboid_mask(cuboids, sweep_poses, max_range, vertical_half_fov)
    visible_by_class = scored_cuboids_by_class(cuboids, class_ids, box_visible)
    margins = class_margin_table(cuboids, class_ids)
    # Step 2: per class, the grown boxes restricted to the visible region.
    neighbourhoods: Dict[str, Dict[str, Any]] = {}
    for class_name, class_id in scored_class_ids.items():
        # No box of the class in the session (margin None): no growth (as build_class_setup).
        margin = margins[class_id] or 0.0
        neighbourhoods[class_name] = {
            "class_id": class_id, "cuboids": visible_by_class[class_id], "margin": margin,
            "neighbourhood": visible_neighbourhood_mask(
                visible_by_class[class_id], margin, grid_origin, voxel_size, grid_shape, sweep_poses,
                max_range, vertical_half_fov),
        }
    return neighbourhoods


def scored_cell_set(
    neighbourhoods: Dict[str, Dict[str, Any]], grid: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """Union of the class neighbourhoods as a list of cells.

    Args:
        neighbourhoods: ``scored_neighbourhoods`` output.
        grid: GT occupancy arrays.

    Returns:
        (cell_indices (V,3) int64, centres (V,3) map frame, {class name: (V,) bool membership},
        gt_class_ids (V,) uint8 = the grid's class id at each cell, 0 = none).
    """
    grid_origin, voxel_size = np.asarray(grid["origin"], dtype=np.float64), float(grid["voxel_size"])
    # Step 1: union mask, then its cells in C order (np.argwhere), the order every array shares.
    union_mask = np.logical_or.reduce([entry["neighbourhood"] for entry in neighbourhoods.values()])
    cell_indices = np.argwhere(union_mask)
    centres = voxel_centers(cell_indices, grid_origin, voxel_size)
    # Step 2: per-class membership and the grid label of every cell.
    neighbourhood_by_class = {
        class_name: entry["neighbourhood"][cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]]
        for class_name, entry in neighbourhoods.items()}
    gt_class_ids = grid["class_ids"][cell_indices[:, 0], cell_indices[:, 1], cell_indices[:, 2]].astype(np.uint8)
    return cell_indices, centres, neighbourhood_by_class, gt_class_ids


def object_membership(
    neighbourhoods: Dict[str, Dict[str, Any]], cell_indices: np.ndarray, centres: np.ndarray,
    grid: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """CSR cell lists of every scored object (one per scored cuboid, all scored classes).

    N_k = scored cells whose centre lies in cuboid k grown horizontally by its class margin (the
    same growth and centre-in-box test the neighbourhood uses); inside = the cell centre also lies
    in the original box k. A cell shared by two overlapping boxes is listed under both.

    Args:
        neighbourhoods: ``scored_neighbourhoods`` output.
        cell_indices: (V,3) scored cells (kept for interface symmetry; the lists index rows of it).
        centres: (V,3) cell centres.
        grid: GT occupancy arrays (unused beyond the shared interface).

    Returns:
        (object_box_ids (K,), object_class_ids (K,), offsets (K+1,), cells (nnz,) row indices into
        the scored cell list, inside (nnz,) bool).
    """
    del grid  # The centres already carry the grid geometry.
    if len(cell_indices) != len(centres):
        raise ValueError("cell_indices and centres must have the same length")
    box_ids: List[Any] = []
    class_ids: List[int] = []
    cell_lists: List[np.ndarray] = []
    inside_lists: List[np.ndarray] = []
    for entry in neighbourhoods.values():
        for cuboid in entry["cuboids"]:
            # Step 1: scored cells in the grown box (AABB prefilter + exact rotated test inside).
            grown_cells = cuboid_point_masks(centres, [grow_cuboid_horizontally(cuboid, entry["margin"])])[0]
            # Step 2: of those, the cells whose centre is in the original box.
            inside_mask = points_in_cuboid(compute_cuboid_geometry(cuboid), centres[grown_cells])
            box_ids.append(cuboid.box_id)
            class_ids.append(entry["class_id"])
            cell_lists.append(grown_cells.astype(np.int64))
            inside_lists.append(inside_mask)
    offsets, cells, inside = csr_from_lists(cell_lists, inside_lists)
    return np.asarray(box_ids), np.asarray(class_ids, dtype=np.int64), offsets, cells, inside



def build_scored_cells(
    sweeps: Iterable[SessionSweep], cuboids: Sequence[Any], scored_class_ids: Dict[str, int],
    grid: Dict[str, np.ndarray], vertical_half_fov: float, scored_range_quantile: float,
) -> Dict[str, Any]:
    """Compose the scored cells and their objects from one pass over the session's sweeps.

    Args:
        sweeps: Every session sweep, consumed once (a one-shot iterator is fine).
        cuboids: All GT boxes.
        scored_class_ids: Class name -> id.
        grid: GT occupancy arrays.
        vertical_half_fov: Half vertical FOV (rad).
        scored_range_quantile: Share of all real returns within the scored range limit. Over ~2000
            sweeps the farthest single return is a stray (~200 m on June, against 40 m for 99.99 % of
            returns), so the region that decides which boxes are scored uses this robust limit.

    Returns:
        Dict with the arrays of scored_cells.npz (cell_indices, centres, gt_class_ids,
        neighbourhood_by_class, object_box_ids, object_class_ids, object_cell_offsets, object_cells,
        object_inside) plus scan, scored_range_m, neighbourhoods.
    """
    # Step 1: one scan of the sweeps gives the poses and the return-range histogram.
    scan = scan_sweeps(sweeps)
    scored_range_m = range_quantile_from_histogram(scan.range_histogram_counts, scored_range_quantile)
    # Step 2: scored region, its cells and the per-object cell lists.
    neighbourhoods = scored_neighbourhoods(
        cuboids, scored_class_ids, scan.sweep_poses, scored_range_m, vertical_half_fov, grid)
    cell_indices, centres, neighbourhood_by_class, gt_class_ids = scored_cell_set(neighbourhoods, grid)
    object_box_ids, object_class_ids, offsets, cells, inside = object_membership(
        neighbourhoods, cell_indices, centres, grid)
    return {
        "scan": scan, "scored_range_m": scored_range_m, "neighbourhoods": neighbourhoods,
        "cell_indices": cell_indices, "centres": centres, "gt_class_ids": gt_class_ids,
        "neighbourhood_by_class": neighbourhood_by_class, "object_box_ids": object_box_ids,
        "object_class_ids": object_class_ids, "object_cell_offsets": offsets, "object_cells": cells,
        "object_inside": inside,
    }
