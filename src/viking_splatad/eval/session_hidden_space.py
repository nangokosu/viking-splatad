"""Session-wide hidden space H: voxel cells that no recorded lidar sweep of the session could see.

A scored cell is hidden when, over ALL recorded Ouster sweeps (train, guard and test), it was never
seen, at least one beam was blocked before it, and no beam passed through it (T78 evidence flags,
T91 rule). The range margin ``m`` that decides "short of the cell" is measured from the data (T92).
Everything here is evaluation-side: ground-truth boxes only define which cells are scored and which
objects own them, they never reach a parser, seed set or loss (project rule 6).
"""

import multiprocessing
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

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
from viking_splatad.eval.hidden_geometry import EVIDENCE_FLAG_COUNT, hidden_space_mask, recorded_ray_evidence
from viking_splatad.eval.hidden_metrics import csr_from_lists
from viking_splatad.eval.hidden_settings import HiddenEvalSettings
from viking_splatad.eval.range_margin import (
    cross_sweep_plane_residuals,
    points_within_range,
    range_margin,
    sweep_pair_indices,
)
from viking_splatad.eval.session_rays import (
    DEFAULT_LIDAR_SENSOR_NAME,
    RETURN_XYZ_COLUMNS,
    SPACE_DIMENSION,
    SessionSweep,
    count_session_sweeps,
    sweep_rays_in_map,
)
from viking_splatad.labels.cuboids import compute_cuboid_geometry, points_in_cuboid
from viking_splatad.splatad.viking_conversions import load_lidar_sweep, open_dataset_view, parse_ego_filter

# Sweeps handed to a worker per task: a task must be long enough to hide scheduling cost and short
# enough to balance load; the value only affects speed, never the result (OR-reduction).
SWEEPS_PER_TASK = 4
# Tasks kept in flight per worker; bounds memory because each task holds SWEEPS_PER_TASK sweeps.
IN_FLIGHT_TASKS_PER_WORKER = 2
# Half of a length: the half diagonal of a cube of edge a is a * sqrt(SPACE_DIMENSION) * HALF.
HALF = 0.5
# Fraction of the GT cell edge allowed as neighbour reach in the margin's plane fits. Half a cell
# is the spatial scale at which the margin must resolve cell membership; neighbours farther than
# that sample a different cell, so their plane fit measures surface shape rather than range noise
# plus pose error (June: m = 0.398 m unbounded vs 0.103 m with neighbours within 0.25 m).
NEIGHBOUR_REACH_VOXEL_FRACTION = 0.5
# Bin width (m) of the streamed return-range histogram behind the scored range limit. 1 cm is far
# below the measured range margin (0.1 to 0.4 m on June, with or without the neighbour reach bound), so binning cannot move the limit by a meaningful
# amount, and ~2e4 bins cover the farthest Ouster return (~200 m) in a few hundred kB.
RANGE_HISTOGRAM_BIN_M = 0.01


@dataclass(frozen=True)
class SweepScan:
    """Cheap per-session facts gathered in one streaming pass over the sweeps.

    Attributes:
        sweep_poses: lidar_to_map (4,4) of every sweep, time order.
        max_range_m: Farthest real return over all sweeps (``range_limit_from_real_points``). Exact
            bound for the per-sweep reach cull (no beam is longer).
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


# Map-frame axes that make the horizontal distance: x and y (z is up in the map frame).
HORIZONTAL_AXIS_COUNT = 2


def nearest_horizontal_distance(centres: np.ndarray, sweep_origins: np.ndarray) -> np.ndarray:
    """Horizontal (x, y) distance of each cell centre to the nearest sweep origin of the session.

    One KD-tree on the S sweep origins; query cost O(V log S).

    Args:
        centres: (V, 3) cell centres, map frame.
        sweep_origins: (S, 3) lidar translations, map frame.

    Returns:
        (V,) float64 distances (m).
    """
    tree = cKDTree(np.asarray(sweep_origins, dtype=np.float64)[:, :HORIZONTAL_AXIS_COUNT])
    return tree.query(np.asarray(centres, dtype=np.float64)[:, :HORIZONTAL_AXIS_COUNT])[0]


def near_band_hidden_mask(hidden: np.ndarray, nearest_sweep_distance_m: np.ndarray, near_band_m: float) -> np.ndarray:
    """Hidden cells within the near band of the driven path: H_near = H AND d_nearest <= near_band_m.

    Args:
        hidden: (V,) bool session hidden space H (T91).
        nearest_sweep_distance_m: (V,) horizontal distance to the nearest sweep origin (m).
        near_band_m: Band half-width (m), configs/eval.yaml hidden_space.near_band_m.

    Returns:
        (V,) bool.
    """
    return np.asarray(hidden, dtype=bool) & (np.asarray(nearest_sweep_distance_m) <= near_band_m)


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


def sweep_evidence(
    sweep: SessionSweep, centres: np.ndarray, voxel_size: float, range_margin_m: float, max_range: float,
) -> np.ndarray:
    """T78 evidence of one sweep on the cells it can reach.

    Cull: a beam has depth <= max_range. A cell can be seen only if entry - m <= depth, and passed
    only if exit + m < depth, so both need a cube entry <= max_range + m; the cube's nearest point
    is at least (centre distance - half diagonal) from the origin, so cells with centre distance
    > max_range + m + half diagonal give no seen/passed flag. Such cells are dropped. Their only
    possible flag, "blocked", is dropped with them on purpose: a sensor cannot observe beyond its
    range, so a wall in front of an out-of-range cell is no evidence about it.

    Args:
        sweep: One recorded sweep.
        centres: (V,3) cell centres in the map frame.
        voxel_size: Cell edge (m).
        range_margin_m: Range margin m (>= 0).
        max_range: Farthest real return of the session (m).

    Returns:
        (V,4) bool [seen, blocked, no_return, passed] (False on culled cells).
    """
    evidence = np.zeros((len(centres), EVIDENCE_FLAG_COUNT), dtype=bool)
    # Step 1: rays in the map frame (T93).
    origin, directions, depths, did_return = sweep_rays_in_map(
        sweep.returns_xyz, sweep.no_return_directions, sweep.lidar_to_map)
    # Step 2: cells that the sweep can reach (derivation in the docstring).
    half_diagonal = voxel_size * np.sqrt(float(SPACE_DIMENSION)) * HALF
    relative_centres = centres - origin
    reachable = np.linalg.norm(relative_centres, axis=1) <= max_range + range_margin_m + half_diagonal
    if not reachable.any():
        return evidence
    # Step 3: T78 on the reachable cells only, written back to their rows.
    evidence[reachable] = recorded_ray_evidence(
        relative_centres[reachable], voxel_size, directions, depths, did_return, range_margin=range_margin_m)
    return evidence


# Worker-side copy of the read-only inputs, set once per process by the pool initialiser so a task
# carries only its sweeps.
_WORKER_CONTEXT: Dict[str, Any] = {}


def _initialise_worker(centres: np.ndarray, voxel_size: float, range_margin_m: float, max_range: float) -> None:
    """Store the per-run constants in the worker process.

    Args:
        centres: (V,3) cell centres.
        voxel_size: Cell edge (m).
        range_margin_m: Range margin (m).
        max_range: Farthest return (m).
    """
    _WORKER_CONTEXT.update(
        centres=centres, voxel_size=voxel_size, range_margin_m=range_margin_m, max_range=max_range)


def _evidence_of_chunk(sweep_chunk: List[SessionSweep]) -> np.ndarray:
    """OR of ``sweep_evidence`` over a chunk of sweeps, using the worker context.

    Args:
        sweep_chunk: Sweeps of one task.

    Returns:
        (V,4) bool.
    """
    context = _WORKER_CONTEXT
    chunk_evidence = np.zeros((len(context["centres"]), EVIDENCE_FLAG_COUNT), dtype=bool)
    for sweep in sweep_chunk:
        chunk_evidence |= sweep_evidence(
            sweep, context["centres"], context["voxel_size"], context["range_margin_m"], context["max_range"])
    return chunk_evidence


def _chunks_of(sweeps: Iterable[SessionSweep], chunk_size: int):
    """Group an iterable of sweeps into lists of ``chunk_size`` (the last may be shorter).

    Args:
        sweeps: Sweep iterable (consumed lazily).
        chunk_size: Sweeps per chunk (>= 1).

    Yields:
        Lists of sweeps.
    """
    chunk: List[SessionSweep] = []
    for sweep in sweeps:
        chunk.append(sweep)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def accumulate_session_evidence(
    sweeps_iterable: Iterable[SessionSweep], centres: np.ndarray, voxel_size: float, range_margin_m: float,
    max_range: float, worker_count: Optional[int],
    progress_callback: Optional[Callable[[int], None]] = None,
) -> np.ndarray:
    """OR the T78 evidence of every sweep into one (V,4) array.

    The flags only ever turn on, so OR over sweeps is commutative and associative: the result does
    not depend on the order or on how sweeps are split between workers (deterministic).
    Sweeps are pulled lazily and at most ``worker_count * IN_FLIGHT_TASKS_PER_WORKER`` chunks are in
    flight, so memory stays bounded however many sweeps there are.

    Args:
        sweeps_iterable: Sweeps (any order).
        centres: (V,3) cell centres, map frame.
        voxel_size: Cell edge (m).
        range_margin_m: Range margin m (>= 0), from ``estimate_range_margin``.
        max_range: Farthest real return of the session (m), for the reach cull.
        worker_count: Processes; None = ``os.cpu_count()``; 1 runs in-process.
        progress_callback: Called with the number of sweeps finished so far.

    Returns:
        (V,4) bool [seen, blocked, no_return, passed].

    Raises:
        ValueError: If ``worker_count`` < 1.
    """
    resolved_worker_count = os.cpu_count() if worker_count is None else worker_count
    if resolved_worker_count is None or resolved_worker_count < 1:
        raise ValueError(f"worker_count must be >= 1, got {worker_count}")
    centres = np.asarray(centres, dtype=np.float64)
    total_evidence = np.zeros((len(centres), EVIDENCE_FLAG_COUNT), dtype=bool)
    finished_sweep_count = 0
    # Step 1: single process: the same functions, no pool.
    if resolved_worker_count == 1:
        _initialise_worker(centres, voxel_size, range_margin_m, max_range)
        for sweep_chunk in _chunks_of(sweeps_iterable, SWEEPS_PER_TASK):
            total_evidence |= _evidence_of_chunk(sweep_chunk)
            finished_sweep_count += len(sweep_chunk)
            if progress_callback is not None:
                progress_callback(finished_sweep_count)
        return total_evidence
    # Step 2: pool with bounded submission; results OR-ed as they arrive.
    in_flight_limit = resolved_worker_count * IN_FLIGHT_TASKS_PER_WORKER
    pending: deque = deque()

    def collect_oldest() -> None:
        """Wait for the oldest task, OR its evidence into the total and report progress.

        Returns:
            None; updates total_evidence and finished_sweep_count in place.
        """
        nonlocal finished_sweep_count
        task, chunk_size = pending.popleft()
        total_evidence.__ior__(task.get())
        finished_sweep_count += chunk_size
        if progress_callback is not None:
            progress_callback(finished_sweep_count)

    with multiprocessing.Pool(
        resolved_worker_count, initializer=_initialise_worker,
        initargs=(centres, voxel_size, range_margin_m, max_range),
    ) as worker_pool:
        for sweep_chunk in _chunks_of(sweeps_iterable, SWEEPS_PER_TASK):
            pending.append((worker_pool.apply_async(_evidence_of_chunk, (sweep_chunk,)), len(sweep_chunk)))
            if len(pending) >= in_flight_limit:
                collect_oldest()
        while pending:
            collect_oldest()
    return total_evidence


def neighbour_reach_from_voxel(voxel_size: float) -> float:
    """Largest neighbour distance allowed in the margin's plane fits, from the GT cell edge.

    Args:
        voxel_size: GT cell edge (m), > 0.

    Returns:
        ``NEIGHBOUR_REACH_VOXEL_FRACTION * voxel_size`` (m).

    Raises:
        ValueError: If ``voxel_size`` <= 0.
    """
    if voxel_size <= 0:
        raise ValueError(f"voxel_size must be > 0, got {voxel_size}")
    return NEIGHBOUR_REACH_VOXEL_FRACTION * voxel_size


def estimate_range_margin(
    dataset_dir: Path, split_settings: Dict[str, Any], settings: HiddenEvalSettings, voxel_size: float,
    lidar_sensor_name: str = DEFAULT_LIDAR_SENSOR_NAME,
) -> Dict[str, Any]:
    """Range margin m from cross-sweep point-to-plane residuals (T92) over deterministic sweep pairs.

    For each pair (first, second) of ``sweep_pair_indices``: A = returns of the first sweep within
    ``margin_max_range_m`` (lidar frame test) mapped to the map frame; B = all returns of the second
    mapped to the map frame; residuals of A against local planes of B. m = ``margin_quantile`` of
    the pooled residuals. Only the sweeps of the pairs are loaded.

    Args:
        dataset_dir: Extracted dataset directory.
        split_settings: Accepted for interface symmetry with ``iter_session_sweeps``; the margin does
            not depend on the split (all sweeps are candidates).
        settings: Validated hidden-space settings.
        voxel_size: GT cell edge (m); bounds the plane-fit neighbour reach via
            ``neighbour_reach_from_voxel``.
        lidar_sensor_name: Lidar stream name.

    Returns:
        {"margin_m", "residual_count", "pair_count", "quantile", "neighbour_reach_m"}.

    Raises:
        ValueError: If ``voxel_size`` <= 0, or no pair gives a residual (``range_margin`` refuses an empty set).
    """
    del split_settings  # The margin uses every sweep regardless of split.
    # Step 0: neighbour reach shared by every plane fit.
    reach = neighbour_reach_from_voxel(voxel_size)
    # Step 1: pairs of time-ordered sweep indices.
    view = open_dataset_view(Path(dataset_dir))
    ego_radius, ego_box = parse_ego_filter(view.manifest["lidar"]["min_lidar_dist"])
    sweep_paths = view.sensor_frame_paths[lidar_sensor_name]
    sweep_poses = view.sensor_poses[lidar_sensor_name]
    pairs = sweep_pair_indices(count_session_sweeps(dataset_dir, lidar_sensor_name),
                               settings.margin_sweep_stride, settings.margin_sweep_gap)

    def returns_in_map(sweep_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Returns of one sweep in the lidar frame and the map frame (p_map = R p + t).

        Args:
            sweep_index: Time-order index of the sweep.

        Returns:
            (lidar-frame points (n,3), map-frame points (n,3)), float64, ego-filtered.
        """
        lidar_points = load_lidar_sweep(sweep_paths[sweep_index], ego_radius, ego_box)[
            :, RETURN_XYZ_COLUMNS].astype(np.float64)
        pose = np.asarray(sweep_poses[sweep_index], dtype=np.float64)
        rotation, translation = pose[:SPACE_DIMENSION, :SPACE_DIMENSION], pose[:SPACE_DIMENSION, SPACE_DIMENSION]
        return lidar_points, lidar_points @ rotation.T + translation

    # Step 2: residuals per pair, pooled.
    residual_arrays: List[np.ndarray] = []
    for first_index, second_index in pairs:
        first_lidar, first_map = returns_in_map(first_index)
        _, second_map = returns_in_map(second_index)
        near_mask = points_within_range(first_lidar, settings.margin_max_range_m)
        residual_arrays.append(cross_sweep_plane_residuals(
            first_map[near_mask], second_map, settings.margin_neighbour_count, settings.margin_planarity_min,
            max_neighbour_reach_m=reach))
    residuals = np.concatenate(residual_arrays) if residual_arrays else np.empty(0, dtype=np.float64)
    # Step 3: the quantile is the margin.
    return {"margin_m": range_margin(residuals, settings.margin_quantile), "residual_count": int(residuals.size),
            "pair_count": len(pairs), "quantile": settings.margin_quantile, "neighbour_reach_m": reach}


def build_hidden_space(
    sweeps_factory: Callable[[], Iterable[SessionSweep]], cuboids: Sequence[Any], scored_class_ids: Dict[str, int],
    grid: Dict[str, np.ndarray], vertical_half_fov: float, range_margin_m: float, worker_count: Optional[int],
    scored_range_quantile: float, near_band_m: float, progress_callback: Optional[Callable[[int], None]] = None,
) -> Dict[str, Any]:
    """Compose the scored cells, their objects and the session evidence into one result.

    Two passes over the sweeps (hence a factory): a cheap scan for poses and the farthest return,
    then the evidence accumulation.

    Args:
        sweeps_factory: Zero-argument callable returning a fresh iterable of every session sweep.
        cuboids: All GT boxes.
        scored_class_ids: Class name -> id.
        grid: GT occupancy arrays.
        vertical_half_fov: Half vertical FOV (rad).
        range_margin_m: Range margin (m).
        worker_count: Processes (None = all cores).
        scored_range_quantile: Share of all real returns within the scored range limit. Over ~2000
            sweeps the farthest single return is a stray (~200 m on June, against 40 m for 99.99 % of
            returns), so the region that decides which boxes are scored uses this robust limit; the
            reach cull keeps the exact farthest return.
        near_band_m: Horizontal distance from the driven path that bounds the headline hidden space
            H_near (``near_band_hidden_mask``).
        progress_callback: Sweeps finished so far, during accumulation.

    Returns:
        Dict with the arrays of hidden_space.npz (cell_indices, centres, evidence, hidden, hidden_near,
        nearest_sweep_distance_m, gt_class_ids, neighbourhood_by_class, object_*) plus scan,
        scored_range_m, neighbourhoods.
    """
    voxel_size = float(grid["voxel_size"])
    # Step 1: scored region from every sweep.
    scan = scan_sweeps(sweeps_factory())
    scored_range_m = range_quantile_from_histogram(scan.range_histogram_counts, scored_range_quantile)
    neighbourhoods = scored_neighbourhoods(
        cuboids, scored_class_ids, scan.sweep_poses, scored_range_m, vertical_half_fov, grid)
    cell_indices, centres, neighbourhood_by_class, gt_class_ids = scored_cell_set(neighbourhoods, grid)
    object_box_ids, object_class_ids, offsets, cells, inside = object_membership(
        neighbourhoods, cell_indices, centres, grid)
    # Step 2: evidence over all sweeps, then H.
    evidence = accumulate_session_evidence(
        sweeps_factory(), centres, voxel_size, range_margin_m, scan.max_range_m, worker_count, progress_callback)
    hidden = hidden_space_mask(evidence)
    # Step 3: distance of every cell to the driven path, and the near-band headline hidden space.
    nearest_sweep_distance_m = nearest_horizontal_distance(
        centres, np.asarray([pose[:SPACE_DIMENSION, SPACE_DIMENSION] for pose in scan.sweep_poses]))
    return {
        "nearest_sweep_distance_m": nearest_sweep_distance_m,
        "hidden_near": near_band_hidden_mask(hidden, nearest_sweep_distance_m, near_band_m),
        "scan": scan, "scored_range_m": scored_range_m, "neighbourhoods": neighbourhoods, "cell_indices": cell_indices, "centres": centres,
        "evidence": evidence, "hidden": hidden, "gt_class_ids": gt_class_ids,
        "neighbourhood_by_class": neighbourhood_by_class, "object_box_ids": object_box_ids,
        "object_class_ids": object_class_ids, "object_cell_offsets": offsets, "object_cells": cells,
        "object_inside": inside,
    }
