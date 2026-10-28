"""Rendered test lidar scans -> map-frame point sets (T67, T68), numpy only.

Reads the per-sweep npz files written by ``scripts/08_render_lidar_pointcloud.py`` (keys in
``runs/RUN_FOLDER_GLOSSARY.md``) and gives, per sweep, two rendered point sets in the lidar frame:

* **gated**: ``pred_points`` as saved (SplatAD's rule: ray-drop probability <= 0.5, valid ray);
* **ungated** (PLAN §5 P1, binding): every valid ray whose accumulation Σw_i = 1 − ∏(1 − α_i)
  reaches the threshold, at its rendered depth (T67).

SplatAD builds a point as ``depth · direction(az, el) + v · t_c`` (splatad.py
``filter_lidar_pred_and_gt``): ``v`` is the sweep's local linear velocity and ``t_c`` the ray time
after the rolling-shutter centring shift (t − (max t + min t)/2 over the raster). ``v`` is not saved,
so T67 fits it on the gated rows, which carry the full point and ``t_c`` (their ``t`` column), checks
that the fit rebuilds them, and applies it to the ungated rays. Points then go to the map frame with the stored ``lidar_to_map`` (T50) and onto the
GT occupancy grid (T68).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.registry import transform

# SplatAD's ray-drop gate (splatad.py filter_lidar_pred_and_gt: sigmoid(ray_drop) <= 0.5).
RAY_DROP_GATE = 0.5
# Columns of pred_points (scripts/08 RENDERED_POINT_COLUMNS): x, y, z, intensity, t.
PRED_POINT_INTENSITY_COLUMN = 3
PRED_POINT_TIME_COLUMN = 4
# Columns of raster_pts (scripts/08 docstring): azimuth deg, elevation deg, depth m, t, intensity.
RASTER_AZIMUTH_COLUMN = 0
RASTER_ELEVATION_COLUMN = 1
RASTER_DEPTH_COLUMN = 2
RASTER_TIME_COLUMN = 3
# xyz columns shared by pred_points and gt_points.
XYZ_COLUMNS = slice(0, 3)
# Float32 roundings in the chain that made a stored point and its rebuild: SplatAD's deg->rad,
# cos/sin, depth·direction, velocity·time, sum and store (6), plus the stored depth and the stored
# time inputs (2). The rebuild may differ by at most one unit in the last place (at the largest
# coordinate) per rounding; a misaligned ray is off by centimetres or more.
REBUILD_ROUNDING_COUNT = 8
# Lidar npz suffix (scripts/08 writes <idx>_<ts_ns>.npz).
LIDAR_NPZ_SUFFIX = ".npz"


@dataclass(frozen=True)
class RenderedSweep:
    """One rendered test sweep, points in its own lidar frame.

    Attributes:
        stem: ``<idx>_<ts_ns>`` file stem (the input sweep's name).
        lidar_to_map: (4, 4) float64 sweep pose in the map frame (T50).
        gated_points: (M, 3) ``pred_points`` xyz.
        ungated_points: (K, 3) T67 points, or None when the npz has no ``pred_accumulation``
            (renders made before 2026-10-26).
        real_points: (N, 3) ``gt_points`` xyz: the real sweep's returns.
    """

    stem: str
    lidar_to_map: np.ndarray
    gated_points: np.ndarray
    ungated_points: Optional[np.ndarray]
    real_points: np.ndarray


def ray_directions(azimuth_degrees: np.ndarray, elevation_degrees: np.ndarray) -> np.ndarray:
    """Unit ray directions in the lidar frame, SplatAD's spherical convention.

    d = (cos el · cos az, cos el · sin az, sin el): x forward, y left, z up.

    Args:
        azimuth_degrees: (R,) azimuths.
        elevation_degrees: (R,) elevations.

    Returns:
        (R, 3) float64 unit vectors.
    """
    azimuth = np.deg2rad(np.asarray(azimuth_degrees, dtype=np.float64))
    elevation = np.deg2rad(np.asarray(elevation_degrees, dtype=np.float64))
    return np.stack(
        [np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], axis=-1
    )


def smallest_positive_step(values: np.ndarray) -> float:
    """Smallest gap between distinct values.

    Args:
        values: Any array.

    Returns:
        min positive difference of the sorted unique values; 0.0 when fewer than two distinct values.
    """
    unique_values = np.unique(np.asarray(values, dtype=np.float64))
    if unique_values.size < 2:
        return 0.0
    return float(np.min(np.diff(unique_values)))


def resolution_at_gate(stored_probabilities: np.ndarray, gate: float) -> float:
    """Resolution of a stored (quantised) probability array at the gate value.

    = gap between the largest stored value <= gate and the smallest stored value > gate. A ray whose
    exact probability passed the gate is stored at most this far above it, whatever the storage
    precision (the saved ``pred_ray_drop_prob`` is coarser than the value the gate used).

    Args:
        stored_probabilities: Stored probabilities of the valid rays.
        gate: Gate value.

    Returns:
        The gap; 0.0 when no value lies on one side of the gate (nothing is ambiguous then).
    """
    unique_values = np.unique(np.asarray(stored_probabilities, dtype=np.float64))
    at_or_below, above = unique_values[unique_values <= gate], unique_values[unique_values > gate]
    if at_or_below.size == 0 or above.size == 0:
        return 0.0
    return float(above.min() - at_or_below.max())


def rolling_shutter_time_shift(raster_time: np.ndarray) -> float:
    """SplatAD's lidar rolling-shutter centring shift: (max + min) / 2 of the raster times.

    splatad.py (``time_to_center_adjustment``) subtracts it from every ray time before building
    points, over the whole raster (empty cells included), so ``pred_points`` holds t − shift.

    Args:
        raster_time: Raster time channel, any shape.

    Returns:
        Shift (s).
    """
    return float((np.max(raster_time) + np.min(raster_time)) / 2.0)


def align_gated_rays(
    ray_drop_probability: np.ndarray,
    ray_intensity: np.ndarray,
    ray_time: np.ndarray,
    valid_ray_mask: np.ndarray,
    gated_points: np.ndarray,
    time_shift: float,
) -> np.ndarray:
    """Find the raster ray of every ``pred_points`` row.

    Rows of ``pred_points`` are the kept rays in flattened raster order. A ray can have been kept only
    if it is valid and its stored probability is <= 0.5 + the stored resolution at the gate
    (``resolution_at_gate``). A two-pointer walk over those candidates assigns the next row to a
    candidate iff both keys agree: the rendered intensity (same tensor, so bit-equal) and the
    centred time, t_ray − shift = t_row within half the smallest raster time step. Intensity alone is
    not enough: it is low precision and repeats. O(R).

    Args:
        ray_drop_probability: (R,) flattened ``pred_ray_drop_prob``.
        ray_intensity: (R,) flattened ``pred_intensity``.
        ray_time: (R,) flattened raster time (before the centring shift).
        valid_ray_mask: (R,) bool, ray exists (raster depth > 0).
        gated_points: (M, 5) ``pred_points``.
        time_shift: ``rolling_shutter_time_shift`` of the raster.

    Returns:
        (M,) raster indices of the rows.

    Raises:
        ValueError: If the walk cannot assign every row (the arrays do not come from one render).
    """
    # Step 1: rays that may have passed the gate.
    gate_resolution = resolution_at_gate(ray_drop_probability[valid_ray_mask], RAY_DROP_GATE)
    candidate_rays = np.flatnonzero(valid_ray_mask & (ray_drop_probability <= RAY_DROP_GATE + gate_resolution))
    row_count = gated_points.shape[0]
    row_times = gated_points[:, PRED_POINT_TIME_COLUMN].astype(np.float64)
    # Step 2: half the smallest raster time step: closer than that, two ray times cannot be confused.
    time_tolerance = smallest_positive_step(ray_time[valid_ray_mask]) / 2.0
    # Step 3: two-pointer walk.
    row_rays = np.empty(row_count, dtype=np.int64)
    row_index = 0
    for candidate_ray in candidate_rays:
        if row_index == row_count:
            break
        keys_match = (
            ray_intensity[candidate_ray] == gated_points[row_index, PRED_POINT_INTENSITY_COLUMN]
            and abs(ray_time[candidate_ray] - time_shift - row_times[row_index]) <= time_tolerance
        )
        if keys_match:
            row_rays[row_index] = candidate_ray
            row_index += 1
    if row_index != row_count:
        raise ValueError(f"assigned {row_index} of {row_count} pred_points rows to raster rays")
    return row_rays


def fit_sweep_velocity(gated_points: np.ndarray, row_base_points: np.ndarray) -> np.ndarray:
    """Least-squares local linear velocity v from residual = pred_xyz − depth·direction = v · t_c.

    Per axis, minimising Σ (r − v t)² gives v = Σ t r / Σ t² (a line through the origin).

    Args:
        gated_points: (M, 5) ``pred_points`` (xyz, intensity, centred time t_c).
        row_base_points: (M, 3) depth · direction of each row's ray.

    Returns:
        (3,) velocity in the lidar frame (m/s); zeros if every t_c is 0 (no motion term is observable).
    """
    centred_times = gated_points[:, PRED_POINT_TIME_COLUMN].astype(np.float64)
    residuals = gated_points[:, XYZ_COLUMNS].astype(np.float64) - row_base_points
    time_square_sum = float(np.sum(centred_times ** 2))
    if time_square_sum == 0.0:
        return np.zeros(3)
    return (centred_times[:, None] * residuals).sum(axis=0) / time_square_sum


@transform("T67", "Rendered lidar raster -> ungated points: accumulation >= threshold, depth·dir + v·t_c")
def ungated_points_from_raster(
    raster_pts: np.ndarray,
    pred_depth: np.ndarray,
    pred_accumulation: np.ndarray,
    pred_ray_drop_prob: np.ndarray,
    pred_intensity: np.ndarray,
    pred_points: np.ndarray,
    accumulation_threshold: float,
) -> np.ndarray:
    """Points of every valid ray whose rendered accumulation reaches the threshold (PLAN §5 P1).

    p = depth · d(az, el) + v · (t − shift), the same formula SplatAD uses for ``pred_points``, with v
    and shift recovered from the gated rows (``align_gated_rays``, ``fit_sweep_velocity``).

    Args:
        raster_pts: [C, H, W, 5] GT ray raster (azimuth deg, elevation deg, depth, t, intensity).
        pred_depth: [C, H, W, 1] rendered expected depth.
        pred_accumulation: [C, H, W, 1] Σ w_i per ray.
        pred_ray_drop_prob: [C, H, W, 1] rendered ray-drop probability.
        pred_intensity: [C, H, W, 1] rendered intensity.
        pred_points: [M, 5] the gated points of the same render.
        accumulation_threshold: Minimum accumulation for a ray to give a point.

    Returns:
        (K, 3) float64 points in the lidar frame, in flattened raster order.
    """
    # Step 1: flatten the raster; a ray exists iff its GT depth > 0 (SplatAD's "valid").
    flat_raster = raster_pts.reshape(-1, raster_pts.shape[-1]).astype(np.float64)
    flat_depth = pred_depth.reshape(-1).astype(np.float64)
    valid_ray_mask = flat_raster[:, RASTER_DEPTH_COLUMN] > 0
    directions = ray_directions(flat_raster[:, RASTER_AZIMUTH_COLUMN], flat_raster[:, RASTER_ELEVATION_COLUMN])
    base_points = flat_depth[:, None] * directions
    ray_time = flat_raster[:, RASTER_TIME_COLUMN]
    # Step 2: motion term from the gated rows (SplatAD's shift, fitted velocity).
    time_shift = rolling_shutter_time_shift(ray_time)
    row_rays = align_gated_rays(
        pred_ray_drop_prob.reshape(-1), pred_intensity.reshape(-1), ray_time, valid_ray_mask, pred_points, time_shift
    )
    velocity = fit_sweep_velocity(pred_points, base_points[row_rays]) if row_rays.size else np.zeros(3)
    # Step 3: the rebuilt gated rows must reproduce pred_points, else the alignment is wrong.
    rebuilt_rows = base_points[row_rays] + velocity * (ray_time[row_rays] - time_shift)[:, None]
    stored_rows = pred_points[:, XYZ_COLUMNS]
    if row_rays.size:
        rebuild_tolerance = REBUILD_ROUNDING_COUNT * float(np.spacing(np.float32(np.abs(stored_rows).max())))
        rebuild_error = float(np.abs(rebuilt_rows - stored_rows).max())
        if rebuild_error > rebuild_tolerance:
            raise ValueError(f"rebuilt pred_points differ by {rebuild_error:.3g} m (> {rebuild_tolerance:.3g} m)")
    # Step 4: ungated selection and the full point formula.
    ungated_mask = valid_ray_mask & (pred_accumulation.reshape(-1) >= accumulation_threshold)
    return base_points[ungated_mask] + velocity * (ray_time[ungated_mask] - time_shift)[:, None]


@transform("T68", "Lidar-frame points -> map frame (lidar_to_map) -> voxel indices on the GT occupancy grid")
def points_to_map_voxels(
    points_lidar: np.ndarray,
    lidar_to_map: np.ndarray,
    grid_origin: np.ndarray,
    voxel_size: float,
    grid_shape: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move points to the map frame and index them on the GT grid.

    p_map = R p + t from the 4x4 pose; voxel index = ⌊(p_map − origin) / voxel_size⌋, the inverse of
    the grid's centre rule origin + (index + 0.5) · voxel_size.

    Args:
        points_lidar: (N, 3) points in the lidar frame.
        lidar_to_map: (4, 4) pose.
        grid_origin: (3,) map-frame minimum corner of the grid.
        voxel_size: Voxel edge (m).
        grid_shape: (nx, ny, nz).

    Returns:
        ((N, 3) map-frame points, (N, 3) int64 voxel indices, (N,) bool inside-grid mask).
    """
    # Step 1: rigid transform in row-vector form: p @ R^T + t.
    pose = np.asarray(lidar_to_map, dtype=np.float64)
    points_map = np.asarray(points_lidar, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
    # Step 2: voxel indices and the in-grid test (0 <= index < shape on every axis).
    voxel_indices = np.floor((points_map - np.asarray(grid_origin, dtype=np.float64)) / voxel_size).astype(np.int64)
    inside_grid = np.all((voxel_indices >= 0) & (voxel_indices < np.asarray(grid_shape)), axis=1)
    return points_map, voxel_indices, inside_grid


def occupied_voxel_grid(voxel_indices: np.ndarray, inside_grid: np.ndarray, grid_shape: Sequence[int]) -> np.ndarray:
    """Boolean grid with every voxel holding at least one in-grid point set.

    Args:
        voxel_indices: (N, 3) T68 indices.
        inside_grid: (N,) T68 mask.
        grid_shape: (nx, ny, nz).

    Returns:
        Bool array of ``grid_shape``.
    """
    occupied = np.zeros(tuple(grid_shape), dtype=bool)
    kept_indices = voxel_indices[inside_grid]
    occupied[kept_indices[:, 0], kept_indices[:, 1], kept_indices[:, 2]] = True
    return occupied


def load_rendered_sweep(npz_path: Path, accumulation_threshold: float) -> RenderedSweep:
    """Read one rendered sweep and build its gated, ungated and real point sets.

    Args:
        npz_path: ``<render dir>/<idx>_<ts_ns>.npz``.
        accumulation_threshold: T67 threshold.

    Returns:
        RenderedSweep (``ungated_points`` None without ``pred_accumulation``).
    """
    with np.load(npz_path) as arrays:
        gated_rows = arrays["pred_points"]
        ungated_points = None
        if "pred_accumulation" in arrays.files:
            ungated_points = ungated_points_from_raster(
                arrays["raster_pts"], arrays["pred_depth"], arrays["pred_accumulation"],
                arrays["pred_ray_drop_prob"], arrays["pred_intensity"], gated_rows, accumulation_threshold,
            )
        return RenderedSweep(
            stem=Path(npz_path).stem,
            lidar_to_map=np.asarray(arrays["lidar_to_map"], dtype=np.float64),
            gated_points=gated_rows[:, XYZ_COLUMNS].astype(np.float64),
            ungated_points=ungated_points,
            real_points=arrays["gt_points"][:, XYZ_COLUMNS].astype(np.float64),
        )


def load_test_lidar_sweeps(
    render_dir: Path, expected_stems: Sequence[str], accumulation_threshold: float
) -> List[RenderedSweep]:
    """Load every rendered sweep of a render folder, checked against the expected frames.

    Args:
        render_dir: Folder of the lidar npz files (``.../renders/test/lidar``).
        expected_stems: ``<idx>_<ts_ns>`` stems the split (or render selection) says were rendered.
        accumulation_threshold: T67 threshold.

    Returns:
        Sweeps in stem order.

    Raises:
        ValueError: If the folder's stems differ from the expected ones (missing or extra files).
    """
    found_stems = sorted(npz_path.stem for npz_path in Path(render_dir).glob(f"*{LIDAR_NPZ_SUFFIX}"))
    missing_stems = sorted(set(expected_stems) - set(found_stems))
    unexpected_stems = sorted(set(found_stems) - set(expected_stems))
    if missing_stems or unexpected_stems:
        raise ValueError(
            f"{render_dir}: renders differ from the expected frames "
            f"(missing {missing_stems[:5]}, unexpected {unexpected_stems[:5]})"
        )
    return [
        load_rendered_sweep(Path(render_dir) / f"{stem}{LIDAR_NPZ_SUFFIX}", accumulation_threshold)
        for stem in found_stems
    ]
