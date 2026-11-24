"""Per-run P2 inputs on the shared scored cells: recorded evidence, P1 hits and the Gaussian factors.

P2 (PLAN §5) scores T81 peak-alpha occupancy from a run's Gaussians on the voxel cells of the scored
neighbourhoods. This module builds what scripts/19 needs beside that occupancy, all on the same
cells: the recorded-beam evidence of the test sweeps (T78), the cells the run's ungated rendered
points hit (P1, T68), and the factorized checkpoint Gaussians (T56), which need torch and so are
exported inside the reference image.
"""

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from viking_splatad.eval.hidden_geometry import recorded_ray_evidence
from viking_splatad.eval.rendered_pointcloud import RenderedSweep, points_to_map_voxels, ray_directions

# Rendered lidar npz keys (scripts/08) holding the recorded scan: per ray (azimuth deg, elevation deg,
# depth, ...) and whether the beam returned. Layout of raster_pts: 5 values per ray.
RASTER_POINTS_KEY = "raster_pts"
RASTER_DID_RETURN_KEY = "raster_did_return"
LIDAR_TO_MAP_KEY = "lidar_to_map"
RASTER_VALUES_PER_RAY = 5
RASTER_AZIMUTH_COLUMN, RASTER_ELEVATION_COLUMN, RASTER_DEPTH_COLUMN = 0, 1, 2
# T78 evidence columns: seen, blocked, no_return, passed_through (hidden_geometry.recorded_ray_evidence).
EVIDENCE_COLUMN_COUNT = 4
# Factorized archive keys (gaussian_occupancy.save_gaussian_factors).
FACTOR_KEYS = ("means", "principal_axes", "axis_std", "opacities", "checkpoint_step")
# Mount points inside the reference image; the repo path matches scripts/test_in_image.sh.
IMAGE_REPO_MOUNT = "/workspace/viking-splatad"
IMAGE_FACTORS_MOUNT = "/factors"
EXPORT_SCRIPT_RELATIVE_PATH = Path("scripts") / "10_export_gaussians.py"
# Per-class key prefixes of the cells archive (scripts/19 writes neighbourhood_<class> and truth_<class>).
NEIGHBOURHOOD_KEY_PREFIX = "neighbourhood_"
TRUTH_KEY_PREFIX = "truth_"


def accumulate_recorded_evidence(render_dir: Path, stems: Sequence[str], centres: np.ndarray, voxel_size: float) -> np.ndarray:
    """OR the T78 recorded-beam evidence of every test sweep over the scored cells.

    Per sweep, the recorded rays are rotated into map-grid axes and the cells are taken relative to
    the sweep origin; rotating the cell centres alone would rotate the voxel boxes (scripts/17).

    Args:
        render_dir: ``renders/test/lidar`` of the run (one ``<stem>.npz`` per test sweep).
        stems: Test sweep stems to use.
        centres: (V, 3) map-frame voxel centres.
        voxel_size: Voxel edge in metres.

    Returns:
        (V, 4) bool evidence flags (seen, blocked, no_return, passed_through), OR-ed over sweeps.
    """
    evidence = np.zeros((len(centres), EVIDENCE_COLUMN_COUNT), dtype=bool)
    for stem in stems:
        with np.load(Path(render_dir) / f"{stem}.npz", allow_pickle=False) as sweep_arrays:
            # Step 1: the recorded raster, one row per ray; rays without a finite positive depth carry no geometry.
            raster = sweep_arrays[RASTER_POINTS_KEY].reshape(-1, RASTER_VALUES_PER_RAY)
            return_mask = sweep_arrays[RASTER_DID_RETURN_KEY].reshape(-1)
            lidar_to_map = sweep_arrays[LIDAR_TO_MAP_KEY]
        valid_rays = np.isfinite(raster).all(axis=1) & (raster[:, RASTER_DEPTH_COLUMN] > 0)
        # Step 2: ray directions in the map axes (R · d); renormalised against float rounding of R.
        map_directions = ray_directions(raster[valid_rays, RASTER_AZIMUTH_COLUMN],
                                        raster[valid_rays, RASTER_ELEVATION_COLUMN]) @ lidar_to_map[:3, :3].T
        map_directions /= np.linalg.norm(map_directions, axis=1, keepdims=True)
        # Step 3: cells relative to this sweep's origin; T78 flags, OR-ed so any sweep's evidence counts.
        evidence |= recorded_ray_evidence(centres - lidar_to_map[:3, 3], voxel_size, map_directions,
                                          raster[valid_rays, RASTER_DEPTH_COLUMN], return_mask[valid_rays])
    return evidence


def p1_hit_cells(sweeps: Sequence[RenderedSweep], cell_indices: np.ndarray, grid_origin: np.ndarray,
                 voxel_size: float, grid_shape: Sequence[int]) -> np.ndarray:
    """Cells holding at least one ungated rendered P1 point of any sweep (T68 voxelisation).

    Args:
        sweeps: The run's rendered test sweeps (ungated points required).
        cell_indices: (V, 3) integer grid indices of the scored cells.
        grid_origin: GT grid origin. voxel_size: Voxel edge. grid_shape: GT grid shape.

    Returns:
        (V,) bool.

    Raises:
        ValueError: If a sweep has no ungated points (binding P1 cannot be rebuilt).
    """
    # Cells and points are compared by flat grid index (row-major ravel), so membership is one isin.
    wanted_keys = np.ravel_multi_index(cell_indices.T, grid_shape)
    hit = np.zeros(len(cell_indices), dtype=bool)
    for sweep in sweeps:
        if sweep.ungated_points is None:
            raise ValueError(f"{sweep.stem}: no ungated points; binding P1 cannot be rebuilt")
        _, point_indices, inside_grid = points_to_map_voxels(sweep.ungated_points, sweep.lidar_to_map,
                                                             grid_origin, voxel_size, grid_shape)
        hit |= np.isin(wanted_keys, np.ravel_multi_index(point_indices[inside_grid].T, grid_shape))
    return hit


def factor_export_command(project_root: Path, train_dir: Path, factors_dir: Path, image_name: str,
                          archive_name: str) -> List[str]:
    """``docker run`` command that writes a run's factorized Gaussians (T56) inside the reference image.

    torch (needed to read the checkpoint) exists only in the image (CLAUDE.md rules 9-10). The repo is
    mounted read-only and only ``factors_dir`` is writable, so the run folder cannot change.

    Args:
        project_root: Repo root on the host. train_dir: The run's trainer directory (inside the repo).
        factors_dir: Host folder for the archive. image_name: Reference image tag.
        archive_name: File name of the archive inside ``factors_dir``.

    Returns:
        The command, program first.

    Raises:
        ValueError: If ``train_dir`` is outside the repo (it would not be visible in the container).
    """
    project_root, train_dir = Path(project_root).resolve(), Path(train_dir).resolve()
    if not train_dir.is_relative_to(project_root):
        raise ValueError(f"{train_dir} is outside the repo; the image only sees the repo")
    return ["docker", "run", "--rm", "--platform", "linux/amd64", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-e", f"PYTHONPATH={IMAGE_REPO_MOUNT}/src",
            "-v", f"{project_root}:{IMAGE_REPO_MOUNT}:ro", "-v", f"{Path(factors_dir).resolve()}:{IMAGE_FACTORS_MOUNT}",
            "-w", IMAGE_REPO_MOUNT, image_name, "python", str(EXPORT_SCRIPT_RELATIVE_PATH),
            "--train-dir", str(train_dir.relative_to(project_root)),
            "--output-path", f"{IMAGE_FACTORS_MOUNT}/{archive_name}", "--factorized"]


def load_factors(archive_path: Path) -> Dict[str, np.ndarray]:
    """Read a factorized Gaussian archive into memory.

    Args:
        archive_path: ``save_gaussian_factors`` output.

    Returns:
        FACTOR_KEYS -> arrays (checkpoint_step as a 0-d array).
    """
    with np.load(archive_path, allow_pickle=False) as archive:
        return {key: archive[key] for key in FACTOR_KEYS}


def p2_cell_outcomes(cells: Mapping[str, np.ndarray], class_names: Sequence[str]) -> Dict[str, np.ndarray]:
    """Classify the scored cells of a P2 cells archive with the rule the P2 metrics count.

    scripts/19 scores class C on its neighbourhood cells N_C with occupied = occupancy >= threshold
    and truth T_C (GT of C inside N_C), so P2 precision_C = |occupied & T_C| / |occupied & N_C|.
    Over several classes the picture takes the union: a cell is scored when it lies in any N_C and
    on ground truth when it lies in any T_C & N_C (an occupied cell on GT for one class and off it
    for another counts as on GT, as scripts/15's P1 map does). For one class, occupied & scored &
    on_ground_truth are exactly the P2 true positives and occupied & scored & ~on_ground_truth the
    false positives.

    Args:
        cells: Arrays of ``gaussian_occupancy_cells.npz``: ``occupancy`` (V,), ``threshold`` scalar,
            and per class ``neighbourhood_<C>`` / ``truth_<C>`` (V,) bool.
        class_names: Scored class names to include (at least one).

    Returns:
        {"occupied", "scored", "on_ground_truth"}: (V,) bool each.

    Raises:
        ValueError: If no class is given.
        KeyError: If a class has no neighbourhood/truth arrays in ``cells``.
    """
    if not class_names:
        raise ValueError("p2_cell_outcomes needs at least one class name")
    # Step 1: T81 occupancy against the archived binding threshold (the same >= cut scripts/19 applies).
    occupied = np.asarray(cells["occupancy"]) >= float(cells["threshold"])
    scored = np.zeros(occupied.shape, dtype=bool)
    on_ground_truth = np.zeros(occupied.shape, dtype=bool)
    # Step 2: union of the class neighbourhoods and of each class's GT within its own neighbourhood.
    for class_name in class_names:
        class_neighbourhood = np.asarray(cells[f"{NEIGHBOURHOOD_KEY_PREFIX}{class_name}"], dtype=bool)
        scored |= class_neighbourhood
        on_ground_truth |= np.asarray(cells[f"{TRUTH_KEY_PREFIX}{class_name}"], dtype=bool) & class_neighbourhood
    return {"occupied": occupied, "scored": scored, "on_ground_truth": on_ground_truth}


# Archive key of the session-wide hidden flag H per cell (hidden-voxel evaluation).
HIDDEN_KEY = "hidden"
# The near-path hidden flag (H_near = H within the near band of the driven path), the headline mask.
HIDDEN_NEAR_KEY = "hidden_near"
# Hidden-band choices of hidden_cell_outcomes -> the archive key holding that band's hidden flag.
HIDDEN_BAND_NEAR, HIDDEN_BAND_FULL = "near", "full"
HIDDEN_KEY_BY_BAND = {HIDDEN_BAND_NEAR: HIDDEN_NEAR_KEY, HIDDEN_BAND_FULL: HIDDEN_KEY}
# Archive key of the P2 thresholds the hidden metric was evaluated at.
P2_THRESHOLDS_KEY = "p2_thresholds"


def hidden_cell_outcomes(cells: Mapping[str, np.ndarray], class_name: str, threshold: float,
                         band: str = HIDDEN_BAND_NEAR) -> Dict[str, np.ndarray]:
    """Classify the scored cells of one class (or group) on the hidden space with the P2 hidden metric's sets.

    With O = occupancy >= threshold (same inclusive cut as ``p2_cell_outcomes``), G = ``truth_<name>``
    (GT inside the neighbourhood), N = ``neighbourhood_<name>`` and H = the band's hidden flag (``hidden_near`` for band "near", ``hidden`` for "full"):
        recall_H    = |O & G & H| / |G & H|
        precision_H = |O & G & H| / |O & N & H|
    The returned sets are exactly the terms of those ratios:
        context     = H & N              (every scored hidden cell of the class)
        hit         = O & G & H          (true positives of both ratios)
        false_alarm = O & N & H & ~G     (false positives of precision_H)
        missed      = G & H & ~O         (false negatives of recall_H)
    so recall_H = |hit| / (|hit| + |missed|) and precision_H = |hit| / (|hit| + |false_alarm|).

    Args:
        cells: Arrays of ``gaussian_occupancy_cells.npz``: ``occupancy``, the band's hidden flag
            (``hidden_near`` or ``hidden``), ``p2_thresholds``
            and ``neighbourhood_<name>`` / ``truth_<name>`` (V,) bool (name = a class or a group such as ``obstacles``).
        class_name: Class or group name.
        threshold: Occupancy cut; must be one of the archive's ``p2_thresholds``.
        band: "near" (H_near, default) or "full" (H).

    Returns:
        {"context", "hit", "false_alarm", "missed"}: (V,) bool each.

    Raises:
        KeyError: If the archive lacks the band's hidden key (``hidden_near`` / ``hidden``), ``p2_thresholds`` or the name's arrays (it predates the hidden evaluation).
        ValueError: If ``threshold`` is not an evaluated threshold of the archive, or ``band`` is unknown.
    """
    # Step 0: the band picks the hidden flag.
    if band not in HIDDEN_KEY_BY_BAND:
        raise ValueError(f"band {band!r} is not one of {sorted(HIDDEN_KEY_BY_BAND)}")
    hidden_key = HIDDEN_KEY_BY_BAND[band]
    # Step 1: every required key must exist; an old archive is reported by key name.
    required_keys = (hidden_key, P2_THRESHOLDS_KEY, f"{NEIGHBOURHOOD_KEY_PREFIX}{class_name}", f"{TRUTH_KEY_PREFIX}{class_name}")
    for key in required_keys:
        if key not in cells:
            raise KeyError(f"cells archive has no '{key}': it predates the hidden evaluation (re-run scripts/19)")
    # Step 2: pictures exist only for thresholds the metric was evaluated at.
    evaluated_thresholds = np.asarray(cells[P2_THRESHOLDS_KEY], dtype=float).reshape(-1)
    matching_thresholds = evaluated_thresholds[np.isclose(evaluated_thresholds, float(threshold))]
    if not len(matching_thresholds):
        raise ValueError(f"threshold {threshold} was not evaluated; archive has {evaluated_thresholds.tolist()}")
    # Step 3: the set algebra of the docstring, cut at the archived value scripts/19 scored with (not the
    # caller's float, which may differ by rounding and would move cells sitting exactly on the cut).
    occupied = np.asarray(cells["occupancy"]) >= matching_thresholds[0]
    hidden = np.asarray(cells[hidden_key], dtype=bool)
    neighbourhood = np.asarray(cells[f"{NEIGHBOURHOOD_KEY_PREFIX}{class_name}"], dtype=bool)
    truth = np.asarray(cells[f"{TRUTH_KEY_PREFIX}{class_name}"], dtype=bool)
    return {"context": hidden & neighbourhood,
            "hit": occupied & truth & hidden,
            "false_alarm": occupied & neighbourhood & hidden & ~truth,
            "missed": truth & hidden & ~occupied}
