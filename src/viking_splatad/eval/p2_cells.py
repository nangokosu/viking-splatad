"""Per-run P2 inputs on the shared scored cells: P1 hits and the Gaussian factors.

P2 (PLAN §5) scores T81 peak-alpha occupancy from a run's Gaussians on the voxel cells of the scored
neighbourhoods. This module builds what scripts/19 needs beside that occupancy, all on the same
cells: the cells the run's ungated rendered points hit (P1, T68), and the factorized checkpoint
Gaussians (T56), which need torch and so are exported inside the reference image. It also
classifies the archived cells for the P2 precision picture (``p2_cell_outcomes``).
"""

from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np

from viking_splatad.eval.rendered_pointcloud import RenderedSweep, points_to_map_voxels

# Factorized archive keys (gaussian_occupancy.save_gaussian_factors).
FACTOR_KEYS = ("means", "principal_axes", "axis_std", "opacities", "checkpoint_step")
# Mount points inside the reference image; the repo path matches scripts/test_in_image.sh.
IMAGE_REPO_MOUNT = "/workspace/viking-splatad"
IMAGE_FACTORS_MOUNT = "/factors"
EXPORT_SCRIPT_RELATIVE_PATH = Path("scripts") / "10_export_gaussians.py"
# Per-class key prefixes of the cells archive (scripts/19 writes neighbourhood_<class> and truth_<class>).
NEIGHBOURHOOD_KEY_PREFIX = "neighbourhood_"
TRUTH_KEY_PREFIX = "truth_"


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


# Archive key of the P2 thresholds the metrics were evaluated at.
P2_THRESHOLDS_KEY = "p2_thresholds"
