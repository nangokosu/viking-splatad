"""Pure-numpy pieces of the PyVista per-pose viewer (scripts/15_view_pose_pyvista.py).

Nothing here imports PyVista, so the tests run in the reference image. The viewer is display only:
GT cuboids are drawn and used to colour points, never fed to SplatAD as seeds, filters or losses
(CLAUDE.md rule 6). The inside/outside masks come from ``eval.detection`` (the metric's own code),
so the picture cannot disagree with the numbers.
"""

import itertools
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.labels.cuboids import Cuboid, compute_cuboid_geometry

# Corners of one box, in the order ``compute_cuboid_geometry`` writes them: the sign triple
# (sx, sy, sz) in {-1, +1}^3 of each corner, from itertools.product (last axis fastest).
CORNER_SIGN_TRIPLES = np.array(list(itertools.product((-1, 1), repeat=3)))
# VTK "lines" cell layout: [point_count, point_a, point_b] per edge, and a box has 8 corners.
VTK_POINTS_PER_EDGE = 2
CORNERS_PER_CUBOID = CORNER_SIGN_TRIPLES.shape[0]
# Display colours (0-255 RGB). They are a presentation choice with no data-derived value: points
# inside a scored cuboid are red, all other rendered points blue, real returns grey.
INSIDE_CUBOID_RGB = (214, 39, 40)
OUTSIDE_CUBOID_RGB = (31, 119, 180)
REAL_POINT_RGB = (160, 160, 160)
# Voxel maps of the picture (0-255 RGB and 0-1 opacity are presentation choices with no data-derived
# value). Ground-truth voxels are grey, rendered voxels light blue in the overlay, a hit is green and
# a rendered voxel off every GT voxel red. In the overlay the GT is the more opaque layer, so it
# shows through the translucent render. Wireframes are dark so they do not compete with the voxels.
GROUND_TRUTH_VOXEL_RGB = (205, 205, 205)
RENDER_VOXEL_RGB = (70, 160, 255)
HIT_VOXEL_RGB = (44, 160, 44)
NON_HIT_VOXEL_RGB = (214, 39, 40)
CUBOID_WIREFRAME_RGB = (60, 60, 60)
GROUND_TRUTH_OVERLAY_OPACITY = 0.8
SOLID_VOXEL_OPACITY = 1.0
# Red (non-hit) render voxels of the precision map are translucent so green hits stay visible behind them.
NON_HIT_PRECISION_OPACITY = 0.3
RENDER_OVERLAY_OPACITY = 0.55
# The three maps drawn for every pose, in file-name order: <stem>_<map name>.png.
MAP_NAMES = ("overlay", "recall", "precision")
# Script that renders the pictures, and the isolated project (viz/) that has PyVista installed.
VIEWER_SCRIPT_RELATIVE_PATH = Path("scripts") / "15_view_pose_pyvista.py"
VIEWER_PROJECT_RELATIVE_DIR = Path("viz")


def cuboid_edge_index_pairs() -> np.ndarray:
    """The 12 edges of a box as pairs of corner indices.

    Two corners share an edge exactly when their sign triples differ in one position (Hamming
    distance 1). A cube has 8 corners of degree 3, so 8 * 3 / 2 = 12 edges.

    Returns:
        (12, 2) int array, each pair (lower index, higher index), in index order.
    """
    edge_pairs = []
    # Step 1: compare every corner pair once (lower index first).
    for first_corner, second_corner in itertools.combinations(range(CORNERS_PER_CUBOID), 2):
        # Step 2: count the axes on which the two corners lie on opposite faces.
        differing_axis_count = int(np.count_nonzero(
            CORNER_SIGN_TRIPLES[first_corner] != CORNER_SIGN_TRIPLES[second_corner]))
        if differing_axis_count == 1:
            edge_pairs.append((first_corner, second_corner))
    return np.array(edge_pairs, dtype=np.int64)


def cuboid_wireframe_arrays(cuboids: Sequence[Cuboid]) -> Tuple[np.ndarray, np.ndarray]:
    """Corner points and VTK line cells that draw the cuboids as wireframes.

    Args:
        cuboids: Boxes in the map frame.

    Returns:
        (corner points (8B, 3) float64 with box k at rows 8k..8k+7, VTK ``lines`` array of length
        3 * 12 * B laid out as [2, a, b, 2, a, b, ...]). Both are empty for no boxes.
    """
    if not cuboids:
        return np.zeros((0, 3)), np.zeros(0, dtype=np.int64)
    edge_pairs = cuboid_edge_index_pairs()
    corner_blocks, line_blocks = [], []
    for cuboid_index, cuboid in enumerate(cuboids):
        # Step 1: this box's 8 map-frame corners (T30).
        corner_blocks.append(compute_cuboid_geometry(cuboid).corners)
        # Step 2: its 12 edges, shifted to this box's slice of the stacked corner array.
        shifted_pairs = edge_pairs + cuboid_index * CORNERS_PER_CUBOID
        point_count_column = np.full((shifted_pairs.shape[0], 1), VTK_POINTS_PER_EDGE, dtype=np.int64)
        line_blocks.append(np.hstack([point_count_column, shifted_pairs]).reshape(-1))
    return np.vstack(corner_blocks), np.concatenate(line_blocks)


def inside_outside_colors(
    inside_mask: np.ndarray,
    inside_rgb: Tuple[int, int, int] = INSIDE_CUBOID_RGB,
    outside_rgb: Tuple[int, int, int] = OUTSIDE_CUBOID_RGB,
) -> np.ndarray:
    """Per-point RGB: one colour for points inside a scored cuboid, another for the rest.

    Args:
        inside_mask: (N,) bool, e.g. ``eval.detection.points_in_any_cuboid``.
        inside_rgb: 0-255 RGB of inside points.
        outside_rgb: 0-255 RGB of all other points.

    Returns:
        (N, 3) uint8.
    """
    # Broadcast the mask over the three colour channels: pick inside_rgb where True, else outside_rgb.
    return np.where(np.asarray(inside_mask, dtype=bool)[:, None], np.array(inside_rgb), np.array(outside_rgb)).astype(np.uint8)


def viz_command(
    run_dirs: Sequence[Path],
    output_dir: Path,
    eval_config_path: Path,
    accumulation_threshold: float,
    rerender_label: Optional[str] = None,
    radius_meters: Optional[float] = None,
    class_names: Optional[Sequence[str]] = None,
) -> List[str]:
    """Argument list that renders pictures for runs, in the isolated viz/ environment.

    Script 13 runs in the root environment, which has no PyVista (adding it to the root pyproject
    would change a recipe file and force a cluster rebuild), so it launches the viewer with
    ``uv run --project viz`` as a subprocess.

    Args:
        run_dirs: Fetched run folders to draw (several = side-by-side panels).
        output_dir: Folder for the PNGs.
        eval_config_path: configs/eval.yaml the evaluation used.
        accumulation_threshold: T67 threshold the evaluation used (so the points match).
        rerender_label: ``rerenders/<label>`` of the (single) run, or None.
        radius_meters: Horizontal radius around each sweep to draw, or None for the config value.
        class_names: Scored class names to draw, or None for the config value (all scored classes).

    Returns:
        The command, program first.
    """
    command = ["uv", "run", "--project", str(VIEWER_PROJECT_RELATIVE_DIR), "python", str(VIEWER_SCRIPT_RELATIVE_PATH)]
    for run_dir in run_dirs:
        command += ["--run", str(run_dir)]
    command += ["--output-dir", str(output_dir), "--eval-config", str(eval_config_path),
                "--accumulation-threshold", repr(float(accumulation_threshold))]
    if rerender_label is not None:
        command += ["--rerender", rerender_label]
    if radius_meters is not None:
        command += ["--radius-meters", repr(float(radius_meters))]
    if class_names is not None:
        command += ["--classes", ",".join(class_names)]
    return command
