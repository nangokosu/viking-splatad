"""Ground-truth extraction: labels JSON -> cuboids.json, occupancy.npz, summary.md.

Ground truth is written to ``session.gt_dir`` and is deliberately kept apart from the
SplatAD inputs (it is used only for occupancy evaluation).

Usage:
    uv run python scripts/05_extract_cuboids.py --config configs/sessions/<session>.yaml
        [--voxel-size METERS] [--max-grid-mb MEGABYTES]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from viking_splatad.labels.cuboids import Cuboid, compute_cuboid_geometry, parse_cuboids
from viking_splatad.labels.occupancy import (
    EMPTY_CLASS_ID,
    OccupancyGrid,
    compute_default_bounds,
    voxelize_cuboids,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Class id whose box width is the "trunk diameter" reference (Viking Hill protocol: 1 = tree_trunk).
TRUNK_CLASS_ID = 1

# Default voxel size = this fraction of the median trunk diameter. 0.5 puts two voxels across
# a typical trunk (Nyquist-style minimum to resolve it at all) while keeping the grid small.
TRUNK_DIAMETER_VOXEL_FRACTION = 0.5

# Default memory budget in MiB for the voxelisation peak. 2048 MiB is a quarter of a 8 GiB
# machine, leaving ample room for the OS and the JSON/NumPy overhead on the Mac.
DEFAULT_MAX_GRID_MB = 2048.0

# Bytes bound for one MiB.
BYTES_PER_MB = 1024.0 * 1024.0

# voxelize_cuboids keeps an int32 class grid and a float64 rank grid alive together, so the peak
# per voxel is the sum of their item sizes; the saved grid is the much smaller uint8 copy.
VOXELIZE_PEAK_BYTES_PER_VOXEL = np.dtype(np.int32).itemsize + np.dtype(np.float64).itemsize

# dtype of the saved class-id grid (class ids must fit; checked below).
SAVED_CLASS_DTYPE = np.uint8


def load_session(config_path: Path) -> Dict[str, Any]:
    """Load the session config via the project loader, or a yaml fallback.

    Args:
        config_path: Path of the session YAML.

    Returns:
        Nested dict whose ``session.*`` path entries are absolute ``Path`` objects.
    """
    try:
        from viking_splatad.config import load_session_config
    except ImportError:
        # Fallback: plain YAML, resolving relative session paths against the project root.
        import yaml

        config = yaml.safe_load(Path(config_path).read_text())
        for key, value in config["session"].items():
            if key.endswith(("_dir", "_json")):
                config["session"][key] = (PROJECT_ROOT / value).resolve()
        return config
    return load_session_config(config_path)


def build_class_names(document: Dict[str, Any]) -> Dict[int, str]:
    """Build the id -> lower-cased name table from the file's own categories.

    Args:
        document: Parsed labels JSON.

    Returns:
        Mapping class id -> lower-case name.
    """
    categories = document["dataset"]["task_attributes"]["categories"]
    return {int(category["id"]): category["name"].lower() for category in categories}


def derive_default_voxel_size(cuboids: Sequence[Cuboid]) -> float:
    """Voxel size = TRUNK_DIAMETER_VOXEL_FRACTION x median trunk diameter.

    The trunk diameter of a box is the mean of its two horizontal full sizes (x, y).

    Args:
        cuboids: All parsed boxes.

    Returns:
        Voxel edge length in meters.

    Raises:
        ValueError: If there is no trunk box to derive from.
    """
    # Horizontal mean size per trunk box.
    diameters = [
        0.5 * (box.dimensions_xyz[0] + box.dimensions_xyz[1])
        for box in cuboids
        if box.class_id == TRUNK_CLASS_ID
    ]
    if not diameters:
        raise ValueError("no trunk boxes to derive a voxel size from; pass --voxel-size")
    return TRUNK_DIAMETER_VOXEL_FRACTION * float(np.median(diameters))


def plan_grid(cuboids: Sequence[Cuboid], voxel_size: float, max_grid_mb: float) -> np.ndarray:
    """Compute the grid shape from the box bounds and enforce the memory budget.

    Mirrors voxelize_cuboids' default bounds (union AABB + one voxel margin) and shape
    ceil((max - min) / voxel_size). Nothing is allocated here.

    Args:
        cuboids: Non-empty boxes.
        voxel_size: Voxel edge length (meters).
        max_grid_mb: Peak-memory budget (MiB).

    Returns:
        (3,) integer grid shape.

    Raises:
        MemoryError: If the voxelisation peak would exceed the budget.
    """
    bounds_min, bounds_max = compute_default_bounds(cuboids, voxel_size)
    grid_shape = np.ceil((bounds_max - bounds_min) / voxel_size).astype(np.int64)
    voxel_count = int(np.prod(grid_shape))
    peak_mb = voxel_count * VOXELIZE_PEAK_BYTES_PER_VOXEL / BYTES_PER_MB
    saved_mb = voxel_count * np.dtype(SAVED_CLASS_DTYPE).itemsize / BYTES_PER_MB
    print(
        f"grid {tuple(int(size) for size in grid_shape)} = {voxel_count} voxels at {voxel_size:.4f} m; "
        f"peak {peak_mb:.1f} MiB, saved {saved_mb:.1f} MiB (budget {max_grid_mb:.1f} MiB)"
    )
    if peak_mb > max_grid_mb:
        raise MemoryError(
            f"voxel grid would need {peak_mb:.1f} MiB (> --max-grid-mb {max_grid_mb:.1f}); "
            f"increase --voxel-size or --max-grid-mb"
        )
    return grid_shape


def cuboid_to_json(box: Cuboid) -> Dict[str, Any]:
    """Serialise one box, including its 8 map-frame corners.

    Args:
        box: Parsed cuboid.

    Returns:
        JSON-ready dict.
    """
    corners = compute_cuboid_geometry(box).corners
    return {
        "box_id": box.box_id,
        "class_id": box.class_id,
        "class_name": box.class_name,
        "center_xyz": list(box.center_xyz),
        "dimensions_xyz": list(box.dimensions_xyz),
        "quaternion_xyzw": list(box.quaternion_xyzw),
        "corners_xyz": corners.tolist(),
    }


def build_summary(
    cuboids: Sequence[Cuboid], grid: OccupancyGrid, session_name: str, labels_path: Path
) -> str:
    """Render summary.md: counts, extents, occupied voxels, map-frame caveat.

    Args:
        cuboids: All boxes.
        grid: Occupancy grid.
        session_name: Session label.
        labels_path: Source labels file.

    Returns:
        Markdown text.
    """
    all_corners = np.concatenate([compute_cuboid_geometry(box).corners for box in cuboids])
    extent_min, extent_max = all_corners.min(axis=0), all_corners.max(axis=0)
    lines: List[str] = [
        f"# Ground-truth summary: {session_name}",
        "",
        f"- source: `{labels_path}`",
        f"- total cuboids: {len(cuboids)}",
        f"- box extent min (x, y, z): {np.round(extent_min, 3).tolist()}",
        f"- box extent max (x, y, z): {np.round(extent_max, 3).tolist()}",
        f"- extent size (m): {np.round(extent_max - extent_min, 3).tolist()}",
        f"- voxel grid shape: {list(grid.class_ids.shape)}, voxel size {grid.voxel_size:.4f} m",
        f"- grid origin: {np.round(grid.origin, 3).tolist()}",
        "",
        "| class_id | class_name | boxes | median full size (x, y, z) m | occupied voxels |",
        "|---|---|---|---|---|",
    ]
    for class_id in sorted({box.class_id for box in cuboids}):
        class_boxes = [box for box in cuboids if box.class_id == class_id]
        median_size = np.median([box.dimensions_xyz for box in class_boxes], axis=0)
        occupied = int(np.count_nonzero(grid.class_ids == class_id))
        lines.append(
            f"| {class_id} | {class_boxes[0].class_name} | {len(class_boxes)} | "
            f"{np.round(median_size, 3).tolist()} | {occupied} |"
        )
    lines += [
        "",
        f"Total occupied voxels: {int(np.count_nonzero(grid.class_ids != EMPTY_CLASS_ID))}",
        "",
        "## Map-frame caveat",
        "",
        "The label file carries NO coordinate-frame information. It is UNVERIFIED that its map "
        "frame equals the bag `/icp_odom` `map` frame; do not use these boxes for evaluation "
        "against bag-frame data until an alignment check has confirmed it.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list (default sys.argv[1:]).

    Returns:
        Process exit code (0 ok, 2 on budget/derivation errors).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="session YAML")
    parser.add_argument("--voxel-size", type=float, default=None, help="meters (default: derived)")
    parser.add_argument("--max-grid-mb", type=float, default=DEFAULT_MAX_GRID_MB, help="peak MiB budget")
    arguments = parser.parse_args(argv)

    # 1) Session config and labels document.
    session = load_session(arguments.config)["session"]
    labels_path = Path(session["labels_json"])
    document = json.loads(labels_path.read_text())
    # 2) Parse boxes using the file's own category table.
    cuboids = parse_cuboids(
        document,
        class_names=build_class_names(document),
        labelset_name="ground-truth",
        sample_key=session["labels_sample_key"],
    )
    if not cuboids:
        print("error: no cuboids parsed", file=sys.stderr)
        return 2
    if max(box.class_id for box in cuboids) > np.iinfo(SAVED_CLASS_DTYPE).max:
        print("error: class id does not fit the saved dtype", file=sys.stderr)
        return 2
    # 3) Voxel size, grid report and budget check BEFORE any allocation.
    try:
        voxel_size = arguments.voxel_size or derive_default_voxel_size(cuboids)
        plan_grid(cuboids, voxel_size, arguments.max_grid_mb)
    except (MemoryError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    # 4) Voxelise, then shrink to uint8 and drop the big temporary.
    grid = voxelize_cuboids(cuboids, voxel_size)
    grid = OccupancyGrid(grid.class_ids.astype(SAVED_CLASS_DTYPE), grid.origin, grid.voxel_size)
    # 5) Write outputs.
    gt_dir = Path(session["gt_dir"])
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / "cuboids.json").write_text(json.dumps([cuboid_to_json(box) for box in cuboids], indent=1))
    np.savez_compressed(
        gt_dir / "occupancy.npz",
        class_ids=grid.class_ids,
        origin=grid.origin,
        voxel_size=np.float64(grid.voxel_size),
    )
    summary = build_summary(cuboids, grid, session["name"], labels_path)
    (gt_dir / "summary.md").write_text(summary)
    print(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
