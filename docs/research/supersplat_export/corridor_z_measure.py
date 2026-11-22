"""Measure robust z bounds for the corridor crop: lidar z tails inside the corridor vs the Gaussians.

The corridor (T87, production ``train_lidar_corridor_map``) bounds z by the raw min/max of the train lidar
returns inside it, which single stray returns set (May 30k: -76 / +72 m). This script reads EVERY train
sweep, keeps the returns inside the corridor (same path and width as production) and reports z
percentiles at several tails, and for the run's exported Gaussians (mapped back to the map frame with the
sidecar) how many lie outside each candidate [low tail, high tail] bound.

    uv run python docs/research/supersplat_export/corridor_z_measure.py --run-dir runs/<run> \
        --dataset data/intermediate/<session> --output <json>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from viking_splatad.export.crop import iter_train_sweeps_map, train_lidar_corridor_map
from viking_splatad.export.supersplat_ply import read_supersplat_ply
from viking_splatad.export.viewer_frame import VIEWER_FRAME_FILE_NAME, read_viewer_frame_record, viewer_to_map_points

# Symmetric tail fractions (percent) tried as z bounds: [tail, 100 - tail]. 0 = raw min/max (production).
TAIL_PERCENTS = (0.0, 0.001, 0.01, 0.1)
# Every sweep: the tails are sparse, a subsample would bias them.
EVERY_SWEEP = 1
HORIZONTAL_AXES = slice(0, 2)
Z_AXIS = 2


def main() -> None:
    """Print and save the z-tail measurements.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    # Step 1: production corridor (path + width), then every return inside it (z only, float32).
    corridor = train_lidar_corridor_map(arguments.run_dir, arguments.dataset)
    path_tree = cKDTree(corridor.path_xy)
    corridor_z = []
    for _, points_map in iter_train_sweeps_map(arguments.run_dir, arguments.dataset, EVERY_SWEEP):
        distance, _ = path_tree.query(points_map[:, HORIZONTAL_AXES])
        corridor_z.append(points_map[distance <= corridor.width_m, Z_AXIS].astype(np.float32))
    corridor_z = np.concatenate(corridor_z)
    # Step 2: exported Gaussians back in the map frame, restricted to the horizontal corridor.
    renders = arguments.run_dir / "renders"
    record = read_viewer_frame_record(renders / VIEWER_FRAME_FILE_NAME)
    fields = read_supersplat_ply(renders / "supersplat.ply")
    means = viewer_to_map_points(np.stack([fields[axis] for axis in "xyz"], axis=1).astype(np.float64),
                                 record["origin_map"], record["rotation"])
    gaussian_z = means[:, Z_AXIS]
    # Step 3: candidate bounds per tail and the Gaussians each would cut.
    candidates = {}
    for tail in TAIL_PERCENTS:
        z_lower = float(np.percentile(corridor_z, tail))
        z_upper = float(np.percentile(corridor_z, 100.0 - tail))
        candidates[str(tail)] = {
            "z_lower": z_lower, "z_upper": z_upper,
            "gaussians_below": int(np.count_nonzero(gaussian_z < z_lower)),
            "gaussians_above": int(np.count_nonzero(gaussian_z > z_upper)),
        }
    report = {
        "run_dir": arguments.run_dir.as_posix(), "corridor_width_m": corridor.width_m,
        "corridor_returns": int(len(corridor_z)), "gaussians": int(len(gaussian_z)),
        "gaussian_z_percentiles": {str(q): float(np.percentile(gaussian_z, q))
                                   for q in (0.001, 0.01, 0.1, 1, 99, 99.9, 99.99, 99.999)},
        "candidates_by_tail_percent": candidates,
    }
    arguments.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
