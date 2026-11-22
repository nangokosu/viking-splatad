"""Measure a driven-path corridor crop for the SuperSplat export, to choose its width from data.

For one run: (1) the horizontal range of every train lidar return from its own sensor origin
(percentiles), (2) the train sensor positions (map frame), (3) for corridor widths equal to those
range percentiles, how many of the PLY's Gaussians lie within that horizontal distance of the
nearest train sensor position, and the extent of what is kept; (4) the z percentiles of the train
lidar returns inside each corridor.

The PLY is already cropped to the lidar min/max box, so every corridor count is a subset of it.
Reuses the export's own readers (``read_train_lidar_rows``, ``open_dataset_view``, ``load_lidar_sweep``,
T70 ``sweep_points_to_map``, ``read_supersplat_ply``).

    uv run python docs/research/supersplat_export/corridor_crop_measure.py --run-dir runs/<run>
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from viking_splatad.export.crop import check_dataset_identity, read_train_lidar_rows
from viking_splatad.export.supersplat_ply import read_supersplat_ply
from viking_splatad.splatad.viking_conversions import load_lidar_sweep, open_dataset_view, parse_ego_filter
from viking_splatad.viz.viewer_export import sweep_points_to_map

# Read every n-th train sweep: percentiles of millions of returns are stable on a 1-in-10 sample,
# and the full set takes minutes. Sensor positions use every train sweep (cheap: poses only).
SWEEP_SAMPLE_STRIDE = 10
# Range percentiles evaluated as corridor widths.
RANGE_PERCENTILES = (50, 75, 90, 95, 99)
# z percentiles reported for the lidar returns inside a corridor (lower tail, upper tail).
Z_PERCENTILES = (0.1, 1, 99, 99.9)
HORIZONTAL_AXES = slice(0, 2)


def main() -> None:
    """Print and save the measurements as JSON.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    check_dataset_identity(arguments.run_dir, arguments.dataset)
    view = open_dataset_view(arguments.dataset)
    ego_radius, ego_box = parse_ego_filter(view.manifest["lidar"]["min_lidar_dist"])
    # Step 1: train sensor positions (all train sweeps) and sampled returns with their horizontal range.
    sensor_positions, sampled_points, sampled_ranges = [], [], []
    for sensor_name, rows in read_train_lidar_rows(arguments.run_dir).items():
        poses = view.sensor_poses[sensor_name]
        frame_indices = [int(row["frame_index"]) for row in rows]
        sensor_positions.append(poses[frame_indices, :3, 3])
        for frame_index in frame_indices[::SWEEP_SAMPLE_STRIDE]:
            sweep = load_lidar_sweep(view.sensor_frame_paths[sensor_name][frame_index], ego_radius, ego_box)
            points_map = sweep_points_to_map(sweep[:, :3], poses[frame_index])
            sampled_points.append(points_map)
            # Horizontal range from this sweep's own sensor origin (map xy).
            sampled_ranges.append(np.linalg.norm(points_map[:, HORIZONTAL_AXES] - poses[frame_index, :2, 3], axis=1))
    sensor_positions = np.concatenate(sensor_positions)
    sampled_points = np.concatenate(sampled_points)
    sampled_ranges = np.concatenate(sampled_ranges)
    path_tree = cKDTree(sensor_positions[:, HORIZONTAL_AXES])
    # Step 2: PLY means and their horizontal distance to the driven path.
    fields = read_supersplat_ply(arguments.run_dir / "renders" / "supersplat.ply")
    means = np.stack([fields[axis] for axis in "xyz"], axis=1).astype(np.float64)
    gaussian_path_distance, _ = path_tree.query(means[:, HORIZONTAL_AXES])
    point_path_distance, _ = path_tree.query(sampled_points[:, HORIZONTAL_AXES])
    # Step 3: one row per corridor width = a range percentile.
    corridors = {}
    for percentile in RANGE_PERCENTILES:
        width = float(np.percentile(sampled_ranges, percentile))
        kept = gaussian_path_distance <= width
        lidar_z = sampled_points[point_path_distance <= width, 2]
        corridors[f"p{percentile}"] = {
            "width_m": width,
            "gaussians_kept": int(kept.sum()),
            "gaussians_kept_fraction": float(kept.mean()),
            "kept_extent_min": means[kept].min(axis=0).round(1).tolist(),
            "kept_extent_max": means[kept].max(axis=0).round(1).tolist(),
            "lidar_z_percentiles": {str(q): float(np.percentile(lidar_z, q)) for q in Z_PERCENTILES},
            "lidar_z_min_max": [float(lidar_z.min()), float(lidar_z.max())],
            "gaussian_z_percentiles": {str(q): float(np.percentile(means[kept, 2], q)) for q in Z_PERCENTILES},
            # Gaussians of the corridor outside the lidar z range [p0.1, p99.9] (would be cut by that bound).
            "gaussians_outside_lidar_z_p01_p999": int(np.count_nonzero(
                (means[kept, 2] < np.percentile(lidar_z, Z_PERCENTILES[0]))
                | (means[kept, 2] > np.percentile(lidar_z, Z_PERCENTILES[-1])))),
        }
    report = {
        "run_dir": arguments.run_dir.as_posix(), "sweep_sample_stride": SWEEP_SAMPLE_STRIDE,
        "train_sensor_positions": int(len(sensor_positions)),
        "sensor_position_extent_min": sensor_positions.min(axis=0).round(1).tolist(),
        "sensor_position_extent_max": sensor_positions.max(axis=0).round(1).tolist(),
        "sampled_returns": int(len(sampled_ranges)),
        "horizontal_range_percentiles_m": {str(q): float(np.percentile(sampled_ranges, q)) for q in RANGE_PERCENTILES},
        "ply_gaussians": int(len(means)),
        "corridors": corridors,
    }
    arguments.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
