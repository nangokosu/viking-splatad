"""T86 crop box and kept fraction on a real run, plus the percentile box (information only).

Reads the run's `renders/gaussians.npz` and its train lidar (through the repo's `crop` module),
reports the observed-lidar crop box, how many Gaussians it keeps at the run's `mcmc_min_opacity`
(read from its config.yml, as the export does), and what a 0.1/99.9 percentile box would keep instead.
The box comes from `crop.train_lidar_bounds_map`, which checks the dataset identity and every frame
row; the raw points for the percentiles are gathered right after it from the same rows.

Run with the project venv (JSON is written next to this script):
  uv run python docs/research/supersplat_export/crop_real_may.py
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np

from viking_splatad.export import crop
from viking_splatad.export.splatad_colours import model_settings_from_config_yml
from viking_splatad.splatad.viking_conversions import load_lidar_sweep, open_dataset_view, parse_ego_filter
from viking_splatad.viz.viewer_export import sweep_points_to_map

DEFAULT_RUN_DIR = Path("runs/may_short_grass-may_scene_a-20261116-135432-17bb7a1")
DEFAULT_DATASET_DIR = Path("data/intermediate/may_short_grass")
DEFAULT_OUTPUT_PATH = Path(__file__).with_name("crop_real_may_output.json")
# Script 22 (export): its ``find_train_dir`` resolves the run's train folder from job_info.txt.
EXPORT_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "22_export_supersplat_ply.py"
# Percentile box in percent (low, high): the information-only alternative to the observed box.
PERCENTILE_BOUNDS_PERCENT = (0.1, 99.9)


def load_export_script_module():
    """Import scripts/22 by path (its file name starts with a digit) to reuse its run-layout helpers.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("export_supersplat_ply_script", EXPORT_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def gather_train_points_map(run_dir: Path, dataset_dir: Path) -> np.ndarray:
    """Concatenate all train-split lidar sweeps of a run in the map frame.

    The dataset must be the run's (``check_dataset_identity``). Row-level checks are those that
    ``crop.train_lidar_bounds_map`` already ran on the same rows just before.

    Args:
        run_dir: run folder (provides the train frame list).
        dataset_dir: extracted dataset folder.
    Returns:
        (N, 3) float32 points in the map frame.
    """
    crop.check_dataset_identity(run_dir, dataset_dir)
    rows_by_sensor = crop.read_train_lidar_rows(run_dir)
    view = open_dataset_view(dataset_dir)
    radius, ego_box = parse_ego_filter(view.manifest["lidar"]["min_lidar_dist"])
    clouds = []
    for sensor, sensor_rows in rows_by_sensor.items():
        for row in sensor_rows:
            frame_index = int(row["frame_index"])
            sweep = load_lidar_sweep(view.sensor_frame_paths[sensor][frame_index], radius, ego_box)
            clouds.append(sweep_points_to_map(sweep[:, :3], view.sensor_poses[sensor][frame_index]).astype(np.float32))
    return np.concatenate(clouds)


def main() -> None:
    """Compute both boxes and their kept fractions; print and write JSON."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    arguments = parser.parse_args()

    # Step 0: the opacity floor of the run, read from its config.yml like the export does.
    export_module = load_export_script_module()
    train_dir = export_module.find_train_dir(arguments.run_dir)
    min_opacity = float(model_settings_from_config_yml(train_dir / export_module.CONFIG_FILE_NAME)["mcmc_min_opacity"])
    # Step 1: observed-lidar crop box (T86) and its timing.
    start_time = time.perf_counter()
    lower, upper = crop.train_lidar_bounds_map(arguments.run_dir, arguments.dataset_dir)
    box_runtime_seconds = time.perf_counter() - start_time
    # Step 2: apply it to the trained Gaussians.
    gaussians = np.load(arguments.run_dir / "renders" / "gaussians.npz")
    means, opacities = gaussians["means"], gaussians["opacities"]
    keep = crop.crop_mask(means, opacities, lower, upper, min_opacity)
    # Step 3: percentile box over all train lidar points, for comparison only.
    points = gather_train_points_map(arguments.run_dir, arguments.dataset_dir)
    low_percentile_box, high_percentile_box = (
        np.percentile(points, percent, axis=0) for percent in PERCENTILE_BOUNDS_PERCENT)
    keep_percentile = crop.crop_mask(means, opacities, low_percentile_box, high_percentile_box, min_opacity)

    result = {
        "inputs": {"run_dir": arguments.run_dir.as_posix(), "dataset_dir": arguments.dataset_dir.as_posix(),
                   "min_opacity": min_opacity, "percentile_bounds_percent": list(PERCENTILE_BOUNDS_PERCENT)},
        "observed_box": {"lower": lower.tolist(), "upper": upper.tolist(), "runtime_seconds": box_runtime_seconds,
                         "kept": int(keep.sum()), "total": int(len(keep)), "kept_fraction": float(keep.mean())},
        "percentile_box": {"lower": low_percentile_box.tolist(), "upper": high_percentile_box.tolist(),
                           "lidar_point_count": int(len(points)), "kept": int(keep_percentile.sum()),
                           "kept_fraction": float(keep_percentile.mean())},
    }
    print(f"observed box kept {keep.sum()} of {len(keep)} ({100 * keep.mean():.1f}%)")
    print(f"percentile box kept {keep_percentile.sum()} ({100 * keep_percentile.mean():.1f}%)")
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
