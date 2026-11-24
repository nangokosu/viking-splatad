"""Per-run quality readers: image guard metrics and lidar render report-only metrics.

Both read files a finished run already holds (no recomputation):
    overfitting/overfitting_curves.csv   (scripts/11_plot_overfitting_curves.py)
    renders/test/lidar_metrics.json      (scripts/08_render_lidar_pointcloud.py)
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict

# Run-relative locations of the two files.
OVERFITTING_CSV_RELATIVE_PATH = Path("overfitting") / "overfitting_curves.csv"
LIDAR_METRICS_RELATIVE_PATH = Path("renders") / "test" / "lidar_metrics.json"
# CSV column names (header written by scripts/11: metric,split,step,value).
METRIC_COLUMN, SPLIT_COLUMN, STEP_COLUMN, VALUE_COLUMN = "metric", "split", "step", "value"
# The split of the training-time eval (test + guard frames).
TEST_SPLIT_NAME = "test"
# Image metrics reported, in output order.
IMAGE_METRIC_NAMES = ("psnr", "ssim", "lpips")
# Key of the final step in the image result.
STEP_KEY = "step"
# Lidar metrics reported from the JSON's per-metric means (report-only, never gating).
REPORTED_LIDAR_METRICS = ("chamfer_distance", "median_chamfer_distance", "depth_median_l2", "depth_mean_rel_l2")
# Top-level key of the lidar JSON holding the mean over scans.
LIDAR_MEAN_KEY = "mean"
# Keys of the run_quality result.
IMAGE_RESULT_KEY, LIDAR_RESULT_KEY = "image", "lidar_render"


def final_image_metrics(overfitting_csv_path: Path) -> Dict[str, Any]:
    """Test-split image metrics at the last step where all three were evaluated.

    The ``test`` split of the curves is the training-time eval (nerfstudio's "val"): test + guard
    frames, not held-out test frames only. It is a guard on image quality, not a held-out-only score.

    Args:
        overfitting_csv_path: ``overfitting_curves.csv`` (columns metric,split,step,value).

    Returns:
        {"psnr", "ssim", "lpips": float values at the latest test step shared by all three,
        "step": int}. A log can miss one metric's final eval (seen on a May run: ssim's last row at
        19600 while psnr and lpips reach 20000), so the step is the latest common one and is reported.

    Raises:
        FileNotFoundError: The CSV does not exist.
        ValueError: A metric has no test rows, or the three metrics share no test step.
    """
    overfitting_csv_path = Path(overfitting_csv_path)
    if not overfitting_csv_path.is_file():
        raise FileNotFoundError(f"{overfitting_csv_path} missing")
    # Step 1: collect the test-split values per image metric, keyed by step.
    values_by_metric_and_step: Dict[str, Dict[int, float]] = {name: {} for name in IMAGE_METRIC_NAMES}
    with open(overfitting_csv_path, newline="") as csv_handle:
        for row in csv.DictReader(csv_handle):
            if row[SPLIT_COLUMN] == TEST_SPLIT_NAME and row[METRIC_COLUMN] in values_by_metric_and_step:
                values_by_metric_and_step[row[METRIC_COLUMN]][int(row[STEP_COLUMN])] = float(row[VALUE_COLUMN])
    # Step 2: every metric needs test rows.
    for metric_name, values_by_step in values_by_metric_and_step.items():
        if not values_by_step:
            raise ValueError(f"{overfitting_csv_path}: no '{TEST_SPLIT_NAME}' rows for metric '{metric_name}'")
    # Step 3: the latest step at which every metric was logged (one eval logs all three; a missing
    # row at the last eval falls back to the previous complete one).
    common_steps = set.intersection(*(set(values) for values in values_by_metric_and_step.values()))
    if not common_steps:
        last_step_by_metric = {name: max(values) for name, values in values_by_metric_and_step.items()}
        raise ValueError(f"{overfitting_csv_path}: the image metrics share no test step (last steps {last_step_by_metric})")
    final_step = max(common_steps)
    # Step 4: values at that step.
    result: Dict[str, Any] = {name: values_by_metric_and_step[name][final_step] for name in IMAGE_METRIC_NAMES}
    result[STEP_KEY] = final_step
    return result


def lidar_render_metrics(lidar_metrics_json_path: Path) -> Dict[str, float]:
    """Mean lidar render metrics reported (not gated) for a run.

    Args:
        lidar_metrics_json_path: ``lidar_metrics.json`` ({"mean": {...}, "per_scan": {...}}).

    Returns:
        {metric: mean value} for each of REPORTED_LIDAR_METRICS.

    Raises:
        FileNotFoundError: The JSON does not exist.
        KeyError: The ``mean`` block or a reported metric is missing (the message names it).
    """
    lidar_metrics_json_path = Path(lidar_metrics_json_path)
    if not lidar_metrics_json_path.is_file():
        raise FileNotFoundError(f"{lidar_metrics_json_path} missing")
    record = json.loads(lidar_metrics_json_path.read_text())
    # Step 1: the mean block.
    if LIDAR_MEAN_KEY not in record:
        raise KeyError(f"{lidar_metrics_json_path}: missing key '{LIDAR_MEAN_KEY}'")
    mean_values = record[LIDAR_MEAN_KEY]
    # Step 2: each reported metric must be present (chamfer is absent when a cloud was empty).
    missing_metrics = [name for name in REPORTED_LIDAR_METRICS if name not in mean_values]
    if missing_metrics:
        raise KeyError(f"{lidar_metrics_json_path}: missing mean metric(s) {missing_metrics}")
    return {name: float(mean_values[name]) for name in REPORTED_LIDAR_METRICS}


def run_quality(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Image guard metrics and lidar render metrics of one fetched run folder.

    Args:
        run_dir: ``runs/<run_id>``.

    Returns:
        {"image": final_image_metrics(...), "lidar_render": lidar_render_metrics(...)}.

    Raises:
        FileNotFoundError: Either file is missing (the message names the path).
        ValueError: From ``final_image_metrics`` (a metric has no test rows or no common step).
        KeyError: From ``lidar_render_metrics`` (the mean block or a reported metric is missing).
    """
    run_dir = Path(run_dir)
    # The files sit at fixed run-relative paths; no split.json is needed to read them.
    return {
        IMAGE_RESULT_KEY: final_image_metrics(run_dir / OVERFITTING_CSV_RELATIVE_PATH),
        LIDAR_RESULT_KEY: lidar_render_metrics(run_dir / LIDAR_METRICS_RELATIVE_PATH),
    }
