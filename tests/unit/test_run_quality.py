"""Tests for the per-run quality readers (synthetic files, expected values written by hand)."""

import json

import pytest

from viking_splatad.eval.run_quality import (
    LIDAR_METRICS_RELATIVE_PATH,
    OVERFITTING_CSV_RELATIVE_PATH,
    REPORTED_LIDAR_METRICS,
    final_image_metrics,
    lidar_render_metrics,
    run_quality,
)

CSV_HEADER = "metric,split,step,value"


def write_curves(csv_path, rows):
    """Write a curves CSV from (metric, split, step, value) tuples."""
    lines = [CSV_HEADER] + [",".join(str(field) for field in row) for row in rows]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("\n".join(lines) + "\n")


def complete_rows():
    """Two eval steps (100, 200) for all metrics, plus train psnr and a later train-only step."""
    return [
        ("psnr", "train", 300, 99.0),
        ("psnr", "train", 200, 30.0),
        ("psnr", "test", 100, 20.0),
        ("psnr", "test", 200, 22.0),
        ("ssim", "test", 100, 0.5),
        ("ssim", "test", 200, 0.6),
        ("lpips", "test", 100, 0.4),
        ("lpips", "test", 200, 0.3),
        ("chamfer_distance", "test", 300, 7.0),
    ]


def write_lidar(json_path, mean_values):
    """Write a lidar_metrics.json with the given mean block."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"mean": mean_values, "per_scan": {}}))


def full_lidar_means():
    """Distinct value per reported metric."""
    return {name: float(index + 1) for index, name in enumerate(REPORTED_LIDAR_METRICS)}


def test_final_image_metrics_picks_largest_test_step(tmp_path):
    csv_path = tmp_path / "curves.csv"
    write_curves(csv_path, complete_rows())
    assert final_image_metrics(csv_path) == {"psnr": 22.0, "ssim": 0.6, "lpips": 0.3, "step": 200}


def test_final_image_metrics_uses_latest_common_step(tmp_path):
    """lpips logged at 300 but psnr/ssim not: the latest step all three share is 200."""
    csv_path = tmp_path / "curves.csv"
    write_curves(csv_path, complete_rows() + [("lpips", "test", 300, 0.2)])
    assert final_image_metrics(csv_path) == {"psnr": 22.0, "ssim": 0.6, "lpips": 0.3, "step": 200}


def test_final_image_metrics_no_common_step_raises(tmp_path):
    """Metrics logged at disjoint steps share no eval: refused."""
    csv_path = tmp_path / "curves.csv"
    write_curves(csv_path, [("psnr", "test", 100, 20.0), ("ssim", "test", 200, 0.6), ("lpips", "test", 300, 0.3)])
    with pytest.raises(ValueError, match="share no test step"):
        final_image_metrics(csv_path)


def test_final_image_metrics_missing_metric_raises(tmp_path):
    csv_path = tmp_path / "curves.csv"
    write_curves(csv_path, [row for row in complete_rows() if row[0] != "ssim"])
    with pytest.raises(ValueError, match="ssim"):
        final_image_metrics(csv_path)


def test_final_image_metrics_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="nothing.csv"):
        final_image_metrics(tmp_path / "nothing.csv")


def test_lidar_render_metrics_reads_mean_values(tmp_path):
    json_path = tmp_path / "lidar_metrics.json"
    write_lidar(json_path, {**full_lidar_means(), "intensity_rmse": 9.0})
    assert lidar_render_metrics(json_path) == full_lidar_means()


def test_lidar_render_metrics_missing_key_raises(tmp_path):
    json_path = tmp_path / "lidar_metrics.json"
    means = full_lidar_means()
    del means["median_chamfer_distance"]
    write_lidar(json_path, means)
    with pytest.raises(KeyError, match="median_chamfer_distance"):
        lidar_render_metrics(json_path)


def test_run_quality_locates_both_files(tmp_path):
    write_curves(tmp_path / OVERFITTING_CSV_RELATIVE_PATH, complete_rows())
    write_lidar(tmp_path / LIDAR_METRICS_RELATIVE_PATH, full_lidar_means())
    result = run_quality(tmp_path)
    assert result["image"]["step"] == 200
    assert result["lidar_render"] == full_lidar_means()


def test_run_quality_missing_file_names_path(tmp_path):
    write_curves(tmp_path / OVERFITTING_CSV_RELATIVE_PATH, complete_rows())
    with pytest.raises(FileNotFoundError, match="lidar_metrics.json"):
        run_quality(tmp_path)
