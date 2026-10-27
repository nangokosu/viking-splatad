"""Tests for the pure training-curve helpers (T63 eval cadence, T64 train/test tag pairing)."""

import numpy as np
import pytest

from viking_splatad.eval.training_curves import (
    MetricPanel,
    count_eval_points,
    curve_rows,
    eval_interval_for_minimum_points,
    pair_metric_tags,
    reduced_resolution_end_step,
    rolling_median,
    smoothing_window,
)

TRAIN = "Train Metrics Dict/"
TEST = "Eval Images Metrics Dict (all images)/"


@pytest.mark.covers("T63")
@pytest.mark.parametrize("max_steps", [3000, 30001, 51, 1000])
def test_interval_gives_at_least_minimum_points(max_steps):
    """Brute-force count of eval steps 1..N-1 divisible by k is >= P and matches the formula."""
    minimum_points = 50
    interval = eval_interval_for_minimum_points(max_steps, minimum_points)
    brute_force = sum(1 for step in range(1, max_steps) if step % interval == 0)
    assert brute_force == count_eval_points(max_steps, interval)
    assert brute_force >= minimum_points
    # Known answers: floor((N-1)/P).
    assert eval_interval_for_minimum_points(30001, 50) == 600
    assert eval_interval_for_minimum_points(3000, 50) == 59


@pytest.mark.covers("T63")
@pytest.mark.parametrize("max_steps", [50, 40])
def test_interval_refuses_too_short_runs(max_steps):
    """Runs with fewer than P usable steps cannot give P evals."""
    with pytest.raises(ValueError):
        eval_interval_for_minimum_points(max_steps, 50)


@pytest.mark.covers("T64")
def test_pairing_pairs_shared_metrics_and_keeps_test_only():
    """Shared metrics pair first (sorted), test-only next; train-only and speed tags vanish."""
    tags = [
        TRAIN + "psnr", TRAIN + "depth_median_l2", TRAIN + "camera_opt_vel_max",
        TEST + "psnr", TEST + "depth_median_l2", TEST + "lpips", TEST + "fps", "Train Loss",
    ]
    assert pair_metric_tags(tags) == [
        MetricPanel("depth_median_l2", TRAIN + "depth_median_l2", TEST + "depth_median_l2"),
        MetricPanel("psnr", TRAIN + "psnr", TEST + "psnr"),
        MetricPanel("lpips", None, TEST + "lpips"),
    ]


@pytest.mark.covers("T64")
def test_pairing_lidar_only_run_has_test_only_psnr():
    """Without a train psnr tag the psnr panel is test-only."""
    panels = pair_metric_tags([TEST + "psnr", "Eval Images Metrics/psnr"])
    assert panels == [MetricPanel("psnr", None, TEST + "psnr")]


def test_smoothing_window_and_rolling_median():
    """Window = round(k / step spacing) forced odd; median removes a spike, keeps constants."""
    train_steps = np.arange(0, 1000, 10)
    assert smoothing_window(train_steps, 600) == 61
    assert smoothing_window(train_steps, 5) == 1
    assert smoothing_window(train_steps, 600 - 10) == 59  # 59 is odd already
    assert smoothing_window(train_steps, 20) == 3  # 2 is even -> 3
    spike = np.array([0, 0, 9, 0, 0.0])
    assert np.array_equal(rolling_median(spike, 3), np.zeros(5))
    constant = np.full(7, 2.5)
    assert np.array_equal(rolling_median(constant, 5), constant)
    # Edge uses available neighbours: first element median of [1, 5] = 3.
    assert rolling_median(np.array([1.0, 5.0, 9.0]), 3)[0] == 3.0


def test_reduced_resolution_end_step():
    """Resolution is reduced until schedule x downscales steps."""
    assert reduced_resolution_end_step(3000, 2) == 6000


def test_curve_rows_sorted_and_typed():
    """Rows sorted by (metric, split, step) with int steps and float values."""
    series = {
        ("psnr", "train"): (np.array([20, 10]), np.array([1, 2])),
        ("lpips", "test"): (np.array([5]), np.array([0.5])),
    }
    rows = curve_rows(series)
    assert rows == [("lpips", "test", 5, 0.5), ("psnr", "train", 10, 2.0), ("psnr", "train", 20, 1.0)]
    assert all(type(row[2]) is int and type(row[3]) is float for row in rows)
