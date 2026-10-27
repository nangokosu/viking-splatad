"""Pure helpers for the overfitting monitor's train-vs-test curves (T63, T64).

Numpy-only so they import in the Mac venv (no tensorboard / matplotlib):

    * T63: pick the eval interval so a run logs at least P all-images evals;
    * T64: pair TensorBoard train and test metric tags into plot panels;
    * smoothing, the reduced-resolution shading step and the CSV rows of the curves.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from viking_splatad.registry import transform

# TensorBoard tag prefixes nerfstudio writes for per-step train metrics and for the all-images eval.
TRAIN_METRIC_PREFIX = "Train Metrics Dict/"
TEST_METRIC_PREFIX = "Eval Images Metrics Dict (all images)/"
TRAIN_LOSS_TAG = "Train Loss"
# Throughput counters nerfstudio logs next to the quality metrics; they measure speed, not quality,
# so they never get a panel. Hardcoded because they are fixed nerfstudio/SplatAD logging names.
NON_QUALITY_METRICS = frozenset({"fps", "num_camera_rays_per_sec", "num_lidar_rays_per_sec", "num_rays"})


@transform("T63", "eval cadence: largest interval giving at least P all-images evals in N training steps")
def eval_interval_for_minimum_points(max_num_iterations: int, minimum_eval_points: int) -> int:
    """Largest interval k with at least ``minimum_eval_points`` evals in a run.

    nerfstudio's ``step_check`` fires on steps 1..N-1 divisible by k (step 0 is skipped), so a run
    logs floor((N-1)/k) evals. With k = floor((N-1)/P) we have k <= (N-1)/P, hence
    (N-1)/k >= P and floor((N-1)/k) >= P because P is an integer. Any larger k gives fewer than P.

    Args:
        max_num_iterations: Training steps N.
        minimum_eval_points: Required number of evals P (>= 1).

    Returns:
        The interval k (>= 1).

    Raises:
        ValueError: If N - 1 < P, i.e. even k = 1 cannot give P evals.
    """
    # Step 1: usable eval steps are 1..N-1.
    usable_steps = max_num_iterations - 1
    if minimum_eval_points < 1 or usable_steps < minimum_eval_points:
        raise ValueError(
            f"{max_num_iterations} steps cannot give {minimum_eval_points} evals (needs N-1 >= P)"
        )
    # Step 2: k = floor((N-1)/P), see the proof above.
    return usable_steps // minimum_eval_points


def count_eval_points(max_num_iterations: int, eval_interval: int) -> int:
    """Number of evals in steps 1..N-1 for interval k: floor((N-1)/k).

    Args:
        max_num_iterations: Training steps N.
        eval_interval: Eval interval k (>= 1).

    Returns:
        How many steps in 1..N-1 are divisible by k.
    """
    return max(0, (max_num_iterations - 1) // eval_interval)


@dataclass(frozen=True)
class MetricPanel:
    """One plot panel: a metric with its optional train tag and its test tag."""

    metric: str
    train_tag: Optional[str]
    test_tag: str


@transform("T64", "train/test metric pairing from TensorBoard tags")
def pair_metric_tags(scalar_tags: Iterable[str]) -> List[MetricPanel]:
    """Pair train and test metric tags by metric name.

    Only tags under the train and all-images-eval prefixes count; others (losses, other eval
    groups) are ignored, as are ``NON_QUALITY_METRICS``. A metric needs a test tag to get a panel;
    train-only metrics are dropped.

    Args:
        scalar_tags: All scalar tag names of a TensorBoard run.

    Returns:
        Panels with a train tag first, then test-only panels, each group sorted by metric name.
    """
    tag_set = set(scalar_tags)
    panels_paired: List[MetricPanel] = []
    panels_test_only: List[MetricPanel] = []
    # Step 1: every test tag defines a candidate metric.
    test_metrics = sorted(
        tag[len(TEST_METRIC_PREFIX):] for tag in tag_set if tag.startswith(TEST_METRIC_PREFIX)
    )
    for metric in test_metrics:
        if metric in NON_QUALITY_METRICS:
            continue
        # Step 2: attach the train tag when the run logged one.
        train_tag = TRAIN_METRIC_PREFIX + metric
        panel = MetricPanel(
            metric, train_tag if train_tag in tag_set else None, TEST_METRIC_PREFIX + metric
        )
        (panels_paired if panel.train_tag else panels_test_only).append(panel)
    return panels_paired + panels_test_only


def smoothing_window(train_steps: np.ndarray, eval_interval: int) -> int:
    """Odd median-window length (in train samples) spanning about one eval interval.

    window = max(1, round(k / median(diff(steps)))), +1 if even so the median is centred.

    Args:
        train_steps: Steps of the train-metric samples (increasing).
        eval_interval: Eval interval k in steps.

    Returns:
        Odd integer >= 1.
    """
    # Step 1: samples per eval interval, from the typical logging spacing.
    typical_spacing = float(np.median(np.diff(train_steps)))
    window = max(1, int(round(eval_interval / typical_spacing)))
    # Step 2: force odd.
    return window + 1 if window % 2 == 0 else window


def rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Centred rolling median; edges use the available neighbours (truncated window).

    Args:
        values: 1-D samples.
        window: Odd window length.

    Returns:
        Array of the same length as ``values``.
    """
    half_window = window // 2
    medians = np.empty(len(values), dtype=float)
    for sample_index in range(len(values)):
        # Window [index - half, index + half], clipped to the array bounds.
        start_index = max(0, sample_index - half_window)
        medians[sample_index] = np.median(values[start_index:sample_index + half_window + 1])
    return medians


def reduced_resolution_end_step(resolution_schedule: int, num_downscales: int) -> int:
    """Step where training reaches full resolution: schedule x number of downscales.

    Args:
        resolution_schedule: Steps per downscale level.
        num_downscales: Number of downscale levels.

    Returns:
        Last step of the reduced-resolution phase (end of the shaded region).
    """
    return resolution_schedule * num_downscales


def curve_rows(
    series: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]],
) -> List[Tuple[str, str, int, float]]:
    """Flatten curves into CSV rows sorted by (metric, split, step).

    Args:
        series: (metric, "train"|"test"|"train_loss") -> (steps, values).

    Returns:
        Rows (metric, split, step, value) with int steps and float values.
    """
    rows = [
        (metric, split, int(step), float(value))
        for (metric, split), (steps, values) in series.items()
        for step, value in zip(steps, values)
    ]
    return sorted(rows, key=lambda row: (row[0], row[1], row[2]))
