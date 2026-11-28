"""docs/proposal/poster/poster_numbers.py: poster numbers and strings from the whole-scene summaries."""

import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "docs" / "proposal" / "poster" / "poster_numbers.py"
METHODS = ("splatad", "splatad-radar-init", "splatad-radar-init-lidar")
METRIC_NAMES = ("recall", "precision", "iou")
THRESHOLD_TEXT = "0.5"
PERCENT_FACTOR = 100
RUN_COUNT = 5
# (mean, min, max) per method for recall: stock [0.30, 0.35], init + loss fully above it (separated),
# radar init inside stock's range (overlap).
RECALL_RANGES = {"splatad": (0.32, 0.30, 0.35), "splatad-radar-init": (0.33, 0.31, 0.36),
                 "splatad-radar-init-lidar": (0.40, 0.38, 0.42)}
# Same for the other metrics: all overlapping.
OTHER_RANGES = {"splatad": (0.50, 0.45, 0.55), "splatad-radar-init": (0.51, 0.46, 0.56),
                "splatad-radar-init-lidar": (0.49, 0.44, 0.54)}


def load_module():
    """Import the poster numbers module by path.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("poster_numbers_under_test", MODULE_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def entry(mean_value, minimum_value, maximum_value, run_count=RUN_COUNT):
    """One summary metric entry.

    Args:
        mean_value: Mean (None for undefined).
        minimum_value: Smallest repeat.
        maximum_value: Largest repeat.
        run_count: Number of repeats.

    Returns:
        Entry in ``describe_values`` layout.
    """
    return {"n": run_count, "mean": mean_value, "std": 0.01, "min": minimum_value, "max": maximum_value}


def synthetic_summary(recall_ranges=None, other_ranges=None):
    """A one-session summary with three arms.

    Args:
        recall_ranges: Method -> (mean, min, max) of recall.
        other_ranges: Method -> (mean, min, max) of precision and IoU.

    Returns:
        Summary dict with the flat metric keys.
    """
    recall_ranges = recall_ranges or RECALL_RANGES
    other_ranges = other_ranges or OTHER_RANGES
    arms = {}
    for method in METHODS:
        metrics = {}
        for metric_name in METRIC_NAMES:
            values = recall_ranges[method] if metric_name == "recall" else other_ranges[method]
            metrics[f"obstacles/P2@{THRESHOLD_TEXT}/{metric_name}/pooled"] = entry(*values)
        arms[f"profile_{method}"] = {"train_method": method, "metrics": metrics}
    return {"arms": arms, "p2_thresholds": [THRESHOLD_TEXT]}


def both_sessions(module, **kwargs):
    """The synthetic summary under both session names.

    Args:
        module: The loaded module.
        **kwargs: Passed to ``synthetic_summary``.

    Returns:
        Session name -> summary.
    """
    return {session: synthetic_summary(**kwargs) for session in module.SESSIONS}


def test_format_percent_range_rounds_fraction_times_hundred():
    """Mean and [min, max] print as fraction x 100, rounded to the requested decimals."""
    module = load_module()
    metric = module.MetricRange(mean=0.3804, minimum=0.3761, maximum=0.3829, run_count=RUN_COUNT)
    expected = (f"{0.3804 * PERCENT_FACTOR:.1f} [{0.3761 * PERCENT_FACTOR:.1f}, {0.3829 * PERCENT_FACTOR:.1f}]")
    assert module.format_percent_range(metric) == expected
    assert module.format_percent_range(metric, decimals=2) == "38.04 [37.61, 38.29]"


def test_load_session_metrics_returns_fractions_straight_from_summary():
    """Every arm and metric is read unscaled, with its run count."""
    module = load_module()
    metrics = module.load_session_metrics(synthetic_summary())
    mean_value, minimum_value, maximum_value = RECALL_RANGES["splatad-radar-init-lidar"]
    recall = metrics["splatad-radar-init-lidar"]["recall"]
    assert (recall.mean, recall.minimum, recall.maximum, recall.run_count) == (
        mean_value, minimum_value, maximum_value, RUN_COUNT)
    assert set(metrics) == set(METHODS)
    assert set(metrics["splatad"]) == set(METRIC_NAMES)


def test_undefined_metric_raises():
    """A metric whose mean is None is refused."""
    module = load_module()
    summary = synthetic_summary()
    summary["arms"]["profile_splatad"]["metrics"][f"obstacles/P2@{THRESHOLD_TEXT}/recall/pooled"] = entry(None, None, None)
    with pytest.raises(ValueError, match="undefined"):
        module.load_session_metrics(summary)


def test_single_run_raises():
    """A metric with fewer than MINIMUM_RUN_COUNT runs is refused."""
    module = load_module()
    summary = synthetic_summary()
    summary["arms"]["profile_splatad"]["metrics"][f"obstacles/P2@{THRESHOLD_TEXT}/recall/pooled"] = entry(
        0.3, 0.3, 0.3, run_count=1)
    with pytest.raises(ValueError, match="run"):
        module.load_session_metrics(summary)


def test_gap_in_points_and_separation_flags():
    """Gap = (arm mean - stock mean) x 100; separated iff the [min, max] ranges do not overlap."""
    module = load_module()
    gaps = module.arm_gaps(synthetic_summary())
    loss_recall = gaps["splatad-radar-init-lidar"]["recall"]
    assert loss_recall.gap_points == pytest.approx((0.40 - 0.32) * PERCENT_FACTOR)
    # Ranges [0.38, 0.42] and [0.30, 0.35] do not overlap.
    assert loss_recall.separated is True
    init_recall = gaps["splatad-radar-init"]["recall"]
    assert init_recall.gap_points == pytest.approx((0.33 - 0.32) * PERCENT_FACTOR)
    # [0.31, 0.36] overlaps [0.30, 0.35].
    assert init_recall.separated is False
    assert gaps["splatad-radar-init-lidar"]["precision"].gap_points == pytest.approx((0.49 - 0.50) * PERCENT_FACTOR)
    assert set(gaps) == {"splatad-radar-init", "splatad-radar-init-lidar"}


def test_format_gap_signs():
    """Positive gaps get '+', negative gaps the true minus sign (U+2212), never an ASCII hyphen."""
    module = load_module()
    assert module.format_gap(module.GapResult(3.84, True)) == "+3.8 pts"
    negative_text = module.format_gap(module.GapResult(-0.46, False))
    assert negative_text == "−0.5 pts"
    assert "-" not in negative_text
    assert module.format_gap(module.GapResult(3.84, True), decimals=2) == "+3.84 pts"


def test_table_shape_and_order():
    """Header plus one row per metric; columns are sessions x arms in METHOD_ORDER."""
    module = load_module()
    rows = module.table_rows(both_sessions(module))
    column_count = 1 + len(module.SESSIONS) * len(module.METHOD_ORDER)
    assert len(rows) == 1 + len(module.METRIC_LABELS)
    assert all(len(row) == column_count for row in rows)
    assert rows[0][0] == "Obstacles (%)"
    assert rows[0][1] == f"{module.SESSION_LABELS[module.SESSIONS[0]]} · {module.ARM_LABEL_BY_METHOD[METHODS[0]]}"
    assert rows[0][-1] == f"{module.SESSION_LABELS[module.SESSIONS[-1]]} · {module.ARM_LABEL_BY_METHOD[METHODS[-1]]}"
    assert [row[0] for row in rows[1:]] == [label for _, label in module.METRIC_LABELS]
    # First data cell: first session, stock, recall.
    mean_value, minimum_value, maximum_value = RECALL_RANGES["splatad"]
    expected_cell = module.format_percent_range(module.MetricRange(mean_value, minimum_value, maximum_value, RUN_COUNT))
    assert rows[1][1] == expected_cell
    # Last cell: last session, init + loss, IoU.
    other_mean, other_min, other_max = OTHER_RANGES["splatad-radar-init-lidar"]
    assert rows[-1][-1] == module.format_percent_range(module.MetricRange(other_mean, other_min, other_max, RUN_COUNT))


def test_key_findings_follow_flags():
    """Two sentences per session whose range wording follows the separation flags."""
    module = load_module()
    findings = module.key_findings(both_sessions(module))
    assert len(findings) == 2 * len(module.SESSIONS)
    expected_loss_sentence = (
        "June · radar init + loss vs stock: recall +8.0 pts (ranges separated), precision −1.0 pts (ranges overlap)")
    assert findings[0] == expected_loss_sentence
    assert findings[1] == "June · radar init vs stock: recall +1.0 pts (ranges overlap)"
    # Flip init + loss recall to overlap: the wording must change.
    overlapping_ranges = dict(RECALL_RANGES)
    overlapping_ranges["splatad-radar-init-lidar"] = (0.34, 0.30, 0.38)
    flipped = module.key_findings(both_sessions(module, recall_ranges=overlapping_ranges))
    assert "recall +2.0 pts (ranges overlap)" in flipped[0]
    assert "ranges separated" not in flipped[0]


def test_real_june_summary_recall_text_matches_json():
    """The table's June init + loss recall cell equals the committed summary JSON, formatted."""
    module = load_module()
    summaries = module.load_all_summaries()
    summary_path = (Path(__file__).resolve().parents[2] / "reports" / "gaussian_occupancy" / "june_tall_grass"
                    / "gaussian_occupancy_summary.json")
    raw_summary = json.loads(summary_path.read_text())
    arm = next(arm for arm in raw_summary["arms"].values() if arm["train_method"] == "splatad-radar-init-lidar")
    statistics = arm["metrics"]["obstacles/P2@0.5/recall/pooled"]
    expected = module.format_percent_range(module.MetricRange(
        statistics["mean"], statistics["min"], statistics["max"], statistics["n"]))
    recall_row = module.table_rows(summaries)[1]
    # June is the first session; init + loss is its third arm.
    assert recall_row[1 + 2] == expected


def test_format_gap_never_prints_negative_zero():
    """A gap that rounds to zero prints as +0.0 (the sign follows the rounded value)."""
    module = load_module()
    tiny_negative_points = -0.04
    assert module.format_gap(module.GapResult(tiny_negative_points, False)) == "+0.0 pts"


def test_undefined_gap_raises():
    """An arm with no defined repeat has no gap; arm_gaps refuses rather than printing a crash or a blank."""
    module = load_module()
    summary = synthetic_summary()
    undefined_entry = entry(None, None, None, run_count=0)  # describe_values: no defined repeat -> n = 0
    for arm in summary["arms"].values():
        if arm["train_method"] == module.LOSS_METHOD:
            arm["metrics"][f"obstacles/P2@{THRESHOLD_TEXT}/recall/pooled"] = undefined_entry
    with pytest.raises(ValueError, match="undefined"):
        module.arm_gaps(summary)
