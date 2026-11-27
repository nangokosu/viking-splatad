"""docs/research/build_summary_report.py: whole-scene tables and the image / lidar render quality rows."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "docs" / "research" / "build_summary_report.py"
THRESHOLD_TEXT = "0.5"
METHODS = ("splatad", "splatad-radar-init", "splatad-radar-init-lidar")
# One distinct mean per method, so a row shows which arm each cell belongs to.
MEAN_BY_METHOD = {"splatad": 0.25, "splatad-radar-init": 0.5, "splatad-radar-init-lidar": 0.75}
STD_VALUE = 0.125
# Defined repeats per arm in the synthetic summary (the campaign's 5; at least MINIMUM_REPEATS_FOR_SPREAD).
REPEAT_COUNT = 5


def load_report_module():
    """Import the report builder as a module.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("build_summary_report_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def metric_statistics(mean_value: float) -> dict:
    """One summary metric entry.

    Args:
        mean_value: Mean over the repeats.

    Returns:
        Entry with n, mean, std, min, max (``describe_values`` layout).
    """
    return {"n": REPEAT_COUNT, "mean": mean_value, "std": STD_VALUE, "min": mean_value - STD_VALUE, "max": mean_value + STD_VALUE}


def synthetic_summary(report_module) -> dict:
    """A one-session summary holding the whole-scene P2 and the quality keys of three arms.

    Args:
        report_module: The loaded builder (for its key helpers).

    Returns:
        Summary dict as scripts/20 writes it, reduced to the keys the report reads.
    """
    arms = {}
    for method in METHODS:
        metrics = {report_module.metric_key(THRESHOLD_TEXT, metric_name): metric_statistics(MEAN_BY_METHOD[method])
                   for metric_name, _ in report_module.METRIC_LABELS}
        for result_key, metric_names in ((report_module.IMAGE_RESULT_KEY, report_module.IMAGE_METRIC_NAMES),
                                         (report_module.LIDAR_RESULT_KEY, report_module.REPORTED_LIDAR_METRICS)):
            for metric_name in metric_names:
                metrics[report_module.quality_key(result_key, metric_name)] = metric_statistics(MEAN_BY_METHOD[method])
        arms[f"profile_{method}"] = {"train_method": method, "metrics": metrics}
    return {"arms": arms, "p2_thresholds": [THRESHOLD_TEXT]}


def test_quality_table_has_one_row_per_metric_with_mean_std_per_arm():
    """Each quality row lists mean ± std of every arm in method order, for the image and lidar groups."""
    report_module = load_report_module()
    summary = synthetic_summary(report_module)
    for result_key, metric_names in ((report_module.IMAGE_RESULT_KEY, report_module.IMAGE_METRIC_NAMES),
                                     (report_module.LIDAR_RESULT_KEY, report_module.REPORTED_LIDAR_METRICS)):
        lines = report_module.quality_table(summary, result_key, metric_names)
        # Header, rule, one row per metric, trailing blank.
        assert len(lines) == 2 + len(metric_names) + 1
        expected_cells = [f"{MEAN_BY_METHOD[method]:.{report_module.DECIMALS}f} ± {STD_VALUE:.{report_module.DECIMALS}f}"
                          for method in METHODS]
        assert lines[2] == f"| {metric_names[0]} | " + " | ".join(expected_cells) + " |"


def test_whole_scene_gap_is_arm_minus_stock():
    """The gap row of a compared arm is its mean minus stock's, labelled separated when the ranges do not overlap."""
    report_module = load_report_module()
    summary = synthetic_summary(report_module)
    lines = report_module.whole_scene_tables({session: summary for session in report_module.SESSIONS}, THRESHOLD_TEXT)
    recall_gap_row = [line for line in lines if line.startswith("| recall |")][-1]
    # init + loss minus stock: 0.75 - 0.25 = +0.500, ranges [0.625, 0.875] and [0.125, 0.375] are disjoint.
    assert "+0.500 (sep)" in recall_gap_row
    assert "hidden" not in "\n".join(lines).lower()
    assert pytest.approx(report_module.gap_and_separation(metric_statistics(0.75), metric_statistics(0.25))[0]) == 0.5


def test_single_repeat_and_undefined_statistics_print_not_available():
    """Like scripts/20 and 21: one defined repeat has no spread (n/a, never sep); no defined repeat prints n/a."""
    report_module = load_report_module()
    single = {"n": 1, "mean": 0.75, "std": None, "min": 0.75, "max": 0.75}
    other_single = {"n": 1, "mean": 0.25, "std": None, "min": 0.25, "max": 0.25}
    undefined = {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    # Disjoint one-point ranges would read "sep" without the minimum-repeat rule.
    assert report_module.format_gap(*report_module.gap_and_separation(single, other_single)) == "+0.500 (n/a)"
    assert report_module.format_gap(*report_module.gap_and_separation(undefined, other_single)) == "n/a"
    assert report_module.format_mean_std(single) == "0.750 ± n/a"
    assert report_module.format_mean_std(undefined) == "n/a"
