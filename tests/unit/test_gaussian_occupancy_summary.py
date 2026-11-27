"""Tests of the whole-scene summary handling of scripts/20 and scripts/21 and of eval/p2_repeats.flatten_run_metrics."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval.p2_repeats import compare_arms, describe_arm, describe_values, flatten_quality_metrics, flatten_run_metrics
from viking_splatad.eval.scene_metrics import class_group_masks, class_metric_report

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
GROUP_CLASS_IDS = (1, 3)
OTHER_CLASS_ID = 2
P2_THRESHOLDS = [0.5, 0.8]
OCCUPANCY_THRESHOLD = 0.5
RUN_NAME = "synthetic_run"
GROUP_NAME = "obstacles"
CLASS_NAME = "tree_trunk"
# Detection curve of one report class: T79 rate at each f_min of a shared grid.
DETECTION_CURVE = {"f_min": [0.25, 0.5], "detection_rate": [1.0, 0.5]}


def load_script(file_name: str, module_name: str):
    """Import a numbered script as a module.

    Args:
        file_name: File name in scripts/.
        module_name: Name to give the module.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / file_name)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def synthetic_archive_and_report():
    """Six cells, two classes (group + tree_trunk), a cells archive and the matching schema-3 report.

    Returns:
        (cells archive dict, report dict) where the report's P1/P2 blocks are computed from the archive.
    """
    gt_class_ids = np.array([1, 1, 3, OTHER_CLASS_ID, 0, 0])
    neighbourhood_by_class_id = {1: np.array([1, 1, 0, 1, 1, 0], dtype=bool), 3: np.array([0, 0, 1, 1, 1, 1], dtype=bool)}
    objects = {"object_box_ids": np.array([10, 11]), "object_class_ids": np.array([1, 3]),
               "object_cell_offsets": np.array([0, 4, 8]), "object_cells": np.array([0, 1, 3, 4, 2, 3, 4, 5]),
               "object_inside": np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)}
    cells = {"occupancy": np.array([0.9, 0.9, 0.3, 0.9, 0.6, 0.9]), "p1_hit": np.array([1, 0, 0, 0, 0, 1], dtype=bool),
             "threshold": np.float64(OCCUPANCY_THRESHOLD), "p2_thresholds": np.array(P2_THRESHOLDS), **objects}
    classes = {}
    for entry_name, class_ids in ((GROUP_NAME, GROUP_CLASS_IDS), (CLASS_NAME, (1,))):
        neighbourhood, truth = class_group_masks(gt_class_ids, class_ids, neighbourhood_by_class_id)
        cells[f"neighbourhood_{entry_name}"], cells[f"truth_{entry_name}"] = neighbourhood, truth
        metric_report = class_metric_report(class_ids, neighbourhood, truth, cells["p1_hit"], cells["occupancy"],
                                            P2_THRESHOLDS, objects)
        classes[entry_name] = {"class_ids": list(class_ids), **json.loads(json.dumps(metric_report)),
                               "detection": {"curve": DETECTION_CURVE}}
    return cells, {"classes": classes}


@pytest.mark.covers("T94")
@pytest.mark.covers("T95")
def test_flatten_run_metrics_keys_and_values():
    """Keys follow class/test/metric/variant; values equal the report's whole-scene blocks."""
    _, report = synthetic_archive_and_report()
    flat = flatten_run_metrics(report)
    group = report["classes"][GROUP_NAME]
    assert flat[f"{GROUP_NAME}/P2@0.5/recall/pooled"] == group["P2"]["0.5"]["pooled"]["recall"]
    assert flat[f"{GROUP_NAME}/P2@0.8/precision/mean"] == group["P2"]["0.8"]["per_object"]["mean_precision"]
    assert flat[f"{GROUP_NAME}/P2@0.5/iou/n_defined"] == group["P2"]["0.5"]["per_object"]["defined_counts"]["iou"]
    assert flat[f"{GROUP_NAME}/P1/recall/pooled"] == group["P1"]["pooled"]["recall"]
    # First-principles pooled recall of P2@0.5: group GT cells are 0, 1 (class 1) and 2 (class 3); occupancy
    # >= 0.5 holds on cells 0 and 1 only, so 2 of 3.
    assert flat[f"{GROUP_NAME}/P2@0.5/recall/pooled"] == pytest.approx(2 / 3)
    # Per class: (1 P1 + 2 P2 thresholds) x 3 metrics x 3 key kinds + one detection rate per f_min value.
    per_class_key_count = (1 + len(P2_THRESHOLDS)) * 3 * 3 + len(DETECTION_CURVE["f_min"])
    assert len(flat) == 2 * per_class_key_count
    assert flat[f"{CLASS_NAME}/P2/detection_rate@{DETECTION_CURVE['f_min'][0]!r}"] == DETECTION_CURVE["detection_rate"][0]
    assert set(flat) >= {f"{CLASS_NAME}/P2@0.8/iou/n_defined", f"{CLASS_NAME}/P1/precision/pooled"}
    # No hidden-cell or stratum keys survive.
    assert not any("_H" in key or "hidden" in key for key in flat)


def test_flatten_quality_metrics_keys():
    """Image metrics keep psnr/ssim/lpips only (no step); lidar metrics keep their names."""
    quality = {"image": {"psnr": 20.0, "ssim": 0.5, "lpips": 0.4, "step": 30000},
               "lidar_render": {"chamfer_distance": 1.0, "median_chamfer_distance": 0.5,
                                "depth_median_l2": 0.2, "depth_mean_rel_l2": 0.1}}
    flat = flatten_quality_metrics(quality)
    assert flat["quality/image/psnr"] == 20.0 and "quality/image/step" not in flat
    assert flat["quality/lidar_render/depth_mean_rel_l2"] == 0.1 and len(flat) == 7


def test_verify_cells_accepts_consistent_and_rejects_tampered_value():
    """A report computed from the archive passes; changing one whole-scene value (or an occupancy) is refused."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_verify")
    cells, report = synthetic_archive_and_report()
    script.verify_cells_reproduce_report(report, cells, Path(RUN_NAME))
    tampered_report = copy.deepcopy(report)
    tampered_report["classes"][GROUP_NAME]["P2"]["0.5"]["pooled"]["true_positive"] += 1
    with pytest.raises(ValueError, match="P2 metrics"):
        script.verify_cells_reproduce_report(tampered_report, cells, Path(RUN_NAME))
    tampered_p1_report = copy.deepcopy(report)
    tampered_p1_report["classes"][CLASS_NAME]["P1"]["per_object"]["mean_recall"] = 0.123
    with pytest.raises(ValueError, match="P1 metrics"):
        script.verify_cells_reproduce_report(tampered_p1_report, cells, Path(RUN_NAME))
    tampered_cells = dict(cells)
    tampered_cells["occupancy"] = np.zeros_like(cells["occupancy"])
    with pytest.raises(ValueError, match="P2 metrics"):
        script.verify_cells_reproduce_report(report, tampered_cells, Path(RUN_NAME))


def test_script_20_schemas_and_scored_cells_is_shared_field():
    """Schema 3 is required and runs scored on different scored-cell builds are not pooled."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_schema")
    assert script.RUN_REPORT_SCHEMA.endswith("gaussian-occupancy/3") and script.REPORT_SCHEMA.endswith("summary/3")
    assert "scored_cells" in script.SHARED_REPORT_FIELDS and "hidden_space" not in script.SHARED_REPORT_FIELDS


def test_headline_keys_exist_in_flattened_metrics_in_order():
    """Every headline key is produced by flatten_run_metrics; pooled precedes mean; P2 leads, then P1."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_keys")
    _, report = synthetic_archive_and_report()
    flat = flatten_run_metrics(report)
    keys = script.headline_keys(GROUP_NAME, "0.5")
    assert all(key in flat for key in keys)
    assert keys[0].endswith("P2@0.5/recall/pooled") and keys[1].endswith("P2@0.5/recall/mean")
    assert keys.index(f"{GROUP_NAME}/P2@0.5/iou/mean") < keys.index(f"{GROUP_NAME}/P1/recall/pooled")
    assert keys[-1] == f"{GROUP_NAME}/P1/precision/mean" and len(keys) == 10
    assert not any("recall_blocked" in key or "hidden" in key or "_H" in key for key in keys)


def test_describe_and_compare_handle_metrics_undefined_in_some_runs():
    """A metric None in one repeat is described from the defined ones; None in every repeat gives no gap."""
    description = describe_values([0.5, None, 0.7])
    assert description["n"] == 2 and description["undefined"] == 1 and description["mean"] == pytest.approx(0.6)
    first = describe_arm([{"metric": 0.5, "all_none": None}, {"metric": None, "all_none": None},
                          {"metric": 0.7, "all_none": None}])
    second = describe_arm([{"metric": 0.1, "all_none": 0.2}, {"metric": 0.2, "all_none": 0.3}])
    comparison = compare_arms(first, second)
    assert comparison["metric"]["gap"] == pytest.approx(0.6 - 0.15)
    assert comparison["metric"]["separated"] is True
    assert comparison["all_none"]["gap"] is None and comparison["all_none"]["separated"] is None
    assert comparison["all_none"]["first_range"] is None
    # One defined repeat in an arm: no spread, so no separation claim.
    assert compare_arms(describe_arm([{"metric": 0.9}, {"metric": None}]), second)["metric"]["separated"] is None


def test_summary_markdown_lists_obstacles_first_with_guard_tables():
    """The markdown has per-class threshold tables (0.5 before 0.8), then the guard and report-only tables."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_markdown")
    _, report = synthetic_archive_and_report()
    quality = {"image": {"psnr": 20.0, "ssim": 0.5, "lpips": 0.4, "step": 30000},
               "lidar_render": {"chamfer_distance": 1.0, "median_chamfer_distance": 0.5,
                                "depth_median_l2": 0.2, "depth_mean_rel_l2": 0.1}}
    metrics = {**flatten_run_metrics(report), **flatten_quality_metrics(quality)}
    summary = {"session": "synthetic_session", "class_names": [GROUP_NAME, CLASS_NAME], "p2_thresholds": ["0.5", "0.8"],
               "arms": {"arm_a": {"train_method": "splatad", "metrics": describe_arm([metrics, metrics]),
                                  "runs": [{"run_id": "run_a", "git_sha": "0123456789", "split": "ok"}]}},
               "comparisons": []}
    markdown = script.summary_markdown(summary)
    assert markdown.index(f"## {GROUP_NAME}, P2 occupied at >= 0.5") < markdown.index(f"## {GROUP_NAME}, P2 occupied at >= 0.8")
    assert markdown.index(f"## {GROUP_NAME}, P2") < markdown.index(f"## {CLASS_NAME}, P2")
    assert markdown.index(script.GUARD_TABLE_TITLE) < markdown.index(script.LIDAR_TABLE_TITLE)
    assert "| image/psnr |" in markdown and "| lidar_render/chamfer_distance |" in markdown


def test_script_21_headline_metrics_and_schema():
    """Script 21 reads summary schema 3, lists obstacles first and expands the P2 rows per threshold."""
    script = load_script("21_compare_sessions.py", "compare_sessions_whole_scene")
    assert script.SUMMARY_SCHEMA.endswith("gaussian-occupancy-summary/3")
    assert script.HEADLINE_CLASSES == ("obstacles", "tree_trunk", "rock", "bush_or_small_tree")
    suffixes = script.headline_metric_suffixes(["0.5", "0.8"])
    assert suffixes[0] == "P2@0.5/recall/pooled" and "P2@0.8/iou/pooled" in suffixes
    assert suffixes.index("P2@0.8/iou/pooled") < suffixes.index("P1/recall/pooled")
    assert suffixes[-2:] == list(script.OPTIONAL_METRIC_SUFFIXES)
    assert not any("_H" in suffix or "hidden" in suffix for suffix in suffixes)
    _, report = synthetic_archive_and_report()
    flat = flatten_run_metrics(report)
    # Every non-optional row is a real flattened key, so the tables are never silently empty.
    required = [suffix for suffix in suffixes if suffix not in script.OPTIONAL_METRIC_SUFFIXES]
    assert all(f"{GROUP_NAME}/{suffix}" in flat for suffix in required)
