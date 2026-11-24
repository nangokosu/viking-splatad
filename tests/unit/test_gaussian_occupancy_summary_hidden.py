"""Tests of the hidden-block handling of scripts/20 and scripts/21 and of eval/p2_repeats.flatten_hidden_metrics."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval.hidden_geometry import occupied_cell_metrics
from viking_splatad.eval.hidden_metrics import class_group_masks, hidden_class_report
from viking_splatad.eval.p2_repeats import compare_arms, describe_arm, describe_values, flatten_hidden_metrics, flatten_quality_metrics

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
GROUP_CLASS_IDS = (1, 3)
OTHER_CLASS_ID = 2
P2_THRESHOLDS = [0.5, 0.8]
OCCUPANCY_THRESHOLD = 0.5
RUN_NAME = "synthetic_run"
GROUP_NAME = "obstacles"
CLASS_NAME = "tree_trunk"


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
    """Six cells, two classes (group + tree_trunk), a cells archive and the matching schema-2 report.

    Returns:
        (cells archive dict, report dict) where the report's P1/P2/hidden blocks are computed from the archive.
    """
    gt_class_ids = np.array([1, 1, 3, OTHER_CLASS_ID, 0, 0])
    neighbourhood_by_class_id = {1: np.array([1, 1, 0, 1, 1, 0], dtype=bool), 3: np.array([0, 0, 1, 1, 1, 1], dtype=bool)}
    objects = {"object_box_ids": np.array([10, 11]), "object_class_ids": np.array([1, 3]),
               "object_cell_offsets": np.array([0, 4, 8]), "object_cells": np.array([0, 1, 3, 4, 2, 3, 4, 5]),
               "object_inside": np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)}
    cells = {"occupancy": np.array([0.9, 0.9, 0.3, 0.9, 0.6, 0.9]), "p1_hit": np.array([1, 0, 0, 0, 0, 1], dtype=bool),
             "hidden": np.array([1, 0, 1, 1, 1, 0], dtype=bool),
             # H_near drops hidden cell 2 (a class-3 GT cell beyond the near band), so it differs from H.
             "hidden_near": np.array([1, 0, 0, 1, 1, 0], dtype=bool), "threshold": np.float64(OCCUPANCY_THRESHOLD),
             "evidence": np.zeros((6, 4), dtype=bool), "p2_thresholds": np.array(P2_THRESHOLDS), **objects}
    occupied = cells["occupancy"] >= OCCUPANCY_THRESHOLD
    classes = {}
    for entry_name, class_ids in ((GROUP_NAME, GROUP_CLASS_IDS), (CLASS_NAME, (1,))):
        neighbourhood, truth = class_group_masks(gt_class_ids, class_ids, neighbourhood_by_class_id)
        cells[f"neighbourhood_{entry_name}"], cells[f"truth_{entry_name}"] = neighbourhood, truth
        evidence = cells["evidence"][neighbourhood]
        classes[entry_name] = {
            "class_ids": list(class_ids),
            "P1": json.loads(json.dumps(occupied_cell_metrics(cells["p1_hit"][neighbourhood], truth[neighbourhood], evidence))),
            "P2": json.loads(json.dumps(occupied_cell_metrics(occupied[neighbourhood], truth[neighbourhood], evidence))),
            "hidden": json.loads(json.dumps(hidden_class_report(
                class_ids, neighbourhood, truth, cells["hidden"], cells["p1_hit"], cells["occupancy"], P2_THRESHOLDS, objects))),
            "hidden_near": json.loads(json.dumps(hidden_class_report(
                class_ids, neighbourhood, truth, cells["hidden_near"], cells["p1_hit"], cells["occupancy"], P2_THRESHOLDS, objects)))}
    return cells, {"classes": classes}


@pytest.mark.covers("T94")
@pytest.mark.covers("T95")
def test_flatten_hidden_metrics_keys_and_values():
    """Keys follow class/test/metric/variant; values equal the report's blocks; context counts are present."""
    _, report = synthetic_archive_and_report()
    flat = flatten_hidden_metrics(report)
    hidden = report["classes"][GROUP_NAME]["hidden"]
    assert flat[f"{GROUP_NAME}/hidden/gt_cells"] == hidden["context"]["hidden_gt_cells"]
    assert flat[f"{GROUP_NAME}/hidden/objects_with_hidden_gt"] == hidden["context"]["objects_with_hidden_gt"]
    assert flat[f"{GROUP_NAME}/P2@0.5/recall_H/pooled"] == hidden["P2"]["0.5"]["pooled"]["recall"]
    assert flat[f"{GROUP_NAME}/P2@0.8/precision_H/mean"] == hidden["P2"]["0.8"]["per_object"]["mean_precision"]
    assert flat[f"{GROUP_NAME}/P2@0.5/iou/pooled"] == hidden["P2"]["0.5"]["baseline_pooled"]["iou"]
    assert flat[f"{GROUP_NAME}/P1/recall/mean"] == hidden["P1"]["baseline_per_object"]["mean_recall"]
    assert flat[f"{GROUP_NAME}/P1/recall_H/n_defined"] == hidden["P1"]["per_object"]["defined_counts"]["recall"]
    # Full-H keys (everything not _Hnear / hidden_near): 2 classes x (2 context + 3 tests x 2 variants x 3 metrics x 3 key kinds).
    full_keys = [key for key in flat if "_Hnear" not in key and "/hidden_near/" not in key]
    assert len(full_keys) == 2 * (2 + 3 * 2 * 3 * 3)
    # H_near keys: 2 classes x (2 context + 3 tests x 1 variant (no baseline repeat) x 3 metrics x 3 key kinds).
    assert len(flat) - len(full_keys) == 2 * (2 + 3 * 1 * 3 * 3)
    assert set(flat) >= {f"{CLASS_NAME}/P2@0.8/iou_H/n_defined", f"{CLASS_NAME}/P1/precision/pooled"}


@pytest.mark.covers("T94")
@pytest.mark.covers("T95")
def test_flatten_hidden_metrics_near_keys_differ_from_full():
    """_Hnear keys come from the hidden_near block and differ from _H where the band removed a hidden cell."""
    cells, report = synthetic_archive_and_report()
    flat = flatten_hidden_metrics(report)
    near = report["classes"][GROUP_NAME]["hidden_near"]
    full = report["classes"][GROUP_NAME]["hidden"]
    assert flat[f"{GROUP_NAME}/hidden_near/gt_cells"] == near["context"]["hidden_gt_cells"]
    assert flat[f"{GROUP_NAME}/hidden_near/objects_with_hidden_gt"] == near["context"]["objects_with_hidden_gt"]
    assert flat[f"{GROUP_NAME}/P2@0.5/recall_Hnear/pooled"] == near["P2"]["0.5"]["pooled"]["recall"]
    assert flat[f"{GROUP_NAME}/P2@0.8/precision_Hnear/mean"] == near["P2"]["0.8"]["per_object"]["mean_precision"]
    assert flat[f"{GROUP_NAME}/P2@0.5/iou_Hnear/n_defined"] == near["P2"]["0.5"]["per_object"]["defined_counts"]["iou"]
    assert flat[f"{GROUP_NAME}/P1/recall_Hnear/pooled"] == near["P1"]["pooled"]["recall"]
    # First-principles check: hidden GT cells of the group are G & H (cells 0, 2) and G & H_near (cell 0 only; cell 2 is beyond the band).
    group_truth = cells[f"truth_{GROUP_NAME}"]
    assert flat[f"{GROUP_NAME}/hidden/gt_cells"] == np.count_nonzero(group_truth & cells["hidden"])
    assert flat[f"{GROUP_NAME}/hidden_near/gt_cells"] == np.count_nonzero(group_truth & cells["hidden_near"])
    assert flat[f"{GROUP_NAME}/hidden_near/gt_cells"] < flat[f"{GROUP_NAME}/hidden/gt_cells"]
    # Cell 2 (occupancy 0.3, missed at 0.5): H recall 1/2, H_near recall 1/1.
    assert full["P2"]["0.5"]["pooled"]["recall"] != near["P2"]["0.5"]["pooled"]["recall"]
    assert flat[f"{GROUP_NAME}/P2@0.5/recall_Hnear/pooled"] != flat[f"{GROUP_NAME}/P2@0.5/recall_H/pooled"]
    # The baseline is mask-independent and is not repeated for H_near.
    assert f"{GROUP_NAME}/P2@0.5/recall_Hnear/pooled" in flat and not any("Hnear" in key and key.endswith("/recall/pooled") for key in flat)


def test_flatten_quality_metrics_keys():
    """Image metrics keep psnr/ssim/lpips only (no step); lidar metrics keep their names."""
    quality = {"image": {"psnr": 20.0, "ssim": 0.5, "lpips": 0.4, "step": 30000},
               "lidar_render": {"chamfer_distance": 1.0, "median_chamfer_distance": 0.5,
                                "depth_median_l2": 0.2, "depth_mean_rel_l2": 0.1}}
    flat = flatten_quality_metrics(quality)
    assert flat["quality/image/psnr"] == 20.0 and "quality/image/step" not in flat
    assert flat["quality/lidar_render/depth_mean_rel_l2"] == 0.1 and len(flat) == 7


def test_verify_cells_accepts_consistent_and_rejects_tampered_hidden_value():
    """A report computed from the archive passes; changing one hidden value (or the hidden flags) is refused."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_hidden")
    cells, report = synthetic_archive_and_report()
    script.verify_cells_reproduce_report(report, cells, Path(RUN_NAME))
    tampered_report = copy.deepcopy(report)
    tampered_report["classes"][GROUP_NAME]["hidden"]["P2"]["0.5"]["pooled"]["true_positive"] += 1
    with pytest.raises(ValueError, match="hidden metrics"):
        script.verify_cells_reproduce_report(tampered_report, cells, Path(RUN_NAME))
    tampered_cells = dict(cells)
    tampered_cells["hidden"] = ~cells["hidden"]
    with pytest.raises(ValueError, match="hidden metrics"):
        script.verify_cells_reproduce_report(report, tampered_cells, Path(RUN_NAME))


def test_verify_cells_recomputes_hidden_near_block():
    """The H_near block is recomputed too: a tampered value, a tampered mask or a missing block is refused."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_hidden_near_verify")
    cells, report = synthetic_archive_and_report()
    tampered_report = copy.deepcopy(report)
    tampered_report["classes"][GROUP_NAME]["hidden_near"]["P2"]["0.5"]["pooled"]["true_positive"] += 1
    with pytest.raises(ValueError, match="hidden_near metrics"):
        script.verify_cells_reproduce_report(tampered_report, cells, Path(RUN_NAME))
    tampered_cells = dict(cells)
    tampered_cells["hidden_near"] = cells["hidden"]
    with pytest.raises(ValueError, match="hidden_near metrics"):
        script.verify_cells_reproduce_report(report, tampered_cells, Path(RUN_NAME))
    archive_without_near = {key: value for key, value in cells.items() if key != "hidden_near"}
    with pytest.raises(ValueError, match="hidden_near"):
        script.verify_cells_reproduce_report(report, archive_without_near, Path(RUN_NAME))


def test_script_20_schemas_and_hidden_space_is_shared_field():
    """Schema 2 is required and runs scored on different hidden-space builds are not pooled."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_hidden_schema")
    assert script.RUN_REPORT_SCHEMA.endswith("gaussian-occupancy/2") and script.REPORT_SCHEMA.endswith("summary/2")
    assert "hidden_space" in script.SHARED_REPORT_FIELDS


def test_headline_keys_exist_in_flattened_metrics_in_order():
    """Every headline key is produced by flatten_hidden_metrics; pooled precedes mean; recall_blocked is gone."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_hidden_keys")
    _, report = synthetic_archive_and_report()
    flat = flatten_hidden_metrics(report)
    keys = script.headline_keys(GROUP_NAME, "0.5")
    assert all(key in flat for key in keys)
    # H_near (headline) leads; the full-H rows follow the H_near ones.
    assert keys[0].endswith("P2@0.5/recall_Hnear/pooled") and keys[1].endswith("P2@0.5/recall_Hnear/mean")
    assert keys.index(f"{GROUP_NAME}/hidden_near/gt_cells") < keys.index(f"{GROUP_NAME}/P2@0.5/recall_H/pooled")
    assert f"{GROUP_NAME}/P1/recall_Hnear/pooled" in keys and f"{GROUP_NAME}/hidden/gt_cells" in keys
    assert not any("recall_blocked" in key for key in keys)


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
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_hidden_markdown")
    _, report = synthetic_archive_and_report()
    quality = {"image": {"psnr": 20.0, "ssim": 0.5, "lpips": 0.4, "step": 30000},
               "lidar_render": {"chamfer_distance": 1.0, "median_chamfer_distance": 0.5,
                                "depth_median_l2": 0.2, "depth_mean_rel_l2": 0.1}}
    metrics = {**flatten_hidden_metrics(report), **flatten_quality_metrics(quality)}
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
    """Script 21 reads summary schema 2, lists obstacles first and expands the P2 rows per threshold."""
    script = load_script("21_compare_sessions.py", "compare_sessions_hidden")
    assert script.SUMMARY_SCHEMA.endswith("gaussian-occupancy-summary/2")
    assert script.HEADLINE_CLASSES == ("obstacles", "tree_trunk", "rock", "bush_or_small_tree")
    suffixes = script.headline_metric_suffixes(["0.5", "0.8"])
    assert suffixes[0] == "P2@0.5/recall_Hnear/pooled" and "P2@0.8/iou_H/mean" in suffixes and "P2@0.8/iou_Hnear/mean" in suffixes
    assert suffixes.index("P2@0.5/iou_Hnear/mean") < suffixes.index("P2@0.5/recall_H/pooled")
    assert "P1/recall_Hnear/pooled" in suffixes
    assert "P1/recall_H/pooled" in suffixes and suffixes[-2:] == list(script.OPTIONAL_METRIC_SUFFIXES)
    assert suffixes.index("P2@0.5/iou/pooled") < suffixes.index("P2@0.8/recall_H/pooled")
    _, report = synthetic_archive_and_report()
    flat = flatten_hidden_metrics(report)
    # Every non-optional row is a real flattened key, so the tables are never silently empty.
    required = [suffix for suffix in suffixes if suffix not in script.OPTIONAL_METRIC_SUFFIXES]
    assert all(f"{GROUP_NAME}/{suffix}" in flat for suffix in required)
