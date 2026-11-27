"""Tests of the whole-scene report logic of scripts/19_eval_gaussian_occupancy.py (pure parts, synthetic cells)."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from viking_splatad.eval.occupancy_settings import parse_occupancy_settings
from viking_splatad.eval.scene_metrics import OBJECT_ARRAY_KEYS, class_group_masks, select_object_lists

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "19_eval_gaussian_occupancy.py"
# Synthetic class ids: 1 and 3 form the group, 2 is an annotated class outside it.
GROUP_CLASS_IDS = (1, 3)
OTHER_CLASS_ID = 2
SESSION_NAME = "synthetic_session"
VOXEL_SIZE_M = 0.5
# Minimal eval section accepted by parse_occupancy_settings (values only need to be valid, not realistic).
EVAL_SECTION = {
    "render_accumulation_threshold": 0.5, "p2_thresholds": [0.5, 0.8],
    "scored_class_ids": {"tree_trunk": 1, "other": OTHER_CLASS_ID, "rock": 3},
    "class_groups": {"obstacles": [1, 3]},
    "scored_cells": {"scored_range_quantile": 0.999},
}
# Keys of one class entry of the schema-3 report, and of the cells archive scripts/20 and scripts/15 read.
CLASS_ENTRY_KEYS = {"class_id", "class_ids", "scored_cuboids", "margin_m", "scored_cells", "ground_truth_cells", "P1", "P2",
                    "detection", "threshold_sweep"}
CELLS_ARCHIVE_KEYS = {"cell_indices", "centres", "occupancy", "p1_hit", "threshold", "grid_origin", "voxel_size",
                      "gt_class_ids", "p2_thresholds", *OBJECT_ARRAY_KEYS}
BLOCK_KEYS = {"pooled", "per_object"}


def load_script():
    """Import scripts/19 as a module.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("eval_gaussian_occupancy_report", SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def synthetic_cells():
    """Six cells: GT labels, two class neighbourhoods, probabilities, P1 hits and two objects.

    Cell 3 carries the other annotated class (label 2) and lies in both neighbourhoods.

    Returns:
        dict of arrays and the objects dict (CSR lists of one object per class).
    """
    gt_class_ids = np.array([1, 1, 3, OTHER_CLASS_ID, 0, 0])
    neighbourhood_by_class_id = {1: np.array([1, 1, 0, 1, 1, 0], dtype=bool), 3: np.array([0, 0, 1, 1, 1, 1], dtype=bool)}
    occupancy = np.array([0.9, 0.9, 0.3, 0.9, 0.6, 0.9])
    p1_hit = np.array([1, 0, 0, 0, 0, 1], dtype=bool)
    # Object 0 (class 1): grown box covers cells 0,1,3,4, centre inside only for 0,1. Object 1 (class 3): cells 2..5.
    objects = {"object_box_ids": np.array([10, 11]), "object_class_ids": np.array([1, 3]),
               "object_cell_offsets": np.array([0, 4, 8]), "object_cells": np.array([0, 1, 3, 4, 2, 3, 4, 5]),
               "object_inside": np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)}
    return {"gt": gt_class_ids, "neighbourhoods": neighbourhood_by_class_id, "occupancy": occupancy,
            "p1_hit": p1_hit, "objects": objects}


def synthetic_class_report(script, entry_ids, p2_thresholds=(0.5, 0.8)):
    """The whole-scene metric blocks of one entry on the synthetic cells.

    Args:
        script: Loaded scripts/19 module.
        entry_ids: Class ids of the entry.
        p2_thresholds: P2 cuts.

    Returns:
        (class_metric_report output, neighbourhood, truth, cells).
    """
    cells = synthetic_cells()
    neighbourhood, truth = class_group_masks(cells["gt"], entry_ids, cells["neighbourhoods"])
    report = script.class_metric_report(entry_ids, neighbourhood, truth, cells["p1_hit"], cells["occupancy"],
                                        list(p2_thresholds), cells["objects"])
    return report, neighbourhood, truth, cells


@pytest.mark.covers("T94")
@pytest.mark.covers("T95")
def test_group_report_applies_other_class_mask_and_threshold_keys():
    """Group N excludes the other-class cell; per-threshold keys are "0.5" and "0.8"; values match set algebra."""
    script = load_script()
    report, neighbourhood, truth, cells = synthetic_class_report(script, GROUP_CLASS_IDS)
    assert list(report["P2"]) == ["0.5", "0.8"]
    for threshold in (0.5, 0.8):
        occupied = cells["occupancy"] >= threshold
        # Expected values from the set definitions: |O&G| / |G| and |O&G| / |O&N| (N excludes cell 3).
        true_positive = np.count_nonzero(occupied & truth)
        pooled = report["P2"][str(float(threshold))]["pooled"]
        assert pooled["recall"] == pytest.approx(true_positive / np.count_nonzero(truth))
        assert pooled["occupied_cells"] == np.count_nonzero(occupied & neighbourhood)
        assert pooled["precision"] == pytest.approx(true_positive / np.count_nonzero(occupied & neighbourhood))
    # The other-class cell (3) is occupied at 0.5 but must count for neither the group's N nor its precision.
    assert not neighbourhood[3] and not truth[3]
    assert report["P2"]["0.5"]["pooled"]["occupied_cells"] == 4
    # P1 block has the same two keys and no per-object list in the JSON.
    assert set(report["P1"]) == BLOCK_KEYS and set(report["P2"]["0.5"]) == BLOCK_KEYS
    assert "per_object" not in report["P1"]["per_object"] and "per_object" not in report["P2"]["0.5"]["per_object"]
    json.dumps(report)


@pytest.mark.covers("T95")
def test_object_selection():
    """A group keeps both objects, a single class only its own; the CSR selection slices the right cells."""
    script = load_script()
    group_report, _, _, cells = synthetic_class_report(script, GROUP_CLASS_IDS, [0.5])
    assert group_report["P2"]["0.5"]["per_object"]["object_count"] == 2
    class_report, _, _, _ = synthetic_class_report(script, (1,), [0.5])
    assert class_report["P2"]["0.5"]["per_object"]["object_count"] == 1
    offsets, selected_cells, selected_inside = select_object_lists(
        cells["objects"]["object_class_ids"], cells["objects"]["object_cell_offsets"], cells["objects"]["object_cells"],
        cells["objects"]["object_inside"], (3,))
    assert offsets.tolist() == [0, 4] and selected_cells.tolist() == [2, 3, 4, 5] and selected_inside.tolist() == [True, False, False, False]


def test_entries_list_group_first_and_group_setup_is_union():
    """class_entries puts ``obstacles`` before the classes; its setup has class_id None and the members' cuboids."""
    script = load_script()
    settings = parse_occupancy_settings(EVAL_SECTION)
    entries = script.class_entries(settings, EVAL_SECTION["scored_class_ids"])
    assert list(entries) == ["obstacles", "tree_trunk", "other", "rock"]
    assert entries["obstacles"] == (1, 3) and entries["rock"] == (3,)
    cuboids = [SimpleNamespace(box_id=10, class_id=1), SimpleNamespace(box_id=11, class_id=3),
               SimpleNamespace(box_id=12, class_id=3), SimpleNamespace(box_id=13, class_id=OTHER_CLASS_ID)]
    objects = synthetic_cells()["objects"]
    group_setup = script.entry_setup((1, 3), True, cuboids, objects, {1: 0.4, 3: 0.2})
    # Box 12 is not a scored-cells object, so it is not scored; box 13 is another class.
    assert [cuboid.box_id for cuboid in group_setup["cuboids"]] == [10, 11]
    assert group_setup["class_id"] is None and group_setup["class_ids"] == [1, 3] and group_setup["margin"] is None
    assert script.entry_setup((3,), False, cuboids, objects, {1: 0.4, 3: 0.2})["margin"] == 0.2


def test_markdown_lists_obstacles_first_with_threshold_tables():
    """The tables come with one section per threshold, in report order (obstacles first), P2 rows then P1 rows."""
    script = load_script()
    classes = {}
    for entry_name, entry_ids in (("obstacles", GROUP_CLASS_IDS), ("tree_trunk", (1,))):
        classes[entry_name] = {**synthetic_class_report(script, entry_ids)[0], "detection": {"curve": {"0.0": 1.0}}}
    report = {"run_id": "r", "profile": "p", "train_method": "m", "git_sha": "g", "gaussians": {"count": 1, "checkpoint_step": 0},
              "settings": {"occupancy_threshold": 0.5}, "scored_cell_count": 6, "classes": classes}
    text = script.report_markdown(report)
    assert text.index("### obstacles, P2 occupied at >= 0.5") < text.index("### obstacles, P2 occupied at >= 0.8") < text.index("### tree_trunk, P2 occupied at >= 0.5")
    for row_label in ("P2 recall", "P2 precision", "P2 IoU", "P1 recall", "P1 precision", "P1 IoU", "Detection rate over f_min"):
        assert row_label in text
    # Table values come from the blocks: obstacles P2 pooled recall at 0.5 is 2 of 3 truth cells.
    assert "| P2 recall | 0.667 |" in text


def test_viz_command_draws_only_the_p2_picture():
    """The picture command carries --p2 after --run and the output dir, and no other picture flag."""
    script = load_script()
    command = script.p2_viz_command(Path("/r"), Path("/o"), Path("/c.yaml"))
    assert command[command.index("--run") + 1] == "/r" and "--p2" in command
    assert command[command.index("--output-dir") + 1] == "/o"
    assert not [argument for argument in command if argument.startswith("--p2-")]


def sha256_of(path):
    """Hex sha256 of a file (independent of the module under test).

    Args:
        path: File.

    Returns:
        Digest text.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_synthetic_project(tmp_path):
    """A repo-like folder with GT files, a reference split record and consistent scored cells.

    Args:
        tmp_path: pytest temporary folder.

    Returns:
        (project_root, gt_dir, inputs dict as ``load_run_inputs`` provides the used keys).
    """
    project_root = tmp_path / "project"
    gt_dir = project_root / "data" / "gt" / SESSION_NAME
    split_dir = project_root / "configs" / "reference_splits"
    gt_dir.mkdir(parents=True), split_dir.mkdir(parents=True)
    (gt_dir / "cuboids.json").write_text("[]")
    grid_origin = np.array([1.0, 2.0, 3.0])
    np.savez(gt_dir / "occupancy.npz", class_ids=np.zeros((2, 2, 2), dtype=np.uint8), origin=grid_origin, voxel_size=VOXEL_SIZE_M)
    split_path = split_dir / f"{SESSION_NAME}.json"
    split_path.write_text("{}")
    cells = synthetic_cells()
    cell_count = len(cells["gt"])
    arrays = {"cell_indices": np.zeros((cell_count, 3), dtype=np.int64), "centres": np.zeros((cell_count, 3)),
              "gt_class_ids": cells["gt"].astype(np.uint8),
              "neighbourhood_tree_trunk": cells["neighbourhoods"][1], "neighbourhood_other": np.zeros(cell_count, dtype=bool),
              "neighbourhood_rock": cells["neighbourhoods"][3],
              **cells["objects"], "voxel_size": np.float64(VOXEL_SIZE_M), "grid_origin": grid_origin}
    np.savez_compressed(gt_dir / "scored_cells.npz", **arrays)
    input_hashes = {f"data/gt/{SESSION_NAME}/cuboids.json": sha256_of(gt_dir / "cuboids.json"),
                    f"data/gt/{SESSION_NAME}/occupancy.npz": sha256_of(gt_dir / "occupancy.npz"),
                    f"configs/reference_splits/{SESSION_NAME}.json": sha256_of(split_path)}
    record = {"schema": "viking-splatad/scored-cells/1", "debug_max_sweeps": None, "scored_range_m": 40.0,
              "scored_range_quantile": 0.999, "max_range_m": 200.0, "vertical_half_fov_rad": 0.2, "input_sha256": input_hashes,
              "output_npz": {"path": "x", "sha256": sha256_of(gt_dir / "scored_cells.npz")}}
    (gt_dir / "scored_cells.json").write_text(json.dumps(record))
    inputs = {"session_config": {"session": {"name": SESSION_NAME}},
              "grid": {"voxel_size": VOXEL_SIZE_M, "origin": grid_origin}}
    return project_root, gt_dir, inputs


def test_loader_accepts_matching_scored_cells(tmp_path):
    """Scored cells built from the very files in use load, with their scored range (not max_range_m)."""
    script = load_script()
    project_root, gt_dir, inputs = build_synthetic_project(tmp_path)
    loaded = script.load_session_scored_cells(gt_dir, inputs, project_root, {"tree_trunk": 1})
    assert loaded["scored_range_m"] == 40.0 and loaded["scored_range_quantile"] == 0.999
    assert loaded["neighbourhood_by_class_name"]["tree_trunk"].tolist() == [True, True, False, True, True, False]
    assert set(loaded["objects"]) == set(OBJECT_ARRAY_KEYS)
    assert loaded["sha256"] == sha256_of(gt_dir / "scored_cells.npz")


def test_loader_refuses_missing_and_mismatched_inputs(tmp_path):
    """Missing scored cells, a changed cuboids/split file, a changed npz or another grid are refused."""
    script = load_script()
    project_root, gt_dir, inputs = build_synthetic_project(tmp_path)
    classes = {"tree_trunk": 1}
    # Another voxel size than the GT grid.
    with pytest.raises(ValueError, match="voxel_size"):
        script.load_session_scored_cells(gt_dir, {**inputs, "grid": {**inputs["grid"], "voxel_size": 2 * VOXEL_SIZE_M}}, project_root, classes)
    # Each hashed input changed after the build.
    for relative_path in (Path("data/gt") / SESSION_NAME / "cuboids.json", Path("configs/reference_splits") / f"{SESSION_NAME}.json"):
        original = (project_root / relative_path).read_bytes()
        (project_root / relative_path).write_text("changed")
        with pytest.raises(ValueError, match="scripts/24"):
            script.load_session_scored_cells(gt_dir, inputs, project_root, classes)
        (project_root / relative_path).write_bytes(original)
    # The recorded npz hash no longer matches the file.
    record = json.loads((gt_dir / "scored_cells.json").read_text())
    record["output_npz"]["sha256"] = "0" * 64
    (gt_dir / "scored_cells.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scored_cells.npz"):
        script.load_session_scored_cells(gt_dir, inputs, project_root, classes)
    # No scored cells at all.
    (gt_dir / "scored_cells.npz").unlink()
    with pytest.raises(ValueError, match="scripts/24"):
        script.load_session_scored_cells(gt_dir, inputs, project_root, classes)


def test_loader_refuses_wrong_schema_and_debug_build(tmp_path):
    """A record of another schema or a debug build is refused with the scripts/24 hint."""
    script = load_script()
    project_root, gt_dir, inputs = build_synthetic_project(tmp_path)
    original_record = json.loads((gt_dir / "scored_cells.json").read_text())
    for field_name, bad_value, message in (("schema", "viking-splatad/scored-cells/0", "schema"), ("debug_max_sweeps", 5, "debug build")):
        (gt_dir / "scored_cells.json").write_text(json.dumps({**original_record, field_name: bad_value}))
        with pytest.raises(ValueError, match=message) as refusal:
            script.load_session_scored_cells(gt_dir, inputs, project_root, {"tree_trunk": 1})
        assert "scripts/24" in str(refusal.value)


@pytest.mark.covers("T94")
@pytest.mark.covers("T95")
def test_evaluate_run_report_and_cells_archive_layout(tmp_path, monkeypatch):
    """Schema-3 layout: exact class-entry keys, settings, scored_cells provenance and the cells archive key set."""
    script = load_script()
    project_root, gt_dir, inputs = build_synthetic_project(tmp_path)
    cells = synthetic_cells()
    cell_count = len(cells["gt"])
    run_inputs = {
        **inputs, "job_info": {"MAX_NUM_ITERATIONS": "11"}, "threshold": 0.5, "gt_dir": gt_dir, "profile": "profile",
        "split_record": {"run": {"run_id": "run", "git_sha": "sha"}, "dataset": {"name": "dataset"}, "membership_sha256": "m"},
        "cuboids": [SimpleNamespace(box_id=10, class_id=1), SimpleNamespace(box_id=11, class_id=3)],
        "grid": {**inputs["grid"], "class_ids": np.zeros((2, 2, 2), dtype=np.uint8)},
        "sweeps": [], "eval_root": tmp_path / "run"}
    factor_arrays = {"means": np.zeros((3, 3)), "opacities": np.zeros(3), "principal_axes": np.zeros((3, 3, 3)),
                     "axis_std": np.zeros((3, 3)), "checkpoint_step": np.int64(10)}
    monkeypatch.setattr(script, "load_run_inputs", lambda *arguments: run_inputs)
    monkeypatch.setattr(script, "gaussian_factors", lambda *arguments: {"arrays": factor_arrays, "sha256": "f"})
    monkeypatch.setattr(script, "final_train_directory", lambda run_dir: tmp_path / "train" / "method" / "run")
    monkeypatch.setattr(script, "p1_hit_cells", lambda *arguments: cells["p1_hit"])
    monkeypatch.setattr(script, "peak_alpha_voxel_occupancy", lambda *arguments: cells["occupancy"])
    monkeypatch.setattr(script, "class_margin_table", lambda *arguments: {1: 0.4, 2: 0.3, 3: 0.2})
    monkeypatch.setattr(script, "cuboid_occupied_fractions", lambda detected_cuboids, *arguments: {
        "gt_cells": np.ones(len(detected_cuboids), dtype=np.int64), "occupied_cells": np.ones(len(detected_cuboids), dtype=np.int64),
        "fraction": np.ones(len(detected_cuboids))})
    eval_settings = {**EVAL_SECTION, "gaussian_occupancy_chunk_size": 4, "p2_f_min_grid_points": 3}
    result = script.evaluate_run(tmp_path / "run", eval_settings, "configs/eval.yaml", project_root)
    report, archive = result["report"], result["cells"]
    assert report["schema"] == "viking-splatad/gaussian-occupancy/3"
    assert list(report["classes"]) == ["obstacles", "tree_trunk", "other", "rock"]
    for class_entry in report["classes"].values():
        assert set(class_entry) == CLASS_ENTRY_KEYS
        assert list(class_entry["P2"]) == ["0.5", "0.8"] and set(class_entry["P1"]) == BLOCK_KEYS
        assert set(class_entry["detection"]) == {"box_ids", "gt_cells", "occupied_cells", "curve"}
    assert report["settings"]["scored_range_m"] == 40.0 and report["settings"]["scored_range_quantile"] == 0.999
    assert "max_range_m" not in report["settings"]
    assert report["scored_cells"] == {"path": f"data/gt/{SESSION_NAME}/scored_cells.npz", "sha256": sha256_of(gt_dir / "scored_cells.npz")}
    # The headline whole-scene recall equals the set-algebra value on the synthetic cells.
    group_neighbourhood, group_truth = class_group_masks(cells["gt"], GROUP_CLASS_IDS, cells["neighbourhoods"])
    expected_recall = np.count_nonzero((cells["occupancy"] >= 0.5) & group_truth) / np.count_nonzero(group_truth)
    assert report["classes"]["obstacles"]["P2"]["0.5"]["pooled"]["recall"] == pytest.approx(expected_recall)
    entry_names = ["obstacles", "tree_trunk", "other", "rock"]
    expected_archive_keys = CELLS_ARCHIVE_KEYS | {f"neighbourhood_{name}" for name in entry_names} | {f"truth_{name}" for name in entry_names}
    assert set(archive) == expected_archive_keys
    assert len(archive["occupancy"]) == cell_count and archive["p1_hit"].dtype == bool
    json.dumps(report)
