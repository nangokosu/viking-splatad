"""Tests of the hidden-voxel report logic of scripts/19_eval_gaussian_occupancy.py (pure parts, synthetic cells)."""

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from viking_splatad.eval.hidden_metrics import class_group_masks
from viking_splatad.eval.hidden_settings import parse_hidden_settings

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "19_eval_gaussian_occupancy.py"
# Synthetic class ids: 1 and 3 form the group, 2 is an annotated class outside it.
GROUP_CLASS_IDS = (1, 3)
OTHER_CLASS_ID = 2
SESSION_NAME = "synthetic_session"
VOXEL_SIZE_M = 0.5
# Minimal eval section accepted by parse_hidden_settings (values only need to be valid, not realistic).
EVAL_SECTION = {
    "render_accumulation_threshold": 0.5, "p2_thresholds": [0.5, 0.8],
    "scored_class_ids": {"tree_trunk": 1, "other": OTHER_CLASS_ID, "rock": 3},
    "class_groups": {"obstacles": [1, 3]},
    "hidden_space": {"margin_quantile": 0.9, "margin_sweep_gap": 1, "margin_sweep_stride": 1, "margin_neighbour_count": 5,
                     "margin_planarity_min": 0.5, "margin_max_range_m": 10.0, "scored_range_quantile": 0.999, "near_band_m": 20.0,
                     "worker_count": None},
}


def load_script():
    """Import scripts/19 as a module.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("eval_gaussian_occupancy_hidden", SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def synthetic_cells():
    """Six cells: GT labels, two class neighbourhoods, hidden flags, probabilities and two objects.

    Cell 3 carries the other annotated class (label 2) and lies in both neighbourhoods.

    Returns:
        dict of arrays and the objects dict (CSR lists of one object per class).
    """
    gt_class_ids = np.array([1, 1, 3, OTHER_CLASS_ID, 0, 0])
    neighbourhood_by_class_id = {1: np.array([1, 1, 0, 1, 1, 0], dtype=bool), 3: np.array([0, 0, 1, 1, 1, 1], dtype=bool)}
    hidden = np.array([1, 0, 1, 1, 1, 0], dtype=bool)
    occupancy = np.array([0.9, 0.9, 0.3, 0.9, 0.6, 0.9])
    p1_hit = np.array([1, 0, 0, 0, 0, 1], dtype=bool)
    # Object 0 (class 1): grown box covers cells 0,1,3,4, centre inside only for 0,1. Object 1 (class 3): cells 2..5.
    objects = {"object_box_ids": np.array([10, 11]), "object_class_ids": np.array([1, 3]),
               "object_cell_offsets": np.array([0, 4, 8]), "object_cells": np.array([0, 1, 3, 4, 2, 3, 4, 5]),
               "object_inside": np.array([1, 1, 0, 0, 1, 0, 0, 0], dtype=bool)}
    return {"gt": gt_class_ids, "neighbourhoods": neighbourhood_by_class_id, "hidden": hidden, "occupancy": occupancy,
            "p1_hit": p1_hit, "objects": objects}


@pytest.mark.covers("T94")
@pytest.mark.covers("T95")
def test_group_report_applies_other_class_mask_and_threshold_keys():
    """Group N excludes the other-class cell; per-threshold keys are "0.5" and "0.8"; values match set algebra."""
    script = load_script()
    cells = synthetic_cells()
    neighbourhood, truth = class_group_masks(cells["gt"], GROUP_CLASS_IDS, cells["neighbourhoods"])
    report = script.hidden_class_report(GROUP_CLASS_IDS, neighbourhood, truth, cells["hidden"], cells["p1_hit"],
                                        cells["occupancy"], [0.5, 0.8], cells["objects"])
    assert list(report["P2"]) == ["0.5", "0.8"]
    for threshold in (0.5, 0.8):
        occupied = cells["occupancy"] >= threshold
        hidden = cells["hidden"]
        # Expected values from the set definitions: |O&G&H| / |G&H| and |O&G&H| / |O&N&H| (N excludes cell 3).
        true_positive = np.count_nonzero(occupied & truth & hidden)
        pooled = report["P2"][script.threshold_key(threshold)]["pooled"]
        assert pooled["recall"] == pytest.approx(true_positive / np.count_nonzero(truth & hidden))
        assert pooled["occupied_cells"] == np.count_nonzero(occupied & neighbourhood & hidden)
        assert pooled["precision"] == pytest.approx(true_positive / np.count_nonzero(occupied & neighbourhood & hidden))
        baseline = report["P2"][script.threshold_key(threshold)]["baseline_pooled"]
        assert baseline["recall"] == pytest.approx(np.count_nonzero(occupied & truth) / np.count_nonzero(truth))
    # The other-class cell (3) is occupied at 0.5 but must count for neither the group's N nor its precision.
    assert not neighbourhood[3] and not truth[3]
    assert report["P2"]["0.5"]["pooled"]["occupied_cells"] == 2
    # P1 block exists with the same four keys and no per-object list in the JSON.
    assert set(report["P1"]) == {"pooled", "per_object", "baseline_pooled", "baseline_per_object"}
    assert "per_object" not in report["P1"]["per_object"] and "per_object" not in report["P2"]["0.5"]["per_object"]
    json.dumps(report)


@pytest.mark.covers("T95")
def test_context_counts_and_object_selection():
    """Context = |G&H|, objects with hidden GT; a single class keeps only its own objects."""
    script = load_script()
    cells = synthetic_cells()
    neighbourhood, truth = class_group_masks(cells["gt"], GROUP_CLASS_IDS, cells["neighbourhoods"])
    group_report = script.hidden_class_report(GROUP_CLASS_IDS, neighbourhood, truth, cells["hidden"], cells["p1_hit"],
                                              cells["occupancy"], [0.5], cells["objects"])
    assert group_report["context"] == {"hidden_gt_cells": int(np.count_nonzero(truth & cells["hidden"])),
                                       "objects_with_hidden_gt": 2, "object_count": 2}
    class_neighbourhood, class_truth = class_group_masks(cells["gt"], (1,), cells["neighbourhoods"])
    class_report = script.hidden_class_report((1,), class_neighbourhood, class_truth, cells["hidden"], cells["p1_hit"],
                                              cells["occupancy"], [0.5], cells["objects"])
    assert class_report["context"]["object_count"] == 1
    offsets, selected_cells, selected_inside = script.select_object_lists(
        cells["objects"]["object_class_ids"], cells["objects"]["object_cell_offsets"], cells["objects"]["object_cells"],
        cells["objects"]["object_inside"], (3,))
    assert offsets.tolist() == [0, 4] and selected_cells.tolist() == [2, 3, 4, 5] and selected_inside.tolist() == [True, False, False, False]


def test_entries_list_group_first_and_group_setup_is_union():
    """class_entries puts ``obstacles`` before the classes; its setup has class_id None and the members' cuboids."""
    script = load_script()
    settings = parse_hidden_settings(EVAL_SECTION)
    entries = script.class_entries(settings, EVAL_SECTION["scored_class_ids"])
    assert list(entries) == ["obstacles", "tree_trunk", "other", "rock"]
    assert entries["obstacles"] == (1, 3) and entries["rock"] == (3,)
    cuboids = [SimpleNamespace(box_id=10, class_id=1), SimpleNamespace(box_id=11, class_id=3),
               SimpleNamespace(box_id=12, class_id=3), SimpleNamespace(box_id=13, class_id=OTHER_CLASS_ID)]
    objects = synthetic_cells()["objects"]
    group_setup = script.entry_setup((1, 3), True, cuboids, objects, {1: 0.4, 3: 0.2})
    # Box 12 is not a hidden-space object, so it is not scored; box 13 is another class.
    assert [cuboid.box_id for cuboid in group_setup["cuboids"]] == [10, 11]
    assert group_setup["class_id"] is None and group_setup["class_ids"] == [1, 3] and group_setup["margin"] is None
    assert script.entry_setup((3,), False, cuboids, objects, {1: 0.4, 3: 0.2})["margin"] == 0.2


def test_markdown_lists_obstacles_first_with_threshold_tables():
    """The hidden tables come with one section per threshold, in report order (obstacles first)."""
    script = load_script()
    cells = synthetic_cells()
    blocks = {}
    for entry_name, entry_ids in (("obstacles", GROUP_CLASS_IDS), ("tree_trunk", (1,))):
        neighbourhood, truth = class_group_masks(cells["gt"], entry_ids, cells["neighbourhoods"])
        blocks[entry_name] = script.hidden_class_report(entry_ids, neighbourhood, truth, cells["hidden"], cells["p1_hit"],
                                                        cells["occupancy"], [0.5, 0.8], cells["objects"])
    lines = []
    for entry_name, block in blocks.items():
        for threshold_text in block["P2"]:
            lines += script.hidden_table_lines(entry_name, block, threshold_text)
    text = "\n".join(lines)
    assert text.index("### obstacles, P2 occupied at >= 0.5") < text.index("### obstacles, P2 occupied at >= 0.8") < text.index("### tree_trunk, P2 occupied at >= 0.5")
    for row_label in ("P2 recall_H", "P2 precision_H", "P2 IoU_H", "P2 recall (baseline)", "P1 recall_H", "Hidden GT cells"):
        assert row_label in text


def test_hidden_viz_command_carries_flags():
    """The hidden picture command has the --p2-hidden flags after --run and the same output dir."""
    script = load_script()
    command = script.p2_viz_command(Path("/r"), Path("/o"), Path("/c.yaml"), script.hidden_viz_arguments("obstacles", 0.5))
    assert command[command.index("--p2-hidden"):command.index("--p2-hidden") + 5] == [
        "--p2-hidden", "--hidden-class", "obstacles", "--hidden-threshold", "0.5"]
    assert command[command.index("--output-dir") + 1] == "/o"
    assert "--p2" in script.p2_viz_command(Path("/r"), Path("/o"), Path("/c.yaml"))


def sha256_of(path):
    """Hex sha256 of a file (independent of the module under test).

    Args:
        path: File.

    Returns:
        Digest text.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_synthetic_project(tmp_path):
    """A repo-like folder with GT files, a reference split record and a consistent hidden space.

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
    arrays = {"cell_indices": np.zeros((3, 3), dtype=np.int64), "centres": np.zeros((3, 3)), "evidence": np.zeros((3, 4), dtype=bool),
              "hidden": np.array([1, 0, 1], dtype=bool), "hidden_near": np.array([1, 0, 0], dtype=bool), "gt_class_ids": np.zeros(3, dtype=np.uint8),
              "neighbourhood_tree_trunk": np.ones(3, dtype=bool), "object_box_ids": np.zeros(0), "object_class_ids": np.zeros(0),
              "object_cell_offsets": np.zeros(1, dtype=np.int64), "object_cells": np.zeros(0, dtype=np.int64),
              "object_inside": np.zeros(0, dtype=bool), "voxel_size": np.float64(VOXEL_SIZE_M), "grid_origin": grid_origin,
              "range_margin_m": np.float64(0.25)}
    np.savez_compressed(gt_dir / "hidden_space.npz", **arrays)
    input_hashes = {f"data/gt/{SESSION_NAME}/cuboids.json": sha256_of(gt_dir / "cuboids.json"),
                    f"data/gt/{SESSION_NAME}/occupancy.npz": sha256_of(gt_dir / "occupancy.npz"),
                    f"configs/reference_splits/{SESSION_NAME}.json": sha256_of(split_path)}
    record = {"schema": "viking-splatad/hidden-space/1", "debug_max_sweeps": None, "max_range_m": 40.0,
              "vertical_half_fov_rad": 0.2, "near_band_m": 20.0, "input_sha256": input_hashes,
              "output_npz": {"path": "x", "sha256": sha256_of(gt_dir / "hidden_space.npz")}}
    (gt_dir / "hidden_space.json").write_text(json.dumps(record))
    inputs = {"session_config": {"session": {"name": SESSION_NAME}},
              "grid": {"voxel_size": VOXEL_SIZE_M, "origin": grid_origin}}
    return project_root, gt_dir, inputs


def test_loader_accepts_matching_hidden_space(tmp_path):
    """A hidden space built from the very files in use loads, with its margin and range."""
    script = load_script()
    project_root, gt_dir, inputs = build_synthetic_project(tmp_path)
    loaded = script.load_session_hidden_space(gt_dir, inputs, project_root, {"tree_trunk": 1})
    assert loaded["range_margin_m"] == 0.25 and loaded["max_range_m"] == 40.0
    assert loaded["hidden"].tolist() == [True, False, True]
    assert loaded["hidden_near"].tolist() == [True, False, False] and loaded["near_band_m"] == 20.0
    assert loaded["sha256"] == sha256_of(gt_dir / "hidden_space.npz")


def test_loader_refuses_missing_and_mismatched_inputs(tmp_path):
    """Missing hidden space, a changed cuboids/occupancy/split file, a changed npz or another grid are refused."""
    script = load_script()
    project_root, gt_dir, inputs = build_synthetic_project(tmp_path)
    classes = {"tree_trunk": 1}
    # Another voxel size than the GT grid.
    with pytest.raises(ValueError, match="voxel_size"):
        script.load_session_hidden_space(gt_dir, {**inputs, "grid": {**inputs["grid"], "voxel_size": 2 * VOXEL_SIZE_M}}, project_root, classes)
    # Each hashed input changed after the build.
    for relative_path in (Path("data/gt") / SESSION_NAME / "cuboids.json", Path("configs/reference_splits") / f"{SESSION_NAME}.json"):
        original = (project_root / relative_path).read_bytes()
        (project_root / relative_path).write_text("changed")
        with pytest.raises(ValueError, match="scripts/24"):
            script.load_session_hidden_space(gt_dir, inputs, project_root, classes)
        (project_root / relative_path).write_bytes(original)
    # The recorded npz hash no longer matches the file.
    record = json.loads((gt_dir / "hidden_space.json").read_text())
    record["output_npz"]["sha256"] = "0" * 64
    (gt_dir / "hidden_space.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hidden_space.npz"):
        script.load_session_hidden_space(gt_dir, inputs, project_root, classes)
    # No hidden space at all.
    (gt_dir / "hidden_space.npz").unlink()
    with pytest.raises(ValueError, match="scripts/24"):
        script.load_session_hidden_space(gt_dir, inputs, project_root, classes)


def test_loader_refuses_hidden_space_without_near_band(tmp_path):
    """A hidden space from before the near band (no hidden_near array) is refused with the scripts/24 rerun hint."""
    script = load_script()
    project_root, gt_dir, inputs = build_synthetic_project(tmp_path)
    with np.load(gt_dir / "hidden_space.npz") as archive:
        arrays = {key: archive[key] for key in archive.files if key != "hidden_near"}
    np.savez_compressed(gt_dir / "hidden_space.npz", **arrays)
    record = json.loads((gt_dir / "hidden_space.json").read_text())
    record["output_npz"]["sha256"] = sha256_of(gt_dir / "hidden_space.npz")
    (gt_dir / "hidden_space.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="scripts/24"):
        script.load_session_hidden_space(gt_dir, inputs, project_root, {"tree_trunk": 1})


def test_markdown_h_near_tables_precede_full_h_tables_and_names_band():
    """Markdown: H_near section (titled with the band) before the full-H section; rows carry _Hnear / _H names."""
    script = load_script()
    cells = synthetic_cells()
    neighbourhood, truth = class_group_masks(cells["gt"], GROUP_CLASS_IDS, cells["neighbourhoods"])
    near_mask = cells["hidden"].copy()
    near_mask[2] = False
    blocks = {key: script.hidden_class_report(GROUP_CLASS_IDS, neighbourhood, truth, mask, cells["p1_hit"], cells["occupancy"],
                                              [0.5, 0.8], cells["objects"])
              for key, mask in (("hidden_near", near_mask), ("hidden", cells["hidden"]))}
    # H_near drops hidden GT cell 2, so the two blocks differ.
    assert blocks["hidden_near"]["context"]["hidden_gt_cells"] < blocks["hidden"]["context"]["hidden_gt_cells"]
    report = {"run_id": "r", "profile": "p", "train_method": "m", "git_sha": "g", "gaussians": {"count": 1, "checkpoint_step": 0},
              "settings": {"occupancy_threshold": 0.5}, "scored_cells": 6, "hidden_space": {"near_band_m": 20.0},
              "classes": {"obstacles": {**blocks, "P1": {"strata": {}}, "P2": {"strata": {}}, "present_but_hidden": {}, "detection": {"curve": {}}}}}
    text = "\n".join(script.hidden_section_lines(report))
    assert text.index("within 20 m of the path (H_near") < text.index("whole session hidden space (H;")
    assert "P2 recall_Hnear" in text and "P2 recall_H |" in text and "P1 recall_Hnear" in text
