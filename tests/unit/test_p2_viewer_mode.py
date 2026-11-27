"""Tests of the ``--p2`` mode of scripts/15_view_pose_pyvista.py (pure parts; no PyVista needed)."""

import ast
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval.p2_cells import NEIGHBOURHOOD_KEY_PREFIX, TRUTH_KEY_PREFIX

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
VIEWER_SCRIPT_PATH = SCRIPTS_DIR / "15_view_pose_pyvista.py"
P2_EVAL_SCRIPT_PATH = SCRIPTS_DIR / "19_eval_gaussian_occupancy.py"
# The viewer's P2 file names and the scripts/19 constants they must equal.
SHARED_NAME_PAIRS = {"P2_EVAL_DIR_NAME": "EVAL_OUTPUT_DIR_NAME", "P2_REPORT_JSON_NAME": "REPORT_JSON_NAME",
                     "P2_CELLS_ARCHIVE_NAME": "CELLS_ARCHIVE_NAME"}


def load_viewer_script():
    """Import scripts/15 as a module (PyVista is imported lazily inside it).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("view_pose_pyvista_p2", VIEWER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def module_string_constants(script_path: Path) -> dict:
    """Top-level ``NAME = "literal"`` assignments of a script, read without importing it.

    Args:
        script_path: Python file.

    Returns:
        Name -> string value.
    """
    constants = {}
    for statement in ast.parse(script_path.read_text()).body:
        if (isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name)
                and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)):
            constants[statement.targets[0].id] = statement.value.value
    return constants


def test_viewer_reads_the_files_scripts_19_writes():
    """The viewer's eval folder, report and archive names equal scripts/19's, so it reads what scripts/19 wrote."""
    viewer_constants = module_string_constants(VIEWER_SCRIPT_PATH)
    writer_constants = module_string_constants(P2_EVAL_SCRIPT_PATH)
    for viewer_name, writer_name in SHARED_NAME_PAIRS.items():
        assert viewer_constants[viewer_name] == writer_constants[writer_name]


def test_p2_cell_layers_splits_occupied_scored_cells_and_cuts_only_the_drawing_to_the_radius():
    """Green = occupied & scored & on GT, red = occupied & scored & off GT; totals count the whole region."""
    viewer = load_viewer_script()
    radius_meters = 5.0
    # Four occupied trunk cells and one unoccupied one: x positions put cells 0-2 within the radius of a lidar at
    # the origin and cell 3 (on GT) beyond it; cell 4 is on GT but below the threshold.
    centres = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [2 * radius_meters, 0.0, 0.0], [4.0, 0.0, 0.0]])
    cells = {
        "centres": centres,
        "occupancy": np.array([0.9, 0.8, 0.7, 0.9, 0.1]),
        "threshold": np.float64(0.5),
        f"{NEIGHBOURHOOD_KEY_PREFIX}tree_trunk": np.array([True, True, False, True, True]),
        f"{TRUTH_KEY_PREFIX}tree_trunk": np.array([True, False, False, True, True]),
    }
    layers = viewer.p2_cell_layers(cells, ["tree_trunk"], np.zeros((1, 2)), radius_meters)
    # Cell 2 is outside the neighbourhood (never scored), cell 4 unoccupied: on GT = {0, 3}, off GT = {1}.
    assert layers["totals"] == {"on_truth": 2, "off_truth": 1}
    # Drawn: cell 3 lies 2 radii away, so only cell 0 stays green; cell 1 stays red.
    np.testing.assert_array_equal(layers["on_truth"], centres[[0]])
    np.testing.assert_array_equal(layers["off_truth"], centres[[1]])


@pytest.mark.parametrize("extra_arguments", [
    ["--run", "runs/b"],
    ["--rerender", "label"],
    ["--ripples"],
    ["--pose", "000001"],
    ["--show-points"],
])
def test_p2_refuses_several_runs_and_other_modes_before_loading(extra_arguments):
    """--p2 draws one run's pooled cells; the refusal comes before any run is read (the paths do not exist)."""
    viewer = load_viewer_script()
    assert viewer.main(["--run", "runs/a", "--p2", *extra_arguments]) == 1


def test_parser_rejects_the_removed_hidden_picture_flag():
    """``--p2-hidden`` no longer exists: argparse exits with its usage error (code 2)."""
    viewer = load_viewer_script()
    with pytest.raises(SystemExit) as exit_info:
        viewer.build_argument_parser().parse_args(["--run", "runs/a", "--p2-hidden"])
    assert exit_info.value.code == 2


def test_caption_reads_the_schema3_pooled_counts_at_the_drawn_threshold():
    """The caption quotes TP / (TP + FP) and TP / (TP + FN) of the P2 block keyed by the drawn cut, not another cut."""
    viewer = load_viewer_script()
    # Two P2 cuts with different counts: only the block of occupancy_threshold may be quoted.
    drawn_block = {"pooled": {"true_positive": 3, "false_positive": 1, "false_negative": 2}}
    other_block = {"pooled": {"true_positive": 7, "false_positive": 7, "false_negative": 7}}
    report = {"settings": {"occupancy_estimator": "T81 peak alpha in voxel", "occupancy_threshold": 0.5},
              "classes": {"obstacles": {"P2": {"0.5": drawn_block, "0.8": other_block}}}}
    scene = type("Scene", (), {"run_id": "run-a"})()
    layers = {"totals": {"on_truth": 3, "off_truth": 1}}
    caption = viewer.p2_caption(scene, report, ["obstacles"], layers, 25.0)
    # Expected ratios from the drawn block's counts: precision 3 / (3 + 1), recall 3 / (3 + 2).
    assert f"precision {viewer.ratio_text(3, 3 + 1)}" in caption
    assert f"recall {viewer.ratio_text(3, 3 + 2)}" in caption
    assert viewer.ratio_text(7, 7 + 7) not in caption
