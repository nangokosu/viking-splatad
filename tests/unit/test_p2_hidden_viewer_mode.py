"""Tests of the ``--p2-hidden`` mode of scripts/15_view_pose_pyvista.py (pure parts; no PyVista needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

VIEWER_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "15_view_pose_pyvista.py"


def load_viewer_script():
    """Import scripts/15 as a module (PyVista is imported lazily inside it).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("view_pose_pyvista_hidden", VIEWER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def test_parser_defaults_and_options():
    """Defaults: group obstacles, threshold from the archive (None); explicit values parse."""
    viewer = load_viewer_script()
    defaults = viewer.build_argument_parser().parse_args(["--run", "runs/a", "--p2-hidden"])
    assert defaults.p2_hidden and defaults.hidden_class == viewer.DEFAULT_HIDDEN_CLASS == "obstacles"
    assert defaults.hidden_threshold is None
    explicit = viewer.build_argument_parser().parse_args(
        ["--run", "runs/a", "--p2-hidden", "--hidden-class", "tree_trunk", "--hidden-threshold", "0.8"])
    assert explicit.hidden_class == "tree_trunk" and explicit.hidden_threshold == 0.8


@pytest.mark.parametrize("extra_arguments", [
    ["--run", "runs/b"], ["--p2"], ["--rerender", "label"], ["--ripples"], ["--pose", "000001"], ["--show-points"],
])
def test_p2_hidden_refuses_other_modes_before_loading(extra_arguments):
    """--p2-hidden excludes --p2 and every other mode; refusal precedes any run loading (paths do not exist)."""
    viewer = load_viewer_script()
    assert viewer.main(["--run", "runs/a", "--p2-hidden", *extra_arguments]) == 1


def test_p2_refuses_p2_hidden():
    """The exclusion holds from the --p2 side too."""
    viewer = load_viewer_script()
    assert viewer.main(["--run", "runs/a", "--p2", "--p2-hidden"]) == 1


def test_threshold_text_and_caption_counts():
    """File-name threshold 0.5 -> 0p5; layer totals count the whole region, ratios come from the counts."""
    viewer = load_viewer_script()
    assert viewer.hidden_threshold_text(0.5) == "0p5" and viewer.hidden_threshold_text(0.8) == "0p8"
    cells = {
        "centres": np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]]),
        "occupancy": np.array([0.9, 0.9, 0.1, 0.9]),
        "threshold": np.float64(0.5), "p2_thresholds": np.array([0.5, 0.8]),
        "hidden": np.array([True, True, True, True]),
        "hidden_near": np.array([True, True, True, True]),
        "neighbourhood_obstacles": np.array([True, True, True, True]),
        "truth_obstacles": np.array([True, False, True, True]),
    }
    layers = viewer.hidden_cell_layers(cells, "obstacles", 0.5, np.zeros((1, 2)), 100.0, "full")
    # hit {0, 3}, false alarm {1}, missed {2}.
    assert layers["totals"] == {"context": 4, "hit": 2, "false_alarm": 1, "missed": 1}
    scene = type("SceneStub", (), {"run_id": "run_x"})()
    caption = viewer.hidden_caption(scene, "obstacles", 0.5, layers, 100.0, "full")
    # recall 2/3, precision 2/3.
    assert "recall_H 0.67, precision_H 0.67" in caption and "hit 2" in caption


def test_band_option_caption_and_default():
    """--hidden-band defaults to near; the caption names the band and its metric suffix; full differs from near."""
    viewer = load_viewer_script()
    parser = viewer.build_argument_parser()
    assert parser.parse_args(["--run", "runs/a", "--p2-hidden"]).hidden_band == "near"
    assert parser.parse_args(["--run", "runs/a", "--p2-hidden", "--hidden-band", "full"]).hidden_band == "full"
    cells = {
        "centres": np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [4.0, 0, 0]]),
        "occupancy": np.array([0.9, 0.9, 0.1, 0.9]),
        "threshold": np.float64(0.5), "p2_thresholds": np.array([0.5, 0.8]),
        "hidden": np.array([True, True, True, True]),
        # Cell 2 (a miss) lies beyond the near band.
        "hidden_near": np.array([True, True, False, True]),
        "neighbourhood_obstacles": np.array([True, True, True, True]),
        "truth_obstacles": np.array([True, False, True, True]),
    }
    scene = type("SceneStub", (), {"run_id": "run_x"})()
    positions = np.zeros((1, 2))
    near_layers = viewer.hidden_cell_layers(cells, "obstacles", 0.5, positions, 100.0)
    full_layers = viewer.hidden_cell_layers(cells, "obstacles", 0.5, positions, 100.0, "full")
    assert near_layers["totals"]["missed"] == 0 and full_layers["totals"]["missed"] == 1
    near_caption = viewer.hidden_caption(scene, "obstacles", 0.5, near_layers, 100.0)
    full_caption = viewer.hidden_caption(scene, "obstacles", 0.5, full_layers, 100.0, "full")
    # Near: recall 2/2, precision 2/3. Full: recall 2/3, precision 2/3.
    assert "near band" in near_caption and "recall_Hnear 1.00, precision_Hnear 0.67" in near_caption
    assert "whole session" in full_caption and "recall_H 0.67, precision_H 0.67" in full_caption
    with pytest.raises(ValueError, match="band"):
        viewer.hidden_cell_layers(cells, "obstacles", 0.5, positions, 100.0, "bogus")


def test_p2_hidden_main_draws_with_report_class_ids(monkeypatch, tmp_path):
    """--p2-hidden reaches the renderer with the hidden class's ids taken from the scripts/19 report.

    Regression: the mode discarded the report on load and the drawing step then read an unbound
    ``p2_report`` (UnboundLocalError), so every hidden picture failed. Loaders and the renderer are
    stubbed, so no run files or PyVista are needed.
    """
    viewer = load_viewer_script()
    # Report entry of a group: its member class ids are what the renderer must receive.
    group_class_ids = [3, 5, 7]
    report = {"classes": {"obstacles": {"class_ids": group_class_ids}}}
    archived_threshold = 0.5
    cells = {"threshold": np.float64(archived_threshold)}
    scene = type("SceneStub", (), {"run_id": "run_x"})()
    rendered_calls = []
    monkeypatch.setattr(viewer, "load_run_scene", lambda *arguments, **keywords: scene)
    monkeypatch.setattr(viewer, "check_scenes_comparable", lambda scenes: None)
    monkeypatch.setattr(viewer, "load_p2_inputs", lambda run_dir: (report, cells))
    monkeypatch.setattr(viewer, "hidden_cell_outcomes", lambda *arguments, **keywords: None)
    monkeypatch.setattr(viewer, "render_p2_hidden", lambda *arguments, **keywords: rendered_calls.append(arguments))
    exit_code = viewer.main(["--run", str(tmp_path), "--p2-hidden", "--output-dir", str(tmp_path / "viz")])
    assert exit_code == 0
    assert len(rendered_calls) == 1
    # Positional order of render_p2_hidden: scene, cells, class ids, class name, threshold, ...
    drawn_scene, drawn_cells, drawn_class_ids, drawn_class_name, drawn_threshold = rendered_calls[0][:5]
    assert drawn_scene is scene and drawn_cells is cells
    assert drawn_class_ids == group_class_ids and drawn_class_name == "obstacles"
    assert drawn_threshold == archived_threshold
