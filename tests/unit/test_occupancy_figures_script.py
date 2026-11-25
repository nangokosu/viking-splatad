"""Tests of the pure helpers and the failure bookkeeping of scripts/26_occupancy_figures.py (no PyVista needed)."""

import dataclasses
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from viking_splatad.viz import figure_suite_io as suite_io
from viking_splatad.viz.figure_runs import load_figure_settings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "26_occupancy_figures.py"
EVAL_YAML_PATH = PROJECT_ROOT / "configs" / "eval.yaml"
SESSION_NAME = "june_tall_grass"
ARMS = ("stock", "radar_init", "radar_init_loss")
VOXEL_EDGE = 0.5
CELL_COUNT = 6


def load_script():
    """Import scripts/26 as a module (PyVista is imported lazily inside it).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("occupancy_figures_script", SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def tiny_cells(occupancy_values):
    """A minimal cells archive (6 cells) with the keys the helpers read.

    Args:
        occupancy_values: (6,) occupancy per cell.

    Returns:
        dict of arrays.
    """
    return {
        "cell_indices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0], [2, 1, 0]]),
        "centres": np.zeros((CELL_COUNT, 3)),
        "occupancy": np.asarray(occupancy_values, dtype=float),
        "threshold": np.array(0.5),
        "p2_thresholds": np.array([0.5, 0.8]),
        "voxel_size": np.array(VOXEL_EDGE),
        "grid_origin": np.zeros(3),
        "hidden": np.ones(CELL_COUNT, dtype=bool),
        "hidden_near": np.ones(CELL_COUNT, dtype=bool),
        "neighbourhood_obstacles": np.array([True, True, True, True, False, False]),
        "truth_obstacles": np.array([True, True, False, False, False, False]),
    }


def test_threshold_grid_contains_p2_thresholds_or_raises():
    """21 points give step 0.05 containing 0.5 and 0.8; 20 points (step 1/19) do not contain 0.5."""
    script = load_script()
    grid = script.threshold_grid(21, [0.5, 0.8])
    assert len(grid) == 21 and np.isclose(grid[0], 0.0) and np.isclose(grid[-1], 1.0)
    assert np.allclose(np.diff(grid), 1.0 / 20)
    with pytest.raises(ValueError, match="not on the"):
        script.threshold_grid(20, [0.5])


def test_campaign_envelope_one_band_per_curve_key():
    """Per threshold each curve key has its own min/max over repeats (not mixed with its sibling); all-NaN stays NaN."""
    script = load_script()
    thresholds = np.array([0.0, 0.5, 1.0])
    nan = np.nan
    repeat_one = {"thresholds": thresholds, "pooled_recall": np.array([0.2, 0.4, nan]),
                  "mean_recall": np.array([0.3, 0.1, nan]), "pooled_precision": np.array([1.0, 0.5, 0.2]),
                  "mean_precision": np.array([0.9, 0.6, 0.1])}
    repeat_two = {"thresholds": thresholds, "pooled_recall": np.array([0.25, 0.35, nan]),
                  "mean_recall": np.array([0.5, 0.2, nan]), "pooled_precision": np.array([0.8, 0.7, nan]),
                  "mean_precision": np.array([0.85, 0.65, 0.3])}
    envelope = script.campaign_envelope([repeat_one, repeat_two])
    assert set(envelope) == {"pooled_recall", "mean_recall", "pooled_precision", "mean_precision"}
    # Expected: elementwise min / max of that key's two repeats only (NaN-ignoring).
    expected = {"pooled_recall": ([0.2, 0.35, nan], [0.25, 0.4, nan]),
                "mean_recall": ([0.3, 0.1, nan], [0.5, 0.2, nan]),
                "pooled_precision": ([0.8, 0.5, 0.2], [1.0, 0.7, 0.2]),
                "mean_precision": ([0.85, 0.6, 0.1], [0.9, 0.65, 0.3])}
    for curve_key, (expected_lower, expected_upper) in expected.items():
        returned_thresholds, lower, upper = envelope[curve_key]
        assert np.allclose(returned_thresholds, thresholds)
        assert np.allclose(lower, expected_lower, equal_nan=True)
        assert np.allclose(upper, expected_upper, equal_nan=True)


def test_scene_outcomes_follow_set_algebra():
    """hit = O & G, false alarm = O & N & ~G, missed = G & ~O, with O = occupancy >= 0.5 (inclusive)."""
    script = load_script()
    cells = tiny_cells([0.5, 0.2, 0.9, 0.1, 0.9, 0.0])
    outcomes = script.scene_outcomes(cells, "obstacles")
    # Cells: 0 on truth & occupied (0.5 counts); 1 truth & empty; 2 neighbourhood off truth & occupied; 3 empty
    # off truth; 4 occupied but outside the neighbourhood; 5 empty outside.
    assert outcomes["hit"].tolist() == [True, False, False, False, False, False]
    assert outcomes["missed"].tolist() == [False, True, False, False, False, False]
    assert outcomes["false_alarm"].tolist() == [False, False, True, False, False, False]
    assert outcomes["context"].tolist() == [True, True, True, True, False, False]


def test_short_run_id_strips_session_prefix():
    """The session prefix is removed once; other ids are unchanged."""
    script = load_script()
    assert script.short_run_id("june_tall_grass-profile-2026", "june_tall_grass") == "profile-2026"
    assert script.short_run_id("other-profile", "june_tall_grass") == "other-profile"


def make_context(script, tmp_path):
    """A hand-built FigureContext over tiny archives, one run folder per arm under tmp_path.

    Args:
        script: Loaded script module.
        tmp_path: Pytest temporary folder.

    Returns:
        The context.
    """
    eval_section = yaml.safe_load(EVAL_YAML_PATH.read_text())["eval"]
    settings = load_figure_settings(eval_section)
    drawn = {}
    for arm in ARMS:
        run_dir = tmp_path / f"{SESSION_NAME}-{arm}-run"
        run_dir.mkdir()
        drawn[arm] = script.ArmRun(arm, run_dir, tiny_cells(np.zeros(CELL_COUNT)), f"{arm}_profile")
        # scene_counts hashes every drawn archive, so the file must exist (its content is irrelevant here).
        drawn[arm].cells_path.parent.mkdir(parents=True, exist_ok=True)
        drawn[arm].cells_path.write_bytes(arm.encode())
    return script.FigureContext(
        session=SESSION_NAME, settings=settings, eval_section=eval_section,
        class_ids=(1, 3, 4), p2_thresholds=np.array([0.5, 0.8]), threshold=0.5,
        curve_grid=np.linspace(settings.threshold_curve_min, settings.threshold_curve_max,
                               settings.threshold_curve_points), drawn=drawn,
        repeats={arm: [drawn[arm]] for arm in ARMS},
        git={"head_sha": "abc", "dirty": False},
        arms_info={arm: {"drawn_run_id": drawn[arm].run_id, "drawn_by": "median repeat"} for arm in ARMS}, hashes=script.HashRecorder())


def stub_all_figures(script, monkeypatch, failing_name=None):
    """Replace every figure function of the script by a stub; one may raise.

    Args:
        script: Loaded script module.
        monkeypatch: Pytest monkeypatch fixture.
        failing_name: Attribute name of the stub that raises, or None.
    """
    def make_stub(name):
        """Return a stub that raises for the failing figure and returns an empty record otherwise."""
        def stub(*arguments, **keyword_arguments):
            """Stand-in for a figure function."""
            if name == failing_name:
                raise RuntimeError(f"renderer {name} exploded")
            return {}
        return stub
    for name in ("draw_arm_outcome_panels", "draw_threshold_curves"):
        monkeypatch.setattr(script, name, make_stub(name))


def test_cross_arm_failure_also_returns_one(tmp_path, monkeypatch):
    """A failure in a cross-arm renderer is recorded in the arms manifest and returns exit code 1."""
    script = load_script()
    context = make_context(script, tmp_path)
    monkeypatch.setattr(script, "build_context", lambda arguments: context)
    stub_all_figures(script, monkeypatch, failing_name="draw_threshold_curves")
    output_dir = tmp_path / "arms_out"
    assert script.main(["--session", SESSION_NAME, "--output-dir", str(output_dir)]) == 1
    arms_manifest = json.loads((output_dir / suite_io.MANIFEST_FILE_NAME).read_text())
    assert arms_manifest["figures"]["threshold_curves_scene"]["status"] == "failed"
    assert arms_manifest["figures"]["arms_scene_3d"]["status"] == "written"


def test_all_figures_ok_returns_zero(tmp_path, monkeypatch):
    """With no failing renderer main returns 0 and writes the arms manifest."""
    script = load_script()
    context = make_context(script, tmp_path)
    monkeypatch.setattr(script, "build_context", lambda arguments: context)
    stub_all_figures(script, monkeypatch)
    output_dir = tmp_path / "arms_out"
    assert script.main(["--session", SESSION_NAME, "--output-dir", str(output_dir)]) == 0
    assert (output_dir / suite_io.MANIFEST_FILE_NAME).exists()


def test_refused_setup_returns_one(monkeypatch, capsys):
    """A setup error (non-comparable runs, unknown arm ...) is a refusal with exit code 1."""
    script = load_script()

    def refuse(arguments):
        """Stand-in for build_context that refuses."""
        raise ValueError("runs are not comparable")
    monkeypatch.setattr(script, "build_context", refuse)
    assert script.main(["--session", SESSION_NAME]) == 1
    assert "refused: runs are not comparable" in capsys.readouterr().err


def test_short_title_and_run_notes(tmp_path):
    """Titles are '<session> · [arm ·] subject'; notes carry run ids without the session prefix."""
    script = load_script()
    context = make_context(script, tmp_path)
    assert context.short_title("subject") == f"{SESSION_NAME}{script.TITLE_SEPARATOR}subject"
    single_title = context.short_title("subject", "stock")
    assert single_title.split(script.TITLE_SEPARATOR) == [SESSION_NAME, context.arm_label("stock"), "subject"]
    note = context.run_note()
    assert note.startswith("drawn runs: ")
    for arm in ARMS:
        expected_id = script.short_run_id(context.drawn[arm].run_id, SESSION_NAME)
        assert f"{context.arm_label(arm)} = {expected_id}" in note
    assert SESSION_NAME + "-" not in note


def test_threshold_grid_snaps_matching_points_to_the_archived_threshold():
    """Regression: a grid point matching a P2 threshold equals the archived value exactly (curves = scripts/19)."""
    script = load_script()
    archived_threshold = 0.5 + 1e-12  # differs from the linspace point by far less than isclose's tolerance
    grid = script.threshold_grid(21, [archived_threshold, 0.8])
    assert archived_threshold in grid.tolist() and 0.8 in grid.tolist()
    assert not (grid == 0.5).any()
    assert np.allclose(grid, np.linspace(0.0, 1.0, 21))


def test_repeat_count_text_states_the_actual_counts():
    """Equal positive counts read 'n = 5 repeats per arm'; unequal or empty ones are listed per arm."""
    script = load_script()
    labels = {"stock": "stock", "radar_init": "radar init"}
    assert script.repeat_count_text({"stock": 5, "radar_init": 5}, labels) == "n = 5 repeats per arm"
    text = script.repeat_count_text({"stock": 5, "radar_init": 0}, labels)
    assert text == "repeats: stock 5, radar init 0 (no band)"
    assert script.repeat_count_text({"stock": 5, "radar_init": 4}, labels) == "repeats: stock 5, radar init 4"


def test_difference_colour_names_follow_the_arm_constants(tmp_path):
    """The legend colour keys are 'only_<arm>' of the constants' arms and a missing colour is refused."""
    script = load_script()
    context = make_context(script, tmp_path)
    assert script.difference_colour_names(context.settings) == (f"only_{script.DIFFERENCE_FIRST_ARM}",
                                                                 f"only_{script.DIFFERENCE_SECOND_ARM}")
    colours_without_second = {name: value for name, value in context.settings.colours.items()
                              if name != f"only_{script.DIFFERENCE_SECOND_ARM}"}
    broken_settings = dataclasses.replace(context.settings, colours=colours_without_second)
    with pytest.raises(ValueError, match=f"only_{script.DIFFERENCE_SECOND_ARM}"):
        script.difference_colour_names(broken_settings)


# Cell layout of the drawing tests: object 7 has cells 0, 1, 2 (0 and 1 inside its box), object 8 has cells 0, 1, 2
# (1 and 2 inside). Truth = cells 0 and 1, so both objects have hidden GT cells (2 and 1 on the all-hidden band).
OBJECT_BOX_IDS = [7, 8]
OBLIQUE_CLASS_ID = 1
OBJECT_CELL_OFFSETS = [0, 3, 6]
OBJECT_CELLS = [0, 1, 2, 0, 1, 2]
OBJECT_INSIDE = [True, True, False, False, True, True]


def cells_with_objects(occupancy_values, hidden_near_flags=None):
    """The tiny archive plus two headline-class objects and the arrays the figure functions read.

    Args:
        occupancy_values: (6,) occupancy per cell.
        hidden_near_flags: Optional (6,) hidden flags (default: all hidden).

    Returns:
        dict of arrays.
    """
    cells = tiny_cells(occupancy_values)
    cells["centres"] = np.arange(CELL_COUNT)[:, None] * np.array([VOXEL_EDGE, 0.0, 0.0])
    cells["evidence"] = np.zeros((CELL_COUNT, 4), dtype=bool)
    if hidden_near_flags is not None:
        cells["hidden_near"] = np.asarray(hidden_near_flags, dtype=bool)
    cells["hidden"] = cells["hidden_near"].copy()
    cells["object_box_ids"] = np.array(OBJECT_BOX_IDS)
    cells["object_class_ids"] = np.array([OBLIQUE_CLASS_ID, OBLIQUE_CLASS_ID])
    cells["object_cell_offsets"] = np.array(OBJECT_CELL_OFFSETS)
    cells["object_cells"] = np.array(OBJECT_CELLS)
    cells["object_inside"] = np.array(OBJECT_INSIDE)
    return cells


def context_with_objects(script, tmp_path, occupancy_by_arm):
    """A context whose drawn runs (and repeats) carry the object-bearing archive.

    Args:
        script: Loaded script module.
        tmp_path: Pytest temporary folder.
        occupancy_by_arm: arm -> (6,) occupancy.

    Returns:
        The context (the archive files exist so hashing works).
    """
    context = make_context(script, tmp_path)
    for arm in ARMS:
        context.drawn[arm].cells = cells_with_objects(occupancy_by_arm[arm])
    context.class_ids = (OBLIQUE_CLASS_ID,)
    # The hashing of inputs needs every archive file to exist (their content is irrelevant here).
    for arm in ARMS:
        cells_path = context.drawn[arm].cells_path
        cells_path.parent.mkdir(parents=True, exist_ok=True)
        cells_path.write_bytes(arm.encode())
    return context


def test_threshold_curve_cell_counts_equal_the_mask_sizes(tmp_path):
    """cell_counts = |G| and |N minus G| of the archive (tiny_cells: G = 2 cells, N = 4 cells), equal across arms."""
    script = load_script()
    context = context_with_objects(script, tmp_path, {arm: [0.5] * CELL_COUNT for arm in ARMS})
    cells = context.drawn["stock"].cells
    truth_size = int(cells["truth_obstacles"].sum())
    off_truth_size = int((cells["neighbourhood_obstacles"] & ~cells["truth_obstacles"]).sum())
    reported = script.draw_threshold_curves(context, tmp_path / "curves.png", script.BAND_ALL)
    assert reported["cell_counts"] == {"on_obstacle": truth_size, "next_to_obstacle": off_truth_size}
    assert (truth_size, off_truth_size) == (2, 2)
    # An arm with a different ground-truth mask is refused.
    context.drawn["radar_init"].cells["truth_obstacles"] = np.zeros(CELL_COUNT, dtype=bool)
    with pytest.raises(ValueError, match="differ between arms"):
        script.whole_scene_cell_counts(context)


def test_threshold_curve_numbers_and_title_state_the_repeat_count(tmp_path):
    """reported numbers hold the grid, the drawn curves and the band envelopes; curves at 0.5 match T94 by hand."""
    script = load_script()
    occupancy = {"stock": [0.5, 0.2, 0.9, 0.1, 0.0, 0.0], "radar_init": [0.5, 0.6, 0.0, 0.0, 0.0, 0.0],
                 "radar_init_loss": [0.9, 0.9, 0.9, 0.0, 0.0, 0.0]}
    context = context_with_objects(script, tmp_path, occupancy)
    reported = script.draw_threshold_curves(context, tmp_path / "curves.png", script.HEADLINE_BAND)
    assert reported["repeat_counts"] == {arm: 1 for arm in ARMS}
    grid = reported["curve_grid"]
    assert len(grid) == context.settings.threshold_curve_points and 0.5 in grid
    half_position = grid.index(0.5)
    # Stock at threshold 0.5 on H_near (all hidden): occupied = cells 0 (0.5, inclusive) and 2 (0.9). Neighbourhood
    # cells 0-3, truth cells 0, 1: hits = 1, false alarms = 1 (cell 2), missed = 1 (cell 1), so recall = precision = 1/2.
    stock_curves = reported["drawn_curves"]["stock"]
    assert stock_curves["pooled_recall"][half_position] == pytest.approx(1 / 2)
    assert stock_curves["pooled_precision"][half_position] == pytest.approx(1 / 2)
    band = reported["campaign_bands"]["stock"]["pooled_recall"]
    assert len(band["lower"]) == len(band["upper"]) == len(grid)
    assert (tmp_path / "curves.png").is_file()


def test_threshold_curves_without_campaign_runs_has_no_band(tmp_path):
    """An arm whose repeats are empty (an override stands in) is drawn without a band and counted as 0."""
    script = load_script()
    occupancy = {arm: [0.9, 0.9, 0.0, 0.0, 0.0, 0.0] for arm in ARMS}
    context = context_with_objects(script, tmp_path, occupancy)
    context.repeats["radar_init_loss"] = []
    reported = script.draw_threshold_curves(context, tmp_path / "curves.png", script.BAND_ALL)
    assert reported["repeat_counts"]["radar_init_loss"] == 0
    assert "radar_init_loss" not in reported["campaign_bands"]
    assert "radar_init_loss" in reported["drawn_curves"]


def test_threshold_grid_over_0_2_to_0_9_contains_the_p2_thresholds():
    """15 points over [0.2, 0.9] (step 0.05) contain 0.2, 0.5, 0.8 and 0.9."""
    script = load_script()
    grid = script.threshold_grid(15, [0.5, 0.8], 0.2, 0.9)
    assert len(grid) == 15 and np.allclose(np.diff(grid), 0.05)
    for expected_point in (0.2, 0.5, 0.8, 0.9):
        assert np.isclose(grid, expected_point).any()


def test_threshold_grid_refuses_a_missing_p2_threshold_and_an_empty_range():
    """A P2 threshold off the grid raises, and so does min >= max."""
    script = load_script()
    with pytest.raises(ValueError, match="not on the"):
        script.threshold_grid(14, [0.5], 0.2, 0.9)
    with pytest.raises(ValueError, match="below maximum"):
        script.threshold_grid(15, [0.5], 0.9, 0.2)
    with pytest.raises(ValueError, match="below maximum"):
        script.threshold_grid(15, [0.5], 0.5, 0.5)


def test_scene_counts_are_written_before_and_outside_every_figure(tmp_path, monkeypatch):
    """scene_counts holds the cell counts and each arm's outcome counts even when every figure fails."""
    script = load_script()
    occupancy = {"stock": [0.5, 0.2, 0.9, 0.1, 0.0, 0.0], "radar_init": [0.5, 0.6, 0.0, 0.0, 0.0, 0.0],
                 "radar_init_loss": [0.9, 0.9, 0.9, 0.0, 0.0, 0.0]}
    context = context_with_objects(script, tmp_path, occupancy)
    for name in ("draw_arm_outcome_panels", "draw_threshold_curves"):
        monkeypatch.setattr(script, name, lambda *arguments, **keyword_arguments: (_ for _ in ()).throw(
            RuntimeError("figure exploded")))
    output_dir = tmp_path / "out"
    recorder = script.generate_arm_figures(context, output_dir)
    assert recorder.any_failed()
    counts = json.loads((output_dir / suite_io.MANIFEST_FILE_NAME).read_text())["reported_numbers"]["scene_counts"]
    assert counts["band"] == "all" and counts["cell_counts"] == {"on_obstacle": 2, "next_to_obstacle": 2}
    # Stock O = cells 0 (0.5, inclusive) and 2; N = cells 0-3, G = cells 0, 1: hit 1, false alarm 1 (cell 2), missed 1
    # (cell 1), context |N| = 4. Computed from the sets, not from the code under test.
    assert counts["outcome_counts_by_arm"]["stock"] == {"hit": 1, "false_alarm": 1, "missed": 1, "context": 4}
    # Init+loss: O = cells 0, 1, 2 -> hit 2, false alarm 1, missed 0.
    assert counts["outcome_counts_by_arm"]["radar_init_loss"] == {"hit": 2, "false_alarm": 1, "missed": 0, "context": 4}
    assert set(counts["outcome_counts_by_arm"]) == set(ARMS)


def test_missing_cuboids_file_fails_only_the_scene_figure(tmp_path):
    """Without the GT cuboid file arms_scene_3d fails; the curves and scene_counts are still written."""
    script = load_script()
    occupancy = {arm: [0.9, 0.9, 0.0, 0.0, 0.0, 0.0] for arm in ARMS}
    context = context_with_objects(script, tmp_path, occupancy)
    context.cuboids_path = tmp_path / "no_such_dir" / script.CUBOIDS_FILE_NAME
    output_dir = tmp_path / "out"
    recorder = script.generate_arm_figures(context, output_dir)
    manifest = json.loads((output_dir / suite_io.MANIFEST_FILE_NAME).read_text())
    assert recorder.any_failed()
    assert manifest["figures"]["arms_scene_3d"]["status"] == "failed"
    assert manifest["figures"]["threshold_curves_scene"]["status"] == "written"
    assert "scene_counts" in manifest["reported_numbers"]
    assert "threshold_curves_scene" in manifest["reported_numbers"]


def test_ordered_box_counts_follow_the_class_id_order_not_the_encounter_order(tmp_path):
    """Counts are ordered like context.class_ids (the group's member order), whatever the order of the boxes."""
    script = load_script()
    context = make_context(script, tmp_path)
    class_name_by_id = {int(class_id): class_name for class_name, class_id in context.eval_section["scored_class_ids"].items()}
    expected_names = [class_name_by_id[int(class_id)] for class_id in context.class_ids]
    # Boxes listed in reverse member order, so encounter order differs from the group order.
    boxes = [SimpleNamespace(class_name=class_name) for class_name in reversed(expected_names)]
    assert list(script.ordered_box_counts(context, boxes)) == expected_names
    # A class without boxes is left out; the others keep their order.
    assert list(script.ordered_box_counts(context, boxes[:-1])) == expected_names[1:]


def test_difference_arms_missing_from_the_arm_order_raise_a_clear_error(tmp_path):
    """A settings arm order without a compared arm is refused with a message naming the arm."""
    script = load_script()
    context = make_context(script, tmp_path)
    script.check_difference_arms(context)
    reduced_order = tuple(arm for arm in context.settings.arm_order if arm != script.DIFFERENCE_SECOND_ARM)
    context.settings = dataclasses.replace(context.settings, arm_order=reduced_order)
    with pytest.raises(ValueError, match=script.DIFFERENCE_SECOND_ARM):
        script.check_difference_arms(context)
