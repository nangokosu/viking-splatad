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


def test_in_window_membership_is_closed_on_both_bounds():
    """Window = projected extent padded by metres; membership is closed on both bounds."""
    script = load_script()
    corners = np.array([[1.0, 2.0], [3.0, 6.0], [2.0, 4.0]])
    padding_m = 3 * VOXEL_EDGE
    lower, upper = script.padded_window(corners, padding_m)
    assert np.allclose(lower, corners.min(axis=0) - padding_m)
    assert np.allclose(upper, corners.max(axis=0) + padding_m)
    inside = script.in_window(np.array([lower, upper, [0.0, 4.0], upper + 0.01]), (lower, upper))
    assert inside.tolist() == [True, True, True, False]


def test_convex_polygon_drops_interior_points():
    """The hull of a square plus its centre has the four corners only."""
    script = load_script()
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [1.0, 1.0]])
    hull = script.convex_polygon(square)
    assert sorted(map(tuple, hull)) == sorted(map(tuple, square[:4]))


def test_difference_image_nan_rule():
    """NaN only where both are NaN; a one-sided NaN counts as 0; else first - second."""
    script = load_script()
    nan = np.nan
    first = np.array([[0.8, nan, nan, 0.2]])
    second = np.array([[0.3, 0.4, nan, nan]])
    difference = script.difference_image(first, second)
    expected = np.array([[0.8 - 0.3, 0.0 - 0.4, nan, 0.2 - 0.0]])
    assert np.allclose(difference, expected, equal_nan=True)


def test_campaign_envelope_min_max_over_repeats_and_both_curves():
    """Per threshold the band is min/max over repeats of the pooled and mean curves; all-NaN stays NaN."""
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
    _, recall_lower, recall_upper = envelope["recall"]
    assert np.allclose(recall_lower, [0.2, 0.1, nan], equal_nan=True)
    assert np.allclose(recall_upper, [0.5, 0.4, nan], equal_nan=True)
    _, precision_lower, precision_upper = envelope["precision"]
    assert np.allclose(precision_lower, [0.8, 0.5, 0.1])
    assert np.allclose(precision_upper, [1.0, 0.7, 0.3])


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


def test_seeds_path_for_profile_uses_anchor_or_none():
    """Stock profiles have no seeds; radar profiles read the anchor's artifact, else their own."""
    script = load_script()
    profiles = {"stock_profile": {}, "init_profile": {"radar_variant": "init"},
                "loss_profile": {"radar_variant": "init_loss", "radar_anchor_profile": "init_profile"}}
    root = Path("intermediate")
    assert script.seeds_path_for_profile("stock_profile", profiles, SESSION_NAME, root) is None
    expected_init = root / SESSION_NAME / "radar_inputs" / "init_profile" / "radar_seeds.npz"
    assert script.seeds_path_for_profile("init_profile", profiles, SESSION_NAME, root) == expected_init
    assert script.seeds_path_for_profile("loss_profile", profiles, SESSION_NAME, root) == expected_init


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
    return script.FigureContext(
        session=SESSION_NAME, settings=settings, eval_section=eval_section, cuboids=[],
        sweep_origins=np.zeros((1, 3)), class_ids=(1, 3, 4), p2_thresholds=np.array([0.5, 0.8]), threshold=0.5,
        curve_grid=np.linspace(0.0, 1.0, settings.threshold_curve_points), drawn=drawn,
        repeats={arm: [drawn[arm]] for arm in ARMS}, seeds_paths={arm: None for arm in ARMS},
        git={"head_sha": "abc", "dirty": False},
        arms_info={arm: {"drawn_run_id": drawn[arm].run_id, "drawn_by": "median repeat"} for arm in ARMS}, hashes=script.HashRecorder(),
        cuboids_path=tmp_path / "cuboids.json")


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
    for name in ("draw_occupancy_bev", "draw_occupancy_outcomes_3d", "draw_gaussian_density_bev",
                 "draw_arm_outcome_panels", "draw_trunk_gallery", "draw_occlusion_sections",
                 "draw_threshold_curves", "draw_occupancy_histograms", "draw_hidden_near_heat"):
        monkeypatch.setattr(script, name, make_stub(name))


def test_renderer_failure_is_recorded_and_exit_code_is_one(tmp_path, monkeypatch):
    """A raising renderer is recorded as failed (others written), the manifests exist and main returns 1."""
    script = load_script()
    context = make_context(script, tmp_path)
    monkeypatch.setattr(script, "build_context", lambda arguments: context)
    stub_all_figures(script, monkeypatch, failing_name="draw_occupancy_outcomes_3d")
    output_dir = tmp_path / "arms_out"
    exit_code = script.main(["--session", SESSION_NAME, "--output-dir", str(output_dir)])
    assert exit_code == 1
    for arm in ARMS:
        manifest = json.loads((context.drawn[arm].run_dir / script.PER_RUN_FIGURE_DIR
                               / suite_io.MANIFEST_FILE_NAME).read_text())
        assert manifest["schema"] == "occupancy-figures/1"
        assert manifest["figures"]["occupancy_outcomes_3d"]["status"] == "failed"
        assert "exploded" in manifest["figures"]["occupancy_outcomes_3d"]["error"]
        assert manifest["figures"]["occupancy_bev"]["status"] == "written"
    arms_manifest = json.loads((output_dir / suite_io.MANIFEST_FILE_NAME).read_text())
    assert arms_manifest["session"] == SESSION_NAME and arms_manifest["binding_threshold"] == 0.5
    assert all(entry["status"] == "written" for entry in arms_manifest["figures"].values())


def test_cross_arm_failure_also_returns_one(tmp_path, monkeypatch):
    """A failure in a cross-arm renderer is recorded in the arms manifest and returns exit code 1."""
    script = load_script()
    context = make_context(script, tmp_path)
    monkeypatch.setattr(script, "build_context", lambda arguments: context)
    stub_all_figures(script, monkeypatch, failing_name="draw_threshold_curves")
    output_dir = tmp_path / "arms_out"
    assert script.main(["--session", SESSION_NAME, "--output-dir", str(output_dir), "--arms-only"]) == 1
    arms_manifest = json.loads((output_dir / suite_io.MANIFEST_FILE_NAME).read_text())
    assert arms_manifest["figures"]["threshold_curves"]["status"] == "failed"
    assert arms_manifest["figures"]["arms_scene_3d"]["status"] == "written"


def test_all_figures_ok_returns_zero(tmp_path, monkeypatch):
    """With no failing renderer main returns 0 and --per-run-only writes no arms manifest."""
    script = load_script()
    context = make_context(script, tmp_path)
    monkeypatch.setattr(script, "build_context", lambda arguments: context)
    stub_all_figures(script, monkeypatch)
    output_dir = tmp_path / "arms_out"
    assert script.main(["--session", SESSION_NAME, "--output-dir", str(output_dir), "--per-run-only"]) == 0
    assert not (output_dir / suite_io.MANIFEST_FILE_NAME).exists()


def test_refused_setup_returns_one(monkeypatch, capsys):
    """A setup error (non-comparable runs, unknown arm ...) is a refusal with exit code 1."""
    script = load_script()

    def refuse(arguments):
        """Stand-in for build_context that refuses."""
        raise ValueError("runs are not comparable")
    monkeypatch.setattr(script, "build_context", refuse)
    assert script.main(["--session", SESSION_NAME]) == 1
    assert "refused: runs are not comparable" in capsys.readouterr().err


def test_padded_window_pads_each_side_by_metres():
    """The window is [min - padding, max + padding] per axis (expected values computed from the corners)."""
    script = load_script()
    corners = np.array([[10.0, 2.0], [12.0, 2.0], [12.0, 9.0], [10.0, 9.0]])
    padding_m = 2.0
    lower, upper = script.padded_window(corners, padding_m)
    np.testing.assert_allclose(lower, corners.min(axis=0) - padding_m)
    np.testing.assert_allclose(upper, corners.max(axis=0) + padding_m)


def test_boxes_inside_extent_keeps_only_boxes_with_centre_inside():
    """A box centred outside the extent is dropped even when a corner lies inside."""
    script = load_script()
    inside_box = np.array([[1.0, 1.0], [3.0, 1.0], [3.0, 3.0], [1.0, 3.0]])
    straddling_box = inside_box + np.array([9.0, 0.0])  # centre x = 11 > extent max 10, left corner inside
    kept = script.boxes_inside_extent([inside_box, straddling_box], (0.0, 10.0, 0.0, 10.0))
    assert len(kept) == 1 and np.array_equal(kept[0], inside_box)


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


def test_finite_summary_counts_and_range():
    """Finite columns are counted and the range ignores NaN; an all-NaN map has no range."""
    script = load_script()
    image = np.array([[0.2, np.nan, 0.9], [np.nan, -0.1, np.nan]])
    assert script.finite_summary(image) == {"finite_columns": 3, "min": -0.1, "max": 0.9}
    assert script.finite_summary(np.full((2, 2), np.nan)) == {"finite_columns": 0, "min": None, "max": None}


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
        The context (cuboids_path exists so hashing works).
    """
    context = make_context(script, tmp_path)
    for arm in ARMS:
        context.drawn[arm].cells = cells_with_objects(occupancy_by_arm[arm])
    context.class_ids = (OBLIQUE_CLASS_ID,)
    context.cuboids_path.write_text("[]")
    # The hashing of inputs needs every archive file to exist (their content is irrelevant here).
    for arm in ARMS:
        cells_path = context.drawn[arm].cells_path
        cells_path.parent.mkdir(parents=True, exist_ok=True)
        cells_path.write_bytes(arm.encode())
    return context


def test_histogram_numbers_are_reported_per_arm_and_series(tmp_path):
    """reported numbers hold the counts of the figure's own bins: they sum to n and match np.histogram."""
    script = load_script()
    occupancy = {"stock": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "radar_init": [0.5, 1.0, 0.2, 0.0, 0.0, 0.0],
                 "radar_init_loss": [1.0, 1.0, 0.9, 0.9, 0.0, 0.0]}
    context = context_with_objects(script, tmp_path, occupancy)
    reported = script.draw_occupancy_histograms(context, tmp_path / "histograms.png")
    assert (tmp_path / "histograms.png").is_file()
    edges = np.linspace(0.0, 1.0, script.HISTOGRAM_BIN_COUNT + 1)
    assert reported["bin_edges"] == edges.tolist()
    # radar_init: truth cells 0, 1 have occupancy 0.5 and 1.0; off-truth neighbourhood cells 2, 3 have 0.2 and 0.0.
    truth_counts = reported["counts"]["radar_init"]["truth"]
    assert truth_counts["n"] == 2 and sum(truth_counts["counts"]) == 2
    assert truth_counts["counts"] == np.histogram([0.5, 1.0], bins=edges)[0].tolist()
    off_truth_counts = reported["counts"]["radar_init"]["off_truth"]
    assert off_truth_counts["counts"] == np.histogram([0.2, 0.0], bins=edges)[0].tolist()


def test_threshold_curve_numbers_and_title_state_the_repeat_count(tmp_path):
    """reported numbers hold the grid, the drawn curves and the band envelopes; curves at 0.5 match T94 by hand."""
    script = load_script()
    occupancy = {"stock": [0.5, 0.2, 0.9, 0.1, 0.0, 0.0], "radar_init": [0.5, 0.6, 0.0, 0.0, 0.0, 0.0],
                 "radar_init_loss": [0.9, 0.9, 0.9, 0.0, 0.0, 0.0]}
    context = context_with_objects(script, tmp_path, occupancy)
    reported = script.draw_threshold_curves(context, tmp_path / "curves.png")
    assert reported["repeat_counts"] == {arm: 1 for arm in ARMS}
    grid = reported["curve_grid"]
    assert len(grid) == context.settings.threshold_curve_points and 0.5 in grid
    half_position = grid.index(0.5)
    # Stock at threshold 0.5 on H_near (all hidden): occupied = cells 0 (0.5, inclusive) and 2 (0.9). Neighbourhood
    # cells 0-3, truth cells 0, 1: hits = 1, false alarms = 1 (cell 2), missed = 1 (cell 1), so recall = precision = 1/2.
    stock_curves = reported["drawn_curves"]["H_near"]["stock"]
    assert stock_curves["pooled_recall"][half_position] == pytest.approx(1 / 2)
    assert stock_curves["pooled_precision"][half_position] == pytest.approx(1 / 2)
    band = reported["campaign_bands"]["H_near"]["stock"]["recall"]
    assert len(band["lower"]) == len(band["upper"]) == len(grid)
    assert (tmp_path / "curves.png").is_file()


def test_threshold_curves_without_campaign_runs_has_no_band(tmp_path):
    """An arm whose repeats are empty (an override stands in) is drawn without a band and counted as 0."""
    script = load_script()
    occupancy = {arm: [0.9, 0.9, 0.0, 0.0, 0.0, 0.0] for arm in ARMS}
    context = context_with_objects(script, tmp_path, occupancy)
    context.repeats["radar_init_loss"] = []
    reported = script.draw_threshold_curves(context, tmp_path / "curves.png")
    assert reported["repeat_counts"]["radar_init_loss"] == 0
    assert "radar_init_loss" not in reported["campaign_bands"]["H_near"]
    assert "radar_init_loss" in reported["drawn_curves"]["H_near"]


def test_heat_numbers_are_reported(tmp_path):
    """The heat map reports each arm's finite columns and range and the difference's range."""
    script = load_script()
    occupancy = {"stock": [0.2, 0.2, 0.2, 0.2, 0.0, 0.0], "radar_init": [0.2, 0.2, 0.2, 0.2, 0.0, 0.0],
                 "radar_init_loss": [0.9, 0.9, 0.9, 0.9, 0.0, 0.0]}
    context = context_with_objects(script, tmp_path, occupancy)
    reported = script.draw_hidden_near_heat(context, tmp_path / "heat.png")
    # Mask = hidden_near & neighbourhood_obstacles = cells 0-3, which sit in 4 columns.
    assert reported["arm_maps"]["stock"]["finite_columns"] == 4
    assert reported["arm_maps"]["radar_init_loss"]["max"] == pytest.approx(0.9)
    # Difference = init+loss minus stock = 0.9 - 0.2 in every column.
    assert reported["difference"]["min"] == pytest.approx(0.9 - 0.2)
    assert reported["difference"]["max"] == pytest.approx(0.9 - 0.2)
    assert len(reported["extent"]) == 4


def test_occlusion_sections_skip_when_no_trunk_has_hidden_gt(tmp_path):
    """Regression: no trunk with a hidden GT cell is a recorded skip (not a failed figure, no crash)."""
    script = load_script()
    context = context_with_objects(script, tmp_path, {arm: np.zeros(CELL_COUNT) for arm in ARMS})
    trunk_class_id = context.eval_section["scored_class_ids"][script.TRUNK_CLASS_NAME]
    context.reference_cells["object_class_ids"] = np.array([trunk_class_id, trunk_class_id])
    context.reference_cells["neighbourhood_tree_trunk"] = context.reference_cells["neighbourhood_obstacles"]
    context.reference_cells["truth_tree_trunk"] = context.reference_cells["truth_obstacles"]
    context.reference_cells["hidden_near"] = np.zeros(CELL_COUNT, dtype=bool)  # nothing hidden
    recorder = script.FigureRecorder()
    output_path = tmp_path / "sections.png"
    recorder.attempt("occlusion_sections", output_path, lambda: script.draw_occlusion_sections(context, output_path))
    assert recorder.figures["occlusion_sections"]["status"] == "skipped"
    assert "no tree_trunk object has a hidden GT cell" in recorder.figures["occlusion_sections"]["error"]
    assert not recorder.any_failed() and not output_path.exists()


def test_gallery_computes_outcomes_once_per_arm_and_titles_the_page_number(tmp_path, monkeypatch):
    """hidden_cell_outcomes runs once per arm (not per box and arm); the title carries the page number passed in."""
    script = load_script()
    context = context_with_objects(script, tmp_path, {arm: [0.9] * CELL_COUNT for arm in ARMS})
    context.cuboids = [SimpleNamespace(box_id=box_id, class_name="trunk") for box_id in OBJECT_BOX_IDS]
    outcome_calls = []
    real_outcomes = script.hidden_cell_outcomes

    def counting_outcomes(*arguments, **keyword_arguments):
        """Count the calls and delegate."""
        outcome_calls.append(arguments)
        return real_outcomes(*arguments, **keyword_arguments)

    saved_titles = []
    monkeypatch.setattr(script, "hidden_cell_outcomes", counting_outcomes)
    monkeypatch.setattr(script, "import_pyvista", lambda: None)
    monkeypatch.setattr(script, "cuboid_wireframe_arrays", lambda cuboids: None)
    monkeypatch.setattr(script, "render_object_tile", lambda *arguments, **keyword_arguments: None)
    monkeypatch.setattr(script, "save_matplotlib_figure",
                        lambda figure, output_path, dpi: saved_titles.append(figure._suptitle.get_text()))
    recorder = script.FigureRecorder()
    reported = script.draw_trunk_gallery(context, tmp_path, recorder)
    assert len(outcome_calls) == len(ARMS)
    assert sorted(reported) == [str(box_id) for box_id in OBJECT_BOX_IDS]
    assert len(saved_titles) == 1 and saved_titles[0].endswith("page 1")
    assert recorder.figures["trunk_gallery_p01"]["status"] == "written"


def test_gallery_without_hidden_objects_is_one_skipped_entry(tmp_path):
    """No object with hidden GT: one skipped 'trunk_gallery' entry and no page entries."""
    script = load_script()
    context = context_with_objects(script, tmp_path, {arm: np.zeros(CELL_COUNT) for arm in ARMS})
    for arm in ARMS:
        context.drawn[arm].cells["hidden_near"] = np.zeros(CELL_COUNT, dtype=bool)
    recorder = script.FigureRecorder()
    assert script.draw_trunk_gallery(context, tmp_path, recorder) == {}
    assert list(recorder.figures) == ["trunk_gallery"] and recorder.figures["trunk_gallery"]["status"] == "skipped"


def test_gallery_preparation_failure_is_recorded_once_as_failed(tmp_path):
    """A failure while preparing the tiles is one failed 'trunk_gallery' entry (through the shared recorder path)."""
    script = load_script()
    context = context_with_objects(script, tmp_path, {arm: np.zeros(CELL_COUNT) for arm in ARMS})
    del context.drawn["stock"].cells["object_box_ids"]
    recorder = script.FigureRecorder()
    assert script.draw_trunk_gallery(context, tmp_path, recorder) == {}
    assert recorder.figures["trunk_gallery"]["status"] == "failed" and recorder.any_failed()

