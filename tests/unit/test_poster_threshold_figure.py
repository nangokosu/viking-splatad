"""docs/proposal/poster/poster_threshold_figure.py: axis limits, manifest parsing, exact-size drawing."""

import importlib.util
import math
from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "docs" / "proposal" / "poster" / "poster_threshold_figure.py"
ARMS = ("stock", "radar_init", "radar_init_loss")
GRID = [0.2, 0.5, 0.8]
TEST_WIDTH_IN = 4.0
TEST_HEIGHT_IN = 3.0
TEST_DPI = 100


def load_module():
    """Import the poster figure script as a module.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("poster_threshold_figure_under_test", SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def synthetic_manifest() -> dict:
    """Build a manifest with one null value and distinct curves per arm.

    Returns:
        Manifest dict with the keys the script reads.
    """
    drawn, bands = {}, {}
    for arm_position, arm in enumerate(ARMS):
        base = 0.3 + 0.1 * arm_position
        curve = [base, base + 0.05, None if arm_position == 0 else base + 0.1]
        drawn[arm] = {"pooled_recall": curve, "pooled_precision": [0.5 + 0.05 * arm_position] * len(GRID)}
        bands[arm] = {
            key: {
                "lower": [None if value is None else value - 0.02 for value in values],
                "upper": [None if value is None else value + 0.02 for value in values],
            }
            for key, values in drawn[arm].items()
        }
    section = {
        "curve_grid": GRID, "drawn_curves": drawn, "campaign_bands": bands,
        "repeat_counts": {arm: 5 for arm in ARMS},
    }
    return {"reported_numbers": {"threshold_curves_scene": section}, "p2_thresholds": [0.5, 0.8]}


def test_data_axis_limits_known_answer():
    """Limits follow floor/ceil of min/max padded by the margin, in tick steps."""
    module = load_module()
    values = [0.32, 0.55, float("nan"), 0.41]
    margin_fraction, tick_step = 0.1, 0.1
    # Expected from the rule: range 0.23, margin 0.023 -> 0.297 -> 0.2 ; 0.573 -> 0.6.
    data_range = 0.55 - 0.32
    expected_lower = math.floor((0.32 - margin_fraction * data_range) / tick_step) * tick_step
    expected_upper = math.ceil((0.55 + margin_fraction * data_range) / tick_step) * tick_step
    lower, upper = module.data_axis_limits(values, margin_fraction, tick_step)
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)


def test_data_axis_limits_value_on_tick_does_not_jump():
    """A constant series sitting exactly on ticks stays on them (no extra step from float noise)."""
    module = load_module()
    lower, upper = module.data_axis_limits([0.3, 0.6], 0.0, 0.1)
    assert lower == pytest.approx(0.3)
    assert upper == pytest.approx(0.6)


def test_data_axis_limits_rejects_all_nan():
    """No finite value is an error."""
    module = load_module()
    with pytest.raises(ValueError):
        module.data_axis_limits([float("nan")], 0.1, 0.1)


def test_parse_maps_null_to_nan():
    """Nulls become NaN in curves and bands; the grid and guides are floats."""
    module = load_module()
    parsed = module.parse_threshold_curves(synthetic_manifest())
    assert np.isnan(parsed["curves"]["stock"]["pooled_recall"][2])
    assert np.isnan(parsed["bands"]["stock"]["pooled_recall"][0][2])
    assert parsed["curves"]["radar_init"]["pooled_recall"][2] == pytest.approx(0.5)
    assert parsed["grid"].tolist() == GRID
    assert parsed["p2_thresholds"] == [0.5, 0.8]


def test_draw_has_exact_pixel_size(tmp_path):
    """The PNG is exactly round(inches * dpi) pixels on each side, and y limits are returned for both panels."""
    from viking_splatad.viz.figure_runs import load_figure_settings
    import yaml

    module = load_module()
    settings = load_figure_settings(yaml.safe_load(module.EVAL_CONFIG_PATH.read_text())["eval"])
    parsed = module.parse_threshold_curves(synthetic_manifest())
    output_path = tmp_path / "figure.png"
    y_limits = module.draw_poster_figure(parsed, settings, TEST_WIDTH_IN, TEST_HEIGHT_IN, TEST_DPI, output_path)
    image = matplotlib.image.imread(output_path)
    assert image.shape[1] == round(TEST_WIDTH_IN * TEST_DPI)
    assert image.shape[0] == round(TEST_HEIGHT_IN * TEST_DPI)
    assert set(y_limits) == {"pooled_recall", "pooled_precision"}


def test_constant_values_on_a_tick_still_give_an_open_axis():
    """All values equal and on a tick: zero range would make lower == upper, so one tick step is added."""
    module = load_module()
    tick_step = 0.05
    lower, upper = module.data_axis_limits([0.8, 0.8, 0.8], margin_fraction=0.1, tick_step=tick_step)
    assert lower == pytest.approx(0.8)
    assert upper == pytest.approx(0.8 + tick_step)


def test_arm_without_band_is_parsed_and_contributes_its_curve():
    """An arm drawn from an override has no campaign band: it parses without a band and its curve still counts."""
    module = load_module()
    manifest = {"p2_thresholds": [0.5],
                "reported_numbers": {"threshold_curves_scene": {
                    "curve_grid": [0.2, 0.5],
                    "drawn_curves": {"stock": {"pooled_recall": [0.4, 0.3], "pooled_precision": [0.7, 0.8]}},
                    "campaign_bands": {}}}}
    parsed = module.parse_threshold_curves(manifest)
    assert "stock" not in parsed["bands"]
    assert sorted(module.panel_values(parsed, "pooled_recall")) == [0.3, 0.4]
