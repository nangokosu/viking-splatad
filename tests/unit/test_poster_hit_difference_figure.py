"""Tests of the pure helpers of docs/proposal/poster/poster_hit_difference_figure.py."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "docs" / "proposal" / "poster" / "poster_hit_difference_figure.py"
_spec = importlib.util.spec_from_file_location("poster_hit_difference_for_test", SCRIPT_PATH)
poster = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(poster)

# Synthetic known-answer inputs: two arbitrary RGBA float colours and a configured 0-255 colour table.
HALF_RGBA = (0.5, 0.0, 1.0, 1.0)
SECOND_RGBA = (0.0, 1.0, 0.2, 1.0)
CONFIGURED = {"only_first_arm": (1, 2, 3), "only_second_arm": (4, 5, 6), "both": (7, 8, 9)}


def test_rgba_to_rgb_ints_scales_channels():
    """Each channel is scaled by the 8-bit maximum and rounded; alpha is dropped."""
    expected = (round(0.5 * poster.RGB_CHANNEL_MAX), 0, poster.RGB_CHANNEL_MAX)
    assert poster.rgba_to_rgb_ints(HALF_RGBA) == expected


def test_poster_colours_override_only_arms_and_keep_both():
    """The two only-arm names take the arm colours; both stays; the input mapping is not mutated."""
    colours = poster.poster_colours(CONFIGURED, "only_first_arm", "only_second_arm", HALF_RGBA, SECOND_RGBA)
    assert colours["only_first_arm"] == poster.rgba_to_rgb_ints(HALF_RGBA)
    assert colours["only_second_arm"] == poster.rgba_to_rgb_ints(SECOND_RGBA)
    assert colours["both"] == CONFIGURED["both"]
    assert CONFIGURED["only_first_arm"] == (1, 2, 3)


def test_render_window_size_follows_width_and_aspect():
    """width = round(inches x dpi); height = round(width / aspect)."""
    width_in, dpi, aspect = 12.5, 200, 2.0
    expected_width = round(width_in * dpi)
    assert poster.render_window_size(width_in, dpi, aspect) == (expected_width, round(expected_width / aspect))


def test_legend_labels_use_thousands_separators():
    """Counts print with commas, in the order only first, only second, both."""
    labels = poster.legend_labels({"a": "A arm", "b": "B arm"}, "a", "b",
                                  {"only_first": 1356, "only_second": 846, "both": 3474})
    assert labels == ["only A arm: 1,356 cells", "only B arm: 846 cells", "both: 3,474 cells"]


def test_manifest_check_accepts_equal_counts():
    """Equal counts raise nothing."""
    counts = {"only_first": 5, "only_second": 6, "both": 7}
    poster.check_counts_against_manifest(counts, dict(counts))


def test_manifest_check_rejects_mismatch():
    """One differing count raises and names the key."""
    counts = {"only_first": 5, "only_second": 6, "both": 7}
    with pytest.raises(ValueError, match="only_second"):
        poster.check_counts_against_manifest(counts, {**counts, "only_second": 99})


def test_stack_label_moves_the_count_to_a_second_line():
    """The count after the last separator goes on its own line; a label without one is unchanged."""
    labels = poster.legend_labels({"a": "A arm", "b": "B arm"}, "a", "b",
                                  {"only_first": 1356, "only_second": 846, "both": 3474})
    assert [poster.stack_label(label) for label in labels] == [
        "only A arm\n1,356 cells", "only B arm\n846 cells", "both\n3,474 cells"]
    assert poster.stack_label("no count here") == "no count here"


def test_stacked_legend_is_narrower_than_one_line_legend():
    """Moving the counts to a second line narrows the measured legend block."""
    labels = poster.legend_labels({"a": "radar init + loss", "b": "stock"}, "a", "b",
                                  {"only_first": 1356, "only_second": 846, "both": 3474})
    test_dpi = 100
    one_line_in = poster.measure_legend_width_in(labels, test_dpi)
    stacked_in = poster.measure_legend_width_in([poster.stack_label(label) for label in labels], test_dpi)
    assert stacked_in < one_line_in
