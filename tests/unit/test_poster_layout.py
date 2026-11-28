"""docs/proposal/poster/poster_layout.py and the pure helpers of build_poster_v2.py (no pptx needed).

Known-answer tests: expected values are computed here from each helper's stated rule.
"""

import importlib.util
from pathlib import Path

import pytest

POSTER_DIR = Path(__file__).resolve().parents[2] / "docs" / "proposal" / "poster"


def load_module(module_name: str, file_name: str):
    """Import a poster script by path.

    Args:
        module_name: Name to register it under.
        file_name: File inside ``docs/proposal/poster``.

    Returns:
        The module.
    """
    module_spec = importlib.util.spec_from_file_location(module_name, POSTER_DIR / file_name)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


layout = load_module("poster_layout_under_test", "poster_layout.py")
builder = load_module("build_poster_v2_under_test", "build_poster_v2.py")


def test_column_widths_split_by_weight_after_gutters():
    """(total − gutters) is shared in proportion to the weights."""
    total_width_in, gutter_in, weights = 23.0, 0.4, (1.0, 1.1, 1.0)
    widths = layout.column_widths(total_width_in, gutter_in, weights)
    content_width_in = total_width_in - gutter_in * (len(weights) - 1)
    expected = [content_width_in * weight / sum(weights) for weight in weights]
    assert widths == pytest.approx(expected)
    assert sum(widths) + gutter_in * (len(weights) - 1) == pytest.approx(total_width_in)


def test_column_widths_reject_non_positive_weight():
    """A zero or negative weight has no meaning as a width share."""
    with pytest.raises(ValueError):
        layout.column_widths(10.0, 0.4, (1.0, 0.0))


def test_fit_image_keeps_aspect_and_binds_on_the_tighter_side():
    """A 2:1 image in a square box is width-bound; in a wide box it is height-bound."""
    width_in, height_in = layout.fit_image(200, 100, 4.0, 4.0)
    assert (width_in, height_in) == pytest.approx((4.0, 2.0))
    width_in, height_in = layout.fit_image(200, 100, 10.0, 3.0)
    assert (width_in, height_in) == pytest.approx((6.0, 3.0))


def test_stack_blocks_places_each_block_after_the_previous_gap():
    """Tops accumulate height + gap; the returned bottom excludes the last block's gap."""
    blocks = [layout.Block(kind="text", height_in=1.0, gap_after_in=0.3),
              layout.Block(kind="text", height_in=2.0, gap_after_in=0.5),
              layout.Block(kind="text", height_in=0.5, gap_after_in=9.0)]
    bottom_in = layout.stack_blocks(blocks, top_in=4.0)
    assert [block.top_in for block in blocks] == pytest.approx([4.0, 5.3, 7.8])
    assert bottom_in == pytest.approx(8.3)


def test_paragraph_height_is_lines_times_pitch_plus_space_after():
    """One short word fits one line: height = font x pitch factor + space after, in inches."""
    font_pt, space_after_pt = 20.0, 6.0
    paragraph = layout.Paragraph(runs=(layout.TextRun("word"),), font_pt=font_pt, space_after_pt=space_after_pt)
    expected_in = (font_pt * layout.LINE_PITCH_FACTOR + space_after_pt) / layout.POINTS_PER_INCH
    assert layout.paragraph_height_in(paragraph, width_in=10.0) == pytest.approx(expected_in)


def test_narrower_box_never_gives_fewer_lines():
    """Greedy wrapping is monotone in width; a box narrower than any word gives one word per line."""
    words = "alpha beta gamma delta epsilon zeta eta theta"
    paragraph = layout.Paragraph(runs=(layout.TextRun(words),), font_pt=24.0)
    line_counts = [len(layout.wrap_paragraph(paragraph, width_in)) for width_in in (20.0, 6.0, 3.0, 1.0, 0.01)]
    assert line_counts == sorted(line_counts)
    assert line_counts[-1] == len(words.split())


def test_text_height_drops_the_last_space_after():
    """Inside one box nothing follows the last paragraph, so its space after is not counted."""
    font_pt, space_after_pt = 18.0, 12.0
    paragraph = layout.Paragraph(runs=(layout.TextRun("x-ray"),), font_pt=font_pt, space_after_pt=space_after_pt)
    single_line_in = font_pt * layout.LINE_PITCH_FACTOR / layout.POINTS_PER_INCH
    expected_in = 2 * single_line_in + space_after_pt / layout.POINTS_PER_INCH
    assert layout.text_height_in([paragraph, paragraph], width_in=10.0) == pytest.approx(expected_in)


def test_thousands_steps_text():
    """Whole thousands print as <n>k; anything else keeps its separators."""
    assert builder.thousands_steps_text(20000) == "20k"
    assert builder.thousands_steps_text(19600) == "19,600"


def test_single_value_rejects_disagreeing_values():
    """The poster states one value only when every source agrees."""
    assert builder.single_value([5, 5, 5], "run count") == 5
    with pytest.raises(ValueError, match="run count"):
        builder.single_value([5, 4], "run count")


def make_facts(june_precision_separated: bool) -> object:
    """Synthetic facts with the gap pattern the takeaway wording assumes (or one premise flipped).

    Args:
        june_precision_separated: True breaks the "June precision ranges overlap" premise.

    Returns:
        A ``PosterFacts``.
    """
    gap = builder.numbers.GapResult
    loss, init = builder.numbers.LOSS_METHOD, builder.numbers.INIT_METHOD
    gaps = {
        "june_tall_grass": {loss: {"recall": gap(3.8, True), "precision": gap(0.7, june_precision_separated),
                                   "iou": gap(3.3, True)},
                            init: {"recall": gap(-3.0, True), "precision": gap(-0.5, False), "iou": gap(-2.6, True)}},
        "may_short_grass": {loss: {"recall": gap(9.2, True), "precision": gap(-7.9, True), "iou": gap(6.7, True)},
                            init: {"recall": gap(-5.8, True), "precision": gap(-2.7, True), "iou": gap(-5.4, True)}},
    }
    return builder.PosterFacts(metrics={}, gaps=gaps, threshold_text="0.5", scored_thresholds=("0.5", "0.8"),
                               voxel_size_m=0.52, run_count=5, training_steps=20000, steps_vary=False,
                               obstacle_cells={}, hit_counts={}, arm_colours={}, both_hex="000000")


def test_takeaway_states_magnitudes_from_the_gaps():
    """The sentence quotes each gap and words the May precision drop as a positive magnitude."""
    text = "".join(run.text for run in builder.results_takeaway(make_facts(june_precision_separated=False)))
    assert "+3.8 pts in June" in text and "+9.2 pts in May" in text
    assert "precision falls 7.9 pts" in text


def test_takeaway_refuses_wording_the_data_does_not_support():
    """If June precision ranges were separated, "ranges overlap" would be false, so the builder raises."""
    with pytest.raises(ValueError, match="June precision"):
        builder.results_takeaway(make_facts(june_precision_separated=True))


def test_fits_on_one_line_detects_a_line_too_wide_for_its_box():
    """A line fits a box at least as wide as its measured width plus the safety margin, and not one narrower."""
    font_pt = 40.0
    words = ("Seeing", "through", "tall", "grass")
    paragraph = layout.Paragraph(runs=(layout.TextRun(" ".join(words), bold=True),), font_pt=font_pt)
    # Measured line width = words + the spaces between them, in points.
    line_width_pt = (sum(layout.word_width_pt(word, font_pt, True) for word in words)
                     + (len(words) - 1) * layout.word_width_pt(" ", font_pt, True))
    # Box width that leaves exactly that usable width after the safety margin.
    exact_width_in = line_width_pt / layout.POINTS_PER_INCH / (1.0 - layout.WIDTH_SAFETY_FRACTION)
    assert layout.fits_on_one_line(paragraph, exact_width_in * 1.01)
    assert not layout.fits_on_one_line(paragraph, exact_width_in * 0.9)


def test_banner_lines_fit_the_banner():
    """Every configured title line and the byline fit one line across the banner at their sizes."""
    banner_width_in = builder.PAGE_WIDTH_IN - 2 * builder.MARGIN_IN
    for line in builder.TITLE_LINES:
        title = layout.Paragraph(runs=(layout.TextRun(line, bold=True),), font_pt=builder.TITLE_PT)
        assert layout.fits_on_one_line(title, banner_width_in), line
    byline = layout.Paragraph(runs=(layout.TextRun(builder.AUTHORS_TEXT, bold=True),
                                    layout.TextRun(builder.BYLINE_SEPARATOR + builder.AFFILIATION_TEXT)),
                              font_pt=builder.BYLINE_PT)
    assert layout.fits_on_one_line(byline, banner_width_in)
