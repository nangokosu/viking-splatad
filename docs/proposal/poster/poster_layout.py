"""Pure layout helpers of the poster builder: text measurement, word wrapping and vertical stacking.

No pptx here (``build_poster_v2.py`` owns the slide), so these helpers are unit-testable in the reference
image. Text is measured with matplotlib's font machinery on the same font family the slide uses, so the
builder can size every text box to its content and check that each column fits before writing a slide.

Measurement model:
    * a word's width = its glyph advance width at the run's size and weight (``TextToPath``), in points;
    * a line holds words greedily until the next word (plus one space) would exceed the usable width;
    * a paragraph's height = line count x line pitch, where line pitch = font size x ``LINE_PITCH_FACTOR``
      (the builder writes the same exact line spacing into the slide, so measure and render agree);
    * the usable width is the box width shrunk by ``WIDTH_SAFETY_FRACTION`` to absorb small metric
      differences between the measuring font and the renderer's font.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

# Points per inch (typographic definition).
POINTS_PER_INCH = 72.0
# Line pitch as a multiple of the font size. The builder sets the same exact spacing on every paragraph.
LINE_PITCH_FACTOR = 1.2
# Fraction of the box width held back when wrapping (renderer fonts differ slightly from the measured one).
WIDTH_SAFETY_FRACTION = 0.05
# Font family of every text box (a metric-safe font that ships with Office and macOS; LibreOffice maps it
# to the metric-compatible Liberation Sans where Arial is absent).
FONT_FAMILY = "Arial"


@dataclass(frozen=True)
class TextRun:
    """A stretch of text with one style.

    Attributes:
        text: The characters (words separated by single spaces).
        bold: Bold weight when True.
        colour_hex: Six-digit RGB hex without ``#``; None = the paragraph's default colour.
    """

    text: str
    bold: bool = False
    colour_hex: Optional[str] = None


@dataclass(frozen=True)
class Paragraph:
    """One paragraph of runs at one size.

    Attributes:
        runs: The runs, in reading order.
        font_pt: Font size in points.
        space_after_pt: Gap after the paragraph, in points.
        align: ``left`` or ``center``.
        colour_hex: Default colour of runs that set none.
    """

    runs: Tuple[TextRun, ...]
    font_pt: float
    space_after_pt: float = 0.0
    align: str = "left"
    colour_hex: str = "000000"


@dataclass
class Block:
    """A vertically stacked element of a column.

    Attributes:
        kind: Builder-defined drawing kind (``build_poster_v2.py`` uses ``text``, ``picture`` and ``photo_pair``).
        height_in: Height the block occupies, in inches.
        payload: Whatever the builder needs to draw it (paragraphs, image bytes, ...).
        gap_after_in: Vertical gap below the block, in inches.
        top_in: Filled in by ``stack_blocks``: the block's top edge on the slide.
    """

    kind: str
    height_in: float
    payload: object = None
    gap_after_in: float = 0.0
    top_in: float = field(default=0.0)


@lru_cache(maxsize=None)
def _text_measurer():
    """Build matplotlib's text-to-path converter once.

    Returns:
        A ``TextToPath`` instance (glyph metrics without a figure or backend).
    """
    from matplotlib.textpath import TextToPath
    return TextToPath()


@lru_cache(maxsize=None)
def word_width_pt(word: str, font_pt: float, bold: bool) -> float:
    """Advance width of a word at a size and weight.

    Args:
        word: The text to measure (no line breaks).
        font_pt: Font size in points.
        bold: Bold weight when True.

    Returns:
        Width in points (TextToPath returns metrics in the units of the given font size, i.e. points).
    """
    from matplotlib.font_manager import FontProperties
    properties = FontProperties(family=FONT_FAMILY, weight="bold" if bold else "normal", size=font_pt)
    width_pt, _height_pt, _descent_pt = _text_measurer().get_text_width_height_descent(word, properties, ismath=False)
    return width_pt


def wrap_paragraph(paragraph: Paragraph, width_in: float) -> List[List[Tuple[str, TextRun]]]:
    """Greedy word wrap of a paragraph into lines.

    Each word keeps its run (style). A word longer than the whole line gets a line of its own, never split.
    Complexity: O(words) width lookups (cached per word, size and weight).

    Args:
        paragraph: The paragraph.
        width_in: Box width in inches (before the safety margin).

    Returns:
        Lines; each line is a list of (word, run) pairs.
    """
    # Step 1: usable width in points, shrunk by the safety fraction.
    usable_width_pt = width_in * POINTS_PER_INCH * (1.0 - WIDTH_SAFETY_FRACTION)
    lines: List[List[Tuple[str, TextRun]]] = []
    current_line: List[Tuple[str, TextRun]] = []
    current_width_pt = 0.0
    # Step 2: words in reading order across runs; a line break starts when the next word does not fit.
    for run in paragraph.runs:
        for word in run.text.split():
            word_pt = word_width_pt(word, paragraph.font_pt, run.bold)
            space_pt = word_width_pt(" ", paragraph.font_pt, run.bold) if current_line else 0.0
            if current_line and current_width_pt + space_pt + word_pt > usable_width_pt:
                lines.append(current_line)
                current_line, current_width_pt, space_pt = [], 0.0, 0.0
            current_line.append((word, run))
            current_width_pt += space_pt + word_pt
    # Step 3: the last, partly filled line (an empty paragraph still occupies one line).
    lines.append(current_line)
    return lines


def fits_on_one_line(paragraph: Paragraph, width_in: float) -> bool:
    """Whether a paragraph wraps to a single line in a box of the given width.

    Args:
        paragraph: The paragraph.
        width_in: Box width in inches (before the safety margin).

    Returns:
        True when ``wrap_paragraph`` yields exactly one line.
    """
    return len(wrap_paragraph(paragraph, width_in)) == 1


def paragraph_height_in(paragraph: Paragraph, width_in: float) -> float:
    """Height of a wrapped paragraph including its space after.

    Args:
        paragraph: The paragraph.
        width_in: Box width in inches.

    Returns:
        line count x line pitch + space after, in inches.
    """
    line_count = len(wrap_paragraph(paragraph, width_in))
    line_pitch_pt = paragraph.font_pt * LINE_PITCH_FACTOR
    return (line_count * line_pitch_pt + paragraph.space_after_pt) / POINTS_PER_INCH


def text_height_in(paragraphs: Sequence[Paragraph], width_in: float) -> float:
    """Height of a text box holding several paragraphs.

    The last paragraph's space after is not counted (nothing follows it inside the box).

    Args:
        paragraphs: The paragraphs, top to bottom.
        width_in: Box width in inches.

    Returns:
        Height in inches.
    """
    if not paragraphs:
        return 0.0
    total_in = sum(paragraph_height_in(paragraph, width_in) for paragraph in paragraphs)
    return total_in - paragraphs[-1].space_after_pt / POINTS_PER_INCH


def stack_blocks(blocks: Sequence[Block], top_in: float) -> float:
    """Assign each block its top edge, stacking downward with each block's gap after.

    Args:
        blocks: Blocks in order (their ``top_in`` is set in place).
        top_in: Top edge of the first block.

    Returns:
        Bottom edge of the last block (its trailing gap excluded), in inches.
    """
    cursor_in = top_in
    bottom_in = top_in
    for block in blocks:
        block.top_in = cursor_in
        bottom_in = cursor_in + block.height_in
        cursor_in = bottom_in + block.gap_after_in
    return bottom_in


def column_widths(total_width_in: float, gutter_in: float, weights: Sequence[float]) -> List[float]:
    """Split a width into columns by weight, leaving equal gutters between them.

    Args:
        total_width_in: Width available for columns plus gutters.
        gutter_in: Gap between adjacent columns.
        weights: Relative column widths (positive).

    Returns:
        Column widths: (total − gutters) x weight / sum(weights).
    """
    if not weights or any(weight <= 0 for weight in weights):
        raise ValueError(f"column weights must be positive, got {list(weights)}")
    content_width_in = total_width_in - gutter_in * (len(weights) - 1)
    weight_total = sum(weights)
    return [content_width_in * weight / weight_total for weight in weights]


def fit_image(image_width_px: int, image_height_px: int, box_width_in: float, box_height_in: float) -> Tuple[float, float]:
    """Largest size of an image inside a box with its aspect ratio kept.

    Args:
        image_width_px: Image width in pixels.
        image_height_px: Image height in pixels.
        box_width_in: Box width in inches.
        box_height_in: Box height in inches.

    Returns:
        (width, height) in inches: scale = min(box_w / img_w, box_h / img_h) applied to both sides.
    """
    scale = min(box_width_in / image_width_px, box_height_in / image_height_px)
    return image_width_px * scale, image_height_px * scale
