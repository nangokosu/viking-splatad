"""Build the redesigned 24 x 36 in poster (v2) as a .pptx and, optionally, its PDF.

Usage (from the repo root; python-pptx, matplotlib and PyVista live in the viz environment):
    uv run --project viz python docs/proposal/poster/build_poster_v2.py --variant curves --pdf
    uv run --project viz python docs/proposal/poster/build_poster_v2.py --variant curves_3d --pdf

Variants:
    curves     results show the table and the June / May threshold curves (large).
    curves_3d  the same plus the June 3D hit-difference figure (the curves get less height).

What it does:
    1. Reads every number from the committed whole-scene summaries (``poster_numbers.py``) and the figure
       manifests; nothing on the poster is typed by hand.
    2. Lays the page out on a fixed grid: text is measured (``poster_layout.py``) so every box is sized to its
       content; the top cards grow to their tallest column, and when the results section is left too little
       height (curves below ``MIN_CURVE_HEIGHT_IN``, or a 3D row too short for its takeaway) the build raises.
    3. Renders the poster-scale figures at exactly their slot size by calling ``poster_threshold_figure.py``
       and ``poster_hit_difference_figure.py`` (skipped when the existing PNG's sidecar already matches).
    4. Opens the original poster (for its slide size, master and the three photos), clears the slide and
       draws the new layout, then saves ``build/<name>.pptx``; with ``--pdf`` converts it with LibreOffice
       into ``docs/proposal/<name>.pdf`` (the deliverable).

The original poster ``docs/proposal/Viking-Forest-Poster-Obstacles.pptx`` is only read, never written.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Repo root: this file lives in docs/proposal/poster/.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
POSTER_DIR = Path(__file__).resolve().parent
# The poster this redesign starts from (read only).
ORIGINAL_POSTER_PATH = PROJECT_ROOT / "docs" / "proposal" / "Viking-Forest-Poster-Obstacles.pptx"
# Intermediate .pptx files (not the deliverable; git-ignored) and the deliverable PDFs.
BUILD_DIR = POSTER_DIR / "build"
FIGURE_DIR = POSTER_DIR / "figures"
PDF_DIR = PROJECT_ROOT / "docs" / "proposal"
# Output stem per variant.
OUTPUT_STEMS = {"curves": "Viking-Forest-Poster-Obstacles-v2", "curves_3d": "Viking-Forest-Poster-Obstacles-v2-3d"}
VARIANTS = tuple(OUTPUT_STEMS)
# Committed inputs.
SUMMARY_PATH_TEMPLATE = "reports/gaussian_occupancy/{session}/gaussian_occupancy_summary.json"
MANIFEST_PATH_TEMPLATE = "reports/gaussian_occupancy/figures/{session}/figures_manifest.json"
EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"
# Session whose 3D hit difference is shown (the tall-grass question).
HIT_DIFFERENCE_SESSION = "june_tall_grass"

# ---- Page grid (inches). A presentation choice; every other position is derived from these. ----
PAGE_WIDTH_IN = 24.0
PAGE_HEIGHT_IN = 36.0
MARGIN_IN = 0.5
GUTTER_IN = 0.4
# Banner: padding above and below its text, and the title-to-byline gap as a fraction of the title size.
BANNER_PADDING_IN = 0.35
BANNER_TITLE_GAP_FRACTION = 0.2
CARD_PADDING_IN = 0.4
# Relative widths of the three top columns (motivation, method, evaluation): the method column carries the
# formula, so it is a little wider.
TOP_COLUMN_WEIGHTS = (1.0, 1.1, 1.0)
# Height cap of the reference-box image in the evaluation card (keeps that column within the top band).
REFERENCE_IMAGE_MAX_HEIGHT_IN = 2.4
# Vertical gaps: between blocks in a card, and after a sub-heading.
BLOCK_GAP_IN = 0.3
HEADING_GAP_IN = 0.12
# Share of the results width taken by the table; the callout grid (CALLOUT_GRID_COLUMNS per row) takes the rest.
TABLE_WIDTH_SHARE = 0.64
# Inner padding of every table cell (inches).
TABLE_CELL_INSET_IN = 0.08
CALLOUT_GRID_COLUMNS = 2
# Smallest threshold-curve height the curves variant accepts (curves take all leftover height there) before it
# reports the page as overfull.
MIN_CURVE_HEIGHT_IN = 5.5
# curves_3d: the curves get this fixed, shorter height so the 3D hit-difference row below takes the rest (the
# scene render is about 2:1, so its size is set by the row height). Chosen by rendering and inspecting.
CURVES_3D_HEIGHT_IN = 4.3
# curves_3d bottom row: the 3D figure's (render + legend) share of the row width; the takeaway panel takes the rest.
HIT_DIFFERENCE_WIDTH_SHARE = 0.7

# ---- Banner text (user's wording choices, 2026-11-27). Each title line must fit on one line of the banner. ----
TITLE_LINES = ("Seeing through tall grass: 4D radar for", "3D Gaussian Splatting of forest scenes")
AUTHORS_TEXT = "Khai Nguyen, Dorsa Molaverdi, Neel Joshi"
AFFILIATION_TEXT = "Computational Imaging"
# Separator between the authors and the affiliation in the byline.
BYLINE_SEPARATOR = "  ·  "

# ---- Type scale (points at print size; the slide is the print, 1 pt = 1 pt). ----
TITLE_PT = 66
BYLINE_PT = 34
SECTION_PT = 46
SUBHEAD_PT = 32
BODY_PT = 26
CAPTION_PT = 21
FORMULA_PT = 30
REFERENCE_PT = 19
TABLE_HEADER_PT = 22
TABLE_MEAN_PT = 28
TABLE_RANGE_PT = 19
CALLOUT_VALUE_PT = 50
CALLOUT_LABEL_PT = 22
PARAGRAPH_GAP_PT = 10

# ---- Colours (RGB hex). The navy family is the original poster's; arm colours come from configs/eval.yaml. ----
NAVY_HEX = "17385D"
BANNER_HEX = "112A46"
BODY_HEX = "1F2933"
CAPTION_HEX = "4A5A68"
PAGE_HEX = "EEF2F6"
CARD_HEX = "FFFFFF"
CARD_LINE_HEX = "C9D4DF"
TINT_HEX = "E3ECF5"
TABLE_RULE_HEX = "C9D4DF"
WHITE_HEX = "FFFFFF"
# Colour of a negative gap callout (matplotlib tab10 red, the palette the arm colours come from).
NEGATIVE_HEX = "D62728"
CARD_LINE_PT = 2.0
# Rounded-corner radius of cards as a fraction of the shorter side (python-pptx adjustment value).
CARD_CORNER_RADIUS_IN = 0.2
# Largest OOXML roundRect adjustment (radius = half the shorter side: a full pill).
MAX_CORNER_ADJUSTMENT = 0.5
EMU_PER_INCH = 914400
# Largest value of an 8-bit colour channel (float channels in [0, 1] scale to it).
RGB_CHANNEL_MAX = 255

# Arm keys of the figure settings, by SplatAD method (configs/eval.yaml figures.arm_order).
ARM_KEY_BY_METHOD = {"splatad": "stock", "splatad-radar-init": "radar_init", "splatad-radar-init-lidar": "radar_init_loss"}
# Picture order in the original poster: May photo, June photo, June reference boxes (left to right, top down).
ORIGINAL_PICTURE_ROLES = ("may_photo", "june_photo", "reference_boxes", "old_threshold_curves", "old_hidden_figure")
# The pictures the new layout reuses (the two old results figures are replaced by poster-scale renders).
REUSED_PICTURE_ROLES = ("may_photo", "june_photo", "reference_boxes")
# The two photos of the motivation card, left to right.
PHOTO_PAIR_ROLES = ("may_photo", "june_photo")


def load_module(module_name: str, path: Path):
    """Import a script that is not part of a package.

    Args:
        module_name: Name to register the module under.
        path: File path.

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


numbers = load_module("poster_numbers_for_build", POSTER_DIR / "poster_numbers.py")
layout = load_module("poster_layout_for_build", POSTER_DIR / "poster_layout.py")
Paragraph, TextRun, Block = layout.Paragraph, layout.TextRun, layout.Block


# ============================================================ facts read from the committed data

@dataclass(frozen=True)
class PosterFacts:
    """Every data-derived value the poster text uses.

    Attributes:
        metrics: session -> method -> metric -> MetricRange.
        gaps: session -> compared method -> metric -> GapResult.
        threshold_text: Binding P2 threshold, e.g. ``0.5``.
        scored_thresholds: Every P2 threshold the summaries score (the curves mark each), e.g. (0.5, 0.8).
        voxel_size_m: Scoring cell edge (summary provenance).
        run_count: Repeats per arm (identical across arms and sessions, checked).
        training_steps: Largest training step among the scored checkpoints.
        steps_vary: True when the scored checkpoints are not all at ``training_steps``.
        obstacle_cells: session -> number of scored obstacle cells (figure manifest).
        hit_counts: Hit-difference cell counts of ``HIT_DIFFERENCE_SESSION`` (figure manifest).
        arm_colours: method -> RGB hex of the arm (tab10 by arm position, as in the figures).
        both_hex: RGB hex of cells both compared arms fill (configs/eval.yaml figures.colours.both).
    """

    metrics: Dict[str, Any]
    gaps: Dict[str, Any]
    threshold_text: str
    scored_thresholds: Tuple[str, ...]
    voxel_size_m: float
    run_count: int
    training_steps: int
    steps_vary: bool
    obstacle_cells: Dict[str, int]
    hit_counts: Dict[str, int]
    arm_colours: Dict[str, str]
    both_hex: str


def single_value(values: Sequence[Any], what: str) -> Any:
    """The one value of a collection that must not vary.

    Args:
        values: The collected values.
        what: Name for the error message.

    Returns:
        The common value.

    Raises:
        ValueError: When the values differ (the poster would otherwise state one of several).
    """
    distinct = sorted(set(values))
    if len(distinct) != 1:
        raise ValueError(f"{what} differs across arms/sessions: {distinct}")
    return distinct[0]


def figure_colour_hex() -> Tuple[Dict[str, str], str]:
    """Each arm's line colour and the "both arms" cell colour, exactly as the figures draw them.

    Returns:
        (method -> RGB hex from ``figures_mpl._arm_colour`` on the configured arm order,
         RGB hex of the configured ``both`` colour).
    """
    import yaml
    from viking_splatad.viz.figure_runs import load_figure_settings
    from viking_splatad.viz.figures_mpl import _arm_colour
    # Step 1: the figure settings of configs/eval.yaml (the same loader the figure scripts use).
    eval_config = yaml.safe_load(EVAL_CONFIG_PATH.read_text())
    settings = load_figure_settings(eval_config["eval"])
    # Step 2: RGBA floats in [0, 1] -> two-digit hex per channel.
    colours = {}
    for method, arm_key in ARM_KEY_BY_METHOD.items():
        red, green, blue, _alpha = _arm_colour(settings, arm_key)
        colours[method] = "".join(f"{round(channel * RGB_CHANNEL_MAX):02X}" for channel in (red, green, blue))
    # Step 3: the configured 0-255 "both" colour.
    both_hex = "".join(f"{channel:02X}" for channel in settings.colours["both"])
    return colours, both_hex


def read_facts() -> PosterFacts:
    """Collect every data-derived value of the poster from the committed summaries and manifests.

    Returns:
        The facts.
    """
    summaries = numbers.load_all_summaries()
    manifests = {session: json.loads((PROJECT_ROOT / MANIFEST_PATH_TEMPLATE.format(session=session)).read_text())
                 for session in numbers.SESSIONS}
    # Step 1: the binding threshold = the first P2 threshold of every summary (they must agree).
    threshold_text = single_value([summary["p2_thresholds"][0] for summary in summaries.values()], "binding threshold")
    # Step 2: per-arm metrics and gaps at that threshold.
    metrics = {session: numbers.load_session_metrics(summary, threshold_text) for session, summary in summaries.items()}
    gaps = {session: numbers.arm_gaps(summary, threshold_text) for session, summary in summaries.items()}
    # Step 3: run count, training steps and voxel size, each required to be one value.
    run_count = single_value([metric.run_count for session_metrics in metrics.values()
                              for arm_metrics in session_metrics.values() for metric in arm_metrics.values()], "run count")
    # Scored checkpoints are not all at the same step (two May runs were scored at an earlier checkpoint), so
    # the poster prints the largest step and marks it approximate when they differ.
    scored_steps = [run["image_step"] for summary in summaries.values() for arm in summary["arms"].values()
                    for run in arm["runs"]]
    training_steps = max(scored_steps)
    steps_vary = len(set(scored_steps)) > 1
    voxel_size_m = single_value([summary["provenance"]["voxel_size_m"] for summary in summaries.values()], "voxel size")
    # Step 4: figure-manifest counts.
    obstacle_cells = {session: manifest["reported_numbers"]["threshold_curves_scene"]["cell_counts"]["on_obstacle"]
                      for session, manifest in manifests.items()}
    hit_counts = manifests[HIT_DIFFERENCE_SESSION]["reported_numbers"]["hit_difference_scene"]["difference"]
    check_hit_difference_arms(hit_counts, manifests[HIT_DIFFERENCE_SESSION])
    # The figures were drawn at the threshold the table uses.
    for session, manifest in manifests.items():
        if str(manifest["binding_threshold"]) != threshold_text:
            raise ValueError(f"{session} figures use threshold {manifest['binding_threshold']}, table {threshold_text}")
    arm_colours, both_hex = figure_colour_hex()
    scored_thresholds = single_value([tuple(summary["p2_thresholds"]) for summary in summaries.values()],
                                     "scored thresholds")
    return PosterFacts(metrics=metrics, gaps=gaps, threshold_text=threshold_text, scored_thresholds=scored_thresholds,
                       voxel_size_m=voxel_size_m,
                       run_count=run_count, training_steps=training_steps, steps_vary=steps_vary,
                       obstacle_cells=obstacle_cells,
                       hit_counts=hit_counts, arm_colours=arm_colours, both_hex=both_hex)


def check_hit_difference_arms(hit_counts: Dict[str, int], manifest: Dict[str, Any]) -> None:
    """Confirm which arm each hit-difference set belongs to, so the poster's colour legend names the right arm.

    Identity: an arm's hits = the cells only it fills + the cells both fill.

    Args:
        hit_counts: ``only_first``, ``only_second`` and ``both`` cell counts.
        manifest: The session's figures manifest (``scene_counts.outcome_counts_by_arm``).

    Raises:
        ValueError: When the counts do not add up to radar init + loss (first) and stock (second).
    """
    arm_hits = {arm: counts["hit"] for arm, counts in
                manifest["reported_numbers"]["scene_counts"]["outcome_counts_by_arm"].items()}
    first_arm = ARM_KEY_BY_METHOD[numbers.LOSS_METHOD]
    second_arm = ARM_KEY_BY_METHOD[numbers.BASELINE_METHOD]
    if hit_counts["only_first"] + hit_counts["both"] != arm_hits[first_arm] or \
            hit_counts["only_second"] + hit_counts["both"] != arm_hits[second_arm]:
        raise ValueError(f"hit-difference sets {hit_counts} do not match arm hits {arm_hits} for "
                         f"first = {first_arm}, second = {second_arm}")


def check_measuring_font() -> None:
    """Refuse to lay out with a fallback font: line breaks and card heights depend on the measured font.

    Raises:
        RuntimeError: When matplotlib does not resolve ``FONT_FAMILY`` to a font file of that family.
    """
    from matplotlib.font_manager import FontProperties, findfont
    resolved = Path(findfont(FontProperties(family=layout.FONT_FAMILY), fallback_to_default=False))
    if not resolved.stem.lower().startswith(layout.FONT_FAMILY.lower()):
        raise RuntimeError(f"text is measured with {resolved.name}, not {layout.FONT_FAMILY}; install "
                           f"{layout.FONT_FAMILY} so the layout matches the rendered poster")


def thousands_steps_text(step_count: int) -> str:
    """Training length as printed, e.g. 20000 -> ``20k``.

    Args:
        step_count: Training steps.

    Returns:
        ``<n>k`` when a whole number of thousands, else the plain count with separators.
    """
    steps_per_thousand = 1000
    if step_count % steps_per_thousand == 0:
        return f"{step_count // steps_per_thousand}k"
    return f"{step_count:,}"


def points_text(gap: Any) -> str:
    """A gap in percentage points as printed in callouts: ``+3.8 pts`` (true minus sign when negative).

    Args:
        gap: ``GapResult``.

    Returns:
        Text from ``poster_numbers.format_gap``.
    """
    return numbers.format_gap(gap)


def results_takeaway(facts: PosterFacts) -> List[TextRun]:
    """The results conclusion, worded from the gaps and separation flags.

    Every clause is checked against the data first; a clause whose premise does not hold raises, so the
    poster can never state a direction or a separation the numbers do not support.

    Args:
        facts: Poster facts.

    Returns:
        Runs of one paragraph.

    Raises:
        ValueError: When a premise of the wording fails.
    """
    june, may = "june_tall_grass", "may_short_grass"
    loss, init = numbers.LOSS_METHOD, numbers.INIT_METHOD
    june_loss, may_loss = facts.gaps[june][loss], facts.gaps[may][loss]
    # Step 1: premises of each clause.
    premises = {
        "loss raises recall in both sessions, ranges separated":
            all(gap["recall"].gap_points > 0 and gap["recall"].separated for gap in (june_loss, may_loss)),
        "June precision ranges overlap": june_loss["precision"].separated is False,
        "May precision falls, ranges separated":
            may_loss["precision"].gap_points < 0 and may_loss["precision"].separated is True,
        "seeds alone lower recall in both sessions":
            all(facts.gaps[session][init]["recall"].gap_points < 0 for session in (june, may)),
    }
    failed = [name for name, holds in premises.items() if not holds]
    if failed:
        raise ValueError(f"results wording no longer matches the data: {failed}")
    # Step 2: the sentence, numbers from the gaps.
    return [
        TextRun("Takeaway: ", bold=True, colour_hex=NAVY_HEX),
        TextRun("radar seeds plus the opacity loss fill more obstacle cells than stock in both seasons "
                f"(recall {points_text(june_loss['recall'])} in June, {points_text(may_loss['recall'])} in May; "
                f"ranges separated). In June precision holds ({points_text(june_loss['precision'])}, ranges overlap); "
                "in May precision falls "
                f"{points_text(numbers.GapResult(abs(may_loss['precision'].gap_points), may_loss['precision'].separated)).lstrip('+')}"
                ". Seeds alone lower recall in both seasons."),
    ]


# ============================================================ content (text) of each card

def body(text_runs: Sequence[TextRun], font_pt: float = BODY_PT, colour_hex: str = BODY_HEX,
         space_after_pt: float = PARAGRAPH_GAP_PT, align: str = "left") -> Paragraph:
    """A body paragraph.

    Args:
        text_runs: Runs, or one string per run.
        font_pt: Size.
        colour_hex: Default colour.
        space_after_pt: Gap after.
        align: ``left`` or ``center``.

    Returns:
        The paragraph.
    """
    runs = tuple(run if isinstance(run, TextRun) else TextRun(run) for run in text_runs)
    return Paragraph(runs=runs, font_pt=font_pt, space_after_pt=space_after_pt, align=align, colour_hex=colour_hex)


def subheading(text: str) -> Paragraph:
    """A navy bold sub-heading paragraph.

    Args:
        text: Heading text.

    Returns:
        The paragraph.
    """
    return Paragraph(runs=(TextRun(text, bold=True),), font_pt=SUBHEAD_PT, colour_hex=NAVY_HEX)


def section_title(text: str) -> Paragraph:
    """A card's section title.

    Args:
        text: Title text.

    Returns:
        The paragraph.
    """
    return Paragraph(runs=(TextRun(text, bold=True),), font_pt=SECTION_PT, colour_hex=NAVY_HEX)


def text_block(paragraphs: Sequence[Paragraph], width_in: float, gap_after_in: float = BLOCK_GAP_IN,
               tinted: bool = False) -> Block:
    """A measured text block.

    Args:
        paragraphs: Paragraphs.
        width_in: Box width.
        gap_after_in: Gap below.
        tinted: Draw it on a light tint panel (the research question).

    Returns:
        The block, its height measured.
    """
    # A tinted panel insets its text by CARD_PADDING_IN / 2 on each side (as ``draw_blocks`` draws it) and adds
    # the same total padding vertically (the text is centred in the panel).
    inner_width_in = width_in - (2 * CARD_PADDING_IN / 2 if tinted else 0.0)
    height_in =layout.text_height_in(paragraphs, inner_width_in) + (CARD_PADDING_IN if tinted else 0.0)
    return Block(kind="text", height_in=height_in, gap_after_in=gap_after_in,
                 payload={"paragraphs": list(paragraphs), "tinted": tinted})


def heading_and_text(heading: str, paragraphs: Sequence[Paragraph], width_in: float) -> List[Block]:
    """A sub-heading block followed by its text block.

    Args:
        heading: Sub-heading text.
        paragraphs: Body paragraphs.
        width_in: Box width.

    Returns:
        Two blocks.
    """
    return [text_block([subheading(heading)], width_in, gap_after_in=HEADING_GAP_IN),
            text_block(paragraphs, width_in)]


def picture_pair_block(images: Dict[str, Tuple[bytes, int, int]], captions: Sequence[Paragraph], width_in: float) -> Block:
    """Two photos side by side with a caption under each.

    Args:
        images: role -> (bytes, width px, height px) for ``may_photo`` and ``june_photo``.
        captions: One caption paragraph per photo.
        width_in: Width of the pair.

    Returns:
        The block (height = the taller photo at the shared width + the taller caption).
    """
    photo_width_in = (width_in - GUTTER_IN) / 2
    # Each photo keeps its own aspect at the shared width; the row is as tall as the taller one.
    photo_heights_in = {role: photo_width_in * images[role][2] / images[role][1] for role in PHOTO_PAIR_ROLES}
    photo_height_in = max(photo_heights_in.values())
    caption_height_in = max(layout.text_height_in([caption], photo_width_in) for caption in captions)
    return Block(kind="photo_pair", height_in=photo_height_in + HEADING_GAP_IN + caption_height_in,
                 gap_after_in=BLOCK_GAP_IN,
                 payload={"photo_width_in": photo_width_in, "photo_height_in": photo_height_in,
                          "photo_heights_in": photo_heights_in, "captions": list(captions)})


def picture_block(image: Tuple[bytes, int, int], width_in: float, max_height_in: float, gap_after_in: float) -> Block:
    """One picture fitted to a width (and a height cap), centred.

    Args:
        image: (bytes, width px, height px).
        width_in: Available width.
        max_height_in: Height cap.
        gap_after_in: Gap below.

    Returns:
        The block.
    """
    _blob, image_width_px, image_height_px = image
    fitted_width_in, fitted_height_in = layout.fit_image(image_width_px, image_height_px, width_in, max_height_in)
    return Block(kind="picture", height_in=fitted_height_in, gap_after_in=gap_after_in,
                 payload={"image": image, "width_in": fitted_width_in})


def arm_name_runs(facts: PosterFacts, method: str) -> TextRun:
    """An arm's name as a bold run in its figure colour, so text, table and curves share one colour key.

    Args:
        facts: Poster facts (arm colours).
        method: SplatAD method of the arm.

    Returns:
        The run.
    """
    return TextRun(numbers.ARM_LABEL_BY_METHOD[method], bold=True, colour_hex=facts.arm_colours[method])


def motivation_blocks(images, width_in: float) -> List[Block]:
    """Motivation card content.

    Args:
        images: Original poster pictures by role.
        width_in: Card inner width.

    Returns:
        Blocks, top to bottom.
    """
    captions = [body([TextRun("May: ", bold=True), "short grass, trunk bases visible."], CAPTION_PT, CAPTION_HEX, 0),
                body([TextRun("June: ", bold=True), "tall grass hides lower trunks."], CAPTION_PT, CAPTION_HEX, 0)]
    return [
        text_block([section_title("Motivation")], width_in, gap_after_in=BLOCK_GAP_IN),
        text_block([body(["Forest robots need 3D maps of ", TextRun("obstacles", bold=True),
                          ": tree trunks, rocks and bushes. In June, tall grass blocks LiDAR beams and camera "
                          "views, leaving gaps where obstacles stand."])], width_in),
        picture_pair_block(images, captions, width_in),
        *heading_and_text("What 4D radar adds", [body([
            "Radar sees through grass but places returns less precisely than LiDAR. Returns that recur "
            "across scans hint at solid geometry the model missed."])], width_in),
        text_block([body([TextRun("Can radar cues recover obstacles that LiDAR never reached?", bold=True)],
                         BODY_PT, NAVY_HEX, 0)], width_in, gap_after_in=0.0, tinted=True),
    ]


def method_blocks(facts: PosterFacts, width_in: float) -> List[Block]:
    """Method card content.

    Args:
        facts: Poster facts.
        width_in: Card inner width.

    Returns:
        Blocks, top to bottom.
    """
    formula = Paragraph(runs=(TextRun("loss = λ · mean[ −log(opacity) ]", bold=True),),
                        font_pt=FORMULA_PT, align="center", colour_hex=NAVY_HEX)
    stock, init, loss = numbers.METHOD_ORDER
    steps_text = f"{'≈' if facts.steps_vary else ''}{thousands_steps_text(facts.training_steps)}"
    return [
        text_block([section_title("Method")], width_in, gap_after_in=BLOCK_GAP_IN),
        *heading_and_text("1  Find recurring radar returns", [body([
            "Align radar scans with the robot's motion. Returns that recur in one place likely come from "
            "solid objects."])], width_in),
        *heading_and_text("2  Seed the 3D model", [body([
            "Add a 3D Gaussian at each recurring return, coloured from the camera where LiDAR sees clearly."])],
            width_in),
        *heading_and_text("3  Keep likely obstacles solid", [body([
            "A label-free loss keeps Gaussians near radar returns opaque:"])], width_in),
        text_block([formula], width_in, tinted=True),
        text_block([body(["We compare ", arm_name_runs(facts, stock), " SplatAD (camera + LiDAR), ",
                          arm_name_runs(facts, init), " (seeds only) and ", arm_name_runs(facts, loss),
                          f". Each is trained {facts.run_count} times for {steps_text} steps per season on the "
                          "same frames. Radar is never rendered."], space_after_pt=0)], width_in, gap_after_in=0.0),
    ]


def evaluation_blocks(facts: PosterFacts, images, width_in: float) -> List[Block]:
    """Data, scoring and references card content.

    Args:
        facts: Poster facts.
        images: Original poster pictures by role.
        width_in: Card inner width.

    Returns:
        Blocks, top to bottom.
    """
    references = [
        body(["[1] Viking Hill Dataset. arXiv:2606.19154 (2026)."], REFERENCE_PT, CAPTION_HEX, 2),
        body(["[2] SplatAD. CVPR 2025. arXiv:2411.16816."], REFERENCE_PT, CAPTION_HEX, 2),
        body(["[3] Kerbl et al. 3D Gaussian Splatting. ACM TOG 2023."], REFERENCE_PT, CAPTION_HEX, 0),
    ]
    return [
        text_block([section_title("Data and scoring")], width_in, gap_after_in=BLOCK_GAP_IN),
        text_block([body(["A ground robot drives one forest trail in May and June [1]. SplatAD [2] renders camera "
                          "and LiDAR from 3D Gaussians [3]."])], width_in),
        picture_block(images["reference_boxes"], width_in, max_height_in=REFERENCE_IMAGE_MAX_HEIGHT_IN,
                      gap_after_in=HEADING_GAP_IN),
        text_block([body(["Reference boxes (June), for scoring only."], CAPTION_PT, CAPTION_HEX, 0)], width_in),
        text_block([subheading("Scoring")], width_in, gap_after_in=HEADING_GAP_IN),
        text_block([
            body([f"Cells of {facts.voxel_size_m:.2f} m; a cell is filled when Gaussian occupancy ≥ "
                  f"{facts.threshold_text}."]),
            body([TextRun("Recall: ", bold=True), "obstacle cells filled."], space_after_pt=4),
            body([TextRun("Precision: ", bold=True), "filled cells near obstacles that are obstacles."],
                 space_after_pt=4),
            body([TextRun("IoU: ", bold=True), "overlap ÷ union."], space_after_pt=0),
        ], width_in),
        text_block(references, width_in, gap_after_in=0.0),
    ]


# ============================================================ results section

def callout_specs(facts: PosterFacts) -> List[Dict[str, Any]]:
    """The four stat callouts of the results card, each a gap against stock.

    Args:
        facts: Poster facts.

    Returns:
        Dicts with ``value`` text, ``label`` text and ``colour`` hex (arm colour when the gap is positive,
        the negative colour otherwise).
    """
    june, may = "june_tall_grass", "may_short_grass"
    loss = numbers.LOSS_METHOD
    chosen = [  # (session, method, metric, label)
        (june, loss, "recall", "June recall"),
        (june, loss, "iou", "June IoU"),
        (may, loss, "recall", "May recall"),
        (may, loss, "precision", "May precision"),
    ]
    specs = []
    for session, method, metric, label in chosen:
        gap = facts.gaps[session][method][metric]
        separation = (numbers.UNAVAILABLE_TEXT if gap.separated is None
                      else numbers.SEPARATED_TEXT if gap.separated else numbers.OVERLAP_TEXT)
        colour = facts.arm_colours[method] if gap.gap_points >= 0 else NEGATIVE_HEX
        comparison = f"{numbers.ARM_LABEL_BY_METHOD[method]} vs {numbers.ARM_LABEL_BY_METHOD[numbers.BASELINE_METHOD]}"
        specs.append({"value": points_text(gap), "label": f"{label}: {comparison}, {separation}", "colour": colour})
    return specs


def results_table_cells(facts: PosterFacts) -> Dict[str, Any]:
    """Header and body cells of the results table, with the best mean per session and metric marked.

    Args:
        facts: Poster facts.

    Returns:
        ``columns`` [(session, method)], ``rows`` [(metric label, [(mean text, range text, is_best)])].
    """
    columns = [(session, method) for session in numbers.SESSIONS for method in numbers.METHOD_ORDER]
    rows = []
    for metric_name, metric_label in numbers.METRIC_LABELS:
        # Best = highest mean of the session (all three metrics are better when higher).
        printed_mean = lambda session, method: numbers.format_percent_range(  # noqa: E731
            facts.metrics[session][method][metric_name]).split(" ", 1)[0]
        best_by_session = {session: max(float(printed_mean(session, method)) for method in numbers.METHOD_ORDER)
                           for session in numbers.SESSIONS}
        cells = []
        for session, method in columns:
            metric = facts.metrics[session][method][metric_name]
            range_text = numbers.format_percent_range(metric).split(" ", 1)[1]
            mean_text = numbers.format_percent_range(metric).split(" ", 1)[0]
            cells.append((mean_text, range_text, float(mean_text) == best_by_session[session]))
        rows.append((metric_label[0].upper() + metric_label[1:], cells))
    return {"columns": columns, "rows": rows}


# ============================================================ figures

def figure_png(kind: str, session: str, variant: str) -> Path:
    """Output path of a poster figure.

    Args:
        kind: ``threshold`` or ``hit_difference``.
        session: Session name.
        variant: Poster variant.

    Returns:
        PNG path under ``figures/``.
    """
    if kind == "threshold":
        return FIGURE_DIR / f"{session}_threshold_curves_{variant}.png"
    return FIGURE_DIR / f"{session}_hit_difference_poster.png"


def sidecar_slot(png_path: Path) -> Optional[Tuple[float, float]]:
    """Slot size recorded in a figure's sidecar JSON.

    Args:
        png_path: The PNG.

    Returns:
        (width, height) in inches, or None when there is no sidecar.
    """
    sidecar_path = png_path.with_suffix(".json")
    if not sidecar_path.exists():
        return None
    sidecar = json.loads(sidecar_path.read_text())
    if "slot_in" in sidecar:
        return tuple(sidecar["slot_in"])
    return sidecar["width_in"], sidecar["height_in"]


def ensure_figure(kind: str, session: str, variant: str, width_in: float, height_in: float, rebuild: bool) -> Path:
    """Render a poster figure at its exact slot size unless an up-to-date one exists.

    Args:
        kind: ``threshold`` or ``hit_difference``.
        session: Session name.
        variant: Poster variant.
        width_in: Slot width.
        height_in: Slot height.
        rebuild: Re-render even when the sidecar matches.

    Returns:
        The PNG path.
    """
    # Step 1: the wanted slot (rounded) and what the existing sidecar records.
    png_path = figure_png(kind, session, variant)
    slot_digits = 2  # slot sizes are compared at 0.01 in, far below what the eye can see on the print
    wanted = (round(width_in, slot_digits), round(height_in, slot_digits))
    existing = sidecar_slot(png_path)
    # The hit-difference figure carries its own stacked legend (arm colours, counts on a second line, which
    # leaves more width to the render); a PNG drawn otherwise is stale.
    sidecar_path = png_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
    legend_matches = kind != "hit_difference" or sidecar.get("legend_stacked") is True
    manifest_path = PROJECT_ROOT / MANIFEST_PATH_TEMPLATE.format(session=session)
    input_matches = sidecar.get("manifest_sha256") == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    # Step 2: reuse the PNG only when size, legend setting and input manifest all match.
    if not rebuild and png_path.exists() and existing is not None and legend_matches and input_matches and \
            tuple(round(value, slot_digits) for value in existing) == wanted:
        return png_path
    # Step 3: otherwise render it at the rounded slot size with the matching figure script.
    script = POSTER_DIR / ("poster_threshold_figure.py" if kind == "threshold" else "poster_hit_difference_figure.py")
    command = [sys.executable, str(script), "--session", session, "--width-in", f"{wanted[0]}",
               "--height-in", f"{wanted[1]}", "--output", str(png_path)] + (
                   ["--stacked-legend"] if kind == "hit_difference" else [])
    print("rendering:", " ".join(command[1:]))
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)
    return png_path


# ============================================================ slide drawing (python-pptx)

def emu(inches: float) -> int:
    """Inches -> EMU (English Metric Units, 914400 per inch).

    Args:
        inches: Length in inches.

    Returns:
        Integer EMU.
    """
    return int(round(inches * EMU_PER_INCH))


def rgb(hex_text: str):
    """python-pptx colour from hex.

    Args:
        hex_text: Six-digit hex.

    Returns:
        ``RGBColor``.
    """
    from pptx.dml.color import RGBColor
    return RGBColor.from_string(hex_text)


def add_rectangle(slide, left_in: float, top_in: float, width_in: float, height_in: float, fill_hex: str,
                  line_hex: Optional[str] = None, rounded: bool = True):
    """Draw a filled (rounded) rectangle with an optional outline and no shadow.

    Args:
        slide: Target slide.
        left_in, top_in, width_in, height_in: Geometry in inches.
        fill_hex: Fill colour.
        line_hex: Outline colour, or None for no outline.
        rounded: Rounded corners when True.

    Returns:
        The shape.
    """
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Pt
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if rounded else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(shape_type, emu(left_in), emu(top_in), emu(width_in), emu(height_in))
    if rounded:
        # The adjustment is the corner radius as a fraction of the shorter side.
        shape.adjustments[0] = min(MAX_CORNER_ADJUSTMENT, CARD_CORNER_RADIUS_IN / min(width_in, height_in))
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill_hex)
    if line_hex is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = rgb(line_hex)
        shape.line.width = Pt(CARD_LINE_PT)
    shape.shadow.inherit = False
    shape.text_frame.text = ""
    return shape


def write_paragraphs(text_frame, paragraphs: Sequence[Paragraph]) -> None:
    """Fill a text frame with styled paragraphs (exact line spacing = the measured line pitch).

    Args:
        text_frame: python-pptx text frame (empty).
        paragraphs: Paragraphs to write.
    """
    from pptx.enum.text import PP_ALIGN
    from pptx.util import Pt
    for paragraph_index, paragraph in enumerate(paragraphs):
        target = text_frame.paragraphs[0] if paragraph_index == 0 else text_frame.add_paragraph()
        target.alignment = PP_ALIGN.CENTER if paragraph.align == "center" else PP_ALIGN.LEFT
        target.line_spacing = Pt(paragraph.font_pt * layout.LINE_PITCH_FACTOR)
        target.space_after = Pt(paragraph.space_after_pt)
        target.space_before = Pt(0)
        # Runs carry their own spacing (a run ends with a space when the next word follows it).
        for text_run in paragraph.runs:
            run = target.add_run()
            run.text = text_run.text
            run.font.size = Pt(paragraph.font_pt)
            run.font.bold = text_run.bold
            run.font.name = layout.FONT_FAMILY
            run.font.color.rgb = rgb(text_run.colour_hex or paragraph.colour_hex)


def add_text(slide, left_in: float, top_in: float, width_in: float, height_in: float,
             paragraphs: Sequence[Paragraph], anchor: str = "top"):
    """Draw a borderless text box with zero insets.

    Args:
        slide: Target slide.
        left_in, top_in, width_in, height_in: Geometry in inches.
        paragraphs: Content.
        anchor: ``top`` or ``middle``.

    Returns:
        The shape.
    """
    from pptx.enum.text import MSO_ANCHOR, MSO_AUTO_SIZE
    box = slide.shapes.add_textbox(emu(left_in), emu(top_in), emu(width_in), emu(height_in))
    frame = box.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_left = frame.margin_right = frame.margin_top = frame.margin_bottom = 0
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE if anchor == "middle" else MSO_ANCHOR.TOP
    write_paragraphs(frame, paragraphs)
    return box


def add_picture(slide, image_bytes: bytes, left_in: float, top_in: float, width_in: float, height_in: float):
    """Place an image.

    Args:
        slide: Target slide.
        image_bytes: Encoded image.
        left_in, top_in, width_in, height_in: Geometry in inches.

    Returns:
        The picture shape.
    """
    return slide.shapes.add_picture(io.BytesIO(image_bytes), emu(left_in), emu(top_in), emu(width_in), emu(height_in))


def draw_blocks(slide, blocks: Sequence[Block], left_in: float, width_in: float, images) -> None:
    """Draw stacked blocks of one column.

    Args:
        slide: Target slide.
        blocks: Blocks with ``top_in`` set.
        left_in: Column left edge.
        width_in: Column width.
        images: Pictures by role.
    """
    for block in blocks:
        if block.kind == "text":
            # Text: plain box, or a tint panel with the text inset and vertically centred.
            paragraphs = block.payload["paragraphs"]
            if block.payload["tinted"]:
                add_rectangle(slide, left_in, block.top_in, width_in, block.height_in, TINT_HEX)
                inset_in = CARD_PADDING_IN / 2
                add_text(slide, left_in + inset_in, block.top_in, width_in - 2 * inset_in, block.height_in,
                         paragraphs, anchor="middle")
            else:
                add_text(slide, left_in, block.top_in, width_in, block.height_in, paragraphs)
        elif block.kind == "photo_pair":
            # Photo pair: May left, June right, a gutter apart; each caption fills the space below its photo.
            photo_width_in, photo_height_in = block.payload["photo_width_in"], block.payload["photo_height_in"]
            for photo_index, role in enumerate(PHOTO_PAIR_ROLES):
                photo_left_in = left_in + photo_index * (photo_width_in + GUTTER_IN)
                add_picture(slide, images[role][0], photo_left_in, block.top_in, photo_width_in,
                            block.payload["photo_heights_in"][role])
                caption_top_in = block.top_in + photo_height_in + HEADING_GAP_IN
                add_text(slide, photo_left_in, caption_top_in, photo_width_in,
                         block.height_in - photo_height_in - HEADING_GAP_IN, [block.payload["captions"][photo_index]])
        elif block.kind == "picture":
            # Single picture: centred horizontally in the column.
            picture_width_in = block.payload["width_in"]
            add_picture(slide, block.payload["image"][0], left_in + (width_in - picture_width_in) / 2, block.top_in,
                        picture_width_in, block.height_in)


def draw_card(slide, left_in: float, top_in: float, width_in: float, height_in: float) -> None:
    """A white section card with a thin outline.

    Args:
        slide: Target slide.
        left_in, top_in, width_in, height_in: Geometry in inches.
    """
    add_rectangle(slide, left_in, top_in, width_in, height_in, CARD_HEX, CARD_LINE_HEX)


def draw_table(slide, cells: Dict[str, Any], facts: PosterFacts, left_in: float, top_in: float, width_in: float) -> float:
    """The results table: two header rows (season, arm) and one row per metric (mean over range).

    Column widths and row heights are measured from the text: the label column is as wide as its widest label,
    the value columns share the rest equally, and each row is as tall as its tallest wrapped cell, so no cell
    text is squeezed or breaks inside a word.

    Args:
        slide: Target slide.
        cells: ``results_table_cells`` output.
        facts: Poster facts (arm colours).
        left_in, top_in, width_in: Geometry.

    Returns:
        Table height in inches.
    """
    from pptx.enum.text import MSO_ANCHOR
    columns, rows = cells["columns"], cells["rows"]
    arm_count = len(numbers.METHOD_ORDER)

    def header(text: str, colour: str = WHITE_HEX, align: str = "center") -> Paragraph:
        """A bold header paragraph.

        Args:
            text: Header text.
            colour: Text colour hex.
            align: ``left`` or ``center``.

        Returns:
            The paragraph at ``TABLE_HEADER_PT``.
        """
        return Paragraph(runs=(TextRun(text, bold=True),), font_pt=TABLE_HEADER_PT, align=align, colour_hex=colour)

    # Step 1: every cell's paragraphs and fill; None marks a cell covered by a merge.
    season_row = [([header(numbers.TABLE_CORNER_LABEL, align="left")], NAVY_HEX)]
    for session in numbers.SESSIONS:
        season_row += [([header(numbers.SESSION_LABELS[session])], NAVY_HEX)] + [None] * (arm_count - 1)
    arm_row = [None] + [([header(numbers.ARM_LABEL_BY_METHOD[method], facts.arm_colours[method])], TINT_HEX)
                        for _session, method in columns]
    grid = [season_row, arm_row]
    for row_index, (metric_label, row_cells) in enumerate(rows):
        fill_hex = CARD_HEX if row_index % 2 == 0 else PAGE_HEX
        metric_row = [([Paragraph(runs=(TextRun(metric_label, bold=True),), font_pt=TABLE_MEAN_PT, colour_hex=NAVY_HEX)],
                       fill_hex)]
        for mean_text, range_text, is_best in row_cells:
            metric_row.append(([Paragraph(runs=(TextRun(mean_text, bold=is_best),), font_pt=TABLE_MEAN_PT, align="center",
                                          colour_hex=NAVY_HEX if is_best else BODY_HEX),
                                Paragraph(runs=(TextRun(range_text),), font_pt=TABLE_RANGE_PT, align="center",
                                          colour_hex=CAPTION_HEX)], fill_hex))
        grid.append(metric_row)
    # Step 2: column widths. Label column = widest single-line label + insets; values share the rest.
    label_width_in = max(sum(layout.word_width_pt(run.text, paragraph.font_pt, run.bold) for run in paragraph.runs)
                         for row in grid if row[0] is not None for paragraph in row[0][0]) / layout.POINTS_PER_INCH
    label_width_in = label_width_in / (1.0 - layout.WIDTH_SAFETY_FRACTION) + 2 * TABLE_CELL_INSET_IN
    value_width_in = (width_in - label_width_in) / len(columns)
    column_width_in = [label_width_in] + [value_width_in] * len(columns)

    def cell_width(row: List[Any], column_index: int) -> float:
        """Width of a cell including the columns its merge covers.

        Only the season row (``grid[0]``) has horizontal merges: its trailing ``None`` cells extend the span.

        Args:
            row: A grid row.
            column_index: Column of the cell.

        Returns:
            Summed width of the spanned columns, in inches.
        """
        span = 1
        while column_index + span < len(row) and row[column_index + span] is None and row is grid[0]:
            span += 1
        return sum(column_width_in[column_index:column_index + span])

    # Step 3: row heights = tallest wrapped cell + insets (the merged corner cell is excluded from row 2).
    row_height_in = []
    for row in grid:
        heights = [layout.text_height_in(entry[0], cell_width(row, column_index) - 2 * TABLE_CELL_INSET_IN)
                   for column_index, entry in enumerate(row) if entry is not None]
        row_height_in.append(max(heights) + 2 * TABLE_CELL_INSET_IN)
    total_height_in = sum(row_height_in)
    # Step 4: the table shape, sized and unstyled (every cell is styled below).
    frame = slide.shapes.add_table(len(grid), len(column_width_in), emu(left_in), emu(top_in), emu(width_in),
                                   emu(total_height_in))
    table = frame.table
    table.first_row = table.horz_banding = False
    for column_index, width in enumerate(column_width_in):
        table.columns[column_index].width = emu(width)
    for row_index, height in enumerate(row_height_in):
        table.rows[row_index].height = emu(height)
    # Step 5: merges (corner over both header rows; each season over its arms), then content.
    table.cell(0, 0).merge(table.cell(1, 0))
    for session_index in range(len(numbers.SESSIONS)):
        first_column = 1 + session_index * arm_count
        table.cell(0, first_column).merge(table.cell(0, first_column + arm_count - 1))
    for row_index, row in enumerate(grid):
        for column_index, entry in enumerate(row):
            if entry is None:
                continue
            paragraphs, fill_hex = entry
            cell = table.cell(row_index, column_index)
            cell.fill.solid()
            cell.fill.fore_color.rgb = rgb(fill_hex)
            cell.margin_left = cell.margin_right = cell.margin_top = cell.margin_bottom = emu(TABLE_CELL_INSET_IN)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.text_frame.text = ""
            write_paragraphs(cell.text_frame, paragraphs)
    return total_height_in


def draw_callouts(slide, specs: Sequence[Dict[str, Any]], left_in: float, top_in: float, width_in: float,
                  grid_columns: int) -> float:
    """A grid of stat callouts (big gap value over a short label) on tinted panels.

    Args:
        slide: Target slide.
        specs: ``callout_specs`` output, filled row by row.
        left_in, top_in, width_in: Geometry of the whole grid.
        grid_columns: Callouts per row.

    Returns:
        Grid height in inches (rows x panel height + gaps between rows).
    """
    callout_width_in = layout.column_widths(width_in, GUTTER_IN, [1.0] * grid_columns)[0]
    labels = [Paragraph(runs=(TextRun(spec["label"]),), font_pt=CALLOUT_LABEL_PT, align="center", colour_hex=CAPTION_HEX)
              for spec in specs]
    inset_in = CARD_PADDING_IN / 2
    # Step 1: one panel height for all (the tallest label decides), so the grid is regular.
    value_height_in = CALLOUT_VALUE_PT * layout.LINE_PITCH_FACTOR / layout.POINTS_PER_INCH
    label_height_in = max(layout.text_height_in([label], callout_width_in - 2 * inset_in) for label in labels)
    panel_height_in = value_height_in + label_height_in + 2 * inset_in
    # Step 2: panels row by row.
    for callout_index, (spec, label) in enumerate(zip(specs, labels)):
        row_index, column_index = divmod(callout_index, grid_columns)
        panel_left_in = left_in + column_index * (callout_width_in + GUTTER_IN)
        panel_top_in = top_in + row_index * (panel_height_in + GUTTER_IN)
        add_rectangle(slide, panel_left_in, panel_top_in, callout_width_in, panel_height_in, TINT_HEX)
        value = Paragraph(runs=(TextRun(spec["value"], bold=True),), font_pt=CALLOUT_VALUE_PT, align="center",
                          colour_hex=spec["colour"])
        add_text(slide, panel_left_in + inset_in, panel_top_in + inset_in, callout_width_in - 2 * inset_in,
                 value_height_in, [value])
        add_text(slide, panel_left_in + inset_in, panel_top_in + inset_in + value_height_in,
                 callout_width_in - 2 * inset_in, label_height_in, [label])
    row_count = -(-len(specs) // grid_columns)  # ceiling division
    return row_count * panel_height_in + (row_count - 1) * GUTTER_IN


# ============================================================ page assembly

def original_pictures(presentation) -> Dict[str, Tuple[bytes, int, int]]:
    """The photos and the reference-box image of the original poster, by role.

    Args:
        presentation: The opened original poster.

    Returns:
        role -> (bytes, width px, height px); roles in ``ORIGINAL_PICTURE_ROLES`` order of appearance.
    """
    from pptx.enum.shapes import MSO_SHAPE_TYPE
    pictures = [shape for shape in presentation.slides[0].shapes if shape.shape_type == MSO_SHAPE_TYPE.PICTURE]
    if len(pictures) != len(ORIGINAL_PICTURE_ROLES):
        raise ValueError(f"original poster has {len(pictures)} pictures, expected {len(ORIGINAL_PICTURE_ROLES)} "
                         f"({', '.join(ORIGINAL_PICTURE_ROLES)}); the roles would be assigned wrongly")
    images = {}
    for role, picture in zip(ORIGINAL_PICTURE_ROLES, pictures):
        if role not in REUSED_PICTURE_ROLES:
            continue
        image_width_px, image_height_px = picture.image.size
        images[role] = (picture.image.blob, image_width_px, image_height_px)
    return images


def clear_slide(slide) -> None:
    """Remove every shape of a slide (the slide size, master and theme stay).

    Args:
        slide: The slide.
    """
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)


def build(variant: str, rebuild_figures: bool) -> Path:
    """Lay out and write one poster variant.

    Args:
        variant: ``curves`` or ``curves_3d``.
        rebuild_figures: Re-render the figures even when their sidecars match.

    Returns:
        Path of the written .pptx.

    Raises:
        ValueError: When the original slide is not the grid's page size, or the results section does not fit
            (curves below ``MIN_CURVE_HEIGHT_IN``, or content running into the bottom row).
    """
    from pptx import Presentation
    check_measuring_font()
    facts = read_facts()
    presentation = Presentation(str(ORIGINAL_POSTER_PATH))
    # Step 0: the page must be the size the grid assumes.
    page_width_in = presentation.slide_width / EMU_PER_INCH
    page_height_in = presentation.slide_height / EMU_PER_INCH
    if (round(page_width_in, 3), round(page_height_in, 3)) != (PAGE_WIDTH_IN, PAGE_HEIGHT_IN):
        raise ValueError(f"original slide is {page_width_in} x {page_height_in} in, expected {PAGE_WIDTH_IN} x {PAGE_HEIGHT_IN}")
    images = original_pictures(presentation)
    slide = presentation.slides[0]
    clear_slide(slide)
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = rgb(PAGE_HEX)

    # Step 1: banner (full bleed) with title, byline and affiliation.
    # Title lines are explicit paragraphs; the last one carries the gap to the byline.
    title_paragraphs = [Paragraph(runs=(TextRun(line, bold=True),), font_pt=TITLE_PT, align="center",
                                  colour_hex=WHITE_HEX,
                                  space_after_pt=TITLE_PT * BANNER_TITLE_GAP_FRACTION if is_last else 0.0)
                        for is_last, line in zip([False] * (len(TITLE_LINES) - 1) + [True], TITLE_LINES)]
    banner_text = title_paragraphs + [
        Paragraph(runs=(TextRun(AUTHORS_TEXT, bold=True), TextRun(BYLINE_SEPARATOR + AFFILIATION_TEXT)),
                  font_pt=BYLINE_PT, align="center", colour_hex=TINT_HEX),
    ]
    banner_width_in = PAGE_WIDTH_IN - 2 * MARGIN_IN
    # A title line that wrapped would silently add a third line: refuse it.
    for paragraph in banner_text:
        if not layout.fits_on_one_line(paragraph, banner_width_in):
            raise ValueError(f"banner line too wide for {banner_width_in} in at {paragraph.font_pt} pt: "
                             f"{' '.join(run.text for run in paragraph.runs)!r}; shorten it or lower its size")
    banner_height_in = layout.text_height_in(banner_text, banner_width_in) + 2 * BANNER_PADDING_IN
    add_rectangle(slide, 0.0, 0.0, PAGE_WIDTH_IN, banner_height_in, BANNER_HEX, rounded=False)
    add_text(slide, MARGIN_IN, 0.0, banner_width_in, banner_height_in, banner_text, anchor="middle")

    # Step 2: three top cards; each column's blocks are measured and stacked, the cards share the tallest height.
    content_width_in = PAGE_WIDTH_IN - 2 * MARGIN_IN
    card_widths = layout.column_widths(content_width_in, GUTTER_IN, TOP_COLUMN_WEIGHTS)
    card_top_in = banner_height_in + GUTTER_IN
    column_blocks = [motivation_blocks(images, card_widths[0] - 2 * CARD_PADDING_IN),
                     method_blocks(facts, card_widths[1] - 2 * CARD_PADDING_IN),
                     evaluation_blocks(facts, images, card_widths[2] - 2 * CARD_PADDING_IN)]
    column_bottoms = [layout.stack_blocks(blocks, card_top_in + CARD_PADDING_IN) for blocks in column_blocks]
    top_card_height_in = max(column_bottoms) + CARD_PADDING_IN - card_top_in
    card_left_in = MARGIN_IN
    for card_width_in, blocks in zip(card_widths, column_blocks):
        draw_card(slide, card_left_in, card_top_in, card_width_in, top_card_height_in)
        draw_blocks(slide, blocks, card_left_in + CARD_PADDING_IN, card_width_in - 2 * CARD_PADDING_IN, images)
        card_left_in += card_width_in + GUTTER_IN

    # Step 3: the full-width results card; fixed-height parts first, the figures take what is left.
    results_top_in = card_top_in + top_card_height_in + GUTTER_IN
    results_bottom_in = PAGE_HEIGHT_IN - MARGIN_IN
    draw_card(slide, MARGIN_IN, results_top_in, content_width_in, results_bottom_in - results_top_in)
    inner_left_in = MARGIN_IN + CARD_PADDING_IN
    inner_width_in = content_width_in - 2 * CARD_PADDING_IN
    cursor_in = results_top_in + CARD_PADDING_IN
    title_paragraphs = [section_title("Experimental results")]
    title_height_in = layout.text_height_in(title_paragraphs, inner_width_in)
    add_text(slide, inner_left_in, cursor_in, inner_width_in, title_height_in, title_paragraphs)
    cursor_in += title_height_in + BLOCK_GAP_IN
    # Table (left) and a 2 x 2 callout grid (right) share one row; the note runs under the table.
    table_width_in = inner_width_in * TABLE_WIDTH_SHARE
    callout_left_in = inner_left_in + table_width_in + GUTTER_IN
    table_height_in = draw_table(slide, results_table_cells(facts), facts, inner_left_in, cursor_in, table_width_in)
    table_note = [body([f"Mean over {facts.run_count} runs; below it, the [min, max] over those runs. Bold: best mean "
                        f"of the season. Threshold {facts.threshold_text}, pooled over all scored obstacle cells "
                        f"(June {facts.obstacle_cells['june_tall_grass']:,}; "
                        f"May {facts.obstacle_cells['may_short_grass']:,})."], CAPTION_PT, CAPTION_HEX, 0)]
    table_note_height_in = layout.text_height_in(table_note, table_width_in)
    add_text(slide, inner_left_in, cursor_in + table_height_in + HEADING_GAP_IN, table_width_in, table_note_height_in,
             table_note)
    callout_height_in = draw_callouts(slide, callout_specs(facts), callout_left_in, cursor_in,
                                      inner_left_in + inner_width_in - callout_left_in, CALLOUT_GRID_COLUMNS)
    cursor_in += max(table_height_in + HEADING_GAP_IN + table_note_height_in, callout_height_in) + BLOCK_GAP_IN

    # Step 4: measure the texts below the curves so the figures can take the rest of the card.
    curve_caption = [body([TextRun("Recall and precision vs occupancy threshold. ", bold=True),
                           "Line: the median run of each version; band: min–max over its "
                           f"{facts.run_count} runs. Dashed lines: thresholds {' and '.join(facts.scored_thresholds)}; "
                           f"the table uses {facts.threshold_text}."],
                          CAPTION_PT, CAPTION_HEX, 0)]
    takeaway = [body(results_takeaway(facts), BODY_PT, BODY_HEX, 0)]
    caption_height_in = layout.text_height_in(curve_caption, inner_width_in)
    session_label_height_in = SUBHEAD_PT * layout.LINE_PITCH_FACTOR / layout.POINTS_PER_INCH
    bottom_in = results_bottom_in - CARD_PADDING_IN
    # Height left for the curves row and the bottom row (takeaway, plus the 3D figure in curves_3d).
    rows_space_in = bottom_in - cursor_in - (session_label_height_in + 2 * HEADING_GAP_IN + caption_height_in
                                             + BLOCK_GAP_IN)
    if variant == "curves_3d":
        # Curves get a fixed, shorter height; the bottom row takes the rest: the 3D figure with its own legend
        # under a one-line heading (left), beside a tinted panel holding the takeaway (right).
        curve_height_in = CURVES_3D_HEIGHT_IN
        bottom_row_height_in = rows_space_in - curve_height_in
        hit_width_in = inner_width_in * HIT_DIFFERENCE_WIDTH_SHARE
        panel_width_in = inner_width_in - hit_width_in - GUTTER_IN
        hit_heading = [Paragraph(runs=(TextRun(f"Where the loss helps (June, threshold {facts.threshold_text}, "
                                               "median runs)", bold=True),),
                                 font_pt=SUBHEAD_PT, align="center", colour_hex=NAVY_HEX)]
        if not layout.fits_on_one_line(hit_heading[0], hit_width_in):
            raise ValueError("3D figure heading does not fit on one line")
        hit_heading_height_in = layout.text_height_in(hit_heading, hit_width_in)
        panel_height_in = layout.text_height_in(takeaway, panel_width_in - CARD_PADDING_IN) + CARD_PADDING_IN
        if panel_height_in > bottom_row_height_in:
            raise ValueError(f"takeaway panel needs {panel_height_in:.2f} in but the 3D row is "
                             f"{bottom_row_height_in:.2f} in; lower CURVES_3D_HEIGHT_IN or widen the panel")
    else:
        bottom_row_height_in = layout.text_height_in(takeaway, inner_width_in - CARD_PADDING_IN) + CARD_PADDING_IN
        curve_height_in = rows_space_in - bottom_row_height_in
    if variant == "curves" and curve_height_in < MIN_CURVE_HEIGHT_IN:
        raise ValueError(f"results section overfull: curves would be {curve_height_in:.2f} in tall "
                         f"(minimum {MIN_CURVE_HEIGHT_IN}); top card column heights "
                         f"{[round(column_bottom - card_top_in, 2) for column_bottom in column_bottoms]} in, results "
                         f"start at {results_top_in:.2f} in; shorten text or shrink the top cards")

    # Step 5: threshold curves, June | May, each under a season label, then their caption.
    curve_widths = layout.column_widths(inner_width_in, GUTTER_IN, [1.0] * len(numbers.SESSIONS))
    figure_left_in = inner_left_in
    for session, curve_width_in in zip(numbers.SESSIONS, curve_widths):
        label = [Paragraph(runs=(TextRun(numbers.SESSION_LABELS[session], bold=True),), font_pt=SUBHEAD_PT,
                           align="center", colour_hex=NAVY_HEX)]
        add_text(slide, figure_left_in, cursor_in, curve_width_in, session_label_height_in, label)
        png_path = ensure_figure("threshold", session, variant, curve_width_in, curve_height_in, rebuild_figures)
        add_picture(slide, png_path.read_bytes(), figure_left_in, cursor_in + session_label_height_in + HEADING_GAP_IN,
                    curve_width_in, curve_height_in)
        figure_left_in += curve_width_in + GUTTER_IN
    cursor_in += session_label_height_in + HEADING_GAP_IN + curve_height_in + HEADING_GAP_IN
    add_text(slide, inner_left_in, cursor_in, inner_width_in, caption_height_in, curve_caption)
    cursor_in += caption_height_in + BLOCK_GAP_IN

    # Step 6: the bottom row, anchored to the card's bottom edge.
    row_top_in = bottom_in - bottom_row_height_in
    if cursor_in > row_top_in + 1e-6:
        raise ValueError(f"results content ends at {cursor_in:.2f} in, below the bottom row top {row_top_in:.2f} in")
    if variant == "curves_3d":
        add_text(slide, inner_left_in, row_top_in, hit_width_in, hit_heading_height_in, hit_heading)
        hit_top_in = row_top_in + hit_heading_height_in + HEADING_GAP_IN
        hit_height_in = row_top_in + bottom_row_height_in - hit_top_in
        png_path = ensure_figure("hit_difference", HIT_DIFFERENCE_SESSION, variant, hit_width_in, hit_height_in,
                                 rebuild_figures)
        add_picture(slide, png_path.read_bytes(), inner_left_in, hit_top_in, hit_width_in, hit_height_in)
        panel_left_in = inner_left_in + hit_width_in + GUTTER_IN
        add_rectangle(slide, panel_left_in, row_top_in, panel_width_in, bottom_row_height_in, TINT_HEX)
        add_text(slide, panel_left_in + CARD_PADDING_IN / 2, row_top_in, panel_width_in - CARD_PADDING_IN,
                 bottom_row_height_in, takeaway, anchor="middle")
    else:
        add_rectangle(slide, inner_left_in, row_top_in, inner_width_in, bottom_row_height_in, TINT_HEX)
        add_text(slide, inner_left_in + CARD_PADDING_IN / 2, row_top_in, inner_width_in - CARD_PADDING_IN,
                 bottom_row_height_in, takeaway, anchor="middle")

    # Step 7: speaker notes with the provenance (not printed).
    slide.notes_slide.notes_text_frame.text = (
        "Numbers: reports/gaussian_occupancy/<session>/gaussian_occupancy_summary.json (scripts/20), "
        f"{facts.run_count} runs per arm at {'≈' if facts.steps_vary else ''}{facts.training_steps} steps, threshold {facts.threshold_text}, pooled "
        "obstacles (tree trunks, rocks, bushes). Curves: figures_manifest.json of the same campaign. "
        "Built by docs/proposal/poster/build_poster_v2.py.")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    output_path = BUILD_DIR / f"{OUTPUT_STEMS[variant]}.pptx"
    presentation.save(str(output_path))
    print(f"wrote {output_path.relative_to(PROJECT_ROOT)} (top cards {top_card_height_in:.2f} in, "
          f"curves {curve_height_in:.2f} in)")
    return output_path


def convert_to_pdf(pptx_path: Path) -> Path:
    """Convert a .pptx to PDF with LibreOffice into ``docs/proposal/``.

    Args:
        pptx_path: The built poster.

    Returns:
        The PDF path.

    Raises:
        FileNotFoundError: When ``soffice`` is not on PATH.
    """
    soffice = shutil.which("soffice")
    if soffice is None:
        raise FileNotFoundError("soffice (LibreOffice) not found on PATH; install LibreOffice to export the PDF")
    subprocess.run([soffice, "--headless", "--convert-to", "pdf", "--outdir", str(PDF_DIR), str(pptx_path)],
                   check=True, cwd=PROJECT_ROOT, stdout=subprocess.DEVNULL)
    pdf_path = PDF_DIR / f"{pptx_path.stem}.pdf"
    # soffice can exit 0 without converting (e.g. while a GUI instance holds the profile).
    if not pdf_path.exists() or pdf_path.stat().st_mtime < pptx_path.stat().st_mtime:
        raise RuntimeError(f"LibreOffice did not write {pdf_path.name}; close any running LibreOffice and retry")
    print(f"wrote {pdf_path.relative_to(PROJECT_ROOT)}")
    return pdf_path


def main(argument_list: Optional[List[str]] = None) -> None:
    """CLI entry point.

    Args:
        argument_list: Arguments (default: ``sys.argv``).
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variant", choices=VARIANTS + ("all",), default="all")
    parser.add_argument("--pdf", action="store_true", help="also export the PDF with LibreOffice")
    parser.add_argument("--rebuild-figures", action="store_true", help="re-render figures even when up to date")
    arguments = parser.parse_args(argument_list)
    variants = VARIANTS if arguments.variant == "all" else (arguments.variant,)
    for variant in variants:
        pptx_path = build(variant, arguments.rebuild_figures)
        if arguments.pdf:
            convert_to_pdf(pptx_path)


if __name__ == "__main__":
    main()
