"""Poster-scale hit-difference figure (which obstacle cells each arm alone, or both, reconstruct).

Re-renders the figure of ``scripts/26_occupancy_figures.py`` (``draw_difference_figure``) for a large-format
poster slot: script 26's own functions compute the difference sets and render the 3D panel; this script only
sets the window size from the slot, recolours the arms to match the poster's threshold curves (tab10 by arm
position), and composes a large legend beside the render (no title, no footer: the poster has its own captions).

Run in the isolated viz environment (PyVista lives there):
    uv run --project viz python docs/proposal/poster/poster_hit_difference_figure.py \
        --session june_tall_grass --width-in 22.2 --height-in 5.8 \
        --output docs/proposal/poster/figures/june_tall_grass_hit_difference_poster.png

The three cell counts are checked against the committed figures manifest; a sidecar JSON records the inputs.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_26_PATH = REPOSITORY_ROOT / "scripts" / "26_occupancy_figures.py"
MANIFEST_PATTERN = "reports/gaussian_occupancy/figures/{session}/figures_manifest.json"
# 8-bit colour channel maximum: converts matplotlib's 0-1 floats to the 0-255 ints PyVista settings use.
RGB_CHANNEL_MAX = 255
# Render resolution (pixels per inch of slot width): >= 200 keeps cell edges crisp at poster print size.
RENDER_DPI = 200
# Default output resolution of the composed figure (print standard for large-format posters).
DEFAULT_OUTPUT_DPI = 300
# Legend font size in points: readable from poster-viewing distance (user's specification).
LEGEND_FONT_POINTS = 30
# Legend swatch edge as a multiple of the font size (square swatch about as tall as a capital letter).
SWATCH_SIZE_FONT_RATIO = 0.9
# Gap between swatch and text, and between render and legend, as multiples of the font size.
SWATCH_TEXT_GAP_FONT_RATIO = 0.5
RENDER_LEGEND_GAP_FONT_RATIO = 1.0
# Vertical spacing between legend entries as a multiple of the font size.
ENTRY_SPACING_FONT_RATIO = 2.0
# Line pitch inside a multi-line legend entry, as a multiple of the font size (each extra line adds one pitch
# to the entry spacing).
LEGEND_LINE_PITCH_FONT_RATIO = 1.2
# Separator between a set's name and its count in a one-line label (stacked labels break the line here).
LABEL_COUNT_SEPARATOR = ": "
POINTS_PER_INCH = 72.0  # typographic definition of a point


def load_script_26() -> Any:
    """Load ``scripts/26_occupancy_figures.py`` as a module (its name starts with a digit).

    Returns:
        The imported module.
    """
    # Step 1: import by path, as tests/unit/test_figures_pyvista.py does.
    module_spec = importlib.util.spec_from_file_location("script26_for_poster_difference", SCRIPT_26_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def rgba_to_rgb_ints(rgba: Sequence[float]) -> Tuple[int, int, int]:
    """Convert a matplotlib RGBA float colour to 0-255 RGB ints.

    Args:
        rgba: (red, green, blue[, alpha]) floats in [0, 1].

    Returns:
        (red, green, blue) ints in [0, 255].
    """
    # Step 1: scale each of the first three channels and round to the nearest int.
    return tuple(int(round(channel * RGB_CHANNEL_MAX)) for channel in rgba[:3])


def poster_colours(configured_colours: Mapping[str, Sequence[int]], first_colour_name: str,
                   second_colour_name: str, first_arm_rgba: Sequence[float],
                   second_arm_rgba: Sequence[float]) -> Dict[str, Tuple[int, ...]]:
    """Copy of the configured colours with the two only-arm entries set to the arms' curve colours.

    Args:
        configured_colours: ``FigureSettings.colours`` (name -> 0-255 RGB); left unmodified.
        first_colour_name: Name of the "only first arm" colour.
        second_colour_name: Name of the "only second arm" colour.
        first_arm_rgba: Curve colour (matplotlib RGBA) of the first arm.
        second_arm_rgba: Curve colour (matplotlib RGBA) of the second arm.

    Returns:
        New colour mapping (``both`` and every other entry as configured).
    """
    # Step 1: copy so the shared settings stay untouched.
    colours = {name: tuple(value) for name, value in configured_colours.items()}
    # Step 2: override the two only-arm colours with the poster's tab10 arm colours.
    colours[first_colour_name] = rgba_to_rgb_ints(first_arm_rgba)
    colours[second_colour_name] = rgba_to_rgb_ints(second_arm_rgba)
    return colours


def render_window_size(image_width_in: float, render_dpi: int, camera_aspect: float) -> Tuple[int, int]:
    """Pixel window of the 3D render for a slot width and the scene's projected aspect.

    width_px = round(image_width_in * render_dpi); height_px = round(width_px / aspect), aspect = width / height.

    Args:
        image_width_in: Width (inches) the render may occupy.
        render_dpi: Render pixels per inch.
        camera_aspect: Width / height of the framed points (``scene_camera_aspect``).

    Returns:
        (width, height) in pixels, each at least 1.
    """
    # Step 1: width from the slot, height from the aspect.
    width_px = max(1, round(image_width_in * render_dpi))
    height_px = max(1, round(width_px / camera_aspect))
    return width_px, height_px


def legend_labels(arm_labels: Mapping[str, str], first_arm: str, second_arm: str,
                  cell_counts: Mapping[str, int]) -> Sequence[str]:
    """Legend texts with thousands separators: only first, only second, both.

    Args:
        arm_labels: Arm key -> display label.
        first_arm: Key of the first compared arm.
        second_arm: Key of the second compared arm.
        cell_counts: ``only_first`` / ``only_second`` / ``both`` -> cell count.

    Returns:
        Three label strings in legend order.
    """
    # Step 1: "<set>: <count> cells", count formatted with a comma separator.
    return [f"only {arm_labels[first_arm]}: {cell_counts['only_first']:,} cells",
            f"only {arm_labels[second_arm]}: {cell_counts['only_second']:,} cells",
            f"both: {cell_counts['both']:,} cells"]


def stack_label(label: str) -> str:
    """Put a label's count on its own line: "only A: 1,356 cells" -> "only A\n1,356 cells".

    A narrower legend leaves more of the slot width to the render.

    Args:
        label: One-line label from ``legend_labels``.

    Returns:
        The label broken at its last ``LABEL_COUNT_SEPARATOR``; unchanged if it has none.
    """
    name, separator, count = label.rpartition(LABEL_COUNT_SEPARATOR)
    return f"{name}\n{count}" if separator else label


def check_counts_against_manifest(cell_counts: Mapping[str, int], manifest_difference: Mapping[str, int]) -> None:
    """Refuse counts that differ from the committed manifest's ``hit_difference_scene.difference``.

    Args:
        cell_counts: Counts computed now (``only_first``, ``only_second``, ``both``).
        manifest_difference: The manifest's counts.

    Raises:
        ValueError: When any of the three counts differs.
    """
    # Step 1: compare each of the three keys.
    mismatches = {key: (cell_counts[key], manifest_difference[key]) for key in ("only_first", "only_second", "both")
                  if cell_counts[key] != manifest_difference[key]}
    if mismatches:
        raise ValueError("hit-difference counts differ from the figures manifest (key: (computed, manifest)): "
                         f"{mismatches}")


def sha256_of_file(path: Path) -> str:
    """SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def measure_legend_width_in(labels: Sequence[str], dpi: int) -> float:
    """Width (inches) of the legend block: swatch + gap + the widest label at the legend font size.

    Args:
        labels: Legend texts.
        dpi: Figure dpi used for the measurement.

    Returns:
        Block width in inches.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    # Step 1: render each label off-screen and take the widest pixel extent.
    figure = Figure(dpi=dpi)
    renderer = FigureCanvasAgg(figure).get_renderer()
    widest_px = max(figure.text(0, 0, label, fontsize=LEGEND_FONT_POINTS).get_window_extent(renderer).width
                    for label in labels)
    # Step 2: add the swatch and the swatch-text gap (font-size multiples, converted points -> inches).
    font_in = LEGEND_FONT_POINTS / POINTS_PER_INCH
    return widest_px / dpi + font_in * (SWATCH_SIZE_FONT_RATIO + SWATCH_TEXT_GAP_FONT_RATIO)


def compose_poster_figure(image: Any, labels: Sequence[str], swatch_rgb: Sequence[Sequence[int]],
                          width_in: float, height_in: float, legend_width_in: float) -> Any:
    """Slot-sized figure: the render on the left (aspect preserved) and the legend on its right.

    With no labels (``--no-legend``) the render is centred in the slot and no legend is drawn.

    Args:
        image: (H, W, 3) uint8 render.
        labels: Legend texts.
        swatch_rgb: 0-255 RGB per entry.
        width_in: Slot width (inches).
        height_in: Slot height (inches).
        legend_width_in: Legend block width (inches).

    Returns:
        The matplotlib figure.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle
    font_in = LEGEND_FONT_POINTS / POINTS_PER_INCH
    gap_in = font_in * RENDER_LEGEND_GAP_FONT_RATIO if labels else 0.0
    figure = Figure(figsize=(width_in, height_in))
    FigureCanvasAgg(figure)  # attach the Agg canvas so savefig works without pyplot
    # Step 1: render axes = the largest box of the image's aspect inside (slot minus legend minus gap) x height.
    image_aspect = image.shape[1] / image.shape[0]
    available_width_in = width_in - legend_width_in - gap_in
    image_width_in = min(available_width_in, height_in * image_aspect)
    image_height_in = image_width_in / image_aspect
    # Without a legend the render is centred horizontally; with one it sits flush left.
    image_left_in = 0.0 if labels else (width_in - image_width_in) / 2
    image_axes = figure.add_axes([image_left_in / width_in, (height_in - image_height_in) / 2 / height_in,
                                  image_width_in / width_in, image_height_in / height_in])
    image_axes.imshow(image, aspect="equal", interpolation="lanczos")
    image_axes.axis("off")
    if not labels:
        return figure
    # Step 2: legend axes in inches (data units = inches), right of the render, entries centred vertically.
    legend_left_in = image_width_in + gap_in
    legend_axes = figure.add_axes([legend_left_in / width_in, 0, (width_in - legend_left_in) / width_in, 1])
    legend_axes.set_xlim(0, width_in - legend_left_in)
    legend_axes.set_ylim(0, height_in)
    legend_axes.axis("off")
    # Entry spacing: the one-line spacing plus one line pitch per extra line of the tallest label.
    extra_lines = max(label.count("\n") for label in labels)
    spacing_in = font_in * (ENTRY_SPACING_FONT_RATIO + LEGEND_LINE_PITCH_FONT_RATIO * extra_lines)
    swatch_in = font_in * SWATCH_SIZE_FONT_RATIO
    block_height_in = spacing_in * (len(labels) - 1)
    for entry_index, (label, rgb) in enumerate(zip(labels, swatch_rgb)):
        # Entry centre line, first entry on top.
        centre_y_in = height_in / 2 + block_height_in / 2 - entry_index * spacing_in
        legend_axes.add_patch(Rectangle((0, centre_y_in - swatch_in / 2), swatch_in, swatch_in,
                                        facecolor=[channel / RGB_CHANNEL_MAX for channel in rgb], edgecolor="none"))
        legend_axes.text(swatch_in + font_in * SWATCH_TEXT_GAP_FONT_RATIO, centre_y_in, label,
                         fontsize=LEGEND_FONT_POINTS, va="center", ha="left")
    return figure


def build_parser() -> argparse.ArgumentParser:
    """Command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="session name, e.g. june_tall_grass")
    parser.add_argument("--width-in", type=float, required=True, help="poster slot width (inches)")
    parser.add_argument("--height-in", type=float, required=True, help="poster slot height (inches)")
    parser.add_argument("--output", type=Path, required=True, help="output PNG path")
    parser.add_argument("--dpi", type=int, default=DEFAULT_OUTPUT_DPI, help="output dpi (default 300)")
    parser.add_argument("--no-legend", action="store_true",
                        help="draw the render only, centred (the poster then states colours and counts in text)")
    parser.add_argument("--stacked-legend", action="store_true",
                        help="put each legend entry's cell count on a second line (narrower legend, larger render)")
    return parser


def main(argument_list: Any = None) -> int:
    """Render the poster hit-difference figure, check the counts and write the PNG and its sidecar JSON.

    Args:
        argument_list: Optional argv (tests); default ``sys.argv``.

    Returns:
        Process exit code (0 on success).
    """
    arguments = build_parser().parse_args(argument_list)
    # Step 1: script 26's context for the default (campaign) profile set, and its difference sets.
    script = load_script_26()
    context = script.build_context(script.build_argument_parser().parse_args(["--session", arguments.session]))
    script.check_difference_arms(context)
    _, difference_sets = script.scene_outcomes_and_difference(context)
    cell_counts = {name: int(mask.sum()) for name, mask in difference_sets.items()}
    # Step 2: counts must equal the committed manifest.
    manifest_path = REPOSITORY_ROOT / MANIFEST_PATTERN.format(session=arguments.session)
    manifest = json.loads(manifest_path.read_text())
    check_counts_against_manifest(
        cell_counts, manifest["reported_numbers"]["hit_difference_scene"]["difference"])
    # Step 3: poster colours (tab10 by arm position, as the threshold curves).
    from viking_splatad.viz.figures_mpl import _arm_colour
    first_colour_name, second_colour_name = script.difference_colour_names(context.settings)
    colours = poster_colours(context.settings.colours, first_colour_name, second_colour_name,
                             _arm_colour(context.settings, script.DIFFERENCE_FIRST_ARM),
                             _arm_colour(context.settings, script.DIFFERENCE_SECOND_ARM))
    # Step 4: legend texts and width, then the render window from the remaining slot width.
    labels = legend_labels(context.settings.arm_labels, script.DIFFERENCE_FIRST_ARM,
                           script.DIFFERENCE_SECOND_ARM, cell_counts)
    if arguments.stacked_legend:
        labels = [stack_label(label) for label in labels]
    if arguments.no_legend:
        labels = []
    legend_width_in = measure_legend_width_in(labels, arguments.dpi) if labels else 0.0
    from viking_splatad.viz.figures_pyvista import (difference_frame_points, render_difference_panel,
                                                    scene_camera_aspect)
    camera_points = difference_frame_points(context.reference_cells["centres"], difference_sets, context.voxel_size)
    # As in scripts/26 draw_difference_figure: no cell in any difference set leaves nothing to frame.
    if camera_points.shape[0] == 0:
        raise ValueError("no cell in any hit-difference set; nothing to render")
    gap_in = LEGEND_FONT_POINTS / POINTS_PER_INCH * RENDER_LEGEND_GAP_FONT_RATIO if labels else 0.0
    render_width_in = arguments.width_in - legend_width_in - gap_in
    if render_width_in <= 0:
        raise ValueError(f"slot {arguments.width_in} in is narrower than the legend ({legend_width_in:.2f} in + "
                         f"{gap_in:.2f} in gap); widen the slot or pass --no-legend")
    window_size = render_window_size(render_width_in, RENDER_DPI,
                                     scene_camera_aspect(camera_points))
    # Step 5: render with script 26's own panel renderer.
    image, crop_report = render_difference_panel(script.import_pyvista(), context.reference_cells["centres"],
                                                 context.voxel_size, difference_sets, colours, window_size,
                                                 first_colour_name=first_colour_name,
                                                 second_colour_name=second_colour_name)
    # Step 6: compose at the exact slot size (no tight bbox) and save.
    swatch_rgb = [colours[first_colour_name], colours[second_colour_name], colours["both"]]
    figure = compose_poster_figure(image, labels, swatch_rgb, arguments.width_in, arguments.height_in,
                                   legend_width_in)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=arguments.dpi, facecolor="white")
    # Step 7: sidecar JSON.
    sidecar = {"session": arguments.session, "cell_counts": cell_counts,
               "colours_rgb": {name: list(colours[name]) for name in (first_colour_name, second_colour_name, "both")},
               "slot_in": [arguments.width_in, arguments.height_in], "dpi": arguments.dpi,
               "render_window_px": list(window_size), "panel_crop": crop_report, "legend": bool(labels),
               "legend_stacked": bool(labels) and arguments.stacked_legend,
               "manifest_path": str(manifest_path.relative_to(REPOSITORY_ROOT)),
               "manifest_sha256": sha256_of_file(manifest_path)}
    arguments.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2, default=str))
    print(f"wrote {arguments.output} counts={cell_counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
