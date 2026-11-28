"""Poster-scale threshold-curve figure (recall and precision of class obstacles vs occupancy threshold).

Redraws the committed ``threshold_curves_scene`` numbers of ``figures_manifest.json`` at print scale
(1 figure point = 1 printed point). Nothing is recomputed: the pooled curves and the min-max bands over the
repeats are read as stored. Arm order, labels and colours come from the repo's figure settings.

Run in the isolated viz environment:
    uv run --project viz python docs/proposal/poster/poster_threshold_figure.py \
        --session june_tall_grass --width-in 10.9 --height-in 8.5 --output <png>
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless backend: the figure is only written to a file
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402
from matplotlib.ticker import MaxNLocator, PercentFormatter  # noqa: E402

from viking_splatad.viz.figure_runs import FigureSettings, load_figure_settings  # noqa: E402
from viking_splatad.viz.figures_mpl import _arm_colour  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_CONFIG_PATH = PROJECT_ROOT / "configs" / "eval.yaml"
MANIFEST_PATH_TEMPLATE = "reports/gaussian_occupancy/figures/{session}/figures_manifest.json"

# Manifest keys (stored layout of reported_numbers.threshold_curves_scene).
CURVES_SECTION_KEY = "threshold_curves_scene"
PANEL_CURVE_KEYS = (("pooled_recall", "Recall"), ("pooled_precision", "Precision"))

# Print-scale sizes in points (poster viewed from a distance; the screen figure used 7-8 pt).
TICK_FONT_PT = 24
AXIS_LABEL_FONT_PT = 28
PANEL_TITLE_FONT_PT = 32
LEGEND_FONT_PT = 26
GUIDE_LABEL_FONT_PT = 20
CURVE_LINE_WIDTH_PT = 4.0
MARKER_SIZE_PT = 8.0
GUIDE_LINE_WIDTH_PT = 2.0
# Band opacity: light enough to see overlapping arms, dark enough to survive printing.
BAND_ALPHA = 0.25
GUIDE_COLOUR = "0.65"
GRID_COLOUR = "0.88"
GRID_LINE_WIDTH_PT = 1.5
# Axis margin as a fraction of the data range, and the step the y limits are rounded outward to
# (1 = 100 %; 0.1 = 10 percentage points; the drawn ticks follow Y_MAJOR_TICK_MANTISSAS).
Y_MARGIN_FRACTION = 0.08
Y_TICK_STEP = 0.1
# Allowed major y tick steps, as MaxNLocator mantissas of the 0.1 decade: 5 or 10 percentage points (whole-percent
# labels stay exact). nbins="auto" picks the finest step whose labels do not crowd for the axis height and tick
# font, so a short poster slot gets 10-point ticks and a tall one 5-point ticks.
Y_MAJOR_TICK_MANTISSAS = (5, 10)
# X padding (threshold units) so the first/last markers and the corner tick labels do not collide.
X_PADDING = 0.03
# Decimals kept when rounding limits, removing float noise such as 0.7000000000000001.
LIMIT_DECIMALS = 10
# A percentage cannot leave [0, 1].
Y_LOWER_BOUND = 0.0
Y_UPPER_BOUND = 1.0
# Number of guides (lowest thresholds first) that carry a text label; more would clutter.
LABELLED_GUIDE_COUNT = 1
DEFAULT_DPI = 300
# Rounding guard so a value exactly on a tick does not jump a whole step because of float noise.
FLOAT_GUARD = 1e-9


def data_axis_limits(values: Sequence[float], margin_fraction: float, tick_step: float) -> Tuple[float, float]:
    """Return y limits that enclose the data with a margin, rounded outward to a tick step.

    Rule: with data range r = max - min, lower = floor((min - margin_fraction * r) / tick_step) * tick_step
    and upper = ceil((max + margin_fraction * r) / tick_step) * tick_step. NaNs are ignored.

    Args:
        values: All values the axis must show (curves and bands).
        margin_fraction: Margin as a fraction of the data range.
        tick_step: Step the limits are rounded outward to.

    Returns:
        (lower, upper) limits.

    Raises:
        ValueError: If no finite value is given or tick_step is not positive.
    """
    # Step 1: keep the finite values only.
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        raise ValueError("no finite value to derive axis limits from")
    if tick_step <= 0:
        raise ValueError("tick_step must be positive")
    # Step 2: pad by a fraction of the range.
    data_min, data_max = float(finite_values.min()), float(finite_values.max())
    margin = margin_fraction * (data_max - data_min)
    # Step 3: round outward to multiples of the tick step (guard against float noise at exact ticks).
    lower = math.floor((data_min - margin) / tick_step + FLOAT_GUARD) * tick_step
    upper = math.ceil((data_max + margin) / tick_step - FLOAT_GUARD) * tick_step
    # Step 4: constant data sitting on a tick would give lower == upper; widen to one tick step above.
    if upper - lower < tick_step * FLOAT_GUARD:
        upper = lower + tick_step
    return round(lower, LIMIT_DECIMALS), round(upper, LIMIT_DECIMALS)


def to_float_array(values: Sequence[Any]) -> np.ndarray:
    """Convert a JSON list to floats, mapping null to NaN.

    Args:
        values: JSON list of numbers or None.

    Returns:
        Float array.
    """
    # None -> NaN so matplotlib leaves a gap instead of failing.
    return np.array([np.nan if value is None else float(value) for value in values], dtype=float)


def parse_threshold_curves(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract the pooled curves, min-max bands and guides from a figures manifest.

    Args:
        manifest: Parsed figures_manifest.json.

    Returns:
        Dict with ``grid`` (array), ``curves`` (arm -> curve key -> array), ``bands`` (arm -> curve key ->
        (lower, upper) arrays; an arm without campaign repeats has no band entry), ``repeat_counts`` and
        ``p2_thresholds`` (list of floats).
    """
    # Step 1: locate the section.
    section = manifest["reported_numbers"][CURVES_SECTION_KEY]
    curve_keys = [curve_key for curve_key, _ in PANEL_CURVE_KEYS]
    curves: Dict[str, Dict[str, np.ndarray]] = {}
    bands: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    # Step 2: read every drawn arm's curves and bands.
    for arm, arm_curves in section["drawn_curves"].items():
        curves[arm] = {curve_key: to_float_array(arm_curves[curve_key]) for curve_key in curve_keys}
        # scripts/26 writes a band only for arms with campaign repeats; an arm drawn from an override has none.
        stored_bands = section.get("campaign_bands", {}).get(arm)
        if stored_bands is None:
            continue
        bands[arm] = {curve_key: (to_float_array(stored_bands[curve_key]["lower"]),
                                  to_float_array(stored_bands[curve_key]["upper"])) for curve_key in curve_keys}
    return {
        "grid": to_float_array(section["curve_grid"]),
        "curves": curves,
        "bands": bands,
        "repeat_counts": dict(section.get("repeat_counts", {})),
        "p2_thresholds": [float(threshold) for threshold in manifest["p2_thresholds"]],
    }


def panel_values(parsed: Mapping[str, Any], curve_key: str) -> List[float]:
    """Collect every value a panel must show (curves and both band edges, all arms).

    Args:
        parsed: Output of parse_threshold_curves.
        curve_key: Panel curve key.

    Returns:
        Flat list of values.
    """
    collected: List[float] = []
    for arm, arm_curves in parsed["curves"].items():
        collected.extend(arm_curves[curve_key].tolist())
        if arm in parsed["bands"]:
            lower, upper = parsed["bands"][arm][curve_key]
            collected.extend(lower.tolist() + upper.tolist())
    return collected


def draw_poster_figure(
    parsed: Mapping[str, Any],
    settings: FigureSettings,
    width_in: float,
    height_in: float,
    dpi: int,
    output_path: Path,
) -> Dict[str, List[float]]:
    """Draw the two-panel figure and save it at exactly width_in x height_in inches.

    Args:
        parsed: Output of parse_threshold_curves.
        settings: Repo figure settings (arm order, labels, colours).
        width_in: Slot width in inches.
        height_in: Slot height in inches.
        dpi: Output resolution.
        output_path: PNG path.

    Returns:
        y limits used, keyed by curve key.
    """
    # Step 1: figure of the exact slot size; constrained layout, no tight bbox so the size stays exact.
    figure, panel_axes = plt.subplots(1, len(PANEL_CURVE_KEYS), figsize=(width_in, height_in), layout="constrained")
    grid = parsed["grid"]
    y_limits_used: Dict[str, List[float]] = {}
    arm_handles = {}
    for panel_axis, (curve_key, panel_title) in zip(np.atleast_1d(panel_axes), PANEL_CURVE_KEYS):
        # Step 2: independent data-driven y range per panel, clamped to the valid [0, 1].
        lower_limit, upper_limit = data_axis_limits(panel_values(parsed, curve_key), Y_MARGIN_FRACTION, Y_TICK_STEP)
        lower_limit, upper_limit = max(lower_limit, Y_LOWER_BOUND), min(upper_limit, Y_UPPER_BOUND)
        y_limits_used[curve_key] = [lower_limit, upper_limit]
        # Step 3: min-max band then pooled curve per arm, in the settings' arm order.
        for arm in settings.arm_order:
            if arm not in parsed["curves"]:
                continue
            colour = _arm_colour(settings, arm)
            if arm in parsed["bands"]:
                band_lower, band_upper = parsed["bands"][arm][curve_key]
                panel_axis.fill_between(grid, band_lower, band_upper, color=colour, alpha=BAND_ALPHA, linewidth=0)
            (line,) = panel_axis.plot(
                grid,
                parsed["curves"][arm][curve_key],
                color=colour,
                linewidth=CURVE_LINE_WIDTH_PT,
                marker="o",
                markersize=MARKER_SIZE_PT,
                label=settings.arm_labels[arm],
            )
            arm_handles[arm] = line
        # Step 4: dashed guides at the manifest's P2 thresholds; only the first LABELLED_GUIDE_COUNT are labelled.
        for guide_index, threshold in enumerate(parsed["p2_thresholds"]):
            panel_axis.axvline(threshold, color=GUIDE_COLOUR, linestyle="--", linewidth=GUIDE_LINE_WIDTH_PT, zorder=0)
            if guide_index < LABELLED_GUIDE_COUNT:
                panel_axis.text(
                    threshold, upper_limit, f" threshold {threshold:g}", color=GUIDE_COLOUR,
                    fontsize=GUIDE_LABEL_FONT_PT, va="top", ha="left",
                )
        # Step 5: axes cosmetics.
        panel_axis.set_ylim(lower_limit, upper_limit)
        panel_axis.set_xlim(float(np.nanmin(grid)) - X_PADDING, float(np.nanmax(grid)) + X_PADDING)
        panel_axis.yaxis.set_major_locator(MaxNLocator(nbins="auto", steps=list(Y_MAJOR_TICK_MANTISSAS)))
        panel_axis.set_title(panel_title, fontsize=PANEL_TITLE_FONT_PT, fontweight="bold")
        panel_axis.set_xlabel("Occupancy threshold", fontsize=AXIS_LABEL_FONT_PT)
        panel_axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        panel_axis.tick_params(labelsize=TICK_FONT_PT)
        panel_axis.grid(axis="y", color=GRID_COLOUR, linewidth=GRID_LINE_WIDTH_PT)
        panel_axis.set_axisbelow(True)
        for spine_name in ("top", "right"):
            panel_axis.spines[spine_name].set_visible(False)
    # Step 6: one shared legend above the panels, in arm order.
    ordered_arms = [arm for arm in settings.arm_order if arm in arm_handles]
    figure.legend(
        [arm_handles[arm] for arm in ordered_arms],
        [settings.arm_labels[arm] for arm in ordered_arms],
        loc="outside upper center",
        ncol=len(ordered_arms),
        fontsize=LEGEND_FONT_PT,
        frameon=False,
    )
    # Step 7: save at the exact slot size and release the figure.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi)
    plt.close(figure)
    return y_limits_used


def file_sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", required=True)
    parser.add_argument("--width-in", type=float, required=True)
    parser.add_argument("--height-in", type=float, required=True)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None, help="default: the session's figures_manifest.json")
    parser.add_argument("--eval-config", type=Path, default=EVAL_CONFIG_PATH)
    return parser


def main() -> None:
    """Draw the figure and write its PNG and audit sidecar JSON."""
    arguments = build_parser().parse_args()
    # Step 1: inputs.
    manifest_path = arguments.manifest or PROJECT_ROOT / MANIFEST_PATH_TEMPLATE.format(session=arguments.session)
    manifest = json.loads(manifest_path.read_text())
    settings = load_figure_settings(yaml.safe_load(arguments.eval_config.read_text())["eval"])
    # The sidecar names the manifest repo-relative (no machine path); outside the repo, by file name only.
    # Computed before drawing so a bad path fails before any PNG is written.
    resolved_manifest = manifest_path.resolve()
    manifest_label = (str(resolved_manifest.relative_to(PROJECT_ROOT)) if resolved_manifest.is_relative_to(PROJECT_ROOT)
                      else resolved_manifest.name)
    # Step 2: draw.
    y_limits = draw_poster_figure(
        parse_threshold_curves(manifest), settings, arguments.width_in, arguments.height_in,
        arguments.dpi, arguments.output,
    )
    # Step 3: audit sidecar next to the PNG.
    sidecar = {
        "manifest_path": manifest_label,
        "manifest_sha256": file_sha256(manifest_path),
        "session": arguments.session,
        "width_in": arguments.width_in,
        "height_in": arguments.height_in,
        "dpi": arguments.dpi,
        "y_limits": y_limits,
    }
    arguments.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(f"wrote {arguments.output} y_limits={y_limits}")


if __name__ == "__main__":
    main()
