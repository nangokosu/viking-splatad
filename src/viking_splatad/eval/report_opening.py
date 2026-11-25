"""Shared opening of the evaluation reports: definitions, whole-scene tables and the figures of ``FIGURE_NAMES``.

Scripts/20 (one session) and scripts/21 (all sessions) both open with these texts, and scripts/21 also
writes them as a stand-alone terminology document. Everything is rendered from a scripts/20 summary and
the figure manifest of scripts/26, so the reports cannot disagree with each other or with the figures.
"""

import json
import posixpath
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from viking_splatad.eval.evaluation_terminology import (
    ARM_LABEL_BY_FIGURE_KEY,
    ARM_LABEL_BY_METHOD,
    FIGURE_NAMES,
    GAUSSIANFORMER_REFERENCE,
    KERBL_REFERENCE,
    SPLATAD_REFERENCE,
    definition_values,
    definitions_markdown,
    figure_reading_markdown,
    format_reference,
)
from viking_splatad.eval.terminology_glossary import CODE_REFERENCES, glossary_markdown

# Decimal places in the tables (metrics are fractions in [0, 1]; 3 places resolve 0.1 %).
TABLE_DECIMALS = 3
# Whole-scene (all scored cells) P2 metrics of the tables, with the table's row label.
WHOLE_SCENE_METRICS = (("recall", "recall"), ("precision", "precision"), ("iou", "IoU"))
# The pooled variant (cells pooled over objects) is the whole-scene headline.
WHOLE_SCENE_VARIANT = "pooled"
# Figure title (alt text and caption) by figure name; keys equal FIGURE_NAMES.
FIGURE_TITLES = {"arms_scene_3d": "Whole scene in 3D, per arm",
                 "threshold_curves_scene": "Recall and precision against the occupancy threshold"}
# Figure manifest of scripts/26: location under the project root, per session, and the command that writes it.
FIGURES_RELATIVE_DIR = Path("reports") / "gaussian_occupancy" / "figures"
FIGURES_MANIFEST_NAME = "figures_manifest.json"
FIGURE_SUITE_COMMAND = "uv run --project viz python scripts/26_occupancy_figures.py --session {session}"
# Manifest entry holding the whole-scene counts, computed outside any figure.
SCENE_COUNTS_KEY = "scene_counts"
# Title of the stand-alone terminology document.
TERMINOLOGY_TITLE = "# Terminology of the evaluation reports"
# Note shown when a figure is not embedded; the manifest's error (or its absence) follows.
MISSING_FIGURE_NOTE = "Figure '{title}' is not available (manifest status {status!r}): {reason}"
NO_REASON_TEXT = "no error recorded in the manifest"


def format_value(value: Optional[float]) -> str:
    """A metric value for the tables.

    Args:
        value: Number or None (undefined).

    Returns:
        Fixed-point text, or "n/a".
    """
    return "n/a" if value is None else f"{value:.{TABLE_DECIMALS}f}"


def separation_label(separated: Optional[bool]) -> str:
    """Table text for compare_arms' "separated" flag.

    Args:
        separated: True (ranges disjoint), False (overlap) or None (an arm has < 2 defined repeats).

    Returns:
        "sep", "ovl" or "n/a".
    """
    if separated is None:
        return "n/a"
    return "sep" if separated else "ovl"


def format_description(description: Mapping[str, Optional[float]]) -> str:
    """An arm's metric as "mean ± std [min, max] (n=repeats)".

    Args:
        description: ``describe_values`` output (mean, std, min, max, n).

    Returns:
        Table cell text.
    """
    return (f"{format_value(description['mean'])} ± {format_value(description['std'])} "
            f"[{format_value(description['min'])}, {format_value(description['max'])}] (n={description['n']})")


def load_figure_manifest(project_root: Path, session: str) -> Dict[str, Any]:
    """Read a session's scripts/26 manifest, refusing clearly when it is missing.

    Args:
        project_root: Repository root.
        session: Session name.

    Returns:
        The parsed ``figures_manifest.json``.

    Raises:
        ValueError: The manifest file does not exist (the message names the scripts/26 command).
    """
    manifest_path = project_root / FIGURES_RELATIVE_DIR / session / FIGURES_MANIFEST_NAME
    if not manifest_path.is_file():
        raise ValueError(f"no figure manifest at {manifest_path}; run {FIGURE_SUITE_COMMAND.format(session=session)} first")
    return json.loads(manifest_path.read_text())


def check_manifest_matches_summary(summary: Mapping, manifest: Mapping) -> None:
    """Refuse a manifest that was drawn from other runs, another session or another cut-off than the summary.

    Args:
        summary: scripts/20 summary (``session``, ``p2_thresholds``, ``arms`` with ``runs``).
        manifest: scripts/26 manifest (``session``, ``binding_threshold``, ``arms`` with ``drawn_run_id``).

    Raises:
        ValueError: Any mismatch; the message says to re-run scripts/26.
    """
    rerun_hint = f"re-run {FIGURE_SUITE_COMMAND.format(session=summary['session'])}"
    # Check 1: same session.
    if manifest.get("session") != summary["session"]:
        raise ValueError(f"manifest is for session {manifest.get('session')!r}, summary for {summary['session']!r}; {rerun_hint}")
    # Check 2: same binding cut-off (the first threshold of the summary).
    if float(manifest["binding_threshold"]) != float(summary["p2_thresholds"][0]):
        raise ValueError(f"manifest binding threshold {manifest['binding_threshold']} differs from the summary's "
                         f"{summary['p2_thresholds'][0]}; {rerun_hint}")
    # Check 3: every drawn run is one of the summary's runs of the matching arm. The manifest arm's
    # ``profile`` equals the summary's arm key; without it, match by the arm's label via train_method.
    summary_arm_keys_by_label = {ARM_LABEL_BY_METHOD.get(arm["train_method"], arm["train_method"]): arm_key
                                 for arm_key, arm in summary["arms"].items()}
    for figure_arm_key, manifest_arm in manifest["arms"].items():
        summary_arm_key = manifest_arm.get("profile")
        if summary_arm_key not in summary["arms"]:
            summary_arm_key = summary_arm_keys_by_label.get(ARM_LABEL_BY_FIGURE_KEY.get(figure_arm_key, figure_arm_key))
        if summary_arm_key is None:
            raise ValueError(f"manifest arm {figure_arm_key!r} matches no summary arm; {rerun_hint}")
        summary_run_ids = {run["run_id"] for run in summary["arms"][summary_arm_key]["runs"]}
        if manifest_arm["drawn_run_id"] not in summary_run_ids:
            raise ValueError(f"manifest arm {figure_arm_key!r} drew run {manifest_arm['drawn_run_id']}, which is not among "
                             f"the summary's runs of {summary_arm_key!r}; {rerun_hint}")


def manifest_definition_values(summary: Mapping, manifest: Mapping) -> Dict[str, Any]:
    """Definition placeholders from a summary and the figure manifest.

    Args:
        summary: scripts/20 summary with ``provenance``.
        manifest: scripts/26 ``figures_manifest.json`` content.

    Returns:
        ``definition_values`` output. A missing curve grid (the curve figure failed) gives "not available"
        curve placeholders instead of an error.

    Raises:
        KeyError: The manifest lacks ``reported_numbers["scene_counts"]`` or its fields.
        ValueError: Manifest and summary disagree (see ``check_manifest_matches_summary``), plus
            ``definition_values``' own errors.
    """
    check_manifest_matches_summary(summary, manifest)
    reported = manifest["reported_numbers"]
    scene_counts = reported[SCENE_COUNTS_KEY]
    # The curve grid sits with the curve figure; absent or empty means the figure failed.
    curve_grid = reported.get("threshold_curves_scene", {}).get("curve_grid")
    curve_settings = None
    if curve_grid:
        # The grid is increasing, so its ends are the smallest and largest threshold.
        curve_settings = {"min": curve_grid[0], "max": curve_grid[-1], "points": len(curve_grid)}
    # Repeats behind the curve band, per figure arm; used only when every arm reports a count.
    curve_repeat_counts = {arm_key: arm["campaign_run_count"] for arm_key, arm in manifest["arms"].items()
                           if "campaign_run_count" in arm}
    if len(curve_repeat_counts) != len(manifest["arms"]):
        curve_repeat_counts = None
    return definition_values(summary, curve_settings, scene_counts["cell_counts"], scene_counts["outcome_counts_by_arm"],
                             curve_repeat_counts)


def whole_scene_row_key(class_name: str, threshold_text: str, metric_name: str) -> str:
    """Flat metric key of a whole-scene P2 row.

    Args:
        class_name: Class or group.
        threshold_text: P2 threshold key.
        metric_name: recall, precision or iou.

    Returns:
        ``<class>/P2@<t>/<metric>/pooled``.
    """
    return f"{class_name}/P2@{threshold_text}/{metric_name}/{WHOLE_SCENE_VARIANT}"


def arm_display_name(summary: Mapping, profile: str) -> str:
    """Table label of an arm: its plain label (as in the definitions) with its Slurm profile.

    Args:
        summary: scripts/20 summary (``arms[profile]["train_method"]``).
        profile: Arm key of the summary (the Slurm profile).

    Returns:
        E.g. "stock (`june_scene_a_cam`)"; an unknown method falls back to the method name.
    """
    method = summary["arms"][profile]["train_method"]
    return f"{ARM_LABEL_BY_METHOD.get(method, method)} (`{profile}`)"


def whole_scene_tables_markdown(summary: Mapping) -> str:
    """Per class and threshold: the arms' whole-scene rows, then the pair gaps on the same rows.

    Args:
        summary: scripts/20 summary (``arms``, ``comparisons``, ``class_names``, ``p2_thresholds``).

    Returns:
        Markdown lines joined by newlines (headings at level 3 and 4).
    """
    arm_names = list(summary["arms"])
    lines: List[str] = []
    # Part 1: arm tables, class order as in the summary, thresholds in the summary's order.
    for class_name in summary["class_names"]:
        for threshold_text in summary["p2_thresholds"]:
            lines += [f"### {class_name}, P2 occupied at >= {threshold_text}", "",
                      "| metric | " + " | ".join(arm_display_name(summary, arm_name) for arm_name in arm_names) + " |", "|---|" + "---|" * len(arm_names)]
            for metric_name, row_label in WHOLE_SCENE_METRICS:
                metric_key = whole_scene_row_key(class_name, threshold_text, metric_name)
                lines.append(f"| {row_label} | " + " | ".join(
                    format_description(summary["arms"][arm_name]["metrics"][metric_key]) for arm_name in arm_names) + " |")
            lines.append("")
    # Part 2: pair gaps (first - second) with the separation flag, as in scripts/20's comparison tables.
    for comparison in summary["comparisons"]:
        lines += [f"### Gap {arm_display_name(summary, comparison['first'])} - {arm_display_name(summary, comparison['second'])} (sep = the arms' [min, max] do not overlap, ovl = they overlap)", "",
                  "| metric | gap | sep / ovl |", "|---|---|---|"]
        for class_name in summary["class_names"]:
            for threshold_text in summary["p2_thresholds"]:
                for metric_name, _ in WHOLE_SCENE_METRICS:
                    metric_key = whole_scene_row_key(class_name, threshold_text, metric_name)
                    row = comparison["metrics"][metric_key]
                    lines.append(f"| {metric_key} | {format_value(row['gap'])} | {separation_label(row['separated'])} |")
        lines.append("")
    return "\n".join(lines)


def figure_block_markdown(figure_name: str, manifest: Mapping, figures_relative_dir: str, values: Mapping) -> str:
    """One figure embedded with its reading guide, or a note saying why it is missing.

    Args:
        figure_name: One of ``FIGURE_NAMES``.
        manifest: scripts/26 manifest.
        figures_relative_dir: Folder of the figures as seen from the report's own folder.
        values: ``definition_values`` output.

    Returns:
        Markdown block.
    """
    title = FIGURE_TITLES[figure_name]
    entry = manifest.get("figures", {}).get(figure_name, {})
    status = entry.get("status")
    # Only a drawn figure is embedded; anything else gets a one-line reason.
    if status != "written":
        return MISSING_FIGURE_NOTE.format(title=title, status=status, reason=entry.get("error") or NO_REASON_TEXT) + "\n"
    # The manifest path is repo-relative; only its file name is joined to the report-relative folder.
    image_path = posixpath.join(figures_relative_dir, posixpath.basename(entry["path"]))
    return f"![{title}]({image_path})\n\n{figure_reading_markdown(figure_name, values)}\n"


def whole_scene_section_markdown(summary: Mapping, manifest: Mapping, figures_relative_dir: str,
                                 values: Mapping, heading: str = "## Whole scene") -> str:
    """The whole-scene section: tables, pair gaps, then the figures of ``FIGURE_NAMES`` in order.

    Args:
        summary: scripts/20 summary.
        manifest: scripts/26 manifest of the same session.
        figures_relative_dir: Figures folder relative to the report's folder.
        values: ``definition_values`` output of this session.
        heading: Section heading line.

    Returns:
        Markdown.
    """
    parts = [heading, "", whole_scene_tables_markdown(summary)]
    parts += [figure_block_markdown(figure_name, manifest, figures_relative_dir, values) for figure_name in FIGURE_NAMES]
    return "\n".join(parts) + "\n"


def whole_scene_opening_markdown(summary: Mapping, manifest: Mapping, figures_relative_dir: str) -> str:
    """Definitions followed by the whole-scene section: the opening of the scripts/20 report.

    Args:
        summary: scripts/20 summary with ``provenance``.
        manifest: scripts/26 manifest of the session.
        figures_relative_dir: Figures folder relative to the report's folder (e.g. ``../figures/<session>``).

    Returns:
        Markdown.

    Raises:
        KeyError: The summary or manifest lacks a required field.
    """
    values = manifest_definition_values(summary, manifest)
    return (definitions_markdown(values) + "\n"
            + whole_scene_section_markdown(summary, manifest, figures_relative_dir, values))


# ---- Stand-alone terminology document ----
# Section headings of the terminology document, in order (the tests check this order).
SECTION_WHAT_IS_EVALUATED = "## What is evaluated and why"
SECTION_GLOSSARY = "## Glossary"
SECTION_DEFINITIONS = "## Definitions"  # heading that definitions_markdown itself prints
SECTION_FIGURES = "## Reading the figures"
SECTION_VALUES = "## Values used, per session"
SECTION_CAVEATS = "## Assumptions and caveats"
SECTION_REFERENCES = "## References"
SECTION_CODE = "### Code"
TERMINOLOGY_SECTION_ORDER = (SECTION_WHAT_IS_EVALUATED, SECTION_GLOSSARY, SECTION_DEFINITIONS, SECTION_FIGURES,
                             SECTION_VALUES, SECTION_CAVEATS, SECTION_REFERENCES, SECTION_CODE)
# Figures whose reading guide the terminology document prints: every figure of the reports.
TERMINOLOGY_FIGURE_NAMES = FIGURE_NAMES
# Section 1: the research question (CLAUDE.md, first paragraph) and how the comparison answers it.
WHAT_IS_EVALUATED_TEXT = (
    "The research question is: does adding 4D radar to lidar + camera SplatAD improve tree-trunk reconstruction "
    "when tall grass occludes the trunks? It is answered by training three arms (stock, radar init, radar init + loss) "
    "several times on a May and a June session, scoring each trained model's occupancy of the obstacles (trunks, rocks, "
    "bushes) against hand-labelled boxes, and comparing arms only by the gap against the spread over the repeats.")
# Value-table rows: (row label, key of ``definition_values``, format spec of the value). Specs match the definitions' text.
VALUE_TABLE_ROWS = (
    ("Voxel (cell) edge (m)", "voxel_size", "g"),
    ("Margin, tree_trunk (m)", "margin_tree_trunk", ".3f"),
    ("Margin, rock (m)", "margin_rock", ".3f"),
    ("Margin, bush_or_small_tree (m)", "margin_bush_or_small_tree", ".3f"),
    ("Scored range (m)", "max_range_m", "g"),
    ("Vertical half field of view (degrees)", "vertical_half_fov_deg", ".1f"),
    ("Near band (m)", "near_band_m", "g"),
    ("Cells on obstacles, whole scene", "n_on", "d"),
    ("Cells next to obstacles, whole scene", "n_off", "d"),
    ("Repeats per arm", "repeat_count", "d"),
    ("P2 cut-offs (first = binding)", "p2_thresholds", "s"),
    ("Curve range, from", "curve_min", "g"),
    ("Curve range, to", "curve_max", "g"),
    ("Curve points", "curve_points", "d"),
)
# Section 6: assumptions and caveats, each stated as the evaluation reports state them.
CAVEAT_BULLETS = (
    "**Centre-of-cell rule and thin trunks.** A cell belongs to a box by its centre only, and boxes are rectangular. A trunk "
    "thinner than one cell therefore gets 0, 1 or 2 cells across depending on where the grid lines fall.",
    "**Unlabelled grass and ground next to obstacles.** The ring of cells next to an obstacle also holds things nobody labelled "
    "(grass, ground, small plants) that really are solid, so precision stays below 1 even for a correct model. This is a likely "
    "cause of low precision, not a tested finding.",
    "**Occupancy is a blocking probability at a point, not filled volume.** It is evaluated at the cell centre (or at a "
    "Gaussian's peak for the cell holding its mean) and assumes each Gaussian blocks light independently.",
    "**One drawn run per arm in the 3D figure.** The 3D figure shows the median repeat of each arm. It illustrates and decides nothing; "
    "arms are separated only by the spread over all repeats.",
    "**Run-to-run noise.** SplatAD runs with the same seed differ, so every comparison needs repeats and is read as the gap against the "
    "spread within the arms.",
)


def format_table_value(value: Any, format_spec: str) -> str:
    """One value-table cell; text values (e.g. "not available", unequal repeat counts) print as they are.

    Args:
        value: Value from ``definition_values``.
        format_spec: Format spec of the row.

    Returns:
        Cell text.
    """
    return value if isinstance(value, str) and format_spec != "s" else format(value, format_spec)


def values_table_markdown(session_values: Mapping[str, Mapping[str, Any]]) -> str:
    """The values each session's evaluation used, one column per session.

    Args:
        session_values: Session name -> ``definition_values`` output.

    Returns:
        Markdown table; every cell is read from ``session_values`` and formatted with the row's spec.

    Raises:
        KeyError: A session's values lack a row's key.
    """
    session_names = list(session_values)
    lines = ["| value | " + " | ".join(session_names) + " |", "|---|" + "---|" * len(session_names)]
    for row_label, value_key, format_spec in VALUE_TABLE_ROWS:
        cells = [format_table_value(session_values[session_name][value_key], format_spec) for session_name in session_names]
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def references_markdown() -> str:
    """The reference list (SplatAD, Kerbl, GaussianFormer) and the code map.

    Returns:
        Markdown: a bullet per reference, then a ``### Code`` list of definition -> file and symbol.
    """
    lines = [f"- {format_reference(reference)}"
             for reference in (SPLATAD_REFERENCE, KERBL_REFERENCE, GAUSSIANFORMER_REFERENCE)]
    lines += ["", SECTION_CODE, ""]
    lines += [f"- {definition}: `{source.path}` (`{source.anchor}`)" for definition, source in CODE_REFERENCES]
    return "\n".join(lines)


def definitions_note(first_session: str) -> str:
    """One line saying whose values the definitions carry.

    Args:
        first_session: Session whose values fill the definitions.

    Returns:
        Markdown line pointing to the per-session values for the other sessions.
    """
    return (f"*The definitions below are rendered once, with the values of session {first_session}; "
            "the per-session values table and sections give the other sessions' values.*")


def terminology_markdown(values: Mapping, session_values: Mapping[str, Mapping[str, Any]]) -> str:
    """Stand-alone terminology document: what is evaluated, glossary, definitions, figure guides, values, caveats, references.

    Args:
        values: ``definition_values`` output the definitions and figure guides are rendered with (first session).
        session_values: Session name -> ``definition_values`` output, for the value table.

    Returns:
        Markdown.
    """
    first_session = values["session"]
    parts = [TERMINOLOGY_TITLE, "", SECTION_WHAT_IS_EVALUATED, "", WHAT_IS_EVALUATED_TEXT, "",
             SECTION_GLOSSARY, "", glossary_markdown(), "",
             definitions_note(first_session), "", definitions_markdown(values), "", SECTION_FIGURES]
    for figure_name in TERMINOLOGY_FIGURE_NAMES:
        parts += ["", f"### How to read: {FIGURE_TITLES[figure_name]}", "", figure_reading_markdown(figure_name, values)]
    parts += ["", SECTION_VALUES, "", values_table_markdown(session_values), "", SECTION_CAVEATS, ""]
    parts += [f"- {bullet}" for bullet in CAVEAT_BULLETS]
    parts += ["", SECTION_REFERENCES, "", references_markdown()]
    return "\n".join(parts) + "\n"


def write_terminology(output_path: Path, values: Mapping, session_values: Mapping[str, Mapping[str, Any]]) -> None:
    """Write the stand-alone terminology document.

    Args:
        output_path: File to write (parent folder is created).
        values: ``definition_values`` output for the definitions and figure guides.
        session_values: Session name -> ``definition_values`` output, for the value table.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(terminology_markdown(values, session_values))
