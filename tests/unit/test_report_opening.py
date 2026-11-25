"""Tests of the shared report opening, the provenance of scripts/20 and the terminology file of scripts/21."""

import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval import report_opening
from viking_splatad.eval.report_opening import SCENE_COUNTS_KEY
from viking_splatad.eval.evaluation_terminology import FIGURE_NAMES
from viking_splatad.eval.p2_repeats import compare_arms, describe_arm

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
CLASS_NAMES = ["obstacles", "tree_trunk", "rock", "bush_or_small_tree"]
THRESHOLDS = ["0.5", "0.8"]
SESSIONS = ("may_session", "june_session")
FIGURES_RELATIVE_DIR = "../figures/{session}"
CURVE_GRID = [0.1, 0.3, 0.5, 0.9]
CELL_COUNTS = {"on_obstacle": 111, "next_to_obstacle": 222}
# Drawn runs' whole-scene outcome counts (manifest scene_counts); stock hit + missed equals
# CELL_COUNTS["on_obstacle"] (11 + 100 = 111), as definition_values requires.
OUTCOME_COUNTS_BY_ARM = {arm_key: {"hit": 11, "missed": 100, "false_alarm": 5, "context": 333}
                         for arm_key in ("stock", "radar_init", "radar_init_loss")}
PROVENANCE = {"voxel_size_m": 0.25, "near_band_m": 20.0, "max_range_m": 70.0, "vertical_half_fov_rad": 0.5,
              "margin_m_by_class": {"tree_trunk": 0.5, "rock": 0.6, "bush_or_small_tree": 0.9}}
REPEAT_VALUES = {"arm_stock": [0.2, 0.3, 0.4], "arm_radar": [0.6, 0.7, 0.8]}
ERROR_TEXT = "boom in drawing"


def load_script(file_name: str, module_name: str):
    """Import a numbered script as a module.

    Args:
        file_name: File name in scripts/.
        module_name: Module name.

    Returns:
        The module.
    """
    module_spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIR / file_name)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def synthetic_summary(session: str):
    """A summary with two arms of three repeats, whole-scene P2 rows and one comparison.

    Args:
        session: Session name.

    Returns:
        Summary dict.
    """
    metric_keys = [f"{class_name}/P2@{threshold}/{metric}/pooled" for class_name in CLASS_NAMES
                   for threshold in THRESHOLDS for metric in ("recall", "precision", "iou")]
    descriptions = {arm_name: describe_arm([{key: value for key in metric_keys} for value in values])
                    for arm_name, values in REPEAT_VALUES.items()}
    comparison = {"first": "arm_radar", "second": "arm_stock",
                  "metrics": compare_arms(descriptions["arm_radar"], descriptions["arm_stock"])}
    return {"session": session, "class_names": CLASS_NAMES, "p2_thresholds": THRESHOLDS,
            "arms": {arm_name: {"train_method": "m",
                                "runs": [{"run_id": f"{session}-{arm_name}-{run_index}"} for run_index, _ in enumerate(values)],
                                "metrics": descriptions[arm_name]}
                     for arm_name, values in REPEAT_VALUES.items()},
            "comparisons": [comparison], "provenance": copy.deepcopy(PROVENANCE)}


def synthetic_manifest(session: str, missing=()):
    """A manifest with all three figures written, except those named in ``missing``.

    Args:
        session: Session name.
        missing: Figure names to mark as failed.

    Returns:
        Manifest dict.
    """
    figures = {name: {"status": "written", "path": f"reports/gaussian_occupancy/figures/{session}/{name}.png"}
               for name in FIGURE_NAMES}
    for name in missing:
        figures[name] = {"status": "failed", "error": ERROR_TEXT}
    # Manifest arms: figure key -> summary arm key (profile); the drawn run is the arm's first run.
    arms = {figure_key: {"profile": arm_name, "drawn_run_id": f"{session}-{arm_name}-0", "campaign_run_count": len(REPEAT_VALUES[arm_name])}
            for figure_key, arm_name in (("stock", "arm_stock"), ("radar_init", "arm_radar"))}
    return {"session": session, "binding_threshold": float(THRESHOLDS[0]), "arms": arms, "figures": figures,
            "reported_numbers": {
                "threshold_curves_scene": {"curve_grid": CURVE_GRID},
                SCENE_COUNTS_KEY: {"band": "all", "cell_counts": CELL_COUNTS, "outcome_counts_by_arm": OUTCOME_COUNTS_BY_ARM}}}


def test_opening_order_definitions_then_scene_then_figures_in_order():
    """Definitions precede the whole-scene heading; the figures follow in FIGURE_NAMES order with relative links."""
    summary, manifest = synthetic_summary(SESSIONS[0]), synthetic_manifest(SESSIONS[0])
    relative_dir = FIGURES_RELATIVE_DIR.format(session=SESSIONS[0])
    text = report_opening.whole_scene_opening_markdown(summary, manifest, relative_dir)
    assert text.index("## Definitions") < text.index("## Whole scene")
    link_positions = [text.index(f"]({relative_dir}/{name}.png)") for name in FIGURE_NAMES]
    assert link_positions == sorted(link_positions) and link_positions[0] > text.index("## Whole scene")
    assert "](reports/" not in text and "](/" not in text


def test_opening_table_values_equal_the_summary():
    """Each arm cell is the formatted description of its summary row; the gap row has the summary gap and flag."""
    summary = synthetic_summary(SESSIONS[0])
    text = report_opening.whole_scene_opening_markdown(summary, synthetic_manifest(SESSIONS[0]), "figs")
    for class_name in CLASS_NAMES:
        for threshold in THRESHOLDS:
            key = f"{class_name}/P2@{threshold}/recall/pooled"
            for arm_name in REPEAT_VALUES:
                assert report_opening.format_description(summary["arms"][arm_name]["metrics"][key]) in text
            row = summary["comparisons"][0]["metrics"][key]
            assert f"| {key} | {row['gap']:.3f} | {report_opening.separation_label(row['separated'])} |" in text
    # First-principles: the radar arm mean is 0.7, std 0.1, range [0.6, 0.8], n = 3.
    assert "0.700 ± 0.100 [0.600, 0.800] (n=3)" in text


def test_figure_not_written_is_not_embedded_and_note_shows_error():
    """A failed figure has no image link; a one-line note carries the manifest's error."""
    summary = synthetic_summary(SESSIONS[0])
    manifest = synthetic_manifest(SESSIONS[0], missing=("threshold_curves_scene",))
    text = report_opening.whole_scene_opening_markdown(summary, manifest, "figs")
    assert "threshold_curves_scene.png" not in text and ERROR_TEXT in text
    assert "figs/arms_scene_3d.png" in text


def test_manifest_values_feed_the_definitions():
    """Curve grid ends and cell counts come from the manifest."""
    values = report_opening.manifest_definition_values(synthetic_summary(SESSIONS[0]), synthetic_manifest(SESSIONS[0]))
    assert (values["curve_min"], values["curve_max"], values["curve_points"]) == (CURVE_GRID[0], CURVE_GRID[-1], len(CURVE_GRID))
    assert (values["n_on"], values["n_off"]) == (CELL_COUNTS["on_obstacle"], CELL_COUNTS["next_to_obstacle"])


def synthetic_report(margin_shift: float = 0.0):
    """The report fields run_provenance reads.

    Args:
        margin_shift: Added to every class margin.

    Returns:
        Report dict.
    """
    return {"ground_truth": {"voxel_size_m": PROVENANCE["voxel_size_m"]}, "hidden_space": {"near_band_m": PROVENANCE["near_band_m"]},
            "settings": {"max_range_m": PROVENANCE["max_range_m"], "vertical_half_fov_rad": PROVENANCE["vertical_half_fov_rad"]},
            "classes": {name: {"margin_m": margin + margin_shift} for name, margin in PROVENANCE["margin_m_by_class"].items()}}


def test_run_provenance_copies_report_fields():
    """The provenance equals the values placed in the report."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_provenance")
    assert script.run_provenance(synthetic_report()) == PROVENANCE


def test_margin_mismatch_across_runs_raises():
    """summarize refuses pooled runs whose class margins differ (checked before any metric is pooled)."""
    script = load_script("20_summarize_gaussian_occupancy.py", "summarize_provenance_mismatch")
    cells = {"a": np.zeros(2)}
    runs = [{"run_dir": Path(f"run{index}"), "cells": cells,
             "report": {**synthetic_report(margin_shift), "run_id": f"run{index}", "profile": "p", "train_method": "splatad",
                        "ground_truth": synthetic_report()["ground_truth"], "settings": synthetic_report()["settings"],
                        "split_membership_sha256": "s", "hidden_space": synthetic_report()["hidden_space"]}}
            for index, margin_shift in enumerate((0.0, 0.1))]
    with pytest.raises(ValueError, match="provenance"):
        script.summarize(runs)


def test_script_21_writes_terminology_and_both_session_sections(tmp_path):
    """Both sessions get a whole-scene section after one definitions block; terminology.md has definitions and guides."""
    script = load_script("21_compare_sessions.py", "compare_openings")
    sessions = [{"session": name, "summary": synthetic_summary(name), "p2_thresholds": THRESHOLDS, "arms": {}} for name in SESSIONS]
    manifests = {name: synthetic_manifest(name) for name in SESSIONS}
    relative_dirs = {name: f"figures/{name}" for name in SESSIONS}
    opening = script.session_openings_markdown(sessions, manifests, relative_dirs)
    assert opening.count("## Definitions") == 1
    assert all(f"## Whole scene: {name}" in opening for name in SESSIONS)
    assert f"](figures/{SESSIONS[1]}/arms_scene_3d.png)" in opening
    # render_markdown keeps its old call form and places the opening after the intro when given.
    render_sessions = [{"session": name, "p2_thresholds": THRESHOLDS, "arms": {}} for name in SESSIONS]
    markdown = script.render_markdown(render_sessions, [], opening)
    assert markdown.index("## Definitions") < markdown.index(f"## {SESSIONS[0]}: arms")
    # The stand-alone file.
    values = report_opening.manifest_definition_values(sessions[0]["summary"], manifests[SESSIONS[0]])
    terminology_path = tmp_path / "terminology.md"
    report_opening.write_terminology(terminology_path, values, {sessions[0]["session"]: values})
    terminology_text = terminology_path.read_text()
    assert "## Definitions" in terminology_text and "## Whole scene" not in terminology_text
    assert terminology_text.count("## How to read") == len(report_opening.TERMINOLOGY_FIGURE_NAMES)


def test_arm_display_name_uses_the_plain_label_and_keeps_the_profile():
    """Known methods print their plain label with the profile; an unknown method falls back to its name."""
    summary = {"arms": {"june_scene_a_cam": {"train_method": "splatad"},
                        "june_scene_a_radar_init_loss_cam": {"train_method": "splatad-radar-init-lidar"},
                        "other_profile": {"train_method": "unknown-method"}}}
    assert report_opening.arm_display_name(summary, "june_scene_a_cam") == "stock (`june_scene_a_cam`)"
    assert (report_opening.arm_display_name(summary, "june_scene_a_radar_init_loss_cam")
            == "radar init + loss (`june_scene_a_radar_init_loss_cam`)")
    assert report_opening.arm_display_name(summary, "other_profile") == "unknown-method (`other_profile`)"


def test_failed_curve_figure_still_renders_with_not_available_curve_values():
    """A manifest without the curve grid renders: curve values read "not available" and the figure note shows."""
    manifest = synthetic_manifest(SESSIONS[0], missing=("threshold_curves_scene",))
    del manifest["reported_numbers"]["threshold_curves_scene"]
    values = report_opening.manifest_definition_values(synthetic_summary(SESSIONS[0]), manifest)
    assert values["curve_min"] == "not available" and values["curve_points"] == "not available"
    text = report_opening.whole_scene_opening_markdown(synthetic_summary(SESSIONS[0]), manifest, "figs")
    assert ERROR_TEXT in text and "threshold_curves_scene.png" not in text
    # The value table prints the text instead of failing on a numeric format spec.
    assert "not available" in report_opening.values_table_markdown({SESSIONS[0]: values})


def test_manifest_session_threshold_and_run_mismatches_raise():
    """Each of the three checks refuses a manifest drawn from other data, naming scripts/26."""
    summary = synthetic_summary(SESSIONS[0])
    # Wrong session.
    manifest = synthetic_manifest(SESSIONS[0])
    manifest["session"] = SESSIONS[1]
    with pytest.raises(ValueError, match="scripts/26"):
        report_opening.manifest_definition_values(summary, manifest)
    # Wrong binding threshold.
    manifest = synthetic_manifest(SESSIONS[0])
    manifest["binding_threshold"] = 0.8
    with pytest.raises(ValueError, match="scripts/26"):
        report_opening.manifest_definition_values(summary, manifest)
    # A drawn run that is not among the summary's runs of the arm.
    manifest = synthetic_manifest(SESSIONS[0])
    manifest["arms"]["stock"]["drawn_run_id"] = "some-other-run"
    with pytest.raises(ValueError, match="scripts/26"):
        report_opening.manifest_definition_values(summary, manifest)
    # The matching manifest passes.
    report_opening.manifest_definition_values(summary, synthetic_manifest(SESSIONS[0]))


def test_arm_matched_by_method_label_when_profile_is_absent():
    """Without a ``profile`` field the manifest arm is matched through the figure key's label and train_method."""
    summary = synthetic_summary(SESSIONS[0])
    summary["arms"]["arm_stock"]["train_method"] = "splatad"
    manifest = synthetic_manifest(SESSIONS[0])
    del manifest["arms"]["stock"]["profile"]
    report_opening.check_manifest_matches_summary(summary, manifest)
    manifest["arms"]["stock"]["drawn_run_id"] = "other"
    with pytest.raises(ValueError, match="scripts/26"):
        report_opening.check_manifest_matches_summary(summary, manifest)


def test_load_figure_manifest_reads_the_file_and_refuses_when_missing(tmp_path):
    """The helper reads the session's manifest and a missing file raises with the scripts/26 command."""
    with pytest.raises(ValueError, match="scripts/26_occupancy_figures.py --session may_session"):
        report_opening.load_figure_manifest(tmp_path, "may_session")
    manifest_path = tmp_path / report_opening.FIGURES_RELATIVE_DIR / "may_session" / report_opening.FIGURES_MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"session": "may_session"}))
    assert report_opening.load_figure_manifest(tmp_path, "may_session") == {"session": "may_session"}


def test_opening_names_the_session_and_curve_guide_uses_manifest_repeat_counts():
    """The definitions name the session; the curve band text states what it spans and the manifest's counts."""
    text = report_opening.whole_scene_opening_markdown(synthetic_summary(SESSIONS[0]), synthetic_manifest(SESSIONS[0]), "figs")
    assert f"Values for session {SESSIONS[0]}: trunk" in text and f"drawn run in {SESSIONS[0]} at t" in text
    assert "lowest and highest value of both the pooled and the mean-over-objects curve" in text
    assert "conservative reading" in text and "over the 3 repeats of the arm" in text
    # Unequal campaign counts show per arm.
    manifest = synthetic_manifest(SESSIONS[0])
    manifest["arms"]["radar_init"]["campaign_run_count"] = 2
    values = report_opening.manifest_definition_values(synthetic_summary(SESSIONS[0]), manifest)
    assert values["curve_repeat_count"] == "2–3 (stock 3, radar init 2)"


def test_terminology_names_the_session_of_its_definitions():
    """The terminology text says the definitions carry the first session's values."""
    values = report_opening.manifest_definition_values(synthetic_summary(SESSIONS[0]), synthetic_manifest(SESSIONS[0]))
    text = report_opening.terminology_markdown(values, {SESSIONS[0]: values})
    assert f"rendered once, with the values of session {SESSIONS[0]}" in text
