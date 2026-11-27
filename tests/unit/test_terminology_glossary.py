"""Tests of the terminology document: section order, glossary sources, value table and topic coverage."""

import re
from pathlib import Path

import pytest

from viking_splatad.eval import report_opening
from viking_splatad.eval.evaluation_terminology import definition_values
from viking_splatad.eval.terminology_glossary import (
    CODE_REFERENCES,
    GLOSSARY,
    PYTHON_SUFFIX,
    TOPIC_ORDER,
    WhereRef,
    glossary_by_topic,
    glossary_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SESSION_NAMES = ("may_x", "june_y")
# Matches a transform id anchor such as T31.
TRANSFORM_ANCHOR = re.compile(r"^T\d+$")


def synthetic_values(offset: float) -> dict:
    """``definition_values`` of a synthetic session; ``offset`` makes the two sessions differ.

    Args:
        offset: Added to the margins, range and counts so each session's column is distinct.

    Returns:
        ``definition_values`` output.
    """
    summary = {"session": "s", "p2_thresholds": ["0.5", "0.8"], "arms": {"a": {"runs": [{}, {}]}, "b": {"runs": [{}, {}]}},
               "provenance": {"voxel_size_m": 0.25 + offset, "scored_range_m": 70.0 + offset, "scored_range_quantile": 0.9999,
                              "vertical_half_fov_rad": 0.5,
                              "margin_m_by_class": {"tree_trunk": 0.5 + offset, "rock": 0.6, "bush_or_small_tree": 0.9}}}
    return definition_values(summary, {"min": 0.1, "max": 0.9, "points": 9},
                             {"on_obstacle": int(100 + offset), "next_to_obstacle": 200},
                             {"stock": {"hit": 40, "missed": int(60 + offset), "false_alarm": 10, "context": 300}})


def anchor_found(where_ref: WhereRef) -> bool:
    """Whether a reference's anchor occurs in its file, by the rule of its file type.

    Args:
        where_ref: Reference to check.

    Returns:
        True for a Python identifier that is defined (``def``, ``class``, constant, field or parameter line) or a
        transform id registered with ``@transform``, or for a phrase that occurs verbatim in the file.
    """
    source_text = (REPO_ROOT / where_ref.path).read_text()
    if not where_ref.path.endswith(PYTHON_SUFFIX):
        return where_ref.anchor in source_text
    if TRANSFORM_ANCHOR.match(where_ref.anchor):
        return f'@transform("{where_ref.anchor}"' in source_text
    # A phrase (not an identifier) is matched verbatim.
    if not re.fullmatch(r"\w+", where_ref.anchor):
        return where_ref.anchor in source_text
    name = re.escape(where_ref.anchor)
    # An identifier must be defined: def, class, module constant, or an annotated field / parameter line.
    definition_pattern = rf"^\s*(def|class) {name}\b|^{name}\s*[:=]|^\s+{name}\s*:"
    return re.search(definition_pattern, source_text, re.MULTILINE) is not None


def test_every_section_present_in_order():
    """All sections of the terminology document appear, in the specified order."""
    values = synthetic_values(0.0)
    text = report_opening.terminology_markdown(values, {name: values for name in SESSION_NAMES})
    positions = [text.index(heading) for heading in report_opening.TERMINOLOGY_SECTION_ORDER]
    assert positions == sorted(positions) and len(set(positions)) == len(positions)
    assert text.count("## How to read") == len(report_opening.TERMINOLOGY_FIGURE_NAMES)
    assert "occupancy_boxplots" not in text and "Occupancy on and next to obstacles" not in text


@pytest.mark.parametrize("entry", GLOSSARY, ids=lambda entry: entry.term)
def test_glossary_sources_exist(entry):
    """Each entry names a source; its file exists and the named symbol or snippet is in it."""
    assert entry.where, f"{entry.term}: no source"
    for where_ref in entry.where:
        assert (REPO_ROOT / where_ref.path).is_file(), f"{entry.term}: missing file {where_ref.path}"
        assert anchor_found(where_ref), f"{entry.term}: {where_ref.anchor!r} not found in {where_ref.path}"


@pytest.mark.parametrize("code_reference", CODE_REFERENCES, ids=lambda reference: reference[0])
def test_code_references_exist(code_reference):
    """Each entry of the code list points at an existing symbol."""
    _, where_ref = code_reference
    assert anchor_found(where_ref), f"{where_ref.anchor!r} not found in {where_ref.path}"


def test_every_topic_has_an_entry_and_is_rendered():
    """Every topic of the specification has at least one entry, and its heading is printed."""
    grouped = glossary_by_topic()
    assert all(grouped[topic] for topic in TOPIC_ORDER)
    rendered = glossary_markdown()
    assert all(f"### {topic}" in rendered for topic in TOPIC_ORDER)
    assert all(f"**{entry.term}.**" in rendered for entry in GLOSSARY)


def test_values_table_equals_the_session_values():
    """The table has one column per session and every cell equals the formatted session value."""
    session_values = {SESSION_NAMES[0]: synthetic_values(0.0), SESSION_NAMES[1]: synthetic_values(1.0)}
    table_lines = report_opening.values_table_markdown(session_values).split("\n")
    assert table_lines[0] == "| value | " + " | ".join(SESSION_NAMES) + " |"
    body_lines = table_lines[2:]
    assert len(body_lines) == len(report_opening.VALUE_TABLE_ROWS)
    for body_line, (row_label, value_key, format_spec) in zip(body_lines, report_opening.VALUE_TABLE_ROWS):
        expected_cells = [format(session_values[name][value_key], format_spec) for name in SESSION_NAMES]
        assert body_line == f"| {row_label} | " + " | ".join(expected_cells) + " |"
    # First-principles spot check: the two sessions differ in the voxel edge, so their cells differ.
    assert body_lines[0].split("|")[2].strip() != body_lines[0].split("|")[3].strip()


def test_whole_scene_gap_table_prints_sep_ovl_na():
    """The separated flag prints as sep / ovl / n/a, never as True / False / None."""
    assert [report_opening.separation_label(flag) for flag in (True, False, None)] == ["sep", "ovl", "n/a"]


def test_glossary_has_no_hidden_cell_terms():
    """The rendered glossary names no hidden-cell, near-band or range-margin term."""
    rendered = glossary_markdown()
    for forbidden in ("hidden", "Hidden", "H_near", "near band", "Range margin", "T92", "T93"):
        assert forbidden not in rendered
    assert not any(code_reference[1].anchor == "T92" for code_reference in CODE_REFERENCES)
