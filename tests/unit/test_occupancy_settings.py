"""Tests for the occupancy evaluation settings parser (configs/eval.yaml boundary)."""

import copy
from pathlib import Path

import pytest
import yaml

from viking_splatad.eval.occupancy_settings import class_group_names_ordered, parse_occupancy_settings

EVAL_YAML_PATH = Path(__file__).resolve().parents[2] / "configs" / "eval.yaml"


def _valid_section():
    """Return a fresh synthetic valid eval section."""
    return {
        "render_accumulation_threshold": 0.5,
        "p2_thresholds": [0.5, 0.8],
        "scored_class_ids": {"tree_trunk": 1, "bush_or_small_tree": 4, "rock": 3},
        "class_groups": {"obstacles": [1, 3, 4]},
        "scored_cells": {"scored_range_quantile": 0.999},
    }


def test_real_config_parses():
    """The committed configs/eval.yaml is valid and has the designed values."""
    section = yaml.safe_load(EVAL_YAML_PATH.read_text())["eval"]
    settings = parse_occupancy_settings(section)
    assert settings.p2_thresholds == (0.5, 0.8)
    assert settings.class_groups["obstacles"] == (1, 3, 4)
    assert settings.binding_threshold == 0.5


def test_synthetic_values_pass_through():
    """Parsed fields equal the input values."""
    settings = parse_occupancy_settings(_valid_section())
    assert settings.scored_range_quantile == 0.999
    assert settings.binding_threshold == 0.5 and settings.p2_thresholds == (0.5, 0.8)


def test_group_names_first_then_classes():
    """Report order is groups, then scored classes in config order."""
    section = _valid_section()
    settings = parse_occupancy_settings(section)
    assert class_group_names_ordered(settings, section["scored_class_ids"]) == [
        "obstacles", "tree_trunk", "bush_or_small_tree", "rock"]


def _mutate_scored(key, value):
    """Return a valid section with scored_cells[key] = value."""
    section = _valid_section()
    section["scored_cells"][key] = value
    return section


def _mutate_top(key, value):
    """Return a valid section with section[key] = value."""
    section = _valid_section()
    section[key] = value
    return section


@pytest.mark.parametrize("section, key", [
    (_mutate_top("p2_thresholds", [0.6, 0.8]), "p2_thresholds"),          # binding missing
    (_mutate_top("p2_thresholds", [0.5, 1.5]), "p2_thresholds"),          # out of range
    (_mutate_top("p2_thresholds", [0.0, 0.5]), "p2_thresholds"),          # 0 excluded
    (_mutate_top("p2_thresholds", [0.8, 0.5]), "p2_thresholds"),          # unsorted
    (_mutate_top("p2_thresholds", [0.5, 0.5, 0.8]), "p2_thresholds"),     # duplicate
    (_mutate_top("class_groups", {"obstacles": [1, 9]}), "class_groups"),  # unknown id
    (_mutate_top("class_groups", {"rock": [1, 3]}), "class_groups"),       # name collision
    (_mutate_scored("scored_range_quantile", 0.0), "scored_range_quantile"),
    (_mutate_scored("scored_range_quantile", 1.5), "scored_range_quantile"),
])
def test_invalid_value_raises_naming_key(section, key):
    """Each violated rule raises ValueError that names the key."""
    with pytest.raises(ValueError, match=key):
        parse_occupancy_settings(section)


def test_missing_scored_cells_key_raises():
    """A missing scored_cells key is reported by name."""
    section = copy.deepcopy(_valid_section())
    del section["scored_cells"]["scored_range_quantile"]
    with pytest.raises(ValueError, match="scored_range_quantile"):
        parse_occupancy_settings(section)
