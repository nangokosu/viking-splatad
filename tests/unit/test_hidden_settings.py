"""Tests for the hidden-voxel evaluation settings parser (configs/eval.yaml boundary)."""

import copy
from pathlib import Path

import pytest
import yaml

from viking_splatad.eval.hidden_settings import class_group_names_ordered, parse_hidden_settings

EVAL_YAML_PATH = Path(__file__).resolve().parents[2] / "configs" / "eval.yaml"


def _valid_section():
    """Return a fresh synthetic valid eval section."""
    return {
        "render_accumulation_threshold": 0.5,
        "p2_thresholds": [0.5, 0.8],
        "scored_class_ids": {"tree_trunk": 1, "bush_or_small_tree": 4, "rock": 3},
        "class_groups": {"obstacles": [1, 3, 4]},
        "hidden_space": {
            "margin_quantile": 0.9, "margin_sweep_gap": 5, "margin_sweep_stride": 50,
            "margin_neighbour_count": 8, "margin_planarity_min": 0.4,
            "margin_max_range_m": 15.0, "scored_range_quantile": 0.999, "near_band_m": 20.0, "worker_count": None,
        },
    }


def test_real_config_parses():
    """The committed configs/eval.yaml is valid and has the designed values."""
    section = yaml.safe_load(EVAL_YAML_PATH.read_text())["eval"]
    settings = parse_hidden_settings(section)
    assert settings.p2_thresholds == (0.5, 0.8)
    assert settings.class_groups["obstacles"] == (1, 3, 4)
    assert settings.binding_threshold == 0.5


def test_synthetic_values_pass_through():
    """Parsed fields equal the input values."""
    settings = parse_hidden_settings(_valid_section())
    assert settings.margin_sweep_gap == 5 and settings.worker_count is None
    assert settings.margin_max_range_m == 15.0


def test_group_names_first_then_classes():
    """Report order is groups, then scored classes in config order."""
    section = _valid_section()
    settings = parse_hidden_settings(section)
    assert class_group_names_ordered(settings, section["scored_class_ids"]) == [
        "obstacles", "tree_trunk", "bush_or_small_tree", "rock"]


def _mutate_hidden(key, value):
    """Return a valid section with hidden_space[key] = value."""
    section = _valid_section()
    section["hidden_space"][key] = value
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
    (_mutate_hidden("margin_quantile", 1.0), "margin_quantile"),
    (_mutate_hidden("margin_quantile", 0.0), "margin_quantile"),
    (_mutate_hidden("margin_sweep_gap", 0), "margin_sweep_gap"),
    (_mutate_hidden("margin_sweep_gap", 2.5), "margin_sweep_gap"),
    (_mutate_hidden("margin_sweep_stride", -1), "margin_sweep_stride"),
    (_mutate_hidden("margin_neighbour_count", 2), "margin_neighbour_count"),
    (_mutate_hidden("margin_planarity_min", 1.1), "margin_planarity_min"),
    (_mutate_hidden("margin_planarity_min", -0.1), "margin_planarity_min"),
    (_mutate_hidden("margin_max_range_m", 0.0), "margin_max_range_m"),
    (_mutate_hidden("scored_range_quantile", 0.0), "scored_range_quantile"),
    (_mutate_hidden("scored_range_quantile", 1.5), "scored_range_quantile"),
    (_mutate_hidden("near_band_m", 0.0), "near_band_m"),
    (_mutate_hidden("worker_count", 0), "worker_count"),
    (_mutate_hidden("worker_count", "all"), "worker_count"),
])
def test_invalid_value_raises_naming_key(section, key):
    """Each violated rule raises ValueError that names the key."""
    with pytest.raises(ValueError, match=key):
        parse_hidden_settings(section)


def test_missing_hidden_key_raises():
    """A missing hidden_space key is reported by name."""
    section = copy.deepcopy(_valid_section())
    del section["hidden_space"]["margin_quantile"]
    with pytest.raises(ValueError, match="margin_quantile"):
        parse_hidden_settings(section)
