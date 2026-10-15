"""Unit tests for config.py (session YAML loading, validation, path resolution)."""

from pathlib import Path

import pytest
import yaml

from viking_splatad.config import (
    REQUIRED_SESSION_KEYS,
    REQUIRED_TOP_LEVEL_KEYS,
    PATH_SESSION_KEYS,
    find_project_root,
    load_session_config,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SESSION_CONFIG_DIR = PROJECT_ROOT / "configs" / "sessions"


def test_project_root_contains_pyproject():
    """The discovered project root is the directory holding pyproject.toml."""
    assert find_project_root() == PROJECT_ROOT
    assert (find_project_root() / "pyproject.toml").is_file()


@pytest.mark.parametrize("session_file_name", ["may_short_grass.yaml", "june_tall_grass.yaml"])
def test_real_session_configs_load_with_absolute_paths(session_file_name):
    """Both real sessions load; every session path is an absolute Path under the root."""
    config = load_session_config(SESSION_CONFIG_DIR / session_file_name)
    # Every required top-level section is present.
    assert set(REQUIRED_TOP_LEVEL_KEYS) <= set(config)
    for path_key in PATH_SESSION_KEYS:
        resolved_path = config["session"][path_key]
        assert isinstance(resolved_path, Path)
        assert resolved_path.is_absolute()
        # Expected value is recomputed independently from the raw YAML text.
        raw_value = yaml.safe_load((SESSION_CONFIG_DIR / session_file_name).read_text())[
            "session"
        ][path_key]
        assert resolved_path == PROJECT_ROOT / raw_value


def test_may_and_june_use_distinct_radar_variants():
    """May and June must select different extrinsics variants."""
    may_config = load_session_config(SESSION_CONFIG_DIR / "may_short_grass.yaml")
    june_config = load_session_config(SESSION_CONFIG_DIR / "june_tall_grass.yaml")
    assert may_config["session"]["extrinsics_variant"] == "short_grass"
    assert june_config["session"]["extrinsics_variant"] == "high_grass"


def write_config_without(tmp_path, top_level_key=None, session_key=None):
    """Write a copy of the May config with one key removed; return its path."""
    raw_config = yaml.safe_load((SESSION_CONFIG_DIR / "may_short_grass.yaml").read_text())
    if top_level_key is not None:
        del raw_config[top_level_key]
    if session_key is not None:
        del raw_config["session"][session_key]
    broken_path = tmp_path / "broken.yaml"
    broken_path.write_text(yaml.safe_dump(raw_config))
    return broken_path


@pytest.mark.parametrize("missing_key", REQUIRED_TOP_LEVEL_KEYS)
def test_missing_top_level_key_names_the_key(tmp_path, missing_key):
    """Dropping any top-level section raises ValueError naming it."""
    with pytest.raises(ValueError, match=missing_key):
        load_session_config(write_config_without(tmp_path, top_level_key=missing_key))


@pytest.mark.parametrize("missing_key", REQUIRED_SESSION_KEYS)
def test_missing_session_key_names_the_key(tmp_path, missing_key):
    """Dropping any session key raises ValueError naming it."""
    with pytest.raises(ValueError, match=missing_key):
        load_session_config(write_config_without(tmp_path, session_key=missing_key))
