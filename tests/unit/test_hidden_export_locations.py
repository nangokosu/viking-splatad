"""Checkpoint discovery must cover both method names without guessing a trainer path."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize("method", ["splatad", "splatad-radar-lidar"])
def test_factor_export_finds_stock_and_radar_trainer_directories(tmp_path, method):
    """A real trainer layout resolves identically for both supported method names."""
    source = Path(__file__).resolve().parents[2] / "scripts/17_compare_hidden_trunks.py"
    specification = importlib.util.spec_from_file_location("compare_export_locations", source)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    directory = tmp_path / "train" / "june" / method / "run-id"
    directory.mkdir(parents=True)
    (directory / "dataparser_transforms.json").write_text("{}")
    assert module.final_train_directory(tmp_path) == directory
    other = tmp_path / "train" / "june" / method / "other-run"
    other.mkdir()
    (other / "dataparser_transforms.json").write_text("{}")
    with pytest.raises(ValueError, match="one final"):
        module.final_train_directory(tmp_path)
