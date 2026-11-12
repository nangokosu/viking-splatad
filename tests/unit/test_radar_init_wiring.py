"""Deployment regressions for the stock and radar-initialization method paths."""

import importlib.util
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUBMIT_SCRIPT = PROJECT_ROOT / "scripts" / "slurm" / "submit_run.py"


def _load_submit_module():
    """Load ``submit_run.py`` without requiring it to be a Python package."""
    module_spec = importlib.util.spec_from_file_location("radar_submit_run", SUBMIT_SCRIPT)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _write_session_config(config_directory: Path, method: str, radar_directory) -> None:
    """Write the subset of a synthetic session YAML consumed by method resolution."""
    config_directory.mkdir(parents=True)
    config = {
        "split": {},
        "splatad": {
            "method": method,
            "radar_inputs_dir": radar_directory,
            "radar_seed_opacity": 0.25,
        },
    }
    (config_directory / "synthetic.yaml").write_text(yaml.safe_dump(config))


def test_method_resolution_supports_stock_and_initialization_only(tmp_path):
    """Stock has no radar artifact while the new method exports all three run values."""
    submit_module = _load_submit_module()
    config_directory = tmp_path / "sessions"
    _write_session_config(config_directory, "splatad", None)
    assert submit_module.session_radar_initialization(
        "synthetic", tmp_path / "data", config_directory
    ) == ("splatad", None, 0.25)
    _write_session_config(config_directory, "splatad-radar-init", "radar_inputs")
    assert submit_module.session_radar_initialization(
        "synthetic", tmp_path / "data", config_directory
    ) == ("splatad-radar-init", "radar_inputs", 0.25)


def test_method_resolution_rejects_loss_branch_and_absolute_artifact_path(tmp_path):
    """Initialization wiring cannot silently select the excluded loss method or a local path."""
    submit_module = _load_submit_module()
    config_directory = tmp_path / "sessions"
    _write_session_config(config_directory, "splatad-radar-lidar", None)
    with pytest.raises(ValueError, match="splatad.method"):
        submit_module.session_radar_initialization("synthetic", tmp_path / "data", config_directory)
    _write_session_config(config_directory, "splatad-radar-init", "/personal/radar_inputs")
    with pytest.raises(ValueError, match="relative"):
        submit_module.session_radar_initialization("synthetic", tmp_path / "data", config_directory)


def test_batch_script_keeps_literal_stock_command_and_separate_radar_command():
    """The original stock invocation remains intact and the extension adds no loss flags."""
    batch_text = (PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch").read_text()
    assert "\nns-train splatad \\\n" in batch_text
    assert "\nns-train splatad-radar-init \\\n" in batch_text
    radar_command = batch_text[batch_text.index("ns-train splatad-radar-init"):]
    assert '--radar-inputs-dir "$RADAR_INPUTS_DIR"' in radar_command
    assert '--pipeline.model.radar-seed-opacity "$RADAR_SEED_OPACITY"' in radar_command
    assert "radar-lidar-loss" not in radar_command
