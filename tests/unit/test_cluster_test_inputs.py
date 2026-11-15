"""Cluster validation uses real prerequisites rather than weakening skip enforcement."""

import importlib.util
from pathlib import Path

import pytest
import yaml


def load_inputs_module():
    """Load the provisioning helper directly; return its module without executing its CLI."""
    script_path = Path(__file__).resolve().parents[2] / "scripts/slurm/prepare_test_inputs.py"
    specification = importlib.util.spec_from_file_location("cluster_test_inputs", script_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_test_inputs_rebase_real_fixture_download_and_expose_expected_paths(tmp_path, monkeypatch):
    """May metadata/labels go on the workdisk and upstream source is linked from the pinned env."""
    module = load_inputs_module()
    code_dir, stage_root, env_prefix = (tmp_path / name for name in ("code", "stage", "env"))
    config_dir = code_dir / "configs/sessions"
    config_dir.mkdir(parents=True)
    session_paths = {name: f"data/{name}" for name in (
        "bag_dir", "calibration_dir", "labels_json", "output_dir", "gt_dir",
    )}
    (config_dir / "may_short_grass.yaml").write_text(yaml.safe_dump({"session": session_paths}))
    upstream = env_prefix / "viking-sources/neurad-studio"
    upstream.mkdir(parents=True)
    recorded_calls = []

    def fake_download(arguments, check):
        """Record the downloader arguments and emulate creating the shared fixture root."""
        assert check is True
        recorded_calls.append(arguments)
        (stage_root / "source-data/data").mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module.subprocess, "run", fake_download)
    module.prepare_test_inputs(code_dir, stage_root, env_prefix)
    runtime = yaml.safe_load((stage_root / "test-inputs/may.yaml").read_text())
    for name, relative_path in session_paths.items():
        assert runtime["session"][name] == str(stage_root / "source-data" / relative_path)
    assert recorded_calls[0][-1] == "--fetch-calibration"
    assert "--chunks" not in recorded_calls[0]
    assert (code_dir / "data").resolve() == (stage_root / "source-data/data").resolve()
    assert (code_dir / "third_party/neurad-studio").resolve() == upstream.resolve()
    # Provisioning can be retried after validation failure without replacing its inputs.
    module.prepare_test_inputs(code_dir, stage_root, env_prefix)


def test_conflicting_test_input_is_preserved(tmp_path):
    """Provisioning must not silently delete a dataset or source tree supplied by a user."""
    module = load_inputs_module()
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    with pytest.raises(FileExistsError):
        module.ensure_symlink(existing_dir, tmp_path / "target")
    assert existing_dir.is_dir()
