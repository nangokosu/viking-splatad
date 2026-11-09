"""Completed extraction can be validated and published without another conversion."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest
import yaml

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.dataset.writer import write_dataset


def publication_fixture(tmp_path, monkeypatch):
    """Create completed synthetic output and record validator calls; return helper and paths."""
    script_path = Path(__file__).resolve().parents[2] / "scripts/slurm/publish_prepared_data.py"
    specification = importlib.util.spec_from_file_location("publish_data", script_path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    stage_root, code_dir = tmp_path / "stage", tmp_path / "code"
    dataset_dir = stage_root / "data-prep/workspace/dataset"
    write_dataset(dataset_dir, build_synthetic_dataset(num_frames=3))
    config_dir = code_dir / "configs/sessions"
    config_dir.mkdir(parents=True)
    chunk_files = ["synthetic_0.mcap"]
    (config_dir / "synthetic.yaml").write_text(yaml.safe_dump({
        "session": {"name": "synthetic"},
        "window": {"chunk_files": chunk_files}, "splatad": {"add_missing_points": False},
    }))
    (dataset_dir / "extract_state.json").write_text(json.dumps({
        "session_name": "synthetic", "extracted_chunks": chunk_files,
        "radar_frame_count": len(list((dataset_dir / "sidecar/radar").glob("*.npy"))),
    }))
    reference_dir = code_dir / "configs/reference_splits"
    reference_dir.mkdir()
    (reference_dir / "synthetic.json").write_text("{}")
    calls = []

    def fake_validation(arguments, check):
        """Record actual gate arguments without launching subprocesses."""
        assert check is True
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_validation)
    return module, stage_root, code_dir, dataset_dir, calls


def test_complete_output_is_published_only_through_validation(tmp_path, monkeypatch):
    """Both gates run, no extractor runs, and manifest/poses bytes remain identical."""
    module, stage_root, code_dir, dataset_dir, calls = publication_fixture(tmp_path, monkeypatch)
    original_files = {name: (dataset_dir / name).read_bytes() for name in ("manifest.json", "poses.npz")}
    destination = module.publish_prepared(stage_root, code_dir, "synthetic", dataset_dir)
    assert [Path(arguments[1]).name for arguments in calls] == ["06_validate_contract.py", "16_check_reference_split.py"]
    assert not dataset_dir.exists()
    for name, contents in original_files.items():
        assert (destination / name).read_bytes() == contents
    assert json.loads((destination / "preparation.json").read_text())["validation_only_recovery"]


def test_incomplete_extraction_is_never_published(tmp_path, monkeypatch):
    """Missing completion records are rejected before any validation or movement."""
    module, stage_root, code_dir, dataset_dir, calls = publication_fixture(tmp_path, monkeypatch)
    state_path = dataset_dir / "extract_state.json"
    state = json.loads(state_path.read_text())
    state["extracted_chunks"] = []
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match="complete extraction"):
        module.publish_prepared(stage_root, code_dir, "synthetic", dataset_dir)
    assert dataset_dir.exists()
    assert calls == []
    assert not (stage_root / "data/synthetic").exists()
