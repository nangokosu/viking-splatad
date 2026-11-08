"""Synthetic data-job orchestration checks; no network, bag extraction, or cluster access."""

import json
from pathlib import Path
import subprocess

import pytest
import yaml


def run_preparation_worker(monkeypatch, tmp_path: Path, fail_contract: bool = False):
    """Execute the batch worker with fake pipeline tools; return stage root and calls.

    The synthetic extractor writes only a sidecar directory. Contract failures
    simulate a rejected dataset and must never publish that dataset.
    """
    stage_root = tmp_path / "stage"
    code_dir = tmp_path / "code"
    config_dir = code_dir / "configs/sessions"
    config_dir.mkdir(parents=True)
    reference_dir = code_dir / "configs/reference_splits"
    reference_dir.mkdir()
    (reference_dir / "june.json").write_text("{}")
    config = {
        "session": {
            "name": "june", "bag_dir": "data/raw/june/extracted/forest",
            "calibration_dir": "data/raw/june/extracted/calibration",
            "labels_json": "data/raw/june/extracted/labels.json", "gt_dir": "data/gt/june",
            "output_dir": "data/intermediate/june",
        },
        "window": {"chunk_files": ["forest_20.mcap", "forest_21.mcap", "forest_22.mcap"]},
        "splatad": {"add_missing_points": True},
    }
    (config_dir / "june.yaml").write_text(yaml.safe_dump(config))
    for name, value in {"STAGE_ROOT": stage_root, "CODE_DIR": code_dir, "SESSION": "june"}.items():
        monkeypatch.setenv(name, str(value))
    calls = []

    def fake_run(arguments, check):
        """Record argv, emulate extraction, and optionally reject contract validation."""
        assert check is True
        calls.append(arguments)
        if arguments[1].endswith("03_extract.py"):
            runtime_config = yaml.safe_load(Path(arguments[arguments.index("--config") + 1]).read_text())
            output_dir = Path(runtime_config["session"]["output_dir"])
            (output_dir / "lidar_no_return").mkdir(parents=True)
            assert runtime_config["window"] == config["window"]
            assert Path(runtime_config["session"]["bag_dir"]).is_relative_to(stage_root)
        if fail_contract and arguments[1].endswith("06_validate_contract.py"):
            raise subprocess.CalledProcessError(1, arguments)
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    script_path = Path(__file__).resolve().parents[2] / "scripts/slurm/prepare_data.sbatch"
    worker_text = script_path.read_text().split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    exec(compile(worker_text, str(script_path), "exec"), {"__name__": "__main__"})
    return stage_root, calls


def test_preparation_uses_pinned_download_and_extraction_and_publishes_after_checks(monkeypatch, tmp_path):
    """Published data has passed contract/reference checks; extraction receives no chunk override."""
    stage_root, calls = run_preparation_worker(monkeypatch, tmp_path)
    assert [Path(arguments[1]).name for arguments in calls] == [
        "00_fetch_chunk.py", "03_extract.py", "06_validate_contract.py", "16_check_reference_split.py",
    ]
    download_arguments = calls[0]
    assert download_arguments[download_arguments.index("--chunks") + 1:] == ["20", "21", "22"]
    assert "--fetch-calibration" in download_arguments
    assert "--chunks" not in calls[1]
    receipt = json.loads((stage_root / "data/june/preparation.json").read_text())
    assert receipt["contract"] == "passed"
    assert receipt["reference_split"] == "passed"
    assert not (stage_root / "data-prep/june.lock").exists()


def test_contract_failure_keeps_data_unpublished_and_releases_lock(monkeypatch, tmp_path):
    """A failing validator leaves diagnostics, never a ready dataset."""
    with pytest.raises(subprocess.CalledProcessError):
        run_preparation_worker(monkeypatch, tmp_path, fail_contract=True)
    stage_root = tmp_path / "stage"
    assert not (stage_root / "data/june").exists()
    assert not (stage_root / "data-prep/june.lock").exists()
    assert list((stage_root / "data-prep").glob("june-*/dataset"))
