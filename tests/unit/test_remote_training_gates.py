"""Remote training remains gated by tests, data validation, and matching reviewed preflight."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest
import yaml

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.dataset.writer import write_dataset


def remote_fixture(tmp_path, monkeypatch):
    """Build a synthetic workdisk/code snapshot and fake successful cluster subprocesses."""
    root = Path(__file__).resolve().parents[2]
    code_dir, stage_root, env_prefix = (tmp_path / name for name in ("code", "stage", "env"))
    (code_dir / "scripts/slurm").mkdir(parents=True)
    shutil.copy(root / "scripts/slurm/submit_run.py", code_dir / "scripts/slurm/submit_run.py")
    (code_dir / "configs/sessions").mkdir(parents=True)
    config = yaml.safe_load((root / "configs/slurm.yaml").read_text())
    config["profiles"] = {"tiny": config["profiles"]["may_scene_a"]}
    (code_dir / "configs/slurm.yaml").write_text(yaml.safe_dump(config))
    session_config = yaml.safe_load((root / "configs/sessions/june_tall_grass.yaml").read_text())
    session_config["splatad"]["add_missing_points"] = False
    (code_dir / "configs/sessions/synthetic.yaml").write_text(yaml.safe_dump(session_config))
    write_dataset(stage_root / "data/synthetic", build_synthetic_dataset())
    specification = importlib.util.spec_from_file_location("remote_worker", root / "scripts/slurm/remote_run.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    calls = []

    def fake_run(arguments, **kwargs):
        """Record validation calls and emulate one generated overlay."""
        calls.append(arguments)
        if any(str(argument).endswith("preflight_dataparser.py") for argument in arguments):
            output_dir = Path(arguments[arguments.index("--out") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "overlay.png").write_bytes(b"synthetic diagnostic artifact")
        return subprocess.CompletedProcess(arguments, 0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    return module, code_dir, stage_root, env_prefix, calls


def test_remote_preflight_records_validation_and_training_uses_remote_metadata(tmp_path, monkeypatch):
    """Successful validation produces a receipt and launch fractions come from workdisk data."""
    module, code_dir, stage_root, env_prefix, calls = remote_fixture(tmp_path, monkeypatch)
    module.run_remote(stage_root, code_dir, env_prefix, "synthetic", "tiny", "revision", "run", True)
    receipt_path = stage_root / "preflights/synthetic/tiny/revision/validation.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["validation_backend"] == "native-cluster"
    assert receipt["overlays"]
    assert any("pytest" in arguments for arguments in calls)
    assert any(any(str(value).endswith("06_validate_contract.py") for value in arguments) for arguments in calls)
    executed = []
    monkeypatch.setattr(module.os, "execve", lambda executable, arguments, environment: executed.append(environment))
    module.run_remote(stage_root, code_dir, env_prefix, "synthetic", "tiny", "revision", "run", False)
    assert executed[0]["DATASET_START_FRACTION"] == "0.0"
    assert executed[0]["DATASET_END_FRACTION"] == "1.0"
    assert executed[0]["TRAIN_SPLIT_FRACTION"] == "0.8"


def test_modified_dataset_invalidates_preflight_before_any_training(tmp_path, monkeypatch):
    """A changed manifest cannot reuse an earlier preflight's approval."""
    module, code_dir, stage_root, env_prefix, calls = remote_fixture(tmp_path, monkeypatch)
    module.run_remote(stage_root, code_dir, env_prefix, "synthetic", "tiny", "revision", "run", True)
    manifest_path = stage_root / "data/synthetic/manifest.json"
    manifest_path.write_text(manifest_path.read_text() + "\n")
    with pytest.raises(ValueError, match="provenance differs"):
        module.run_remote(stage_root, code_dir, env_prefix, "synthetic", "tiny", "revision", "run", False)


def test_failed_revalidation_removes_old_preflight_receipt(tmp_path, monkeypatch):
    """A failed check leaves no usable receipt even after an earlier successful preflight."""
    module, code_dir, stage_root, env_prefix, calls = remote_fixture(tmp_path, monkeypatch)
    module.run_remote(stage_root, code_dir, env_prefix, "synthetic", "tiny", "revision", "run", True)

    def fail_validation(arguments, **kwargs):
        """Reject the first validation subprocess to simulate a failed rerun."""
        raise subprocess.CalledProcessError(1, arguments)

    monkeypatch.setattr(module.subprocess, "run", fail_validation)
    with pytest.raises(subprocess.CalledProcessError):
        module.run_remote(stage_root, code_dir, env_prefix, "synthetic", "tiny", "revision", "run", True)
    assert not (stage_root / "preflights/synthetic/tiny/revision/validation.json").exists()
