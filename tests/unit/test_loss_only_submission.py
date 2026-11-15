"""Loss-only profiles preserve baseline settings and cannot bypass radar provenance gates."""

import json
from pathlib import Path

import pytest
import yaml

from tests.unit.test_remote_training_gates import remote_fixture
from tests.unit.test_slurm_submit import load_submit_module


def test_loss_only_profiles_differ_only_by_explicit_radar_flag():
    """Training/split/resource settings are byte-equivalent after removing the arm switch."""
    helpers = load_submit_module()
    config = helpers.load_slurm_config()
    for baseline_name, loss_name in (("june_scene_a", "june_loss_only"),
                                      ("smoke_a4500_june", "smoke_a4500_june_loss_only")):
        baseline, loss_only = config["profiles"][baseline_name], dict(config["profiles"][loss_name])
        assert loss_only.pop("radar_loss_only") is True
        assert loss_only == baseline
    original = {"method": "splatad", "loss_weight": 0., "gate_quantile": 0.95,
                "chunk_size": 4096, "inputs_dir": None}
    resolved = helpers.profile_radar_lidar_options(config["profiles"]["june_loss_only"], original,
                                                    "/stage", "june", "june_loss_only", "revision")
    assert original["method"] == "splatad" and original["inputs_dir"] is None
    assert resolved["method"] == "splatad-radar-lidar" and resolved["auto_weight"] is True
    assert resolved["loss_weight"] == 0.
    assert resolved["inputs_dir"] == "/stage/radar-inputs/june/june_loss_only/revision"
    with pytest.raises(ValueError, match="train_lidar_only"):
        helpers.profile_radar_lidar_options({"radar_loss_only": True}, original, "/stage", "june", "L", "rev")


def test_loss_only_worker_checks_artifact_and_records_automatic_weight(tmp_path, monkeypatch):
    """Real worker routing validates the exact artifact, hashes it and exports loss-only flags."""
    module, code_dir, stage_root, env_prefix, calls = remote_fixture(tmp_path, monkeypatch)
    config_path = code_dir / "configs/slurm.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["profiles"]["tiny"].update(radar_loss_only=True, train_lidar_only=True)
    config_path.write_text(yaml.safe_dump(config))
    artifact = stage_root / "radar-inputs/synthetic/tiny/revision"
    artifact.mkdir(parents=True)
    (artifact / "radar_seeds.npz").write_bytes(b"synthetic checksum-pinned artifact")
    (artifact / "radar_inputs.json").write_text("{}")
    validations = []
    from viking_splatad.radar_init import artifact as artifact_module
    monkeypatch.setattr(artifact_module, "load_verified_anchors", lambda *arguments: validations.append(arguments))
    module.run_remote(stage_root, code_dir, env_prefix, "synthetic", "tiny", "revision", "run", True)
    assert validations and validations[0][0] == artifact
    overlay_call = next(arguments for arguments in calls
                        if any(str(argument).endswith("preflight_dataparser.py") for argument in arguments))
    assert overlay_call[overlay_call.index("--radar-inputs-dir") + 1] == str(artifact)
    receipt = json.loads((stage_root / "preflights/synthetic/tiny/revision/validation.json").read_text())
    assert receipt["radar_artifact_sha256"] and receipt["radar_metadata_sha256"]
    executed = []
    monkeypatch.setattr(module.os, "execve", lambda executable, arguments, environment: executed.append(environment))
    module.run_remote(stage_root, code_dir, env_prefix, "synthetic", "tiny", "revision", "run", False)
    assert executed[0]["RADAR_LIDAR_AUTO_WEIGHT"] == "True"
    assert executed[0]["RADAR_LIDAR_LOSS_WEIGHT"] == "0.0"
    assert Path(executed[0]["RADAR_INPUTS_DIR"]) == artifact
    assert executed[0]["TRAIN_LIDAR_ONLY"] == "True"


def test_prepare_radar_cli_dispatches_without_local_data_work(monkeypatch):
    """Mac entrypoint dispatches a CPU job; it does not run numerical radar work locally."""
    helpers = load_submit_module()
    called = []
    monkeypatch.setattr(helpers, "submit_radar_preparation", lambda *arguments: called.append(arguments))
    assert helpers.main(["prepare-radar", "--session", "june_tall_grass", "--profile", "june_loss_only",
                         "--afterok", "123"]) == 0
    assert called[0][1:] == ("june_tall_grass", "june_loss_only", "123")
