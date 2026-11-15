"""User decisions 2026-11-14 on radar seeds: random features_rest, opacity from stock, stock-row hash, contract robustness."""

import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit.test_loss_only_artifact import synthetic_artifact
from tests.unit.test_slurm_submit import load_submit_module
from viking_splatad.contract.splatad_contract import validate_radar_inputs
from viking_splatad.dataset.run_split import file_sha256
from viking_splatad.radar_init.build import build_loss_only_artifact


def _neurad_ready() -> bool:
    """Return whether the pinned torch/neurad stack is importable (true in the reference image)."""
    try:
        from nerfstudio.cameras.lidars import LidarType
    except ImportError:
        return False
    return hasattr(LidarType, "OUSTER_OS0_32")


needs_image = pytest.mark.skipif(not _neurad_ready(), reason="needs the reference image")

# Synthetic stock Gaussians and seeds; only shapes and distinct values matter here.
STOCK_ROW_COUNT = 3
FEATURES_REST_WIDTH = 5
FEATURES_DC_WIDTH = 3
SEED_COUNT = 2
SEED_COVARIANCE_VARIANCE = 0.01
RNG_SEED = 1234
STOCK_INIT_OPACITY_AT_APPEND = 0.3  # value of config.init_opacities when the seeds are appended
OVERRIDE_OPACITY = 0.25
SESSION_NAME = "synthetic"
REPOSITORY_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- model helpers


def _stock_factory():
    """Return a stand-in stock create_gauss_param_dict that draws features_rest from the torch RNG."""
    import torch

    def create_stock(model_instance, *arguments, **keyword_arguments):
        """Build STOCK_ROW_COUNT stock rows; features_rest is randn like stock SplatAD."""
        return torch.nn.ParameterDict({
            "means": torch.nn.Parameter(torch.arange(STOCK_ROW_COUNT * 3, dtype=torch.float32).reshape(-1, 3)),
            "scales": torch.nn.Parameter(torch.zeros((STOCK_ROW_COUNT, 3))),
            "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(STOCK_ROW_COUNT, 1)),
            "features_dc": torch.nn.Parameter(torch.zeros((STOCK_ROW_COUNT, FEATURES_DC_WIDTH))),
            "features_rest": torch.nn.Parameter(torch.randn(STOCK_ROW_COUNT, FEATURES_REST_WIDTH)),
            "opacities": torch.nn.Parameter(torch.zeros((STOCK_ROW_COUNT, 1))),
            "id": torch.nn.Parameter(torch.zeros((STOCK_ROW_COUNT, 1)), requires_grad=False),
        })

    return create_stock


def _seeds(with_seeds=True):
    """Return radar_seeds metadata (or None) with SEED_COUNT painted isotropic anchors."""
    import torch

    if not with_seeds:
        return None
    means = torch.arange(SEED_COUNT * 3, dtype=torch.float32).reshape(-1, 3) + 50.0
    covariances = (torch.eye(3) * SEED_COVARIANCE_VARIANCE).repeat(SEED_COUNT, 1, 1)
    return {"means": means, "covariances": covariances,
            "rgb": torch.full((SEED_COUNT, 3), 128, dtype=torch.uint8)}


def _build_gaussians(monkeypatch, with_seeds=True, init_opacities=None, **config_kwargs):
    """Run RadarInitSplatADModel.create_gauss_param_dict with stock patched; return (params, rng seeded run).

    Args:
        monkeypatch: pytest monkeypatch fixture.
        with_seeds: Whether radar_seeds metadata exists.
        init_opacities: Value assigned to config.init_opacities just before the call (None keeps it).
        **config_kwargs: RadarInitSplatADModelConfig overrides.

    Returns:
        The ParameterDict created under torch.manual_seed(RNG_SEED).
    """
    import torch
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_init_model import RadarInitSplatADModel, RadarInitSplatADModelConfig

    monkeypatch.setattr(SplatADModel, "create_gauss_param_dict", _stock_factory())
    model = RadarInitSplatADModel.__new__(RadarInitSplatADModel)
    torch.nn.Module.__init__(model)
    model.config = RadarInitSplatADModelConfig(**config_kwargs)
    if init_opacities is not None:
        model.config.init_opacities = init_opacities
    seeds = _seeds(with_seeds)
    model.kwargs = {"metadata": {} if seeds is None else {"radar_seeds": seeds}}
    torch.manual_seed(RNG_SEED)
    return model.create_gauss_param_dict([], [])


# ---------------------------------------------------------------- 1. features_rest


@needs_image
def test_seed_features_rest_are_random_like_stock(monkeypatch):
    """Seed rows equal torch.randn drawn right after stock's own draw; shape and dtype match stock."""
    import torch

    parameters = _build_gaussians(monkeypatch)
    torch.manual_seed(RNG_SEED)
    torch.randn(STOCK_ROW_COUNT, FEATURES_REST_WIDTH)  # stock's draw
    expected_seed_rows = torch.randn(SEED_COUNT, FEATURES_REST_WIDTH)
    seed_rows = parameters["features_rest"].detach()[STOCK_ROW_COUNT:]
    assert seed_rows.shape == (SEED_COUNT, FEATURES_REST_WIDTH)
    assert seed_rows.dtype == torch.float32
    assert not torch.all(seed_rows == 0)
    assert torch.equal(seed_rows, expected_seed_rows)


@needs_image
def test_stock_rows_identical_with_and_without_seeds(monkeypatch):
    """Appending seeds leaves every stock row bit-identical for the same RNG seed."""
    import torch

    with_seeds = _build_gaussians(monkeypatch, with_seeds=True)
    without_seeds = _build_gaussians(monkeypatch, with_seeds=False)
    for name in without_seeds:
        assert torch.equal(with_seeds[name].detach()[:STOCK_ROW_COUNT], without_seeds[name].detach())


# ---------------------------------------------------------------- 2. opacity from stock


@needs_image
def test_unset_seed_opacity_follows_init_opacities_at_append_time(monkeypatch):
    """With no override, seed opacity logits equal logit(config.init_opacities) when appended."""
    import torch

    parameters = _build_gaussians(monkeypatch, init_opacities=STOCK_INIT_OPACITY_AT_APPEND)
    seed_logits = parameters["opacities"].detach()[STOCK_ROW_COUNT:]
    expected_logit = math.log(STOCK_INIT_OPACITY_AT_APPEND / (1.0 - STOCK_INIT_OPACITY_AT_APPEND))
    assert torch.allclose(seed_logits, torch.full_like(seed_logits, expected_logit))


@needs_image
def test_override_wins_over_init_opacities(monkeypatch):
    """An explicit radar_seed_opacity is used instead of init_opacities."""
    import torch

    parameters = _build_gaussians(monkeypatch, init_opacities=STOCK_INIT_OPACITY_AT_APPEND,
                                  radar_seed_opacity=OVERRIDE_OPACITY)
    seed_logits = parameters["opacities"].detach()[STOCK_ROW_COUNT:]
    expected_logit = math.log(OVERRIDE_OPACITY / (1.0 - OVERRIDE_OPACITY))
    assert torch.allclose(seed_logits, torch.full_like(seed_logits, expected_logit))


@needs_image
def test_seed_opacity_default_is_none_and_validated_only_when_set():
    """None is accepted; a set value must lie strictly in (0, 1)."""
    from viking_splatad.splatad.radar_init_model import RadarInitSplatADModelConfig

    assert RadarInitSplatADModelConfig().radar_seed_opacity is None
    for invalid_value in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            RadarInitSplatADModelConfig(radar_seed_opacity=invalid_value)


# ---------------------------------------------------------------- 2b. resolver and environment


def _resolve(submit_module, session_extra, profile_extra):
    """Resolve an init_loss profile against a synthetic session/profile with optional opacity keys."""
    config = {"profiles": {"probe": {"seed": 0, "radar_variant": "init_loss", **profile_extra}}}
    session_config = {"session": {"name": SESSION_NAME}, "splatad": dict(session_extra)}
    return config, submit_module.profile_radar_variant_options(config, session_config, "probe")


def test_resolver_omits_opacity_unless_configured():
    """seed_opacity is None by default, taken from the session, and the profile overrides the session."""
    submit_module = load_submit_module()
    _, unset = _resolve(submit_module, {}, {})
    assert unset["seed_opacity"] is None
    _, from_session = _resolve(submit_module, {"radar_seed_opacity": 0.3}, {})
    assert from_session["seed_opacity"] == 0.3
    _, from_profile = _resolve(submit_module, {"radar_seed_opacity": 0.3}, {"radar_seed_opacity": 0.2})
    assert from_profile["seed_opacity"] == 0.2


def test_committed_session_yamls_do_not_set_seed_opacity():
    """Both session YAMLs leave the opacity to stock's init_opacities."""
    import yaml

    for session_name in ("may_short_grass", "june_tall_grass"):
        session = yaml.safe_load((REPOSITORY_ROOT / "configs" / "sessions" / f"{session_name}.yaml").read_text())
        assert "radar_seed_opacity" not in session["splatad"]


def test_job_environment_exports_opacity_only_when_set():
    """RADAR_SEED_OPACITY is absent when seed_opacity is None and present otherwise (real profile)."""
    import yaml

    submit_module = load_submit_module()
    slurm_config = submit_module.load_slurm_config(REPOSITORY_ROOT / "configs" / "slurm.yaml")
    profile_name = "may_scene_a_radar_init_loss"
    session_config = yaml.safe_load((REPOSITORY_ROOT / "configs" / "sessions" / "may_short_grass.yaml").read_text())
    exported = {}
    for label, session_extra in (("unset", {}), ("set", {"radar_seed_opacity": 0.3})):
        session_config["splatad"] = {**session_config["splatad"], **session_extra}
        options = submit_module.profile_radar_variant_options(slurm_config, session_config, profile_name)
        exported[label] = submit_module.build_job_environment(
            slurm_config["profiles"][profile_name], "may_short_grass", "run-id", "a" * 40, "/stage",
            {"env_prefix": "/env"}, radar_lidar_options=options)
    assert "RADAR_SEED_OPACITY" not in exported["unset"]
    assert float(exported["set"]["RADAR_SEED_OPACITY"]) == 0.3


def test_sbatch_passes_opacity_flag_only_when_set():
    """The sbatch no longer requires RADAR_SEED_OPACITY and adds the flag under a set-check."""
    text = (REPOSITORY_ROOT / "scripts" / "slurm" / "splatad_run.sbatch").read_text()
    assert "requires RADAR_SEED_OPACITY" not in text
    flag_line = next(line for line in text.splitlines() if "--pipeline.model.radar-seed-opacity" in line)
    flag_position = text.index(flag_line)
    guard = 'if [ -n "${RADAR_SEED_OPACITY:-}" ]; then'
    assert guard in text[:flag_position] and flag_position - text.rindex(guard, 0, flag_position) < 200


# ---------------------------------------------------------------- 3. matched-initialisation hash


@needs_image
def test_stock_gaussian_hash_covers_only_stock_rows(monkeypatch, tmp_path):
    """stock_gaussian_sha256 equals the hash of the stock rows; gaussian_sha256 still covers all rows."""
    import torch
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_init_lidar_model import (
        RadarInitLidarSplatADModel, RadarInitLidarSplatADModelConfig)
    from viking_splatad.splatad.radar_lidar_model import gaussian_parameter_fingerprint

    diagnostics_path = tmp_path / "diagnostics.jsonl"
    stock_parameters = _stock_factory()(None)
    monkeypatch.setattr(SplatADModel, "create_gauss_param_dict", lambda model_instance, *a, **k: stock_parameters)

    def stock_populate_modules(model_instance):
        """Stand-in for stock setup: only the Gaussian parameters."""
        model_instance.device_indicator_param = torch.nn.Parameter(torch.empty(0))
        model_instance.gauss_params = model_instance.create_gauss_param_dict([], [])

    monkeypatch.setattr(SplatADModel, "populate_modules", stock_populate_modules)
    model = RadarInitLidarSplatADModel.__new__(RadarInitLidarSplatADModel)
    torch.nn.Module.__init__(model)
    model.config = RadarInitLidarSplatADModelConfig(radar_lidar_diagnostics_path=diagnostics_path)
    model.kwargs = {"metadata": {"radar_seeds": _seeds()}}
    model.populate_modules()
    record = next(json.loads(line) for line in diagnostics_path.read_text().splitlines()
                  if json.loads(line)["event"] == "stock_initialization")
    assert record["stock_gaussian_sha256"] == gaussian_parameter_fingerprint(stock_parameters)
    assert record["gaussian_sha256"] == gaussian_parameter_fingerprint(model.gauss_params)
    assert record["gaussian_sha256"] != record["stock_gaussian_sha256"]


# ---------------------------------------------------------------- 4. contract robustness


def _built_artifact(tmp_path):
    """Build a real schema /1 artifact; return (root, settings, output directory)."""
    root, settings, config, calibration, options, _ = synthetic_artifact(tmp_path)
    output = tmp_path / "anchors"
    build_loss_only_artifact(root, config, settings, options, output, calibration)
    return root, settings, output


def test_corrupt_npz_with_matching_checksum_fails_the_report(tmp_path):
    """A truncated radar_seeds.npz whose recorded checksum matches gives a failed check, not BadZipFile."""
    root, settings, output = _built_artifact(tmp_path)
    # Truncating a real archive keeps the zip magic but loses the central directory (BadZipFile).
    intact_bytes = (output / "radar_seeds.npz").read_bytes()
    (output / "radar_seeds.npz").write_bytes(intact_bytes[: len(intact_bytes) // 2])
    metadata = json.loads((output / "radar_inputs.json").read_text())
    metadata["artifact_sha256"] = file_sha256(output / "radar_seeds.npz")
    (output / "radar_inputs.json").write_text(json.dumps(metadata))
    report = validate_radar_inputs(output, SESSION_NAME, root, settings, require_seed_colours=False)
    assert not report.passed


def test_list_top_level_metadata_fails_the_report(tmp_path):
    """radar_inputs.json holding a list gives a failed check, not AttributeError."""
    root, settings, output = _built_artifact(tmp_path)
    (output / "radar_inputs.json").write_text(json.dumps([1, 2, 3]))
    report = validate_radar_inputs(output, SESSION_NAME, root, settings, require_seed_colours=False)
    assert not report.passed
