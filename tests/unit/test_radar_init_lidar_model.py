"""Combined radar initialization + radar opacity loss model (W1-C): MRO, config, seeds, loss, calibration."""

from dataclasses import fields
import json
import math
from types import SimpleNamespace

import pytest

from viking_splatad.dataset.schema import values_equal


def _neurad_ready() -> bool:
    """Return whether the pinned torch/neurad stack is importable (true in the reference image)."""
    try:
        from nerfstudio.cameras.lidars import LidarType
    except ImportError:
        return False
    return hasattr(LidarType, "OUSTER_OS0_32")


pytestmark = pytest.mark.skipif(not _neurad_ready(), reason="needs the reference image")

# Synthetic scene. Stock Gaussians sit far from every anchor except one "near" stock row.
SEED_OPACITY_PROBABILITY = 0.25
ANCHOR_VARIANCE = 0.01  # isotropic anchor covariance [m^2]
ANCHOR_MEANS = [[1.0, 2.0, 3.0], [-4.0, 0.5, 2.0]]
# Stock rows: the first two are far from all anchors (unsupported); the third lies one-tenth of a
# standard deviation from anchor 0 (Mahalanobis^2 = 0.01), well inside the chi-square gate.
FAR_STOCK_OFFSET = 100.0
NEAR_STOCK_OFFSET_FRACTION_OF_SIGMA = 0.1
STOCK_OPACITY_LOGITS = [0.3, -0.7, 1.1]
FIXED_LOSS_WEIGHT = 0.7
STOCK_REGULARIZER = 0.0025  # synthetic stock mcmc_opacity_reg, distinct from the radar term
FEATURES_REST_WIDTH = 5  # arbitrary spherical-harmonics tail width; only shapes matter here
FEATURES_DC_WIDTH = 3  # RGB


def _anchor_tensors():
    """Return (means [S,3], covariances [S,3,3]) anchors as float32 tensors."""
    import torch

    means = torch.tensor(ANCHOR_MEANS)
    covariances = (torch.eye(3) * ANCHOR_VARIANCE).repeat(means.shape[0], 1, 1)
    return means, covariances


def _seed_metadata(with_rgb=True):
    """Return a radar_seeds metadata dict for the synthetic anchors (rgb optional)."""
    import torch

    means, covariances = _anchor_tensors()
    seeds = {"means": means, "covariances": covariances}
    if with_rgb:
        seeds["rgb"] = torch.tensor([[255, 128, 0], [0, 64, 255]], dtype=torch.uint8)[: means.shape[0]]
    return seeds


def _stock_parameters():
    """Return a stock-like ParameterDict: 2 far rows and 1 row near anchor 0, all static (id 0)."""
    import torch

    sigma = math.sqrt(ANCHOR_VARIANCE)
    near_row = [ANCHOR_MEANS[0][0] + NEAR_STOCK_OFFSET_FRACTION_OF_SIGMA * sigma,
                ANCHOR_MEANS[0][1], ANCHOR_MEANS[0][2]]
    means = torch.tensor([[FAR_STOCK_OFFSET] * 3, [-FAR_STOCK_OFFSET] * 3, near_row])
    row_count = means.shape[0]
    return torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(torch.zeros((row_count, 3))),
        "quats": torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(row_count, 1)),
        "features_dc": torch.nn.Parameter(torch.zeros((row_count, FEATURES_DC_WIDTH))),
        "features_rest": torch.nn.Parameter(torch.zeros((row_count, FEATURES_REST_WIDTH))),
        "opacities": torch.nn.Parameter(torch.tensor(STOCK_OPACITY_LOGITS)[:, None].clone()),
        "id": torch.nn.Parameter(torch.zeros((row_count, 1)), requires_grad=False),
    })


def _combined_model(monkeypatch, config_kwargs=None, seeds="default"):
    """Build a combined model via object.__new__ with stock create_gauss_param_dict patched.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        config_kwargs: Extra RadarInitLidarSplatADModelConfig fields.
        seeds: radar_seeds metadata dict, None for no seeds, or "default" for the synthetic anchors.

    Returns:
        (model, stock_parameters) with gauss_params already created by the combined model.
    """
    import torch
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_init_lidar_model import (
        RadarInitLidarSplatADModel, RadarInitLidarSplatADModelConfig)

    stock_parameters = _stock_parameters()
    monkeypatch.setattr(SplatADModel, "create_gauss_param_dict",
                        lambda model_instance, *arguments, **keyword_arguments: stock_parameters)
    model = RadarInitLidarSplatADModel.__new__(RadarInitLidarSplatADModel)
    torch.nn.Module.__init__(model)
    model.config = RadarInitLidarSplatADModelConfig(
        radar_seed_opacity=SEED_OPACITY_PROBABILITY, **(config_kwargs or {}))
    model.device_indicator_param = torch.nn.Parameter(torch.empty(0))
    model.dynamic_actors = SimpleNamespace(actor_sizes=torch.zeros((0, 3)))
    radar_seeds = _seed_metadata() if isinstance(seeds, str) else seeds
    model.kwargs = {"metadata": {} if radar_seeds is None else {"radar_seeds": radar_seeds}}
    model.register_buffer("radar_calibrated_weight", torch.tensor(float("nan"), dtype=torch.float64))
    model._radar_last_logged_step = None
    model.gauss_params = model.create_gauss_param_dict([], [])
    model.step = 0
    return model, stock_parameters


def _static_logits_of_supported_rows(model):
    """Hand-pick the supported rows: the 2 appended seeds and the near stock row (index 2)."""
    import torch

    logits = model.gauss_params["opacities"].detach().reshape(-1)
    stock_row_count = len(STOCK_OPACITY_LOGITS)
    supported = [stock_row_count - 1] + list(range(stock_row_count, len(logits)))
    return logits[torch.tensor(supported)]


def _unweighted_radar_loss_by_hand(model):
    """Return mean(-log sigmoid(logit)) over the supported static rows (T43 formula), computed in numpy."""
    import numpy as np

    logits = _static_logits_of_supported_rows(model).double().numpy()
    return float(np.mean(-np.log(1.0 / (1.0 + np.exp(-logits)))))


# ---------------------------------------------------------------- 1. MRO


def test_mro_orders_radar_lidar_before_radar_init_before_stock_and_resolves_methods():
    """Method resolution gives the loss/diagnostics to RadarLidar and seed appending to RadarInit."""
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_init_lidar_model import RadarInitLidarSplatADModel
    from viking_splatad.splatad.radar_init_model import RadarInitSplatADModel
    from viking_splatad.splatad.radar_lidar_model import RadarLidarSplatADModel

    mro = RadarInitLidarSplatADModel.__mro__
    assert mro.index(RadarLidarSplatADModel) < mro.index(RadarInitSplatADModel) < mro.index(SplatADModel)
    assert RadarInitLidarSplatADModel.create_gauss_param_dict is RadarInitSplatADModel.create_gauss_param_dict
    assert RadarInitLidarSplatADModel.get_loss_dict is RadarLidarSplatADModel.get_loss_dict
    assert RadarInitLidarSplatADModel.populate_modules is RadarLidarSplatADModel.populate_modules


# ---------------------------------------------------------------- 2. configuration


def test_combined_config_has_both_field_sets_with_parent_defaults_and_target():
    """Every parent field exists with the parent's default and the target is the combined model."""
    from viking_splatad.splatad.radar_init_lidar_model import (
        RadarInitLidarSplatADModel, RadarInitLidarSplatADModelConfig)
    from viking_splatad.splatad.radar_init_model import RadarInitSplatADModelConfig
    from viking_splatad.splatad.radar_lidar_model import RadarLidarSplatADModelConfig

    combined = RadarInitLidarSplatADModelConfig()
    init_defaults = RadarInitSplatADModelConfig()
    lidar_defaults = RadarLidarSplatADModelConfig()
    assert combined._target is RadarInitLidarSplatADModel
    assert combined.radar_seed_opacity == init_defaults.radar_seed_opacity
    lidar_field_names = [spec.name for spec in fields(lidar_defaults) if spec.name.startswith("radar_lidar_")]
    assert lidar_field_names, "the loss config must expose radar_lidar_* fields"
    for field_name in lidar_field_names:
        assert getattr(combined, field_name) == getattr(lidar_defaults, field_name), field_name


def test_method_spec_uses_combined_config_and_keeps_stock_fields():
    """The registered method is named correctly, targets the combined config and keeps stock settings."""
    from nerfstudio.configs.method_configs import method_configs
    from viking_splatad.splatad.radar_init_lidar_method import RADAR_INIT_LIDAR_METHOD_SPEC
    from viking_splatad.splatad.radar_init_lidar_model import RadarInitLidarSplatADModelConfig

    assert RADAR_INIT_LIDAR_METHOD_SPEC.config.method_name == "splatad-radar-init-lidar"
    extended = RADAR_INIT_LIDAR_METHOD_SPEC.config.pipeline.model
    assert isinstance(extended, RadarInitLidarSplatADModelConfig)
    original = method_configs["splatad"].pipeline.model
    for specification in fields(original):
        if specification.name != "_target":
            assert values_equal(getattr(original, specification.name), getattr(extended, specification.name)), \
                specification.name


# ---------------------------------------------------------------- 3. seeds appended and counted


def test_seeds_are_appended_after_stock_rows_and_counted(monkeypatch):
    """N stock rows become N+S rows; the model records S as _radar_gaussians_added."""
    model, stock_parameters = _combined_model(monkeypatch)
    stock_row_count = len(STOCK_OPACITY_LOGITS)
    seed_count = len(ANCHOR_MEANS)
    assert model.gauss_params["means"].shape[0] == stock_row_count + seed_count
    assert model._radar_gaussians_added == seed_count
    import torch

    torch.testing.assert_close(model.gauss_params["means"][stock_row_count:].detach(),
                               torch.tensor(ANCHOR_MEANS))


def test_no_seeds_leaves_stock_rows_and_counts_zero(monkeypatch):
    """Without radar_seeds metadata the stock rows pass through and the count is zero."""
    model, _ = _combined_model(monkeypatch, seeds=None)
    assert model.gauss_params["means"].shape[0] == len(STOCK_OPACITY_LOGITS)
    assert model._radar_gaussians_added == 0


# ---------------------------------------------------------------- 4. seed rows are supported


def test_loss_includes_appended_seed_rows_and_matches_hand_value(monkeypatch):
    """Seed Gaussians sit on the anchors, so they are supported and enter the mean softplus(-logit)."""
    import torch

    model, _ = _combined_model(monkeypatch, {"radar_lidar_loss_weight": FIXED_LOSS_WEIGHT})
    model.step = 1
    from nerfstudio.models.splatad import SplatADModel

    monkeypatch.setattr(SplatADModel, "get_loss_dict",
                        lambda *arguments: {"mcmc_opacity_reg": torch.tensor(STOCK_REGULARIZER)})
    metrics = {}
    losses = model.get_loss_dict({}, {"raster_pts": torch.zeros(1)}, metrics)
    supported_count = len(_static_logits_of_supported_rows(model))
    assert supported_count == 1 + len(ANCHOR_MEANS)
    expected_unweighted = _unweighted_radar_loss_by_hand(model)
    assert metrics["radar_lidar_supported_gaussians"] == supported_count
    assert losses["radar_lidar_loss"].item() == pytest.approx(FIXED_LOSS_WEIGHT * expected_unweighted, rel=1e-6)
    # Seed logit is logit(p) = log(p / (1 - p)); -log sigmoid(logit) = -log p.
    seed_logit = math.log(SEED_OPACITY_PROBABILITY / (1.0 - SEED_OPACITY_PROBABILITY))
    assert model.gauss_params["opacities"][-1].item() == pytest.approx(seed_logit, rel=1e-6)


# ---------------------------------------------------------------- 5. step-0 camera calibration


def _patch_stock_loss(monkeypatch):
    """Make stock get_loss_dict return only a synthetic mcmc_opacity_reg."""
    import torch
    from nerfstudio.models.splatad import SplatADModel

    monkeypatch.setattr(SplatADModel, "get_loss_dict",
                        lambda *arguments: {"mcmc_opacity_reg": torch.tensor(STOCK_REGULARIZER)})


def test_step_zero_camera_batch_calibrates_without_adding_the_radar_term(monkeypatch):
    """Calibration runs on a camera batch; only the following lidar batch carries the radar term."""
    import torch

    model, _ = _combined_model(monkeypatch, {"radar_lidar_auto_weight": True})
    _patch_stock_loss(monkeypatch)
    unweighted = _unweighted_radar_loss_by_hand(model)
    camera_losses = model.get_loss_dict({}, {}, {})
    assert "radar_lidar_loss" not in camera_losses
    assert torch.isfinite(model.radar_calibrated_weight).item()
    expected_weight = STOCK_REGULARIZER / unweighted
    assert model.radar_calibrated_weight.item() == pytest.approx(expected_weight, rel=1e-6)
    model.step = 1
    lidar_losses = model.get_loss_dict({}, {"raster_pts": torch.zeros(1)})
    # weight * unweighted == stock regularizer by construction of the calibration.
    assert lidar_losses["radar_lidar_loss"].item() == pytest.approx(STOCK_REGULARIZER, rel=1e-6)
    assert model.radar_calibrated_weight.item() == pytest.approx(expected_weight, rel=1e-6)


def test_step_zero_lidar_batch_still_calibrates_and_adds_the_term(monkeypatch):
    """A lidar batch at step 0 calibrates and adds the weighted term in the same call."""
    import torch

    model, _ = _combined_model(monkeypatch, {"radar_lidar_auto_weight": True})
    _patch_stock_loss(monkeypatch)
    losses = model.get_loss_dict({}, {"raster_pts": torch.zeros(1)})
    assert losses["radar_lidar_loss"].item() == pytest.approx(STOCK_REGULARIZER, rel=1e-6)
    assert torch.isfinite(model.radar_calibrated_weight).item()


# ---------------------------------------------------------------- 6. uncalibrated at step > 0


@pytest.mark.parametrize("batch_has_lidar", [True, False], ids=["lidar_batch", "camera_batch"])
def test_uncalibrated_auto_weight_after_step_zero_raises(monkeypatch, batch_has_lidar):
    """Missing the step-0 calibration is an error, never a silent fallback weight."""
    import torch

    model, _ = _combined_model(monkeypatch, {"radar_lidar_auto_weight": True})
    _patch_stock_loss(monkeypatch)
    model.step = 1
    batch = {"raster_pts": torch.zeros(1)} if batch_has_lidar else {}
    with pytest.raises(ValueError):
        model.get_loss_dict({}, batch)


# ---------------------------------------------------------------- 7. diagnostics


def _stock_initialization_record(diagnostics_path):
    """Return the stock_initialization JSONL record from a diagnostics file."""
    records = [json.loads(line) for line in diagnostics_path.read_text().splitlines()]
    matching = [record for record in records if record["event"] == "stock_initialization"]
    assert len(matching) == 1
    return matching[0]


def _populate_with_patched_stock(model, monkeypatch):
    """Run the model's populate_modules with stock populate_modules reduced to creating gauss_params."""
    import torch
    from nerfstudio.models.splatad import SplatADModel

    def stock_populate_modules(model_instance):
        """Stand-in for stock setup: only the Gaussian parameters and device indicator."""
        model_instance.device_indicator_param = torch.nn.Parameter(torch.empty(0))
        model_instance.gauss_params = model_instance.create_gauss_param_dict([], [])

    monkeypatch.setattr(SplatADModel, "populate_modules", stock_populate_modules)
    model.populate_modules()


def test_combined_model_diagnostics_report_the_appended_seed_count(monkeypatch, tmp_path):
    """stock_initialization records radar_gaussians_added == S and the enlarged Gaussian count."""
    import torch
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_init_lidar_model import (
        RadarInitLidarSplatADModel, RadarInitLidarSplatADModelConfig)

    diagnostics_path = tmp_path / "diagnostics.jsonl"
    stock_parameters = _stock_parameters()
    monkeypatch.setattr(SplatADModel, "create_gauss_param_dict",
                        lambda model_instance, *arguments, **keyword_arguments: stock_parameters)
    model = RadarInitLidarSplatADModel.__new__(RadarInitLidarSplatADModel)
    torch.nn.Module.__init__(model)
    model.config = RadarInitLidarSplatADModelConfig(
        radar_seed_opacity=SEED_OPACITY_PROBABILITY, radar_lidar_diagnostics_path=diagnostics_path)
    model.kwargs = {"metadata": {"radar_seeds": _seed_metadata()}}
    _populate_with_patched_stock(model, monkeypatch)
    record = _stock_initialization_record(diagnostics_path)
    assert record["radar_gaussians_added"] == len(ANCHOR_MEANS)
    assert record["gaussian_count"] == len(STOCK_OPACITY_LOGITS) + len(ANCHOR_MEANS)


def test_loss_only_model_diagnostics_still_report_zero_added(monkeypatch, tmp_path):
    """The loss-only model appends nothing, so its record keeps radar_gaussians_added == 0."""
    import torch
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_lidar_model import RadarLidarSplatADModel, RadarLidarSplatADModelConfig

    diagnostics_path = tmp_path / "diagnostics.jsonl"
    stock_parameters = _stock_parameters()
    monkeypatch.setattr(SplatADModel, "create_gauss_param_dict",
                        lambda model_instance, *arguments, **keyword_arguments: stock_parameters)
    model = RadarLidarSplatADModel.__new__(RadarLidarSplatADModel)
    torch.nn.Module.__init__(model)
    model.config = RadarLidarSplatADModelConfig(radar_lidar_diagnostics_path=diagnostics_path)
    model.kwargs = {"metadata": {"radar_seeds": _seed_metadata(with_rgb=False)}}
    _populate_with_patched_stock(model, monkeypatch)
    record = _stock_initialization_record(diagnostics_path)
    assert record["radar_gaussians_added"] == 0
    assert record["gaussian_count"] == len(STOCK_OPACITY_LOGITS)


# ---------------------------------------------------------------- 8. rgb required


def test_radar_init_model_requires_rgb_in_seeds(monkeypatch):
    """Seeds without painted colours cannot initialize Gaussians; the error names rgb."""
    import torch
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_init_model import RadarInitSplatADModel

    stock_parameters = _stock_parameters()
    monkeypatch.setattr(SplatADModel, "create_gauss_param_dict",
                        lambda model_instance, *arguments, **keyword_arguments: stock_parameters)
    model = object.__new__(RadarInitSplatADModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(radar_seed_opacity=SEED_OPACITY_PROBABILITY)
    model.kwargs = {"metadata": {"radar_seeds": _seed_metadata(with_rgb=False)}}
    with pytest.raises(ValueError, match="rgb"):
        model.create_gauss_param_dict([], [])
