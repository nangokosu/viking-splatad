"""Loss-only configuration equality, analytic gradients and non-random evidence."""

from dataclasses import dataclass, field, fields
import math

import numpy as np
import pytest

from viking_splatad.dataset.schema import values_equal
from viking_splatad.splatad.radar_lidar_method import extend_stock_model_config
from viking_splatad.splatad.radar_lidar_loss import calibrate_opacity_weight


def test_config_extension_preserves_every_stock_field_without_aliasing():
    """Dataclass and nested overrides survive extension instead of reverting to defaults."""
    @dataclass
    class Stock:
        """Synthetic stock model with a deliberately non-default nested override."""
        _target: object = "stock"
        schedule: int = 10
        nested: dict = field(default_factory=lambda: {"value": [1]})

    @dataclass
    class Extended(Stock):
        """Synthetic radar config whose target changes while stock values remain intact."""
        _target: object = "radar"
        weight: float = 0.0

    original = Stock(schedule=37, nested={"value": [7, 9]})
    extended = extend_stock_model_config(original, Extended)
    assert extended.schedule == 37 and extended.nested == original.nested
    extended.nested["value"].append(11)
    assert original.nested == {"value": [7, 9]}
    assert extended._target == "radar"


def test_real_radar_model_preserves_stock_configuration_and_initializer():
    """The actual plugin carries stock fields unchanged and inherits the stock creation hook."""
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_lidar_method import RADAR_LIDAR_METHOD_SPEC
    from viking_splatad.splatad.radar_lidar_model import RadarLidarSplatADModel

    original = method_configs["splatad"].pipeline.model
    extended = RADAR_LIDAR_METHOD_SPEC.config.pipeline.model
    for specification in fields(original):
        if specification.name != "_target":
            assert values_equal(getattr(original, specification.name), getattr(extended, specification.name))
    assert RadarLidarSplatADModel.create_gauss_param_dict is SplatADModel.create_gauss_param_dict


def test_weight_calibration_matches_actual_stock_regularizer():
    """Initial weighted radar term equals the stock weighted opacity regularizer."""
    initial_penalty, regularizer = math.log(2), 0.005 / 2
    weight = calibrate_opacity_weight(initial_penalty, regularizer)
    assert weight * initial_penalty == pytest.approx(regularizer)
    for invalid in (0., -1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            calibrate_opacity_weight(invalid, regularizer)


@pytest.mark.covers("T42")
def test_union_gate_chunks_both_axes_and_includes_the_boundary():
    """Known chi-square boundary is included; overlapping anchors never multiply losses."""
    import torch
    from scipy.stats import chi2
    from viking_splatad.splatad.radar_lidar_loss import radar_support_gate

    quantile = 0.95
    boundary = math.sqrt(chi2.ppf(quantile, df=3))
    means = torch.tensor([[0., 0., 0.], [boundary, 0., 0.], [boundary + 0.1, 0., 0.]], dtype=torch.float64)
    anchors = torch.zeros((3, 3), dtype=torch.float64)
    covariance = torch.eye(3, dtype=torch.float64).repeat(3, 1, 1)
    expected = [True, True, False]
    for chunk_size in (1, 2, 4):
        supported, seed_supported = radar_support_gate(means, anchors, covariance, quantile, chunk_size)
        assert supported.tolist() == expected
        assert seed_supported.tolist() == [True, True, True]


@pytest.mark.covers("T43")
def test_opacity_gradient_is_analytic_and_has_no_position_attraction():
    """Negative log sigmoid changes opacity only, counted once across overlapping anchors."""
    import torch
    from viking_splatad.splatad.radar_lidar_loss import compute_radar_lidar_loss

    means = torch.zeros((2, 3), dtype=torch.float64, requires_grad=True)
    logits = torch.tensor([0., math.log(3)], dtype=torch.float64, requires_grad=True)
    result = compute_radar_lidar_loss(means, logits, torch.tensor([True, True]),
                                     torch.zeros((2, 3), dtype=torch.float64),
                                     torch.eye(3, dtype=torch.float64).repeat(2, 1, 1), 2., 0.95, 1)
    expected = 2 * (-math.log(0.5) - math.log(0.75)) / 2
    assert result.loss.item() == pytest.approx(expected)
    result.loss.backward()
    np.testing.assert_allclose(logits.grad.numpy(), -2 * np.array([1 - 0.5, 1 - 0.75]) / 2)
    assert means.grad is None


def test_initialization_fingerprint_does_not_consume_rng():
    """Hashing captures tensor changes without altering stochastic stock initialization."""
    import torch
    from viking_splatad.splatad.radar_lidar_model import gaussian_parameter_fingerprint

    parameters = {"means": torch.zeros((2, 3)), "opacities": torch.zeros((2, 1))}
    rng_before = torch.random.get_rng_state().clone()
    fingerprint = gaussian_parameter_fingerprint(parameters)
    assert torch.equal(rng_before, torch.random.get_rng_state())
    assert gaussian_parameter_fingerprint(dict(reversed(list(parameters.items())))) == fingerprint
    parameters["means"][0, 0] = 1
    assert gaussian_parameter_fingerprint(parameters) != fingerprint


def test_model_calibrates_once_and_keeps_weight_in_checkpoint(monkeypatch):
    """A synthetic stock LiDAR loss calibrates once, persists lambda and leaves geometry intact."""
    import torch
    from types import SimpleNamespace
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_lidar_model import RadarLidarSplatADModel, RadarLidarSplatADModelConfig

    model = RadarLidarSplatADModel.__new__(RadarLidarSplatADModel)
    torch.nn.Module.__init__(model)
    model.config = RadarLidarSplatADModelConfig(radar_lidar_auto_weight=True)
    model.device_indicator_param = torch.nn.Parameter(torch.empty(0))
    model.gauss_params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.zeros((1, 3))),
        "opacities": torch.nn.Parameter(torch.zeros((1, 1))),
        "id": torch.nn.Parameter(torch.zeros((1, 1)), requires_grad=False),
    })
    model.dynamic_actors = SimpleNamespace(actor_sizes=torch.zeros((0, 3)))
    model.kwargs = {"metadata": {"radar_seeds": {"means": torch.zeros((1, 3)),
                                               "covariances": torch.eye(3)[None]}}}
    model.register_buffer("radar_calibrated_weight", torch.tensor(float("nan"), dtype=torch.float64))
    model._radar_last_logged_step = None
    model.step = 0
    regularizer = 0.0025  # Synthetic stock loss, deliberately distinct from the unweighted radar term.
    monkeypatch.setattr(SplatADModel, "get_loss_dict", lambda *arguments: {"mcmc_opacity_reg": torch.tensor(regularizer)})
    metrics = {}
    losses = model.get_loss_dict({}, {"raster_pts": torch.zeros(1)}, metrics)
    assert losses["radar_lidar_loss"].item() == pytest.approx(regularizer)
    measured_weight = model.radar_calibrated_weight.item()
    assert measured_weight == pytest.approx(regularizer / math.log(2))
    assert model.state_dict()["radar_calibrated_weight"].item() == measured_weight
    assert metrics["radar_lidar_supported_gaussians"] == 1
    model.step = 1
    model.get_loss_dict({}, {"raster_pts": torch.zeros(1)})
    assert model.radar_calibrated_weight.item() == measured_weight
    losses["radar_lidar_loss"].backward()
    assert model.gauss_params["opacities"].grad.item() < 0
    assert model.gauss_params["means"].grad is None
    assert len(model.gauss_params["means"]) == 1
