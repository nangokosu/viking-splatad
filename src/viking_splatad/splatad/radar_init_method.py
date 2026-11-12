"""Nerfstudio registration for initialization-only radar SplatAD."""

import copy
from dataclasses import fields

try:
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.plugins.types import MethodSpecification

    from viking_splatad.splatad.radar_init_model import RadarInitSplatADModelConfig

    RADAR_INIT_METHOD_AVAILABLE = True
except ImportError:
    RADAR_INIT_METHOD_AVAILABLE = False


if RADAR_INIT_METHOD_AVAILABLE:
    _stock_config = copy.deepcopy(method_configs["splatad"])
    _stock_model = _stock_config.pipeline.model
    _model_arguments = {
        model_field.name: copy.deepcopy(getattr(_stock_model, model_field.name))
        for model_field in fields(_stock_model)
        if model_field.init and model_field.name != "_target"
    }
    _stock_config.method_name = "splatad-radar-init"
    _stock_config.pipeline.model = RadarInitSplatADModelConfig(**_model_arguments)
    RADAR_INIT_METHOD_SPEC = MethodSpecification(
        config=_stock_config,
        description="Stock SplatAD with train-only radar-guided Gaussian initialization",
    )
else:
    RADAR_INIT_METHOD_SPEC = None
