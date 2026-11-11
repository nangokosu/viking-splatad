"""Neurad-studio method registration for the radar LiDAR-loss extension."""

import copy
from dataclasses import fields


def extend_stock_model_config(stock_model_config, radar_config_type):
    """Copy every stock model field into the extension, preserving nested config values.

    Args:
        stock_model_config: Actual model instance from the pinned stock trainer.
        radar_config_type: Dataclass subclass adding radar-only settings/target.

    Returns:
        Independent extension config with every stock setting except _target unchanged.
    """
    # asdict would recursively turn nested config dataclasses into dictionaries.
    settings = {spec.name: copy.deepcopy(getattr(stock_model_config, spec.name))
                for spec in fields(stock_model_config) if spec.init and spec.name != "_target"}
    return radar_config_type(**settings)

try:
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.plugins.types import MethodSpecification

    from viking_splatad.splatad.radar_lidar_model import RadarLidarSplatADModelConfig

    RADAR_METHOD_AVAILABLE = True
except ImportError:
    RADAR_METHOD_AVAILABLE = False


if RADAR_METHOD_AVAILABLE:
    _stock_config = copy.deepcopy(method_configs["splatad"])
    _stock_config.method_name = "splatad-radar-lidar"
    _stock_config.pipeline.model = extend_stock_model_config(
        _stock_config.pipeline.model, RadarLidarSplatADModelConfig,
    )
    RADAR_LIDAR_METHOD_SPEC = MethodSpecification(
        config=_stock_config,
        description="SplatAD with an optional radar-supported LiDAR loss",
    )
else:
    RADAR_LIDAR_METHOD_SPEC = None
