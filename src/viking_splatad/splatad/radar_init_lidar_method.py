"""Neurad-studio method registration for radar initialization plus the radar LiDAR loss."""

import copy

from viking_splatad.splatad.radar_lidar_method import extend_stock_model_config

try:
    from nerfstudio.configs.method_configs import method_configs
    from nerfstudio.plugins.types import MethodSpecification

    from viking_splatad.splatad.radar_init_lidar_model import RadarInitLidarSplatADModelConfig

    RADAR_INIT_LIDAR_METHOD_AVAILABLE = True
except ImportError:
    RADAR_INIT_LIDAR_METHOD_AVAILABLE = False


if RADAR_INIT_LIDAR_METHOD_AVAILABLE:
    # Start from a private copy of stock SplatAD so the registered stock method is untouched.
    _stock_config = copy.deepcopy(method_configs["splatad"])
    _stock_config.method_name = "splatad-radar-init-lidar"
    _stock_config.pipeline.model = extend_stock_model_config(
        _stock_config.pipeline.model, RadarInitLidarSplatADModelConfig,
    )
    RADAR_INIT_LIDAR_METHOD_SPEC = MethodSpecification(
        config=_stock_config,
        description="SplatAD with radar seed Gaussians and a radar-supported LiDAR opacity loss",
    )
else:
    RADAR_INIT_LIDAR_METHOD_SPEC = None
