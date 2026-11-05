"""Neurad-studio method registration for the radar LiDAR-loss extension."""

import copy

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
    _stock_config.pipeline.model = RadarLidarSplatADModelConfig(max_steps=_stock_config.max_num_iterations)
    RADAR_LIDAR_METHOD_SPEC = MethodSpecification(
        config=_stock_config,
        description="SplatAD with an optional radar-supported LiDAR loss",
    )
else:
    RADAR_LIDAR_METHOD_SPEC = None
