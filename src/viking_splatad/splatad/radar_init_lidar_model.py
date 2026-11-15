"""SplatAD model combining radar-seed initialization with the radar opacity loss."""

from dataclasses import dataclass, field
from typing import Type

try:
    from viking_splatad.splatad.radar_init_model import RadarInitSplatADModel, RadarInitSplatADModelConfig
    from viking_splatad.splatad.radar_lidar_model import RadarLidarSplatADModel, RadarLidarSplatADModelConfig

    NERFSTUDIO_AVAILABLE = True
except ImportError:
    NERFSTUDIO_AVAILABLE = False


if NERFSTUDIO_AVAILABLE:

    @dataclass
    class RadarInitLidarSplatADModelConfig(RadarLidarSplatADModelConfig, RadarInitSplatADModelConfig):
        """Union of the loss config fields and the seed-opacity field; target is the combined model."""

        _target: Type = field(default_factory=lambda: RadarInitLidarSplatADModel)

    class RadarInitLidarSplatADModel(RadarLidarSplatADModel, RadarInitSplatADModel):
        """Append radar seed Gaussians, then apply the radar opacity loss to all static Gaussians.

        No new logic: the MRO is (this, RadarLidar, RadarInit, stock SplatAD). RadarLidar
        defines populate_modules and get_loss_dict, so they win and reach stock via super();
        it does not define create_gauss_param_dict, so lookup passes RadarLidar and resolves
        to RadarInit's override, which stock populate_modules calls to build the Gaussians
        (seeds included) before RadarLidar fingerprints them.
        """
