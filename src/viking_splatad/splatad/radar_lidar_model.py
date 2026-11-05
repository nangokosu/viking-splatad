"""Project-owned SplatAD model subclass for the optional radar LiDAR term."""

from dataclasses import dataclass, field
from typing import Type

try:
    from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig

    from viking_splatad.splatad.radar_lidar_loss import compute_radar_lidar_loss

    NERFSTUDIO_AVAILABLE = True
except ImportError:
    NERFSTUDIO_AVAILABLE = False


if NERFSTUDIO_AVAILABLE:

    @dataclass
    class RadarLidarSplatADModelConfig(SplatADModelConfig):
        """Configuration for SplatAD with an optional radar-supported LiDAR term."""

        _target: Type = field(default_factory=lambda: RadarLidarSplatADModel)
        radar_lidar_loss_weight: float = 0.0
        radar_lidar_gate_quantile: float = 0.95
        radar_lidar_gate_chunk_size: int = 4096

    class RadarLidarSplatADModel(SplatADModel):
        """Add radar-supported opacity supervision to the stock LiDAR loss."""

        config: RadarLidarSplatADModelConfig

        def get_loss_dict(self, outputs, batch, metrics_dict=None):
            """Return stock losses plus the optional radar-supported LiDAR term.

            Args:
                outputs: SplatAD outputs for the current training batch.
                batch: Camera or LiDAR training batch.
                metrics_dict: Optional metrics dictionary from the training step.

            Returns:
                Loss dictionary consumed by the stock trainer.
            """
            loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
            if "raster_pts" not in batch or self.config.radar_lidar_loss_weight == 0.0:
                return loss_dict
            metadata = self.kwargs.get("metadata", {})
            radar_seeds = metadata.get("radar_seeds")
            if radar_seeds is None:
                raise ValueError("radar_lidar_loss_weight is enabled but radar_seeds metadata is missing")
            static_gaussian_mask = self.gauss_params["id"].reshape(-1) >= self.dynamic_actors.actor_sizes.shape[0]
            result = compute_radar_lidar_loss(
                gaussian_means=self.gauss_params["means"],
                opacity_logits=self.gauss_params["opacities"],
                static_gaussian_mask=static_gaussian_mask,
                radar_seed_means=radar_seeds["means"].to(self.device),
                radar_seed_covariances=radar_seeds["covariances"].to(self.device),
                loss_weight=self.config.radar_lidar_loss_weight,
                gate_quantile=self.config.radar_lidar_gate_quantile,
                chunk_size=self.config.radar_lidar_gate_chunk_size,
            )
            if metrics_dict is not None:
                metrics_dict["radar_lidar_loss"] = float(result.loss.detach())
                metrics_dict["radar_lidar_supported_gaussians"] = result.supported_gaussian_count
                metrics_dict["radar_lidar_supported_seed_fraction"] = result.supported_seed_fraction
            loss_dict["radar_lidar_loss"] = result.loss
            return loss_dict
