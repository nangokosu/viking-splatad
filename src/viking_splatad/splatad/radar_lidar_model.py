"""Project-owned SplatAD model subclass for the optional radar LiDAR term."""

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Optional, Type


def gaussian_parameter_fingerprint(parameters) -> str:
    """Hash sorted Gaussian tensors, including dtype and shape, without consuming RNG state.

    Args:
        parameters: Mapping of parameter names to torch tensors.

    Returns:
        SHA256 of the exact initialized parameters for matched B/L audit records.
    """
    digest = hashlib.sha256()
    for name in sorted(parameters):
        array = parameters[name].detach().cpu().contiguous().numpy()
        digest.update(json.dumps([name, str(array.dtype), list(array.shape)]).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()

try:
    from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig

    from viking_splatad.splatad.radar_lidar_loss import calibrate_opacity_weight, compute_radar_lidar_loss

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
        radar_lidar_auto_weight: bool = False
        """Measure lambda at step zero; fail if initial support or stock regularizer is absent."""
        radar_lidar_diagnostics_path: Optional[Path] = None
        """Per-run JSONL evidence; contains counts, calibration and initial tensor hash."""
        radar_lidar_log_interval: int = 100
        """Coverage cadence specified by PLAN section 4.2."""

    class RadarLidarSplatADModel(SplatADModel):
        """Add radar-supported opacity supervision to the stock LiDAR loss."""

        config: RadarLidarSplatADModelConfig

        def populate_modules(self):
            """Run unchanged stock initialization, then add a persistent non-learned weight.

            No create_gauss_param_dict override, random sampling, Gaussian append
            or new optimizer group is introduced by this loss-only model.
            """
            import torch

            super().populate_modules()
            if self.config.radar_lidar_log_interval <= 0:
                raise ValueError("radar_lidar_log_interval must be positive")
            self.register_buffer("radar_calibrated_weight", torch.tensor(float("nan"), dtype=torch.float64))
            self._radar_initialization_hash = gaussian_parameter_fingerprint(self.gauss_params)
            self._radar_last_logged_step = None
            self._write_radar_diagnostic({"event": "stock_initialization", "step": 0,
                                          "gaussian_sha256": self._radar_initialization_hash,
                                          "gaussian_count": len(self.gauss_params["means"]),
                                          "radar_gaussians_added": 0})

        def _write_radar_diagnostic(self, record):
            """Append one JSON-serializable runtime record to the configured run-local path."""
            path = self.config.radar_lidar_diagnostics_path
            if path is not None:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as output:
                    output.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

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
            if (not self.training or "raster_pts" not in batch
                    or (self.config.radar_lidar_loss_weight == 0.0 and not self.config.radar_lidar_auto_weight)):
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
                # Always measure the unweighted term; weighting happens once below.
                loss_weight=1.0,
                gate_quantile=self.config.radar_lidar_gate_quantile,
                chunk_size=self.config.radar_lidar_gate_chunk_size,
            )
            weight = self.config.radar_lidar_loss_weight
            if self.config.radar_lidar_auto_weight:
                import torch

                if not torch.isfinite(self.radar_calibrated_weight).item():
                    if self.step != 0 or result.empty_support:
                        self._write_radar_diagnostic({"event": "calibration_failed", "step": int(self.step),
                                                      "empty_support": result.empty_support})
                        raise ValueError("automatic radar weight requires supported Gaussians at step zero")
                    weight = calibrate_opacity_weight(float(result.loss.detach()),
                                                      float(loss_dict["mcmc_opacity_reg"].detach()))
                    self.radar_calibrated_weight.fill_(weight)
                    self._write_radar_diagnostic({"event": "weight_calibration", "step": 0,
                                                  "unweighted_radar_loss": float(result.loss.detach()),
                                                  "stock_opacity_regularizer": float(loss_dict["mcmc_opacity_reg"].detach()),
                                                  "weight": weight})
                weight = float(self.radar_calibrated_weight.item())
            weighted_loss = result.loss * weight
            if metrics_dict is not None:
                metrics_dict["radar_lidar_loss"] = float(weighted_loss.detach())
                metrics_dict["radar_lidar_unweighted_loss"] = float(result.loss.detach())
                metrics_dict["radar_lidar_loss_weight"] = weight
                metrics_dict["radar_lidar_supported_gaussians"] = result.supported_gaussian_count
                metrics_dict["radar_lidar_supported_seed_fraction"] = result.supported_seed_fraction
                metrics_dict["radar_lidar_empty_support"] = int(result.empty_support)
            if self.step % self.config.radar_lidar_log_interval == 0 and self._radar_last_logged_step != self.step:
                self._write_radar_diagnostic({"event": "coverage", "step": int(self.step),
                                              "supported_gaussians": result.supported_gaussian_count,
                                              "supported_seed_fraction": result.supported_seed_fraction,
                                              "empty_support": result.empty_support, "weight": weight})
                self._radar_last_logged_step = self.step
            loss_dict["radar_lidar_loss"] = weighted_loss
            return loss_dict
