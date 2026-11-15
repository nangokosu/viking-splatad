"""Project-owned SplatAD model subclass for the optional radar LiDAR term."""

from dataclasses import dataclass, field
import hashlib
import json
import os
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
    from viking_splatad.splatad.radar_support_index import RadarSupportIndex

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
            # Matched-initialisation audit: hash only the stock rows (appended seeds are the last
            # radar_gaussians_added rows of every parameter), comparable to arm B's hash.
            gaussian_count = len(self.gauss_params["means"])
            stock_row_count = gaussian_count - int(getattr(self, "_radar_gaussians_added", 0))
            stock_gaussian_hash = gaussian_parameter_fingerprint(
                {name: parameter[:stock_row_count] for name, parameter in self.gauss_params.items()})
            self._radar_last_logged_step = None
            self._radar_support_index = None
            self._write_radar_diagnostic({"event": "stock_initialization", "step": 0,
                                          "gaussian_sha256": self._radar_initialization_hash,
                                          "stock_gaussian_sha256": stock_gaussian_hash,
                                          "gaussian_count": gaussian_count,
                                          # Set by RadarInitSplatADModel when it is in the MRO; 0 otherwise.
                                          "radar_gaussians_added": int(getattr(self, "_radar_gaussians_added", 0))})

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
            is_lidar_batch = "raster_pts" in batch
            if (not self.training
                    or (self.config.radar_lidar_loss_weight == 0.0 and not self.config.radar_lidar_auto_weight)):
                return loss_dict
            # Auto-weight calibration is needed on any batch type while still uncalibrated.
            needs_calibration = False
            if self.config.radar_lidar_auto_weight:
                import torch

                needs_calibration = not torch.isfinite(self.radar_calibrated_weight).item()
            # A camera batch carries no radar term, so it only matters when it must calibrate.
            if not is_lidar_batch and not needs_calibration:
                return loss_dict
            metadata = self.kwargs.get("metadata", {})
            radar_seeds = metadata.get("radar_seeds")
            if radar_seeds is None:
                raise ValueError("radar_lidar_loss_weight is enabled but radar_seeds metadata is missing")
            if getattr(self, "_radar_support_index", None) is None:
                means = radar_seeds["means"].to(device=self.device, dtype=self.gauss_params["means"].dtype)
                covariances = radar_seeds["covariances"].to(device=self.device, dtype=means.dtype)
                self._radar_support_index = RadarSupportIndex(means, covariances,
                    self.config.radar_lidar_gate_quantile, self.config.radar_lidar_gate_chunk_size,
                    workers=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
            static_gaussian_mask = self.gauss_params["id"].reshape(-1) >= self.dynamic_actors.actor_sizes.shape[0]
            result = compute_radar_lidar_loss(
                gaussian_means=self.gauss_params["means"],
                opacity_logits=self.gauss_params["opacities"],
                static_gaussian_mask=static_gaussian_mask,
                radar_seed_means=self._radar_support_index.means,
                radar_seed_covariances=self._radar_support_index.covariances,
                # Always measure the unweighted term; weighting happens once below.
                loss_weight=1.0,
                gate_quantile=self.config.radar_lidar_gate_quantile,
                chunk_size=self.config.radar_lidar_gate_chunk_size,
                support_index=self._radar_support_index,
            )
            weight = self.config.radar_lidar_loss_weight
            if self.config.radar_lidar_auto_weight:
                if needs_calibration:
                    # Calibrate only at step zero and only with non-empty support.
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
            if not is_lidar_batch:
                # Calibration-only camera batch: the term is added on lidar batches only.
                return loss_dict
            weighted_loss = result.loss * weight
            if metrics_dict is not None:
                metrics_dict["radar_lidar_loss"] = float(weighted_loss.detach())
                metrics_dict["radar_lidar_unweighted_loss"] = float(result.loss.detach())
                metrics_dict["radar_lidar_loss_weight"] = weight
                metrics_dict["radar_lidar_supported_gaussians"] = result.supported_gaussian_count
                metrics_dict["radar_lidar_supported_seed_fraction"] = result.supported_seed_fraction
                metrics_dict["radar_lidar_empty_support"] = int(result.empty_support)
                metrics_dict["radar_lidar_candidate_pairs"] = self._radar_support_index.last_candidate_pairs
            if self.step % self.config.radar_lidar_log_interval == 0 and self._radar_last_logged_step != self.step:
                self._write_radar_diagnostic({"event": "coverage", "step": int(self.step),
                                              "supported_gaussians": result.supported_gaussian_count,
                                              "supported_seed_fraction": result.supported_seed_fraction,
                                              "empty_support": result.empty_support, "weight": weight})
                self._radar_last_logged_step = self.step
            loss_dict["radar_lidar_loss"] = weighted_loss
            return loss_dict
