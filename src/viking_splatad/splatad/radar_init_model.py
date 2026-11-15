"""SplatAD model extension that appends radar-initialized static Gaussians."""

from dataclasses import dataclass, field
from typing import Dict, Optional, Type

import numpy as np
from scipy.spatial.transform import Rotation

from viking_splatad.registry import transform


@transform(
    "T41",
    "Positive-definite radar covariances -> SplatAD log-scales and scalar-first quaternions",
)
def covariance_to_log_scales_and_quaternions(
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Factor covariance matrices into SplatAD scale and rotation parameters.

    SplatAD represents ``Sigma = R diag(exp(log_scale)^2) R^T`` and stores
    quaternions scalar-first (w, x, y, z). Eigenvectors are made right-handed before
    conversion so reflections never enter the quaternion representation.

    Args:
        covariances: Symmetric positive-definite matrices ``[N,3,3]``.

    Returns:
        ``(log_scales [N,3], quaternions_wxyz [N,4])`` as float64 arrays.
    """
    covariance_array = np.asarray(covariances, dtype=np.float64)
    if covariance_array.ndim != 3 or covariance_array.shape[1:] != (3, 3):
        raise ValueError("covariances must have shape [N,3,3]")
    if not np.isfinite(covariance_array).all():
        raise ValueError("covariances must be finite")
    if not np.allclose(covariance_array, covariance_array.transpose(0, 2, 1)):
        raise ValueError("covariances must be symmetric")
    # Sigma = V diag(lambda) V^T, so R = V and exp(log_scale) = sqrt(lambda).
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_array)
    if np.any(eigenvalues <= 0.0):
        raise ValueError("covariances must be positive definite")
    # Negating one eigenvector column flips det(V) to +1 and leaves V diag(lambda) V^T unchanged.
    reflected = np.linalg.det(eigenvectors) < 0.0
    eigenvectors[reflected, :, 0] *= -1.0
    # SciPy returns scalar-last (x, y, z, w); reorder to SplatAD's (w, x, y, z).
    quaternions_xyzw = Rotation.from_matrix(eigenvectors).as_quat()
    quaternions_wxyz = quaternions_xyzw[:, [3, 0, 1, 2]]
    # log(sqrt(lambda)) = 0.5 * log(lambda).
    log_scales = 0.5 * np.log(eigenvalues)
    return log_scales, quaternions_wxyz


try:
    import torch
    from torch import nn

    from nerfstudio.models.splatad import SplatADModel, SplatADModelConfig

    NERFSTUDIO_AVAILABLE = True
except ImportError:
    NERFSTUDIO_AVAILABLE = False


if NERFSTUDIO_AVAILABLE:

    @dataclass
    class RadarInitSplatADModelConfig(SplatADModelConfig):
        """Stock SplatAD model configuration plus radar seed opacity."""

        _target: Type = field(default_factory=lambda: RadarInitSplatADModel)
        radar_seed_opacity: Optional[float] = None
        """Seed opacity probability override; None uses ``init_opacities`` as it is when seeds are appended."""

        def __post_init__(self) -> None:
            """Validate radar opacity after applying stock model validation."""
            parent_post_init = getattr(super(), "__post_init__", None)
            if parent_post_init is not None:
                parent_post_init()
            # Validate only an explicit override; None defers to stock's init_opacities.
            if self.radar_seed_opacity is not None and not 0.0 < self.radar_seed_opacity < 1.0:
                raise ValueError("radar_seed_opacity must be strictly between zero and one")

    class RadarInitSplatADModel(SplatADModel):
        """Append one static Gaussian for every validated train radar seed."""

        config: RadarInitSplatADModelConfig

        def create_gauss_param_dict(self, *args, **kwargs):
            """Run stock initialization, then append covariance-shaped radar seeds.

            Args:
                *args: Positional arguments forwarded unchanged to stock SplatAD.
                **kwargs: Keyword arguments forwarded unchanged to stock SplatAD.

            Returns:
                Stock ``ParameterDict`` when no seeds exist, otherwise the same keys with
                one radar row appended per seed. No optimizer group or renderer is added.
            """
            stock_parameters = super().create_gauss_param_dict(*args, **kwargs)
            radar_seeds = self.kwargs.get("metadata", {}).get("radar_seeds")
            if radar_seeds is None or radar_seeds["means"].shape[0] == 0:
                self._radar_gaussians_added = 0
                return stock_parameters
            # Seeds initialize Gaussian colour, so unpainted (rgb-less) anchors are unusable here.
            if "rgb" not in radar_seeds:
                raise ValueError("radar_seeds metadata has no rgb; radar initialization needs painted seed colours")
            # Seed rows take the device/dtype of the stock parameters they are concatenated onto.
            reference_parameter = stock_parameters["means"]
            device = reference_parameter.device
            floating_dtype = reference_parameter.dtype
            means = radar_seeds["means"].to(device=device, dtype=floating_dtype)
            numpy_covariances = radar_seeds["covariances"].detach().cpu().numpy()
            log_scales_numpy, quaternions_numpy = covariance_to_log_scales_and_quaternions(
                numpy_covariances
            )
            seed_count = means.shape[0]
            # Read by RadarLidarSplatADModel diagnostics.
            self._radar_gaussians_added = int(seed_count)
            # Opacity probability: the override, else the value stock used for its own Gaussians. Stock
            # populate_modules reassigns init_opacities (MCMC) only after create_gauss_param_dict, so
            # reading it here gives the same value stock's rows were built with.
            seed_opacity_probability = (self.config.radar_seed_opacity
                                        if self.config.radar_seed_opacity is not None
                                        else self.config.init_opacities)
            # Per-seed rows: covariance-shaped scales/rotation, colour rgb / dtype max (stock's rgb/255),
            # random appearance features drawn like stock (after stock's own draws, so stock rows are
            # unchanged), and logit(seed opacity) as the opacity logit.
            radar_values: Dict[str, "torch.Tensor"] = {
                "means": means,
                "scales": torch.as_tensor(log_scales_numpy, device=device, dtype=floating_dtype),
                "quats": torch.as_tensor(quaternions_numpy, device=device, dtype=floating_dtype),
                "features_dc": radar_seeds["rgb"].to(
                    device=device, dtype=floating_dtype
                ) / torch.iinfo(radar_seeds["rgb"].dtype).max,
                "features_rest": torch.randn(
                    (seed_count, *stock_parameters["features_rest"].shape[1:]),
                    device=device,
                    dtype=stock_parameters["features_rest"].dtype,
                ),
                "opacities": torch.full(
                    (seed_count, *stock_parameters["opacities"].shape[1:]),
                    torch.logit(torch.tensor(seed_opacity_probability)).item(),
                    device=device,
                    dtype=stock_parameters["opacities"].dtype,
                ),
            }
            # Stock assigns every static Gaussian the actor-count id. Its maximum is that
            # static id because dynamic actors occupy the preceding non-negative ids.
            radar_values["id"] = torch.full(
                (seed_count, *stock_parameters["id"].shape[1:]),
                stock_parameters["id"].detach().max().item(),
                device=device,
                dtype=stock_parameters["id"].dtype,
            )
            # Guard against an upstream SplatAD change adding or renaming a Gaussian parameter.
            if set(radar_values) != set(stock_parameters):
                raise RuntimeError(
                    "pinned SplatAD Gaussian parameter keys changed; radar initialization needs review"
                )
            return nn.ParameterDict({
                parameter_name: nn.Parameter(
                    torch.cat([stock_parameter, radar_values[parameter_name]], dim=0),
                    requires_grad=stock_parameter.requires_grad,
                )
                for parameter_name, stock_parameter in stock_parameters.items()
            })
