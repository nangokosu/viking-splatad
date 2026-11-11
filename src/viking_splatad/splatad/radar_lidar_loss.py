"""Radar-supported occupancy term added to the existing LiDAR training loss."""

from dataclasses import dataclass
import math

from viking_splatad.registry import transform


@dataclass(frozen=True)
class RadarLidarLossResult:
    """Computed radar-supported LiDAR loss and diagnostics."""

    loss: object
    supported_gaussian_count: int
    supported_seed_fraction: float
    empty_support: bool


@transform("T43", "Mean negative log opacity of static Gaussians supported by fixed radar anchors")
def compute_radar_lidar_loss(
    gaussian_means: object,
    opacity_logits: object,
    static_gaussian_mask: object,
    radar_seed_means: object,
    radar_seed_covariances: object,
    loss_weight: float,
    gate_quantile: float,
    chunk_size: int,
) -> RadarLidarLossResult:
    """Compute an opacity term for static Gaussians near radar seeds.

    Args:
        gaussian_means: Torch tensor with shape ``[gaussian_count, 3]``.
        opacity_logits: Torch tensor with shape ``[gaussian_count]`` or
            ``[gaussian_count, 1]``.
        static_gaussian_mask: Boolean Torch tensor with shape ``[gaussian_count]``.
        radar_seed_means: Torch tensor with shape ``[seed_count, 3]``.
        radar_seed_covariances: Torch tensor with shape ``[seed_count, 3, 3]``.
        loss_weight: Non-negative multiplier for the radar-supported term.
        gate_quantile: Chi-square CDF quantile in the open interval ``(0, 1)``.
        chunk_size: Positive number of Gaussian means processed per chunk.

    Returns:
        RadarLidarLossResult containing the loss and support diagnostics.

    Raises:
        ValueError: If tensor shapes, values, or configuration values are invalid.
    """
    import torch
    from scipy.stats import chi2

    _validate_inputs(
        gaussian_means,
        opacity_logits,
        static_gaussian_mask,
        radar_seed_means,
        radar_seed_covariances,
        loss_weight,
        gate_quantile,
        chunk_size,
    )
    if (
        gaussian_means.device != opacity_logits.device
        or gaussian_means.device != static_gaussian_mask.device
        or gaussian_means.device != radar_seed_means.device
        or gaussian_means.device != radar_seed_covariances.device
    ):
        raise ValueError("all radar LiDAR loss tensors must use the same device")
    if (
        gaussian_means.dtype != radar_seed_means.dtype
        or gaussian_means.dtype != radar_seed_covariances.dtype
    ):
        raise ValueError("Gaussian and radar seed tensors must use the same dtype")

    static_indices = torch.nonzero(static_gaussian_mask, as_tuple=False).reshape(-1)
    seed_count = int(radar_seed_means.shape[0])
    if loss_weight == 0.0 or seed_count == 0 or static_indices.numel() == 0:
        zero_loss = opacity_logits.sum() * 0.0
        return RadarLidarLossResult(zero_loss, 0, 0.0, True)

    supported_mask, supported_seed_mask = radar_support_gate(
        gaussian_means[static_indices], radar_seed_means, radar_seed_covariances,
        gate_quantile, chunk_size,
    )
    supported_indices = static_indices[supported_mask]
    if supported_indices.numel() == 0:
        zero_loss = opacity_logits.sum() * 0.0
        return RadarLidarLossResult(zero_loss, 0, 0.0, True)

    flattened_opacity_logits = opacity_logits.reshape(-1)
    # -log(sigmoid(o)) = softplus(-o), stable even for very transparent Gaussians.
    radar_loss = loss_weight * torch.nn.functional.softplus(
        -flattened_opacity_logits[supported_indices]
    ).mean()
    return RadarLidarLossResult(
        radar_loss, int(supported_indices.numel()),
        float(supported_seed_mask.float().mean().item()), False,
    )


@transform("T42", "Inclusive Mahalanobis union gate, bounded over both Gaussians and radar anchors")
def radar_support_gate(gaussian_means, radar_seed_means, radar_seed_covariances,
                       gate_quantile: float, chunk_size: int):
    """Return Gaussian and anchor support masks without position gradients.

    Args:
        gaussian_means: Validated static means [G,3].
        radar_seed_means, radar_seed_covariances: Validated anchors [S,3], [S,3,3].
        gate_quantile: Chi-square probability in (0,1).
        chunk_size: Maximum rows on either pairwise axis.

    Returns:
        Boolean masks [G], [S]. Work is O(G*S), temporary storage O(chunk_size**2).
    """
    import torch
    from scipy.stats import chi2

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if not 0.0 < gate_quantile < 1.0:
        raise ValueError("gate quantile must be in (0, 1)")
    seed_count = int(radar_seed_means.shape[0])
    seed_covariance_inverses = torch.linalg.inv(radar_seed_covariances)
    chi_square_threshold = float(chi2.ppf(gate_quantile, df=3))
    supported_mask = torch.zeros(len(gaussian_means), dtype=torch.bool, device=gaussian_means.device)
    supported_seed_mask = torch.zeros(seed_count, dtype=torch.bool, device=gaussian_means.device)

    # Hard membership is not a differentiable attraction force. Avoid retaining
    # position graphs for pairwise distances; only the selected opacities learn.
    with torch.no_grad():
        for gaussian_start in range(0, len(gaussian_means), chunk_size):
            gaussian_slice = slice(gaussian_start, gaussian_start + chunk_size)
            for seed_start in range(0, seed_count, chunk_size):
                seed_slice = slice(seed_start, seed_start + chunk_size)
                deltas = gaussian_means[gaussian_slice, None, :] - radar_seed_means[None, seed_slice, :]
                squared_distance = torch.einsum("gsi,sij,gsj->gs", deltas,
                                                seed_covariance_inverses[seed_slice], deltas)
                membership = squared_distance <= chi_square_threshold
                supported_mask[gaussian_slice] |= membership.any(dim=1)
                supported_seed_mask[seed_slice] |= membership.any(dim=0)
    return supported_mask, supported_seed_mask


def calibrate_opacity_weight(unweighted_loss: float, stock_regularizer: float) -> float:
    """Return lambda = stock_regularizer / unweighted_loss; reject absent support.

    Args:
        unweighted_loss: Initial mean softplus(-opacity logit) on supported Gaussians.
        stock_regularizer: Actual weighted stock mcmc_opacity_reg from the same state.

    Returns:
        Finite positive multiplier; never invents a fallback weight.
    """
    if not all(math.isfinite(value) and value > 0 for value in (unweighted_loss, stock_regularizer)):
        raise ValueError("initial radar loss and stock opacity regularizer must be finite and positive")
    weight = stock_regularizer / unweighted_loss
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("calibrated opacity weight must be finite and positive")
    return weight


def _validate_inputs(
    gaussian_means: object,
    opacity_logits: object,
    static_gaussian_mask: object,
    radar_seed_means: object,
    radar_seed_covariances: object,
    loss_weight: float,
    gate_quantile: float,
    chunk_size: int,
) -> None:
    """Validate tensor contracts and scalar configuration before GPU work."""
    import torch

    expected_shapes = (
        (gaussian_means, 2, "gaussian_means"),
        (radar_seed_means, 2, "radar_seed_means"),
        (radar_seed_covariances, 3, "radar_seed_covariances"),
    )
    for tensor, expected_rank, tensor_name in expected_shapes:
        if not hasattr(tensor, "ndim") or tensor.ndim != expected_rank:
            raise ValueError(f"{tensor_name} must have rank {expected_rank}")
    if gaussian_means.shape[1] != 3 or radar_seed_means.shape[1] != 3:
        raise ValueError("Gaussian and radar seed means must have shape [N, 3]")
    if radar_seed_covariances.shape[1:] != (3, 3):
        raise ValueError("radar_seed_covariances must have shape [N, 3, 3]")
    if opacity_logits.numel() != gaussian_means.shape[0]:
        raise ValueError("opacity_logits must contain one value per Gaussian")
    if static_gaussian_mask.shape != (gaussian_means.shape[0],):
        raise ValueError("static_gaussian_mask must have shape [gaussian_count]")
    if static_gaussian_mask.dtype != torch.bool:
        raise ValueError("static_gaussian_mask must be boolean")
    if radar_seed_covariances.shape[0] != radar_seed_means.shape[0]:
        raise ValueError("radar seed means and covariances must have equal counts")
    if not (
        gaussian_means.is_floating_point()
        and opacity_logits.is_floating_point()
        and radar_seed_means.is_floating_point()
        and radar_seed_covariances.is_floating_point()
    ):
        raise ValueError("Gaussian and radar seed tensors must use floating-point dtypes")
    if not (
        torch.isfinite(gaussian_means).all()
        and torch.isfinite(opacity_logits).all()
        and torch.isfinite(radar_seed_means).all()
        and torch.isfinite(radar_seed_covariances).all()
    ):
        raise ValueError("Gaussian and radar seed tensors must be finite")
    if not torch.allclose(
        radar_seed_covariances,
        radar_seed_covariances.transpose(-1, -2),
    ):
        raise ValueError("radar seed covariances must be symmetric")
    try:
        torch.linalg.cholesky(radar_seed_covariances)
    except RuntimeError as error:
        raise ValueError("radar seed covariances must be positive definite") from error
    if (not math.isfinite(loss_weight) or loss_weight < 0.0 or not 0.0 < gate_quantile < 1.0
            or isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0):
        raise ValueError("loss weight must be non-negative; quantile must be in (0, 1); chunk size must be positive")
