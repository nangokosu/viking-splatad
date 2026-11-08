"""Radar-supported occupancy term added to the existing LiDAR training loss."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RadarLidarLossResult:
    """Computed radar-supported LiDAR loss and diagnostics."""

    loss: object
    supported_gaussian_count: int
    supported_seed_fraction: float
    empty_support: bool


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

    seed_covariance_inverses = torch.linalg.inv(radar_seed_covariances)
    chi_square_threshold = float(chi2.ppf(gate_quantile, df=3))
    supported_mask = torch.zeros(static_indices.numel(), dtype=torch.bool, device=gaussian_means.device)
    supported_seed_mask = torch.zeros(seed_count, dtype=torch.bool, device=gaussian_means.device)

    for chunk_start in range(0, static_indices.numel(), chunk_size):
        chunk_indices = static_indices[chunk_start:chunk_start + chunk_size]
        chunk_deltas = gaussian_means[chunk_indices, None, :] - radar_seed_means[None, :, :]
        mahalanobis_squared = torch.einsum(
            "gsi,sij,gsj->gs",
            chunk_deltas,
            seed_covariance_inverses,
            chunk_deltas,
        )
        chunk_support = mahalanobis_squared <= chi_square_threshold
        supported_mask[chunk_start:chunk_start + chunk_indices.numel()] = chunk_support.any(dim=1)
        supported_seed_mask |= chunk_support.any(dim=0)

    supported_indices = static_indices[supported_mask]
    if supported_indices.numel() == 0:
        zero_loss = opacity_logits.sum() * 0.0
        return RadarLidarLossResult(zero_loss, 0, 0.0, True)

    flattened_opacity_logits = opacity_logits.reshape(-1)
    # -log(sigmoid(o)) = softplus(-o); this form avoids sigmoid underflow for
    # transparent Gaussians and retains the gradient that should increase opacity.
    radar_loss = loss_weight * torch.nn.functional.softplus(
        -flattened_opacity_logits[supported_indices]
    ).mean()
    return RadarLidarLossResult(
        radar_loss,
        int(supported_indices.numel()),
        float(supported_seed_mask.float().mean().item()),
        False,
    )


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
    if loss_weight < 0.0 or gate_quantile <= 0.0 or gate_quantile >= 1.0 or chunk_size <= 0:
        raise ValueError("loss weight must be non-negative; quantile must be in (0, 1); chunk size must be positive")
