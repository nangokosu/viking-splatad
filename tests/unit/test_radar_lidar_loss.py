"""Synthetic known-answer tests for the radar-supported LiDAR loss."""

import pytest

torch = pytest.importorskip("torch")

from viking_splatad.splatad.radar_lidar_loss import compute_radar_lidar_loss


@pytest.mark.parametrize("chunk_size", [1, 3])
def test_radar_lidar_loss_matches_supported_static_gaussians(chunk_size):
    """The loss includes only static Gaussians inside the seed ellipsoid."""
    gaussian_means = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    opacity_logits = torch.tensor([0.0, 1.0, -2.0])
    static_gaussian_mask = torch.tensor([True, False, True])
    radar_seed_means = torch.tensor([[0.0, 0.0, 0.0]])
    radar_seed_covariances = torch.eye(3).unsqueeze(0)

    result = compute_radar_lidar_loss(
        gaussian_means,
        opacity_logits,
        static_gaussian_mask,
        radar_seed_means,
        radar_seed_covariances,
        loss_weight=2.0,
        gate_quantile=0.95,
        chunk_size=chunk_size,
    )
    expected_loss = 2.0 * (-torch.log(torch.sigmoid(opacity_logits[0])))

    assert torch.allclose(result.loss, expected_loss)
    assert result.supported_gaussian_count == 1
    assert result.supported_seed_fraction == 1.0
    assert not result.empty_support


def test_radar_lidar_loss_excludes_point_outside_ellipsoid():
    """A Gaussian outside the configured Mahalanobis gate is excluded."""
    result = compute_radar_lidar_loss(
        torch.tensor([[2.0, 0.0, 0.0]]),
        torch.tensor([0.0]),
        torch.tensor([True]),
        torch.zeros((1, 3)),
        torch.eye(3).unsqueeze(0),
        loss_weight=1.0,
        gate_quantile=0.5,
        chunk_size=1,
    )

    assert result.loss.item() == 0.0
    assert result.supported_gaussian_count == 0
    assert result.supported_seed_fraction == 0.0
    assert result.empty_support


def test_zero_weight_preserves_zero_radar_term():
    """A disabled radar term returns a differentiable zero on the input device."""
    opacity_logits = torch.tensor([0.25], requires_grad=True)
    result = compute_radar_lidar_loss(
        torch.zeros((1, 3)),
        opacity_logits,
        torch.tensor([True]),
        torch.zeros((1, 3)),
        torch.eye(3).unsqueeze(0),
        loss_weight=0.0,
        gate_quantile=0.95,
        chunk_size=1,
    )

    result.loss.backward()
    assert result.loss.item() == 0.0
    assert opacity_logits.grad is not None
    assert opacity_logits.grad.item() == 0.0
