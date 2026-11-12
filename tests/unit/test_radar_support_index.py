"""Exact sparse spatial gating agrees with dense ellipsoids, including duplicate anchors."""

import numpy as np
import pytest


def test_spatial_index_matches_dense_masks_and_analytic_loss():
    """Conservative radius buckets eliminate pairs without capping neighbors or changing support."""
    import torch
    from viking_splatad.splatad.radar_lidar_loss import compute_radar_lidar_loss, radar_support_gate
    from viking_splatad.splatad.radar_support_index import RadarSupportIndex

    random = np.random.default_rng(7)  # Fixed synthetic geometry, unrelated to trainer initialization.
    means = torch.tensor(random.normal(size=(31, 3)), dtype=torch.float64)
    anchors = torch.tensor(random.normal(size=(19, 3)), dtype=torch.float64)
    anchors[1] = anchors[0]
    scales = torch.tensor(random.uniform(0.05, 1., size=(19, 3)), dtype=torch.float64)
    covariances = torch.diag_embed(scales.square())
    for chunk_size in (2, 7):
        index = RadarSupportIndex(anchors, covariances, 0.95, chunk_size)
        expected = radar_support_gate(means, anchors, covariances, 0.95, chunk_size)
        actual = radar_support_gate(means, anchors, covariances, 0.95, chunk_size, support_index=index)
        assert torch.equal(actual[0], expected[0]) and torch.equal(actual[1], expected[1])
        logits = torch.zeros(len(means), dtype=torch.float64, requires_grad=True)
        result = compute_radar_lidar_loss(means, logits, torch.ones(len(means), dtype=torch.bool),
                                         anchors, covariances, 1., 0.95, chunk_size, support_index=index)
        assert result.loss.item() == pytest.approx(np.log(2))
        assert result.supported_gaussian_count == int(expected[0].sum())
        assert index.last_candidate_pairs <= len(means) * len(anchors)
        with pytest.raises(ValueError, match="immutable"):
            radar_support_gate(means, anchors.clone(), covariances, 0.95, chunk_size, support_index=index)


def test_spatial_index_requeries_current_gaussian_positions():
    """Moving a Gaussian changes support immediately; only target indexes are cached."""
    import torch
    from viking_splatad.splatad.radar_lidar_loss import radar_support_gate
    from viking_splatad.splatad.radar_support_index import RadarSupportIndex

    anchors, covariance = torch.zeros((1, 3)), torch.eye(3)[None]
    means = torch.zeros((1, 3))
    index = RadarSupportIndex(anchors, covariance, 0.95, 1)
    assert radar_support_gate(means, anchors, covariance, 0.95, 1, support_index=index)[0].item()
    means[0, 0] = 100.
    assert not radar_support_gate(means, anchors, covariance, 0.95, 1, support_index=index)[0].item()


def test_full_window_smoke_shares_verified_canonical_anchor_artifact():
    """Only iteration/checkpoint/time budgets change; split/window and GPU remain full-scene."""
    from tests.unit.test_slurm_submit import load_submit_module

    helpers = load_submit_module()
    config = helpers.load_slurm_config()
    full = config["profiles"]["june_loss_only"]
    smoke = config["profiles"]["june_loss_only_smoke"]
    assert "window_seconds" not in smoke and "window_start_seconds" not in smoke
    assert helpers.resolve_job_gres(config, smoke) == config["gres"]
    for key in set(full) - {"max_num_iterations", "steps_per_save", "time_limit"}:
        assert smoke[key] == full[key]
    baseline_options = {"method": "splatad", "loss_weight": 0., "gate_quantile": 0.95,
                        "chunk_size": 4096, "inputs_dir": None}
    options = helpers.profile_radar_lidar_options(smoke, baseline_options, "/stage", "june", "smoke", "rev")
    assert options["inputs_dir"] == "/stage/radar-inputs/june/june_loss_only/rev"
