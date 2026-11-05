"""Known-answer tests for radar seed metadata loading and frame conversion."""

import json

import numpy as np
import pytest

from viking_splatad.splatad.radar_seed_metadata import load_radar_seed_metadata


def test_load_radar_seed_metadata_transforms_means_and_covariances(tmp_path):
    """The parser transform is applied to means and covariances analytically."""
    seed_directory = tmp_path / "radar_inputs"
    seed_directory.mkdir()
    np.savez(
        seed_directory / "radar_seeds.npz",
        means=np.array([[1.0, 2.0, 3.0]]),
        covariances=np.diag([1.0, 2.0, 3.0])[None, ...],
    )
    (seed_directory / "radar_inputs.json").write_text(json.dumps({"session": "june", "frame": "map"}))
    world_to_dataparser = np.array([
        [0.0, -1.0, 0.0, 10.0],
        [1.0, 0.0, 0.0, 20.0],
        [0.0, 0.0, 1.0, 30.0],
    ])

    result = load_radar_seed_metadata(seed_directory, "june", world_to_dataparser)

    np.testing.assert_array_equal(result["means"], np.array([[8.0, 21.0, 33.0]], dtype=np.float32))
    np.testing.assert_allclose(
        result["covariances"],
        np.diag([2.0, 1.0, 3.0])[None, ...].astype(np.float32),
    )


def test_load_radar_seed_metadata_rejects_non_positive_definite_covariance(tmp_path):
    """A covariance with a non-positive eigenvalue cannot define a support ellipsoid."""
    seed_directory = tmp_path / "radar_inputs"
    seed_directory.mkdir()
    np.savez(
        seed_directory / "radar_seeds.npz",
        means=np.zeros((1, 3)),
        covariances=np.diag([1.0, 0.0, 1.0])[None, ...],
    )

    with pytest.raises(ValueError, match="positive definite"):
        load_radar_seed_metadata(seed_directory, "june", np.eye(3, 4))
