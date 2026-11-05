"""Load and transform the radar seed metadata consumed by the LiDAR loss."""

import json
from pathlib import Path
from typing import Dict

import numpy as np


def load_radar_seed_metadata(seed_directory: Path, session_name: str, world_to_dataparser: np.ndarray) -> Dict[str, np.ndarray]:
    """Load radar seed means and covariances and transform them to the parser frame.

    Args:
        seed_directory: Directory containing ``radar_seeds.npz`` and optional
            ``radar_inputs.json`` metadata.
        session_name: Session name that the artifact must identify.
        world_to_dataparser: 3x4 map/world-to-dataparser transform.

    Returns:
        Dictionary containing float32 ``means`` and ``covariances`` arrays.

    Raises:
        FileNotFoundError: If the seed artifact is missing.
        ValueError: If metadata, shapes, finiteness, or covariance definiteness is invalid.
    """
    seed_path = seed_directory / "radar_seeds.npz"
    metadata_path = seed_directory / "radar_inputs.json"
    if not seed_path.is_file():
        raise FileNotFoundError(f"radar seed artifact is missing: {seed_path}")
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("session") != session_name:
            raise ValueError(f"{metadata_path}: session does not match {session_name!r}")
        if metadata.get("frame", "map") != "map":
            raise ValueError(f"{metadata_path}: radar seeds must be in the map frame")

    with np.load(seed_path) as arrays:
        if "means" not in arrays or "covariances" not in arrays:
            raise ValueError(f"{seed_path}: expected means and covariances arrays")
        means = np.asarray(arrays["means"], dtype=np.float64)
        covariances = np.asarray(arrays["covariances"], dtype=np.float64)
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"{seed_path}: means must have shape [N, 3]")
    if covariances.shape != (means.shape[0], 3, 3):
        raise ValueError(f"{seed_path}: covariances must have shape [N, 3, 3]")
    if not np.isfinite(means).all() or not np.isfinite(covariances).all():
        raise ValueError(f"{seed_path}: means and covariances must be finite")
    if not np.allclose(covariances, np.swapaxes(covariances, 1, 2)):
        raise ValueError(f"{seed_path}: covariances must be symmetric")
    if np.any(np.linalg.eigvalsh(covariances) <= 0):
        raise ValueError(f"{seed_path}: covariances must be positive definite")
    transform = np.asarray(world_to_dataparser, dtype=np.float64)
    if transform.shape != (3, 4):
        raise ValueError(f"world_to_dataparser must have shape [3, 4], got {transform.shape}")
    rotation = transform[:, :3]
    means = means @ rotation.T + transform[:, 3]
    covariances = rotation @ covariances @ rotation.T
    return {
        "means": means.astype(np.float32),
        "covariances": covariances.astype(np.float32),
    }
