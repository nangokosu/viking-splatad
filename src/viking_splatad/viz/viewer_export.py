"""Pure-numpy helper that moves a sweep's points into the map frame for the PyVista viewer.

The viewer is display only: GT cuboids are drawn, never SplatAD inputs, seeds or loss targets
(CLAUDE.md rule 6).
"""

import numpy as np

from viking_splatad.registry import transform


@transform("T70", "Sweep points (lidar frame) -> map frame with lidar_to_map, for display")
def sweep_points_to_map(points_lidar: np.ndarray, lidar_to_map: np.ndarray) -> np.ndarray:
    """Move points from a sweep's lidar frame to the map frame shared with the GT cuboids.

    p_map = R p + t, applied to row vectors as p @ R^T + t (the same rule as T68).

    Args:
        points_lidar: (N, 3) points, lidar frame.
        lidar_to_map: (4, 4) rigid pose of the sweep.

    Returns:
        (N, 3) float64 map-frame points.
    """
    pose = np.asarray(lidar_to_map, dtype=np.float64)
    return np.asarray(points_lidar, dtype=np.float64) @ pose[:3, :3].T + pose[:3, 3]
