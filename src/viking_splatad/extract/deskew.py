"""Ego-motion compensation (deskew) of spinning-lidar sweeps (T13)."""

from typing import Callable, Tuple

import numpy as np

from viking_splatad.registry import transform


@transform(
    "T13",
    "Deskew: move each point from the lidar frame at its capture time to the lidar "
    "frame at the reference time; return signed per-point time offsets",
)
def deskew_points(
    points_lidar: np.ndarray,
    point_times: np.ndarray,
    pose_at: Callable[[np.ndarray], np.ndarray],
    reference_time: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Motion-compensate a sweep to the pose at ``reference_time``.

    Math: a point measured at time t_p is p_p in the lidar frame at t_p. With
    T_world_lidar(t) the pose of the lidar in the world,
        p_world = T_world_lidar(t_p) @ p_p
        p_ref   = inv(T_world_lidar(t_ref)) @ p_world
                = inv(T_world_lidar(t_ref)) @ T_world_lidar(t_p) @ p_p.
    The signed offset is t = t_p - t_ref (negative for points captured before the
    reference). SplatAD later undoes the compensation via xyz_rs = xyz - v * t.

    Args:
        points_lidar: [n,3] points, each in the lidar frame at its own capture time.
        point_times: [n] absolute capture times in seconds (float64).
        pose_at: Callable mapping times [m] -> T_world_lidar poses [m,4,4].
        reference_time: Absolute reference time in seconds.

    Returns:
        (points_reference, relative_time): [n,3] points in the reference lidar frame
        and [n] signed seconds t_p - t_ref.
    """
    points_lidar = np.asarray(points_lidar, dtype=np.float64)
    point_times = np.asarray(point_times, dtype=np.float64)
    # Per-point poses at capture time and the single pose at the reference time.
    poses_world_lidar_at_points = pose_at(point_times)
    pose_world_lidar_at_reference = pose_at(np.array([reference_time], dtype=np.float64))[0]
    # Inverse of the reference pose maps world -> reference lidar frame.
    pose_reference_world = np.linalg.inv(pose_world_lidar_at_reference)
    # Homogeneous coordinates so rotation and translation apply in one product.
    homogeneous_points = np.concatenate(
        [points_lidar, np.ones((points_lidar.shape[0], 1))], axis=1
    )
    # Per-point relative transform T_ref_point = inv(T_ref) @ T_point (batched).
    poses_reference_point = np.einsum("ab,nbc->nac", pose_reference_world, poses_world_lidar_at_points)
    points_reference = np.einsum("nac,nc->na", poses_reference_point, homogeneous_points)[:, :3]
    # Signed time offset relative to the reference.
    relative_time = point_times - float(reference_time)
    return points_reference, relative_time
