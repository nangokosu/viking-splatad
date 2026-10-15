"""Camera/lidar pose geometry for SplatAD: T9a, T9b, T18, T20."""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from viking_splatad.extract.pose import Trajectory, interpolate_pose_se3
from viking_splatad.registry import transform

# OpenCV (x right, y down, z forward) -> nerfstudio (x right, y up, z backward): negate y and z.
OPENCV_TO_NERFSTUDIO = np.diag([1.0, -1.0, -1.0])

# Fraction of the sweep duration at which each reference mode samples the trajectory.
SWEEP_REFERENCE_FRACTIONS = {"start": 0.0, "mid": 0.5, "end": 1.0}


@transform("T9a", "Camera-to-world (OpenCV axes): T_world_base @ T_base_camera")
def camera_to_world_opencv(
    T_world_base: np.ndarray, T_base_camera: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compose the camera-to-world matrix.

    Args:
        T_world_base: 4x4 base pose in world.
        T_base_camera: 4x4 camera extrinsic (camera in base).

    Returns:
        (c2w_3x4_float32 as SplatAD expects, c2w_4x4_float64 helper).
    """
    # c2w maps camera-frame points to world: world <- base <- camera.
    c2w_float64 = np.asarray(T_world_base, dtype=np.float64) @ np.asarray(T_base_camera, dtype=np.float64)
    # SplatAD stores only the top 3 rows, as float32.
    return c2w_float64[:3, :].astype(np.float32), c2w_float64


@transform("T9b", "Apply OPENCV_TO_NERFSTUDIO flip: R_out = R_in @ diag(1,-1,-1)")
def apply_opencv_to_nerfstudio(camera_to_world: np.ndarray) -> np.ndarray:
    """Flip camera axes from OpenCV to nerfstudio convention.

    Right-multiplying the rotation by diag(1,-1,-1) negates its y and z columns (the
    camera axes in world); translation is unchanged. The map is an involution.

    Args:
        camera_to_world: 3x4 or 4x4 c2w.

    Returns:
        Array of the same shape and dtype family with flipped rotation block.
    """
    flipped = np.array(camera_to_world, copy=True)
    flipped[:3, :3] = flipped[:3, :3] @ OPENCV_TO_NERFSTUDIO.astype(flipped.dtype)
    return flipped


@transform("T18", "Lidar-to-world at sweep reference time: T_world_base(t_ref) @ T_base_lidar")
def lidar_to_world_at_reference_time(
    trajectory: Trajectory,
    T_base_lidar: np.ndarray,
    sweep_start_time: float,
    sweep_duration: float,
    reference_mode: str = "mid",
) -> Tuple[np.ndarray, float]:
    """Lidar pose in world at start/mid/end of a sweep.

    t_ref = sweep_start_time + fraction * sweep_duration, fraction in {0, 0.5, 1}.

    Args:
        trajectory: Base trajectory T_world_base.
        T_base_lidar: 4x4 lidar extrinsic.
        sweep_start_time: Sweep start, seconds.
        sweep_duration: Sweep length, seconds.
        reference_mode: "start", "mid" or "end".

    Returns:
        (4x4 T_world_lidar, reference time used).

    Raises:
        ValueError: On unknown reference_mode.
    """
    if reference_mode not in SWEEP_REFERENCE_FRACTIONS:
        raise ValueError(f"reference_mode must be one of {sorted(SWEEP_REFERENCE_FRACTIONS)}")
    # Reference time inside the sweep at the mode's fraction of its duration.
    reference_time = sweep_start_time + SWEEP_REFERENCE_FRACTIONS[reference_mode] * sweep_duration
    # Base pose at that time (lerp/slerp), then chain base -> lidar extrinsic.
    T_world_base = interpolate_pose_se3(trajectory, np.array([reference_time]))[0]
    return T_world_base @ np.asarray(T_base_lidar, dtype=np.float64), float(reference_time)


@transform("T20", "Sensor index assignment replicating neurad-studio ad_dataparser")
def assign_sensor_indices(
    camera_names: Sequence[str], lidar_names: Sequence[str]
) -> Tuple[Dict[str, int], List[str]]:
    """Assign sensor indices: cameras 0..n_cam-1; lidars 0..n_lidar-1, offset by n_cam on overlap.

    Args:
        camera_names: Camera names in order.
        lidar_names: Lidar names in order.

    Returns:
        (name->index mapping, sensor_idx_to_name list).
    """
    camera_count = len(camera_names)
    camera_indices = {name: index for index, name in enumerate(camera_names)}
    lidar_indices = {name: index for index, name in enumerate(lidar_names)}
    # If any lidar index collides with a camera index, shift all lidars past the cameras.
    if set(lidar_indices.values()) & set(camera_indices.values()):
        lidar_indices = {name: index + camera_count for name, index in lidar_indices.items()}
    name_to_index = {**camera_indices, **lidar_indices}
    # Invert the mapping into a list addressed by sensor index.
    index_to_name = [""] * len(name_to_index)
    for name, index in name_to_index.items():
        index_to_name[index] = name
    # Contiguity: indices must be exactly 0..N-1 with unique names (a duplicate name
    # would shrink name_to_index and leave a gap or out-of-range index).
    assert sorted(name_to_index.values()) == list(range(len(name_to_index))), "sensor indices not contiguous"
    assert "" not in index_to_name
    return name_to_index, index_to_name
