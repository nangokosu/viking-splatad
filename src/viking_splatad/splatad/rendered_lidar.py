"""Framework-free helpers for saving SplatAD's rendered lidar scans (T50).

``scripts/08_render_lidar_pointcloud.py`` renders every lidar pose of a trained run on the GPU
node. This module holds the numpy-only half, so it is unit-testable on the Mac:

    * T50: take a rendered scan's pose from the dataparser frame back to the map frame, so the
      rendered points can be compared with map-frame data (GT cuboids, raw sweeps + poses.npz);
    * name each rendered scan after the input sweep it re-renders (``<idx>_<ts_ns>``), found by
      matching scan times to the sweep times in ``poses.npz``.

Frames (neurad-studio ``ADDataParser``, ad_dataparser.py:316-323, 649-663):
    * map frame: the "world" of ``poses.npz`` (``<sensor>__poses_world_sensor``);
    * dataparser frame: map frame left-multiplied by ``w2m`` (the inverse of the mean lidar
      pose), stored as ``DataparserOutputs.dataparser_transform`` (3x4). Scale is 1.0.
"""

from pathlib import Path
from typing import List

import numpy as np

from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.registry import transform

# Extension of the per-sweep lidar files written by dataset/writer.py.
SWEEP_FILE_SUFFIX = ".npy"


def _to_homogeneous_4x4(pose_3x4_or_4x4: np.ndarray) -> np.ndarray:
    """Append the [0, 0, 0, 1] row to a 3x4 rigid transform (4x4 input is returned as float64).

    Args:
        pose_3x4_or_4x4: Rigid transform, shape (3, 4) or (4, 4).

    Returns:
        (4, 4) float64 homogeneous matrix.

    Raises:
        ValueError: If the shape is neither (3, 4) nor (4, 4).
    """
    pose_array = np.asarray(pose_3x4_or_4x4, dtype=np.float64)
    if pose_array.shape == (4, 4):
        return pose_array.copy()
    if pose_array.shape != (3, 4):
        raise ValueError(f"expected a (3, 4) or (4, 4) transform, got {pose_array.shape}")
    homogeneous_pose = np.eye(4, dtype=np.float64)
    homogeneous_pose[:3, :] = pose_array
    return homogeneous_pose


@transform("T50", "Dataparser-frame lidar pose -> map-frame pose: inv([w2m; 0 0 0 1]) @ lidar_to_world")
def lidar_pose_in_map_frame(lidar_to_world: np.ndarray, dataparser_transform: np.ndarray) -> np.ndarray:
    """Map-frame pose of a lidar whose pose is given in SplatAD's dataparser frame.

    The dataparser stores ``lidar_to_world_dp = W @ lidar_to_map`` with ``W = [w2m; 0 0 0 1]``,
    so ``lidar_to_map = inv(W) @ lidar_to_world_dp``. W is rigid, hence
    ``inv(W) = [R^T, -R^T t; 0 0 0 1]`` (no general inverse needed, no scale since
    ``dataparser_scale`` is 1.0 for ADDataParser). A lidar-frame point p reaches the map frame as
    ``lidar_to_map @ [p; 1]``.

    Args:
        lidar_to_world: Lidar pose in the dataparser frame, (3, 4) or (4, 4).
        dataparser_transform: ``DataparserOutputs.dataparser_transform`` (w2m), (3, 4) or (4, 4).

    Returns:
        (4, 4) float64 lidar-to-map pose.

    Raises:
        ValueError: If the dataparser transform's rotation is not orthonormal (a scaled or sheared
            W would make the rigid inverse wrong).
    """
    # Step 1: both inputs as 4x4 so they compose by matrix product.
    lidar_to_dataparser = _to_homogeneous_4x4(lidar_to_world)
    map_to_dataparser = _to_homogeneous_4x4(dataparser_transform)
    # Rigid-inverse precondition R^T R = I; the tolerance is float32 resolution (the dataparser
    # stores W as float32), times 10 for the few rounding steps in R^T R.
    float32_tolerance = 10.0 * float(np.finfo(np.float32).eps)
    rotation = map_to_dataparser[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=float32_tolerance):
        raise ValueError("dataparser_transform is not rigid (rotation not orthonormal)")
    # Step 2: rigid inverse of W: rotation transposed, translation -R^T t.
    rotation_transposed = map_to_dataparser[:3, :3].T
    dataparser_to_map = np.eye(4, dtype=np.float64)
    dataparser_to_map[:3, :3] = rotation_transposed
    dataparser_to_map[:3, 3] = -rotation_transposed @ map_to_dataparser[:3, 3]
    # Step 3: lidar -> dataparser -> map.
    return dataparser_to_map @ lidar_to_dataparser


def sorted_sweep_stems(sweep_dir: Path) -> List[str]:
    """File stems (``<idx>_<ts_ns>``) of a sensor's sweeps, ordered by frame index.

    The dataset view (``open_dataset_view``) orders sweeps the same way, and poses.npz
    ``<sensor>__times[k]`` belongs to the k-th sweep, so position k here is frame k there.

    Args:
        sweep_dir: ``data/intermediate/<session>/lidar/<sensor>``.

    Returns:
        Stems in frame-index order.

    Raises:
        ValueError: If a sweep name does not match the writer pattern, or the indices are not
            0..N-1 (a gap would break the position <-> poses.npz index correspondence).
    """
    # Step 1: parse (index, stem) of every sweep file; reject foreign names loudly.
    indexed_stems = []
    for sweep_path in Path(sweep_dir).glob(f"*{SWEEP_FILE_SUFFIX}"):
        name_match = FRAME_FILE_PATTERN.match(sweep_path.name)
        if name_match is None:
            raise ValueError(f"unexpected sweep file name {sweep_path.name}")
        indexed_stems.append((int(name_match.group(1)), sweep_path.stem))
    # Step 2: order by index and require the indices to be exactly 0..N-1.
    indexed_stems.sort()
    frame_indices = [frame_index for frame_index, _ in indexed_stems]
    if frame_indices != list(range(len(frame_indices))):
        raise ValueError(f"sweep indices in {sweep_dir} are not contiguous from 0")
    return [stem for _, stem in indexed_stems]


def match_scan_times_to_frame_indices(
    scan_times_seconds: np.ndarray, frame_times_seconds: np.ndarray, tolerance_seconds: float
) -> np.ndarray:
    """Index of the input sweep each rendered scan re-renders, by nearest time.

    Scan times are ``Lidars.times + DataparserOutputs.time_offset`` (the dataparser subtracts
    the earliest sensor time, ad_dataparser.py:300-314), i.e. the poses.npz time base.
    Nearest-neighbour search uses one sorted search: O((S + F) log F) for S scans, F frames.

    Args:
        scan_times_seconds: (S,) absolute times of the rendered scans.
        frame_times_seconds: (F,) poses.npz ``<sensor>__times``, ascending.
        tolerance_seconds: Largest accepted |scan time - frame time|. Callers pass half the
            sweep period: consecutive sweeps are one period apart, so any larger offset could
            belong to the neighbouring sweep.

    Returns:
        (S,) int64 frame indices.

    Raises:
        ValueError: If frame times are not strictly increasing, a scan has no frame within the
            tolerance, or two scans match the same frame.
    """
    scan_times = np.asarray(scan_times_seconds, dtype=np.float64).reshape(-1)
    frame_times = np.asarray(frame_times_seconds, dtype=np.float64).reshape(-1)
    # Step 1: searchsorted needs strictly increasing frame times.
    if frame_times.size == 0 or np.any(np.diff(frame_times) <= 0):
        raise ValueError("frame times must be non-empty and strictly increasing")
    # Step 2: insertion points; the nearest frame is the one left or right of each.
    right_indices = np.clip(np.searchsorted(frame_times, scan_times), 0, frame_times.size - 1)
    left_indices = np.clip(right_indices - 1, 0, frame_times.size - 1)
    left_offsets = np.abs(scan_times - frame_times[left_indices])
    right_offsets = np.abs(scan_times - frame_times[right_indices])
    nearest_indices = np.where(left_offsets <= right_offsets, left_indices, right_indices).astype(np.int64)
    nearest_offsets = np.minimum(left_offsets, right_offsets)
    # Step 3: every scan must sit on a sweep, and no sweep may be claimed twice.
    if np.any(nearest_offsets > tolerance_seconds):
        worst_scan = int(np.argmax(nearest_offsets))
        raise ValueError(
            f"scan {worst_scan} is {nearest_offsets[worst_scan]:.6f} s from the nearest sweep "
            f"(tolerance {tolerance_seconds:.6f} s)"
        )
    if np.unique(nearest_indices).size != nearest_indices.size:
        raise ValueError("two rendered scans matched the same input sweep")
    return nearest_indices
