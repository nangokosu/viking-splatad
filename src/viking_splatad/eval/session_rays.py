"""Recorded lidar sweeps of a session as map-frame rays (T93), for ray-evidence evaluation.

Feeds ``recorded_ray_evidence`` (``hidden_geometry``, T78): every beam of every Ouster sweep
becomes an origin, a unit direction, a depth and a did-return flag in the map frame.
Nothing here enters a parser, seed set or loss (project rule 6).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np

from viking_splatad.dataset.run_split import build_run_split_with_cores
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.registry import transform
from viking_splatad.splatad.viking_conversions import (
    load_lidar_sweep,
    no_return_path_for_sweep,
    open_dataset_view,
    parse_ego_filter,
)

# neurad-studio's DUMMY_DISTANCE_VALUE (ad_dataparser), the depth the dataparser gives no-return
# beams (T55). Not imported: neurad is absent from the project venv. Any finite positive value
# works for T78, which ignores depth of no-return beams; the same value keeps one convention.
NO_RETURN_DEPTH_PLACEHOLDER_M = 2000.0
# Manifest sensor kind and default stream name of the lidar whose sweeps are the rays.
LIDAR_SENSOR_KIND = "lidar"
DEFAULT_LIDAR_SENSOR_NAME = "ouster"
# Column slice of x, y, z in a returns row (x,y,z,intensity,t[,channel]).
RETURN_XYZ_COLUMNS = slice(0, 3)
# Column slice of dir_x, dir_y, dir_z in a no-return row (dir,t,channel,column).
NO_RETURN_DIRECTION_COLUMNS = slice(0, 3)
# Homogeneous pose sizes accepted by ``sweep_rays_in_map``.
POSE_ROW_COUNT_3X4 = 3
POSE_ROW_COUNT_4X4 = 4
POSE_COLUMN_COUNT = 4
# Space dimension.
SPACE_DIMENSION = 3


@dataclass(frozen=True)
class SessionSweep:
    """One recorded sweep, ready for ``sweep_rays_in_map``.

    Attributes:
        stem: File stem ``<frame_index>_<timestamp_ns>``.
        split: ``train``, ``test`` or ``guard``.
        timestamp_ns: Stamp in the file name.
        returns_xyz: float64 (n,3) returns in the lidar frame, after the ego filter.
        no_return_directions: float64 (m,3) unit directions of no-return beams, lidar frame.
        lidar_to_map: float64 (4,4) pose of the lidar in the map frame.
    """

    stem: str
    split: str
    timestamp_ns: int
    returns_xyz: np.ndarray
    no_return_directions: np.ndarray
    lidar_to_map: np.ndarray


@transform("T93", "Recorded lidar sweep (returns + no-return beams) -> map-frame ray origin, directions, depths, did_return")
def sweep_rays_in_map(
    returns_xyz: np.ndarray, no_return_directions: np.ndarray, lidar_to_map: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rotate one sweep's beams into the map frame.

    For a return p (lidar frame) and pose (R, t): origin = t, depth = ||p||,
    direction = R p / ||p|| (R is orthonormal, so ||R p|| = ||p||). A no-return beam with unit
    direction d has direction R d / ||R d|| and the placeholder depth.

    Args:
        returns_xyz: (n,3) return points, lidar frame.
        no_return_directions: (m,3) beam directions of no-returns, lidar frame.
        lidar_to_map: (4,4) or (3,4) lidar-to-map pose.

    Returns:
        ``(origin (3,), directions (R,3), depths (R,), did_return (R,) bool)``, float64 except
        did_return, with
        R = (returns with finite non-zero range) + m, returns first then no-returns.

    Raises:
        ValueError: If an argument has the wrong shape.
    """
    # Step 1: validate shapes and take the pose as float64 rotation + translation.
    pose = np.asarray(lidar_to_map, dtype=np.float64)
    if pose.shape not in ((POSE_ROW_COUNT_3X4, POSE_COLUMN_COUNT), (POSE_ROW_COUNT_4X4, POSE_COLUMN_COUNT)):
        raise ValueError(f"lidar_to_map must be (3,4) or (4,4), got {pose.shape}")
    return_points = np.asarray(returns_xyz, dtype=np.float64).reshape(-1, SPACE_DIMENSION)
    beam_directions = np.asarray(no_return_directions, dtype=np.float64).reshape(-1, SPACE_DIMENSION)
    rotation = pose[:SPACE_DIMENSION, :SPACE_DIMENSION]
    origin = pose[:SPACE_DIMENSION, SPACE_DIMENSION].copy()
    # Step 2: returns; drop rows whose range is not finite or is zero (no direction).
    return_ranges = np.linalg.norm(return_points, axis=1)
    valid_returns = np.isfinite(return_points).all(axis=1) & np.isfinite(return_ranges) & (return_ranges > 0.0)
    return_points, return_ranges = return_points[valid_returns], return_ranges[valid_returns]
    # Row-vector form of R p: (p R^T).
    return_directions = (return_points @ rotation.T) / return_ranges[:, None]
    # Step 3: no-returns, renormalised after rotation (guards float32 sidecar round-off).
    rotated_beams = beam_directions @ rotation.T
    beam_norms = np.linalg.norm(rotated_beams, axis=1)
    if not (np.isfinite(beam_norms).all() and (beam_norms > 0.0).all()):
        raise ValueError("no-return directions must be finite and non-zero")
    no_return_unit = rotated_beams / beam_norms[:, None]
    # Step 4: returns first, then no-returns.
    directions = np.concatenate([return_directions, no_return_unit], axis=0)
    depths = np.concatenate([return_ranges, np.full(len(no_return_unit), NO_RETURN_DEPTH_PLACEHOLDER_M)])
    did_return = np.concatenate(
        [np.ones(len(return_ranges), dtype=bool), np.zeros(len(no_return_unit), dtype=bool)])
    return origin, directions, depths, did_return


def count_session_sweeps(dataset_dir: Path, lidar_sensor_name: str = DEFAULT_LIDAR_SENSOR_NAME) -> int:
    """Number of recorded sweeps of the lidar, without loading any frame.

    Args:
        dataset_dir: Extracted dataset directory.
        lidar_sensor_name: Lidar stream name.

    Returns:
        Count of sweep files (all of them, regardless of the split window).

    Raises:
        KeyError: If the lidar is not in the dataset.
    """
    return len(open_dataset_view(Path(dataset_dir)).sensor_frame_paths[lidar_sensor_name])


def iter_session_sweeps(
    dataset_dir: Path, split_settings: dict, lidar_sensor_name: str = DEFAULT_LIDAR_SENSOR_NAME
) -> Iterator[SessionSweep]:
    """Yield every recorded sweep of the session in time order with its split label.

    The ego filter and the split are those of the dataparser (``load_lidar_sweep`` with the
    manifest's ``min_lidar_dist``; ``build_run_split_with_cores`` with the session settings).
    ``lidar_to_map`` is ``poses_world_sensor``, the map-frame pose the dataparser also reads.

    Args:
        dataset_dir: Extracted dataset directory.
        split_settings: ``session_split_settings`` dict (dataset_start_fraction,
            dataset_end_fraction, train_fraction, test_split_method, test_render_fraction).
        lidar_sensor_name: Lidar stream name.

    Yields:
        ``SessionSweep`` per sweep.

    Raises:
        ValueError: If a sweep has no split row (outside the split window).
        FileNotFoundError: If a sweep has no no-return file (as the dataparser).
    """
    # Step 1: view, ego filter and split rows keyed by timestamp.
    view = open_dataset_view(Path(dataset_dir))
    ego_radius, ego_box = parse_ego_filter(view.manifest["lidar"]["min_lidar_dist"])
    split_frames, _, _, _ = build_run_split_with_cores(
        view, split_settings["dataset_start_fraction"], split_settings["dataset_end_fraction"],
        split_settings["train_fraction"], test_split_method=split_settings["test_split_method"],
        test_render_fraction=split_settings["test_render_fraction"])
    split_by_timestamp = {
        frame.timestamp_ns: frame.split for frame in split_frames
        if frame.sensor == lidar_sensor_name and frame.kind == LIDAR_SENSOR_KIND}
    sweep_paths = view.sensor_frame_paths[lidar_sensor_name]
    poses = view.sensor_poses[lidar_sensor_name]
    # Step 2: one sweep at a time, in file (time) order.
    for sweep_path, lidar_to_map in zip(sweep_paths, poses):
        timestamp_ns = int(FRAME_FILE_PATTERN.match(sweep_path.name).group(2))
        if timestamp_ns not in split_by_timestamp:
            raise ValueError(f"{sweep_path.name} has no split row (outside the split window)")
        no_return_path = no_return_path_for_sweep(sweep_path)
        if not no_return_path.is_file():
            raise FileNotFoundError(f"{no_return_path} missing (re-extract the dataset with --force)")
        sweep = load_lidar_sweep(sweep_path, ego_radius, ego_box)
        no_return_rows = np.load(no_return_path)
        yield SessionSweep(
            stem=sweep_path.stem, split=split_by_timestamp[timestamp_ns], timestamp_ns=timestamp_ns,
            returns_xyz=sweep[:, RETURN_XYZ_COLUMNS].astype(np.float64),
            no_return_directions=no_return_rows[:, NO_RETURN_DIRECTION_COLUMNS].astype(np.float64),
            lidar_to_map=np.asarray(lidar_to_map, dtype=np.float64))
