"""Recorded lidar sweeps of a session, streamed one at a time.

Feeds the scored-cell scan in ``scored_cells.py``: poses and returns of every Ouster sweep with
its split label. Nothing here enters a parser, seed set or loss (project rule 6).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from viking_splatad.dataset.run_split import build_run_split_with_cores
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.splatad.viking_conversions import (
    load_lidar_sweep,
    open_dataset_view,
    parse_ego_filter,
)

# Manifest sensor kind and default stream name of the lidar whose sweeps are streamed.
LIDAR_SENSOR_KIND = "lidar"
DEFAULT_LIDAR_SENSOR_NAME = "ouster"
# Column slice of x, y, z in a returns row (x,y,z,intensity,t[,channel]).
RETURN_XYZ_COLUMNS = slice(0, 3)


@dataclass(frozen=True)
class SessionSweep:
    """One recorded sweep with its split label and pose.

    Attributes:
        stem: File stem ``<frame_index>_<timestamp_ns>``.
        split: ``train``, ``test`` or ``guard``.
        timestamp_ns: Stamp in the file name.
        returns_xyz: float64 (n,3) returns in the lidar frame, after the ego filter.
        lidar_to_map: float64 (4,4) pose of the lidar in the map frame.
    """

    stem: str
    split: str
    timestamp_ns: int
    returns_xyz: np.ndarray
    lidar_to_map: np.ndarray


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
    for sweep_path, lidar_to_map in zip(sweep_paths, poses, strict=True):
        timestamp_ns = int(FRAME_FILE_PATTERN.match(sweep_path.name).group(2))
        if timestamp_ns not in split_by_timestamp:
            raise ValueError(f"{sweep_path.name} has no split row (outside the split window)")
        sweep = load_lidar_sweep(sweep_path, ego_radius, ego_box)
        yield SessionSweep(
            stem=sweep_path.stem, split=split_by_timestamp[timestamp_ns], timestamp_ns=timestamp_ns,
            returns_xyz=sweep[:, RETURN_XYZ_COLUMNS].astype(np.float64),
            lidar_to_map=np.asarray(lidar_to_map, dtype=np.float64))
