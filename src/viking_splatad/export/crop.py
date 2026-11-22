"""Observed-scene crop for the SuperSplat export (T86).

A trained SplatAD model holds Gaussians that never describe the scene: the init adds a far-hemisphere
cloud and a uniform cloud whose map-frame means can lie far outside the scene, while the scene itself
spans a few hundred metres. A viewer needs only the Gaussians inside the volume the lidar actually
observed and opaque enough to matter. This module builds that volume (the axis-aligned box of the
train-split lidar returns in the map frame) and the keep mask.

Reused, not re-derived: ``split_frames.csv`` as written by scripts/09 from ``dataset.run_split``
(train membership per sensor), ``open_dataset_view`` (poses ``T_world_sensor``, frame paths),
``load_lidar_sweep`` (finite + ego-return filter, exactly as training), T70 ``sweep_points_to_map``
(rigid ``p_map = R p + t``) and ``file_sha256`` (dataset identity). The dataset ``world`` frame of
``poses.npz`` is the map frame.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from viking_splatad.dataset.run_split import REFERENCE_SENSOR_KIND, TRAIN_SPLIT_NAME, file_sha256
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.registry import transform
from viking_splatad.splatad.viking_conversions import (
    load_lidar_sweep,
    open_dataset_view,
    parse_ego_filter,
)
from viking_splatad.viz.viewer_export import sweep_points_to_map

# Name of the per-run frame list written by scripts/09 (columns used: sensor, kind, split, frame_index,
# timestamp_ns, time_s, file).
SPLIT_FRAMES_FILE_NAME = "split_frames.csv"
# Name of the per-run split record holding the dataset hashes.
SPLIT_RECORD_FILE_NAME = "split.json"
# Dataset files whose SHA-256 the split record pins (file name -> key under split.json["dataset"]).
DATASET_HASH_KEYS = {"manifest.json": "manifest_sha256", "poses.npz": "poses_sha256"}
# Key of the dataset block inside split.json.
SPLIT_RECORD_DATASET_KEY = "dataset"
# Every n-th train sweep (per sensor) is read. 1 = all: the extremes of the box are sparse far returns
# that a subsample can miss, so the default reads every sweep. Callers may raise it for speed.
TRAIN_SWEEP_STRIDE = 1
# Tolerance (seconds) when comparing the CSV's time_s with the poses.npz time of the same frame. The
# timestamps have 1 ns resolution, so 1e-9 s is "equal up to float round-off".
TIME_MATCH_ATOL_SECONDS = 1e-9
# Valid range of an opacity (probability, not a logit).
OPACITY_MIN, OPACITY_MAX = 0.0, 1.0


@transform("T86", "Observed-scene crop: map-frame lidar bounding box; keep Gaussians inside it with opacity >= min_opacity")
def observed_bounds_from_lidar(points_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of map-frame lidar points.

    lower = min over points per axis, upper = max over points per axis (O(P)).

    Args:
        points_map: (P, 3) lidar returns in the map frame.

    Returns:
        (lower, upper): two (3,) float64 corners.

    Raises:
        ValueError: If the cloud is not (P, 3) with P >= 1 or has non-finite values.
    """
    # Step 1: validate, since a NaN would silently poison min/max.
    points_map = np.asarray(points_map, dtype=np.float64)
    if points_map.ndim != 2 or points_map.shape[1] != 3 or points_map.shape[0] < 1:
        raise ValueError(f"points_map must be (P>=1, 3), got {points_map.shape}")
    if not np.isfinite(points_map).all():
        raise ValueError("points_map has non-finite values")
    # Step 2: per-axis extremes.
    return points_map.min(axis=0), points_map.max(axis=0)


def crop_mask(
    means: np.ndarray, opacities: np.ndarray, lower: np.ndarray, upper: np.ndarray, min_opacity: float
) -> np.ndarray:
    """Keep Gaussians whose mean is inside the box (inclusive) and whose opacity >= min_opacity.

    Args:
        means: (N, 3) finite Gaussian centres in the map frame.
        opacities: (N,) peak alpha in [0, 1] (activated, not logits).
        lower: (3,) finite box minimum corner.
        upper: (3,) finite box maximum corner, >= lower on every axis.
        min_opacity: Opacity floor, from the run's config ``mcmc_min_opacity``.

    Returns:
        (N,) bool keep mask.

    Raises:
        ValueError: On a wrong shape, non-finite value, opacity outside [0, 1] or lower > upper.
    """
    means = np.asarray(means, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    # Step 1: shapes. (N,1) opacities would broadcast against the (N,) box test and give an (N,N) mask.
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means must be (N, 3), got {means.shape}")
    if opacities.shape != (means.shape[0],):
        raise ValueError(f"opacities must be ({means.shape[0]},), got {opacities.shape}")
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError(f"box corners must be (3,), got {lower.shape} and {upper.shape}")
    # Step 2: finiteness (NaN compares False everywhere and would drop Gaussians silently).
    for array_name, array in (("means", means), ("opacities", opacities), ("lower", lower), ("upper", upper)):
        if not np.isfinite(array).all():
            raise ValueError(f"{array_name} has non-finite values")
    # Step 3: opacities must be probabilities; logits (negative or > 1) would be compared to the floor wrongly.
    if opacities.size and (opacities.min() < OPACITY_MIN or opacities.max() > OPACITY_MAX):
        raise ValueError("opacities must lie in [0, 1]; pass sigmoid(logit), not raw logits")
    # Step 4: an inverted box would keep nothing without any warning.
    if (lower > upper).any():
        raise ValueError(f"box corners swapped: lower {lower} > upper {upper}")
    # Step 5: inside the closed box on all three axes, and opaque enough (inclusive threshold).
    inside_box = np.all((means >= lower) & (means <= upper), axis=1)
    return inside_box & (opacities >= min_opacity)


def check_dataset_identity(run_dir: Path, dataset_dir: Path) -> None:
    """Refuse a dataset other than the one the run trained on.

    Compares the SHA-256 of ``manifest.json`` and ``poses.npz`` in ``dataset_dir`` with the hashes
    recorded in the run's ``split.json``.

    Args:
        run_dir: Run folder holding ``split.json``.
        dataset_dir: Candidate dataset directory.

    Raises:
        ValueError: If a hash is missing from the record or differs from the file's hash.
    """
    # Step 1: the run's record of the dataset it was built from.
    split_record = json.loads((Path(run_dir) / SPLIT_RECORD_FILE_NAME).read_text())
    recorded = split_record.get(SPLIT_RECORD_DATASET_KEY, {})
    # Step 2: hash each pinned file of the candidate dataset and compare.
    for file_name, hash_key in DATASET_HASH_KEYS.items():
        if hash_key not in recorded:
            raise ValueError(f"{SPLIT_RECORD_FILE_NAME} of {run_dir} has no {hash_key}")
        actual_hash = file_sha256(Path(dataset_dir) / file_name)
        if actual_hash != recorded[hash_key]:
            raise ValueError(
                f"{file_name} of {dataset_dir} (sha256 {actual_hash}) is not the dataset the run trained on "
                f"({hash_key} {recorded[hash_key]})"
            )


def read_train_lidar_rows(run_dir: Path) -> Dict[str, List[dict]]:
    """Train-split lidar rows of ``split_frames.csv``, grouped by sensor in CSV order.

    Args:
        run_dir: Run folder holding ``split_frames.csv``.

    Returns:
        sensor name -> list of CSV row dicts (camera rows and non-train rows are dropped).
    """
    rows_by_sensor: Dict[str, List[dict]] = {}
    with open(Path(run_dir) / SPLIT_FRAMES_FILE_NAME, newline="") as split_file:
        for row in csv.DictReader(split_file):
            # Only lidar frames of the train split: they are what the model was fitted to.
            if row["kind"] == REFERENCE_SENSOR_KIND and row["split"] == TRAIN_SPLIT_NAME:
                rows_by_sensor.setdefault(row["sensor"], []).append(row)
    return rows_by_sensor


def train_lidar_bounds_map(
    run_dir: Path, dataset_dir: Path, sweep_stride: int = TRAIN_SWEEP_STRIDE
) -> Tuple[np.ndarray, np.ndarray]:
    """Map-frame bounding box of a run's train lidar sweeps (every ``sweep_stride``-th per sensor).

    Each sweep is stored in the sensor frame and moved with its ``T_world_sensor`` pose
    (p_map = p @ R^T + t, T70). The box is accumulated sweep by sweep (running per-axis min and max),
    so memory is O(one sweep) instead of O(all points). Frames are paired by file name and index,
    never by list position: the CSV row, the sorted frame files and the pose rows must agree.

    Args:
        run_dir: Run folder holding ``split_frames.csv`` and ``split.json``.
        dataset_dir: Dataset the run trained on (``manifest.json``, ``poses.npz``, ``lidar/<sensor>/``).
        sweep_stride: Keep every n-th train sweep of each sensor, n >= 1.

    Returns:
        (lower, upper): two (3,) float64 map-frame corners.

    Raises:
        ValueError: On a stride < 1, a dataset that is not the run's, a file/pose count mismatch,
            gapped frame numbering, a CSV row that disagrees with the file name or poses, or when
            no train lidar point exists.
    """
    # Step 0: a stride below 1 would make the slice below empty or reversed.
    if sweep_stride < 1:
        raise ValueError(f"sweep_stride must be >= 1, got {sweep_stride}")
    # Step 1: the dataset must be the one the run trained on.
    check_dataset_identity(run_dir, dataset_dir)
    # Step 2: train lidar rows per sensor; none at all is an error.
    rows_by_sensor = read_train_lidar_rows(run_dir)
    if not rows_by_sensor:
        raise ValueError(f"no train lidar frames in {run_dir}")
    view = open_dataset_view(Path(dataset_dir))
    ego_radius, ego_box = parse_ego_filter(view.manifest["lidar"]["min_lidar_dist"])
    lower_running = np.full(3, np.inf)
    upper_running = np.full(3, -np.inf)
    for sensor_name, sensor_rows in rows_by_sensor.items():
        frame_paths = view.sensor_frame_paths[sensor_name]
        sensor_poses = view.sensor_poses[sensor_name]
        sensor_times = view.sensor_times[sensor_name]
        # Step 3: one frame file, one pose and one time per frame, numbered 0..N-1 without gaps. Only then
        # does frame index k denote both the k-th file and the k-th pose row.
        if not (len(frame_paths) == len(sensor_poses) == len(sensor_times)):
            raise ValueError(
                f"{sensor_name}: {len(frame_paths)} frame files, {len(sensor_poses)} poses, "
                f"{len(sensor_times)} times"
            )
        file_indices = [int(FRAME_FILE_PATTERN.match(path.name).group(1)) for path in frame_paths]
        if file_indices != list(range(len(frame_paths))):
            raise ValueError(f"{sensor_name}: frame file numbering is not 0..{len(frame_paths) - 1} without gaps")
        # Step 4: the stride applies within this sensor's train rows only.
        for row in sensor_rows[::sweep_stride]:
            frame_index = int(row["frame_index"])
            if not 0 <= frame_index < len(frame_paths):
                raise ValueError(f"{sensor_name}: frame_index {frame_index} outside 0..{len(frame_paths) - 1}")
            frame_path = frame_paths[frame_index]
            # Step 5: cross-check the row against the file name (index, timestamp) and the poses (time).
            if Path(row["file"]).name != frame_path.name:
                raise ValueError(f"{sensor_name}: CSV file {row['file']!r} != dataset file {frame_path.name!r}")
            name_timestamp_ns = int(FRAME_FILE_PATTERN.match(frame_path.name).group(2))
            if int(row["timestamp_ns"]) != name_timestamp_ns:
                raise ValueError(
                    f"{sensor_name}: CSV timestamp_ns {row['timestamp_ns']} != file name stamp {name_timestamp_ns}"
                )
            if not np.isclose(float(row["time_s"]), sensor_times[frame_index], rtol=0.0, atol=TIME_MATCH_ATOL_SECONDS):
                raise ValueError(
                    f"{sensor_name}: CSV time_s {row['time_s']} != poses.npz time {sensor_times[frame_index]}"
                )
            # Step 6: load as in training and move to the map frame with this frame's own pose.
            sweep = load_lidar_sweep(frame_path, ego_radius, ego_box)
            if sweep.shape[0] == 0:
                continue
            sweep_bounds = observed_bounds_from_lidar(sweep_points_to_map(sweep[:, :3], sensor_poses[frame_index]))
            # Step 7: the union box has the min of the minima and the max of the maxima.
            lower_running = np.minimum(lower_running, sweep_bounds[0])
            upper_running = np.maximum(upper_running, sweep_bounds[1])
    if not np.isfinite(lower_running).all():
        raise ValueError(f"no train lidar points survive the ego filter in {run_dir}")
    return lower_running, upper_running
