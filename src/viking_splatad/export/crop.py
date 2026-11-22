"""Scene crops for the SuperSplat export: the lidar bounding box (T86) and the driven-path corridor (T87).

A trained SplatAD model holds Gaussians that never describe the scene: the init adds a far-hemisphere
cloud and a uniform cloud whose map-frame means can lie far outside the scene, while the scene itself
spans a few hundred metres. A viewer needs only the Gaussians inside the volume the lidar actually
observed and opaque enough to matter. This module builds that volume from the train-split lidar
returns in the map frame (two variants, below) and the keep masks.

Reused, not re-derived: ``split_frames.csv`` as written by scripts/09 from ``dataset.run_split``
(train membership per sensor), ``open_dataset_view`` (poses ``T_world_sensor``, frame paths),
``load_lidar_sweep`` (finite + ego-return filter, exactly as training), T70 ``sweep_points_to_map``
(rigid ``p_map = R p + t``) and ``file_sha256`` (dataset identity). The dataset ``world`` frame of
``poses.npz`` is the map frame.

Two crops are offered:

* T86, the axis-aligned box of all train lidar returns. Simple, but the lidar sees far (about
  +-200 m), so the box keeps a sparse shell of large background Gaussians.
* T87, the driven-path corridor: keep Gaussians whose horizontal distance to the train sensor path is
  at most ``width`` (the ``CORRIDOR_RANGE_PERCENTILE``-th percentile of the horizontal range of train
  returns from their own sensor), whose z lies in the [t, 100 - t] percentile range
  (t = ``CORRIDOR_Z_TAIL_PERCENT``) of the z of the lidar returns inside that corridor (tree tops kept,
  stray returns cut) and whose opacity is >= the floor.
"""

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree

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
# The corridor reaches as far as this percentage of the train lidar returns do horizontally from their own
# sensor. Chosen from renders of the May 30k run's train camera under each width
# (docs/research/supersplat_export/corridor_width_render_sweep_may_output.json/.png): p95 leaves 3.9% of the
# pixels as holes (distant trees and background seen from the path are cut), p99 1.1%, p99.9 0.73% with the
# same error vs the photo as the uncropped box, while still dropping the far shell the box kept (kept x/y
# extent 153 x 112 m instead of 416 x 341 m).
CORRIDOR_RANGE_PERCENTILE = 99.9
# Every n-th train sweep per sensor feeds the width percentile: a percentile of millions of returns is
# stable on a 1-in-10 sample (the measurement script used stride 10). The z pass reads every sweep
# because the z tails are sparse.
CORRIDOR_WIDTH_SWEEP_STRIDE = 10
# The corridor z range is the [CORRIDOR_Z_TAIL_PERCENT, 100 - CORRIDOR_Z_TAIL_PERCENT] percentile range of the
# z of the lidar returns inside the corridor (percent, per tail). Measured with
# docs/research/supersplat_export/corridor_z_measure.py (outputs corridor_z_measure_{may,june}_output.json),
# reading every train sweep: raw min/max are -76.0/72.5 m (May) and -1.5/43.6 m (June), set by stray
# returns; the [0.001, 99.999] percentiles are -0.92/22.49 m (May) and 0.15/21.78 m (June), above the
# ~22 m tree tops, and cut 3,208 / 3,073 of ~4.8 M exported Gaussians; the 0.01% tail would cut the
# canopy at ~20 m.
CORRIDOR_Z_TAIL_PERCENT = 0.001
# Number of columns of the corridor path array (map-frame x, y).
PATH_XY_COLUMNS = 2
# Column of a map-frame (x, y, z) point holding z (up).
Z_COLUMN_INDEX = 2
# Bounds of the percentile argument of np.percentile (a percentage, 0 = minimum, 100 = maximum).
PERCENTILE_MIN, PERCENTILE_MAX = 0.0, 100.0
# A tail percentage must stay below this, or the lower percentile would reach the upper one (50 = median).
Z_TAIL_PERCENT_EXCLUSIVE_MAX = 50.0


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


def validate_gaussian_inputs(
    means: np.ndarray, opacities: np.ndarray, min_opacity: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the Gaussian arrays and opacity floor shared by both crop masks.

    Args:
        means: (N, 3) Gaussian centres.
        opacities: (N,) peak alpha in [0, 1] (activated, not logits).
        min_opacity: Opacity floor, a probability in [OPACITY_MIN, OPACITY_MAX].

    Returns:
        (means, opacities) as float64 arrays.

    Raises:
        ValueError: On a wrong shape, a non-finite value, an opacity outside [0, 1], or a
            ``min_opacity`` that is non-finite or outside [0, 1].
    """
    means = np.asarray(means, dtype=np.float64)
    opacities = np.asarray(opacities, dtype=np.float64)
    # Step 1: shapes. (N,1) opacities would broadcast against the (N,) tests and give an (N,N) mask.
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError(f"means must be (N, 3), got {means.shape}")
    if opacities.shape != (means.shape[0],):
        raise ValueError(f"opacities must be ({means.shape[0]},), got {opacities.shape}")
    # Step 2: finiteness (NaN compares False everywhere and would drop Gaussians silently).
    for array_name, array in (("means", means), ("opacities", opacities)):
        if not np.isfinite(array).all():
            raise ValueError(f"{array_name} has non-finite values")
    # Step 3: opacities must be probabilities; logits (negative or > 1) would be compared to the floor wrongly.
    if opacities.size and (opacities.min() < OPACITY_MIN or opacities.max() > OPACITY_MAX):
        raise ValueError("opacities must lie in [0, 1]; pass sigmoid(logit), not raw logits")
    # Step 4: the floor is itself a probability; NaN would keep nothing, a negative floor everything.
    if not np.isfinite(min_opacity) or not OPACITY_MIN <= min_opacity <= OPACITY_MAX:
        raise ValueError(f"min_opacity must be finite and in [{OPACITY_MIN}, {OPACITY_MAX}], got {min_opacity}")
    return means, opacities


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
        ValueError: On a wrong shape, non-finite value, opacity or ``min_opacity`` outside [0, 1]
            or lower > upper.
    """
    # Step 1: shared validation of the Gaussians and the floor.
    means, opacities = validate_gaussian_inputs(means, opacities, min_opacity)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    # Step 2: box corners: shape and finiteness.
    if lower.shape != (3,) or upper.shape != (3,):
        raise ValueError(f"box corners must be (3,), got {lower.shape} and {upper.shape}")
    for array_name, array in (("lower", lower), ("upper", upper)):
        if not np.isfinite(array).all():
            raise ValueError(f"{array_name} has non-finite values")
    # Step 3: an inverted box would keep nothing without any warning.
    if (lower > upper).any():
        raise ValueError(f"box corners swapped: lower {lower} > upper {upper}")
    # Step 4: inside the closed box on all three axes, and opaque enough (inclusive threshold).
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


def iter_validated_train_rows(
    run_dir: Path, dataset_dir: Path, sweep_stride: int
) -> Iterator[Tuple[str, Path, np.ndarray, float]]:
    """Validate a run's train lidar layout and yield one (sensor, file, pose, time) per selected row.

    Frames are paired by file name and index, never by list position: the CSV row, the sorted frame
    files and the pose rows must agree. Checks run lazily as the generator is consumed.

    Args:
        run_dir: Run folder holding ``split_frames.csv`` and ``split.json``.
        dataset_dir: Dataset the run trained on.
        sweep_stride: Keep every n-th train sweep of each sensor, n >= 1.

    Yields:
        (sensor_name, frame_path, pose_world_sensor (4, 4), time_s) for each selected train sweep.

    Raises:
        ValueError: On a stride < 1, a dataset that is not the run's, no train lidar rows, a
            file/pose count mismatch, gapped frame numbering, or a CSV row that disagrees with the
            file name or poses.
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
            yield sensor_name, frame_path, sensor_poses[frame_index], float(sensor_times[frame_index])


def iter_train_sweeps_map(
    run_dir: Path, dataset_dir: Path, sweep_stride: int
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Stream a run's train lidar sweeps in the map frame, one non-empty sweep at a time (O(one sweep) memory).

    Each sweep is stored in the sensor frame and moved with its ``T_world_sensor`` pose
    (p_map = p @ R^T + t, T70); the sensor origin in the map frame is the pose translation t.

    Args:
        run_dir: Run folder holding ``split_frames.csv`` and ``split.json``.
        dataset_dir: Dataset the run trained on (``manifest.json``, ``poses.npz``, ``lidar/<sensor>/``).
        sweep_stride: Keep every n-th train sweep of each sensor, n >= 1.

    Yields:
        (sensor_origin_map (3,), points_map (P, 3)) per sweep with at least one point surviving the
        ego filter.

    Raises:
        ValueError: As ``iter_validated_train_rows``.
    """
    # Step 0: refuse a bad stride and a foreign dataset BEFORE the dataset is opened, so a wrong dataset
    # raises the identity error rather than a loader error from its manifest or poses.
    if sweep_stride < 1:
        raise ValueError(f"sweep_stride must be >= 1, got {sweep_stride}")
    check_dataset_identity(run_dir, dataset_dir)
    # Step 1: the ego filter is the training one, read from the manifest.
    view = open_dataset_view(Path(dataset_dir))
    ego_radius, ego_box = parse_ego_filter(view.manifest["lidar"]["min_lidar_dist"])
    # Step 2: validated rows; load as in training and move to the map frame with this frame's own pose.
    for _sensor_name, frame_path, pose_world_sensor, _time_s in iter_validated_train_rows(
        run_dir, dataset_dir, sweep_stride
    ):
        sweep = load_lidar_sweep(frame_path, ego_radius, ego_box)
        if sweep.shape[0] == 0:
            continue
        yield (
            np.asarray(pose_world_sensor[:3, 3], dtype=np.float64),
            sweep_points_to_map(sweep[:, :3], pose_world_sensor),
        )


def train_sensor_positions_map(run_dir: Path, dataset_dir: Path) -> np.ndarray:
    """Map-frame origins (pose translation) of every train lidar sweep, from the poses only.

    No point file is loaded. Validation is the same as for the sweep generator.

    Args:
        run_dir: Run folder holding ``split_frames.csv`` and ``split.json``.
        dataset_dir: Dataset the run trained on.

    Returns:
        (P, 3) float64 sensor origins, one per train lidar sweep (empty sweeps included).

    Raises:
        ValueError: As ``iter_validated_train_rows``.
    """
    # The origin of a sensor in the map frame is t of T_world_sensor (the image of the sensor-frame origin).
    return np.array(
        [pose[:3, 3] for _sensor, _path, pose, _time in iter_validated_train_rows(run_dir, dataset_dir, 1)],
        dtype=np.float64,
    ).reshape(-1, 3)


def train_lidar_bounds_map(
    run_dir: Path, dataset_dir: Path, sweep_stride: int = TRAIN_SWEEP_STRIDE
) -> Tuple[np.ndarray, np.ndarray]:
    """Map-frame bounding box of a run's train lidar sweeps (every ``sweep_stride``-th per sensor).

    The box is accumulated sweep by sweep (running per-axis min and max), so memory is O(one sweep)
    instead of O(all points).

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
    # Step 1: start from the empty box (+inf lower, -inf upper), the identity of min/max union.
    lower_running = np.full(3, np.inf)
    upper_running = np.full(3, -np.inf)
    for _origin_map, points_map in iter_train_sweeps_map(run_dir, dataset_dir, sweep_stride):
        sweep_bounds = observed_bounds_from_lidar(points_map)
        # The union box has the min of the minima and the max of the maxima.
        lower_running = np.minimum(lower_running, sweep_bounds[0])
        upper_running = np.maximum(upper_running, sweep_bounds[1])
    # Step 2: a box still at +inf means no sweep yielded a point.
    if not np.isfinite(lower_running).all():
        raise ValueError(f"no train lidar points survive the ego filter in {run_dir}")
    return lower_running, upper_running


@dataclass(frozen=True)
class TrainPathCorridor:
    """Driven-path corridor: horizontal distance to ``path_xy`` <= ``width_m``, z in [z_lower, z_upper].

    Attributes:
        path_xy: (P, 2) map-frame x, y of the train sensor origins.
        path_positions_map: (P, 3) the same origins with z, so that a caller needing the full positions
            (the viewer origin) does not read the poses a second time; ``path_xy`` is its first two columns.
        width_m: Corridor half-width in metres (horizontal distance to the path).
        z_lower: Low z bound (map frame): the ``z_tail_percent``-th percentile of the z of the lidar
            returns inside the corridor.
        z_upper: High z bound (map frame): the (100 - ``z_tail_percent``)-th percentile of the same z.
    """

    path_xy: np.ndarray
    width_m: float
    z_lower: float
    z_upper: float
    path_positions_map: Optional[np.ndarray] = None


def check_sensor_positions(sensor_positions_map: np.ndarray) -> np.ndarray:
    """Validate a caller-supplied (P, 3) array of sensor positions and return it as float64.

    Args:
        sensor_positions_map: Candidate map-frame sensor positions.

    Returns:
        (P, 3) float64 array, P >= 1, all finite.

    Raises:
        ValueError: If the shape is not (P >= 1, 3) or a value is not finite.
    """
    positions = np.asarray(sensor_positions_map, dtype=np.float64)
    # Shape first (an empty or (P, 2) path would break the KD-tree or the viewer origin), then finiteness.
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] < 1:
        raise ValueError(f"sensor_positions_map must be (P>=1, 3), got {positions.shape}")
    if not np.isfinite(positions).all():
        raise ValueError("sensor_positions_map has non-finite values")
    return positions


def train_lidar_corridor_map(
    run_dir: Path,
    dataset_dir: Path,
    width_sweep_stride: int = CORRIDOR_WIDTH_SWEEP_STRIDE,
    range_percentile: float = CORRIDOR_RANGE_PERCENTILE,
    sensor_positions_map: Optional[np.ndarray] = None,
    z_tail_percent: float = CORRIDOR_Z_TAIL_PERCENT,
) -> TrainPathCorridor:
    """Driven-path corridor of a run's train lidar (T87 inputs).

    Pass 1 (every ``width_sweep_stride``-th sweep): width = the ``range_percentile``-th percentile of
    r = ||p_xy - o_xy||, the horizontal range of each return p from its own sweep's sensor origin o.
    Pass 2 (every sweep, streaming): the z of returns whose horizontal distance to the nearest path point
    (cKDTree on the path, O(M log P) per sweep of M points) is <= width, then
    z_lower = percentile(z, t) and z_upper = percentile(z, 100 - t) with t = ``z_tail_percent``. Percentiles
    instead of min/max because single stray returns set the extremes (t = 0 gives exactly min/max).

    Args:
        run_dir: Run folder holding ``split_frames.csv`` and ``split.json``.
        dataset_dir: Dataset the run trained on.
        width_sweep_stride: Sweep stride of pass 1, n >= 1.
        range_percentile: Percentile in [0, 100] of the horizontal range that sets the width.
        sensor_positions_map: Optional (P, 3) output of ``train_sensor_positions_map`` for this run and
            dataset, computed once by a caller that also needs it. None = read it here.
        z_tail_percent: Percent cut from each end of the corridor z distribution, in
            [PERCENTILE_MIN, 50). 0 = raw min/max.

    Returns:
        The corridor (``path_positions_map`` holds the positions used).

    Raises:
        ValueError: On the errors of ``iter_validated_train_rows``, a percentile outside [0, 100], a
            ``sensor_positions_map`` that is not finite (P >= 1, 3), a ``z_tail_percent`` outside [0, 50), no return in pass 1, or no return inside the corridor.
    """
    # Step 0: np.percentile accepts only [PERCENTILE_MIN, PERCENTILE_MAX].
    if not PERCENTILE_MIN <= range_percentile <= PERCENTILE_MAX:
        raise ValueError(
            f"range_percentile must be in [{PERCENTILE_MIN}, {PERCENTILE_MAX}], got {range_percentile}"
        )
    # Step 0b: the tails must leave a non-empty central range (t < 50 keeps z_lower <= z_upper).
    if not PERCENTILE_MIN <= z_tail_percent < Z_TAIL_PERCENT_EXCLUSIVE_MAX:
        raise ValueError(
            f"z_tail_percent must be in [{PERCENTILE_MIN}, {Z_TAIL_PERCENT_EXCLUSIVE_MAX}), got {z_tail_percent}"
        )
    # Step 1: the path is the train sensor origins' x, y.
    path_positions_map = (train_sensor_positions_map(run_dir, dataset_dir) if sensor_positions_map is None
                          else check_sensor_positions(sensor_positions_map))
    path_xy = path_positions_map[:, :PATH_XY_COLUMNS]
    # Step 2: pass 1, horizontal ranges from each sweep's own origin on a sweep subsample. Stored as float32
    # (halves memory): a percentile of distances in metres needs no float64 precision (float32 resolves
    # ~1e-5 m at 100 m, far below any meaningful corridor width).
    range_chunks = [
        np.linalg.norm(points_map[:, :PATH_XY_COLUMNS] - origin_map[:PATH_XY_COLUMNS], axis=1).astype(np.float32)
        for origin_map, points_map in iter_train_sweeps_map(run_dir, dataset_dir, width_sweep_stride)
    ]
    if not range_chunks:
        raise ValueError(f"no train lidar points survive the ego filter in {run_dir}")
    width_m = float(np.percentile(np.concatenate(range_chunks), range_percentile))
    # Step 3: pass 2, z of the returns inside the corridor, over every sweep. Streamed per sweep as float32
    # (a percentile of heights in metres needs no float64 precision), then one concatenate: ~45 M returns
    # = ~180 MB.
    path_tree = cKDTree(path_xy)
    corridor_z_chunks = []
    for _origin_map, points_map in iter_train_sweeps_map(run_dir, dataset_dir, 1):
        distance_to_path, _nearest = path_tree.query(points_map[:, :PATH_XY_COLUMNS])
        inside_corridor_z = points_map[distance_to_path <= width_m, Z_COLUMN_INDEX]
        if inside_corridor_z.shape[0] == 0:
            continue
        corridor_z_chunks.append(inside_corridor_z.astype(np.float32))
    if not corridor_z_chunks:
        raise ValueError(f"no train lidar return lies in the corridor of width {width_m} m in {run_dir}")
    # Step 4: z range = [t, 100 - t] percentiles of the pooled z (t = 0 is min/max).
    corridor_z = np.concatenate(corridor_z_chunks)
    z_lower = float(np.percentile(corridor_z, z_tail_percent))
    z_upper = float(np.percentile(corridor_z, PERCENTILE_MAX - z_tail_percent))
    return TrainPathCorridor(path_xy=path_xy, width_m=width_m, z_lower=z_lower, z_upper=z_upper,
                             path_positions_map=path_positions_map)


@transform(
    "T87",
    "Driven-path corridor crop: horizontal distance to the train sensor path <= width, z within the "
    "[t, 100-t] percentiles of the corridor lidar z (percentile trim), opacity >= min_opacity",
)
def corridor_crop_mask(
    means: np.ndarray,
    opacities: np.ndarray,
    path_xy: np.ndarray,
    width_m: float,
    z_lower: float,
    z_upper: float,
    min_opacity: float,
) -> np.ndarray:
    """Keep Gaussians near the driven path, inside the corridor z range and opaque enough.

    Keep iff min over path points q of ||mean_xy - q|| <= width_m (inclusive), z_lower <= mean_z <= z_upper
    (inclusive) and opacity >= min_opacity (inclusive). The nearest path point comes from a cKDTree
    (O((N + P) log P)).

    Args:
        means: (N, 3) finite Gaussian centres in the map frame.
        opacities: (N,) peak alpha in [0, 1] (activated, not logits).
        path_xy: (P, 2) finite map-frame path points, P >= 1.
        width_m: Corridor half-width in metres, > 0.
        z_lower: Lower z bound, finite.
        z_upper: Upper z bound, finite, >= z_lower.
        min_opacity: Opacity floor.

    Returns:
        (N,) bool keep mask.

    Raises:
        ValueError: On a wrong shape, non-finite value, opacity or ``min_opacity`` outside [0, 1],
            width <= 0, z_lower > z_upper or an empty path.
    """
    # Step 1: shared validation of the Gaussians and the floor.
    means, opacities = validate_gaussian_inputs(means, opacities, min_opacity)
    path_xy = np.asarray(path_xy, dtype=np.float64)
    # Step 2: path shape.
    if path_xy.ndim != 2 or path_xy.shape[1] != PATH_XY_COLUMNS or path_xy.shape[0] < 1:
        raise ValueError(f"path_xy must be (P>=1, {PATH_XY_COLUMNS}), got {path_xy.shape}")
    # Step 3: finiteness of the path and the scalar corridor parameters.
    for array_name, array in (
        ("path_xy", path_xy),
        ("width_m", np.float64(width_m)), ("z_lower", np.float64(z_lower)), ("z_upper", np.float64(z_upper)),
    ):
        if not np.isfinite(array).all():
            raise ValueError(f"{array_name} has non-finite values")
    # Step 4: a non-positive width or inverted z range would keep nothing without any warning.
    if width_m <= 0.0:
        raise ValueError(f"width_m must be > 0, got {width_m}")
    if z_lower > z_upper:
        raise ValueError(f"z bounds swapped: z_lower {z_lower} > z_upper {z_upper}")
    # Step 5: horizontal distance to the nearest path point, then the three inclusive tests.
    distance_to_path, _nearest = cKDTree(path_xy).query(means[:, :PATH_XY_COLUMNS])
    near_path = distance_to_path <= width_m
    within_z = (means[:, Z_COLUMN_INDEX] >= z_lower) & (means[:, Z_COLUMN_INDEX] <= z_upper)
    return near_path & within_z & (opacities >= min_opacity)
