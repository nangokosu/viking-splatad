"""Exact train/test frame membership of one SplatAD run, rebuilt without torch (T58, T59, T60, T62).

SplatAD's frame selection has no randomness. ``ADDataParser._generate_dataparser_outputs``
(neurad-studio ad_dataparser.py) does, in order:
    1. ``_filter_based_on_time``: keep every camera image and lidar sweep whose time lies in
       [t_min + D * start_fraction, t_min + D * end_fraction] (inclusive), where t_min and D are
       the min and span of the camera and lidar times together (T58);
    2. ``_get_linspaced_indices``: per sensor, independently, train = ceil(n * fraction) evenly
       spaced frames of the n surviving ones; test ("eval") = the rest (T59).
       ``VikingHillDataParser`` with ``test_block_count = K`` replaces step 2 by K contiguous test
       time blocks shared by every sensor (T60); config.yml records K.
    3. With ``test_block_core_fraction = c < 1`` (``VikingHillDataParser``), only the central
       c of each block stays test; the block's outer frames become ``guard`` frames, in neither
       split (T62): never trained on, never rendered. config.yml records c.
The training seed (``--machine.seed``) only orders sampling during training; it never changes
which frames are train or test. The only seeded step in split selection, ``max_eval_frames``
(``torch.manual_seed(123)``), is None for ns-train and forced to None by ns-render and
scripts/08, so it is refused here rather than emulated.

Two recorded values are NOT the run's split and must not be used instead:
    * ``config.yml`` ``train_split_fraction`` is saved before the dataparser runs, so it holds the
      class default (0.5); the real fraction is the manifest's ``session.train_fraction`` unless
      ``train_split_fraction_override`` is set (``resolve_train_fraction``, T27).
    * ``manifest["pipeline"]["splits"]`` is computed over the whole recording, ignoring the window.
"""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from viking_splatad.dataset.splits import linspace_train_eval_split
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.registry import transform
from viking_splatad.splatad.viking_conversions import DatasetView

# Sensor kinds the dataparser turns into SplatAD sensors (radar never is: CLAUDE.md rule 5).
SPLIT_SENSOR_KINDS = ("camera", "lidar")
# Names of the two splits, as used by ns-render and scripts/08 output folders (renders/<split>/).
TRAIN_SPLIT_NAME = "train"
TEST_SPLIT_NAME = "test"
# Frames inside a T60 test block but outside its core (T62): held out of training, not tested.
GUARD_SPLIT_NAME = "guard"
# Fractions meaning "whole recording": neurad skips the time filter for exactly these values.
FULL_RECORDING_START_FRACTION = 0.0
FULL_RECORDING_END_FRACTION = 1.0
# neurad's special case: at exactly this fraction, train and eval are both every frame.
ALL_FRAMES_TRAIN_FRACTION = 1.0
# T62 core fraction meaning "the whole block is test" (no guard frames; the T60 split as is).
WHOLE_BLOCK_CORE_FRACTION = 1.0


@dataclass(frozen=True)
class SplitFrame:
    """One frame of one sensor with its split membership.

    Attributes:
        sensor: Sensor name (e.g. ``ids_camera``).
        kind: ``camera`` or ``lidar``.
        split: ``train`` or ``test``.
        frame_index: Index in the frame file name ``<frame_index>_<timestamp_ns>.<ext>``.
        timestamp_ns: Stamp in the frame file name (identifies the frame across datasets).
        time_s: ``poses.npz`` time [s] the dataparser uses for this frame.
        file: Frame path relative to the dataset root (e.g. ``camera/ids_camera/000012_<ns>.jpg``).
    """

    sensor: str
    kind: str
    split: str
    frame_index: int
    timestamp_ns: int
    time_s: float
    file: str


@transform("T58", "neurad-studio time-window filter over camera+lidar times (dataset start/end fraction)")
def time_window_masks(
    sensor_times: Dict[str, np.ndarray], start_fraction: float, end_fraction: float
) -> Tuple[Dict[str, np.ndarray], Optional[Tuple[float, float]]]:
    """Which frames of each sensor survive SplatAD's time window.

    Same float64 operations in the same order as ``_filter_based_on_time``:
        t_end = t_min + D * end_fraction;  t_start = t_min + D * start_fraction  (D = t_max - t_min)
    so frames exactly on a bound get the same verdict. Bounds are inclusive.

    Args:
        sensor_times: sensor name -> float64 [N] times of every camera and lidar frame.
        start_fraction: ``--dataset-start-fraction``.
        end_fraction: ``--dataset-end-fraction``.

    Returns:
        (masks, bounds): sensor name -> bool [N] keep-mask, and (t_start, t_end) seconds, or None
        when the fractions are (0, 1) and neurad skips the filter (every frame kept).
    """
    # Step 1: neurad returns early for the whole recording; mirror that (no bound comparisons).
    if start_fraction == FULL_RECORDING_START_FRACTION and end_fraction == FULL_RECORDING_END_FRACTION:
        return {name: np.ones(len(times), dtype=bool) for name, times in sensor_times.items()}, None
    # Step 2: min and span over all camera + lidar times together (torch.cat of both sets).
    all_times = np.concatenate([np.asarray(times, dtype=np.float64) for times in sensor_times.values()])
    end_time = float(all_times.max())
    start_time = float(all_times.min())
    duration = end_time - start_time
    # Step 3: bounds, in neurad's operation order (end first, from the unshifted start).
    end_time = start_time + duration * end_fraction
    start_time += duration * start_fraction
    # Step 4: inclusive mask per sensor (_filter_sensordata_on_time).
    masks = {
        name: (np.asarray(times, dtype=np.float64) >= start_time) & (np.asarray(times, dtype=np.float64) <= end_time)
        for name, times in sensor_times.items()
    }
    return masks, (start_time, end_time)


@transform("T59", "neurad-studio per-sensor LINSPACE train/eval indices, incl. fraction 1.0 case")
def neurad_train_eval_indices(
    num_frames: int, train_fraction: float, max_eval_frames: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Train and eval positions among one sensor's windowed frames (``_get_linspaced_indices``).

    For fraction < 1 this is T21: train = int(linspace(0, n-1, ceil(n * fraction))). With
    ceil(n * f) <= n points the spacing (n-1)/(count-1) is >= 1, so truncation never collides.
    neurad differs from T21 at fraction == 1.0: train AND eval are every frame (eval is for
    viewing only), and for n == 0 both are empty.

    Args:
        num_frames: Frames of this sensor left after the time window.
        train_fraction: Resolved train fraction (manifest or override).
        max_eval_frames: Must be None (the seeded random eval subset is never used by our runs).

    Returns:
        (train_positions, eval_positions): sorted int64 arrays of positions in [0, num_frames).

    Raises:
        ValueError: If ``max_eval_frames`` is set (torch.randperm subset not reproduced here).
    """
    # Step 1: refuse the one seeded branch instead of silently returning a different split.
    if max_eval_frames is not None:
        raise ValueError("max_eval_frames is set: neurad draws a torch.randperm(seed 123) eval subset")
    # Step 2: empty sensor -> empty split.
    if num_frames == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    # Step 3: neurad's full-dataset case: both splits are every frame.
    if train_fraction == ALL_FRAMES_TRAIN_FRACTION:
        all_positions = np.arange(num_frames, dtype=np.int64)
        return all_positions, all_positions.copy()
    # Step 4: interleaved LINSPACE split, identical to T21 for fraction < 1.
    train_positions, eval_positions = linspace_train_eval_split(num_frames, train_fraction)
    return train_positions.astype(np.int64), eval_positions.astype(np.int64)


@transform("T60", "middle-block train/test split: K contiguous test time blocks shared by every sensor")
def middle_block_train_eval_indices(
    sensor_times: Dict[str, np.ndarray], train_fraction: float, test_block_count: int
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], List[Tuple[float, float]]]:
    """Train and test positions per sensor when the test set is K contiguous blocks of time.

    Replaces neurad's per-sensor LINSPACE split (T59), whose test frames sit 0.1 s from a train
    frame and are chosen independently per sensor. Here the blocks are the same time intervals
    for every sensor, so a test camera image and the lidar sweep taken with it are both held out.

    Geometry, over the windowed times of all sensors together (t_lo = min, D = max - min):
        block length  L   = D * (1 - f) / K                      (K blocks hold the test fraction)
        block centres c_k = t_lo + D * k / (K + 1),  k = 1..K    (evenly spaced, away from the ends)
        block k            = [c_k - L/2, c_k + L/2)              (half-open: no frame in two blocks)
    Train stretches: D / (K+1) - L/2 before the first and after the last block, D / (K+1) - L
    between blocks (total D - K*L = D*f). Blocks are disjoint and separated by train frames iff
    L < D / (K + 1), i.e. f > 1 / (K + 1). At a constant frame rate each sensor's test share is ~(1 - f).
    f == 1.0 follows neurad's convention (train and test are both every frame; test is for viewing).
    Cost O(N * K) for N frames in total.

    Args:
        sensor_times: sensor name -> float64 [N] times of that sensor's windowed frames.
        train_fraction: f in (1 / (K + 1), 1].
        test_block_count: K >= 1.

    Returns:
        (positions, blocks): sensor name -> (train_positions, test_positions), sorted int64 positions
        into that sensor's array; and the K (start, end) block bounds in seconds ([] when f == 1.0
        or there are no frames).

    Raises:
        ValueError: If K is not a positive int or f is outside (1 / (K + 1), 1].
    """
    # Step 1: validate (bool is an int subclass; refuse it like any non-integer).
    if isinstance(test_block_count, bool) or not isinstance(test_block_count, int) or test_block_count < 1:
        raise ValueError(f"test_block_count must be a positive int, got {test_block_count!r}")
    minimum_fraction = 1.0 / (test_block_count + 1)
    if not minimum_fraction < train_fraction <= ALL_FRAMES_TRAIN_FRACTION:
        raise ValueError(f"train_fraction {train_fraction} must lie in ({minimum_fraction}, 1] for "
                         f"{test_block_count} test blocks (blocks would touch or overlap)")
    times_by_sensor = {name: np.asarray(times, dtype=np.float64) for name, times in sensor_times.items()}
    all_times = np.concatenate(list(times_by_sensor.values())) if times_by_sensor else np.empty(0)
    # Step 2: neurad's full-dataset case (every frame in both splits), and the no-frame case
    # (every sensor empty, so "every frame" is the empty set): no blocks.
    if train_fraction == ALL_FRAMES_TRAIN_FRACTION or all_times.size == 0:
        positions = {}
        for name, times in times_by_sensor.items():
            all_positions = np.arange(len(times), dtype=np.int64)
            positions[name] = (all_positions, all_positions.copy())
        return positions, []
    # Step 3: block bounds from the shared span (formulas in the docstring).
    span_start = float(all_times.min())
    span_duration = float(all_times.max()) - span_start
    block_length = span_duration * (1.0 - train_fraction) / test_block_count
    block_bounds = []
    for block_number in range(1, test_block_count + 1):
        block_centre = span_start + span_duration * block_number / (test_block_count + 1)
        block_bounds.append((block_centre - block_length / 2.0, block_centre + block_length / 2.0))
    # Step 4: a frame is test iff its time falls in any half-open block.
    positions = {}
    for name, times in times_by_sensor.items():
        test_mask = np.zeros(len(times), dtype=bool)
        for block_start, block_end in block_bounds:
            test_mask |= (times >= block_start) & (times < block_end)
        positions[name] = (np.flatnonzero(~test_mask).astype(np.int64), np.flatnonzero(test_mask).astype(np.int64))
    return positions, block_bounds


@transform("T62", "test block cores: the central fraction of each T60 test block stays test, the rest is guard")
def restrict_test_to_block_cores(
    sensor_times: Dict[str, np.ndarray],
    positions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    block_bounds: List[Tuple[float, float]],
    core_fraction: float,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, np.ndarray], List[Tuple[float, float]]]:
    """Keep as test only the frames in the central part of each T60 block; the rest become guard.

    Rendering every test frame of a full recording is costly to store and copy back; the core of
    each block is the part farthest from any training frame, so it is the most held-out sample.
    Train positions are never changed: guard frames come only out of the test set.

    Geometry, per block [a, b) of length L = b - a and centre m = (a + b) / 2:
        core = [m - c * L / 2, m + c * L / 2)      (half-open, like the block; inside it for c <= 1)
    A test frame is kept iff its time falls in its block's core; the guard is the rest of the block,
    (1 - c) * L / 2 on each side. c == 1 returns the inputs unchanged (no rounding of the bounds).
    Cost O(N * K) for N test frames and K blocks.

    Args:
        sensor_times: sensor name -> float64 [N] windowed times (the array T60 was given).
        positions: T60 output: sensor name -> (train_positions, test_positions) into those times.
        block_bounds: T60 output: the K (start, end) block bounds in seconds.
        core_fraction: c in (0, 1].

    Returns:
        (positions, guard_positions, cores): positions with test reduced to the core frames (train
        untouched); sensor name -> sorted int64 guard positions; the K (start, end) core bounds.

    Raises:
        ValueError: If c is outside (0, 1].
    """
    # Step 1: validate; c == 0 would leave no test frame, c > 1 would reach into train frames.
    if not 0.0 < core_fraction <= WHOLE_BLOCK_CORE_FRACTION:
        raise ValueError(f"test_block_core_fraction {core_fraction} must lie in (0, 1]")
    # Step 2: the whole block is the core: nothing to move.
    if core_fraction == WHOLE_BLOCK_CORE_FRACTION:
        return dict(positions), {name: np.empty(0, dtype=np.int64) for name in positions}, list(block_bounds)
    # Step 3: core bounds around each block centre (formula in the docstring).
    core_bounds = []
    for block_start, block_end in block_bounds:
        block_centre = (block_start + block_end) / 2.0
        core_half_length = core_fraction * (block_end - block_start) / 2.0
        core_bounds.append((block_centre - core_half_length, block_centre + core_half_length))
    # Step 4: split each sensor's test positions into core and guard by time.
    core_positions_by_sensor, guard_positions_by_sensor = {}, {}
    for name, (train_positions, test_positions) in positions.items():
        test_times = np.asarray(sensor_times[name], dtype=np.float64)[test_positions]
        in_core_mask = np.zeros(len(test_positions), dtype=bool)
        for core_start, core_end in core_bounds:
            in_core_mask |= (test_times >= core_start) & (test_times < core_end)
        core_positions_by_sensor[name] = (train_positions, test_positions[in_core_mask].astype(np.int64))
        guard_positions_by_sensor[name] = test_positions[~in_core_mask].astype(np.int64)
    return core_positions_by_sensor, guard_positions_by_sensor, core_bounds


def resolve_run_train_fraction(manifest: dict, train_fraction_override: Optional[float]) -> Tuple[float, str]:
    """The fraction the dataparser actually used, and where it came from.

    Mirrors ``resolve_train_fraction`` (T27) but also names the source for the record.

    Args:
        manifest: Parsed dataset ``manifest.json``.
        train_fraction_override: ``train_split_fraction_override`` of the run, or None.

    Returns:
        (fraction, source) with source ``override`` or ``manifest.session.train_fraction``.
    """
    if train_fraction_override is not None:
        return float(train_fraction_override), "override"
    return float(manifest["session"]["train_fraction"]), "manifest.session.train_fraction"


def build_run_split(
    view: DatasetView,
    start_fraction: float,
    end_fraction: float,
    train_fraction: float,
    max_eval_frames: Optional[int] = None,
    test_block_count: Optional[int] = None,
    test_block_core_fraction: float = WHOLE_BLOCK_CORE_FRACTION,
) -> Tuple[List[SplitFrame], Optional[Tuple[float, float]]]:
    """Every camera/lidar frame of a run with its split, in the dataparser's order.

    Args:
        view: ``open_dataset_view`` of the run's dataset (same file <-> time pairing as the dataparser).
        start_fraction, end_fraction: The run's time window fractions.
        train_fraction: Resolved train fraction.
        max_eval_frames: Must be None (see ``neurad_train_eval_indices``).
        test_block_count: None = neurad's per-sensor LINSPACE split (T59); K = K middle test
            blocks shared by every sensor (T60, the dataparser's ``test_block_count``).
        test_block_core_fraction: c of T62 (the dataparser's ``test_block_core_fraction``): only
            the central c of each block is test, its other frames get split ``guard``. 1.0 = no guard.

    Returns:
        (frames, bounds): frames sorted by (sensor order in the manifest, split, frame index); and
        the window bounds from ``time_window_masks``.

    Raises:
        ValueError: If a sensor's frame files and times differ in count, ``max_eval_frames``
            is set, or c < 1 without test blocks.
    """
    frames, window_bounds, _, _ = build_run_split_with_cores(
        view, start_fraction, end_fraction, train_fraction, max_eval_frames, test_block_count,
        test_block_core_fraction)
    return frames, window_bounds


def build_run_split_with_blocks(
    view: DatasetView,
    start_fraction: float,
    end_fraction: float,
    train_fraction: float,
    max_eval_frames: Optional[int] = None,
    test_block_count: Optional[int] = None,
    test_block_core_fraction: float = WHOLE_BLOCK_CORE_FRACTION,
) -> Tuple[List[SplitFrame], Optional[Tuple[float, float]], List[Tuple[float, float]]]:
    """``build_run_split`` that also returns the T60 test blocks (for the run's split record).

    Args:
        view, start_fraction, end_fraction, train_fraction, max_eval_frames, test_block_count,
        test_block_core_fraction: As in ``build_run_split``.

    Returns:
        (frames, window bounds, test blocks): test blocks are the (start, end) seconds of each
        T60 block, [] for the LINSPACE split.

    Raises:
        ValueError: As in ``build_run_split``.
    """
    frames, window_bounds, test_blocks, _ = build_run_split_with_cores(
        view, start_fraction, end_fraction, train_fraction, max_eval_frames, test_block_count,
        test_block_core_fraction)
    return frames, window_bounds, test_blocks


def build_run_split_with_cores(
    view: DatasetView,
    start_fraction: float,
    end_fraction: float,
    train_fraction: float,
    max_eval_frames: Optional[int] = None,
    test_block_count: Optional[int] = None,
    test_block_core_fraction: float = WHOLE_BLOCK_CORE_FRACTION,
) -> Tuple[List[SplitFrame], Optional[Tuple[float, float]], List[Tuple[float, float]], List[Tuple[float, float]]]:
    """``build_run_split`` that also returns the T60 test blocks and their T62 cores.

    Args:
        view, start_fraction, end_fraction, train_fraction, max_eval_frames, test_block_count,
        test_block_core_fraction: As in ``build_run_split``.

    Returns:
        (frames, window bounds, test blocks, test cores): blocks and cores are (start, end)
        seconds, both [] for the LINSPACE split; cores equal blocks when c == 1.

    Raises:
        ValueError: As in ``build_run_split``.
    """
    # Step 1: the SplatAD sensors, in manifest order (the dataparser's camera/lidar tuples).
    split_sensors = [sensor for sensor in view.manifest["sensors"] if sensor["kind"] in SPLIT_SENSOR_KINDS]
    sensor_times = {sensor["name"]: view.sensor_times[sensor["name"]] for sensor in split_sensors}
    # Step 2: time window over all of them together.
    window_masks, window_bounds = time_window_masks(sensor_times, start_fraction, end_fraction)
    # Step 3: files pair positionally with times, as in build_sensor_arrays; keep the windowed ones.
    kept_positions_by_sensor = {}
    for sensor in split_sensors:
        sensor_name = sensor["name"]
        frame_count = len(view.sensor_frame_paths[sensor_name])
        if frame_count != len(sensor_times[sensor_name]):
            raise ValueError(f"{sensor_name}: {frame_count} frame files but {len(sensor_times[sensor_name])} times")
        kept_positions_by_sensor[sensor_name] = np.flatnonzero(window_masks[sensor_name])
    # Step 4: split positions, relative to each sensor's windowed frames.
    test_blocks: List[Tuple[float, float]] = []
    test_cores: List[Tuple[float, float]] = []
    guard_positions_by_sensor: Dict[str, np.ndarray] = {}
    if test_block_count is None:
        # Cores are parts of T60 blocks; without blocks there is nothing to take a core of.
        if test_block_core_fraction != WHOLE_BLOCK_CORE_FRACTION:
            raise ValueError("test_block_core_fraction < 1 needs the block split (test_block_count)")
        # T59: each sensor on its own (neurad's LINSPACE).
        split_positions_by_sensor = {
            sensor_name: neurad_train_eval_indices(len(kept_positions), train_fraction, max_eval_frames)
            for sensor_name, kept_positions in kept_positions_by_sensor.items()
        }
    else:
        # T60: one set of time blocks over all sensors' windowed times together.
        if max_eval_frames is not None:
            raise ValueError("max_eval_frames is set: the block split never subsamples its test frames")
        windowed_times = {sensor_name: sensor_times[sensor_name][kept_positions]
                          for sensor_name, kept_positions in kept_positions_by_sensor.items()}
        split_positions_by_sensor, test_blocks = middle_block_train_eval_indices(
            windowed_times, train_fraction, test_block_count)
        # T62: only each block's core stays test; the rest of the block is guard.
        split_positions_by_sensor, guard_positions_by_sensor, test_cores = restrict_test_to_block_cores(
            windowed_times, split_positions_by_sensor, test_blocks, test_block_core_fraction)

    frames: List[SplitFrame] = []
    for sensor in split_sensors:
        sensor_name, sensor_kind = sensor["name"], sensor["kind"]
        frame_paths = view.sensor_frame_paths[sensor_name]
        times = sensor_times[sensor_name]
        kept_positions = kept_positions_by_sensor[sensor_name]
        train_positions, eval_positions = split_positions_by_sensor[sensor_name]
        guard_positions = guard_positions_by_sensor.get(sensor_name, np.empty(0, dtype=np.int64))
        for split_name, split_positions in ((TRAIN_SPLIT_NAME, train_positions), (TEST_SPLIT_NAME, eval_positions),
                                            (GUARD_SPLIT_NAME, guard_positions)):
            for windowed_position in split_positions:
                recording_position = int(kept_positions[windowed_position])
                frame_path = frame_paths[recording_position]
                name_match = FRAME_FILE_PATTERN.match(frame_path.name)
                frames.append(
                    SplitFrame(
                        sensor=sensor_name,
                        kind=sensor_kind,
                        split=split_name,
                        frame_index=int(name_match.group(1)),
                        timestamp_ns=int(name_match.group(2)),
                        time_s=float(times[recording_position]),
                        file=frame_path.relative_to(view.root).as_posix(),
                    )
                )
    return frames, window_bounds, test_blocks, test_cores


def split_digest(frames: List[SplitFrame]) -> str:
    """Order-independent fingerprint of split membership, to compare runs or users at a glance.

    Hashes the sorted "sensor<TAB>split<TAB>frame file name" lines (file names carry index and ns
    stamp, so two datasets with the same frames give the same digest).

    Args:
        frames: Output of ``build_run_split``.

    Returns:
        Hex SHA-256.
    """
    membership_lines = sorted(f"{frame.sensor}\t{frame.split}\t{Path(frame.file).name}" for frame in frames)
    return hashlib.sha256("\n".join(membership_lines).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    """SHA-256 of a file's bytes (identifies the exact dataset a split was built from).

    Args:
        path: File to hash.

    Returns:
        Hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as file_handle:
        # 1 MiB blocks: constant memory for any file size.
        for block in iter(lambda: file_handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_with_rendered_stems(
    frames: List[SplitFrame], rendered_stems: Dict[Tuple[str, str], List[str]]
) -> Dict[str, dict]:
    """Check the rebuilt split against the file names SplatAD's renders actually produced.

    ns-render names camera renders ``<idx>_<ts_ns>.jpg`` and scripts/08 names lidar renders
    ``<idx>_<ts_ns>.npz``, each in ``renders/<split>/``; their stems must equal the frame stems.

    Args:
        frames: Output of ``build_run_split``.
        rendered_stems: (kind, split) -> stems found on disk, for the folders that exist.

    Returns:
        "<kind>/<split>" -> {"match": bool, "missing_from_renders": [...], "unexpected_in_renders": [...]}.
    """
    report = {}
    for (sensor_kind, split_name), stems_on_disk in sorted(rendered_stems.items()):
        expected_stems = {
            Path(frame.file).stem for frame in frames if frame.kind == sensor_kind and frame.split == split_name
        }
        found_stems = set(stems_on_disk)
        report[f"{sensor_kind}/{split_name}"] = {
            "match": expected_stems == found_stems,
            "missing_from_renders": sorted(expected_stems - found_stems),
            "unexpected_in_renders": sorted(found_stems - expected_stems),
        }
    return report

