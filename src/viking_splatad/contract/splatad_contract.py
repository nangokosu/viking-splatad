"""SplatAD input contract: named checks over a dataset loaded via the intermediate reader.

These checks are not plan transforms (no ``@transform``); they gate hand-off to the GPU box.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List

import numpy as np

from viking_splatad.dataset.schema import IntermediateDataset
from viking_splatad.dataset.splits import linspace_train_eval_split
from viking_splatad.dataset.writer import SchemaVersionError, read_dataset

# Elevation-mapping length must tile into groups of 8 channels (SplatAD lidar tiling).
ELEVATION_TILE_CHANNELS = 8
# Allowed lidar array widths: x,y,z,intensity,t (5) or plus channel (6).
ALLOWED_LIDAR_COLUMNS = (5, 6)
# Column layout of a lidar sweep.
INTENSITY_COLUMN, TIME_COLUMN, CHANNEL_COLUMN = 3, 4, 5
# Index of k4 inside the stored distortion vector [k1, k2, k3, k4, p1, p2].
DISTORTION_K4_INDEX = 3
# Relative tolerance when asking whether 2*pi/resolution is an integer (float round-off only).
AZIMUTH_INTEGER_RTOL = 1e-9
# World up axis required by SplatAD (z-up).
Z_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one named contract check."""

    name: str
    passed: bool
    message: str


@dataclass
class ContractReport:
    """Collection of check results."""

    results: List[CheckResult]

    @property
    def passed(self) -> bool:
        """True when every check passed."""
        return all(result.passed for result in self.results)

    def failed_check_names(self) -> List[str]:
        """Names of failed checks, in execution order."""
        return [result.name for result in self.results if not result.passed]

    def to_markdown(self) -> str:
        """Render the report as a Markdown table with an overall verdict."""
        lines = [
            f"# SplatAD contract report: {'PASS' if self.passed else 'FAIL'}",
            "",
            "| check | result | message |",
            "|---|---|---|",
        ]
        for result in self.results:
            verdict = "PASS" if result.passed else "FAIL"
            lines.append(f"| {result.name} | {verdict} | {result.message} |")
        return "\n".join(lines) + "\n"


def _result(name: str, problems: List[str]) -> CheckResult:
    """Build a CheckResult: passes iff ``problems`` is empty.

    Args:
        name: Check name.
        problems: Human-readable violation strings collected by the check.

    Returns:
        CheckResult with message "ok" or the problems joined by "; ".
    """
    return CheckResult(name, not problems, "ok" if not problems else "; ".join(problems))


def _lidar_streams(dataset: IntermediateDataset):
    """Select lidar streams.

    Args:
        dataset: Loaded intermediate dataset.

    Returns:
        List of streams whose ``kind`` is "lidar".
    """
    return [stream for stream in dataset.streams.values() if stream.kind == "lidar"]


def _is_plain_int(value) -> bool:
    """True for integers (numpy ints included) but not bools.

    Args:
        value: Any object.

    Returns:
        Whether ``value`` is an int/np.integer and not a bool (bool subclasses int).
    """
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def check_poses(dataset: IntermediateDataset) -> CheckResult:
    """Poses are [N,4,4] with N>=2, finite, rigid last row, finite after float32 cast.

    Enforces SplatAD's pose requirement: it consumes a float32 3x4 world-from-sensor
    block per frame and needs at least two poses per sensor to interpolate motion.

    Args:
        dataset: Loaded intermediate dataset; ``streams`` maps sensor name to a stream
            whose ``poses_world_sensor`` is the pose array under test.

    Returns:
        CheckResult "poses_finite_and_count": passed with message "ok" when every
        stream complies; otherwise failed with one message per violating sensor.
    """
    problems = []
    for name, stream in dataset.streams.items():
        poses = stream.poses_world_sensor
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            problems.append(f"{name}: pose shape {poses.shape} is not [N,4,4]")
            continue
        if poses.shape[0] < 2:
            problems.append(f"{name}: {poses.shape[0]} poses (<2)")
        # The 3x4 float32 pose block must stay finite after the cast.
        as_float32 = poses[:, :3, :].astype(np.float32)
        if not (np.isfinite(poses).all() and np.isfinite(as_float32).all()):
            problems.append(f"{name}: non-finite pose values")
        elif not np.allclose(poses[:, 3, :], [0.0, 0.0, 0.0, 1.0]):
            problems.append(f"{name}: last pose row is not [0,0,0,1]")
    return _result("poses_finite_and_count", problems)


def check_intrinsics_integer_size(dataset: IntermediateDataset) -> CheckResult:
    """Declared width/height are positive integers.

    Enforces SplatAD's requirement that camera image sizes are integer pixel counts.

    Args:
        dataset: Loaded intermediate dataset; ``cameras`` maps camera name to intrinsics
            with ``width`` and ``height`` fields.

    Returns:
        CheckResult "intrinsics_integer_size": passed ("ok") when every camera has
        positive integer width and height; otherwise failed listing offending cameras.
    """
    problems = [
        f"{name}: width/height {camera.width!r}x{camera.height!r} not positive ints"
        for name, camera in dataset.cameras.items()
        if not (_is_plain_int(camera.width) and _is_plain_int(camera.height)
                and camera.width > 0 and camera.height > 0)
    ]
    return _result("intrinsics_integer_size", problems)


def check_image_shape(dataset: IntermediateDataset) -> CheckResult:
    """Every stored image has shape (height, width, ...) equal to the declared size.

    Enforces SplatAD's requirement that images match the intrinsics' resolution, since
    the projection model uses the declared size.

    Args:
        dataset: Loaded intermediate dataset; camera-kind ``streams`` hold image arrays in
            ``frames`` and ``cameras`` holds the declared width/height per camera.

    Returns:
        CheckResult "image_shape_matches": passed ("ok") when all images match; otherwise
        failed listing each mismatching ``camera[frame]``. Cameras with non-integer sizes
        are skipped (reported by check_intrinsics_integer_size).
    """
    problems = []
    for name, stream in dataset.streams.items():
        if stream.kind != "camera" or name not in dataset.cameras:
            continue
        camera = dataset.cameras[name]
        # Non-integer sizes are reported by intrinsics_integer_size instead.
        if not (_is_plain_int(camera.width) and _is_plain_int(camera.height)):
            continue
        for frame_number, image in enumerate(stream.frames):
            if image.shape[:2] != (camera.height, camera.width):
                problems.append(
                    f"{name}[{frame_number}]: image {image.shape[1]}x{image.shape[0]} "
                    f"!= declared {camera.width}x{camera.height}"
                )
    return _result("image_shape_matches", problems)


def check_distortion_k4(dataset: IntermediateDataset) -> CheckResult:
    """Distortion is absent or has k4 == 0.

    Enforces SplatAD's camera model, which supports no fourth radial distortion term.

    Args:
        dataset: Loaded intermediate dataset; ``cameras`` maps camera name to intrinsics
            whose optional ``distortion`` is [k1, k2, k3, k4, p1, p2].

    Returns:
        CheckResult "distortion_k4_zero": passed ("ok") when no camera has nonzero k4;
        otherwise failed listing each camera and its k4 value.
    """
    problems = [
        f"{name}: k4={camera.distortion[DISTORTION_K4_INDEX]}"
        for name, camera in dataset.cameras.items()
        if camera.distortion is not None
        and len(camera.distortion) > DISTORTION_K4_INDEX
        and camera.distortion[DISTORTION_K4_INDEX] != 0.0
    ]
    return _result("distortion_k4_zero", problems)


def check_times(dataset: IntermediateDataset) -> CheckResult:
    """Pose times are float64, finite and strictly increasing per sensor.

    Enforces SplatAD's requirement for float64 timestamps (float32 loses sub-ms
    precision at epoch scale) that are strictly ordered for pose interpolation.

    Args:
        dataset: Loaded intermediate dataset; ``streams`` maps sensor name to a stream
            whose ``times`` array holds one timestamp per pose.

    Returns:
        CheckResult "times_strictly_increasing": passed ("ok") when all streams comply;
        otherwise failed with the first violated property (dtype, finiteness, ordering)
        per offending sensor.
    """
    problems = []
    for name, stream in dataset.streams.items():
        times = stream.times
        if times.dtype != np.float64:
            problems.append(f"{name}: times dtype {times.dtype}")
        elif not np.isfinite(times).all():
            problems.append(f"{name}: non-finite times")
        elif not np.all(np.diff(times) > 0.0):
            problems.append(f"{name}: times not strictly increasing")
    return _result("times_strictly_increasing", problems)


def check_index_contiguity(dataset: IntermediateDataset) -> CheckResult:
    """Camera/lidar frame indices are exactly 0..n-1.

    Enforces SplatAD's requirement that frame indices are contiguous from zero, so they
    can index poses, times and frames directly.

    Args:
        dataset: Loaded intermediate dataset; ``streams`` maps sensor name to a stream
            whose ``frame_indices`` array is under test.

    Returns:
        CheckResult "sensor_index_contiguity": passed ("ok") when every stream's indices
        equal arange(n); otherwise failed listing offending sensors.
    """
    problems = [
        f"{name}: frame indices are not 0..{stream.frame_indices.size - 1}"
        for name, stream in dataset.streams.items()
        if not np.array_equal(stream.frame_indices, np.arange(stream.frame_indices.size))
    ]
    return _result("sensor_index_contiguity", problems)


def check_lidar_format(dataset: IntermediateDataset) -> CheckResult:
    """Each sweep is float32 with shape [n,5] or [n,6].

    Enforces SplatAD's lidar array layout: x, y, z, intensity, time, plus an optional
    channel column.

    Args:
        dataset: Loaded intermediate dataset; lidar-kind ``streams`` hold one point
            array per sweep in ``frames``.

    Returns:
        CheckResult "lidar_array_format": passed ("ok") when all sweeps match; otherwise
        failed listing each ``sensor[frame]`` with its dtype and shape.
    """
    problems = []
    for stream in _lidar_streams(dataset):
        for frame_number, sweep in enumerate(stream.frames):
            if sweep.dtype != np.float32 or sweep.ndim != 2 or sweep.shape[1] not in ALLOWED_LIDAR_COLUMNS:
                problems.append(f"{stream.sensor_name}[{frame_number}]: {sweep.dtype} {sweep.shape}")
    return _result("lidar_array_format", problems)


def check_intensity(dataset: IntermediateDataset) -> CheckResult:
    """Intensity column lies in [0, 1].

    Enforces SplatAD's requirement for normalized lidar intensity, which it regresses
    directly.

    Args:
        dataset: Loaded intermediate dataset; lidar-kind ``streams`` hold sweeps whose
            column ``INTENSITY_COLUMN`` is under test.

    Returns:
        CheckResult "lidar_intensity_range": passed ("ok") when every sweep's intensities
        are finite and within [0, 1]; otherwise failed listing offending sweeps.
    """
    problems = []
    for stream in _lidar_streams(dataset):
        for frame_number, sweep in enumerate(stream.frames):
            intensity = sweep[:, INTENSITY_COLUMN].astype(np.float64)
            if not (np.isfinite(intensity).all() and intensity.min() >= 0.0 and intensity.max() <= 1.0):
                problems.append(f"{stream.sensor_name}[{frame_number}]: intensity outside [0,1]")
    return _result("lidar_intensity_range", problems)


def check_point_time(dataset: IntermediateDataset) -> CheckResult:
    """Per-point time offset lies within +/- one sweep period.

    Enforces SplatAD's requirement that per-point times are offsets relative to the
    sweep timestamp (for rolling-shutter compensation), not absolute times.

    Args:
        dataset: Loaded intermediate dataset; lidar-kind ``streams`` hold sweeps whose
            column ``TIME_COLUMN`` is under test, and ``lidar.sweep_period`` is the bound.

    Returns:
        CheckResult "lidar_point_time_range": passed ("ok") when max |t| <= sweep period
        for every sweep; otherwise failed listing offending sweeps.
    """
    problems = []
    for stream in _lidar_streams(dataset):
        for frame_number, sweep in enumerate(stream.frames):
            if np.abs(sweep[:, TIME_COLUMN]).max() > dataset.lidar.sweep_period:
                problems.append(f"{stream.sensor_name}[{frame_number}]: |t| > sweep period")
    return _result("lidar_point_time_range", problems)


def check_channels(dataset: IntermediateDataset) -> CheckResult:
    """When a channel column exists, its values are integers that are mapping keys.

    Enforces SplatAD's requirement that each point's channel indexes the lidar
    elevation mapping.

    Args:
        dataset: Loaded intermediate dataset; lidar-kind ``streams`` hold sweeps (sweeps
            without ``CHANNEL_COLUMN`` are skipped) and ``lidar.elevation_mapping`` provides
            the valid channel keys.

    Returns:
        CheckResult "lidar_channel_in_mapping": passed ("ok") when all channels are
        integral and valid keys; otherwise failed listing offending sweeps.
    """
    problems = []
    valid_channels = set(dataset.lidar.elevation_mapping)
    for stream in _lidar_streams(dataset):
        for frame_number, sweep in enumerate(stream.frames):
            if sweep.shape[1] <= CHANNEL_COLUMN:
                continue
            channels = sweep[:, CHANNEL_COLUMN]
            is_integral = np.all(channels == np.rint(channels))
            if not is_integral or not set(np.rint(channels).astype(int)) <= valid_channels:
                problems.append(f"{stream.sensor_name}[{frame_number}]: channel not in mapping keys")
    return _result("lidar_channel_in_mapping", problems)


def check_elevation_mapping(dataset: IntermediateDataset) -> CheckResult:
    """Elevation mapping length is a positive multiple of ELEVATION_TILE_CHANNELS (8).

    Enforces SplatAD's lidar tiling, which groups channels in blocks of 8.

    Args:
        dataset: Loaded intermediate dataset; ``lidar.elevation_mapping`` is the
            per-channel elevation table whose length is tested.

    Returns:
        CheckResult "elevation_mapping_multiple_of_8": passed ("ok") when the length is
        positive and divisible by 8; otherwise failed stating the length.
    """
    length = len(dataset.lidar.elevation_mapping)
    problems = [] if length > 0 and length % ELEVATION_TILE_CHANNELS == 0 else [
        f"length {length} not a positive multiple of {ELEVATION_TILE_CHANNELS}"
    ]
    return _result("elevation_mapping_multiple_of_8", problems)


def check_azimuth_tiles(dataset: IntermediateDataset) -> CheckResult:
    """Azimuth bins = 2*pi / resolution is an integer divisible by the tile width.

    Enforces SplatAD's lidar azimuth tiling: a full turn must split into whole bins
    that group evenly into tiles.

    Args:
        dataset: Loaded intermediate dataset; ``lidar`` provides ``azimuth_resolution``
            (radians per bin) and ``azimuth_tile_width`` (bins per tile).

    Returns:
        CheckResult "azimuth_tile_integral": passed ("ok") when the bin count is an
        integer (within AZIMUTH_INTEGER_RTOL) divisible by the tile width; otherwise
        failed describing which condition broke.
    """
    lidar = dataset.lidar
    problems = []
    # Number of azimuth bins around a full turn: B = 2*pi / resolution.
    bin_count = 2.0 * np.pi / lidar.azimuth_resolution
    # Integrality is checked first; only then is divisibility by the tile width meaningful.
    if not np.isclose(bin_count, np.rint(bin_count), rtol=AZIMUTH_INTEGER_RTOL, atol=0.0):
        problems.append(f"2*pi/resolution = {bin_count} is not an integer")
    elif int(np.rint(bin_count)) % lidar.azimuth_tile_width != 0:
        problems.append(f"{int(np.rint(bin_count))} bins not divisible by tile width {lidar.azimuth_tile_width}")
    return _result("azimuth_tile_integral", problems)


def check_splits(dataset: IntermediateDataset) -> CheckResult:
    """Train and eval splits are non-empty for every camera/lidar sensor.

    Enforces SplatAD's need for both training and evaluation frames per sensor, using
    the same linspace split the converter applies.

    Args:
        dataset: Loaded intermediate dataset; ``streams`` give per-sensor frame counts
            and ``session.train_fraction`` sets the split ratio.

    Returns:
        CheckResult "splits_non_empty": passed ("ok") when every sensor has at least one
        train and one eval frame; otherwise failed listing sensors with their counts.
    """
    problems = []
    for name, stream in dataset.streams.items():
        train_indices, eval_indices = linspace_train_eval_split(
            stream.frame_indices.size, dataset.session.train_fraction
        )
        if train_indices.size == 0 or eval_indices.size == 0:
            problems.append(f"{name}: train={train_indices.size} eval={eval_indices.size}")
    return _result("splits_non_empty", problems)


def check_world_up(dataset: IntermediateDataset) -> CheckResult:
    """World up axis is +z.

    Enforces SplatAD's z-up world-frame convention.

    Args:
        dataset: Loaded intermediate dataset; ``session.world_up`` is the 3-vector tested
            against ``Z_UP``.

    Returns:
        CheckResult "world_up_is_z": passed ("ok") when world_up is close to [0, 0, 1];
        otherwise failed showing the actual vector.
    """
    problems = [] if np.allclose(dataset.session.world_up, Z_UP) else [
        f"world_up {dataset.session.world_up.tolist()} != +z"
    ]
    return _result("world_up_is_z", problems)


# Ordered list of checks executed by validate_dataset.
CONTRACT_CHECKS: List[Callable[[IntermediateDataset], CheckResult]] = [
    check_poses, check_intrinsics_integer_size, check_image_shape, check_distortion_k4,
    check_times, check_index_contiguity, check_lidar_format, check_intensity,
    check_point_time, check_channels, check_elevation_mapping, check_azimuth_tiles,
    check_splits, check_world_up,
]


def validate_dataset(root: Path) -> ContractReport:
    """Load an intermediate dataset and run every contract check.

    Args:
        root: Dataset directory written by the intermediate writer.

    Returns:
        ContractReport; a dataset that cannot be loaded yields one failing
        ``dataset_loadable`` check rather than an exception.
    """
    try:
        dataset = read_dataset(root)
    except (SchemaVersionError, OSError, KeyError, ValueError) as load_error:
        return ContractReport([CheckResult("dataset_loadable", False, str(load_error))])
    return ContractReport([check(dataset) for check in CONTRACT_CHECKS])
