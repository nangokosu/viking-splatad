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
    """Build a CheckResult: passes iff ``problems`` is empty."""
    return CheckResult(name, not problems, "ok" if not problems else "; ".join(problems))


def _lidar_streams(dataset: IntermediateDataset):
    """Lidar streams of the dataset."""
    return [stream for stream in dataset.streams.values() if stream.kind == "lidar"]


def _is_plain_int(value) -> bool:
    """True for integers (numpy ints included) but not bools."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def check_poses(dataset: IntermediateDataset) -> CheckResult:
    """Poses are [N,4,4] with N>=2, finite, rigid last row, finite after float32 cast."""
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
    """Declared width/height are positive integers."""
    problems = [
        f"{name}: width/height {camera.width!r}x{camera.height!r} not positive ints"
        for name, camera in dataset.cameras.items()
        if not (_is_plain_int(camera.width) and _is_plain_int(camera.height)
                and camera.width > 0 and camera.height > 0)
    ]
    return _result("intrinsics_integer_size", problems)


def check_image_shape(dataset: IntermediateDataset) -> CheckResult:
    """Every stored image has shape (height, width, ...) equal to the declared size."""
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
    """Distortion is absent or has k4 == 0."""
    problems = [
        f"{name}: k4={camera.distortion[DISTORTION_K4_INDEX]}"
        for name, camera in dataset.cameras.items()
        if camera.distortion is not None
        and len(camera.distortion) > DISTORTION_K4_INDEX
        and camera.distortion[DISTORTION_K4_INDEX] != 0.0
    ]
    return _result("distortion_k4_zero", problems)


def check_times(dataset: IntermediateDataset) -> CheckResult:
    """Pose times are float64, finite and strictly increasing per sensor."""
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
    """Camera/lidar frame indices are exactly 0..n-1."""
    problems = [
        f"{name}: frame indices are not 0..{stream.frame_indices.size - 1}"
        for name, stream in dataset.streams.items()
        if not np.array_equal(stream.frame_indices, np.arange(stream.frame_indices.size))
    ]
    return _result("sensor_index_contiguity", problems)


def check_lidar_format(dataset: IntermediateDataset) -> CheckResult:
    """Each sweep is float32 with shape [n,5] or [n,6]."""
    problems = []
    for stream in _lidar_streams(dataset):
        for frame_number, sweep in enumerate(stream.frames):
            if sweep.dtype != np.float32 or sweep.ndim != 2 or sweep.shape[1] not in ALLOWED_LIDAR_COLUMNS:
                problems.append(f"{stream.sensor_name}[{frame_number}]: {sweep.dtype} {sweep.shape}")
    return _result("lidar_array_format", problems)


def check_intensity(dataset: IntermediateDataset) -> CheckResult:
    """Intensity column lies in [0, 1]."""
    problems = []
    for stream in _lidar_streams(dataset):
        for frame_number, sweep in enumerate(stream.frames):
            intensity = sweep[:, INTENSITY_COLUMN].astype(np.float64)
            if not (np.isfinite(intensity).all() and intensity.min() >= 0.0 and intensity.max() <= 1.0):
                problems.append(f"{stream.sensor_name}[{frame_number}]: intensity outside [0,1]")
    return _result("lidar_intensity_range", problems)


def check_point_time(dataset: IntermediateDataset) -> CheckResult:
    """Per-point time offset lies within +/- one sweep period."""
    problems = []
    for stream in _lidar_streams(dataset):
        for frame_number, sweep in enumerate(stream.frames):
            if np.abs(sweep[:, TIME_COLUMN]).max() > dataset.lidar.sweep_period:
                problems.append(f"{stream.sensor_name}[{frame_number}]: |t| > sweep period")
    return _result("lidar_point_time_range", problems)


def check_channels(dataset: IntermediateDataset) -> CheckResult:
    """When a channel column exists, its values are integers that are mapping keys."""
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
    """Elevation mapping length is a multiple of 8."""
    length = len(dataset.lidar.elevation_mapping)
    problems = [] if length > 0 and length % ELEVATION_TILE_CHANNELS == 0 else [
        f"length {length} not a positive multiple of {ELEVATION_TILE_CHANNELS}"
    ]
    return _result("elevation_mapping_multiple_of_8", problems)


def check_azimuth_tiles(dataset: IntermediateDataset) -> CheckResult:
    """Azimuth bins = 2*pi / resolution is an integer divisible by the tile width."""
    lidar = dataset.lidar
    problems = []
    # Number of azimuth bins around a full turn: B = 2*pi / resolution.
    bin_count = 2.0 * np.pi / lidar.azimuth_resolution
    if not np.isclose(bin_count, np.rint(bin_count), rtol=AZIMUTH_INTEGER_RTOL, atol=0.0):
        problems.append(f"2*pi/resolution = {bin_count} is not an integer")
    elif int(np.rint(bin_count)) % lidar.azimuth_tile_width != 0:
        problems.append(f"{int(np.rint(bin_count))} bins not divisible by tile width {lidar.azimuth_tile_width}")
    return _result("azimuth_tile_integral", problems)


def check_splits(dataset: IntermediateDataset) -> CheckResult:
    """Train and eval splits are non-empty for every camera/lidar sensor."""
    problems = []
    for name, stream in dataset.streams.items():
        train_indices, eval_indices = linspace_train_eval_split(
            stream.frame_indices.size, dataset.session.train_fraction
        )
        if train_indices.size == 0 or eval_indices.size == 0:
            problems.append(f"{name}: train={train_indices.size} eval={eval_indices.size}")
    return _result("splits_non_empty", problems)


def check_world_up(dataset: IntermediateDataset) -> CheckResult:
    """World up axis is +z."""
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
