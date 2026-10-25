"""Dataclasses describing the intermediate dataset handed to the SplatAD converter.

Conventions: poses 4x4 float64 ``T_<target>_<source>``; timestamps float64 seconds;
quaternions (x, y, z, w). Equality is exact for numpy arrays (see ``values_equal``).
"""

import dataclasses
import numbers
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Bump whenever the on-disk layout or manifest fields change; reader rejects others.
SCHEMA_VERSION = 1


def values_equal(left: Any, right: Any) -> bool:
    """Exact structural equality that understands numpy arrays and dataclasses.

    Args:
        left, right: Values to compare (arrays, dataclasses, dicts, lists, scalars).

    Returns:
        True when types, shapes, dtypes and every element match exactly (NaN == NaN).
    """
    # Arrays: both sides must be arrays with identical dtype and shape before comparing
    # elements; NaN equality is enabled only for float dtypes (equal_nan is invalid otherwise).
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return (
            isinstance(left, np.ndarray)
            and isinstance(right, np.ndarray)
            and left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=left.dtype.kind == "f")
        )
    # Dicts: same key set, then recurse on each value.
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            values_equal(left[key], right[key]) for key in left
        )
    # Sequences: list and tuple are interchangeable (JSON round-trips tuples as lists);
    # same length, then recurse element-wise.
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            values_equal(left_item, right_item) for left_item, right_item in zip(left, right)
        )
    if isinstance(left, numbers.Real) and isinstance(right, numbers.Real):
        # np.float64 vs float is immaterial, but int vs float must stay distinguishable
        # (the contract requires integer image sizes).
        return (
            isinstance(left, numbers.Integral) == isinstance(right, numbers.Integral)
            and left == right
        )
    return type(left) is type(right) and left == right


class ExactEqualityMixin:
    """Gives dataclasses (declared with ``eq=False``) array-safe exact equality."""

    def __eq__(self, other: object) -> bool:
        """Field-wise ``values_equal`` comparison.

        Args:
            other: Object to compare against.

        Returns:
            True if every dataclass field is exactly equal; NotImplemented for a different type.
        """
        if type(other) is not type(self):
            return NotImplemented
        return all(
            values_equal(getattr(self, spec.name), getattr(other, spec.name))
            for spec in dataclasses.fields(self)
        )

    __hash__ = None  # mutable, array-holding objects are not hashable


@dataclass(eq=False)
class CameraIntrinsics(ExactEqualityMixin):
    """Pinhole intrinsics. ``width``/``height`` are stored as given (contract checks int).

    ``distortion`` is [k1, k2, k3, k4, p1, p2] (neurad order) or None; SplatAD needs k4 == 0.
    """

    width: Any
    height: Any
    fx: float
    fy: float
    cx: float
    cy: float
    distortion: Optional[np.ndarray] = None


@dataclass(eq=False)
class ClockInfo(ExactEqualityMixin):
    """Clock alignment results: per-sensor offsets [s] to the reference clock + radar drift."""

    reference_sensor: str
    sensor_offsets_seconds: Dict[str, float]
    radar_drift_offset: float
    radar_drift_rate: float


@dataclass(eq=False)
class RollingShutterParams(ExactEqualityMixin):
    """Camera rolling shutter: full-frame readout time [s] and scan direction."""

    readout_seconds: float
    direction: str


@dataclass(eq=False)
class LidarInfo(ExactEqualityMixin):
    """Lidar model.

    ``elevation_mapping`` maps channel id -> elevation angle [rad];
    ``azimuth_resolution`` is radians per azimuth bin; ``azimuth_tile_width`` is the
    number of azimuth bins per rasterisation tile; ``sweep_period`` is seconds/sweep.
    ``min_lidar_dist`` is the ego filter: the extraction pipeline writes (dx, dy, dz) box
    half-extents [m]; a scalar is read as a sphere radius [m].
    """

    lidar_type: str
    elevation_mapping: Dict[int, float]
    azimuth_resolution: float
    azimuth_tile_width: int
    min_lidar_dist: Union[float, Tuple[float, float, float]]
    intensity_scale: float
    sweep_period: float


@dataclass(eq=False)
class WindowInfo(ExactEqualityMixin):
    """Time window [s] of the recording that was extracted."""

    start_time: float
    end_time: float


@dataclass(eq=False)
class SessionInfo(ExactEqualityMixin):
    """Session metadata: name, world up-axis (must be +z) and train fraction."""

    name: str
    world_up: np.ndarray
    train_fraction: float


@dataclass(eq=False)
class SensorStream(ExactEqualityMixin):
    """One sensor's frames and poses.

    ``frames`` holds uint8 HxWx3 images (camera) or float32 [n,5|6] sweeps (lidar,
    columns x, y, z, intensity, t[, channel]). Camera frames are JPEG (lossy), so
    equality compares their shape/dtype only.
    """

    sensor_name: str
    kind: str
    times: np.ndarray
    poses_world_sensor: np.ndarray
    frame_indices: np.ndarray
    frame_timestamps_ns: np.ndarray
    frames: List[np.ndarray] = field(default_factory=list)

    def __eq__(self, other: object) -> bool:
        """Exact equality; camera pixel content is excluded (JPEG is lossy).

        Args:
            other: Object to compare against.

        Returns:
            True if all fields match (camera frames: shape and dtype only);
            NotImplemented for a different type.
        """
        if type(other) is not type(self):
            return NotImplemented
        # Compare field by field, short-circuiting on the first mismatch.
        for spec in dataclasses.fields(self):
            left, right = getattr(self, spec.name), getattr(other, spec.name)
            if spec.name == "frames" and self.kind == "camera":
                same = len(left) == len(right) and all(
                    left_frame.shape == right_frame.shape and left_frame.dtype == right_frame.dtype
                    for left_frame, right_frame in zip(left, right)
                )
            else:
                same = values_equal(left, right)
            if not same:
                return False
        return True

    __hash__ = None


@dataclass(eq=False)
class IntermediateDataset(ExactEqualityMixin):
    """Whole intermediate dataset (manifest + poses + frames + radar sidecar)."""

    schema_version: int
    cameras: Dict[str, CameraIntrinsics]
    extrinsics: Dict[str, np.ndarray]
    clock: ClockInfo
    rolling_shutter: Dict[str, RollingShutterParams]
    lidar: LidarInfo
    window: WindowInfo
    session: SessionInfo
    streams: Dict[str, SensorStream]
    radar_sidecar: Dict[str, np.ndarray] = field(default_factory=dict)
