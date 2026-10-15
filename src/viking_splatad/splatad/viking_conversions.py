"""Framework-free half of the Viking Hill dataparser (T27).

Everything that turns the intermediate dataset (``dataset/writer.py`` layout) into the
numeric arrays neurad-studio's ``ADDataParser`` needs lives here, using only numpy, so it
is importable and unit-testable in the plain project venv (no torch / nerfstudio).
``viking_dataparser.py`` wraps these arrays into ``Cameras`` / ``Lidars`` tensors.

Unit conventions at the boundary (neurad-studio source, verified against the checkout):
    * lidar elevation mappings are in DEGREES, ``{channel_id: elevation_degrees}``;
    * lidar azimuth resolution is in DEGREES per column;
    * the manifest stores both in RADIANS, so they are converted here.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from viking_splatad.contract.splatad_contract import AZIMUTH_INTEGER_RTOL
from viking_splatad.dataset.schema import SCHEMA_VERSION
from viking_splatad.dataset.writer import (
    FRAME_FILE_PATTERN,
    POSE_KEY_SEPARATOR,
    SchemaVersionError,
)
from viking_splatad.extract.geometry import apply_opencv_to_nerfstudio
from viking_splatad.registry import transform

# Name of the LidarType member the patch adds (patches/apply_ouster_patch.py).
OUSTER_LIDAR_TYPE_NAME = "OUSTER_OS0_32"
# Full turn in degrees; azimuth resolution [deg] = FULL_TURN_DEGREES / columns per turn.
FULL_TURN_DEGREES = 360.0


@dataclass(frozen=True)
class DatasetView:
    """Lightweight, pixel-free view of an intermediate dataset on disk.

    Attributes:
        root: Dataset directory.
        manifest: Parsed ``manifest.json``.
        sensor_times: sensor name -> float64 [N] seconds.
        sensor_poses: sensor name -> float64 [N,4,4] ``T_world_sensor``.
        sensor_frame_paths: sensor name -> frame file paths ordered by frame index.
    """

    root: Path
    manifest: dict
    sensor_times: Dict[str, np.ndarray]
    sensor_poses: Dict[str, np.ndarray]
    sensor_frame_paths: Dict[str, List[Path]]


@dataclass(frozen=True)
class NeuradLidarModel:
    """Lidar model in neurad-studio units, derived from the manifest.

    Attributes:
        type_name: ``LidarType`` member name (e.g. ``OUSTER_OS0_32``).
        elevation_mapping_degrees: channel id -> elevation [deg] (mapping order is irrelevant
            to the rasteriser, which sorts the values; ids only matter for missing points).
        azimuth_resolution_degrees: degrees per azimuth column.
        revolution_time_seconds: seconds per sweep.
        min_lidar_radius: points closer than this [m] to the sensor are ego returns.
    """

    type_name: str
    elevation_mapping_degrees: Dict[int, float]
    azimuth_resolution_degrees: float
    revolution_time_seconds: float
    min_lidar_radius: float


@dataclass(frozen=True)
class CameraArrays:
    """Per-image camera arrays, one row per image, all cameras concatenated in config order.

    ``camera_to_worlds`` is already in nerfstudio axes (flip applied exactly once).
    """

    camera_to_worlds: np.ndarray  # [N,3,4] float32
    times: np.ndarray  # [N] float64 seconds
    focal_x: np.ndarray  # [N] float32 pixels
    focal_y: np.ndarray  # [N] float32 pixels
    principal_x: np.ndarray  # [N] float32 pixels
    principal_y: np.ndarray  # [N] float32 pixels
    widths: np.ndarray  # [N] int32
    heights: np.ndarray  # [N] int32
    sensor_indices: np.ndarray  # [N,1] int32, index of the camera inside the configured tuple
    image_paths: List[Path]
    rolling_shutter_seconds: float  # full-frame readout, first camera's value
    time_to_center_pixel_seconds: float  # centre-row offset from the image stamp


@dataclass(frozen=True)
class LidarArrays:
    """Per-sweep lidar arrays, one row per sweep, all lidars concatenated in config order."""

    lidar_to_worlds: np.ndarray  # [N,3,4] float32, pose at the sweep's stamp
    times: np.ndarray  # [N] float64 seconds
    sensor_indices: np.ndarray  # [N,1] int32
    sweep_paths: List[Path]
    model: NeuradLidarModel


def open_dataset_view(root: Path) -> DatasetView:
    """Read manifest, times and poses and list frame files, without loading any frame.

    Unlike ``read_dataset`` this never decodes images or sweeps (a session holds
    thousands of 2056x1542 JPEGs), so the dataparser stays cheap and streams from disk.

    Args:
        root: Dataset directory written by ``write_dataset``.

    Returns:
        A ``DatasetView``.

    Raises:
        SchemaVersionError: If the manifest ``schema_version`` is unsupported.
        FileNotFoundError: If ``manifest.json`` or ``poses.npz`` is missing.
    """
    root = Path(root)
    # 1) manifest, refusing foreign layouts before reading anything else.
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"dataset schema_version {manifest['schema_version']} != supported {SCHEMA_VERSION}"
        )
    # 2) times and poses per sensor (small arrays, loaded fully).
    pose_archive = np.load(root / "poses.npz")
    sensor_times, sensor_poses, sensor_frame_paths = {}, {}, {}
    for sensor in manifest["sensors"]:
        name, kind = sensor["name"], sensor["kind"]
        sensor_times[name] = np.asarray(pose_archive[f"{name}{POSE_KEY_SEPARATOR}times"], dtype=np.float64)
        sensor_poses[name] = np.asarray(
            pose_archive[f"{name}{POSE_KEY_SEPARATOR}poses_world_sensor"], dtype=np.float64
        )
        # 3) frame files ordered by the zero-padded frame index in their names.
        indexed_paths = []
        for path in (root / kind / name).iterdir():
            match = FRAME_FILE_PATTERN.match(path.name)
            if match:
                indexed_paths.append((int(match.group(1)), path))
        indexed_paths.sort(key=lambda record: record[0])
        sensor_frame_paths[name] = [path for _, path in indexed_paths]
    return DatasetView(root, manifest, sensor_times, sensor_poses, sensor_frame_paths)


def neurad_lidar_model_from_manifest(lidar_manifest: dict) -> NeuradLidarModel:
    """Convert the manifest's lidar block (radians) to neurad-studio units (degrees).

    Azimuth snapping: the datamanager computes ``tile_width = ceil(360 / (res_deg * 32))``.
    With ``res_deg = rad2deg(2*pi/B)`` round-off can land a hair above an exact integer
    (e.g. 32.000000000000004 -> ceil = 33, one phantom tile). When ``B = 2*pi/res_rad`` is
    an integer to within ``AZIMUTH_INTEGER_RTOL`` the resolution is therefore rebuilt as
    exactly ``360 / B`` (1024 columns -> 0.3515625, a dyadic float, exact in binary).

    Args:
        lidar_manifest: ``manifest["lidar"]`` dict.

    Returns:
        The lidar model in neurad-studio units.
    """
    # Elevation: radians -> degrees; JSON stores channel ids as string keys.
    elevation_mapping_degrees = {
        int(channel): float(np.rad2deg(angle_radians))
        for channel, angle_radians in lidar_manifest["elevation_mapping"].items()
    }
    # Azimuth: columns per turn B = 2*pi / resolution, snapped to an integer when it is one.
    columns_per_turn = 2.0 * math.pi / lidar_manifest["azimuth_resolution"]
    nearest_column_count = round(columns_per_turn)
    if abs(columns_per_turn - nearest_column_count) <= AZIMUTH_INTEGER_RTOL * nearest_column_count:
        azimuth_resolution_degrees = FULL_TURN_DEGREES / nearest_column_count
    else:
        azimuth_resolution_degrees = float(np.rad2deg(lidar_manifest["azimuth_resolution"]))
    return NeuradLidarModel(
        type_name=str(lidar_manifest["lidar_type"]),
        elevation_mapping_degrees=elevation_mapping_degrees,
        azimuth_resolution_degrees=azimuth_resolution_degrees,
        revolution_time_seconds=float(lidar_manifest["sweep_period"]),
        min_lidar_radius=float(lidar_manifest["min_lidar_dist"]),
    )


def load_lidar_sweep(sweep_path: Path, min_lidar_radius: float) -> np.ndarray:
    """Load one sweep and drop ego returns and non-finite rows.

    The manifest holds a single scalar ``min_lidar_dist``, interpreted as a sphere radius:
    row is kept iff ``||xyz|| >= min_lidar_radius``.

    Args:
        sweep_path: ``.npy`` file with float32 [n,5|6] ``x,y,z,intensity,t[,channel]``.
        min_lidar_radius: Ego-return radius in metres.

    Returns:
        float32 [m,5|6] array, m <= n.
    """
    sweep = np.load(sweep_path).astype(np.float32, copy=False)
    # Range of every point from the sensor origin (sensor frame).
    ranges = np.linalg.norm(sweep[:, :3], axis=1)
    keep_mask = np.isfinite(sweep).all(axis=1) & (ranges >= min_lidar_radius)
    return sweep[keep_mask]


def _stack_with_first_axis(arrays: List[np.ndarray], empty_shape: tuple, dtype) -> np.ndarray:
    """Concatenate per-sensor arrays along axis 0 (empty-safe).

    Args:
        arrays: Arrays sharing trailing dimensions.
        empty_shape: Shape to return when ``arrays`` is empty.
        dtype: Dtype of the empty result.

    Returns:
        The concatenation, or an empty array of ``empty_shape``.
    """
    return np.concatenate(arrays, axis=0) if arrays else np.zeros(empty_shape, dtype=dtype)


@transform(
    "T27",
    "Dataset -> neurad-studio sensor arrays: OPENCV_TO_NERFSTUDIO flip once, int sizes, "
    "seconds, lidar model in degrees",
)
def build_sensor_arrays(
    view: DatasetView,
    camera_names: tuple,
    lidar_names: tuple,
    time_to_center_fraction_of_readout: float,
) -> tuple:
    """Assemble the camera and lidar arrays the ``ADDataParser`` subclass consumes.

    Camera pose: the dataset stores ``T_world_camera`` in OpenCV axes (x right, y down,
    z forward). nerfstudio wants x right, y up, z backward, i.e. right-multiplying the
    rotation by ``diag(1,-1,-1)`` (``apply_opencv_to_nerfstudio``). That flip happens HERE,
    exactly once; nothing upstream applies it and the dataparser does not repeat it.

    Rolling shutter: the centre row is read ``readout/2`` after the first row, so
    ``time_to_center_pixel = fraction * readout`` with fraction 0.5 when the image stamp
    marks the start of readout and 0 when it marks mid-readout.

    Args:
        view: Dataset view from ``open_dataset_view``.
        camera_names: Camera stream names, in config order (index = ``sensor_idx``).
        lidar_names: Lidar stream names, in config order.
        time_to_center_fraction_of_readout: Fraction of the readout time at which the centre
            row is exposed after the image stamp.

    Returns:
        ``(CameraArrays, LidarArrays)``; either may have zero rows if its names are empty
        (lidar then carries a model from the manifest regardless).

    Raises:
        KeyError: If a requested sensor is not in the dataset.
        ValueError: If a stream's frame count differs from its pose/time count.
    """
    manifest = view.manifest
    # ---- cameras ----
    per_camera = {key: [] for key in (
        "c2w", "times", "fx", "fy", "cx", "cy", "width", "height", "sensor_index", "paths")}
    readout_seconds = 0.0
    for camera_index, camera_name in enumerate(camera_names):
        poses, times, paths = (
            view.sensor_poses[camera_name], view.sensor_times[camera_name],
            view.sensor_frame_paths[camera_name],
        )
        if not (len(poses) == len(times) == len(paths)):
            raise ValueError(
                f"{camera_name}: {len(poses)} poses, {len(times)} times, {len(paths)} images"
            )
        intrinsics = manifest["cameras"][camera_name]
        frame_count = len(times)
        # Flip each OpenCV-axes pose to nerfstudio axes, keep the top 3x4 block as float32.
        flipped = np.stack([apply_opencv_to_nerfstudio(pose) for pose in poses])
        per_camera["c2w"].append(flipped[:, :3, :].astype(np.float32))
        per_camera["times"].append(times)
        # Intrinsics are per camera (images are pre-undistorted), repeated per image.
        for key, field_name in (("fx", "fx"), ("fy", "fy"), ("cx", "cx"), ("cy", "cy")):
            per_camera[key].append(np.full(frame_count, intrinsics[field_name], dtype=np.float32))
        per_camera["width"].append(np.full(frame_count, int(intrinsics["width"]), dtype=np.int32))
        per_camera["height"].append(np.full(frame_count, int(intrinsics["height"]), dtype=np.int32))
        per_camera["sensor_index"].append(np.full((frame_count, 1), camera_index, dtype=np.int32))
        per_camera["paths"].extend(paths)
        # Rolling shutter is a per-sensor float in neurad; the first camera defines it.
        if camera_index == 0:
            readout_seconds = float(
                manifest.get("rolling_shutter", {}).get(camera_name, {}).get("readout_seconds", 0.0)
            )
    cameras = CameraArrays(
        camera_to_worlds=_stack_with_first_axis(per_camera["c2w"], (0, 3, 4), np.float32),
        times=_stack_with_first_axis(per_camera["times"], (0,), np.float64),
        focal_x=_stack_with_first_axis(per_camera["fx"], (0,), np.float32),
        focal_y=_stack_with_first_axis(per_camera["fy"], (0,), np.float32),
        principal_x=_stack_with_first_axis(per_camera["cx"], (0,), np.float32),
        principal_y=_stack_with_first_axis(per_camera["cy"], (0,), np.float32),
        widths=_stack_with_first_axis(per_camera["width"], (0,), np.int32),
        heights=_stack_with_first_axis(per_camera["height"], (0,), np.int32),
        sensor_indices=_stack_with_first_axis(per_camera["sensor_index"], (0, 1), np.int32),
        image_paths=per_camera["paths"],
        rolling_shutter_seconds=readout_seconds,
        time_to_center_pixel_seconds=time_to_center_fraction_of_readout * readout_seconds,
    )
    # ---- lidars ----
    per_lidar = {key: [] for key in ("l2w", "times", "sensor_index", "paths")}
    for lidar_index, lidar_name in enumerate(lidar_names):
        poses, times, paths = (
            view.sensor_poses[lidar_name], view.sensor_times[lidar_name],
            view.sensor_frame_paths[lidar_name],
        )
        if not (len(poses) == len(times) == len(paths)):
            raise ValueError(
                f"{lidar_name}: {len(poses)} poses, {len(times)} times, {len(paths)} sweeps"
            )
        # Lidar frame is x-forward, y-left, z-up, which neurad uses as is: no axis flip.
        per_lidar["l2w"].append(poses[:, :3, :].astype(np.float32))
        per_lidar["times"].append(times)
        per_lidar["sensor_index"].append(np.full((len(times), 1), lidar_index, dtype=np.int32))
        per_lidar["paths"].extend(paths)
    lidars = LidarArrays(
        lidar_to_worlds=_stack_with_first_axis(per_lidar["l2w"], (0, 3, 4), np.float32),
        times=_stack_with_first_axis(per_lidar["times"], (0,), np.float64),
        sensor_indices=_stack_with_first_axis(per_lidar["sensor_index"], (0, 1), np.int32),
        sweep_paths=per_lidar["paths"],
        model=neurad_lidar_model_from_manifest(manifest["lidar"]),
    )
    return cameras, lidars


def resolve_train_fraction(view: DatasetView, override: Optional[float]) -> float:
    """Pick the train fraction: explicit override, else the manifest's session value.

    Args:
        view: Dataset view.
        override: Config override, or None to use the dataset's own value.

    Returns:
        Train fraction in (0, 1].
    """
    return float(view.manifest["session"]["train_fraction"] if override is None else override)
