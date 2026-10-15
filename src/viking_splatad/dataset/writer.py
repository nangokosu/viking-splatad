"""Intermediate dataset writer / reader (T26).

Layout under ``root``::

    manifest.json                      schema_version, sensors, intrinsics, extrinsics, clock,
                                       rolling shutter, lidar model, window, session
    poses.npz                          <sensor>__times, <sensor>__poses_world_sensor
    camera/<sensor>/<idx>_<ts_ns>.jpg  one subdirectory per camera (several cameras may exist)
    lidar/<sensor>/<idx>_<ts_ns>.npy   float32 [n,5] or [n,6]
    sidecar/radar/<name>.npy           radar diagnostics (never part of the SplatAD input)
"""

import json
import re
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np

from viking_splatad.dataset.schema import (
    CameraIntrinsics,
    ClockInfo,
    IntermediateDataset,
    LidarInfo,
    RollingShutterParams,
    SCHEMA_VERSION,
    SensorStream,
    SessionInfo,
    WindowInfo,
)
from viking_splatad.registry import transform

# JPEG quality 100 = least lossy setting OpenCV offers (JPEG itself stays lossy).
JPEG_QUALITY = 100
FRAME_FILE_PATTERN = re.compile(r"^(\d+)_(\d+)\.(jpg|npy)$")
# Zero-padding width of the frame index in file names (sorts lexicographically up to 10^6 frames).
INDEX_DIGITS = 6
# Separator between sensor name and array name inside poses.npz keys.
POSE_KEY_SEPARATOR = "__"


class SchemaVersionError(ValueError):
    """Raised when a dataset on disk has a different ``schema_version`` than this code."""


def _array_to_list(array: Any) -> Any:
    """ndarray -> nested list (JSON-safe, float64 repr round-trips exactly); None passes through."""
    return None if array is None else np.asarray(array).tolist()


def _manifest_dict(dataset: IntermediateDataset) -> Dict[str, Any]:
    """Serialise the non-bulk fields of a dataset to a JSON-ready dict.

    Args:
        dataset: Dataset to describe.

    Returns:
        Dict for ``manifest.json``.
    """
    lidar = dataset.lidar
    return {
        "schema_version": dataset.schema_version,
        "sensors": [
            {"name": stream.sensor_name, "kind": stream.kind}
            for stream in dataset.streams.values()
        ],
        "cameras": {
            name: {
                "width": camera.width, "height": camera.height,
                "fx": camera.fx, "fy": camera.fy, "cx": camera.cx, "cy": camera.cy,
                "distortion": _array_to_list(camera.distortion),
            }
            for name, camera in dataset.cameras.items()
        },
        "extrinsics": {name: _array_to_list(pose) for name, pose in dataset.extrinsics.items()},
        "clock": {
            "reference_sensor": dataset.clock.reference_sensor,
            "sensor_offsets_seconds": dataset.clock.sensor_offsets_seconds,
            "radar_drift_offset": dataset.clock.radar_drift_offset,
            "radar_drift_rate": dataset.clock.radar_drift_rate,
        },
        "rolling_shutter": {
            name: {"readout_seconds": params.readout_seconds, "direction": params.direction}
            for name, params in dataset.rolling_shutter.items()
        },
        "lidar": {
            "lidar_type": lidar.lidar_type,
            # JSON object keys must be strings; the reader restores ints.
            "elevation_mapping": {str(channel): angle for channel, angle in lidar.elevation_mapping.items()},
            "azimuth_resolution": lidar.azimuth_resolution,
            "azimuth_tile_width": lidar.azimuth_tile_width,
            "min_lidar_dist": lidar.min_lidar_dist,
            "intensity_scale": lidar.intensity_scale,
            "sweep_period": lidar.sweep_period,
        },
        "window": {"start_time": dataset.window.start_time, "end_time": dataset.window.end_time},
        "session": {
            "name": dataset.session.name,
            "world_up": _array_to_list(dataset.session.world_up),
            "train_fraction": dataset.session.train_fraction,
        },
    }


@transform("T26", "write the intermediate dataset (manifest, poses, frames, sidecar) to disk")
def write_dataset(root: Path, dataset: IntermediateDataset) -> None:
    """Write a dataset in the layout described in the module docstring.

    Args:
        root: Output directory (created if missing).
        dataset: Dataset to write. No validation is done here; see the contract module.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # 1) manifest
    (root / "manifest.json").write_text(json.dumps(_manifest_dict(dataset), indent=2))
    # 2) poses + times, keyed per sensor
    pose_arrays = {}
    for name, stream in dataset.streams.items():
        pose_arrays[f"{name}{POSE_KEY_SEPARATOR}times"] = stream.times
        pose_arrays[f"{name}{POSE_KEY_SEPARATOR}poses_world_sensor"] = stream.poses_world_sensor
    np.savez(root / "poses.npz", **pose_arrays)
    # 3) frames, one directory per sensor
    for name, stream in dataset.streams.items():
        frame_directory = root / stream.kind / name
        frame_directory.mkdir(parents=True, exist_ok=True)
        for frame_index, timestamp_ns, frame in zip(
            stream.frame_indices, stream.frame_timestamps_ns, stream.frames
        ):
            stem = f"{int(frame_index):0{INDEX_DIGITS}d}_{int(timestamp_ns)}"
            if stream.kind == "camera":
                cv2.imwrite(
                    str(frame_directory / f"{stem}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
                )
            else:
                np.save(frame_directory / f"{stem}.npy", frame)
    # 4) radar sidecar (directory always created so the layout is stable)
    radar_directory = root / "sidecar" / "radar"
    radar_directory.mkdir(parents=True, exist_ok=True)
    for name, array in dataset.radar_sidecar.items():
        np.save(radar_directory / f"{name}.npy", array)


def read_dataset(root: Path) -> IntermediateDataset:
    """Read a dataset written by ``write_dataset``, rebuilding the same dataclasses.

    Args:
        root: Dataset directory.

    Returns:
        The reconstructed ``IntermediateDataset``.

    Raises:
        SchemaVersionError: If the manifest ``schema_version`` != ``SCHEMA_VERSION``.
    """
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    # Refuse foreign layouts before interpreting any other field.
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise SchemaVersionError(
            f"dataset schema_version {manifest['schema_version']} != supported {SCHEMA_VERSION}"
        )
    pose_archive = np.load(root / "poses.npz")
    streams = {}
    for sensor in manifest["sensors"]:
        name, kind = sensor["name"], sensor["kind"]
        # Collect (index, timestamp, path) from the file names, ordered by index.
        frame_files = []
        for path in (root / kind / name).iterdir():
            match = FRAME_FILE_PATTERN.match(path.name)
            if match:
                frame_files.append((int(match.group(1)), int(match.group(2)), path))
        frame_files.sort(key=lambda record: record[0])
        frames = [
            cv2.imread(str(path)) if kind == "camera" else np.load(path)
            for _, _, path in frame_files
        ]
        streams[name] = SensorStream(
            sensor_name=name,
            kind=kind,
            times=pose_archive[f"{name}{POSE_KEY_SEPARATOR}times"],
            poses_world_sensor=pose_archive[f"{name}{POSE_KEY_SEPARATOR}poses_world_sensor"],
            frame_indices=np.array([record[0] for record in frame_files], dtype=np.int64),
            frame_timestamps_ns=np.array([record[1] for record in frame_files], dtype=np.int64),
            frames=frames,
        )
    radar_sidecar = {
        path.stem: np.load(path)
        for path in sorted((root / "sidecar" / "radar").glob("*.npy"))
    }
    lidar = manifest["lidar"]
    return IntermediateDataset(
        schema_version=manifest["schema_version"],
        cameras={
            name: CameraIntrinsics(
                width=entry["width"], height=entry["height"],
                fx=entry["fx"], fy=entry["fy"], cx=entry["cx"], cy=entry["cy"],
                distortion=None if entry["distortion"] is None else np.array(entry["distortion"], dtype=np.float64),
            )
            for name, entry in manifest["cameras"].items()
        },
        extrinsics={name: np.array(pose, dtype=np.float64) for name, pose in manifest["extrinsics"].items()},
        clock=ClockInfo(**manifest["clock"]),
        rolling_shutter={
            name: RollingShutterParams(**entry) for name, entry in manifest["rolling_shutter"].items()
        },
        lidar=LidarInfo(
            lidar_type=lidar["lidar_type"],
            elevation_mapping={int(channel): angle for channel, angle in lidar["elevation_mapping"].items()},
            azimuth_resolution=lidar["azimuth_resolution"],
            azimuth_tile_width=lidar["azimuth_tile_width"],
            min_lidar_dist=lidar["min_lidar_dist"],
            intensity_scale=lidar["intensity_scale"],
            sweep_period=lidar["sweep_period"],
        ),
        window=WindowInfo(**manifest["window"]),
        session=SessionInfo(
            name=manifest["session"]["name"],
            world_up=np.array(manifest["session"]["world_up"], dtype=np.float64),
            train_fraction=manifest["session"]["train_fraction"],
        ),
        streams=streams,
        radar_sidecar=radar_sidecar,
    )
