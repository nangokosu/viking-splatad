"""Round-trip tests for the intermediate dataset writer/reader (T26)."""

import json

import numpy as np
import pytest

from viking_splatad.dataset.schema import (
    SCHEMA_VERSION,
    CameraIntrinsics,
    ClockInfo,
    IntermediateDataset,
    LidarInfo,
    RollingShutterParams,
    SensorStream,
    SessionInfo,
    WindowInfo,
)
from viking_splatad.dataset.writer import SchemaVersionError, read_dataset, write_dataset

SECONDS_TO_NANOSECONDS = 1_000_000_000  # unit conversion factor


def build_synthetic_dataset(num_frames: int = 10, seed: int = 0) -> IntermediateDataset:
    """Valid synthetic dataset: 1 camera, 1 lidar (6-column sweeps), radar sidecar.

    Args:
        num_frames: Frames per sensor.
        seed: RNG seed for pixel/point content.

    Returns:
        A dataset that satisfies every contract check.
    """
    rng = np.random.default_rng(seed)
    image_width, image_height = 32, 24
    sweep_period = 0.1
    channel_count = 8  # elevation mapping length must be a multiple of 8
    streams = {}
    for sensor_name, kind in (("cam_front", "camera"), ("lidar_top", "lidar")):
        times = np.arange(num_frames, dtype=np.float64) * sweep_period
        poses = np.tile(np.eye(4), (num_frames, 1, 1))
        poses[:, 0, 3] = np.arange(num_frames) * 0.5  # drive along +x
        if kind == "camera":
            frames = [
                rng.integers(0, 256, size=(image_height, image_width, 3), dtype=np.uint8)
                for frame_number in range(num_frames)
            ]
        else:
            frames = []
            for frame_number in range(num_frames):
                point_count = int(rng.integers(20, 40))
                sweep = np.column_stack(
                    [
                        rng.uniform(-20, 20, point_count),  # x
                        rng.uniform(-20, 20, point_count),  # y
                        rng.uniform(-2, 2, point_count),  # z
                        rng.uniform(0, 1, point_count),  # intensity
                        rng.uniform(-sweep_period / 2, sweep_period / 2, point_count),  # t
                        rng.integers(0, channel_count, point_count),  # channel
                    ]
                ).astype(np.float32)
                frames.append(sweep)
        streams[sensor_name] = SensorStream(
            sensor_name=sensor_name,
            kind=kind,
            times=times,
            poses_world_sensor=poses,
            frame_indices=np.arange(num_frames, dtype=np.int64),
            frame_timestamps_ns=np.rint(times * SECONDS_TO_NANOSECONDS).astype(np.int64),
            frames=frames,
        )
    return IntermediateDataset(
        schema_version=SCHEMA_VERSION,
        cameras={
            "cam_front": CameraIntrinsics(
                width=image_width, height=image_height,
                fx=30.0, fy=31.0, cx=16.0, cy=12.0,
                distortion=np.array([0.01, -0.002, 0.0, 0.0, 0.0005, -0.0003]),
            )
        },
        extrinsics={"cam_front": np.eye(4), "lidar_top": np.eye(4)},
        clock=ClockInfo("lidar_top", {"cam_front": 0.0123}, 0.25, 2e-4),
        rolling_shutter={"cam_front": RollingShutterParams(readout_seconds=0.02, direction="top_to_bottom")},
        lidar=LidarInfo(
            lidar_type="spinning",
            elevation_mapping={channel: np.deg2rad(-10.0 + 2.0 * channel) for channel in range(channel_count)},
            azimuth_resolution=2.0 * np.pi / 512,
            azimuth_tile_width=8,
            min_lidar_dist=1.5,
            intensity_scale=255.0,
            sweep_period=sweep_period,
        ),
        window=WindowInfo(start_time=0.0, end_time=(num_frames - 1) * sweep_period),
        session=SessionInfo(name="synthetic", world_up=np.array([0.0, 0.0, 1.0]), train_fraction=0.8),
        streams=streams,
        radar_sidecar={"radar_front": rng.normal(size=(5, 4))},
    )


@pytest.mark.covers("T26")
def test_write_then_read_round_trip_is_exact(tmp_path):
    """Everything but JPEG pixel values comes back bit-exact; layout files exist."""
    dataset = build_synthetic_dataset()
    write_dataset(tmp_path, dataset)
    loaded = read_dataset(tmp_path)

    assert loaded == dataset  # dataclass equality is exact for numpy arrays
    first_ns = int(dataset.streams["lidar_top"].frame_timestamps_ns[0])
    assert (tmp_path / "manifest.json").is_file()
    assert (tmp_path / "poses.npz").is_file()
    assert (tmp_path / "lidar" / "lidar_top" / f"000000_{first_ns}.npy").is_file()
    assert (tmp_path / "camera" / "cam_front" / f"000000_{first_ns}.jpg").is_file()
    assert (tmp_path / "sidecar" / "radar").is_dir()
    # Lidar sweeps keep float32 dtype and exact values.
    for original, restored in zip(dataset.streams["lidar_top"].frames, loaded.streams["lidar_top"].frames):
        assert restored.dtype == np.float32
        assert np.array_equal(original, restored)
    # JPEG is lossy: shape preserved only.
    assert loaded.streams["cam_front"].frames[0].shape == dataset.streams["cam_front"].frames[0].shape


@pytest.mark.covers("T26")
def test_schema_version_mismatch_raises(tmp_path):
    """A manifest written by another schema version is refused."""
    write_dataset(tmp_path, build_synthetic_dataset(num_frames=3))
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] += 1
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(SchemaVersionError):
        read_dataset(tmp_path)
