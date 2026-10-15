"""Tests for the Viking Hill dataparser (T27).

Pure-numpy tests run in the project venv. The pipeline test needs torch + neurad-studio
with the T28 patch and skips cleanly elsewhere. To run everything on a CPU-only Mac::

    # one-off: neurad env (python 3.11, numpy<2, cpu torch, jaxtyping, matplotlib, tyro, rich,
    # viser, tensorboard, wandb, opencv, scipy, pytest) + `pip install -e .` of this repo
    src/viking_splatad/splatad/install_into_neurad.sh          # patch third_party checkout
    PYTHONPATH=third_party/neurad-studio <neurad_env>/bin/python -m pytest \
        tests/unit/test_viking_dataparser.py tests/unit/test_ouster_patch.py
"""

import dataclasses
import math

import numpy as np
import pytest

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.dataset.writer import write_dataset
from viking_splatad.splatad.viking_conversions import (
    build_sensor_arrays,
    load_lidar_sweep,
    neurad_lidar_model_from_manifest,
    open_dataset_view,
)

SYNTHETIC_CAMERA, SYNTHETIC_LIDAR = "cam_front", "lidar_top"
FRAME_COUNT = 10


def _write_synthetic(tmp_path):
    """Write the synthetic dataset (lidar typed OUSTER_OS0_32) and return (dataset, root)."""
    dataset = build_synthetic_dataset(num_frames=FRAME_COUNT)
    dataset.lidar = dataclasses.replace(dataset.lidar, lidar_type="OUSTER_OS0_32")
    # Give the camera a non-identity rotation so the axis flip is observable.
    angle = 0.3
    rotation_z = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                           [math.sin(angle), math.cos(angle), 0.0],
                           [0.0, 0.0, 1.0]])
    dataset.streams[SYNTHETIC_CAMERA].poses_world_sensor[:, :3, :3] = rotation_z
    root = tmp_path / "session"
    write_dataset(root, dataset)
    return dataset, root


@pytest.mark.covers("T27")
def test_sensor_arrays_counts_dtypes_and_single_axis_flip(tmp_path):
    """Shapes/dtypes follow the contract; the OpenCV->nerfstudio flip is applied exactly once."""
    dataset, root = _write_synthetic(tmp_path)
    cameras, lidars = build_sensor_arrays(open_dataset_view(root), (SYNTHETIC_CAMERA,), (SYNTHETIC_LIDAR,), 0.5)
    assert cameras.camera_to_worlds.shape == (FRAME_COUNT, 3, 4)
    assert cameras.camera_to_worlds.dtype == np.float32
    assert cameras.widths.dtype == np.int32 and cameras.widths[0] == dataset.cameras[SYNTHETIC_CAMERA].width
    assert cameras.times.dtype == np.float64 and np.all(np.diff(cameras.times) > 0)
    assert lidars.lidar_to_worlds.shape == (FRAME_COUNT, 3, 4)
    assert np.all(np.diff(lidars.times) > 0) and np.isfinite(lidars.lidar_to_worlds).all()
    assert len(cameras.image_paths) == len(lidars.sweep_paths) == FRAME_COUNT
    # Expected: stored rotation with its y and z COLUMNS negated (R @ diag(1,-1,-1)), once.
    stored = dataset.streams[SYNTHETIC_CAMERA].poses_world_sensor
    expected_rotation = stored[:, :3, :3].copy()
    expected_rotation[:, :, 1:] *= -1.0
    np.testing.assert_allclose(cameras.camera_to_worlds[:, :, :3], expected_rotation, atol=1e-6)
    np.testing.assert_allclose(cameras.camera_to_worlds[:, :, 3], stored[:, :3, 3], atol=1e-6)
    # Lidar poses are not flipped.
    np.testing.assert_allclose(lidars.lidar_to_worlds, dataset.streams[SYNTHETIC_LIDAR].poses_world_sensor[:, :3], atol=1e-6)
    # Rolling shutter from the manifest; centre row at half the readout.
    readout = dataset.rolling_shutter[SYNTHETIC_CAMERA].readout_seconds
    assert cameras.rolling_shutter_seconds == readout
    assert cameras.time_to_center_pixel_seconds == pytest.approx(0.5 * readout)


@pytest.mark.covers("T27")
def test_lidar_model_units_and_azimuth_snap():
    """Radians -> degrees; 1024 columns snap to exactly 360/1024 so tile_width == 32."""
    ring_count = 32
    elevations_degrees = np.linspace(45.0, -42.0, ring_count)
    manifest_lidar = {
        "lidar_type": "OUSTER_OS0_32",
        "elevation_mapping": {str(ring): float(np.deg2rad(angle)) for ring, angle in enumerate(elevations_degrees)},
        "azimuth_resolution": 2.0 * np.pi / 1024,
        "azimuth_tile_width": 32, "min_lidar_dist": 0.8, "intensity_scale": 1.0, "sweep_period": 0.1,
    }
    model = neurad_lidar_model_from_manifest(manifest_lidar)
    assert model.azimuth_resolution_degrees == 360.0 / 1024
    # Datamanager formula: tile_width = ceil(360 / (res * 32 azimuth columns per tile)).
    azimuth_columns_per_tile = 32
    assert math.ceil(360.0 / (model.azimuth_resolution_degrees * azimuth_columns_per_tile)) == 1024 // azimuth_columns_per_tile
    assert len(model.elevation_mapping_degrees) % 8 == 0
    np.testing.assert_allclose(
        [model.elevation_mapping_degrees[ring] for ring in range(ring_count)], elevations_degrees, atol=1e-9)
    assert model.min_lidar_radius == 0.8 and model.revolution_time_seconds == 0.1


@pytest.mark.covers("T27")
def test_load_lidar_sweep_drops_ego_and_nonfinite_rows(tmp_path):
    """Rows closer than the radius or non-finite are removed; others untouched."""
    sweep = np.array([[0.1, 0.0, 0.0, 0.5, 0.0, 1.0],      # ego (r = 0.1)
                      [3.0, 4.0, 0.0, 0.5, 0.0, 2.0],      # r = 5 keep
                      [np.nan, 0.0, 0.0, 0.5, 0.0, 3.0]],  # non-finite
                     dtype=np.float32)
    path = tmp_path / "sweep.npy"
    np.save(path, sweep)
    kept = load_lidar_sweep(path, min_lidar_radius=1.0)
    assert kept.shape == (1, 6) and kept.dtype == np.float32 and kept[0, 5] == 2.0


def _neurad_ready():
    """True when torch + patched neurad-studio are importable."""
    try:
        from nerfstudio.cameras.lidars import LidarType
    except ImportError:
        return False
    return hasattr(LidarType, "OUSTER_OS0_32")


@pytest.mark.covers("T27")
@pytest.mark.skipif(not _neurad_ready(), reason="needs torch + patched neurad-studio (see module docstring)")
def test_real_adparser_pipeline_on_synthetic_dataset(tmp_path):
    """Synthetic dataset through the real ADDataParser: split counts, dtypes, times, poses, no actors."""
    import torch
    from nerfstudio.cameras.lidars import LidarType, get_lidar_azimuth_resolution, get_lidar_elevation_mapping
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig

    dataset, root = _write_synthetic(tmp_path)
    config = VikingHillDataParserConfig(
        data=tmp_path, session_name="session", cameras=(SYNTHETIC_CAMERA,), lidars=(SYNTHETIC_LIDAR,))
    train = config.setup().get_dataparser_outputs("train")
    evaluation = config.setup().get_dataparser_outputs("eval")
    # Split sizes follow the manifest train fraction: ceil(N * f) train, rest eval.
    train_count = math.ceil(FRAME_COUNT * dataset.session.train_fraction)
    assert len(train.cameras) == train_count and len(evaluation.cameras) == FRAME_COUNT - train_count
    assert len(train.metadata["lidars"]) == train_count
    # Contract-level properties of the produced tensors.
    assert train.cameras.width.dtype in (torch.int32, torch.int64) and train.cameras.distortion_params is None
    assert torch.isfinite(train.cameras.camera_to_worlds).all()
    assert torch.isfinite(train.metadata["lidars"].lidar_to_worlds).all()
    assert (torch.diff(train.cameras.times.squeeze(-1)) > 0).all()
    assert all(cloud.dtype == torch.float32 for cloud in train.metadata["point_clouds"])
    ego_radius = dataset.lidar.min_lidar_dist
    assert all((cloud[:, :3].norm(dim=-1) >= ego_radius).all() for cloud in train.metadata["point_clouds"])
    assert train.metadata["trajectories"] == []
    assert float(train.cameras.metadata["rolling_shutter_time"][0]) == pytest.approx(
        dataset.rolling_shutter[SYNTHETIC_CAMERA].readout_seconds)
    # Camera pose survives the dataparser's re-centering: undo it and compare with the flipped input.
    to_world = torch.cat([train.dataparser_transform, torch.tensor([[0.0, 0.0, 0.0, 1.0]])]).inverse()
    camera_4x4 = torch.cat([train.cameras.camera_to_worlds,
                            torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).expand(train_count, 1, 4)], dim=1)
    restored = (to_world @ camera_4x4)[:, :3, :3]
    stored = dataset.streams[SYNTHETIC_CAMERA].poses_world_sensor[:, :3, :3].copy()
    stored[:, :, 1:] *= -1.0
    train_rows = np.unique(np.linspace(0, FRAME_COUNT - 1, train_count).astype(int))
    np.testing.assert_allclose(restored.numpy(), stored[train_rows], atol=1e-4)
    # The patched accessors now serve the manifest-derived model (no ValueError).
    mapping = get_lidar_elevation_mapping(LidarType.OUSTER_OS0_32)
    np.testing.assert_allclose(
        [mapping[channel] for channel in sorted(mapping)],
        [np.rad2deg(dataset.lidar.elevation_mapping[channel]) for channel in sorted(dataset.lidar.elevation_mapping)],
        atol=1e-9)
    assert get_lidar_azimuth_resolution(LidarType.OUSTER_OS0_32) == pytest.approx(
        np.rad2deg(dataset.lidar.azimuth_resolution))
