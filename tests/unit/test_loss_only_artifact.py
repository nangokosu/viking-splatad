"""End-to-end synthetic fixed-anchor artifact construction and fail-closed provenance gates."""

import json

import numpy as np
import pytest

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.dataset.run_split import file_sha256
from viking_splatad.dataset.writer import write_dataset
from viking_splatad.radar_init.artifact import CLOCK_SCHEMA, anchor_split, load_verified_anchors
from viking_splatad.radar_init.build import build_loss_only_artifact, source_digest
from viking_splatad.splatad.viking_conversions import open_dataset_view


def synthetic_artifact(tmp_path):
    """Return a measured synthetic two-sweep stationary target and its immutable artifact inputs."""
    dataset = build_synthetic_dataset(num_frames=20)
    dataset.extrinsics.update({"base->lidar": np.eye(4), "base->radar": np.eye(4)})
    dataset.radar_sidecar = {}
    root = tmp_path / "dataset"
    write_dataset(root, dataset)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["pipeline"] = {"reference_ns": 0, "radar": {"fields": ["x", "y", "z", "power", "doppler"]}}
    manifest_path.write_text(json.dumps(manifest))
    settings = {"start_fraction": 0., "end_fraction": 1., "train_fraction": 0.8,
                "test_split_method": "shared_interleaved", "test_block_count": 2,
                "test_block_core_fraction": 1., "test_render_fraction": 1.}
    view = open_dataset_view(root)
    _, lidar_frames, _ = anchor_split(view, settings)
    selected = [frame for frame in lidar_frames if frame.split == "train"][:2]
    records = []
    for frame_index, frame in enumerate(selected):
        # The fixture drives +0.5 m per 0.1 s: velocity = 5 m/s.
        point = np.array([10. - 5 * frame.time_s, 0., 1.])
        radial_velocity = -5 * point[0] / np.linalg.norm(point)
        path = root / "sidecar/radar" / f"{frame_index:06d}_{frame.timestamp_ns}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array([[*point, 1., radial_velocity]]))
        records.append((frame.timestamp_ns, path))
    calibration = {"schema": CLOCK_SCHEMA, "session": "synthetic", "reference_ns": 0,
                   "method": "synthetic known clock", "offset_seconds": 0., "drift_rate": 0.,
                   "clock_reference_seconds": 0., "residual_rms_seconds": 0.,
                   "fit_stamps_ns": [stamp for stamp, _ in records], "fit_source_sha256": source_digest(records),
                   "doppler_noise_std_mps": 0.05, "doppler_sign": 1}
    session_config = {"session": {"name": "synthetic"}, "lidar": {"name": "lidar_top"},
                      "frames": {"base": "base", "lidar": "lidar", "radar": "radar"}}
    options = {"range_cell_m": 0.1, "azimuth_cell_degrees": 1.25, "elevation_cell_degrees": 1.7,
               "max_range_m": 42., "confidence_quantile": 0.95, "persistence_quantile": 0.5,
               "clock_search_periods": 0.5}
    return root, settings, session_config, calibration, options, records


def test_builder_publishes_only_loss_targets_and_preserves_baseline_inputs(tmp_path):
    """Two radar sweeps of one world target produce one fixed anchor without editing baseline data."""
    root, settings, config, calibration, options, _ = synthetic_artifact(tmp_path)
    manifest_hash, pose_hash = file_sha256(root / "manifest.json"), file_sha256(root / "poses.npz")
    output = tmp_path / "anchors"
    summary = build_loss_only_artifact(root, config, settings, options, output, calibration)
    arrays, metadata = load_verified_anchors(output, root, "synthetic", settings)
    np.testing.assert_allclose(arrays["means"], [[10., 0., 1.]])
    assert summary["anchor_count"] == 1 and metadata["gaussian_initialization"] is False
    assert file_sha256(root / "manifest.json") == manifest_hash
    assert file_sha256(root / "poses.npz") == pose_hash
    with pytest.raises(FileExistsError):
        build_loss_only_artifact(root, config, settings, options, output, calibration)


def test_artifact_rejects_wrong_split_and_modified_raw_source(tmp_path):
    """A different holdout or modified radar file cannot reuse the verified artifact."""
    root, settings, config, calibration, options, records = synthetic_artifact(tmp_path)
    output = tmp_path / "anchors"
    build_loss_only_artifact(root, config, settings, options, output, calibration)
    with pytest.raises(ValueError, match="split"):
        load_verified_anchors(output, root, "synthetic", dict(settings, train_fraction=0.5))
    np.save(records[0][1], np.zeros((1, 5)))
    with pytest.raises(ValueError, match="source.*checksum"):
        load_verified_anchors(output, root, "synthetic", settings)


def test_clock_report_rejects_held_out_fit_observation(tmp_path):
    """Training loss calibration cannot consume radar observations at a held-out instant."""
    root, settings, config, calibration, options, _ = synthetic_artifact(tmp_path)
    _, frames, _ = anchor_split(open_dataset_view(root), settings)
    heldout = next(frame for frame in frames if frame.split != "train")
    invalid = dict(calibration, fit_stamps_ns=[heldout.timestamp_ns])
    with pytest.raises(ValueError, match="non-training"):
        build_loss_only_artifact(root, config, settings, options, tmp_path / "refused", invalid)
    assert not (tmp_path / "refused").exists()
