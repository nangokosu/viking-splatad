"""Artifact, dataparser conversion and model parameter tests (T39-T41)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.radar_init.artifact import (
    RADAR_ARRAYS_FILENAME,
    load_radar_seed_artifact,
    write_radar_seed_artifact,
)
from viking_splatad.radar_init.seeds import RadarSeedGaussians
from viking_splatad.splatad.radar_init_model import covariance_to_log_scales_and_quaternions
from viking_splatad.splatad.viking_conversions import (
    crop_radar_seeds_to_scene_and_time,
    transform_radar_seeds_to_dataparser,
)


def _neurad_ready() -> bool:
    """Return whether the pinned torch/neurad dataparser is importable."""
    try:
        from nerfstudio.cameras.lidars import LidarType
    except ImportError:
        return False
    return hasattr(LidarType, "OUSTER_OS0_32")


def _seed_arrays() -> RadarSeedGaussians:
    """Return two small positive-definite seeds for artifact tests."""
    return RadarSeedGaussians(
        means=np.array([[1.0, 2.0, 3.0], [20.0, 0.0, 0.0]]),
        covariances=np.array([np.diag([1.0, 4.0, 9.0]), np.eye(3)]),
        rgb=np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8),
        sensor_origins=np.zeros((2, 3)),
        support_count=np.array([2, 1], dtype=np.int64),
        time_spans_s=np.array([[10.0, 12.0], [30.0, 31.0]]),
    )


def test_artifact_round_trip_and_hash_corruption_failure(tmp_path):
    """Every file is hashed and one changed archive byte is rejected before loading."""
    artifact_directory = tmp_path / "radar_inputs"
    write_radar_seed_artifact(
        artifact_directory, _seed_arrays(), "june_tall_grass",
        {"split_digest": "known"}, {"range_resolution_m": 0.1},
    )
    arrays, metadata = load_radar_seed_artifact(artifact_directory, "june_tall_grass")
    np.testing.assert_array_equal(arrays["rgb"], _seed_arrays().rgb)
    assert metadata["seed_count"] == 2
    array_path = artifact_directory / RADAR_ARRAYS_FILENAME
    array_path.write_bytes(array_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_radar_seed_artifact(artifact_directory, "june_tall_grass")


@pytest.mark.covers("T39")
def test_radar_seed_similarity_transform_rotates_covariance():
    """T39 applies translation/scale to points and scale-squared rotation to covariance."""
    rotation = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    transform = np.column_stack([rotation, [10.0, 0.0, 0.0]])
    arrays = {field_name: np.asarray(getattr(_seed_arrays(), field_name))
              for field_name in _seed_arrays().__dataclass_fields__}
    transformed = transform_radar_seeds_to_dataparser(arrays, transform, 2.0)
    np.testing.assert_allclose(transformed["means"][0], [16.0, 2.0, 6.0], atol=1e-12)
    expected_covariance = 4.0 * rotation @ arrays["covariances"][0] @ rotation.T
    np.testing.assert_allclose(transformed["covariances"][0], expected_covariance, atol=1e-12)


@pytest.mark.covers("T40")
def test_crop_prevents_time_and_scene_leakage_and_makes_times_relative():
    """Only a seed overlapping the train sensor window and AABB reaches metadata."""
    arrays = {field_name: np.asarray(getattr(_seed_arrays(), field_name))
              for field_name in _seed_arrays().__dataclass_fields__}
    cropped, counts = crop_radar_seeds_to_scene_and_time(
        arrays, 11.0, 4.0, np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    )
    assert counts == {
        "input": 2, "outside_scene": 1, "outside_time": 1,
        "retained": 1, "retained_fraction": 0.5,
    }
    np.testing.assert_allclose(cropped["time_spans_s"], [[0.0, 1.0]])


@pytest.mark.covers("T41")
def test_covariance_parameters_reconstruct_the_original_covariance():
    """T41 eigenscales and scalar-first quaternions reconstruct covariance exactly."""
    rotation = Rotation.from_euler("xyz", [20.0, -10.0, 35.0], degrees=True).as_matrix()
    covariance = rotation @ np.diag([0.01, 0.04, 0.09]) @ rotation.T
    log_scales, quaternions_wxyz = covariance_to_log_scales_and_quaternions(covariance[None])
    reconstructed_rotation = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]]).as_matrix()[0]
    reconstructed = reconstructed_rotation @ np.diag(np.exp(log_scales[0]) ** 2) @ reconstructed_rotation.T
    np.testing.assert_allclose(reconstructed, covariance, atol=1e-12)


@pytest.mark.skipif(not _neurad_ready(), reason="needs the reference image")
def test_model_appends_static_radar_parameters_without_new_parameter_keys(monkeypatch):
    """The model adds exact seed geometry/color while retaining stock parameter groups."""
    import torch
    from types import SimpleNamespace
    from nerfstudio.models.splatad import SplatADModel
    from viking_splatad.splatad.radar_init_model import RadarInitSplatADModel

    stock_parameters = torch.nn.ParameterDict({
        "means": torch.zeros((2, 3)),
        "scales": torch.zeros((2, 3)),
        "quats": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
        "features_dc": torch.zeros((2, 3)),
        "features_rest": torch.zeros((2, 13)),
        "opacities": torch.zeros((2, 1)),
        "id": torch.nn.Parameter(torch.zeros((2, 1)), requires_grad=False),
    })
    model = object.__new__(RadarInitSplatADModel)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(radar_seed_opacity=0.25)
    model.kwargs = {"metadata": {"radar_seeds": {
        "means": torch.tensor([[1.0, 2.0, 3.0]]),
        "covariances": torch.tensor([np.diag([0.01, 0.04, 0.09])]),
        "rgb": torch.tensor([[255, 128, 0]], dtype=torch.uint8),
    }}}
    monkeypatch.setattr(
        SplatADModel, "create_gauss_param_dict",
        lambda model_instance, *arguments, **keyword_arguments: stock_parameters,
    )
    appended = RadarInitSplatADModel.create_gauss_param_dict(model, [], [])
    assert set(appended) == set(stock_parameters)
    assert appended["means"].shape[0] == 3
    torch.testing.assert_close(appended["means"][-1], torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(appended["features_dc"][-1], torch.tensor([1.0, 128.0 / 255.0, 0.0]))
    assert appended["id"][-1].item() == stock_parameters["id"].max().item()
    assert not appended["id"].requires_grad


def test_radar_builder_never_imports_or_reads_ground_truth():
    """The offline builder has no GT/label dependency or path reference."""
    builder_text = (Path(__file__).parents[2] / "scripts" / "07_build_radar_seeds.py").read_text()
    assert "viking_splatad.labels" not in builder_text
    assert "gt_dir" not in builder_text
    assert "labels_json" not in builder_text


@pytest.mark.skipif(not _neurad_ready(), reason="needs the reference image")
def test_dataparser_attaches_radar_only_to_train_and_stock_path_is_unchanged(tmp_path):
    """Configured seeds are train-only; a stock parser exposes no radar metadata key."""
    from tests.unit.test_viking_dataparser import (
        SYNTHETIC_CAMERA,
        SYNTHETIC_LIDAR,
        _write_synthetic,
    )
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig
    from viking_splatad.dataset.run_split import build_run_split_with_cores, split_digest
    from viking_splatad.radar_init.artifact import sha256_file
    from viking_splatad.splatad.viking_conversions import open_dataset_view

    dataset, dataset_root = _write_synthetic(tmp_path)
    first_time = min(stream.times.min() for stream in dataset.streams.values())
    last_time = max(stream.times.max() for stream in dataset.streams.values())
    seeds = RadarSeedGaussians(
        means=np.array([[0.0, 0.0, 0.0]]),
        covariances=np.array([np.eye(3) * 0.01]),
        rgb=np.array([[127, 127, 127]], dtype=np.uint8),
        sensor_origins=np.array([[-1.0, 0.0, 0.0]]),
        support_count=np.array([2]),
        time_spans_s=np.array([[first_time, last_time]]),
    )
    dataset_view = open_dataset_view(dataset_root)
    split_frames, _, _, _ = build_run_split_with_cores(
        dataset_view, 0.0, 1.0, dataset.session.train_fraction
    )
    write_radar_seed_artifact(
        dataset_root / "radar_inputs",
        seeds,
        "session",
        {
            "manifest_sha256": sha256_file(dataset_root / "manifest.json"),
            "poses_sha256": sha256_file(dataset_root / "poses.npz"),
            "split_digest": split_digest(split_frames),
        },
        {},
    )
    configured_parser = VikingHillDataParserConfig(
        data=tmp_path,
        session_name="session",
        cameras=(SYNTHETIC_CAMERA,),
        lidars=(SYNTHETIC_LIDAR,),
        radar_inputs_dir=Path("radar_inputs"),
    ).setup()
    train_outputs = configured_parser.get_dataparser_outputs("train")
    evaluation_outputs = configured_parser.get_dataparser_outputs("eval")
    assert train_outputs.metadata["radar_seeds"]["means"].shape == (1, 3)
    assert "radar_seeds" not in evaluation_outputs.metadata
    stock_outputs = VikingHillDataParserConfig(
        data=tmp_path,
        session_name="session",
        cameras=(SYNTHETIC_CAMERA,),
        lidars=(SYNTHETIC_LIDAR,),
    ).setup().get_dataparser_outputs("train")
    assert "radar_seeds" not in stock_outputs.metadata
