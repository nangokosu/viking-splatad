"""Radar-initialization model parameter, ground-truth isolation and dataparser tests (T41)."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.splatad.radar_init_model import covariance_to_log_scales_and_quaternions


def _neurad_ready() -> bool:
    """Return whether the pinned torch/neurad dataparser is importable."""
    try:
        from nerfstudio.cameras.lidars import LidarType
    except ImportError:
        return False
    return hasattr(LidarType, "OUSTER_OS0_32")


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
