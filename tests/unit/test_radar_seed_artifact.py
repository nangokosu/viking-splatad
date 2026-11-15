"""Radar-initialization model parameter, ground-truth isolation and dataparser tests (T41).

The dataparser test builds its artifact with the real builder (`build_loss_only_artifact`, plus the
schema /2 painting helper), so the train-only attachment is checked against a verified artifact.
"""

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
    """The offline builder (script 07 and the modules it wraps) has no GT/label dependency or path reference."""
    project_root = Path(__file__).parents[2]
    builder_paths = [
        project_root / "scripts" / "07_build_radar_seeds.py",
        project_root / "src" / "viking_splatad" / "radar_init" / "build.py",
        project_root / "src" / "viking_splatad" / "radar_init" / "seeds.py",
    ]
    for builder_path in builder_paths:
        builder_text = builder_path.read_text()
        assert "viking_splatad.labels" not in builder_text, builder_path
        assert "gt_dir" not in builder_text, builder_path
        assert "labels_json" not in builder_text, builder_path


def _build_parser_fixture(tmp_path, painted: bool):
    """Build a synthetic OUSTER-typed dataset plus a verified radar anchor artifact for the parser.

    Args:
        tmp_path: pytest temporary directory (parser data root).
        painted: Whether to rewrite the artifact into schema /2 with `rgb`.

    Returns:
        (parser config kwargs, artifact rgb array or None).
    """
    import json

    from tests.unit.test_loss_only_artifact import synthetic_artifact
    from tests.unit.test_radar_painted_artifact import make_painted
    from viking_splatad.radar_init.build import build_loss_only_artifact

    dataset_root, split_settings, session_config, calibration, options, _ = synthetic_artifact(tmp_path)
    # Step 1: the parser needs a typed lidar and a session name equal to the dataset directory name.
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["lidar"]["lidar_type"] = "OUSTER_OS0_32"
    manifest["session"]["name"] = dataset_root.name
    manifest_path.write_text(json.dumps(manifest))
    session_config["session"]["name"] = dataset_root.name
    calibration["session"] = dataset_root.name
    # Step 2: build into the session directory so the relative radar_inputs_dir resolves.
    artifact_directory = dataset_root / "radar_inputs"
    build_loss_only_artifact(dataset_root, session_config, split_settings, options, artifact_directory, calibration)
    artifact_rgb = None
    if painted:
        with np.load(artifact_directory / "radar_seeds.npz", allow_pickle=False) as archive:
            anchor_count = len(archive["means"])
        artifact_rgb = np.full((anchor_count, 3), 127, dtype=np.uint8)
        make_painted(artifact_directory, artifact_rgb, np.zeros(anchor_count, dtype=np.int64))
    # Step 3: parser split settings equal the artifact's (parser_split_settings maps the field names).
    parser_kwargs = dict(
        data=tmp_path, session_name=dataset_root.name, cameras=("cam_front",), lidars=("lidar_top",),
        dataset_start_fraction=split_settings["start_fraction"], dataset_end_fraction=split_settings["end_fraction"],
        train_split_fraction_override=split_settings["train_fraction"],
        test_block_count=split_settings["test_block_count"],
        test_block_core_fraction=split_settings["test_block_core_fraction"],
        test_split_method=split_settings["test_split_method"],
        test_render_fraction=split_settings["test_render_fraction"],
    )
    return parser_kwargs, artifact_rgb


@pytest.mark.skipif(not _neurad_ready(), reason="needs the reference image")
@pytest.mark.parametrize("painted", [False, True])
def test_dataparser_attaches_radar_only_to_train_and_stock_path_is_unchanged(tmp_path, painted):
    """Configured seeds are train-only (rgb as integers when painted); a stock parser has no radar key."""
    import torch
    from viking_splatad.splatad.viking_dataparser import VikingHillDataParserConfig

    parser_kwargs, artifact_rgb = _build_parser_fixture(tmp_path, painted)
    configured_parser = VikingHillDataParserConfig(radar_inputs_dir=Path("radar_inputs"), **parser_kwargs).setup()
    train_outputs = configured_parser.get_dataparser_outputs("train")
    evaluation_outputs = configured_parser.get_dataparser_outputs("eval")
    train_seeds = train_outputs.metadata["radar_seeds"]
    # Train outputs carry anchors with (N, 3) means; eval outputs carry none.
    assert train_seeds["means"].ndim == 2 and train_seeds["means"].shape[1] == 3
    assert train_seeds["means"].shape[0] >= 1
    assert "radar_seeds" not in evaluation_outputs.metadata
    # rgb reaches the train metadata, as an integer tensor, only for a painted artifact.
    if painted:
        assert train_seeds["rgb"].shape == (train_seeds["means"].shape[0], 3)
        assert not train_seeds["rgb"].dtype.is_floating_point
        np.testing.assert_array_equal(train_seeds["rgb"].numpy(), artifact_rgb)
    else:
        assert "rgb" not in train_seeds
    # A stock parser (no radar_inputs_dir) exposes no radar metadata key.
    stock_outputs = VikingHillDataParserConfig(**parser_kwargs).setup().get_dataparser_outputs("train")
    assert "radar_seeds" not in stock_outputs.metadata
