"""Known-answer tests for the SuperSplat colour bake (T85) against the upstream RGBDecoderCNN."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
splatad_module = pytest.importorskip("nerfstudio.models.splatad")

from viking_splatad.export.splatad_colours import (
    bake_gaussian_colours,
    camera_sensor_index,
    load_upstream_rgb_decoder,
    model_settings_from_config_yml,
)

RGBDecoderCNN = splatad_module.RGBDecoderCNN

# Realistic sizes (match the real runs) and a seed; the values only need to be varied, not exact.
SETTINGS = {
    "feature_dim": 13,
    "appearance_dim": 8,
    "rgb_decoder_hidden_dim": 32,
    "rgb_decoder_kernel_size": 3,
    "rgb_decoder_num_hidden_blocks": 1,
    "mcmc_min_opacity": 0.005,
    "cameras": ["ids_camera"],
}
RANDOM_SEED = 1234
GAUSSIAN_COUNT = 5
DIRECTION_COUNT = 3
# Weight scale large enough that the decoder output is not ~0 (upstream last layer is scaled 1e-2).
LAST_LAYER_GAIN = 100.0
FLOAT_TOLERANCE = 1e-5


def _build_random_decoder():
    """Upstream decoder with seeded random weights and an amplified last layer.

    Returns:
        ``RGBDecoderCNN`` in eval mode, zero padding (upstream default).
    """
    torch.manual_seed(RANDOM_SEED)
    decoder = RGBDecoderCNN(
        SETTINGS["feature_dim"] + SETTINGS["appearance_dim"] + 3,
        hidden_dim=SETTINGS["rgb_decoder_hidden_dim"],
        kernel_size=SETTINGS["rgb_decoder_kernel_size"],
        num_hidden_blocks=SETTINGS["rgb_decoder_num_hidden_blocks"],
    )
    with torch.no_grad():
        decoder.net[-1].weight.mul_(LAST_LAYER_GAIN)
    return decoder.eval()


def _receptive_radius(decoder) -> int:
    """Width of the border whose outputs zero padding can change.

    Each stride-1 k x k conv widens the dependency by (k-1)/2 pixels per side, and stacked convs add,
    so summing over every Conv2d (residual 1x1 convs add 0) bounds the longest path.

    Args:
        decoder: Module whose Conv2d layers are summed.

    Returns:
        Receptive radius in pixels.
    """
    return sum((module.kernel_size[0] - 1) // 2 for module in decoder.modules() if isinstance(module, torch.nn.Conv2d))


@pytest.mark.covers("T85")
def test_bake_matches_upstream_forward_on_uniform_patch():
    """The baked colour equals the centre pixel of the unmodified forward on a uniform patch."""
    decoder = _build_random_decoder()
    reference_decoder = copy.deepcopy(decoder)  # unmodified, zero padding
    # Seeded random Gaussian features, camera embedding and unit directions.
    generator = torch.Generator().manual_seed(RANDOM_SEED)
    features_dc = torch.rand(GAUSSIAN_COUNT, 3, generator=generator)
    features_rest = torch.randn(GAUSSIAN_COUNT, SETTINGS["feature_dim"], generator=generator)
    appearance_vector = torch.randn(SETTINGS["appearance_dim"], generator=generator)
    directions = torch.nn.functional.normalize(torch.randn(DIRECTION_COUNT, 3, generator=generator), dim=-1)

    baked = bake_gaussian_colours(decoder, features_dc, features_rest, appearance_vector, directions, chunk_size=2)

    # Smallest patch whose centre pixel is out of reach of the zero padding on every side.
    patch_size = 2 * _receptive_radius(reference_decoder) + 1
    centre = patch_size // 2
    assert baked.shape == (GAUSSIAN_COUNT, DIRECTION_COUNT, 3) and baked.dtype == np.float32
    for gaussian_index in range(GAUSSIAN_COUNT):
        # Forward input channels: albedo (3) + rest (F) + appearance (A), uniform over the patch.
        feature_vector = torch.cat([features_dc[gaussian_index], features_rest[gaussian_index], appearance_vector])
        uniform_features = feature_vector.expand(patch_size, patch_size, -1).contiguous()
        for direction_index in range(DIRECTION_COUNT):
            ray_dirs = directions[direction_index].expand(1, patch_size, patch_size, 3).contiguous()
            # Upstream forward on the uniform patch, then the model's clamp, read at the centre.
            with torch.no_grad():
                rendered = reference_decoder(uniform_features, ray_dirs)
            expected = torch.clamp(rendered[0, centre, centre], 0.0, 1.0).numpy()
            np.testing.assert_allclose(baked[gaussian_index, direction_index], expected, atol=FLOAT_TOLERANCE)
    # Non-triviality: the colours are not all clamped to one value.
    assert baked.std() > 0.0


@pytest.mark.covers("T85")
def test_bake_does_not_modify_callers_decoder():
    """Padding mode of the caller's convolutions stays at the upstream default."""
    decoder = _build_random_decoder()
    bake_gaussian_colours(decoder, torch.zeros(1, 3), torch.zeros(1, 13), torch.zeros(8), torch.zeros(1, 3), 1)
    assert all(
        module.padding_mode == "zeros" for module in decoder.modules() if isinstance(module, torch.nn.Conv2d)
    )


@pytest.mark.covers("T85")
def test_load_upstream_decoder_strips_prefixes_and_is_strict():
    """Keys with the model and DDP prefixes load; a missing key raises."""
    source = _build_random_decoder()
    appearance_table = torch.randn(2, SETTINGS["appearance_dim"])
    state = {f"module._model.rgb_decoder.{key}": value for key, value in source.state_dict().items()}
    state["_model.appearance_embedding.weight"] = appearance_table
    # Prefixed keys load into an eval-mode decoder with identical weights.
    loaded, appearance_weight = load_upstream_rgb_decoder(state, SETTINGS)
    assert not loaded.training
    assert torch.equal(appearance_weight, appearance_table)
    for key, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[key], value)

    # Dropping one decoder key must fail the strict load.
    missing_key = next(iter(state))
    del state[missing_key]
    with pytest.raises(RuntimeError):
        load_upstream_rgb_decoder(state, SETTINGS)


@pytest.mark.covers("T85")
def test_camera_sensor_index_is_position():
    """The sensor index of a camera is its position in the camera list."""
    camera_names = ["front", "left", "right"]
    for expected_index, camera_name in enumerate(camera_names):
        assert camera_sensor_index(camera_names, camera_name) == expected_index
    with pytest.raises(ValueError):
        camera_sensor_index(camera_names, "rear")


@pytest.mark.covers("T85")
def test_model_settings_read_from_tagged_config_yml(tmp_path):
    """A config.yml with python tags (as nerfstudio writes it) yields the settings and cameras."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "method_name: splatad\n"
        "pipeline: !!python/object:nerfstudio.pipelines.splatad_pipeline.SplatADPipelineConfig\n"
        "  _target: !!python/name:nerfstudio.pipelines.splatad_pipeline.SplatADPipeline ''\n"
        "  datamanager: !!python/object:a.b.Config\n"
        "    dataparser: !!python/object:c.d.ParserConfig\n"
        "      cameras: !!python/tuple\n"
        "      - cam_one\n"
        "      - cam_two\n"
        "      data: !!python/object/apply:pathlib.PosixPath\n"
        "      - /\n"
        "      - somewhere\n"
        "  model: !!python/object:nerfstudio.models.splatad.SplatADModelConfig\n"
        "    appearance_dim: 7\n"
        "    feature_dim: 11\n"
        "    mcmc_min_opacity: 0.25\n"
        "    rgb_decoder_hidden_dim: 21\n"
        "    rgb_decoder_kernel_size: 5\n"
        "    rgb_decoder_num_hidden_blocks: 2\n"
    )
    settings = model_settings_from_config_yml(config_path)
    assert settings == {
        "feature_dim": 11,
        "appearance_dim": 7,
        "rgb_decoder_hidden_dim": 21,
        "rgb_decoder_kernel_size": 5,
        "rgb_decoder_num_hidden_blocks": 2,
        "mcmc_min_opacity": 0.25,
        "cameras": ["cam_one", "cam_two"],
    }


# Invalid chunk sizes: zero would make range() raise obscurely, negatives would return no colours.
INVALID_CHUNK_SIZES = (0, -1)


def _bake_one_gaussian(decoder, chunk_size: int = 1) -> np.ndarray:
    """Bake one zero-feature Gaussian and one direction with the given decoder.

    Args:
        decoder: Upstream decoder to bake with.
        chunk_size: Chunk size passed through.

    Returns:
        (1, 1, 3) baked colour.
    """
    return bake_gaussian_colours(
        decoder,
        torch.full((1, 3), 0.5),
        torch.zeros(1, SETTINGS["feature_dim"]),
        torch.zeros(SETTINGS["appearance_dim"]),
        torch.tensor([[0.0, 0.0, 1.0]]),
        chunk_size,
    )


@pytest.mark.covers("T85")
@pytest.mark.parametrize("invalid_chunk_size", INVALID_CHUNK_SIZES)
def test_bake_rejects_chunk_size_below_one(invalid_chunk_size):
    """chunk_size 0 and -1 raise ValueError."""
    with pytest.raises(ValueError):
        _bake_one_gaussian(_build_random_decoder(), invalid_chunk_size)


@pytest.mark.covers("T85")
def test_bake_accepts_float64_and_cpu_moved_decoder():
    """A float64 decoder or one moved with .to('cpu') bakes; the colours match the float32 bake."""
    reference_colours = _bake_one_gaussian(_build_random_decoder())
    cpu_moved_colours = _bake_one_gaussian(_build_random_decoder().to("cpu"))
    np.testing.assert_allclose(cpu_moved_colours, reference_colours, atol=FLOAT_TOLERANCE)
    # float64 weights: the module's dtype differs from the float32 inputs; the bake must cope.
    float64_decoder = _build_random_decoder().double()
    try:
        float64_colours = _bake_one_gaussian(float64_decoder)
    except RuntimeError as dtype_error:  # pragma: no cover - documents the failure mode
        pytest.fail(f"float64 decoder crashed the bake: {dtype_error}")
    np.testing.assert_allclose(float64_colours, reference_colours, atol=FLOAT_TOLERANCE)


@pytest.mark.covers("T85")
def test_bake_copy_is_on_cpu_and_callers_weights_unchanged(monkeypatch):
    """The decoder copy used is on CPU, and the caller's weights are identical after the bake."""
    decoder = _build_random_decoder()
    weights_before = {key: value.clone() for key, value in decoder.state_dict().items()}
    seen_devices = []
    original_deepcopy = copy.deepcopy

    def recording_deepcopy(module):
        """deepcopy that records the device of the first parameter of the copied decoder."""
        duplicate = original_deepcopy(module)
        seen_devices.append(next(duplicate.parameters()).device.type)
        return duplicate

    monkeypatch.setattr(copy, "deepcopy", recording_deepcopy)
    _bake_one_gaussian(decoder)
    # The copy itself, after the bake's .cpu(), is on CPU; verify through a CPU-only parameter check.
    assert seen_devices and all(device_type == "cpu" for device_type in seen_devices)
    for key, value in decoder.state_dict().items():
        assert torch.equal(value, weights_before[key])


@pytest.mark.covers("T85")
def test_bake_with_zero_width_appearance():
    """appearance_dim 0 (upstream Embedding(num_sensors, 0)) bakes without error."""
    torch.manual_seed(RANDOM_SEED)
    decoder = RGBDecoderCNN(
        SETTINGS["feature_dim"] + 3,
        hidden_dim=SETTINGS["rgb_decoder_hidden_dim"],
        kernel_size=SETTINGS["rgb_decoder_kernel_size"],
        num_hidden_blocks=SETTINGS["rgb_decoder_num_hidden_blocks"],
    ).eval()
    colours = bake_gaussian_colours(
        decoder,
        torch.full((2, 3), 0.5),
        torch.zeros(2, SETTINGS["feature_dim"]),
        torch.zeros(0),
        torch.tensor([[0.0, 0.0, 1.0]]),
        chunk_size=1,
    )
    assert colours.shape == (2, 1, 3) and np.isfinite(colours).all()


@pytest.mark.covers("T85")
def test_walk_section_null_or_non_mapping_section_raises_key_error(tmp_path):
    """A null or scalar section on the path raises the documented KeyError, not TypeError."""
    from viking_splatad.export.splatad_colours import _walk_section

    with pytest.raises(KeyError):
        _walk_section({"pipeline": None}, ("pipeline", "model"))
    with pytest.raises(KeyError):
        _walk_section({"pipeline": {"model": 3}}, ("pipeline", "model", "x"))
    with pytest.raises(KeyError):
        _walk_section({"pipeline": {"model": None}}, ("pipeline", "model"))


@pytest.mark.covers("T85")
@pytest.mark.parametrize("cameras_block", ["cameras: !!python/tuple []\n", "cameras: !!python/tuple\n      - None\n"])
def test_settings_reject_empty_or_none_cameras(tmp_path, cameras_block):
    """An empty camera tuple or the upstream 'none' sentinel is unresolvable and raises ValueError."""
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "pipeline:\n  datamanager:\n    dataparser:\n      " + cameras_block
        + "  model:\n    appearance_dim: 1\n    feature_dim: 1\n    mcmc_min_opacity: 0.1\n"
        "    rgb_decoder_hidden_dim: 1\n    rgb_decoder_kernel_size: 1\n    rgb_decoder_num_hidden_blocks: 1\n"
    )
    with pytest.raises(ValueError):
        model_settings_from_config_yml(config_path)
