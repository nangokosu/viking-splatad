"""Per-Gaussian RGB for the SuperSplat export, baked with the upstream SplatAD RGB decoder (T85).

SplatAD stores no per-Gaussian RGB: each Gaussian has ``features_dc`` (3, the albedo) and
``features_rest`` (F). The camera render alpha-blends those features per pixel, appends the
appearance embedding and the ray direction, and decodes with ``RGBDecoderCNN``
(``nerfstudio.models.splatad``). To show a colour per Gaussian, this module runs that very decoder
on a feature image that is uniform, so no layer is reimplemented. torch and nerfstudio are
imported lazily, so this module imports without them.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import yaml

from viking_splatad.registry import transform

# Prefix of the RGB decoder's keys in the checkpoint's ``pipeline`` state dict.
RGB_DECODER_KEY_PREFIX = "_model.rgb_decoder."
# Key of the appearance embedding table in the same state dict.
APPEARANCE_EMBEDDING_KEY = "_model.appearance_embedding.weight"
# Prefix DistributedDataParallel puts before every key (checked after the optional strip).
DDP_KEY_PREFIX = "module."
# Smallest valid number of Gaussians per forward batch.
MIN_CHUNK_SIZE = 1
# Sentinel (case-insensitive) the upstream ad_dataparser reads as "no cameras" (cameras == ("none",)).
NO_CAMERAS_SENTINEL = "none"
# Width of the ray direction appended to the decoder input (splatad.py: ``viewdir_dim = 3``).
VIEW_DIRECTION_DIM = 3
# Camera colour range the model clamps to (splatad.py: ``torch.clamp(rgb, 0.0, 1.0)``).
RGB_MIN, RGB_MAX = 0.0, 1.0
# Field names read from the model section (MODEL_SECTION_PATH) of the run's config.yml.
MODEL_SETTING_FIELDS = (
    "feature_dim",
    "appearance_dim",
    "rgb_decoder_hidden_dim",
    "rgb_decoder_kernel_size",
    "rgb_decoder_num_hidden_blocks",
    "mcmc_min_opacity",
)
# Location of the model and dataparser sections in nerfstudio's config.yml.
MODEL_SECTION_PATH = ("pipeline", "model")
DATAPARSER_SECTION_PATH = ("pipeline", "datamanager", "dataparser")


class _TagTolerantLoader(yaml.SafeLoader):
    """SafeLoader that reads ``!!python/...`` nodes as plain data, so nerfstudio is not imported."""


def _construct_python_tag_as_data(loader: yaml.SafeLoader, tag_suffix: str, node: yaml.Node) -> Any:
    """Build the plain-data equivalent of a ``!!python/...`` node.

    Args:
        loader: The active loader.
        tag_suffix: Part of the tag after ``tag:yaml.org,2002:python/`` (unused: data only).
        node: The YAML node to convert.

    Returns:
        A dict for a mapping node, a list for a sequence node (tuples included), else the scalar text.
    """
    # Step 1: pick the plain container from the node kind; the Python type named by the tag is ignored.
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_TagTolerantLoader.add_multi_constructor("tag:yaml.org,2002:python/", _construct_python_tag_as_data)


def _walk_section(document: Mapping[str, Any], key_path: Tuple[str, ...]) -> Mapping[str, Any]:
    """Descend ``key_path`` in a nested mapping.

    Args:
        document: Parsed config.yml.
        key_path: Keys from the root to the wanted section.

    Returns:
        The section mapping.

    Raises:
        KeyError: If a key on the path is missing, or a section on the path is null or not a mapping.
    """
    section = document
    # Step 1: one level per key; a missing key, or a parent that is null / not a mapping (so it
    # cannot hold the key), names the whole path for a readable error.
    for key in key_path:
        if not isinstance(section, Mapping) or key not in section:
            raise KeyError(f"config.yml has no {'.'.join(key_path)} (missing {key!r})")
        section = section[key]
    # Step 2: the wanted section itself must be a mapping too.
    if not isinstance(section, Mapping):
        raise KeyError(f"config.yml has no {'.'.join(key_path)} (not a mapping)")
    return section


def model_settings_from_config_yml(config_path: Path) -> Dict[str, Any]:
    """Read the RGB-decoder and camera settings of a run from its nerfstudio ``config.yml``.

    The file's ``!!python/...`` tags are loaded as plain data by a tag-tolerant SafeLoader, because
    the existing ``io.nerfstudio_config.read_config_scalar`` reads only one single-token scalar per
    call by regex and cannot return the camera list.

    Args:
        config_path: The run's ``config.yml``.

    Returns:
        Dict with ``feature_dim, appearance_dim, rgb_decoder_hidden_dim, rgb_decoder_kernel_size,
        rgb_decoder_num_hidden_blocks, mcmc_min_opacity`` and ``cameras`` (list of camera names in
        the dataparser's order).

    Raises:
        KeyError: If a section or field is absent.
        ValueError: If ``cameras`` is empty, the upstream ``"none"`` sentinel, or not a list of names
            (the real run configs write the explicit camera names as a tuple; upstream
            ``sensor_idx_to_name`` enumerates ``cameras + lidars``, so no name means no camera index).
    """
    # Step 1: parse the whole file with the tag-tolerant loader.
    document = yaml.load(Path(config_path).read_text(), Loader=_TagTolerantLoader)
    # Step 2: copy the model fields.
    model_section = _walk_section(document, MODEL_SECTION_PATH)
    model_settings = {field_name: model_section[field_name] for field_name in MODEL_SETTING_FIELDS}
    # Step 3: the cameras tuple is a list after loading; keep it as a list of str.
    dataparser_section = _walk_section(document, DATAPARSER_SECTION_PATH)
    if "cameras" not in dataparser_section:
        raise KeyError(f"config.yml has no {'.'.join(DATAPARSER_SECTION_PATH)}.cameras")
    raw_cameras = dataparser_section["cameras"]
    # A bare string would iterate per character; only a sequence of names is resolvable.
    if not isinstance(raw_cameras, (list, tuple)):
        raise ValueError(f"dataparser cameras must be a list of names, got {raw_cameras!r}")
    camera_names = [str(camera_name) for camera_name in raw_cameras]
    # Step 4: ad_dataparser turns ("none",) into no cameras, and no cameras leaves nothing to index.
    if not camera_names or (len(camera_names) == 1 and camera_names[0].lower() == NO_CAMERAS_SENTINEL):
        raise ValueError("dataparser cameras is empty or 'none': no camera appearance row to bake with")
    model_settings["cameras"] = camera_names
    return model_settings


def camera_sensor_index(cameras: List[str], camera_name: str) -> int:
    """Row of a camera in the appearance embedding table.

    ad_dataparser enumerates ``cameras + lidars``, so a camera's sensor index is its position in the
    camera list.

    Args:
        cameras: Camera names in dataparser order.
        camera_name: Camera whose index is wanted.

    Returns:
        Position of ``camera_name`` in ``cameras``.

    Raises:
        ValueError: If the camera is not in the list.
    """
    # Step 1: list.index raises ValueError for an unknown name, which is the wanted behaviour.
    return list(cameras).index(camera_name)


def _strip_state_key(state_key: str) -> str:
    """Drop an optional DDP ``module.`` prefix from a state-dict key.

    Args:
        state_key: Key as stored in the checkpoint.

    Returns:
        The key without the leading ``module.``.
    """
    return state_key[len(DDP_KEY_PREFIX):] if state_key.startswith(DDP_KEY_PREFIX) else state_key


def load_upstream_rgb_decoder(pipeline_state: Mapping[str, Any], settings: Dict[str, Any]):
    """Build the upstream ``RGBDecoderCNN`` and load its trained weights.

    Args:
        pipeline_state: The checkpoint's ``pipeline`` state dict (keys may carry ``module.``).
        settings: Output of ``model_settings_from_config_yml``.

    Returns:
        ``(decoder, appearance_weight)``: the decoder in eval mode and the appearance embedding
        table, shape (num_sensors, appearance_dim).

    Raises:
        RuntimeError: If a decoder key is missing or unexpected (strict loading).
        KeyError: If the appearance embedding is absent. Upstream always creates
            ``torch.nn.Embedding(num_sensors, appearance_dim)``, so the key exists even for
            ``appearance_dim = 0`` (weight shape (num_sensors, 0), a zero-width appearance).
    """
    # Step 1: lazy imports keep this module importable without torch / nerfstudio.
    from nerfstudio.models.splatad import RGBDecoderCNN

    # Step 2: construct exactly as SplatADModel does: in_dim = feature_dim + appearance_dim + view dir.
    decoder = RGBDecoderCNN(
        settings["feature_dim"] + settings["appearance_dim"] + VIEW_DIRECTION_DIM,
        hidden_dim=settings["rgb_decoder_hidden_dim"],
        kernel_size=settings["rgb_decoder_kernel_size"],
        num_hidden_blocks=settings["rgb_decoder_num_hidden_blocks"],
    )
    # Step 3: keep the decoder's own keys, with the DDP prefix and the model prefix removed.
    decoder_state = {}
    appearance_weight = None
    for state_key, state_value in pipeline_state.items():
        stripped_key = _strip_state_key(state_key)
        if stripped_key.startswith(RGB_DECODER_KEY_PREFIX):
            decoder_state[stripped_key[len(RGB_DECODER_KEY_PREFIX):]] = state_value
        elif stripped_key == APPEARANCE_EMBEDDING_KEY:
            appearance_weight = state_value
    if appearance_weight is None:
        raise KeyError(f"{APPEARANCE_EMBEDDING_KEY} not in the pipeline state")
    # Step 4: strict load, so a settings/checkpoint mismatch fails loudly; then inference mode.
    decoder.load_state_dict(decoder_state, strict=True)
    decoder.eval()
    return decoder, appearance_weight.detach().float().cpu()


@transform("T85", "Per-Gaussian RGB baked with the upstream RGBDecoderCNN on a uniform feature field")
def bake_gaussian_colours(
    decoder,
    features_dc,
    features_rest,
    appearance_vector,
    directions,
    chunk_size: int,
) -> np.ndarray:
    """Bake one RGB per Gaussian and view direction with the upstream decoder.

    Why this equals the upstream decoder on a uniform feature field: ``RGBDecoderCNN.net`` applies
    stride-1 k x k convolutions with zero ``padding = k // 2`` (k = ``rgb_decoder_kernel_size``, odd
    so the padding is symmetric). On an image whose every pixel holds the same feature vector, the
    pixels farther than the receptive radius from every border are unaffected by the padding. A 1x1
    image whose convolutions use ``padding_mode="replicate"`` has every padded neighbour equal to the
    pixel itself, i.e. it is a spatially uniform patch of that Gaussian's feature, so its single
    output equals the centre pixel of a large uniform patch. The colour is then
    ``albedo * (1 + spec[:3]) + spec[3:]`` clamped to [0, 1], as the upstream forward and model do;
    the model's ``(1 - alpha) * background`` term is taken as zero (fully opaque pixel).

    This approximates the real render: there the decoder sees alpha-blended features of many
    Gaussians and their neighbours, so the real pixel colour is not any single Gaussian's colour.

    Cost: N * M single-pixel network evaluations; working memory per forward scales with
    ``chunk_size``, and the returned array holds N * M * 3 float32 values.

    Args:
        decoder: Upstream ``RGBDecoderCNN`` (not modified; a deep copy is used).
        features_dc: (N, 3) albedo features.
        features_rest: (N, F) remaining features.
        appearance_vector: (A,) appearance embedding of the chosen camera.
        directions: (M, 3) unit ray directions (camera towards scene) in the dataparser/world frame,
            as upstream ``get_ray_dirs_pinhole`` produces them.
        chunk_size: Gaussians per forward batch (bounds memory); at least 1.

    Returns:
        (N, M, 3) float32 array of RGB in [0, 1].

    Raises:
        ValueError: If ``chunk_size < 1`` (``range`` with step 0 or a negative step would raise
            obscurely or silently return empty colours).
    """
    import torch

    # Step 0: validate before allocating the output.
    if chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size must be >= {MIN_CHUNK_SIZE}, got {chunk_size}")
    # Step 1: a CPU float32 copy (inputs are CPU float32 tensors, so a CUDA or float64 decoder must not
    # be used as is) with
    # replicate padding, so the caller's zero-padded module stays unchanged. The albedo width is the
    # upstream decoder's own ``skip_dim``.
    replicate_decoder = copy.deepcopy(decoder).cpu().float().eval()
    albedo_channel_count = replicate_decoder.skip_dim
    for module in replicate_decoder.modules():
        if isinstance(module, torch.nn.Conv2d):
            module.padding_mode = "replicate"
    gaussian_count = features_dc.shape[0]
    direction_count = directions.shape[0]
    baked_colours = np.empty((gaussian_count, direction_count, albedo_channel_count), dtype=np.float32)
    # Step 2: plain float32 CPU tensors, whatever the caller passed.
    features_dc = torch.as_tensor(features_dc, dtype=torch.float32, device="cpu")
    features_rest = torch.as_tensor(features_rest, dtype=torch.float32, device="cpu")
    appearance_vector = torch.as_tensor(appearance_vector, dtype=torch.float32, device="cpu")
    directions = torch.as_tensor(directions, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        for direction_index in range(direction_count):
            for chunk_start in range(0, gaussian_count, chunk_size):
                chunk_stop = min(chunk_start + chunk_size, gaussian_count)
                chunk_length = chunk_stop - chunk_start
                # Step 3: spec input channels, in the order of the upstream forward:
                # features_rest, appearance embedding, ray direction (same for all Gaussians).
                appearance_block = appearance_vector.expand(chunk_length, -1)
                direction_block = directions[direction_index].expand(chunk_length, -1)
                spec_input = torch.cat(
                    [features_rest[chunk_start:chunk_stop], appearance_block, direction_block], dim=-1
                )
                # Step 4: (B, C, 1, 1) images through the upstream network.
                network_output = replicate_decoder.net(spec_input[:, :, None, None])[:, :, 0, 0]
                # Step 5: upstream combine albedo * (1 + gain) + offset, then the model's clamp.
                albedo = features_dc[chunk_start:chunk_stop]
                gain = network_output[:, :albedo_channel_count]
                offset = network_output[:, albedo_channel_count:]
                chunk_rgb = torch.clamp(albedo * (1 + gain) + offset, RGB_MIN, RGB_MAX)
                baked_colours[chunk_start:chunk_stop, direction_index] = chunk_rgb.numpy()
    return baked_colours
