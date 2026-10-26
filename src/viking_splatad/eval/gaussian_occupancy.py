"""Gaussians of a trained SplatAD run in the map frame, for occupancy evaluation (P2, T56).

PLAN §5 P2 scores the model's own Gaussians against the GT cuboid grid, without rendering, so a
trunk hidden behind grass Gaussians still counts. This module is the numpy-only half of T56:

    * pick the Gaussian parameters out of a checkpoint's ``pipeline`` state dict
      (``_model.gauss_params.<name>``, one row per Gaussian);
    * decode them as the rasteriser does: std per axis = exp(log-scale), opacity = sigmoid(logit),
      rotation from the normalised (w, x, y, z) quaternion, Σ = R·diag(std)²·Rᵀ
      (gsplat ``_quat_scale_to_covar_preci``, ``quat_to_rotmat``; splatad.py:980-981);
    * move them from the dataparser frame to the map frame with the same inverse(w2m) as T50;
    * save them as ``gaussians.npz`` (keys in GAUSSIANS_NPZ_KEYS, documented in
      runs/RUN_FOLDER_GLOSSARY.md).

Reading the ``.ckpt`` needs torch, so ``scripts/10_export_gaussians.py`` does that in the SplatAD
environment and hands plain numpy arrays to this module.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from viking_splatad.registry import transform
from viking_splatad.splatad.rendered_lidar import lidar_pose_in_map_frame

# State-dict key prefix of the Gaussian parameters: the pipeline holds the model as ``_model``,
# whose ParameterDict is ``gauss_params`` (base_pipeline.py:294, splatad.py:439-464).
GAUSS_PARAMS_KEY_PREFIX = "_model.gauss_params."
# Buffer whose row count is the number of dynamic actors (dynamic_actors.py:156). Gaussians with
# id < actor count live in their actor's box frame; static ones have id = actor count
# (splatad.py:594-596, 849).
ACTOR_SIZES_KEY = "_model.dynamic_actors.actor_sizes"
# Prefix DDP adds to every key; nerfstudio strips it on load (base_pipeline.py:439-441).
DDP_KEY_PREFIX = "module."
# Gaussian parameters T56 reads, by their gauss_params name.
GAUSS_PARAM_NAMES = ("means", "scales", "quats", "opacities", "id")
# nerfstudio checkpoint names, step-<zero-padded step>.ckpt (trainer.py:513).
CHECKPOINT_NAME_PATTERN = re.compile(r"^step-(\d+)\.ckpt$")
# Keys of gaussians.npz. runs/RUN_FOLDER_GLOSSARY.md documents each one (a test keeps the two in
# sync), and save_gaussians_npz refuses to write any other key set.
GAUSSIANS_NPZ_KEYS = (
    "means", "covariances", "axis_std", "opacities", "dataparser_transform", "checkpoint_step",
)


@dataclass(frozen=True)
class MapFrameGaussians:
    """Decoded static Gaussians in the map frame.

    Attributes:
        means: (N, 3) float64 centres μ_k.
        covariances: (N, 3, 3) float64 Σ_k, symmetric positive definite.
        axis_std: (N, 3) float64 std along each principal axis, exp(log-scale); the 3σ ellipsoid
            fits in a sphere of radius 3·max(axis_std) (rotation-invariant).
        opacities: (N,) float64 peak alpha σ(o_k) in (0, 1).
    """

    means: np.ndarray
    covariances: np.ndarray
    axis_std: np.ndarray
    opacities: np.ndarray


def gauss_params_from_state_dict(pipeline_state: Mapping[str, np.ndarray]) -> dict:
    """Pick the Gaussian parameters and actor count out of a checkpoint's pipeline state dict.

    Args:
        pipeline_state: ``torch.load(ckpt)["pipeline"]`` with every tensor converted to numpy
            (keys may carry the DDP ``module.`` prefix).

    Returns:
        Dict with one (N, ...) array per GAUSS_PARAM_NAMES entry, plus ``actor_count`` (int).

    Raises:
        KeyError: If a Gaussian parameter or the actor buffer is missing.
        ValueError: If the parameters disagree on the Gaussian count N.
    """
    # Step 1: drop the DDP prefix, as nerfstudio's load_pipeline does.
    unprefixed_state = {
        (state_key[len(DDP_KEY_PREFIX):] if state_key.startswith(DDP_KEY_PREFIX) else state_key): value
        for state_key, value in pipeline_state.items()
    }
    # Step 2: the parameters by name; a missing one is a format change, so fail loudly.
    missing_keys = [GAUSS_PARAMS_KEY_PREFIX + param_name for param_name in GAUSS_PARAM_NAMES
                    if GAUSS_PARAMS_KEY_PREFIX + param_name not in unprefixed_state]
    if ACTOR_SIZES_KEY not in unprefixed_state:
        missing_keys.append(ACTOR_SIZES_KEY)
    if missing_keys:
        raise KeyError(f"checkpoint pipeline state lacks {missing_keys}")
    gauss_params = {param_name: np.asarray(unprefixed_state[GAUSS_PARAMS_KEY_PREFIX + param_name])
                    for param_name in GAUSS_PARAM_NAMES}
    # Step 3: every parameter has one row per Gaussian.
    row_counts = {param_name: param_array.shape[0] for param_name, param_array in gauss_params.items()}
    if len(set(row_counts.values())) != 1:
        raise ValueError(f"Gaussian parameters disagree on the row count: {row_counts}")
    gauss_params["actor_count"] = int(np.asarray(unprefixed_state[ACTOR_SIZES_KEY]).shape[0])
    return gauss_params


def quaternions_to_rotations(quaternions_wxyz: np.ndarray) -> np.ndarray:
    """Rotation matrices of unnormalised (w, x, y, z) quaternions, as gsplat builds them.

    After normalising q = (w, x, y, z), R = I + 2w[v]ₓ + 2[v]ₓ² with v = (x, y, z), written out
    entry by entry (gsplat ``_quat_scale_to_covar_preci``). Cost O(N).

    Args:
        quaternions_wxyz: (N, 4) quaternions, any non-zero norm.

    Returns:
        (N, 3, 3) float64 rotations.

    Raises:
        ValueError: If a quaternion has zero norm (no rotation defined).
    """
    quaternions = np.asarray(quaternions_wxyz, dtype=np.float64).reshape(-1, 4)
    # Step 1: unit quaternions (gsplat normalises before every use, so the stored norm is free).
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("zero-norm quaternion")
    w_part, x_part, y_part, z_part = (quaternions / norms).T
    # Step 2: the standard unit-quaternion rotation matrix, row by row.
    rotations = np.stack([
        1 - 2 * (y_part ** 2 + z_part ** 2), 2 * (x_part * y_part - w_part * z_part),
        2 * (x_part * z_part + w_part * y_part),
        2 * (x_part * y_part + w_part * z_part), 1 - 2 * (x_part ** 2 + z_part ** 2),
        2 * (y_part * z_part - w_part * x_part),
        2 * (x_part * z_part - w_part * y_part), 2 * (y_part * z_part + w_part * x_part),
        1 - 2 * (x_part ** 2 + y_part ** 2),
    ], axis=-1)
    return rotations.reshape(-1, 3, 3)


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Logistic function σ(z) = 1 / (1 + e^(−z)), without overflow.

    Uses the identity σ(z) = ½(1 + tanh(z/2)), which is finite for every real z.

    Args:
        logits: Any-shape array.

    Returns:
        float64 array of the same shape, values in [0, 1].
    """
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(logits, dtype=np.float64)))


@transform("T56", "Checkpoint Gaussians -> map-frame means, covariances R·diag(e^s)²·Rᵀ, opacities sigmoid(o)")
def checkpoint_gaussians_to_map_frame(
    means: np.ndarray,
    log_scales: np.ndarray,
    quaternions_wxyz: np.ndarray,
    opacity_logits: np.ndarray,
    actor_ids: np.ndarray,
    actor_count: int,
    dataparser_transform: np.ndarray,
    dataparser_scale: float,
) -> MapFrameGaussians:
    """Decode a run's static Gaussians and move them from the dataparser frame to the map frame.

    Dataparser frame: x_dp = W·x_map with W = [A | t] rigid (w2m, scale 1). Hence
    x_map = Aᵀ(x_dp − t) for the centres, and a Gaussian N(μ, Σ) maps to N(Aᵀ(μ − t), AᵀΣA).
    inverse(W) is taken from T50 (``lidar_pose_in_map_frame`` with an identity pose), so the
    Gaussians and the rendered scans share the exact same map frame.

    Args:
        means: (N, 3) centres, dataparser frame (``gauss_params.means``).
        log_scales: (N, 3) log std per axis (``gauss_params.scales``).
        quaternions_wxyz: (N, 4) unnormalised rotations (``gauss_params.quats``).
        opacity_logits: (N,) or (N, 1) logit opacities (``gauss_params.opacities``).
        actor_ids: (N,) or (N, 1) actor index per Gaussian (``gauss_params.id``).
        actor_count: Number of dynamic actors; ids >= actor_count are static.
        dataparser_transform: w2m, (3, 4) or (4, 4) (``dataparser_transforms.json`` "transform").
        dataparser_scale: ``dataparser_transforms.json`` "scale".

    Returns:
        The decoded Gaussians in the map frame.

    Raises:
        ValueError: If any Gaussian belongs to an actor (its mean is in the actor's box frame, so
            no single map-frame transform applies), the scale is not 1 (T50 assumes 1), or the
            transform is not rigid (raised by T50).
    """
    # Step 1: only static Gaussians have dataparser-frame means.
    actor_ids = np.asarray(actor_ids).reshape(-1)
    actor_gaussian_count = int(np.count_nonzero(actor_ids < actor_count))
    if actor_gaussian_count:
        raise ValueError(f"{actor_gaussian_count} Gaussians belong to dynamic actors; "
                         "their means are in actor box frames")
    # The rendered scans (T50) assume scale 1; a different scale would desynchronise the two.
    if dataparser_scale != 1.0:
        raise ValueError(f"dataparser scale {dataparser_scale} != 1 (T50 assumes 1)")
    # Step 2: inverse(W) = [Aᵀ | −Aᵀt], via T50 (which also checks that A is orthonormal).
    dataparser_to_map = lidar_pose_in_map_frame(np.eye(4), dataparser_transform)
    rotation_to_map = dataparser_to_map[:3, :3]
    translation_to_map = dataparser_to_map[:3, 3]
    # Step 3: decode the rasteriser's parameterisation.
    axis_std = np.exp(np.asarray(log_scales, dtype=np.float64).reshape(-1, 3))
    rotations_dataparser = quaternions_to_rotations(quaternions_wxyz)
    opacities = stable_sigmoid(np.asarray(opacity_logits).reshape(-1))
    # Step 4: rotate each Gaussian's axes into the map frame (R_map = Aᵀ·R), then
    # Σ_map = R_map·diag(std²)·R_mapᵀ = AᵀΣ_dpA. Scaling R's columns by std gives M = R·diag(std),
    # and Σ = M·Mᵀ; einsum keeps it one O(N) pass.
    rotations_map = np.einsum("ij,njk->nik", rotation_to_map, rotations_dataparser)
    scaled_axes = rotations_map * axis_std[:, None, :]
    covariances = np.einsum("nij,nkj->nik", scaled_axes, scaled_axes)
    # Step 5: centres, x_map = Aᵀx_dp − Aᵀt.
    means_map = np.asarray(means, dtype=np.float64).reshape(-1, 3) @ rotation_to_map.T + translation_to_map
    return MapFrameGaussians(means=means_map, covariances=covariances, axis_std=axis_std, opacities=opacities)


def latest_checkpoint_path(checkpoint_dir: Path) -> Path:
    """The highest-step ``step-<n>.ckpt`` in a directory (the one ``eval_setup`` and ``08`` load).

    Args:
        checkpoint_dir: A run's ``nerfstudio_models`` directory.

    Returns:
        Path of the checkpoint with the largest step.

    Raises:
        FileNotFoundError: If the directory holds no checkpoint.
    """
    # Compare steps as integers, so the choice does not depend on the zero padding.
    steps_by_path = {}
    for candidate_path in Path(checkpoint_dir).iterdir():
        name_match = CHECKPOINT_NAME_PATTERN.match(candidate_path.name)
        if name_match is not None:
            steps_by_path[candidate_path] = int(name_match.group(1))
    if not steps_by_path:
        raise FileNotFoundError(f"no step-*.ckpt in {checkpoint_dir}")
    return max(steps_by_path, key=steps_by_path.get)


def save_gaussians_npz(output_path: Path, gaussians: MapFrameGaussians, dataparser_transform: np.ndarray,
                       checkpoint_step: int) -> None:
    """Write ``gaussians.npz`` (keys GAUSSIANS_NPZ_KEYS) with float32 arrays and provenance.

    float32 keeps μ to ~1e-7 relative precision (sub-micrometre within a few hundred metres), far
    below any voxel size, at half the file size.

    Args:
        output_path: Destination ``.npz`` (parent directories are created).
        gaussians: Output of ``checkpoint_gaussians_to_map_frame``.
        dataparser_transform: The w2m used, stored as (3, 4).
        checkpoint_step: Training step of the source checkpoint.

    Raises:
        ValueError: If the record's keys differ from GAUSSIANS_NPZ_KEYS.
    """
    record = {
        "means": gaussians.means.astype(np.float32),
        "covariances": gaussians.covariances.astype(np.float32),
        "axis_std": gaussians.axis_std.astype(np.float32),
        "opacities": gaussians.opacities.astype(np.float32),
        "dataparser_transform": np.asarray(dataparser_transform, dtype=np.float32)[:3, :],
        "checkpoint_step": np.int64(checkpoint_step),
    }
    # runs/RUN_FOLDER_GLOSSARY.md documents exactly GAUSSIANS_NPZ_KEYS.
    if set(record) != set(GAUSSIANS_NPZ_KEYS):
        raise ValueError(f"npz keys {sorted(record)} differ from GAUSSIANS_NPZ_KEYS")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **record)
