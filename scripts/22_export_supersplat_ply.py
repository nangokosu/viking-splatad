"""Export a trained SplatAD run's Gaussians as a SuperSplat-readable 3DGS ``.ply`` (T82-T86).

Wires the export modules together for one run, or for every fetched run with ``--all``:

    1. Latest ``step-*.ckpt`` of ``<run>/train/<session>/<method>/<run_id>/nerfstudio_models/`` (the
       train dir named by ``job_info.txt``) loaded on the CPU; only
       the Gaussian parameters, ``features_dc`` / ``features_rest``, the RGB decoder and the
       appearance embedding are kept.
    2. Map-frame geometry with T56 (``checkpoint_gaussians_to_map_frame``).
    3. Crop (T86, unless ``--no-crop``): the box of the train lidar sweeps in the map frame, and
       ``opacity >= mcmc_min_opacity`` from the run's ``config.yml``.
    4. Colour (T85 + T84): SplatAD stores no RGB per Gaussian, so the upstream RGB decoder is run on
       each Gaussian's features along M Fibonacci directions and fitted with SH (degree 0 = the mean).
       Directions are generated in the map frame, d_dp = A d_map for the decoder (the decoder was
       trained in the dataparser frame x_dp = A x_map + t; direction vectors ignore t). The ray
       convention is camera -> scene, the sign of 3DGS' ``dir = mean - campos``. The appearance vector
       is the embedding row of the run's camera (first camera of the config, a note is printed if
       there are several).
    5. Quaternions from the principal axes (T83), ``log_scales = log(axis_std)``, opacity logits as
       stored in the checkpoint (exactly the logit of the activated opacity, so no clipping of 0 / 1
       is needed), then ``<run>/renders/supersplat.ply`` (T82, atomic write).

The session (and so the default dataset ``data/intermediate/<session>``) is the folder under
``train/``, the same one ``scripts/09_extract_run_split.py`` reads from ``job_info.txt``.

Needs torch and nerfstudio (RGB decoder), so it runs in the reference image. The repo is mounted
read-only by ``scripts/test_in_image.sh``; set ``VIKING_WRITABLE_DIR`` (repo-relative) to make one
folder writable, e.g. the run's ``renders``:

    VIKING_WRITABLE_DIR=runs/<run>/renders scripts/test_in_image.sh \\
        python scripts/22_export_supersplat_ply.py --run-dir runs/<run> --sh-degree 0

Which is, spelled out, the ``docker run`` of test_in_image.sh with one extra
``-v "$PWD/runs/<run>/renders":/workspace/viking-splatad/runs/<run>/renders``.

``--dry-run`` does everything except the write (checkpoint, crop, colour bake, SH fit, quaternions)
and reports the projected file size and the peak resident memory, so it needs no writable folder.
"""

import argparse
import functools
import glob
import importlib.util
import json
import resource
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from viking_splatad.export.spherical_harmonics import COLOUR_CHANNEL_COUNT, MAX_SH_DEGREE

# Repository root: this file lives in <root>/scripts/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Default runs folder scanned by --all, and default dataset parent (as scripts/09).
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "runs"
DEFAULT_DATASET_PARENT = PROJECT_ROOT / "data" / "intermediate"
# Files and folders of a run (see runs/RUN_FOLDER_GLOSSARY.md).
TRAIN_FOLDER_NAME = "train"
CHECKPOINT_DIR_NAME = "nerfstudio_models"
CONFIG_FILE_NAME = "config.yml"
DATAPARSER_TRANSFORMS_FILE_NAME = "dataparser_transforms.json"
RENDERS_FOLDER_NAME = "renders"
OUTPUT_FILE_NAME = "supersplat.ply"
# Names of the SH feature tensors inside ``_model.gauss_params`` (splatad.py).
FEATURES_DC_NAME = "features_dc"
FEATURES_REST_NAME = "features_rest"
# Default --sh-degree: 0 = one plain colour per Gaussian (no view dependence, smallest file).
DEFAULT_SH_DEGREE = 0
# Bytes per MiB, for printing sizes.
BYTES_PER_MIB = 1024 * 1024
# Bytes of one float32 PLY field and of one float32 network value.
PLY_FIELD_BYTES = np.dtype(np.float32).itemsize
FLOAT32_BYTES = np.dtype(np.float32).itemsize
# Bytes per KiB: ru_maxrss is in KiB on Linux (the image) and in bytes on macOS.
BYTES_PER_KIB = 1024
# run folder file with the run's identity, and the keys read from it (as scripts/09_extract_run_split.py).
JOB_INFO_FILE_NAME = "job_info.txt"
JOB_INFO_RUN_ID_KEY, JOB_INFO_SESSION_KEY, JOB_INFO_METHOD_KEY = "RUN_ID", "SESSION", "TRAIN_METHOD"
# Script whose ``parse_job_info`` is reused (its file name starts with a digit, so it is loaded by path).
RUN_SPLIT_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "09_extract_run_split.py"
# Free space always left on the output filesystem. 128 MiB is a judgement, not a measured need: it
# covers filesystem metadata and what other processes write during an export of minutes, and is small
# next to a typical ply (100+ MiB) so that it only bites when the disk is nearly full. It does not
# change any result.
DISK_SAFETY_MARGIN_BYTES = 128 * BYTES_PER_MIB
# Memory budget of the decoder forward batch (activations of one batch). Same judgement as
# CHUNK_MEMORY_BUDGET_BYTES of the SH fit: small next to the (N, M, 3) data, large enough for efficient
# convolutions. It does not change any result.
DECODER_MEMORY_BUDGET_BYTES = 256 * BYTES_PER_MIB
# Tensors alive at once in one decoder convolution block: the replicate-padded input, the convolution
# output and the activation output (an upper bound; autograd is off so nothing else is kept).
DECODER_LIVE_TENSOR_COUNT = 3


@contextmanager
def timed_step(step_name: str, step_seconds: Dict[str, float]) -> Iterator[None]:
    """Measure the wall time of a ``with`` block into ``step_seconds[step_name]``.

    Args:
        step_name: Key of the measurement.
        step_seconds: Dict that receives the elapsed seconds (``time.perf_counter``).

    Yields:
        Nothing.
    """
    # Step 1: monotonic clock, recorded even if the block raises.
    start_time = time.perf_counter()
    try:
        yield
    finally:
        step_seconds[step_name] = time.perf_counter() - start_time


@functools.lru_cache(maxsize=None)
def load_job_info_parser() -> Callable[[Path], Dict[str, str]]:
    """Import ``parse_job_info`` from scripts/09_extract_run_split.py (reuse, not a copy).

    Returns:
        The function reading the top-level ``KEY: value`` lines of a ``job_info.txt``.
    """
    # Step 1: the file name starts with a digit, so import it by path.
    module_spec = importlib.util.spec_from_file_location("extract_run_split_script", RUN_SPLIT_SCRIPT_PATH)
    script_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(script_module)
    return script_module.parse_job_info


def find_train_dir(run_dir: Path) -> Path:
    """Resolve ``<run>/train/<session>/<method>/<run_id>/`` from the run's own ``job_info.txt``.

    Only the folder that ``job_info.txt`` names (RUN_ID, SESSION, TRAIN_METHOD) is accepted, never
    the first folder found. A run without TRAIN_METHOD (older job_info.txt) may match any method
    folder, but then exactly one folder must qualify.

    Args:
        run_dir: Run folder.

    Returns:
        The train dir (holds ``config.yml`` and the checkpoint folder).

    Raises:
        FileNotFoundError: No ``job_info.txt``, or no matching folder with config and checkpoints.
        ValueError: ``job_info.txt`` lacks RUN_ID / SESSION, or several folders match.
    """
    run_dir = Path(run_dir)
    # Step 1: identity of the run.
    job_info_path = run_dir / JOB_INFO_FILE_NAME
    if not job_info_path.is_file():
        raise FileNotFoundError(f"{job_info_path} not found")
    job_info = load_job_info_parser()(job_info_path)
    missing_keys = [key for key in (JOB_INFO_RUN_ID_KEY, JOB_INFO_SESSION_KEY) if not job_info.get(key)]
    if missing_keys:
        raise ValueError(f"{job_info_path} lacks {missing_keys}")
    # Step 2: train/<session>/<method>/<run_id>, method wildcarded only when unrecorded.
    method_pattern = glob.escape(job_info[JOB_INFO_METHOD_KEY]) if job_info.get(JOB_INFO_METHOD_KEY) else "*"
    candidate_dirs = sorted(run_dir.glob(
        f"{TRAIN_FOLDER_NAME}/{glob.escape(job_info[JOB_INFO_SESSION_KEY])}/{method_pattern}/"
        f"{glob.escape(job_info[JOB_INFO_RUN_ID_KEY])}"))
    usable_dirs = [candidate_dir for candidate_dir in candidate_dirs
                   if (candidate_dir / CONFIG_FILE_NAME).is_file()
                   and (candidate_dir / CHECKPOINT_DIR_NAME).is_dir()]
    # Step 3: exactly one.
    if not usable_dirs:
        raise FileNotFoundError(f"no {TRAIN_FOLDER_NAME}/<session>/<method>/<run_id> with {CONFIG_FILE_NAME} "
                                f"matching {JOB_INFO_FILE_NAME} in {run_dir}")
    if len(usable_dirs) > 1:
        raise ValueError(f"ambiguous train dirs in {run_dir}: {[str(path) for path in usable_dirs]}")
    return usable_dirs[0]


def session_of_train_dir(train_dir: Path) -> str:
    """Session name = the folder two levels above the train dir (``train/<session>/<method>/<id>``).

    Args:
        train_dir: Output of ``find_train_dir``.

    Returns:
        The session name, e.g. ``may_short_grass``.
    """
    return train_dir.parent.parent.name


def load_checkpoint_tensors(checkpoint_path: Path) -> dict:
    """Load a checkpoint on the CPU and keep only the tensors the export needs.

    Args:
        checkpoint_path: ``step-*.ckpt``.

    Returns:
        Dict with ``step`` (int), ``gauss_params`` (T56 input arrays), ``features_dc`` (N, 3) and
        ``features_rest`` (N, F) float32 numpy, and ``pipeline_state``: the torch tensors of the RGB
        decoder and appearance embedding (keys as stored).

    Raises:
        KeyError: If a feature tensor, a Gaussian parameter or the actor buffer is missing.
        ValueError: If the Gaussian parameters disagree on the row count (``gauss_params_from_state_dict``).
    """
    import torch  # pylint: disable=import-outside-toplevel
    from viking_splatad.eval.gaussian_occupancy import (  # pylint: disable=import-outside-toplevel
        ACTOR_SIZES_KEY, DDP_KEY_PREFIX, GAUSS_PARAM_NAMES, GAUSS_PARAMS_KEY_PREFIX,
        gauss_params_from_state_dict)
    from viking_splatad.export.splatad_colours import (  # pylint: disable=import-outside-toplevel
        APPEARANCE_EMBEDDING_KEY, RGB_DECODER_KEY_PREFIX)

    # Step 1: whole checkpoint on the CPU (as scripts/10); the optimiser state is dropped below.
    loaded_state = torch.load(checkpoint_path, map_location="cpu")
    full_state = loaded_state["pipeline"]
    # Step 2: sort keys (DDP prefix stripped) into geometry tensors and decoder tensors.
    feature_keys = {GAUSS_PARAMS_KEY_PREFIX + FEATURES_DC_NAME: FEATURES_DC_NAME,
                    GAUSS_PARAMS_KEY_PREFIX + FEATURES_REST_NAME: FEATURES_REST_NAME}
    geometry_keys = tuple(GAUSS_PARAMS_KEY_PREFIX + param_name for param_name in GAUSS_PARAM_NAMES)
    geometry_keys += (ACTOR_SIZES_KEY,)
    geometry_state, decoder_state, features = {}, {}, {}
    for state_key, state_value in full_state.items():
        stripped_key = state_key.removeprefix(DDP_KEY_PREFIX)
        if stripped_key in geometry_keys:
            geometry_state[stripped_key] = state_value.detach().cpu().numpy()
        elif stripped_key in feature_keys:
            features[feature_keys[stripped_key]] = state_value.detach().cpu().numpy().astype(np.float32)
        elif stripped_key.startswith(RGB_DECODER_KEY_PREFIX) or stripped_key == APPEARANCE_EMBEDDING_KEY:
            decoder_state[stripped_key] = state_value.detach().cpu()
    missing_features = [name for name in feature_keys.values() if name not in features]
    if missing_features:
        raise KeyError(f"checkpoint pipeline state lacks {missing_features}")
    checkpoint_step = int(loaded_state["step"])
    # Step 3: free the big checkpoint (optimiser moments) before the heavy steps.
    del loaded_state, full_state
    return {"step": checkpoint_step, "gauss_params": gauss_params_from_state_dict(geometry_state),
            "features_dc": features[FEATURES_DC_NAME].reshape(-1, COLOUR_CHANNEL_COUNT),
            "features_rest": features[FEATURES_REST_NAME].reshape(features[FEATURES_REST_NAME].shape[0], -1),
            "pipeline_state": decoder_state}


def decoder_batch_size_for_budget(decoder, memory_budget_bytes: int = DECODER_MEMORY_BUDGET_BYTES) -> int:
    """Gaussians per decoder forward so that one batch's activations fit the memory budget.

    Each Gaussian is a 1 x 1 image run with replicate padding, and a convolution with padding p
    turns it into a (1 + 2p) x (1 + 2p) padded image, p = kernel // 2. The widest layer holds
    ``widest = max over convolutions of max(in_channels, out_channels)`` channels of it, so per Gaussian

        working_bytes = DECODER_LIVE_TENSOR_COUNT * widest * (1 + 2p)^2 * 4 bytes (float32)

    and ``batch = max(1, floor(budget / working_bytes))``. Both ``widest`` and ``p`` are read from the
    decoder's convolutions, so a different hidden width or kernel size changes the batch.

    Args:
        decoder: Upstream ``RGBDecoderCNN``.
        memory_budget_bytes: Byte budget of one batch's activations.

    Returns:
        Batch size >= 1.

    Raises:
        ValueError: If the decoder has no convolution.
    """
    import torch  # pylint: disable=import-outside-toplevel

    # Step 1: the decoder's convolutions give the padding and the layer widths.
    convolutions = [module for module in decoder.modules() if isinstance(module, torch.nn.Conv2d)]
    if not convolutions:
        raise ValueError("decoder has no Conv2d layer")
    widest_channel_count = max(max(convolution.in_channels, convolution.out_channels)
                               for convolution in convolutions)
    padding = max(convolution.padding[0] for convolution in convolutions)
    # Step 2: bytes of one Gaussian's padded activations, then how many fit.
    padded_pixel_count = (1 + 2 * padding) ** 2
    working_bytes = DECODER_LIVE_TENSOR_COUNT * widest_channel_count * padded_pixel_count * FLOAT32_BYTES
    return max(1, memory_budget_bytes // working_bytes)


def rest_coefficients_per_channel(sh_degree: int) -> int:
    """Number of higher-order SH coefficients per colour channel, K = (degree + 1)^2 - 1.

    Args:
        sh_degree: SH degree 0..3.

    Returns:
        K (0 for degree 0).
    """
    # (degree + 1)^2 basis functions in all; the first is the DC term.
    return (sh_degree + 1) ** 2 - 1


def peak_rss_bytes() -> int:
    """Peak resident set size of this process so far.

    Returns:
        Bytes (``ru_maxrss`` is KiB on Linux and bytes on macOS).
    """
    peak_value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak_value if sys.platform == "darwin" else peak_value * BYTES_PER_KIB


def free_bytes_at(path: Path) -> int:
    """Free bytes of the filesystem holding ``path`` (its nearest existing parent).

    Args:
        path: A file or folder that may not exist yet.

    Returns:
        Free bytes for an unprivileged writer.
    """
    existing_path = Path(path)
    while not existing_path.exists():
        existing_path = existing_path.parent
    return shutil.disk_usage(existing_path).free


def ply_byte_count(vertex_count: int, rest_per_channel: int) -> int:
    """Exact size of the ply that ``write_supersplat_ply`` writes: header + rows.

    Args:
        vertex_count: Number of Gaussians N.
        rest_per_channel: K higher-order coefficients per channel.

    Returns:
        ``len(header) + N * (number of fields) * 4`` bytes.
    """
    from viking_splatad.export.supersplat_ply import (  # pylint: disable=import-outside-toplevel
        PLY_END_HEADER_LINE, PLY_FORMAT_LINE, PLY_MAGIC_LINE, PLY_PROPERTY_PREFIX, PLY_VERTEX_PREFIX,
        build_field_names)

    # Step 1: header text exactly as the writer builds it (one line per property).
    field_names = build_field_names(rest_per_channel)
    header_lines = [PLY_MAGIC_LINE, PLY_FORMAT_LINE, f"{PLY_VERTEX_PREFIX}{vertex_count}"]
    header_lines += [f"{PLY_PROPERTY_PREFIX}{name}" for name in field_names]
    header_lines.append(PLY_END_HEADER_LINE)
    header_byte_count = len(("\n".join(header_lines) + "\n").encode("ascii"))
    # Step 2: every row is len(field_names) float32 values.
    return header_byte_count + vertex_count * len(field_names) * PLY_FIELD_BYTES


class InsufficientDiskSpaceError(Exception):
    """The projected ply (plus the safety margin) does not fit the free disk space."""


def rotate_directions_to_dataparser(directions_map: np.ndarray, dataparser_transform: np.ndarray) -> np.ndarray:
    """Rotate map-frame directions into the dataparser frame: d_dp = A d_map, renormalised.

    The decoder was trained in the dataparser frame x_dp = A x_map + t; a direction is a difference of
    two points, so t cancels. A is a rotation times a uniform scale s, so |d_dp| = s |d_map|; dividing
    by the norm restores unit length, the decoder's input contract whatever s is. Scripts 22 and 23 both
    use this one function so the export and its check share one convention.

    Args:
        directions_map: float [N,3] unit directions in the map frame.
        dataparser_transform: float [3,3], [3,4] or [4,4] A (or the full transform): only [:3, :3] is used.

    Returns:
        float [N,3] unit directions in the dataparser frame.
    """
    rotated = directions_map @ np.asarray(dataparser_transform, dtype=np.float64)[:3, :3].T
    return rotated / np.linalg.norm(rotated, axis=1, keepdims=True)


def raise_system_exit_on_sigterm() -> None:
    """Turn SIGTERM (sent by ``timeout`` or Slurm) into ``SystemExit`` so ``finally`` / cleanup code runs.

    By default SIGTERM kills Python without unwinding, which would leave the atomic-write temp file
    behind; ``write_supersplat_ply`` removes it on any exception, and SystemExit is one.

    Returns:
        None.
    """
    def raise_exit(signal_number, stack_frame):  # pylint: disable=unused-argument
        raise SystemExit(f"terminated by signal {signal_number}")

    signal.signal(signal.SIGTERM, raise_exit)


def bake_sh_colours(
    decoder, features_dc: np.ndarray, features_rest: np.ndarray, appearance_vector: np.ndarray,
    directions_map: np.ndarray, rotation_map_to_dataparser: np.ndarray, sh_degree: int,
):
    """Bake colours with the upstream decoder and fit SH coefficients, streamed in chunks.

    For chunk c the (n, M, 3) colours exist only inside the loop; the fit writes into preallocated
    (N, 3) / (N, 3, K) arrays, so memory is O(chunk * M) rather than O(N * M). The decoder forward
    uses its own batch (``decoder_batch_size_for_budget``), separate from the fit chunk.

    Args:
        decoder: Upstream RGB decoder (eval mode).
        features_dc: (N, 3) albedo features.
        features_rest: (N, F) remaining features.
        appearance_vector: (A,) embedding row of the camera.
        directions_map: (M, 3) unit directions, map frame, M = oversampling * (degree+1)^2.
        rotation_map_to_dataparser: A of x_dp = A x_map + t; directions rotate with A only.
        sh_degree: SH degree 0..3.

    Returns:
        (sh_dc (N, 3) float64, sh_rest (N, 3, K) float64 or None when K = 0).
    """
    from viking_splatad.export.spherical_harmonics import (  # pylint: disable=import-outside-toplevel
        apply_sh_fit, chunk_size_for_budget, sh_fit_operator)
    from viking_splatad.export.splatad_colours import bake_gaussian_colours  # pylint: disable=import-outside-toplevel

    # Step 1: the fit operator pinv(B) depends only on the map-frame directions, so it is built once.
    pseudo_inverse = sh_fit_operator(directions_map, sh_degree)
    rest_per_channel = pseudo_inverse.shape[0] - 1
    # Step 2: the decoder lives in the dataparser frame: d_dp = A d_map, renormalised.
    directions_dataparser = rotate_directions_to_dataparser(directions_map, rotation_map_to_dataparser)
    gaussian_count = features_dc.shape[0]
    fit_chunk_size = chunk_size_for_budget(directions_map.shape[0])
    decoder_batch_size = decoder_batch_size_for_budget(decoder)
    sh_dc = np.empty((gaussian_count, COLOUR_CHANNEL_COUNT), dtype=np.float64)
    sh_rest = np.empty((gaussian_count, COLOUR_CHANNEL_COUNT, rest_per_channel), dtype=np.float64)
    # Step 3: bake the chunk (n, M, 3), fit it, keep only the coefficients.
    for chunk_start in range(0, gaussian_count, fit_chunk_size):
        chunk_stop = min(chunk_start + fit_chunk_size, gaussian_count)
        chunk_colours = bake_gaussian_colours(
            decoder, features_dc[chunk_start:chunk_stop], features_rest[chunk_start:chunk_stop],
            appearance_vector, directions_dataparser, decoder_batch_size)
        sh_dc[chunk_start:chunk_stop], sh_rest[chunk_start:chunk_stop] = apply_sh_fit(
            pseudo_inverse, chunk_colours)
    return sh_dc, (sh_rest if rest_per_channel > 0 else None)


def export_run(run_dir: Path, dataset_dir: Optional[Path], sh_degree: int, crop: bool,
               overwrite: bool, dry_run: bool = False) -> dict:
    """Export one run to ``<run>/renders/supersplat.ply`` and print per-step timings and counts.

    Args:
        run_dir: Run folder.
        dataset_dir: Dataset the run trained on, or None for ``data/intermediate/<session>``.
        sh_degree: SH degree 0..3.
        crop: Apply the observed-scene crop (T86).
        overwrite: Replace an existing ply.
        dry_run: Do every step except the disk guard and the write.

    Returns:
        Dict with ``output_path, bytes`` (written, or projected when dry), ``kept, dropped,
        step_seconds, peak_rss_bytes, dry_run``.

    Raises:
        FileExistsError: If the ply exists, ``overwrite`` is false and this is not a dry run.
        FileNotFoundError, ValueError: From ``find_train_dir`` and the loaders.
        ValueError: If the crop keeps no Gaussian (a header-only ply is never written).
        InsufficientDiskSpaceError: If the projected size plus the margin exceeds the free space.
    """
    from viking_splatad.eval.gaussian_occupancy import (  # pylint: disable=import-outside-toplevel
        checkpoint_gaussians_to_map_frame, latest_checkpoint_path)
    from viking_splatad.export.crop import crop_mask, train_lidar_bounds_map  # pylint: disable=import-outside-toplevel
    from viking_splatad.export.spherical_harmonics import (  # pylint: disable=import-outside-toplevel
        DIRECTION_OVERSAMPLING_FACTOR, fibonacci_sphere_directions)
    from viking_splatad.export.splatad_colours import (  # pylint: disable=import-outside-toplevel
        camera_sensor_index, load_upstream_rgb_decoder, model_settings_from_config_yml)
    from viking_splatad.export.supersplat_ply import (  # pylint: disable=import-outside-toplevel
        rotations_to_quaternions_wxyz, write_supersplat_ply)

    run_dir = Path(run_dir)
    output_path = run_dir / RENDERS_FOLDER_NAME / OUTPUT_FILE_NAME
    # A dry run writes nothing, so an existing ply is no obstacle to it.
    if output_path.exists() and not overwrite and not dry_run:
        raise FileExistsError(f"{output_path} exists (use --overwrite)")
    train_dir = find_train_dir(run_dir)
    step_seconds: Dict[str, float] = {}
    # Step 1: checkpoint, config and dataparser transform.
    with timed_step("load_checkpoint", step_seconds):
        checkpoint_path = latest_checkpoint_path(train_dir / CHECKPOINT_DIR_NAME)
        tensors = load_checkpoint_tensors(checkpoint_path)
        settings = model_settings_from_config_yml(train_dir / CONFIG_FILE_NAME)
        transforms = json.loads((train_dir / DATAPARSER_TRANSFORMS_FILE_NAME).read_text())
        dataparser_transform = np.asarray(transforms["transform"], dtype=np.float64)
    # Step 2: map-frame geometry (T56; also asserts scale 1 and no actors).
    gauss_params = tensors.pop("gauss_params")
    with timed_step("map_frame_geometry", step_seconds):
        gaussians = checkpoint_gaussians_to_map_frame(
            means=gauss_params["means"], log_scales=gauss_params["scales"],
            quaternions_wxyz=gauss_params["quats"], opacity_logits=gauss_params["opacities"],
            actor_ids=gauss_params["id"], actor_count=gauss_params["actor_count"],
            dataparser_transform=dataparser_transform, dataparser_scale=float(transforms["scale"]))
        # Keep only what the export uses: the (N, 3, 3) covariances are the largest array and unused.
        means_map, axis_std = gaussians.means, gaussians.axis_std
        principal_axes, activated_opacities = gaussians.principal_axes, gaussians.opacities
        opacity_logits = np.asarray(gauss_params["opacities"], dtype=np.float64).reshape(-1)
        del gaussians, gauss_params
    total_count = means_map.shape[0]
    # Step 3: crop mask (everything kept when --no-crop).
    with timed_step("crop", step_seconds):
        if crop:
            session = session_of_train_dir(train_dir)
            lower_corner, upper_corner = train_lidar_bounds_map(
                run_dir, dataset_dir or DEFAULT_DATASET_PARENT / session)
            keep_mask = crop_mask(means_map, activated_opacities, lower_corner, upper_corner,
                                  float(settings["mcmc_min_opacity"]))
        else:
            keep_mask = np.ones(total_count, dtype=bool)
    kept_count = int(keep_mask.sum())
    # Step 4: an empty selection would give a header-only ply SuperSplat cannot show.
    if kept_count == 0:
        raise ValueError(f"{run_dir.name}: the crop keeps 0 of {total_count} Gaussians; nothing to export "
                         "(check the dataset, or use --no-crop)")
    # Step 5: size and disk check before the expensive colour bake. The old ply (--overwrite) stays
    # until os.replace, so it is not credited: the temp file needs the whole projected size free.
    rest_per_channel = rest_coefficients_per_channel(sh_degree)
    projected_bytes = ply_byte_count(kept_count, rest_per_channel)
    if not dry_run:
        free_bytes = free_bytes_at(output_path)
        if projected_bytes + DISK_SAFETY_MARGIN_BYTES > free_bytes:
            raise InsufficientDiskSpaceError(
                f"projected {projected_bytes} bytes + margin {DISK_SAFETY_MARGIN_BYTES} for {output_path} "
                f"exceeds the free {free_bytes}")
    # Step 6: apply the mask to every per-Gaussian array, then drop the full-size originals.
    means_kept = means_map[keep_mask]
    axis_std_kept = axis_std[keep_mask]
    principal_axes_kept = principal_axes[keep_mask]
    opacity_logits_kept = opacity_logits[keep_mask]
    features_dc_kept = tensors.pop("features_dc")[keep_mask]
    features_rest_kept = tensors.pop("features_rest")[keep_mask]
    del means_map, axis_std, principal_axes, activated_opacities, opacity_logits
    # Step 7: colour. Camera: first of the config's list (single camera today).
    camera_names = settings["cameras"]
    if len(camera_names) > 1:
        print(f"note: several cameras {camera_names}; the appearance of the first ({camera_names[0]}) is used")
    with timed_step("bake_colours_and_sh_fit", step_seconds):
        decoder, appearance_weight = load_upstream_rgb_decoder(tensors["pipeline_state"], settings)
        appearance_vector = appearance_weight[camera_sensor_index(camera_names, camera_names[0])].numpy()
        # M = oversampling * (degree+1)^2 directions: the SH basis has (degree+1)^2 unknowns per channel.
        direction_count = DIRECTION_OVERSAMPLING_FACTOR * (rest_per_channel + 1)
        directions_map = fibonacci_sphere_directions(direction_count)
        rotation_map_to_dataparser = dataparser_transform[:3, :3]
        sh_dc, sh_rest = bake_sh_colours(decoder, features_dc_kept, features_rest_kept, appearance_vector,
                                         directions_map, rotation_map_to_dataparser, sh_degree)
    # Step 8: rotations -> quaternions, scales -> log; opacity logits are the stored ones.
    with timed_step("quaternions_and_write", step_seconds):
        quaternions = rotations_to_quaternions_wxyz(principal_axes_kept)
        log_scales = np.log(axis_std_kept)
        if dry_run:
            bytes_written = projected_bytes
        else:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            bytes_written = write_supersplat_ply(output_path, means_kept, opacity_logits_kept, log_scales,
                                                 quaternions, sh_dc, sh_rest)
    # Step 9: report.
    total_seconds = sum(step_seconds.values())
    peak_bytes = peak_rss_bytes()
    print(f"{run_dir.name}: step {tensors['step']}, kept {kept_count}, dropped {total_count - kept_count} "
          f"of {total_count} (crop={crop}), degree {sh_degree}, {direction_count} directions")
    for step_name, seconds in step_seconds.items():
        print(f"  {step_name}: {seconds:.2f} s")
    action = "dry run, would write" if dry_run else "wrote"
    print(f"  total: {total_seconds:.2f} s; {action} {output_path} ({bytes_written} bytes, "
          f"{bytes_written / BYTES_PER_MIB:.2f} MiB); peak RSS {peak_bytes} bytes "
          f"({peak_bytes / BYTES_PER_MIB:.0f} MiB)")
    return {"output_path": output_path, "bytes": bytes_written, "kept": kept_count,
            "dropped": total_count - kept_count, "step_seconds": step_seconds,
            "peak_rss_bytes": peak_bytes, "dry_run": dry_run}


def discover_runs(runs_root: Path, overwrite: bool = False, dry_run: bool = False) -> Tuple[List[Path], List[Tuple[Path, str]]]:
    """Split ``runs/*`` into runs to export and runs to skip, with the reason.

    Args:
        runs_root: The runs folder.
        overwrite: If true, a run that already has a ply is exported again, not skipped.
        dry_run: If true, a run that already has a ply is still listed (nothing is written).

    Returns:
        (to_export list of Paths, skipped list of (Path, reason)).
    """
    to_export: List[Path] = []
    skipped = []
    for run_dir in sorted(path for path in Path(runs_root).iterdir() if path.is_dir()):
        # Step 1: a usable run has a resolvable train dir with a checkpoint.
        try:
            train_dir = find_train_dir(run_dir)
        except (FileNotFoundError, ValueError) as error:
            skipped.append((run_dir, f"no usable train dir ({error})"))
            continue
        if not list((train_dir / CHECKPOINT_DIR_NAME).glob("step-*.ckpt")):
            skipped.append((run_dir, "no checkpoint"))
        # Step 2: an existing ply means done, unless --overwrite asks for a fresh export.
        elif (run_dir / RENDERS_FOLDER_NAME / OUTPUT_FILE_NAME).exists() and not (overwrite or dry_run):
            skipped.append((run_dir, f"{OUTPUT_FILE_NAME} exists"))
        else:
            to_export.append(run_dir)
    return to_export, skipped


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Returns:
        The argument parser.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--run-dir", type=Path, help="One run folder.")
    target_group.add_argument("--all", action="store_true",
                              help="Every runs/* with a checkpoint and no ply yet (with --overwrite: all of them).")
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT, help="Folder scanned by --all.")
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Dataset of one --run-dir (default data/intermediate/<session>); not with --all.")
    parser.add_argument("--sh-degree", type=int, default=DEFAULT_SH_DEGREE, choices=range(MAX_SH_DEGREE + 1),
                        help=f"SH degree 0..{MAX_SH_DEGREE} (0 = one colour per Gaussian).")
    parser.add_argument("--no-crop", action="store_true", help="Keep every Gaussian.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing supersplat.ply.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Do everything except the write; print the projected size and peak memory.")
    return parser


def main(argument_list: Optional[List[str]] = None) -> int:
    """Export one run or all fetched runs.

    Args:
        argument_list: CLI arguments (None = sys.argv[1:]).

    Returns:
        Exit code: 0 on success, 1 if the disk guard stopped the work or any ``--all`` run failed.

    Raises:
        SystemExit: On invalid arguments (``--dataset-dir`` with ``--all``).
    """
    parser = build_argument_parser()
    arguments = parser.parse_args(argument_list)
    # Step 1: each --all run derives its own session dataset, so one shared dataset is a mistake.
    if arguments.all and arguments.dataset_dir is not None:
        parser.error("--dataset-dir cannot be used with --all (each run uses its own session dataset)")
    crop = not arguments.no_crop
    # Step 2: one run. Errors propagate; only the disk guard is reported as an exit code.
    if not arguments.all:
        try:
            export_run(arguments.run_dir, arguments.dataset_dir, arguments.sh_degree, crop,
                       arguments.overwrite, arguments.dry_run)
        except InsufficientDiskSpaceError as error:
            print(f"stopping: {error}")
            return 1
        return 0
    # Step 3: --all. A disk-guard error stops the batch (the next run would fail the same way);
    # any other failure is recorded with the run id and the loop goes on.
    to_export, skipped = discover_runs(arguments.runs_root, arguments.overwrite, arguments.dry_run)
    for skipped_run, reason in skipped:
        print(f"skipped {skipped_run.name}: {reason}")
    print(f"{len(to_export)} runs to export")
    exported_runs: List[str] = []
    failed_runs: List[Tuple[str, str]] = []
    disk_stopped = False
    for run_dir in to_export:
        try:
            export_run(run_dir, None, arguments.sh_degree, crop, arguments.overwrite, arguments.dry_run)
        except InsufficientDiskSpaceError as error:
            print(f"stopping: {run_dir.name}: {error}")
            disk_stopped = True
            break
        except Exception as error:  # pylint: disable=broad-except  # per-run isolation, reported below
            print(f"FAILED {run_dir.name}: {type(error).__name__}: {error}")
            failed_runs.append((run_dir.name, f"{type(error).__name__}: {error}"))
        else:
            exported_runs.append(run_dir.name)
    # Step 4: summary.
    not_attempted = len(to_export) - len(exported_runs) - len(failed_runs)
    print(f"summary: exported {len(exported_runs)}, skipped {len(skipped)}, failed {len(failed_runs)}"
          + (f", not attempted {not_attempted} (disk guard)" if disk_stopped else ""))
    for failed_name, failure in failed_runs:
        print(f"  failed {failed_name}: {failure}")
    return 1 if (failed_runs or disk_stopped) else 0


if __name__ == "__main__":
    # Only the command-line entry point: SIGTERM (timeout) must unwind so the temp ply is removed.
    raise_system_exit_on_sigterm()
    sys.exit(main())
