"""Versioned, content-addressed radar seed artifacts."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from viking_splatad.radar_init.seeds import RadarSeedGaussians

RADAR_INPUTS_SCHEMA = "viking-splatad/radar-inputs/1"
RADAR_ARRAYS_FILENAME = "radar_seeds.npz"
RADAR_METADATA_FILENAME = "radar_inputs.json"
RADAR_MANIFEST_FILENAME = "manifest.json"
ARRAY_NAMES = (
    "means", "covariances", "rgb", "sensor_origins", "support_count", "time_spans_s"
)


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file.

    Args:
        path: File whose bytes are hashed.

    Returns:
        Sixty-four-character hexadecimal digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for block in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_seed_arrays(arrays: Dict[str, np.ndarray]) -> None:
    """Validate the complete seed array contract or raise ``ValueError``.

    Args:
        arrays: Mapping loaded from the compressed seed archive.

    Returns:
        None.
    """
    missing_names = sorted(set(ARRAY_NAMES) - set(arrays))
    extra_names = sorted(set(arrays) - set(ARRAY_NAMES))
    if missing_names or extra_names:
        raise ValueError(f"radar seed arrays mismatch: missing={missing_names}, extra={extra_names}")
    seed_count = arrays["means"].shape[0] if arrays["means"].ndim == 2 else -1
    expected_shapes = {
        "means": (seed_count, 3),
        "covariances": (seed_count, 3, 3),
        "rgb": (seed_count, 3),
        "sensor_origins": (seed_count, 3),
        "support_count": (seed_count,),
        "time_spans_s": (seed_count, 2),
    }
    for array_name, expected_shape in expected_shapes.items():
        if arrays[array_name].shape != expected_shape:
            raise ValueError(
                f"{array_name} has shape {arrays[array_name].shape}, expected {expected_shape}"
            )
    if seed_count <= 0:
        raise ValueError("radar seed artifact must contain at least one seed")
    for array_name in ARRAY_NAMES:
        if arrays[array_name].dtype.kind not in "biufc":
            raise ValueError(f"{array_name} must have a numeric dtype")
        if not np.isfinite(arrays[array_name]).all():
            raise ValueError(f"{array_name} contains non-finite values")
    if arrays["rgb"].dtype != np.uint8:
        raise ValueError("rgb must have dtype uint8")
    for floating_name in ("means", "covariances", "sensor_origins", "time_spans_s"):
        if arrays[floating_name].dtype != np.float64:
            raise ValueError(f"{floating_name} must have dtype float64")
    if arrays["support_count"].dtype != np.int64 or np.any(arrays["support_count"] <= 0):
        raise ValueError("support_count must contain positive int64 values")
    if np.any(arrays["time_spans_s"][:, 1] < arrays["time_spans_s"][:, 0]):
        raise ValueError("every radar seed time span must be ordered")
    covariances = arrays["covariances"].astype(np.float64)
    symmetry_tolerance = np.finfo(np.float64).eps * np.maximum(1.0, np.abs(covariances)).max()
    if not np.allclose(covariances, covariances.transpose(0, 2, 1), atol=symmetry_tolerance):
        raise ValueError("radar seed covariances must be symmetric")
    if np.any(np.linalg.eigvalsh(covariances) <= 0.0):
        raise ValueError("radar seed covariances must be positive definite")


def write_radar_seed_artifact(
    output_directory: Path,
    seeds: RadarSeedGaussians,
    session_name: str,
    provenance: Dict[str, Any],
    parameters: Dict[str, Any],
) -> None:
    """Write an immutable seed archive plus metadata and file hashes.

    Args:
        output_directory: New, empty output directory.
        seeds: T36 seed arrays in the session map frame.
        session_name: Dataset session tag required by the dataparser.
        provenance: Hashes/tags for the dataset, split and inputs.
        parameters: Effective label-free selection parameters.

    Returns:
        None.

    Raises:
        FileExistsError: If the target already exists.
        ValueError: If the arrays or tags violate the artifact contract.
    """
    output_directory = Path(output_directory)
    if output_directory.exists():
        raise FileExistsError(f"refusing to reuse radar artifact directory: {output_directory}")
    if not session_name:
        raise ValueError("session_name must be non-empty")
    arrays = {array_name: np.asarray(getattr(seeds, array_name)) for array_name in ARRAY_NAMES}
    _validate_seed_arrays(arrays)
    output_directory.mkdir(parents=True)
    array_path = output_directory / RADAR_ARRAYS_FILENAME
    np.savez_compressed(array_path, **arrays)
    metadata = {
        "schema": RADAR_INPUTS_SCHEMA,
        "session": session_name,
        "frame": "map",
        "time_basis": "absolute_seconds",
        "seed_count": int(arrays["means"].shape[0]),
        "provenance": provenance,
        "parameters": parameters,
    }
    metadata_path = output_directory / RADAR_METADATA_FILENAME
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    manifest = {
        "schema": RADAR_INPUTS_SCHEMA,
        "files": {
            RADAR_ARRAYS_FILENAME: sha256_file(array_path),
            RADAR_METADATA_FILENAME: sha256_file(metadata_path),
        },
    }
    (output_directory / RADAR_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def load_radar_seed_artifact(
    input_directory: Path, expected_session_name: str
) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Load and fully validate a radar seed artifact.

    Args:
        input_directory: Directory containing the three versioned artifact files.
        expected_session_name: Session requested by the dataparser.

    Returns:
        ``(arrays, metadata)`` after hash, tag and numerical validation.
    """
    input_directory = Path(input_directory)
    manifest_path = input_directory / RADAR_MANIFEST_FILENAME
    try:
        manifest = json.loads(manifest_path.read_text())
        metadata_path = input_directory / RADAR_METADATA_FILENAME
        array_path = input_directory / RADAR_ARRAYS_FILENAME
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid radar artifact manifest: {error}") from error
    if manifest.get("schema") != RADAR_INPUTS_SCHEMA:
        raise ValueError(f"unsupported radar artifact schema: {manifest.get('schema')!r}")
    expected_files = {RADAR_ARRAYS_FILENAME, RADAR_METADATA_FILENAME}
    if set(manifest.get("files", {})) != expected_files:
        raise ValueError("radar artifact manifest must hash exactly the arrays and metadata files")
    for filename in sorted(expected_files):
        file_path = input_directory / filename
        if not file_path.is_file() or sha256_file(file_path) != manifest["files"][filename]:
            raise ValueError(f"radar artifact hash mismatch: {filename}")
    try:
        metadata = json.loads(metadata_path.read_text())
        with np.load(array_path, allow_pickle=False) as archive:
            arrays = {array_name: archive[array_name] for array_name in archive.files}
    except (json.JSONDecodeError, OSError, ValueError, KeyError) as error:
        raise ValueError(f"invalid radar artifact contents: {error}") from error
    expected_tags = {
        "schema": RADAR_INPUTS_SCHEMA,
        "session": expected_session_name,
        "frame": "map",
        "time_basis": "absolute_seconds",
    }
    for tag_name, expected_value in expected_tags.items():
        if metadata.get(tag_name) != expected_value:
            raise ValueError(
                f"radar artifact {tag_name} is {metadata.get(tag_name)!r}, expected {expected_value!r}"
            )
    _validate_seed_arrays(arrays)
    if metadata.get("seed_count") != arrays["means"].shape[0]:
        raise ValueError("radar artifact seed_count does not match its arrays")
    if not isinstance(metadata.get("provenance"), dict) or not isinstance(
        metadata.get("parameters"), dict
    ):
        raise ValueError("radar artifact provenance and parameters must be objects")
    return arrays, metadata
