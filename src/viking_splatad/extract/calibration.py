"""Extrinsics parsing (T5) and transform comparison (T6)."""

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np

from viking_splatad.registry import transform

# Matches a frame-pair header line such as "base_link -> camera".
FRAME_PAIR_PATTERN = re.compile(r"^\s*([A-Za-z0-9_]+)\s*->\s*([A-Za-z0-9_]+)\s*$")
# Matches a quoted variant header such as '"short_grass__ros1__" bags (2024-05):'.
VARIANT_HEADER_PATTERN = re.compile(r'^\s*"([a-z]+_grass)__ros1__"\s*bags')
# Matches a bracketed list of numbers.
NUMBER_LIST_PATTERN = re.compile(r"\[([^\]]*)\]")
# Path of the extrinsics file below a calibration directory.
EXTRINSICS_RELATIVE_PATH = Path("extrinsics") / "extrinsics.txt"


def quaternion_xyzw_to_rotation(quaternion_xyzw: np.ndarray) -> np.ndarray:
    """Convert a (possibly non-unit) quaternion [x,y,z,w] to a rotation matrix.

    The text stores 3-decimal quaternions (norm != 1), so q is renormalised first:
    q <- q/|q|. Then R = I + 2w[v]x + 2[v]x^2 with v=(x,y,z), the standard
    unit-quaternion formula, which is orthonormal with det +1.

    Args:
        quaternion_xyzw: Length-4 array.

    Returns:
        [3,3] float64 rotation matrix.
    """
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        raise ValueError("Zero-norm quaternion")
    vector_x, vector_y, vector_z, scalar_w = quaternion / norm
    cross_matrix = np.array(
        [[0.0, -vector_z, vector_y], [vector_z, 0.0, -vector_x], [-vector_y, vector_x, 0.0]]
    )
    return np.eye(3) + 2.0 * scalar_w * cross_matrix + 2.0 * (cross_matrix @ cross_matrix)


def parse_number_list(line: str) -> np.ndarray:
    """Parse the first bracketed number list on a line into a float array."""
    match = NUMBER_LIST_PATTERN.search(line)
    if match is None:
        raise ValueError(f"No bracketed list in line: {line!r}")
    return np.array([float(token) for token in match.group(1).split(",")], dtype=np.float64)


@transform(
    "T5",
    "Parse extrinsics.txt (translation + renormalised xyzw quaternion) into 4x4 "
    "T_parent_child matrices, selecting the per-run hugin radar variant",
)
def parse_extrinsics_text(text: str, extrinsics_variant: str) -> Dict[str, np.ndarray]:
    """Parse the dataset's extrinsics.txt.

    Format: a "parent -> child" header, then "- Translation: [x, y, z]" and
    "- Rotation: in Quaternion [x, y, z, w]". The hugin entry holds two blocks under
    quoted variant headers; only the block named ``extrinsics_variant`` is used.

    Args:
        text: File contents.
        extrinsics_variant: "short_grass" or "high_grass".

    Returns:
        Mapping "parent->child" -> [4,4] float64 transform mapping child coords to parent.

    Raises:
        ValueError: If the requested variant is absent from the text.
    """
    transforms: Dict[str, np.ndarray] = {}
    current_key: Optional[str] = None       # frame pair being parsed
    current_variant: Optional[str] = None   # variant header in force (None = shared)
    pending_translation: Optional[np.ndarray] = None
    seen_variants = set()
    # Step 1: scan line by line, tracking headers and capturing translation/rotation.
    for line in text.splitlines():
        pair_match = FRAME_PAIR_PATTERN.match(line)
        variant_match = VARIANT_HEADER_PATTERN.match(line)
        if pair_match:
            current_key = f"{pair_match.group(1)}->{pair_match.group(2)}"
            current_variant, pending_translation = None, None
        elif variant_match:
            current_variant = variant_match.group(1)
            seen_variants.add(current_variant)
            pending_translation = None
        elif "Translation:" in line:
            pending_translation = parse_number_list(line)
        elif "in Quaternion" in line and current_key is not None:
            # Step 2: a quaternion completes an entry; skip blocks of other variants.
            if current_variant is not None and current_variant != extrinsics_variant:
                continue
            if pending_translation is None:
                raise ValueError(f"Quaternion without translation for {current_key}")
            matrix = np.eye(4)
            matrix[:3, :3] = quaternion_xyzw_to_rotation(parse_number_list(line))
            matrix[:3, 3] = pending_translation
            transforms[current_key] = matrix
    # Step 3: the requested variant must exist in the file.
    if extrinsics_variant not in seen_variants:
        raise ValueError(
            f"Extrinsics variant {extrinsics_variant!r} not found (available: {sorted(seen_variants)})"
        )
    return transforms


def load_extrinsics(
    calibration_dir: Union[str, Path], extrinsics_variant: str
) -> Dict[str, np.ndarray]:
    """Read ``calibration_dir/extrinsics/extrinsics.txt`` and parse it.

    Args:
        calibration_dir: Dataset calibration directory.
        extrinsics_variant: "short_grass" or "high_grass".

    Returns:
        Same mapping as parse_extrinsics_text.
    """
    return parse_extrinsics_text(
        (Path(calibration_dir) / EXTRINSICS_RELATIVE_PATH).read_text(), extrinsics_variant
    )


@transform("T6", "Compare two rigid transforms: translation distance and rotation angle")
def compare_transforms(transform_a: np.ndarray, transform_b: np.ndarray) -> Tuple[float, float]:
    """Return (translation_error_m, rotation_error_rad) between two 4x4 transforms.

    Translation error = |t_a - t_b|. Rotation error is the angle of R_a^T R_b:
    theta = arccos((trace(R_a^T R_b) - 1) / 2), the relative rotation's angle.

    Args:
        transform_a: [4,4] transform.
        transform_b: [4,4] transform.

    Returns:
        (translation_error_m, rotation_error_rad), rotation in [0, pi].
    """
    transform_a = np.asarray(transform_a, dtype=np.float64)
    transform_b = np.asarray(transform_b, dtype=np.float64)
    # Step 1: Euclidean distance between translations.
    translation_error = float(np.linalg.norm(transform_a[:3, 3] - transform_b[:3, 3]))
    # Step 2: relative rotation and its angle; clip guards rounding outside [-1, 1].
    relative_rotation = transform_a[:3, :3].T @ transform_b[:3, :3]
    cosine_angle = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return translation_error, float(np.arccos(cosine_angle))
