"""Ground-truth cuboid parsing (T29) and cuboid pose geometry (T30).

Cuboids are Segments.ai point-cloud boxes drawn on a per-session lidar MAP frame
(z-up, aligned with UTM). Conventions: poses 4x4 float64 ``T_<target>_<source>``,
quaternions (x, y, z, w). ``dimensions`` in the JSON are FULL sizes (not half).
"""

import itertools
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from viking_splatad.registry import transform

# Class-id -> name table from the Viking Hill labelling protocol (0 = ground is
# implicit and never drawn as a box). It is only a default: callers may pass their
# own table to ``parse_cuboids``.
DEFAULT_CLASS_NAMES: Dict[int, str] = {
    1: "tree_trunk",
    2: "tree_canopy",
    3: "rock",
    4: "bush_small_tree",
    5: "car",
    6: "building",
    7: "lamp_sign",
    8: "ignore",
}

# Labelset that holds the ground-truth boxes in the Segments.ai export; default
# only, overridable per call.
DEFAULT_LABELSET_NAME = "ground-truth"

# Name used when a class id is absent from the class-name table.
UNKNOWN_CLASS_NAME = "unknown"

# Slack for the point-in-box comparison |local| <= half. Rotation matrices built
# from float64 quaternions carry ~1e-16 relative error, so points exactly on a
# face can land ~1e-15 outside; 1e-9 m is far above that and far below any
# physically meaningful (mm-scale) distance.
DEFAULT_BOUNDARY_TOLERANCE_METERS = 1e-9


@dataclass(frozen=True)
class Cuboid:
    """One labelled box in the map frame.

    Attributes:
        box_id: Annotation id from the JSON.
        class_id: Integer class id.
        class_name: Human-readable class name.
        center_xyz: Box centre in the map frame (meters).
        dimensions_xyz: FULL box sizes along the box's own x, y, z axes (meters).
        quaternion_xyzw: Unit quaternion of R_map_box (x, y, z, w).
    """

    box_id: Any
    class_id: int
    class_name: str
    center_xyz: Tuple[float, float, float]
    dimensions_xyz: Tuple[float, float, float]
    quaternion_xyzw: Tuple[float, float, float, float]


@dataclass(frozen=True)
class CuboidGeometry:
    """Derived geometry of a cuboid.

    Attributes:
        center: (3,) box centre in the map frame.
        rotation_map_box: (3, 3) R_map_box (columns are box axes in the map frame).
        half_extents: (3,) half sizes along box axes.
        corners: (8, 3) corner points in the map frame.
    """

    center: np.ndarray
    rotation_map_box: np.ndarray
    half_extents: np.ndarray
    corners: np.ndarray


def _looks_like_cuboid_list(candidate: Any) -> bool:
    """Return True if ``candidate`` is a non-empty list of cuboid-annotation dicts.

    Args:
        candidate: Any JSON node.

    Returns:
        Whether every element is a dict with ``position`` and ``dimensions``.
    """
    return (
        isinstance(candidate, list)
        and len(candidate) > 0
        and all(
            isinstance(element, dict) and "position" in element and "dimensions" in element
            for element in candidate
        )
    )


def _collect_annotation_lists(node: Any, path: str, found: List[Tuple[str, list]]) -> None:
    """Recursively collect (path, annotation list) pairs from a JSON tree.

    Args:
        node: Current JSON node.
        path: Slash-joined keys/indices leading to ``node``.
        found: Output list, appended in document order.
    """
    # A matching list is a leaf of the search: do not descend into annotations.
    if _looks_like_cuboid_list(node):
        found.append((path, node))
        return
    if isinstance(node, dict):
        for key, child in node.items():
            _collect_annotation_lists(child, f"{path}/{key}", found)
    elif isinstance(node, list):
        for child_index, child in enumerate(node):
            _collect_annotation_lists(child, f"{path}/{child_index}", found)


def find_annotation_list(
    document: Any,
    labelset_name: Optional[str] = DEFAULT_LABELSET_NAME,
    sample_key: Optional[str] = None,
) -> list:
    """Locate the cuboid annotation list inside a labels JSON (tolerant search).

    Documented search path: the document is walked recursively and every list whose
    elements all look like cuboids (dicts with ``position`` and ``dimensions``) is a
    candidate, e.g. ``dataset/samples/0/labels/ground-truth/attributes/annotations``,
    ``annotations`` or the document itself being the list. Candidates are filtered
    to paths containing ``labelset_name`` (when any do) and ``sample_key`` (e.g.
    ``"5cm_map_2024_05_dense.ply"``; matched against the path or the enclosing
    ``name`` fields' path text). Exactly one candidate must remain.

    Args:
        document: Parsed JSON.
        labelset_name: Preferred labelset name; ignored if no path contains it.
        sample_key: Optional substring selecting one sample/map file.

    Returns:
        The annotation dict list.

    Raises:
        ValueError: If none or several candidates remain.
    """
    candidates: List[Tuple[str, list]] = []
    _collect_annotation_lists(document, "", candidates)
    if not candidates:
        raise ValueError(
            "No cuboid annotations found: expected a list of dicts with 'position' and "
            "'dimensions' (e.g. .../labels/<labelset>/attributes/annotations or 'annotations')"
        )
    # Prefer candidates under the requested labelset, but stay tolerant if absent.
    if labelset_name is not None:
        in_labelset = [item for item in candidates if f"/{labelset_name}" in item[0]]
        if in_labelset:
            candidates = in_labelset
    # Sample selection: the key may be a dict key on the path or a sample "name".
    if sample_key is not None:
        selected = [item for item in candidates if sample_key in item[0]]
        if not selected:
            selected = [
                item for item in candidates
                if _path_has_name(document, item[0], sample_key)
            ]
        candidates = selected
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one annotations list but found {len(candidates)} "
            f"({[path for path, _ in candidates]}); pass sample_key to choose one"
        )
    return candidates[0][1]


def _path_has_name(document: Any, path: str, name_fragment: str) -> bool:
    """Check whether any ancestor dict on ``path`` has a ``name`` containing a fragment.

    Args:
        document: Parsed JSON root.
        path: Slash path produced by ``_collect_annotation_lists``.
        name_fragment: Substring to look for in ``name`` fields.

    Returns:
        True if some ancestor's ``name`` string contains ``name_fragment``.
    """
    node = document
    # Path starts with "/" so the first split element is empty.
    for step in path.split("/")[1:]:
        if isinstance(node, dict):
            if isinstance(node.get("name"), str) and name_fragment in node["name"]:
                return True
            node = node[step]
        elif isinstance(node, list):
            node = node[int(step)]
    return False


def _quaternion_from_annotation(annotation: dict) -> Tuple[float, float, float, float]:
    """Read and normalise the box quaternion (x, y, z, w) from an annotation.

    Falls back to a pure z rotation from ``yaw`` when ``rotation`` is absent.

    Args:
        annotation: One cuboid annotation dict.

    Returns:
        Unit quaternion (x, y, z, w).
    """
    rotation = annotation.get("rotation")
    if rotation is not None:
        raw = np.array([rotation["qx"], rotation["qy"], rotation["qz"], rotation["qw"]], dtype=np.float64)
    else:
        # Yaw-only: axis-angle about z gives (0, 0, sin(yaw/2), cos(yaw/2)).
        yaw = float(annotation.get("yaw", 0.0))
        raw = np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])
    # Normalise so rotation-matrix conversion is exactly orthonormal.
    unit = raw / np.linalg.norm(raw)
    return tuple(float(component) for component in unit)


@transform("T29", "Parse Segments.ai cuboid JSON into typed Cuboid dataclasses")
def parse_cuboids(
    document: Any,
    class_names: Optional[Dict[int, str]] = None,
    labelset_name: Optional[str] = DEFAULT_LABELSET_NAME,
    sample_key: Optional[str] = None,
) -> List[Cuboid]:
    """Parse a labels JSON document into typed cuboids.

    Args:
        document: Parsed JSON (see ``find_annotation_list`` for accepted shapes).
        class_names: class id -> name table; defaults to ``DEFAULT_CLASS_NAMES``.
        labelset_name: Labelset to read (default ``DEFAULT_LABELSET_NAME``).
        sample_key: Optional map-file substring to select one sample.

    Returns:
        Cuboids in document order; non-cuboid annotation types are skipped.

    Raises:
        ValueError: If no unique annotations list can be located.
    """
    name_table = DEFAULT_CLASS_NAMES if class_names is None else class_names
    annotations = find_annotation_list(document, labelset_name, sample_key)
    cuboids: List[Cuboid] = []
    for annotation in annotations:
        # Only cuboids are boxes; other annotation types (if any) are ignored.
        if annotation.get("type", "cuboid") != "cuboid":
            continue
        # Extract fields; dimensions stay FULL sizes, rotation is normalised to a unit quaternion.
        class_id = int(annotation["category_id"])
        position = annotation["position"]
        dimensions = annotation["dimensions"]
        cuboids.append(
            Cuboid(
                box_id=annotation.get("id"),
                class_id=class_id,
                class_name=name_table.get(class_id, UNKNOWN_CLASS_NAME),
                center_xyz=(float(position["x"]), float(position["y"]), float(position["z"])),
                dimensions_xyz=(float(dimensions["x"]), float(dimensions["y"]), float(dimensions["z"])),
                quaternion_xyzw=_quaternion_from_annotation(annotation),
            )
        )
    return cuboids


def quaternion_to_rotation_matrix(quaternion_xyzw: Iterable[float]) -> np.ndarray:
    """Convert a unit quaternion (x, y, z, w) to a 3x3 rotation matrix.

    Uses R = I + 2w[v]x + 2[v]x^2 expanded component-wise, v = (x, y, z); valid for
    unit quaternions (the input is renormalised defensively).

    Args:
        quaternion_xyzw: Quaternion components (x, y, z, w).

    Returns:
        (3, 3) float64 rotation matrix.
    """
    quat = np.asarray(list(quaternion_xyzw), dtype=np.float64)
    qx, qy, qz, qw = quat / np.linalg.norm(quat)
    # Rows of the standard unit-quaternion rotation matrix; diagonal terms use
    # 1 - 2(sum of the other two squared components), valid since |q| = 1.
    return np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ]
    )


@transform("T30", "Cuboid pose -> rotation matrix, half-extents, 8 corners (point-in-box)")
def compute_cuboid_geometry(cuboid: Cuboid) -> CuboidGeometry:
    """Derive rotation, half-extents and corner points of a cuboid.

    Corners are ``center + R_map_box @ (sign * half_extents)`` for all 8 sign
    triples in {-1, +1}^3.

    Args:
        cuboid: Box in the map frame.

    Returns:
        CuboidGeometry with (8, 3) corners in the map frame.
    """
    rotation_map_box = quaternion_to_rotation_matrix(cuboid.quaternion_xyzw)
    # Full size -> half size: the box spans [-half, +half] in its own frame.
    half_extents = np.asarray(cuboid.dimensions_xyz, dtype=np.float64) / 2.0
    center = np.asarray(cuboid.center_xyz, dtype=np.float64)
    # Sign pattern: each axis contributes either the -half or the +half face.
    sign_triples = np.array(list(itertools.product((-1.0, 1.0), repeat=3)))
    local_corners = sign_triples * half_extents
    # Row-vector form of R @ local: local_corners @ R^T.
    corners = center + local_corners @ rotation_map_box.T
    return CuboidGeometry(center, rotation_map_box, half_extents, corners)


def points_in_cuboid(
    geometry: CuboidGeometry,
    points_map: np.ndarray,
    tolerance_meters: float = DEFAULT_BOUNDARY_TOLERANCE_METERS,
) -> np.ndarray:
    """Test which map-frame points lie inside (or on) a cuboid.

    Transforms each point into the box frame, p_box = R_map_box^T (p_map - center),
    and accepts it when |p_box| <= half_extents (+ tolerance) on all three axes.

    Args:
        geometry: Output of ``compute_cuboid_geometry``.
        points_map: (N, 3) points in the map frame.
        tolerance_meters: Boundary slack (see DEFAULT_BOUNDARY_TOLERANCE_METERS).

    Returns:
        (N,) boolean mask.
    """
    # Row-vector form of R^T @ offset: offset @ R.
    local_points = (np.asarray(points_map, dtype=np.float64) - geometry.center) @ geometry.rotation_map_box
    return np.all(np.abs(local_points) <= geometry.half_extents + tolerance_meters, axis=1)
