"""Exact perspective-camera fit of a point set, pure numpy (no PyVista).

``tight_oblique_camera`` places the camera as close as possible along the oblique direction while keeping every
given point in view for a given panel aspect, centred on the points' projection; ``tight_projected_aspect``
gives the panel shape that matches the scene.
"""

from typing import List, Sequence, Tuple

import numpy as np

# Oblique camera: unit-less direction from the focal point to the camera (x right, y forward, z up), the
# vertical field of view (PyVista/VTK default 30 degrees) and the image-up vector.
OBLIQUE_CAMERA_DIRECTION = (1.0, -1.0, 0.9)
CAMERA_VIEW_ANGLE_DEGREES = 30.0
VIEW_UP_VECTOR = (0.0, 0.0, 1.0)

# Spatial dimensions of the scene.
SPATIAL_DIMENSIONS = 3
# Direction and view-up count as parallel when |cos| of their angle exceeds 1 - this (unit-less).
PARALLEL_TOLERANCE = 1e-9


def camera_basis(direction: Sequence[float], view_up: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthonormal camera frame for a camera placed along ``direction`` from the focal point.

    Args:
        direction: Vector from the focal point toward the camera (any length).
        view_up: Approximate up vector of the image.

    Returns:
        (forward, right, true_up) unit vectors: forward = -normalised direction (camera to focal point),
        right = normalised forward x view_up, true_up = right x forward.

    Raises:
        ValueError: If direction is zero or parallel to view_up (the image axes are undefined).
    """
    # Step 1: forward looks from the camera to the focal point.
    direction_array = np.asarray(direction, dtype=float)
    direction_norm = np.linalg.norm(direction_array)
    if direction_norm == 0.0:
        raise ValueError("direction must be non-zero")
    forward = -direction_array / direction_norm
    # Step 2: reject a view-up parallel to forward (cross product would vanish).
    view_up_array = np.asarray(view_up, dtype=float)
    view_up_unit = view_up_array / np.linalg.norm(view_up_array)
    if abs(float(forward @ view_up_unit)) > 1.0 - PARALLEL_TOLERANCE:
        raise ValueError("direction is parallel to view_up")
    # Step 3: right = forward x up; true_up = right x forward is orthogonal to both by construction.
    right = np.cross(forward, view_up_array)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)
    return forward, right, true_up


def tight_projected_aspect(points: np.ndarray, direction: Sequence[float], view_up: Sequence[float]) -> float:
    """Width / height of the orthographic projection of the points themselves (not of their bounding box).

    A scene that is diagonal in the ground plane leaves the corners of its axis-aligned box empty, so the box
    over-states the projected extent; the points' own extent is tighter.

    Args:
        points: (N, 3) points (N >= 1, not all projecting to one image row or column).
        direction: Vector from the focal point toward the camera.
        view_up: Approximate image-up vector.

    Returns:
        Extent along ``right`` divided by extent along ``true_up``.
    """
    _, right, true_up = camera_basis(direction, view_up)
    return float(np.ptp(points @ right) / np.ptp(points @ true_up))


def tight_oblique_camera(points: np.ndarray, aspect: float, padding: float, direction: Sequence[float],
                         view_up: Sequence[float], view_angle_degrees: float) -> List[Tuple[float, float, float]]:
    """Closest perspective camera along ``direction`` that keeps every given point in view, centred on them.

    Frustum condition per point, with d the camera-to-focal distance and (r, u, z) the point's right, up and
    toward-camera coordinates relative to the focal point: d >= z + |u| / tan(fov / 2) and
    d >= z + |r| / (aspect * tan(fov / 2)). It is evaluated on the points themselves, with the focal
    point chosen so the projected extent is centred: its right and up coordinates are the mid-ranges
    (max + min) / 2 of the points' right / up coordinates and its depth coordinate the points' mean. Centring
    minimises max |r| and max |u|, hence the distance d.

    Args:
        points: (N, 3) points to keep in view (N >= 1); give cell corners if the cell size matters.
        aspect: Panel width / height (> 0).
        padding: Distance multiplier (> 0; 1 = tight).
        direction: Vector from the focal point toward the camera.
        view_up: Approximate image-up vector.
        view_angle_degrees: Vertical field of view in degrees, in (0, 180).

    Returns:
        [position, focal_point, view_up], each a 3-tuple.

    Raises:
        ValueError: If aspect or padding is not positive, the view angle is out of range, points are not (N, 3)
            with N >= 1, or direction is parallel to view_up.
    """
    # Step 1: validate.
    point_array = np.asarray(points, dtype=float)
    if point_array.ndim != 2 or point_array.shape[1] != SPATIAL_DIMENSIONS or point_array.shape[0] == 0:
        raise ValueError("points must be a non-empty (N, 3) array")
    if aspect <= 0.0 or padding <= 0.0:
        raise ValueError("aspect and padding must be > 0")
    if not 0.0 < view_angle_degrees < 180.0:  # a perspective field of view lies strictly in (0, 180) degrees
        raise ValueError("view_angle_degrees must be in (0, 180)")
    # Step 2: image coordinates and the centred focal point.
    forward, right, true_up = camera_basis(direction, view_up)
    right_coordinates, up_coordinates = point_array @ right, point_array @ true_up
    depth_coordinates = point_array @ (-forward)
    focal_right = (right_coordinates.max() + right_coordinates.min()) / 2.0
    focal_up = (up_coordinates.max() + up_coordinates.min()) / 2.0
    focal_depth = depth_coordinates.mean()
    focal_point = focal_right * right + focal_up * true_up + focal_depth * (-forward)
    # Step 3: binding point per the frustum condition, then step back.
    vertical_half_tangent = np.tan(np.radians(view_angle_degrees) / 2.0)
    required_distances = (depth_coordinates - focal_depth) + np.maximum(
        np.abs(up_coordinates - focal_up) / vertical_half_tangent,
        np.abs(right_coordinates - focal_right) / (aspect * vertical_half_tangent))
    position = focal_point - padding * float(required_distances.max()) * forward
    return [tuple(position), tuple(focal_point), tuple(float(component) for component in view_up)]
