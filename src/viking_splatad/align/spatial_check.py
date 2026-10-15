"""Lidar -> camera reprojection and an edge-alignment score for extrinsic checking."""

from typing import Tuple

import cv2
import numpy as np

from viking_splatad.registry import transform


@transform("T25", "project lidar points into the image with T_camera_lidar and pinhole K")
def project_lidar_to_image(
    points_lidar: np.ndarray,
    T_camera_lidar: np.ndarray,
    camera_matrix: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project lidar points into pixels (OpenCV pinhole, no distortion).

    Pixel convention (OpenCV): u = fx*X/Z + cx, v = fy*Y/Z + cy, where integer
    coordinates are pixel *centres*, so the top-left pixel covers [-0.5, 0.5) and
    the image spans u in [-0.5, width-0.5), v in [-0.5, height-0.5).

    Args:
        points_lidar: [N,3] points in the lidar frame.
        T_camera_lidar: 4x4 float64 transform lidar -> camera.
        camera_matrix: 3x3 intrinsics K.
        image_width: Image width in pixels.
        image_height: Image height in pixels.

    Returns:
        (pixels [N,2] float64, valid [N] bool); valid = in front of the camera
        (Z > 0) and inside the image bounds. Pixels of points with Z <= 0 are NaN.
    """
    # Camera-frame points: p_cam = R p_lidar + t.
    points_camera = points_lidar @ T_camera_lidar[:3, :3].T + T_camera_lidar[:3, 3]
    depth = points_camera[:, 2]
    in_front = depth > 0.0
    # Avoid dividing by non-positive depth; those rows are masked to NaN below.
    safe_depth = np.where(in_front, depth, np.nan)
    pixel_u = camera_matrix[0, 0] * points_camera[:, 0] / safe_depth + camera_matrix[0, 2]
    pixel_v = camera_matrix[1, 1] * points_camera[:, 1] / safe_depth + camera_matrix[1, 2]
    pixels = np.column_stack([pixel_u, pixel_v])
    # NaN comparisons are False, so behind-camera points are excluded automatically.
    inside_image = (
        (pixel_u >= -0.5)
        & (pixel_u < image_width - 0.5)
        & (pixel_v >= -0.5)
        & (pixel_v < image_height - 0.5)
    )
    return pixels, in_front & inside_image


def edge_alignment_score(
    image_gray: np.ndarray,
    pixels: np.ndarray,
    valid: np.ndarray,
    discontinuity: np.ndarray,
) -> float:
    """Discontinuity-weighted mean image-gradient magnitude, normalised by the image maximum.

    score = sum_i d_i * |grad I|(u_i, v_i) / (sum_i d_i * max |grad I|) over valid
    points. It is 1 when every lidar discontinuity lands on the strongest image edge
    and falls as projected edges slide off image edges (a plain Pearson correlation
    was rejected: it is scale-invariant and stays ~1 while edge points merely weaken).

    Args:
        image_gray: [H,W] float/uint8 grayscale image.
        pixels: [N,2] projected (u,v).
        valid: [N] bool mask from the projection.
        discontinuity: [N] non-negative per-point depth-discontinuity measure.

    Returns:
        Score in [0, 1]; 0 if there is no weight or no image gradient.
    """
    # Sobel gradient magnitude sqrt(Gx^2 + Gy^2) on a float32 copy.
    gray_float = image_gray.astype(np.float32)
    gradient_x = cv2.Sobel(gray_float, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Sobel(gray_float, cv2.CV_32F, 0, 1)
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    # Nearest-pixel lookup; clip guards the half-pixel border that rounds to W or H.
    column_index = np.clip(np.rint(pixels[valid, 0]).astype(int), 0, image_gray.shape[1] - 1)
    row_index = np.clip(np.rint(pixels[valid, 1]).astype(int), 0, image_gray.shape[0] - 1)
    sampled_gradient = gradient_magnitude[row_index, column_index]
    weights = discontinuity[valid]
    maximum_gradient = gradient_magnitude.max()
    if weights.sum() == 0.0 or maximum_gradient == 0.0:
        return 0.0
    return float(np.sum(weights * sampled_gradient) / (weights.sum() * maximum_gradient))
