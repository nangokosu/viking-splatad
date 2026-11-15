"""Paint radar seed colours from undistorted train camera frames (camera projection and occlusion)."""

import numpy as np

NEUTRAL_GREY_RGB = np.array([127, 127, 127], dtype=np.uint8)


def project_map_points(
    points_map: np.ndarray,
    transform_map_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple:
    """Project map points with an OpenCV camera pose, returning pixels and depth."""
    transform_camera_map = np.linalg.inv(np.asarray(transform_map_camera, dtype=np.float64))
    points_camera = points_map @ transform_camera_map[:3, :3].T + transform_camera_map[:3, 3]
    depth = points_camera[:, 2]
    safe_depth = np.where(depth > 0.0, depth, np.nan)
    pixel_u = camera_matrix[0, 0] * points_camera[:, 0] / safe_depth + camera_matrix[0, 2]
    pixel_v = camera_matrix[1, 1] * points_camera[:, 1] / safe_depth + camera_matrix[1, 2]
    return np.column_stack([pixel_u, pixel_v]), depth


def paint_visible_seed_colors(
    seed_means_map: np.ndarray,
    transform_map_camera: np.ndarray,
    camera_matrix: np.ndarray,
    image_rgb: np.ndarray,
    occluder_points_map: np.ndarray,
    footprint_radius_pixels: float,
    occlusion_margin_m: float,
) -> tuple:
    """Paint seeds visible in one train image; keep neutral grey otherwise.

    Args:
        seed_means_map: Seed points ``[N,3]`` in the map frame.
        transform_map_camera: OpenCV ``T_map_camera`` ``[4,4]``.
        camera_matrix: Undistorted intrinsics ``[3,3]``.
        image_rgb: RGB image ``[H,W,3]``.
        occluder_points_map: Nearest train LiDAR sweep in map coordinates ``[L,3]``.
        footprint_radius_pixels: Radar angular footprint radius in image pixels.
        occlusion_margin_m: A LiDAR return must be this much closer to count as occluding.

    Returns:
        ``(rgb [N,3] uint8, painted [N] bool)``.
    """
    seed_means = np.asarray(seed_means_map, dtype=np.float64)
    colors = np.repeat(NEUTRAL_GREY_RGB[None, :], len(seed_means), axis=0)
    painted = np.zeros(len(seed_means), dtype=bool)
    seed_pixels, seed_depths = project_map_points(seed_means, transform_map_camera, camera_matrix)
    lidar_pixels, lidar_depths = project_map_points(
        np.asarray(occluder_points_map, dtype=np.float64), transform_map_camera, camera_matrix
    )
    image_height, image_width = image_rgb.shape[:2]
    lidar_valid = (
        (lidar_depths > 0.0)
        & np.isfinite(lidar_pixels).all(axis=1)
        & (lidar_pixels[:, 0] >= 0.0)
        & (lidar_pixels[:, 0] < image_width)
        & (lidar_pixels[:, 1] >= 0.0)
        & (lidar_pixels[:, 1] < image_height)
    )
    valid_lidar_pixels = lidar_pixels[lidar_valid]
    valid_lidar_depths = lidar_depths[lidar_valid]
    for seed_index, (pixel, seed_depth) in enumerate(zip(seed_pixels, seed_depths)):
        inside = (
            seed_depth > 0.0
            and np.isfinite(pixel).all()
            and 0.0 <= pixel[0] < image_width
            and 0.0 <= pixel[1] < image_height
        )
        if not inside:
            continue
        pixel_distances = np.linalg.norm(valid_lidar_pixels - pixel, axis=1)
        closer_return = (
            (pixel_distances <= footprint_radius_pixels)
            & (valid_lidar_depths < seed_depth - occlusion_margin_m)
        )
        if closer_return.any():
            continue
        column_index = int(np.clip(np.rint(pixel[0]), 0, image_width - 1))
        row_index = int(np.clip(np.rint(pixel[1]), 0, image_height - 1))
        colors[seed_index] = image_rgb[row_index, column_index]
        painted[seed_index] = True
    return colors, painted
