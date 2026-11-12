"""Turn selected radar voxels into colored anisotropic Gaussian seeds (T36)."""

from dataclasses import dataclass

import numpy as np

from viking_splatad.radar_init.select import RadarCandidateVoxels
from viking_splatad.registry import transform

# Neutral mid-grey is used only when no unobstructed train-camera sample exists.
NEUTRAL_GREY_RGB = np.array([127, 127, 127], dtype=np.uint8)


@dataclass(frozen=True)
class RadarSeedGaussians:
    """Complete map-frame radar seed artifact arrays."""

    means: np.ndarray
    covariances: np.ndarray
    rgb: np.ndarray
    sensor_origins: np.ndarray
    support_count: np.ndarray
    time_spans_s: np.ndarray


def _ray_basis(unit_ray: np.ndarray) -> np.ndarray:
    """Build a right-handed radial/azimuth/elevation basis for one unit ray."""
    reference_axis = np.array([0.0, 0.0, 1.0])
    if abs(float(unit_ray @ reference_axis)) > 1.0 - np.sqrt(np.finfo(np.float64).eps):
        reference_axis = np.array([1.0, 0.0, 0.0])
    azimuth_axis = np.cross(reference_axis, unit_ray)
    azimuth_axis /= np.linalg.norm(azimuth_axis)
    elevation_axis = np.cross(unit_ray, azimuth_axis)
    basis = np.column_stack([unit_ray, azimuth_axis, elevation_axis])
    if np.linalg.det(basis) < 0.0:
        basis[:, -1] *= -1.0
    return basis


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


@transform(
    "T36",
    "Radar candidate voxels -> colored anisotropic Gaussian seeds from sensor resolution",
)
def build_seed_gaussians(
    candidates: RadarCandidateVoxels,
    rgb: np.ndarray,
    range_resolution_m: float,
    azimuth_resolution_rad: float,
    elevation_resolution_rad: float,
) -> RadarSeedGaussians:
    """Build one Gaussian per radar candidate with covariance in the ray basis.

    ``Sigma = R_ray diag(sigma_r^2, (r*delta_az)^2, (r*delta_el)^2) R_ray^T``.

    Args:
        candidates: T35 output in the map frame.
        rgb: uint8 or numeric colors ``[N,3]`` already painted/neutral-filled.
        range_resolution_m: Radar range resolution (used as radial sigma), metres.
        azimuth_resolution_rad: Radar azimuth resolution, radians.
        elevation_resolution_rad: Radar elevation resolution, radians.

    Returns:
        Complete ``RadarSeedGaussians`` arrays in the map frame.
    """
    means = np.asarray(candidates.means_map, dtype=np.float64)
    origins = np.asarray(candidates.sensor_origins_map, dtype=np.float64)
    colors = np.asarray(rgb)
    if means.ndim != 2 or means.shape[1] != 3 or origins.shape != means.shape:
        raise ValueError("candidate means and origins must both have shape [N,3]")
    if colors.shape != means.shape or not np.isfinite(colors).all():
        raise ValueError("rgb must be finite with shape [N,3]")
    if np.any((colors < 0) | (colors > 255)):
        raise ValueError("rgb values must lie in [0,255]")
    if min(range_resolution_m, azimuth_resolution_rad, elevation_resolution_rad) <= 0.0:
        raise ValueError("radar resolutions must be positive")
    ray_vectors = means - origins
    ranges_m = np.linalg.norm(ray_vectors, axis=1)
    if np.any(ranges_m <= 0.0):
        raise ValueError("every radar candidate must have positive range")
    covariances = np.empty((len(means), 3, 3), dtype=np.float64)
    for seed_index, (ray_vector, range_m) in enumerate(zip(ray_vectors, ranges_m)):
        unit_ray = ray_vector / range_m
        ray_basis = _ray_basis(unit_ray)
        standard_deviations = np.array([
            range_resolution_m,
            range_m * np.tan(azimuth_resolution_rad),
            range_m * np.tan(elevation_resolution_rad),
        ])
        covariances[seed_index] = (
            ray_basis @ np.diag(standard_deviations**2) @ ray_basis.T
        )
    return RadarSeedGaussians(
        means=means,
        covariances=covariances,
        rgb=np.rint(colors).astype(np.uint8),
        sensor_origins=origins,
        support_count=np.asarray(candidates.support_counts, dtype=np.int64),
        time_spans_s=np.asarray(candidates.time_spans_s, dtype=np.float64),
    )
