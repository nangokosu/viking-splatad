"""Paint radar anchor colours from undistorted train camera frames.

Pipeline: project map-frame anchors into the train camera frame nearest in time, reject those
that are outside the image, behind the camera, or occluded by a closer train LiDAR return inside
the radar angular footprint, and read the remaining colours from the image. Unpainted anchors keep
neutral grey. Only train frames are read, so no held-out image leaks into the initialisation.
"""

import cv2
import numpy as np

from viking_splatad.dataset.run_split import nearest_reference_positions
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.registry import transform
from viking_splatad.splatad.viking_conversions import load_lidar_sweep

# Camera-stamp value for an anchor that no train camera frame painted.
UNPAINTED_STAMP_NS = -1

# Colour of an unpainted anchor: uint8 mid-grey (floor(255 / 2)), a prior that favours no hue.
NEUTRAL_GREY_RGB = np.array([127, 127, 127], dtype=np.uint8)


def project_map_points(
    points_map: np.ndarray,
    transform_map_camera: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple:
    """Project map points with an OpenCV camera pose, returning pixels and depth.

    Args:
        points_map: Points ``[N,3]`` in the map frame.
        transform_map_camera: OpenCV ``T_map_camera`` ``[4,4]``.
        camera_matrix: Undistorted intrinsics ``[3,3]``.

    Returns:
        ``(pixels [N,2] (u, v), depth [N])``; pixels are NaN where depth is not positive.
    """
    # Map -> camera: p_c = R^T (p_m - t), applied via the inverted pose.
    transform_camera_map = np.linalg.inv(np.asarray(transform_map_camera, dtype=np.float64))
    points_camera = points_map @ transform_camera_map[:3, :3].T + transform_camera_map[:3, 3]
    # OpenCV camera looks along +z; points at or behind the image plane get NaN pixels.
    depth = points_camera[:, 2]
    safe_depth = np.where(depth > 0.0, depth, np.nan)
    # Pinhole projection: u = fx * x / z + cx, v = fy * y / z + cy.
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
    # Step 1: every seed starts grey and unpainted.
    seed_means = np.asarray(seed_means_map, dtype=np.float64)
    colors = np.repeat(NEUTRAL_GREY_RGB[None, :], len(seed_means), axis=0)
    painted = np.zeros(len(seed_means), dtype=bool)
    # Step 2: project seeds and LiDAR occluders into the same image.
    seed_pixels, seed_depths = project_map_points(seed_means, transform_map_camera, camera_matrix)
    lidar_pixels, lidar_depths = project_map_points(
        np.asarray(occluder_points_map, dtype=np.float64), transform_map_camera, camera_matrix
    )
    image_height, image_width = image_rgb.shape[:2]
    # Step 3: keep only occluders in front of the camera and inside the image.
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
    # Step 4: per seed, O(L) scan of occluders (O(N * L) per image).
    for seed_index, (pixel, seed_depth) in enumerate(zip(seed_pixels, seed_depths)):
        # A seed behind the camera or off the image cannot be painted.
        inside = (
            seed_depth > 0.0
            and np.isfinite(pixel).all()
            and 0.0 <= pixel[0] < image_width
            and 0.0 <= pixel[1] < image_height
        )
        if not inside:
            continue
        # Occluded: a LiDAR return inside the footprint disc and more than the margin nearer.
        pixel_distances = np.linalg.norm(valid_lidar_pixels - pixel, axis=1)
        closer_return = (
            (pixel_distances <= footprint_radius_pixels)
            & (valid_lidar_depths < seed_depth - occlusion_margin_m)
        )
        if closer_return.any():
            continue
        # Nearest pixel (OpenCV pixel centres sit on integers), clipped for u or v just below the edge.
        column_index = int(np.clip(np.rint(pixel[0]), 0, image_width - 1))
        row_index = int(np.clip(np.rint(pixel[1]), 0, image_height - 1))
        colors[seed_index] = image_rgb[row_index, column_index]
        painted[seed_index] = True
    return colors, painted


def _lidar_sweep_filter_arguments(dataset_view) -> tuple:
    """Read the sweep loader's minimum radius and ego box from the manifest.

    Args:
        dataset_view: Dataset view whose manifest holds ``lidar.min_lidar_dist``.

    Returns:
        ``(minimum_radius_m, ego_half_extents_m)``; a list means an ego box, a float a radius.
    """
    minimum_distance = dataset_view.manifest["lidar"]["min_lidar_dist"]
    if isinstance(minimum_distance, list):
        return 0.0, tuple(minimum_distance)
    return float(minimum_distance), (0.0, 0.0, 0.0)


def _train_frame_indices(dataset_view, sensor_name: str, train_files: set) -> np.ndarray:
    """Indices of a sensor's frames whose dataset-relative path is in the train set.

    Args:
        dataset_view: Dataset view of the extracted dataset.
        sensor_name: Sensor stream name.
        train_files: Dataset-relative posix paths belonging to train.

    Returns:
        int64 [T] frame indices.
    """
    return np.array([
        frame_index
        for frame_index, frame_path in enumerate(dataset_view.sensor_frame_paths[sensor_name])
        if frame_path.relative_to(dataset_view.root).as_posix() in train_files
    ], dtype=np.int64)


@transform("T37", "radar anchor colours painted from the nearest unobstructed train camera frame")
def paint_anchors_from_train_cameras(
    anchor_means_map: np.ndarray,
    anchor_time_spans_s: np.ndarray,
    dataset_view,
    train_files: set,
    camera_name: str,
    lidar_name: str,
    azimuth_cell_rad: float,
    elevation_cell_rad: float,
    occlusion_margin_m: float,
) -> tuple:
    """Paint anchors from the train camera frame nearest their mid-time, grey when occluded.

    Footprint: a radar cell spans ``azimuth_cell_rad`` x ``elevation_cell_rad``; at focal length f
    pixels an angle a subtends ``f * tan(a)`` pixels on the image plane, so the LiDAR-occlusion
    radius is ``max(fx * tan(azimuth_cell), fy * tan(elevation_cell))``. Cost: one image and one
    sweep load per distinct chosen frame.

    Args:
        anchor_means_map: float [S,3] anchor positions in the map frame.
        anchor_time_spans_s: float [S,2] (start, end) observation span of each anchor, seconds.
        dataset_view: Dataset view with frame paths, times, poses and manifest.
        train_files: Dataset-relative posix paths of the train frames.
        camera_name: Camera stream providing colour.
        lidar_name: LiDAR stream providing the nearest train occluder sweep.
        azimuth_cell_rad: Radar azimuth resolution.
        elevation_cell_rad: Radar elevation resolution.
        occlusion_margin_m: A LiDAR return must be this much closer than the anchor to occlude.

    Returns:
        ``(rgb uint8 [S,3], painted bool [S], camera_stamps_ns int64 [S])``; stamps are the chosen
        frame's timestamp, ``-1`` where unpainted.
    """
    # Step 1: validate shapes and the train frame sets.
    anchor_means = np.asarray(anchor_means_map, dtype=np.float64)
    time_spans = np.asarray(anchor_time_spans_s, dtype=np.float64)
    if anchor_means.ndim != 2 or anchor_means.shape[1] != 3:
        raise ValueError(f"anchor_means_map must be [S,3], got {anchor_means.shape}")
    if time_spans.shape != (len(anchor_means), 2):
        raise ValueError(f"anchor_time_spans_s must be [{len(anchor_means)},2], got {time_spans.shape}")
    camera_paths = dataset_view.sensor_frame_paths[camera_name]
    lidar_paths = dataset_view.sensor_frame_paths[lidar_name]
    train_camera_indices = _train_frame_indices(dataset_view, camera_name, train_files)
    if train_camera_indices.size == 0:
        raise ValueError("reference split contains no train camera frames")
    train_lidar_indices = _train_frame_indices(dataset_view, lidar_name, train_files)
    if train_lidar_indices.size == 0:
        raise ValueError("reference split contains no train LiDAR sweeps")

    # Step 2: choose each anchor's train camera frame by time nearest its span midpoint.
    anchor_times = time_spans.mean(axis=1)
    train_camera_times = dataset_view.sensor_times[camera_name][train_camera_indices]
    chosen_camera_indices = train_camera_indices[
        nearest_reference_positions(train_camera_times, anchor_times)
    ]

    # Step 3: camera matrix, pixel footprint radius and LiDAR sweep filter settings.
    intrinsics = dataset_view.manifest["cameras"][camera_name]
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ])
    footprint_radius_pixels = max(
        intrinsics["fx"] * np.tan(azimuth_cell_rad),
        intrinsics["fy"] * np.tan(elevation_cell_rad),
    )
    minimum_radius_m, ego_half_extents_m = _lidar_sweep_filter_arguments(dataset_view)

    # Step 4: paint per chosen frame so each image and occluder sweep is loaded once.
    colors = np.repeat(NEUTRAL_GREY_RGB[None, :], len(anchor_means), axis=0)
    painted = np.zeros(len(anchor_means), dtype=bool)
    camera_stamps_ns = np.full(len(anchor_means), UNPAINTED_STAMP_NS, dtype=np.int64)
    train_lidar_times = dataset_view.sensor_times[lidar_name][train_lidar_indices]
    for camera_index in np.unique(chosen_camera_indices):
        frame_rows = np.flatnonzero(chosen_camera_indices == camera_index)
        image_bgr = cv2.imread(str(camera_paths[camera_index]), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"cannot read train image {camera_paths[camera_index]}")
        # Occluders: the train sweep nearest in time to this camera frame, moved to the map frame.
        nearest_lidar_position = nearest_reference_positions(
            train_lidar_times, np.array([dataset_view.sensor_times[camera_name][camera_index]])
        )[0]
        lidar_index = train_lidar_indices[nearest_lidar_position]
        lidar_sweep = load_lidar_sweep(lidar_paths[lidar_index], minimum_radius_m, ego_half_extents_m)
        lidar_pose = dataset_view.sensor_poses[lidar_name][lidar_index]
        occluder_points_map = lidar_sweep[:, :3] @ lidar_pose[:3, :3].T + lidar_pose[:3, 3]
        frame_colors, frame_painted = paint_visible_seed_colors(
            anchor_means[frame_rows],
            dataset_view.sensor_poses[camera_name][camera_index],
            camera_matrix,
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            occluder_points_map,
            footprint_radius_pixels,
            occlusion_margin_m,
        )
        # Frame timestamp = integer after the underscore in "<index>_<timestamp_ns>.<ext>".
        name_match = FRAME_FILE_PATTERN.match(camera_paths[camera_index].name)
        if name_match is None:
            raise ValueError(f"camera frame name {camera_paths[camera_index].name} breaks the pattern")
        painted_rows = frame_rows[frame_painted]
        colors[painted_rows] = frame_colors[frame_painted]
        painted[painted_rows] = True
        camera_stamps_ns[painted_rows] = int(name_match.group(2))
    return colors, painted, camera_stamps_ns
