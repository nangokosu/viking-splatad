"""Paint radar anchor colours from undistorted train camera frames.

Pipeline: project map-frame anchors into the train camera frames in order of increasing time
distance (up to K frames, as stock SplatAD's point painting), reject per frame those that are outside
the image, behind the camera, or occluded by a closer train LiDAR return inside the radar angular
footprint, and read the colour from the first frame that accepts the anchor. Unpainted anchors keep
neutral grey. Only train frames are read, so no held-out image leaks into the initialisation.
"""

import cv2
import numpy as np

from viking_splatad.dataset.run_split import nearest_reference_positions
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN
from viking_splatad.registry import transform
from viking_splatad.splatad.viking_conversions import load_lidar_sweep

# Camera-stamp value for an anchor that no train camera frame painted (the /2 painting-record contract).
UNPAINTED_CAMERA_STAMP_NS = -1

# Candidate frames tried per camera when the caller gives no count: the single nearest frame. The
# configured value (configs/radar/loss_only.yaml: paint_frames_per_camera) overrides this.
DEFAULT_PAINT_FRAMES_PER_CAMERA = 1

# Cap on the anchors x train-frames time-distance matrix held at once (elements); anchors are
# ranked in chunks so memory stays bounded (2**24 float64 = 128 MiB), an arbitrary working-set size.
CANDIDATE_MATRIX_ELEMENT_BUDGET = 2**24

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


class PaintedAnchors(tuple):
    """``(rgb, painted, camera_stamps_ns)`` that also carries the footprint radius the painter used.

    A tuple subclass so callers can keep unpacking three values; the radius is an attribute.

    Attributes:
        footprint_radius_pixels: Occlusion-disc radius in pixels actually applied.
    """

    footprint_radius_pixels: float

    def __new__(cls, rgb, painted, camera_stamps_ns, footprint_radius_pixels):
        """Build the 3-tuple and attach the radius."""
        instance = super().__new__(cls, (rgb, painted, camera_stamps_ns))
        instance.footprint_radius_pixels = float(footprint_radius_pixels)
        return instance


def _nearest_train_frame_ranking(train_camera_times: np.ndarray, anchor_times: np.ndarray,
                                 candidate_count: int) -> np.ndarray:
    """Per anchor, the positions of its ``candidate_count`` nearest train frames, nearest first.

    Ranking is by absolute time distance, as stock SplatAD's ``paint_points`` (``torch.topk`` of
    ``|camera_time - lidar_time|``, smallest first). Ties go to the earlier frame (stable sort).
    Cost: O(S * T log T) time, chunked over anchors so at most CANDIDATE_MATRIX_ELEMENT_BUDGET
    distances exist at once.

    Args:
        train_camera_times: float [T] train camera times, seconds.
        anchor_times: float [S] anchor midpoint times, seconds.
        candidate_count: Frames per anchor to keep (already clipped to T).

    Returns:
        int64 [S, candidate_count] positions into ``train_camera_times``.
    """
    # Step 1: anchors per chunk so chunk_size * T stays within the element budget.
    chunk_size = max(1, CANDIDATE_MATRIX_ELEMENT_BUDGET // len(train_camera_times))
    ranking = np.empty((len(anchor_times), candidate_count), dtype=np.int64)
    for chunk_start in range(0, len(anchor_times), chunk_size):
        chunk_times = anchor_times[chunk_start:chunk_start + chunk_size]
        # Step 2: distance matrix [chunk, T], then the nearest candidate_count by stable argsort.
        distances = np.abs(train_camera_times[None, :] - chunk_times[:, None])
        ranking[chunk_start:chunk_start + chunk_size] = np.argsort(
            distances, axis=1, kind="stable")[:, :candidate_count]
    return ranking


@transform("T37", "radar anchor colours painted from the nearest unobstructed train camera frames")
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
    paint_frames_per_camera: int = DEFAULT_PAINT_FRAMES_PER_CAMERA,
) -> PaintedAnchors:
    """Paint anchors from the first of their K nearest-in-time train frames that sees them.

    Candidates: train camera frames ordered by time distance to the anchor's span midpoint; K is
    ``paint_frames_per_camera`` (one camera stream, so K = cameras x per-camera factor of stock
    SplatAD's ``paint_points_topk``), clipped to the number of train frames. Round r tries every
    still-unpainted anchor's r-th candidate; anchors sharing a candidate frame are batched so each
    image and occluder sweep loads once per round.

    Footprint: a radar cell spans ``azimuth_cell_rad`` x ``elevation_cell_rad``. A beam centred on
    the optical axis covers +-cell/2 about its axis, so the cell's pixel half-width at focal length
    f is ``f * tan(cell / 2)``. The LiDAR-occlusion disc has that radius, i.e. it covers exactly one
    resolution cell (diameter ~ one cell): ``max(fx * tan(azimuth_cell / 2), fy * tan(elevation_cell / 2))``.

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
        paint_frames_per_camera: Candidate train frames tried per anchor (K).

    Returns:
        ``PaintedAnchors``: unpacks as ``(rgb uint8 [S,3], painted bool [S], camera_stamps_ns
        int64 [S])`` (stamp of the painting frame, ``-1`` where unpainted) and has
        ``footprint_radius_pixels``, the radius used.
    """
    # Step 1: validate shapes, K and the train frame sets.
    anchor_means = np.asarray(anchor_means_map, dtype=np.float64)
    time_spans = np.asarray(anchor_time_spans_s, dtype=np.float64)
    if anchor_means.ndim != 2 or anchor_means.shape[1] != 3:
        raise ValueError(f"anchor_means_map must be [S,3], got {anchor_means.shape}")
    if time_spans.shape != (len(anchor_means), 2):
        raise ValueError(f"anchor_time_spans_s must be [{len(anchor_means)},2], got {time_spans.shape}")
    if paint_frames_per_camera < 1:
        raise ValueError(f"paint_frames_per_camera must be at least 1, got {paint_frames_per_camera}")
    camera_paths = dataset_view.sensor_frame_paths[camera_name]
    lidar_paths = dataset_view.sensor_frame_paths[lidar_name]
    train_camera_indices = _train_frame_indices(dataset_view, camera_name, train_files)
    if train_camera_indices.size == 0:
        raise ValueError("reference split contains no train camera frames")
    train_lidar_indices = _train_frame_indices(dataset_view, lidar_name, train_files)
    if train_lidar_indices.size == 0:
        raise ValueError("reference split contains no train LiDAR sweeps")

    # Step 2: rank each anchor's candidate train camera frames by time distance to its span midpoint.
    anchor_times = time_spans.mean(axis=1)
    train_camera_times = dataset_view.sensor_times[camera_name][train_camera_indices]
    candidate_count = min(paint_frames_per_camera, len(train_camera_indices))
    candidate_positions = _nearest_train_frame_ranking(train_camera_times, anchor_times, candidate_count)

    # Step 3: camera matrix, pixel footprint radius (one cell wide) and LiDAR sweep filter settings.
    intrinsics = dataset_view.manifest["cameras"][camera_name]
    camera_matrix = np.array([
        [intrinsics["fx"], 0.0, intrinsics["cx"]],
        [0.0, intrinsics["fy"], intrinsics["cy"]],
        [0.0, 0.0, 1.0],
    ])
    footprint_radius_pixels = float(max(
        intrinsics["fx"] * np.tan(azimuth_cell_rad / 2.0),
        intrinsics["fy"] * np.tan(elevation_cell_rad / 2.0),
    ))
    minimum_radius_m, ego_half_extents_m = _lidar_sweep_filter_arguments(dataset_view)

    # Step 4: rounds over candidate rank; each round batches the unpainted anchors by candidate frame.
    colors = np.repeat(NEUTRAL_GREY_RGB[None, :], len(anchor_means), axis=0)
    painted = np.zeros(len(anchor_means), dtype=bool)
    camera_stamps_ns = np.full(len(anchor_means), UNPAINTED_CAMERA_STAMP_NS, dtype=np.int64)
    train_lidar_times = dataset_view.sensor_times[lidar_name][train_lidar_indices]
    for candidate_rank in range(candidate_count):
        unpainted_rows = np.flatnonzero(~painted)
        if unpainted_rows.size == 0:
            break
        round_camera_indices = train_camera_indices[candidate_positions[unpainted_rows, candidate_rank]]
        for camera_index in np.unique(round_camera_indices):
            frame_rows = unpainted_rows[round_camera_indices == camera_index]
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
    return PaintedAnchors(colors, painted, camera_stamps_ns, footprint_radius_pixels)
