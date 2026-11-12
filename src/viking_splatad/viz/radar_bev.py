"""Label-free radar/LiDAR alignment picture; pixel overlap is not semantic accuracy."""

from pathlib import Path

import cv2
import numpy as np

from viking_splatad.registry import transform


@transform("T47", "Map/parser XY points to inclusive bounded BEV pixels with y increasing upward")
def bev_pixels(points: np.ndarray, bounds: np.ndarray, image_shape):
    """Return integer row/column coordinates for finite XY points inside one shared map extent.

    Args:
        points: [N,2] coordinates; bounds: min/max XY [2,2].
        image_shape: Positive (height,width), each at least two to represent both bounds.

    Returns:
        Rows/columns [K,2] and source keep-mask [N]. Exact upper bounds map to the last pixel.
    """
    points, bounds = np.asarray(points), np.asarray(bounds)
    height, width = image_shape
    if (points.ndim != 2 or points.shape[1] != 2 or bounds.shape != (2, 2)
            or not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0])
            or min(height, width) < 2):
        raise ValueError("finite increasing XY bounds and a nondegenerate image required")
    kept = np.isfinite(points).all(axis=1) & (points >= bounds[0]).all(axis=1) & (points <= bounds[1]).all(axis=1)
    normalized = (points[kept] - bounds[0]) / (bounds[1] - bounds[0])
    coordinates = np.floor(normalized * np.array([width - 1, height - 1])).astype(np.int64)
    return np.column_stack([height - 1 - coordinates[:, 1], coordinates[:, 0]]), kept


def write_radar_lidar_bev(lidar_points: np.ndarray, anchor_points: np.ndarray,
                          path: Path, image_pixels: int = 1024) -> Path:
    """Write both training inputs in one square XY view without GT or selection changes.

    Args:
        lidar_points, anchor_points: Already co-framed xyz [N,3], [S,3].
        path: PNG destination; image_pixels: visual resolution, default 1024 matches preflight previews.

    Returns:
        Written path. Gray = sampled LiDAR, orange = radar anchors, white = same XY pixel.
        This picture diagnoses gross alignment, not trunk recovery or 3D correspondence.
    """
    lidar_points, anchor_points = np.asarray(lidar_points), np.asarray(anchor_points)
    if (not len(lidar_points) or not len(anchor_points) or lidar_points.shape[1] != 3
            or anchor_points.shape[1] != 3 or not np.isfinite(lidar_points).all()
            or not np.isfinite(anchor_points).all()):
        raise ValueError("nonempty finite co-framed LiDAR and anchor points required")
    combined = np.concatenate([lidar_points[:, :2], anchor_points[:, :2]])
    extent = combined.max(axis=0) - combined.min(axis=0)
    side = float(extent.max())
    if side <= 0:
        raise ValueError("BEV inputs have no spatial extent")
    centre = (combined.min(axis=0) + combined.max(axis=0)) / 2
    bounds = np.stack([centre - side / 2, centre + side / 2])
    lidar_pixels, _ = bev_pixels(lidar_points[:, :2], bounds, (image_pixels, image_pixels))
    radar_pixels, _ = bev_pixels(anchor_points[:, :2], bounds, (image_pixels, image_pixels))
    lidar_mask, radar_mask = (np.zeros((image_pixels, image_pixels), dtype=bool) for _ in range(2))
    lidar_mask[tuple(lidar_pixels.T)] = True
    radar_mask[tuple(radar_pixels.T)] = True
    image = np.zeros((image_pixels, image_pixels, 3), dtype=np.uint8)
    # These are display-only BGR colors, not confidence values or semantic labels.
    image[lidar_mask] = [160, 160, 160]
    image[radar_mask] = [0, 140, 255]
    image[lidar_mask & radar_mask] = [255, 255, 255]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"could not write radar/LiDAR BEV: {path}")
    path.with_suffix(".txt").write_text(
        "Label-free training input alignment diagnostic.\nGray: sampled LiDAR. Orange: radar anchors.\n"
        "White: same XY pixel, not GT agreement or 3D correspondence.\n"
        f"XY min/max: {bounds.tolist()}; x right, y up; side {side} m.\n")
    return path
