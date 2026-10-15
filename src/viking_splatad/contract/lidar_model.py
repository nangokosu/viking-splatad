"""Lidar model parameters derived from data: ego box, elevation map, azimuth tiling."""

import math
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from viking_splatad.registry import transform

# SplatAD rasterises lidar in tiles that are 8 channels tall and 32 azimuth columns wide
# (neurad-studio contract), so these are fixed by the target format, not tunable.
CHANNELS_PER_TILE = 8
AZIMUTH_COLUMNS_PER_TILE = 32


@transform(
    "T15",
    "Derive min_lidar_dist (dx,dy,dz) from the near-field robot-body cluster of an "
    "accumulated sensor-frame cloud",
)
def derive_min_lidar_dist(points_xyz: np.ndarray, margin: float) -> Tuple[float, float, float]:
    """Find the robot-body cluster by the first gap in the radial histogram.

    Algorithm: range r = |p|. Histogram r with sqrt(n) bins over [r_min, r_max]
    (data-derived bin width). The body is the first contiguous run of non-empty bins
    starting at the innermost bin; the first empty bin after it is the gap. The
    thresholds are the per-axis max |coordinate| of that cluster plus ``margin``, so
    the removed box {|x|<dx and |y|<dy and |z|<dz} encloses the cluster.
    Complexity O(n).

    Args:
        points_xyz: [n,3] sensor-frame points accumulated over many scans.
        margin: Extra metres added to each half-extent.

    Returns:
        (dx, dy, dz) half-extents; a point is kept iff |x|>=dx or |y|>=dy or |z|>=dz.

    Raises:
        ValueError: If no radial gap separates a near-field cluster from the scene.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    # Drop non-finite points; they carry no range information.
    points_xyz = points_xyz[np.all(np.isfinite(points_xyz), axis=1)]
    ranges = np.linalg.norm(points_xyz, axis=1)
    # Bin count sqrt(n): standard data-derived rule (Excel/Sturges-family heuristic).
    bin_count = max(int(math.ceil(math.sqrt(len(ranges)))), 2)
    counts, bin_edges = np.histogram(ranges, bins=bin_count)
    # First empty bin = start of the gap after the near-field run.
    empty_bin_indices = np.flatnonzero(counts == 0)
    if counts.size == 0 or empty_bin_indices.size == 0:
        raise ValueError("No radial gap found: cannot isolate a robot-body cluster")
    first_gap_bin = int(empty_bin_indices[0])
    # Cluster = points below the lower edge of the gap bin.
    cluster_radius = bin_edges[first_gap_bin]
    cluster = points_xyz[ranges < cluster_radius]
    if cluster.shape[0] == 0:
        raise ValueError("Near-field cluster is empty")
    # Per-axis half-extent enclosing the cluster, plus the safety margin.
    half_extents = np.abs(cluster).max(axis=0) + margin
    return float(half_extents[0]), float(half_extents[1]), float(half_extents[2])


def split_sorted_by_otsu_gap(sorted_values: np.ndarray) -> np.ndarray:
    """Cluster 1-D sorted values by splitting at gaps above an Otsu threshold.

    Otsu: choose the threshold on the gap sizes maximising the between-class variance
    w0*w1*(mean0-mean1)^2, separating many small within-beam gaps from few large
    between-beam gaps.

    Args:
        sorted_values: Ascending 1-D array.

    Returns:
        Integer cluster label per value (0..k-1, ascending).
    """
    gaps = np.diff(sorted_values)
    sorted_gaps = np.sort(gaps)
    gap_total = sorted_gaps.size
    # Prefix sums give class weights/means for every candidate split in O(n).
    cumulative_sum = np.cumsum(sorted_gaps)
    low_counts = np.arange(1, gap_total)  # first class sizes 1..n-1
    low_means = cumulative_sum[:-1] / low_counts
    high_means = (cumulative_sum[-1] - cumulative_sum[:-1]) / (gap_total - low_counts)
    between_variance = low_counts * (gap_total - low_counts) * (low_means - high_means) ** 2
    best_split = int(np.argmax(between_variance))
    # Threshold midway between the two classes' boundary gap values.
    threshold = 0.5 * (sorted_gaps[best_split] + sorted_gaps[best_split + 1])
    # A new cluster starts after every gap above the threshold.
    return np.concatenate([[0], np.cumsum(gaps > threshold)])


@transform(
    "T16",
    "Derive per-channel elevation mapping (channel -> median elevation deg) from data",
)
def derive_elevation_mapping(
    points_xyz: np.ndarray, ring: Optional[np.ndarray] = None
) -> Dict[int, float]:
    """Median elevation per beam, ascending, with the 8-channel tile check.

    Elevation = asin(z/d). With ``ring`` the keys are the ring ids (listed in ascending
    elevation order); without it elevations are clustered by Otsu gap splitting and
    keys are ranks 0..C-1 in ascending elevation.

    Args:
        points_xyz: [n,3] sensor-frame points.
        ring: Optional [n] integer beam index per point.

    Returns:
        Ordered dict channel -> median elevation in degrees (ascending).

    Raises:
        ValueError: If the channel count is not a multiple of 8.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    distances = np.linalg.norm(points_xyz, axis=1)
    # Drop non-finite and zero-range points: asin(z/d) is undefined for them.
    valid = np.isfinite(distances) & (distances > 0)
    # Elevation above the sensor xy-plane: asin(z/d), in degrees.
    elevation_deg = np.degrees(np.arcsin(points_xyz[valid, 2] / distances[valid]))
    if ring is not None:
        ring_valid = np.asarray(ring)[valid]
        medians = {
            int(ring_id): float(np.median(elevation_deg[ring_valid == ring_id]))
            for ring_id in np.unique(ring_valid)
        }
        # Order by elevation so the mapping is "sorted ascending".
        ordered_items = sorted(medians.items(), key=lambda item: item[1])
        mapping = dict(ordered_items)
    else:
        # No ring ids: sort, cluster by gap size, and take each cluster's median as one beam.
        order = np.argsort(elevation_deg)
        sorted_elevation = elevation_deg[order]
        labels = split_sorted_by_otsu_gap(sorted_elevation)
        mapping = {
            int(label): float(np.median(sorted_elevation[labels == label]))
            for label in np.unique(labels)
        }
    if len(mapping) % CHANNELS_PER_TILE != 0:
        raise ValueError(
            f"{len(mapping)} channels is not a multiple of {CHANNELS_PER_TILE}"
        )
    return mapping


@transform(
    "T17",
    "Azimuth resolution from columns per ring and tile_width = ceil(360/(res*32))",
)
def derive_azimuth_tiling(
    columns_per_ring: Union[int, float, Sequence[float]],
    integral_tolerance: float = 1e-6,
) -> Tuple[float, int]:
    """Azimuth resolution and SplatAD tile width from measured columns.

    Math: res = 360/columns. tiles = 360/(res*32) = columns/32 must be integral
    (within ``integral_tolerance``, a float round-off guard) or tiles would not cover
    the sweep exactly; tile_width = ceil(tiles).

    Args:
        columns_per_ring: Measured columns (scalar, or per-ring counts -> median).
        integral_tolerance: Allowed |tiles - round(tiles)|.

    Returns:
        (azimuth_resolution_deg, tile_width).

    Raises:
        ValueError: If tiles is not integral.
    """
    # Median over rings is robust to rings with dropped returns.
    column_count = float(np.median(np.atleast_1d(columns_per_ring)))
    full_circle_deg = 360.0
    resolution_deg = full_circle_deg / column_count
    # Tiles per full sweep (algebraically equal to column_count / 32).
    tiles =full_circle_deg / (resolution_deg * AZIMUTH_COLUMNS_PER_TILE)
    if abs(tiles - round(tiles)) > integral_tolerance:
        raise ValueError(
            f"{column_count} columns gives {tiles} tiles of {AZIMUTH_COLUMNS_PER_TILE}: not integral"
        )
    # Subtracting the tolerance stops ceil from rounding a round-off-high integer up by one.
    return resolution_deg, int(math.ceil(tiles - integral_tolerance))
