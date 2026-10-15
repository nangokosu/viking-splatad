"""Lidar point-cloud handling: PointCloud2 decode, frame change, intensity, channels."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.registry import transform

# sensor_msgs/PointField datatype ids -> numpy base types (ROS message definition).
ROS_DATATYPE_TO_NUMPY = {
    1: np.int8,
    2: np.uint8,
    3: np.int16,
    4: np.uint16,
    5: np.int32,
    6: np.uint32,
    7: np.float32,
    8: np.float64,
}


def read_field_attribute(field, attribute_name: str):
    """Read ``attribute_name`` from a field given as dict or message-like object.

    Args:
        field: PointField as a dict (key lookup) or an object (attribute lookup).
        attribute_name: Name of the key/attribute to read.

    Returns:
        The stored value for ``attribute_name``.
    """
    # Dicts (e.g. decoded JSON/bag dumps) use keys; ROS message objects use attributes.
    if isinstance(field, dict):
        return field[attribute_name]
    return getattr(field, attribute_name)


@transform(
    "T11",
    "Decode a sensor_msgs/PointCloud2 into per-field numpy arrays (offsets, padding, "
    "endianness, ROS datatypes)",
)
def decode_pointcloud2(
    fields: Sequence,
    point_step: int,
    row_step: int,
    width: int,
    height: int,
    is_bigendian: bool,
    data: bytes,
) -> Dict[str, np.ndarray]:
    """Decode raw PointCloud2 bytes into a dict field name -> array.

    A structured dtype with explicit offsets and itemsize=point_step skips padding;
    each row is read from byte ``row_index*row_step`` so row padding is honoured.
    NaN/no-return points are kept (callers filter them).

    Args:
        fields: Items with name/offset/datatype/count (dicts or objects).
        point_step: Bytes per point.
        row_step: Bytes per row (>= width*point_step).
        width: Points per row.
        height: Number of rows.
        is_bigendian: Byte order of the data.
        data: Raw message bytes.

    Returns:
        name -> array of length width*height (shape [n,count] when count > 1),
        in native byte order.
    """
    # Step 1: build a structured dtype describing one point, honouring declared byte order.
    byte_order_prefix = ">" if is_bigendian else "<"
    names, formats, offsets = [], [], []
    for field in fields:
        base_type = np.dtype(ROS_DATATYPE_TO_NUMPY[int(read_field_attribute(field, "datatype"))])
        base_type = base_type.newbyteorder(byte_order_prefix)
        element_count = int(read_field_attribute(field, "count"))
        names.append(read_field_attribute(field, "name"))
        # count>1 becomes a sub-array so one field stays one column group.
        formats.append(base_type if element_count == 1 else (base_type, (element_count,)))
        offsets.append(int(read_field_attribute(field, "offset")))
    point_dtype = np.dtype(
        {"names": names, "formats": formats, "offsets": offsets, "itemsize": point_step}
    )
    # Step 2: read every row at its own start so inter-row padding is skipped.
    row_arrays: List[np.ndarray] = [
        np.frombuffer(data, dtype=point_dtype, count=width, offset=row_index * row_step)
        for row_index in range(height)
    ]
    structured = np.concatenate(row_arrays) if row_arrays else np.empty(0, dtype=point_dtype)
    # Step 3: convert each column to native byte order for downstream numpy/torch use.
    return {
        name: structured[name].astype(structured[name].dtype.newbyteorder("="))
        for name in names
    }


@transform("T12", "Transform lidar points from a source frame into the lidar sensor frame")
def transform_points_to_sensor_frame(
    points_source: np.ndarray, transform_lidar_source: np.ndarray
) -> np.ndarray:
    """Apply T_lidar_source: p_lidar = R @ p_source + t.

    Args:
        points_source: [n,3] points in the source frame.
        transform_lidar_source: [4,4] float64 pose mapping source -> lidar frame.

    Returns:
        [n,3] float64 points in the lidar sensor frame.
    """
    points_source = np.asarray(points_source, dtype=np.float64)
    rotation = transform_lidar_source[:3, :3]
    translation = transform_lidar_source[:3, 3]
    # Row-vector form of R @ p: p @ R^T, then add translation.
    return points_source @ rotation.T + translation


@dataclass(frozen=True)
class IntensityNormalisation:
    """Result of intensity normalisation.

    Attributes:
        normalised_scans: Per-scan float32 arrays clipped to [0,1].
        scale: Divisor used (store in the manifest).
        clipped_count: Values outside [0, scale] before clipping.
        total_count: Total values processed.
        clipped_fraction: clipped_count / total_count.
    """

    normalised_scans: List[np.ndarray]
    scale: float
    clipped_count: int
    total_count: int
    clipped_fraction: float


@transform("T14", "Normalise lidar intensity to [0,1] with a data-derived scale; count clipping")
def normalise_intensity(
    intensity_scans: Sequence[np.ndarray],
    percentile: float = 99.0,
    dtype_max: Optional[float] = None,
) -> IntensityNormalisation:
    """Scale intensities by a global scale and clip to [0,1].

    scale = dtype_max if given, else the global ``percentile`` of finite values over
    all scans (robust to outlier spikes; 99 is a conventional robust-max default).
    normalised = clip(value/scale, 0, 1) is monotone in value.

    Args:
        intensity_scans: Per-scan intensity arrays.
        percentile: Percentile (0-100) used when ``dtype_max`` is None.
        dtype_max: Known sensor maximum (e.g. 255 for uint8), overrides percentile.

    Returns:
        IntensityNormalisation with scale and clipped statistics.

    Raises:
        ValueError: If the derived scale is not positive.
    """
    # Pool all scans so one global scale keeps intensities comparable across scans.
    all_values = np.concatenate([np.asarray(scan, dtype=np.float64).ravel() for scan in intensity_scans])
    # NaN/inf must not influence the percentile.
    finite_values = all_values[np.isfinite(all_values)]
    scale = float(dtype_max) if dtype_max is not None else float(np.percentile(finite_values, percentile))
    # `not scale > 0` also rejects NaN.
    if not scale > 0:
        raise ValueError(f"Non-positive intensity scale {scale}")
    normalised_scans = []
    clipped_count = 0
    for scan in intensity_scans:
        # Divide by scale, so scale maps to 1.0.
        scaled = np.asarray(scan, dtype=np.float64) / scale
        # Clipped = finite values that fall outside the unit interval.
        clipped_count += int(np.count_nonzero(np.isfinite(scaled) & ((scaled > 1.0) | (scaled < 0.0))))
        normalised_scans.append(np.clip(scaled, 0.0, 1.0).astype(np.float32))
    # Denominator is finite values only, matching how clipped_count is counted.
    total_count = int(finite_values.size)
    return IntensityNormalisation(
        normalised_scans, scale, clipped_count, total_count, clipped_count / total_count
    )


@transform("T19", "Assign each point a channel index matching the elevation-mapping keys")
def assign_channels(
    elevation_mapping: Dict[int, float],
    ring: Optional[np.ndarray] = None,
    elevation_deg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Map points to channel keys by ring id, else by nearest elevation.

    Args:
        elevation_mapping: channel -> elevation degrees.
        ring: Optional [n] ring ids; each must be a mapping key.
        elevation_deg: [n] point elevations in degrees (used when ring is None).

    Returns:
        [n] int64 channel indices (keys of the mapping).

    Raises:
        ValueError: If a ring is not a key, or neither input is given.
    """
    channel_keys = np.array(list(elevation_mapping.keys()), dtype=np.int64)
    # Preferred path: ring ids are already channel keys; validate and pass through.
    if ring is not None:
        ring = np.asarray(ring, dtype=np.int64)
        unknown_rings = np.setdiff1d(ring, channel_keys)
        if unknown_rings.size:
            raise ValueError(f"Rings not in elevation mapping: {unknown_rings.tolist()}")
        return ring
    if elevation_deg is None:
        raise ValueError("Need ring or elevation_deg")
    channel_elevations = np.array(list(elevation_mapping.values()), dtype=np.float64)
    # Nearest beam: argmin |elevation - beam elevation| over the beam axis.
    # Builds an [n, n_beams] distance matrix: O(n * n_beams) time and memory.
    distance_to_beams = np.abs(np.asarray(elevation_deg, dtype=np.float64)[:, None] - channel_elevations[None, :])
    return channel_keys[np.argmin(distance_to_beams, axis=1)]


def assemble_scan_array(
    points_xyz: np.ndarray,
    intensity: np.ndarray,
    relative_time: np.ndarray,
    channel: np.ndarray,
    min_lidar_dist: Tuple[float, float, float],
) -> np.ndarray:
    """Stack x,y,z,intensity,t,channel into float32 [n,6] after filtering.

    Removed: non-finite rows, zero-range points, and ego points. A point is kept iff
    |x|>=dx or |y|>=dy or |z|>=dz.

    Args:
        points_xyz: [n,3] sensor-frame (deskewed) points.
        intensity: [n] normalised intensity.
        relative_time: [n] signed seconds relative to the reference time.
        channel: [n] channel indices.
        min_lidar_dist: (dx, dy, dz) ego box half-extents.

    Returns:
        float32 array [m,6].
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    # Column order is the SplatAD scan layout: x, y, z, intensity, t, channel.
    columns = np.column_stack([points_xyz, intensity, relative_time, channel]).astype(np.float64)
    # Drop rows with any NaN/inf in any column.
    finite_mask = np.all(np.isfinite(columns), axis=1)
    # Zero range = no-return placeholder at the sensor origin; nan_to_num avoids NaN norms.
    nonzero_range_mask = np.linalg.norm(np.nan_to_num(points_xyz), axis=1) > 0
    # Keep a point if it lies outside the ego box on at least one axis.
    outside_ego_mask = np.any(np.abs(points_xyz) >= np.asarray(min_lidar_dist), axis=1)
    return columns[finite_mask & nonzero_range_mask & outside_ego_mask].astype(np.float32)
