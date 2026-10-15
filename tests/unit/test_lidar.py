"""Unit tests for extract/lidar.py (T11 decode, T12 frame change, T14 intensity, T19 channels)."""

import numpy as np
import pytest

from viking_splatad.extract.lidar import (
    ROS_DATATYPE_TO_NUMPY,
    assemble_scan_array,
    assign_channels,
    decode_pointcloud2,
    normalise_intensity,
    transform_points_to_sensor_frame,
)

# ROS sensor_msgs/PointField datatype ids (INT8=1 ... FLOAT64=8), written out in the test
# independently of the implementation table so a wrong table entry is caught.
ROS_INT8, ROS_UINT8, ROS_INT16, ROS_UINT16 = 1, 2, 3, 4
ROS_INT32, ROS_UINT32, ROS_FLOAT32, ROS_FLOAT64 = 5, 6, 7, 8


def encode_padded_cloud(point_count, byte_order_prefix, row_padding_bytes, height):
    """Encode a synthetic padded structured cloud as raw PointCloud2 bytes.

    Args:
        point_count: Points per row (width).
        byte_order_prefix: "<" little endian or ">" big endian.
        row_padding_bytes: Extra bytes appended to every row (row_step > width*point_step).
        height: Number of rows.

    Returns:
        (message_kwargs, expected_columns): decode inputs and the ground-truth arrays
        (flattened row-major).
    """
    point_step = 32  # 28 payload bytes + 4 padding bytes; the padding is the point of the test
    # Field layout with deliberate gaps (offset 12..15 and 28..31 unused).
    field_layout = [
        ("x", 0, ROS_FLOAT32, np.float32),
        ("y", 4, ROS_FLOAT32, np.float32),
        ("z", 8, ROS_FLOAT32, np.float32),
        ("intensity", 16, ROS_FLOAT32, np.float32),
        ("t", 20, ROS_UINT32, np.uint32),
        ("ring", 24, ROS_UINT16, np.uint16),
        ("flag", 26, ROS_INT8, np.int8),
    ]
    total_points = point_count * height
    rng = np.random.default_rng(seed=7)
    expected_columns = {
        "x": rng.normal(size=total_points).astype(np.float32),
        "y": rng.normal(size=total_points).astype(np.float32),
        "z": rng.normal(size=total_points).astype(np.float32),
        "intensity": rng.uniform(0, 255, size=total_points).astype(np.float32),
        "t": rng.integers(0, 10**6, size=total_points).astype(np.uint32),
        "ring": rng.integers(0, 64, size=total_points).astype(np.uint16),
        "flag": rng.integers(-100, 100, size=total_points).astype(np.int8),
    }
    # A NaN no-return point must survive decoding untouched.
    expected_columns["x"][0] = np.nan
    row_step = point_count * point_step + row_padding_bytes
    buffer = bytearray(row_step * height)
    for row_index in range(height):
        for column_index in range(point_count):
            flat_index = row_index * point_count + column_index
            base = row_index * row_step + column_index * point_step
            for name, offset, _, numpy_type in field_layout:
                encoded = np.array(expected_columns[name][flat_index], dtype=numpy_type)
                encoded = encoded.astype(np.dtype(numpy_type).newbyteorder(byte_order_prefix))
                buffer[base + offset : base + offset + encoded.nbytes] = encoded.tobytes()
    message_kwargs = dict(
        fields=[
            {"name": name, "offset": offset, "datatype": datatype, "count": 1}
            for name, offset, datatype, _ in field_layout
        ],
        point_step=point_step,
        row_step=row_step,
        width=point_count,
        height=height,
        is_bigendian=byte_order_prefix == ">",
        data=bytes(buffer),
    )
    return message_kwargs, expected_columns


@pytest.mark.covers("T11")
@pytest.mark.parametrize("byte_order_prefix", ["<", ">"])
def test_decode_pointcloud2_roundtrip_with_padding(byte_order_prefix):
    """Encode padded multi-row cloud (both endiannesses) -> decode equals input, NaN kept."""
    message_kwargs, expected_columns = encode_padded_cloud(
        point_count=5, byte_order_prefix=byte_order_prefix, row_padding_bytes=12, height=3
    )
    decoded = decode_pointcloud2(**message_kwargs)
    assert set(decoded) == set(expected_columns)
    for name, expected in expected_columns.items():
        np.testing.assert_array_equal(decoded[name], expected)
        assert decoded[name].dtype == expected.dtype
    # No-return point stays present (filterable), not dropped.
    assert np.isnan(decoded["x"][0]) and decoded["x"].shape[0] == 15


def test_ros_datatype_table_covers_int8_to_float64():
    """The datatype table maps ids 1..8 to the documented numpy kinds."""
    expected = {1: np.int8, 2: np.uint8, 3: np.int16, 4: np.uint16,
                5: np.int32, 6: np.uint32, 7: np.float32, 8: np.float64}
    assert {key: np.dtype(value) for key, value in ROS_DATATYPE_TO_NUMPY.items()} == {
        key: np.dtype(value) for key, value in expected.items()
    }


def make_rigid_transform(yaw_radians, translation):
    """Build a 4x4 rotation-about-z + translation transform from first principles."""
    cos_yaw, sin_yaw = np.cos(yaw_radians), np.sin(yaw_radians)
    transform = np.eye(4)
    transform[:3, :3] = [[cos_yaw, -sin_yaw, 0], [sin_yaw, cos_yaw, 0], [0, 0, 1]]
    transform[:3, 3] = translation
    return transform


@pytest.mark.covers("T12")
def test_transform_points_to_sensor_frame_known_extrinsic():
    """Lidar yawed 90 deg and offset in base_link: analytic sensor-frame coordinates."""
    # Sensor frame origin sits at (1,2,3) in base_link, axes yawed +90 deg about z, so
    # T_base_lidar = [R|t]; the needed transform is its inverse T_lidar_base.
    t_base_lidar = make_rigid_transform(np.pi / 2, [1.0, 2.0, 3.0])
    t_lidar_base = np.linalg.inv(t_base_lidar)
    points_base = np.array([[1.0, 2.0, 3.0], [2.0, 2.0, 3.0], [1.0, 5.0, 4.0]])
    # Hand-derived: lidar x axis = base +y, lidar y axis = base -x.
    # offset (1,0,0) in base -> lidar (0,-1,0); offset (0,3,1) -> lidar (3,0,1).
    expected = np.array([[0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [3.0, 0.0, 1.0]])
    result = transform_points_to_sensor_frame(points_base, t_lidar_base)
    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.covers("T14")
def test_normalise_intensity_scale_monotone_and_clip_count():
    """Percentile scale, monotone mapping into [0,1], correct clipped count."""
    # 100 values 1..100 in two scans; 99th percentile by numpy linear interpolation.
    scans = [np.arange(1, 51, dtype=np.float32), np.arange(51, 101, dtype=np.float32)]
    percentile = 99.0
    result = normalise_intensity(scans, percentile=percentile)
    expected_scale = np.percentile(np.arange(1, 101), percentile)  # = 99.01
    assert result.scale == pytest.approx(expected_scale)
    joined = np.concatenate(result.normalised_scans)
    assert joined.min() >= 0.0 and joined.max() <= 1.0
    assert np.all(np.diff(joined) >= 0)  # input was ascending -> output monotone
    # Values strictly above 99.01 are 100 only -> exactly one clipped.
    assert result.clipped_count == 1
    assert result.total_count == 100
    assert result.clipped_fraction == pytest.approx(1 / 100)
    assert joined[0] == pytest.approx(1 / expected_scale)


@pytest.mark.covers("T14")
def test_normalise_intensity_dtype_max_and_negative_clip():
    """Explicit dtype max is the scale; negatives are clipped and counted."""
    scans = [np.array([-5.0, 0.0, 127.5, 255.0, 300.0], dtype=np.float32)]
    result = normalise_intensity(scans, dtype_max=float(np.iinfo(np.uint8).max))
    assert result.scale == 255.0
    np.testing.assert_allclose(
        result.normalised_scans[0], [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-7
    )
    assert result.clipped_count == 2  # -5 (below) and 300 (above)


@pytest.mark.covers("T19")
def test_assign_channels_by_ring_matches_mapping_keys():
    """Points on known rings get channel == ring key; unknown ring raises."""
    mapping = {channel: -10.0 + 2.5 * channel for channel in range(8)}
    rings = np.array([0, 3, 7, 3, 1])
    channels = assign_channels(mapping, ring=rings)
    np.testing.assert_array_equal(channels, rings)
    with pytest.raises(ValueError):
        assign_channels(mapping, ring=np.array([0, 99]))


@pytest.mark.covers("T19")
def test_assign_channels_by_nearest_elevation():
    """Without ring: nearest-elevation key, with noise below half the beam spacing."""
    mapping = {channel: -10.0 + 2.5 * channel for channel in range(8)}
    true_channels = np.array([0, 5, 7, 2, 2, 4])
    beam_spacing = 2.5
    noise = np.array([0.3, -0.4, 0.1, 0.0, -0.3, 0.2])  # all < spacing/2
    assert np.all(np.abs(noise) < beam_spacing / 2)
    elevation_deg = np.array([mapping[int(key)] for key in true_channels]) + noise
    channels = assign_channels(mapping, elevation_deg=elevation_deg)
    np.testing.assert_array_equal(channels, true_channels)


def test_assemble_scan_array_filters_and_layout():
    """Stack to float32 [n,6]; drop non-finite, zero-range and ego-box points."""
    xyz = np.array([
        [5.0, 0.0, 0.0],      # kept
        [0.1, 0.1, 0.1],      # inside ego box -> removed
        [0.1, 0.1, 2.0],      # |z| >= dz -> kept (OR rule)
        [np.nan, 1.0, 1.0],   # non-finite -> removed
        [0.0, 0.0, 0.0],      # zero range -> removed
        [0.0, 7.0, 0.0],      # kept via |y|>=dy
    ])
    intensity = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    relative_time = np.array([-0.05, -0.02, 0.0, 0.01, 0.02, 0.05])
    channel = np.array([0, 1, 2, 3, 4, 5])
    min_lidar_dist = (1.0, 1.0, 1.0)
    scan = assemble_scan_array(xyz, intensity, relative_time, channel, min_lidar_dist)
    assert scan.dtype == np.float32 and scan.shape == (3, 6)
    np.testing.assert_allclose(
        scan,
        [[5.0, 0.0, 0.0, 0.1, -0.05, 0], [0.1, 0.1, 2.0, 0.3, 0.0, 2], [0.0, 7.0, 0.0, 0.6, 0.05, 5]],
        atol=1e-6,
    )
