"""Unit tests for extract/radar.py (T23 radar cloud -> array)."""

import numpy as np
import pytest

from viking_splatad.extract.lidar import decode_pointcloud2
from viking_splatad.extract.radar import radar_cloud_to_array

ROS_FLOAT32 = 7  # sensor_msgs/PointField FLOAT32 id, written independently of the implementation
FLOAT32_BYTES = 4
RADAR_FIELD_NAMES = ["x", "y", "z", "range", "elevation", "azimuth", "power", "doppler"]


def encode_radar_cloud(columns):
    """Encode float32 columns (dict name -> array) as packed PointCloud2 message kwargs.

    Args:
        columns: Field name -> 1-D float32 array, all equal length.

    Returns:
        Keyword arguments for decode_pointcloud2.
    """
    point_count = len(next(iter(columns.values())))
    point_step = FLOAT32_BYTES * len(columns)
    # Interleave columns point-by-point: row-major [point, field] float32 matrix.
    packed = np.stack([columns[name] for name in columns], axis=1).astype("<f4")
    return dict(
        fields=[
            {"name": name, "offset": FLOAT32_BYTES * field_index, "datatype": ROS_FLOAT32, "count": 1}
            for field_index, name in enumerate(columns)
        ],
        point_step=point_step,
        row_step=point_step * point_count,
        width=point_count,
        height=1,
        is_bigendian=False,
        data=packed.tobytes(),
    )


@pytest.mark.covers("T23")
def test_radar_cloud_to_array_selects_fields_and_drops_nonfinite_rows():
    """Selected columns come out in requested order; NaN/inf rows are removed."""
    rng = np.random.default_rng(seed=3)
    point_count = 12
    columns = {name: rng.normal(size=point_count).astype(np.float32) for name in RADAR_FIELD_NAMES}
    columns["x"][2] = np.nan            # bad row via a requested field
    columns["power"][5] = np.inf        # bad row via a requested field
    columns["azimuth"][7] = np.nan      # NaN in a NON-requested field must not drop the row
    decoded = decode_pointcloud2(**encode_radar_cloud(columns))
    requested = ["x", "y", "z", "power", "doppler"]
    result = radar_cloud_to_array(decoded, requested)
    # Expected computed independently: stack requested columns and mask finite rows.
    expected_full = np.stack([columns[name] for name in requested], axis=1)
    expected = expected_full[np.all(np.isfinite(expected_full), axis=1)]
    assert result.dtype == np.float32
    assert result.shape == (point_count - 2, len(requested))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.covers("T23")
def test_radar_cloud_to_array_missing_field_raises():
    """Requesting an absent field raises KeyError naming it."""
    decoded = decode_pointcloud2(**encode_radar_cloud({"x": np.zeros(3, dtype=np.float32)}))
    with pytest.raises(KeyError, match="power"):
        radar_cloud_to_array(decoded, ["x", "power"])
