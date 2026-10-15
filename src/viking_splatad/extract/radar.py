"""Radar PointCloud2 handling (sidecar for alignment diagnostics; never enters SplatAD)."""

from typing import Dict, Sequence

import numpy as np

from viking_splatad.registry import transform


@transform("T23", "Convert a decoded radar PointCloud2 into a float32 [n,k] array, dropping non-finite rows")
def radar_cloud_to_array(
    decoded_fields: Dict[str, np.ndarray], field_names: Sequence[str]
) -> np.ndarray:
    """Stack selected radar fields into a float32 matrix without NaN/inf rows.

    A row is kept iff every *requested* field is finite; non-requested fields are
    ignored (e.g. a NaN azimuth does not remove a point when azimuth is not asked for).

    Args:
        decoded_fields: Output of decode_pointcloud2 (name -> length-n array).
        field_names: Fields to output, in column order.

    Returns:
        float32 array [m, len(field_names)], m <= n.

    Raises:
        KeyError: If a requested field is absent from the cloud.
    """
    # Step 1: fail loudly and by name on a missing field.
    for field_name in field_names:
        if field_name not in decoded_fields:
            raise KeyError(f"Radar cloud has no field {field_name!r}")
    # Step 2: column-stack requested fields as float32 -> [n, k].
    stacked = np.stack(
        [np.asarray(decoded_fields[field_name], dtype=np.float32) for field_name in field_names],
        axis=1,
    )
    # Step 3: keep rows whose every requested value is finite.
    return stacked[np.all(np.isfinite(stacked), axis=1)]
