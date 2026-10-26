"""Ouster no-returns (Phase B, option C): beam directions and the per-sweep no-return sidecar.

The Ouster publishes an organized cloud, one row per ring and one column per azimuth step.
A cell whose laser got no echo has range 0 and no usable coordinates. SplatAD's ray-drop loss
needs those rays as "did not return" samples (``add_missing_points``), so the converter writes
them to ``lidar_no_return/<sensor>/<idx>_<ts_ns>.npy`` beside the returns file, which stays
returns-only. The dataparser turns them into 2000 m placeholders only when the flag is set.

A no-return has no measured direction, and the bags carry no Ouster beam-angle metadata (no
``/ouster/metadata`` topic, no sensor JSON in the calibration folder; checked 2026-10-26). The
direction of every (ring, column) cell is therefore fitted from the returns of the raw cloud.

Sidecar row layout (float32): ``dir_x, dir_y, dir_z, t, channel, column``. The direction is a
unit vector in the lidar frame at the sweep reference time (motion-compensated like the returns),
t is the signed offset from the reference time in seconds, channel the elevation-mapping key and
column the cloud column of the cell.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import numpy as np

from viking_splatad.extract.deskew import deskew_points
from viking_splatad.registry import transform

# Sidecar row layout (see module docstring).
NO_RETURN_DIRECTION_COLUMNS = slice(0, 3)
NO_RETURN_TIME_COLUMN = 3
NO_RETURN_CHANNEL_COLUMN = 4
NO_RETURN_CELL_COLUMN = 5
NO_RETURN_ROW_WIDTH = 6
# Full turn and half turn in degrees (azimuth wrapping).
FULL_TURN_DEGREES = 360.0
HALF_TURN_DEGREES = 180.0
# Candidate signs of the azimuth step per column: Ouster columns run clockwise (azimuth decreases),
# the synthetic test scene counter-clockwise; the fit picks whichever the data supports.
COLUMN_STEP_SIGNS = (1.0, -1.0)
# Unknowns of the per-ring line fit angle = intercept + slope / range.
LINE_FIT_PARAMETER_COUNT = 2


def wrap_to_half_turn(angle_degrees: np.ndarray) -> np.ndarray:
    """Wrap angles into [-180, 180) degrees.

    Args:
        angle_degrees: Angles in degrees (any shape).

    Returns:
        Wrapped angles, same shape.
    """
    # Shift by a half turn, reduce modulo a full turn, shift back.
    return (np.asarray(angle_degrees, dtype=np.float64) + HALF_TURN_DEGREES) % FULL_TURN_DEGREES - HALF_TURN_DEGREES


def circular_mean_degrees(angle_degrees: np.ndarray) -> float:
    """Mean direction of angles on the circle.

    Math: mean of the unit phasors exp(i*theta), then its argument. Unlike the arithmetic
    mean it is unaffected by the +-180 degree seam.

    Args:
        angle_degrees: [n] angles in degrees.

    Returns:
        Mean angle in degrees, in (-180, 180].
    """
    phasor_mean = np.mean(np.exp(1j * np.radians(angle_degrees)))
    return float(np.degrees(np.angle(phasor_mean)))


def mean_resultant_length(angle_degrees: np.ndarray) -> float:
    """Concentration of angles on the circle: |mean of exp(i*theta)|, in [0, 1].

    1 means all angles are equal; values near 0 mean the angles are spread around the circle.

    Args:
        angle_degrees: [n] angles in degrees.

    Returns:
        The mean resultant length.
    """
    return float(np.abs(np.mean(np.exp(1j * np.radians(angle_degrees)))))


@dataclass(frozen=True)
class BeamDirectionModel:
    """Far-field direction of every (ring, column) cell of the organized cloud.

    Cell direction: elevation = ``elevation_deg[ring]``,
    azimuth = ``azimuth_offset_deg[ring] + column_step_deg * column``.

    Attributes:
        column_count: Columns per ring (cloud width).
        column_step_deg: Signed azimuth step per column, +-360/column_count.
        elevation_deg: ring id -> far-field elevation (degrees).
        azimuth_offset_deg: ring id -> azimuth of column 0 (degrees, in [-180, 180)).
        elevation_parallax_deg_m: ring id -> fitted elevation slope against 1/range (deg*m);
            diagnostic: the beam-origin offset seen from the sensor origin.
        azimuth_parallax_deg_m: ring id -> fitted azimuth slope against 1/range (deg*m); diagnostic.
        median_residual_deg: Median angle between the fitted and the measured return directions.
        residual_percentile: Percentile reported in ``percentile_residual_deg``.
        percentile_residual_deg: That percentile of the same angles (diagnostic).
    """

    column_count: int
    column_step_deg: float
    elevation_deg: Dict[int, float]
    azimuth_offset_deg: Dict[int, float]
    elevation_parallax_deg_m: Dict[int, float]
    azimuth_parallax_deg_m: Dict[int, float]
    median_residual_deg: float
    residual_percentile: float
    percentile_residual_deg: float

    def to_dict(self) -> Dict[str, Any]:
        """JSON-ready dict (ring keys as strings), the inverse of ``from_dict``.

        Returns:
            Plain dict of the fields.
        """
        def ring_keyed(mapping: Dict[int, float]) -> Dict[str, float]:
            """Ring-keyed mapping with string keys (JSON object keys are strings)."""
            return {str(ring_id): float(value) for ring_id, value in mapping.items()}

        return {
            "column_count": int(self.column_count),
            "column_step_deg": float(self.column_step_deg),
            "elevation_deg": ring_keyed(self.elevation_deg),
            "azimuth_offset_deg": ring_keyed(self.azimuth_offset_deg),
            "elevation_parallax_deg_m": ring_keyed(self.elevation_parallax_deg_m),
            "azimuth_parallax_deg_m": ring_keyed(self.azimuth_parallax_deg_m),
            "median_residual_deg": float(self.median_residual_deg),
            "residual_percentile": float(self.residual_percentile),
            "percentile_residual_deg": float(self.percentile_residual_deg),
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BeamDirectionModel":
        """Rebuild the model from ``to_dict`` output.

        Args:
            payload: Dict as written by ``to_dict`` (e.g. read back from ``derived.json``).

        Returns:
            The model, ring keys restored to int.
        """
        def int_keyed(mapping: Dict[str, float]) -> Dict[int, float]:
            """Ring-keyed mapping with int keys."""
            return {int(ring_id): float(value) for ring_id, value in mapping.items()}

        return cls(
            column_count=int(payload["column_count"]),
            column_step_deg=float(payload["column_step_deg"]),
            elevation_deg=int_keyed(payload["elevation_deg"]),
            azimuth_offset_deg=int_keyed(payload["azimuth_offset_deg"]),
            elevation_parallax_deg_m=int_keyed(payload["elevation_parallax_deg_m"]),
            azimuth_parallax_deg_m=int_keyed(payload["azimuth_parallax_deg_m"]),
            median_residual_deg=float(payload["median_residual_deg"]),
            residual_percentile=float(payload["residual_percentile"]),
            percentile_residual_deg=float(payload["percentile_residual_deg"]),
        )


def fit_line_against_inverse_range(angles_deg: np.ndarray, inverse_range: np.ndarray) -> Tuple[float, float]:
    """Least-squares fit angle = intercept + slope * (1/range).

    Args:
        angles_deg: [n] measured angles (degrees, already free of the +-180 seam).
        inverse_range: [n] 1/range (1/m).

    Returns:
        (intercept_deg, slope_deg_m).
    """
    # Design matrix rows [1, 1/r]; lstsq minimises ||design @ [intercept, slope] - angles||^2.
    design = np.column_stack([np.ones_like(inverse_range), inverse_range])
    solution = np.linalg.lstsq(design, angles_deg, rcond=None)[0]
    return float(solution[0]), float(solution[1])


@transform(
    "T51",
    "Fit the far-field direction of every Ouster (ring, column) cell from raw returns "
    "(per-ring angle = intercept + slope/range; column azimuth step +-360/columns)",
)
def derive_beam_direction_model(
    points_xyz: np.ndarray,
    ring: np.ndarray,
    column: np.ndarray,
    column_count: int,
    expected_rings,
    max_median_residual_deg: float,
    residual_percentile: float,
) -> BeamDirectionModel:
    """Fit cell directions from returns of the RAW (not motion-compensated) cloud.

    In the raw cloud every cell keeps a fixed direction in the sensor frame, but the beam starts
    a few centimetres from the sensor origin. Seen from the origin, a return at range r is off
    the beam direction by about offset/r (parallax), so near returns are biased. Per ring:
        elevation(r) = elevation_inf + k_el / r
        azimuth(r, column) - step * column = azimuth_0 + k_az / r
    The intercepts (1/r -> 0) are the far-field direction, which is what a no-return placeholder
    at 2000 m needs. step = +-360/column_count (Ouster columns are equal encoder steps); its
    sign is the one that concentrates the per-ring azimuth residuals (larger mean resultant
    length). Azimuth residuals are centred on their circular mean before the linear fit, so the
    +-180 seam never enters least squares. Cost O(n) per ring.

    Args:
        points_xyz: [n,3] returned points in the raw sensor frame (metres).
        ring: [n] ring id of each point.
        column: [n] cloud column of each point.
        column_count: Columns per ring (cloud width).
        expected_rings: Ring ids that must be fitted (the elevation-mapping keys).
        max_median_residual_deg: Gate on the median angle between fitted and measured
            directions; a larger value means the cell model does not describe this cloud.
        residual_percentile: Percentile of the same angles stored for diagnostics.

    Returns:
        The fitted ``BeamDirectionModel``.

    Raises:
        ValueError: If a ring has fewer distinct ranges than fit parameters, an expected ring has
            no returns, or the median residual exceeds the gate.
    """
    points_xyz = np.asarray(points_xyz, dtype=np.float64)
    ring = np.asarray(ring, dtype=np.int64)
    column = np.asarray(column, dtype=np.int64)
    # Step 1: spherical angles and inverse range of every return, seen from the sensor origin.
    distances = np.linalg.norm(points_xyz, axis=1)
    elevation = np.degrees(np.arcsin(points_xyz[:, 2] / distances))
    azimuth = np.degrees(np.arctan2(points_xyz[:, 1], points_xyz[:, 0]))
    inverse_range = 1.0 / distances
    # Every expected ring needs returns, else its cells would have no direction.
    ring_ids = np.unique(ring)
    missing_rings = sorted(set(int(ring_id) for ring_id in expected_rings) - set(ring_ids.tolist()))
    if missing_rings:
        raise ValueError(f"no returns to fit beam directions for rings {missing_rings}")
    # Step 2: column step sign. The right sign leaves only a constant per-ring offset (plus small
    # parallax), so its residuals are concentrated; the wrong sign spreads them over the circle.
    step_magnitude = FULL_TURN_DEGREES / int(column_count)

    def concentration(step_sign: float) -> float:
        """Mean over rings of the azimuth-residual concentration for one step sign."""
        residual = wrap_to_half_turn(azimuth - step_sign * step_magnitude * column)
        return float(np.mean([mean_resultant_length(residual[ring == ring_id]) for ring_id in ring_ids]))

    column_step = max(COLUMN_STEP_SIGNS, key=concentration) * step_magnitude
    # Step 3: per-ring line fits against 1/range; residual angles for the gate.
    elevation_far, elevation_slope, azimuth_offset, azimuth_slope = {}, {}, {}, {}
    residual_angles = np.empty(ring.size, dtype=np.float64)
    for ring_id in ring_ids:
        ring_mask = ring == ring_id
        # The line is determined only if [1, 1/r] has full column rank; matrix_rank's SVD tolerance
        # also treats ranges equal up to float round-off as one range.
        ring_inverse_range = inverse_range[ring_mask]
        design_rank = np.linalg.matrix_rank(np.column_stack([np.ones_like(ring_inverse_range), ring_inverse_range]))
        if design_rank < LINE_FIT_PARAMETER_COUNT:
            raise ValueError(f"ring {int(ring_id)}: fewer than {LINE_FIT_PARAMETER_COUNT} distinct ranges")
        # Elevation is never near the seam (|elevation| <= 90), so fit it directly.
        elevation_intercept, elevation_k = fit_line_against_inverse_range(elevation[ring_mask], inverse_range[ring_mask])
        # Azimuth: remove the column ramp, centre on the circular mean, then fit the line.
        azimuth_residual = wrap_to_half_turn(azimuth[ring_mask] - column_step * column[ring_mask])
        azimuth_centre = circular_mean_degrees(azimuth_residual)
        azimuth_intercept, azimuth_k = fit_line_against_inverse_range(
            wrap_to_half_turn(azimuth_residual - azimuth_centre), inverse_range[ring_mask])
        # Store far-field intercepts and parallax slopes; adding the centre back undoes the centring.
        elevation_far[int(ring_id)], elevation_slope[int(ring_id)] = elevation_intercept, elevation_k
        azimuth_offset[int(ring_id)] = float(wrap_to_half_turn(azimuth_centre + azimuth_intercept))
        azimuth_slope[int(ring_id)] = azimuth_k
        # Residual angle per return: elevation error and azimuth error scaled by cos(elevation)
        # (an azimuth error shrinks to cos(el) times its size on the unit sphere); small-angle norm.
        elevation_error = elevation[ring_mask] - (elevation_intercept + elevation_k * inverse_range[ring_mask])
        azimuth_error = wrap_to_half_turn(
            azimuth_residual - (azimuth_offset[int(ring_id)] + azimuth_k * inverse_range[ring_mask]))
        residual_angles[ring_mask] = np.hypot(elevation_error, azimuth_error * np.cos(np.radians(elevation[ring_mask])))
    # Step 4: gate on the median (robust to a minority of odd returns, e.g. multipath).
    median_residual = float(np.median(residual_angles))
    if median_residual > max_median_residual_deg:
        raise ValueError(
            f"beam direction fit: median residual {median_residual:.4g} deg exceeds {max_median_residual_deg:.4g} deg"
        )
    return BeamDirectionModel(
        column_count=int(column_count),
        column_step_deg=float(column_step),
        elevation_deg=elevation_far,
        azimuth_offset_deg=azimuth_offset,
        elevation_parallax_deg_m=elevation_slope,
        azimuth_parallax_deg_m=azimuth_slope,
        median_residual_deg=median_residual,
        residual_percentile=float(residual_percentile),
        percentile_residual_deg=float(np.percentile(residual_angles, residual_percentile)),
    )


@transform("T52", "Unit direction of Ouster (ring, column) cells in the raw sensor frame from the beam model")
def beam_unit_directions(model: BeamDirectionModel, ring: np.ndarray, column: np.ndarray) -> np.ndarray:
    """Far-field unit direction of each cell.

    Math: azimuth a = azimuth_offset[ring] + step * column, elevation e = elevation[ring];
    d = (cos e cos a, cos e sin a, sin e), which has unit norm.

    Args:
        model: Fitted beam model (T51).
        ring: [m] ring ids (each must be in the model).
        column: [m] cloud columns.

    Returns:
        [m,3] float64 unit vectors in the raw sensor frame.

    Raises:
        ValueError: If a ring is not in the model.
    """
    ring = np.asarray(ring, dtype=np.int64)
    column = np.asarray(column, dtype=np.float64)
    # Reject rings the model was not fitted for.
    unknown_rings = sorted(set(np.unique(ring).tolist()) - set(model.elevation_deg))
    if unknown_rings:
        raise ValueError(f"rings {unknown_rings} are not in the beam model")
    # Per-point lookups of the ring parameters (rings are few, so a dict lookup per unique ring).
    elevation_deg = np.zeros(ring.size, dtype=np.float64)
    azimuth_offset_deg = np.zeros(ring.size, dtype=np.float64)
    for ring_id in np.unique(ring):
        ring_mask = ring == ring_id
        elevation_deg[ring_mask] = model.elevation_deg[int(ring_id)]
        azimuth_offset_deg[ring_mask] = model.azimuth_offset_deg[int(ring_id)]
    elevation = np.radians(elevation_deg)
    azimuth = np.radians(azimuth_offset_deg + model.column_step_deg * column)
    # Spherical (azimuth, elevation) -> Cartesian unit vector.
    return np.column_stack([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])


@transform("T53", "Locate Ouster no-return cells (ring, column, time) in an organized PointCloud2")
def no_return_cells(
    fields: Dict[str, np.ndarray], column_count: int, row_count: int, lidar_config: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ring, column and sweep-relative time of every cell with range 0.

    The cloud is row-major: cell k sits in column k mod column_count (checked on the real bags:
    the ring field is constant along each row). The Ouster driver writes the column timestamp
    into ``t`` for no-return cells too, so their time is measured, not interpolated.

    Args:
        fields: Decoded cloud (``decode_cloud_fields``), one entry per cell.
        column_count: Cloud width.
        row_count: Cloud height (rings).
        lidar_config: ``lidar`` config section (field names and units).

    Returns:
        (ring [m] int64, column [m] int64, point_time_seconds [m] float64).

    Raises:
        ValueError: If the cloud is not organized (height 1 or cell count != width*height).
    """
    range_values = np.asarray(fields[lidar_config["range_field"]])
    cell_count = range_values.size
    # An unorganized cloud has no (ring, column) grid, so the column of a cell is unknown.
    if row_count <= 1 or cell_count != int(column_count) * int(row_count):
        raise ValueError(
            f"cloud is not organized: {cell_count} cells, width {column_count}, height {row_count}"
        )
    # Range 0 marks a no-return (Ouster convention, range field in sensor units).
    no_return_mask = range_values == 0
    cell_index = np.flatnonzero(no_return_mask)
    # Ring from the ring field; column from the row-major cell index; time converted to seconds.
    ring =np.asarray(fields[lidar_config["ring_field"]])[no_return_mask].astype(np.int64)
    column = (cell_index % int(column_count)).astype(np.int64)
    point_time_seconds = (
        np.asarray(fields[lidar_config["point_time_field"]])[no_return_mask].astype(np.float64)
        * lidar_config["point_time_unit_seconds"]
    )
    return ring, column, point_time_seconds


@transform(
    "T54",
    "Motion-compensate no-return directions to the sweep reference pose and assemble the "
    "sidecar rows [dir_x, dir_y, dir_z, t, channel, column]",
)
def no_return_sweep_rows(
    directions_sensor: np.ndarray,
    absolute_times: np.ndarray,
    pose_at: Callable[[np.ndarray], np.ndarray],
    reference_time: float,
    channel: np.ndarray,
    column: np.ndarray,
) -> np.ndarray:
    """Rotate each direction into the reference lidar frame and stack the sidecar rows.

    T13 maps a point p measured at t_p to T_ref^-1 T(t_p) p = R p + c. A direction has no
    position, so only R applies: R d = deskew(d) - deskew(0) with both evaluated at t_p.
    R is a rotation, so R d stays unit length.

    Args:
        directions_sensor: [m,3] unit directions in the lidar frame at each cell's capture time.
        absolute_times: [m] capture times (seconds, trajectory clock).
        pose_at: Times [k] -> T_world_lidar [k,4,4] (as for ``deskew_points``).
        reference_time: Sweep reference time (seconds, trajectory clock).
        channel: [m] elevation-mapping keys.
        column: [m] cloud columns.

    Returns:
        float32 [m, NO_RETURN_ROW_WIDTH] rows; t = capture time - reference time.
    """
    directions_sensor = np.asarray(directions_sensor, dtype=np.float64).reshape(-1, 3)
    absolute_times = np.asarray(absolute_times, dtype=np.float64)
    direction_count = directions_sensor.shape[0]
    # No no-return cells: empty sidecar that still has the full row width.
    if direction_count == 0:
        return np.zeros((0, NO_RETURN_ROW_WIDTH), dtype=np.float32)
    # Deskew the direction tips and the origin at the same times; their difference is R d.
    stacked_points = np.vstack([directions_sensor, np.zeros_like(directions_sensor)])
    stacked_times = np.concatenate([absolute_times, absolute_times])
    deskewed_points, relative_time = deskew_points(stacked_points, stacked_times, pose_at, reference_time)
    directions_reference = deskewed_points[:direction_count] - deskewed_points[direction_count:]
    # Assemble rows in the sidecar column order (NO_RETURN_* constants).
    return np.column_stack([
        directions_reference,
        relative_time[:direction_count],
        np.asarray(channel, dtype=np.float64),
        np.asarray(column, dtype=np.float64),
    ]).astype(np.float32)
