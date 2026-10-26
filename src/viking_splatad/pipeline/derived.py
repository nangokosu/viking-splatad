"""Globally derived pipeline parameters, computed once and cached in ``derived.json``.

Parameters such as the intensity scale (T14), ego box (T15), elevation mapping (T16) and
azimuth tiling (T17) must be identical for every frame of a session. They are derived from
the first processed chunk and then reloaded for every later chunk, so frames added by a
later run stay consistent with the earlier ones.
"""

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from viking_splatad.contract.lidar_model import (
    derive_azimuth_tiling,
    derive_elevation_mapping,
    derive_min_lidar_dist,
)
from viking_splatad.extract.lidar import normalise_intensity
from viking_splatad.extract.timebase import NANOSECONDS_PER_SECOND
from viking_splatad.registry import transform

# File name of the cache inside the dataset output directory.
DERIVED_FILE_NAME = "derived.json"
# Bump when the meaning of a cached field changes so stale caches are rejected, not reused.
DERIVED_FORMAT_VERSION = 1
# extrinsics.txt prints values with 3 decimals (see calibration/extrinsics/extrinsics.txt).
CALIBRATION_PRINTED_DECIMALS = 3
# Quaternion components are 4 numbers, each rounded by at most half a unit of the last decimal.
QUATERNION_COMPONENT_COUNT = 4
# Fraction of the nearest returns searched for the robot-body cluster (T15). T15 looks for the
# first EMPTY radial-histogram bin, which never occurs when sparse returns trail off from the
# body to the scene (real May data: 10% of returns lie at 0.21-0.4 m, then a sparse tail up to
# ~35 m with no empty bin). Restricting the search to the nearest quarter leaves headroom over
# that 10% while keeping bins fine enough to expose the gap. Overridable via lidar.ego_search_fraction.
DEFAULT_EGO_SEARCH_FRACTION = 0.25
# Space dimensions of a translation vector.
TRANSLATION_COMPONENT_COUNT = 3


@dataclass
class LidarScanSample:
    """One decoded lidar sweep restricted to valid (range > 0) points, in the sensor frame.

    Attributes:
        points_xyz: [n,3] float64 metres.
        ring: [n] integer beam ids.
        intensity: [n] raw (unnormalised) intensity values.
        point_time_seconds: [n] seconds since sweep start.
        column_count: Azimuth columns per ring (the cloud ``width``).
        stamp_ns: Absolute header stamp of the sweep in nanoseconds.
    """

    points_xyz: np.ndarray
    ring: np.ndarray
    intensity: np.ndarray
    point_time_seconds: np.ndarray
    column_count: int
    stamp_ns: int


@dataclass
class DerivedParameters:
    """Everything derived once per session (serialised to ``derived.json``)."""

    session_name: str
    reference_ns: int
    source_chunk: str
    scan_count: int
    intensity_scale: float
    intensity_percentile: float
    min_lidar_dist: List[float]
    min_lidar_dist_fallback: bool
    elevation_mapping_deg: Dict[str, float]
    azimuth_resolution_deg: float
    azimuth_tile_width: int
    columns_per_ring: float
    sweep_period_seconds: float
    camera_source: Dict[str, Any]
    camera_undistorted: Dict[str, Any]
    resize_factor: float
    bag_extrinsics: Dict[str, List[List[float]]] = field(default_factory=dict)
    extrinsic_check: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # T51 beam model (``BeamDirectionModel.to_dict``). None in caches written before Phase B;
    # run_pipeline fits and saves it then, so adding it needs no format-version bump.
    beam_direction_model: Optional[Dict[str, Any]] = None
    format_version: int = DERIVED_FORMAT_VERSION


def save_derived(path: Path, derived: DerivedParameters) -> None:
    """Write the derived parameters as JSON.

    Args:
        path: Target file (parent directory is created).
        derived: Parameters to persist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(derived), indent=2))


def load_derived(path: Path, session_name: str) -> Optional[DerivedParameters]:
    """Load cached derived parameters, or None if absent.

    Args:
        path: ``derived.json`` path.
        session_name: Session this cache must belong to (May and June never mix).

    Returns:
        The parameters, or None when the file does not exist.

    Raises:
        ValueError: If the cache belongs to another session or has another format version.
    """
    path = Path(path)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    # Refuse foreign caches instead of silently producing inconsistent frames.
    if payload.get("session_name") != session_name:
        raise ValueError(
            f"{path} belongs to session {payload.get('session_name')!r}, not {session_name!r}"
        )
    if payload.get("format_version") != DERIVED_FORMAT_VERSION:
        raise ValueError(f"{path}: unsupported derived format version {payload.get('format_version')}")
    return DerivedParameters(**payload)


def calibration_tolerance() -> Tuple[float, float]:
    """Tolerances for comparing the calibration file with the bag's /tf.

    The calibration text prints values with ``CALIBRATION_PRINTED_DECIMALS`` decimals, so
    each printed component is off by at most half a unit in the last place, e = 0.5*10^-d.
    Translation: all 3 components off by e gives |dt| <= sqrt(3)*e.
    Rotation: all 4 quaternion components off by e gives |dq| <= sqrt(4)*e; for unit
    quaternions the rotation angle error is ~2*|dq|.

    Returns:
        (translation_tolerance_m, rotation_tolerance_rad).
    """
    half_unit_in_last_place = 0.5 * 10.0 ** (-CALIBRATION_PRINTED_DECIMALS)
    translation_tolerance = math.sqrt(TRANSLATION_COMPONENT_COUNT) * half_unit_in_last_place
    rotation_tolerance = 2.0 * math.sqrt(QUATERNION_COMPONENT_COUNT) * half_unit_in_last_place
    return translation_tolerance, rotation_tolerance


@transform("T49", "Sweep period that bounds the per-point time of every extracted sweep")
def covering_sweep_period(derived_period_seconds: float, longest_point_time_seconds: float) -> float:
    """Sweep period stored in the manifest: the derived period, raised to any longer point time.

    ``derived_period_seconds`` comes from a small sample of the first chunk, but real sweep
    lengths jitter (Ouster: ~0.1 ms), so a later sweep can end after it. SplatAD requires
    |t| <= sweep period for every point, hence the bound is max(derived, longest observed |t|)
    over ALL extracted sweeps. With reference = sweep start, no extracted value depends on
    this bound, so raising it changes only the manifest/contract value.

    Args:
        derived_period_seconds: ``sweep_period_seconds`` from ``derived.json``.
        longest_point_time_seconds: Largest |t| over every sweep written to the dataset.

    Returns:
        max(derived_period_seconds, longest_point_time_seconds).
    """
    return max(float(derived_period_seconds), float(longest_point_time_seconds))


def derive_lidar_parameters(
    samples: Sequence[LidarScanSample],
    intensity_percentile: float,
    ego_margin_m: float,
    ego_search_fraction: float = DEFAULT_EGO_SEARCH_FRACTION,
) -> Dict[str, Any]:
    """Derive intensity scale, ego box, elevation map, azimuth tiling and sweep period.

    Args:
        samples: Valid-point sweeps pooled for the derivation (T14-T17 inputs).
        intensity_percentile: Percentile for the intensity scale (T14).
        ego_margin_m: Margin added to the robot-body cluster extents (T15).
        ego_search_fraction: Nearest fraction of returns in which the body cluster is searched.

    Returns:
        Dict with the lidar-related fields of ``DerivedParameters``.

    Raises:
        ValueError: If no sweep is given.
    """
    if len(samples) == 0:
        raise ValueError("Need at least one lidar sweep to derive lidar parameters")
    # Step 1: pool points across sweeps (more rings/ego points seen than in a single sweep).
    pooled_xyz = np.concatenate([sample.points_xyz for sample in samples], axis=0)
    pooled_ring = np.concatenate([sample.ring for sample in samples], axis=0)
    # Step 2 (T14): global percentile scale of the raw intensities.
    intensity_result = normalise_intensity(
        [sample.intensity for sample in samples], percentile=intensity_percentile
    )
    # Step 3 (T15): ego box from the near-field cluster, searched among the nearest returns only.
    # Sanity: the box must not delete more points than the search set itself; otherwise (or if no
    # radial gap exists) keep all points (zero half-extents remove nothing) and flag the fallback.
    pooled_range = np.linalg.norm(pooled_xyz, axis=1)
    search_radius = float(np.quantile(pooled_range, ego_search_fraction))
    try:
        min_lidar_dist = list(derive_min_lidar_dist(pooled_xyz[pooled_range <= search_radius], ego_margin_m))
        inside_box = np.all(np.abs(pooled_xyz) < np.asarray(min_lidar_dist), axis=1)
        min_lidar_dist_fallback = bool(inside_box.mean() > ego_search_fraction)
    except ValueError:
        min_lidar_dist_fallback = True
    if min_lidar_dist_fallback:
        min_lidar_dist = [0.0, 0.0, 0.0]
    # Step 4 (T16): per-ring median elevation (degrees), keys = ring ids.
    elevation_mapping_deg = derive_elevation_mapping(pooled_xyz, pooled_ring)
    # Step 5 (T17): azimuth resolution and tile width from the measured columns per ring.
    columns_per_ring = float(np.median([sample.column_count for sample in samples]))
    azimuth_resolution_deg, azimuth_tile_width = derive_azimuth_tiling(columns_per_ring)
    # Step 6: sweep period = median spacing of sweep stamps (robust to a dropped sweep), but
    # never shorter than the longest per-point time in this sample. Later sweeps can be longer
    # (sweep-length jitter), so the manifest raises it to cover every written sweep (T49).
    longest_point_time = max(float(sample.point_time_seconds.max()) for sample in samples)
    stamp_spacings_ns = np.diff(np.array([sample.stamp_ns for sample in samples], dtype=np.int64))
    stamp_spacings_ns = stamp_spacings_ns[stamp_spacings_ns > 0]
    median_spacing = (
        float(np.median(stamp_spacings_ns)) / NANOSECONDS_PER_SECOND if stamp_spacings_ns.size else 0.0
    )
    return {
        "intensity_scale": float(intensity_result.scale),
        "intensity_percentile": float(intensity_percentile),
        "min_lidar_dist": min_lidar_dist,
        "min_lidar_dist_fallback": min_lidar_dist_fallback,
        "elevation_mapping_deg": {
            str(channel): float(angle) for channel, angle in elevation_mapping_deg.items()
        },
        "azimuth_resolution_deg": float(azimuth_resolution_deg),
        "azimuth_tile_width": int(azimuth_tile_width),
        "columns_per_ring": columns_per_ring,
        "sweep_period_seconds": max(median_spacing, longest_point_time),
    }
