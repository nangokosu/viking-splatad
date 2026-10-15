"""ROS timestamp handling (T2): exact ns -> seconds, dedupe, sort."""

from typing import NamedTuple, Optional

import numpy as np

from viking_splatad.registry import transform

# Definition of the SI prefix, not a tunable value.
NANOSECONDS_PER_SECOND = 1_000_000_000


class StampConversion(NamedTuple):
    """Result of converting ROS stamps.

    Attributes:
        seconds: float64 strictly increasing seconds relative to ``reference_ns``.
        kept_indices: Indices into the input arrays of the kept stamps (in output order).
        reference_ns: Integer ns offset subtracted; absolute ns = reference_ns + seconds*1e9.
    """

    seconds: np.ndarray
    kept_indices: np.ndarray
    reference_ns: int


@transform("T2", "ROS (sec,nanosec) -> exact float64 seconds, deduped and strictly increasing")
def ros_stamps_to_increasing_seconds(
    stamp_seconds: np.ndarray,
    stamp_nanoseconds: np.ndarray,
    reference_ns: Optional[int] = None,
) -> StampConversion:
    """Convert ROS stamps to strictly increasing float64 seconds.

    Precision: an absolute epoch time (~1.7e9 s) in float64 has ~2.4e-7 s spacing, coarser
    than 1 ns. So the sum sec*1e9+nanosec is done in int64, a reference is subtracted
    in integers, and only the small difference (<< 2^53 ns) is converted to float.

    Args:
        stamp_seconds: Integer ``sec`` fields.
        stamp_nanoseconds: Integer ``nanosec`` fields.
        reference_ns: Integer ns to subtract; defaults to the earliest stamp.
            Pass 0 for absolute seconds.

    Returns:
        StampConversion with sorted, de-duplicated seconds, kept indices, and reference.
    """
    # Exact integer total nanoseconds (int64 holds ~9.2e18 ns = ~292 years).
    total_ns = (
        np.asarray(stamp_seconds, dtype=np.int64) * NANOSECONDS_PER_SECOND
        + np.asarray(stamp_nanoseconds, dtype=np.int64)
    )
    # Default reference: earliest stamp, so relative values start at 0.
    if reference_ns is None:
        reference_ns = int(total_ns.min())
    # np.unique sorts ascending and returns the first occurrence of each duplicate.
    unique_ns, kept_indices = np.unique(total_ns, return_index=True)
    # Integer subtraction first, float conversion last.
    relative_ns = unique_ns - np.int64(reference_ns)
    seconds = relative_ns.astype(np.float64) / NANOSECONDS_PER_SECOND
    return StampConversion(seconds, kept_indices, int(reference_ns))
