"""Unit tests for extract/timebase.py (T2)."""

import numpy as np
import pytest

from viking_splatad.extract.timebase import ros_stamps_to_increasing_seconds

# Realistic epoch-scale seconds (justification: large enough that naive float math loses ns).
EPOCH_SECONDS = 1_700_000_000
NANOSECONDS_PER_SECOND = 1_000_000_000


@pytest.mark.covers("T2")
def test_duplicates_and_unsorted_become_strictly_increasing():
    """Unsorted + duplicated stamps -> strictly increasing output, first duplicates kept."""
    # Stamps (sec, nanosec) in arrival order; index 3 duplicates index 0, index 2 is earliest.
    seconds_part = np.array([EPOCH_SECONDS + 5, EPOCH_SECONDS + 6, EPOCH_SECONDS + 4, EPOCH_SECONDS + 5])
    nanosec_part = np.array([500, 0, 999_999_999, 500])
    result = ros_stamps_to_increasing_seconds(seconds_part, nanosec_part)
    # Expected: sorted unique order is original indices [2, 0, 1].
    assert result.kept_indices.tolist() == [2, 0, 1]
    assert np.all(np.diff(result.seconds) > 0)
    assert len(result.seconds) == 3


@pytest.mark.covers("T2")
def test_nanosecond_conversion_is_exact():
    """A 1 ns step at epoch scale survives, which naive float64 sec+ns*1e-9 would blur."""
    seconds_part = np.array([EPOCH_SECONDS, EPOCH_SECONDS])
    nanosec_part = np.array([1, 2])
    result = ros_stamps_to_increasing_seconds(seconds_part, nanosec_part)
    # Relative to the reference (first stamp) the second stamp is exactly 1 ns later.
    assert result.seconds[0] == 0.0
    assert result.seconds[1] == 1 / NANOSECONDS_PER_SECOND
    # Absolute time is recoverable from the returned reference.
    absolute_ns = EPOCH_SECONDS * NANOSECONDS_PER_SECOND + 1
    assert result.reference_ns == absolute_ns


@pytest.mark.covers("T2")
def test_explicit_zero_reference_gives_absolute_seconds():
    """reference_ns=0 yields plain absolute seconds."""
    result = ros_stamps_to_increasing_seconds(np.array([3]), np.array([500_000_000]), reference_ns=0)
    assert result.seconds[0] == 3.5
