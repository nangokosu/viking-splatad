"""rosbag2 metadata.yaml parsing: choose which bag chunk files cover a time window."""

import re
from typing import List, Optional

import yaml

from viking_splatad.registry import transform

NANOSECONDS_PER_SECOND = 10**9  # SI definition of the nanosecond


def chunk_index(file_name: str) -> int:
    """Extract the numeric chunk index from a chunk file name.

    Args:
        file_name: e.g. "short_grass__ros2_12.mcap".

    Returns:
        The integer before the file extension (12).

    Raises:
        ValueError: If no trailing index is present.
    """
    # Trailing digits right before the final extension.
    match = re.search(r"_(\d+)\.[^.]+$", file_name)
    if match is None:
        raise ValueError(f"No chunk index in {file_name!r}")
    return int(match.group(1))


@transform("T1", "Select rosbag2 chunk files whose time span intersects the requested window")
def select_bag_chunks(
    metadata_yaml_text: str,
    start_offset_s: Optional[float],
    end_offset_s: Optional[float],
) -> List[str]:
    """Return chunk file names intersecting [bag_start+start, bag_start+end].

    Chunk k spans [s_k, s_k + d_k] (closed). It intersects the window [a, b] iff
    s_k <= b and s_k + d_k >= a. A None bound is open (-inf for start, +inf for end).
    Integer nanoseconds are used for the comparison to avoid float rounding on
    epoch-scale values (~1.7e18 ns exceeds float64's exact-integer range).

    Args:
        metadata_yaml_text: Text of the bag's metadata.yaml.
        start_offset_s: Window start, seconds after the bag start (None = open).
        end_offset_s: Window end, seconds after the bag start (None = open).

    Returns:
        Chunk file names in the metadata's relative_file_paths order.
    """
    # Step 1: parse metadata and fetch per-file records.
    bag_information = yaml.safe_load(metadata_yaml_text)["rosbag2_bagfile_information"]
    file_records = {record["path"]: record for record in bag_information["files"]}
    # Step 2: bag start = earliest chunk start (equals the top-level starting_time).
    bag_start_ns = min(
        int(record["starting_time"]["nanoseconds_since_epoch"]) for record in file_records.values()
    )
    # Step 3: window bounds in absolute integer nanoseconds (None stays open).
    window_start_ns = (
        None if start_offset_s is None
        else bag_start_ns + round(start_offset_s * NANOSECONDS_PER_SECOND)
    )
    window_end_ns = (
        None if end_offset_s is None
        else bag_start_ns + round(end_offset_s * NANOSECONDS_PER_SECOND)
    )
    # Step 4: keep chunks passing the interval-intersection test, in listed order.
    selected_names: List[str] = []
    for file_name in bag_information.get("relative_file_paths", list(file_records)):
        record = file_records[file_name]
        chunk_start_ns = int(record["starting_time"]["nanoseconds_since_epoch"])
        chunk_end_ns = chunk_start_ns + int(record["duration"]["nanoseconds"])
        starts_before_window_end = window_end_ns is None or chunk_start_ns <= window_end_ns
        ends_after_window_start = window_start_ns is None or chunk_end_ns >= window_start_ns
        if starts_before_window_end and ends_after_window_start:
            selected_names.append(file_name)
    return selected_names
