"""Unit tests for io/bag_metadata.py (T1 chunk selection, chunk_index helper)."""

import pytest
import yaml

from viking_splatad.io.bag_metadata import chunk_index, select_bag_chunks

NANOSECONDS_PER_SECOND = 10**9
# Synthetic bag layout: three back-to-back chunks of 100 s, 50 s, 25 s.
BAG_START_NANOSECONDS = 1_700_000_000 * NANOSECONDS_PER_SECOND
CHUNK_DURATIONS_SECONDS = [100, 50, 25]


def build_metadata_text():
    """Build rosbag2 metadata YAML text for contiguous chunks.

    Returns:
        (yaml_text, chunk_windows) where chunk_windows lists (name, start_s, end_s)
        offsets relative to the bag start, computed here from the durations.
    """
    file_entries, chunk_windows = [], []
    running_offset_seconds = 0
    for chunk_number, duration_seconds in enumerate(CHUNK_DURATIONS_SECONDS):
        name = f"demo_{chunk_number}.mcap"
        file_entries.append(
            {
                "path": name,
                "starting_time": {
                    "nanoseconds_since_epoch": BAG_START_NANOSECONDS
                    + running_offset_seconds * NANOSECONDS_PER_SECOND
                },
                "duration": {"nanoseconds": duration_seconds * NANOSECONDS_PER_SECOND},
                "message_count": 1,
            }
        )
        chunk_windows.append(
            (name, running_offset_seconds, running_offset_seconds + duration_seconds)
        )
        running_offset_seconds += duration_seconds
    metadata = {
        "rosbag2_bagfile_information": {
            "relative_file_paths": [name for name, _, _ in chunk_windows],
            "files": file_entries,
        }
    }
    return yaml.safe_dump(metadata), chunk_windows


def expected_selection(chunk_windows, start_offset_s, end_offset_s):
    """Brute-force reference: chunks whose closed interval intersects the window."""
    low = float("-inf") if start_offset_s is None else start_offset_s
    high = float("inf") if end_offset_s is None else end_offset_s
    return [name for name, start, end in chunk_windows if start <= high and end >= low]


@pytest.mark.covers("T1")
@pytest.mark.parametrize(
    "start_offset_s,end_offset_s",
    [
        (None, None),      # open window: everything
        (10.0, 20.0),      # strictly inside chunk 0
        (90.0, 120.0),     # spans chunk 0 and 1
        (120.0, 160.0),    # inside chunk 1 and 2
        (160.0, None),     # open end
        (None, 5.0),       # open start
        (500.0, 600.0),    # past the bag: nothing
    ],
)
def test_select_bag_chunks_matches_reference(start_offset_s, end_offset_s):
    """Selected chunks equal the brute-force intersecting set (none extra, none missing)."""
    metadata_text, chunk_windows = build_metadata_text()
    assert select_bag_chunks(metadata_text, start_offset_s, end_offset_s) == expected_selection(
        chunk_windows, start_offset_s, end_offset_s
    )


@pytest.mark.covers("T1")
def test_select_bag_chunks_boundaries_and_order():
    """A window ending exactly at a chunk boundary touches both neighbours; order is kept."""
    metadata_text, chunk_windows = build_metadata_text()
    boundary_offset_seconds = chunk_windows[0][2]
    selected = select_bag_chunks(
        metadata_text, boundary_offset_seconds, boundary_offset_seconds
    )
    assert selected == [chunk_windows[0][0], chunk_windows[1][0]]
    everything = select_bag_chunks(metadata_text, None, None)
    assert everything == [name for name, _, _ in chunk_windows]


@pytest.mark.covers("T1")
def test_select_bag_chunks_on_real_metadata_shape():
    """Window inside chunk 0 of the real May bag (first chunk covers ~99.8 s) selects only it."""
    import pathlib

    real_path = (
        pathlib.Path(__file__).resolve().parents[2]
        / "data/raw/may_short_grass/extracted/ros2_jazzy/data/2024_05_short_grass_run"
        / "bag/short_grass__ros2/metadata.yaml"
    )
    if not real_path.exists():
        pytest.skip("real metadata not present")
    files = yaml.safe_load(real_path.read_text())["rosbag2_bagfile_information"]["files"]
    first_duration_seconds = files[0]["duration"]["nanoseconds"] / NANOSECONDS_PER_SECOND
    selected = select_bag_chunks(real_path.read_text(), 0.0, first_duration_seconds / 2)
    assert selected == [files[0]["path"]]


@pytest.mark.parametrize(
    "file_name,expected_index",
    [("short_grass__ros2_12.mcap", 12), ("tall_grass__ros2_0.mcap", 0), ("a_b_101.mcap", 101)],
)
def test_chunk_index(file_name, expected_index):
    """Trailing integer before the extension is the chunk index."""
    assert chunk_index(file_name) == expected_index
