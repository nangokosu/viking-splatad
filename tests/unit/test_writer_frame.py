"""Tests for the single-frame writer helper used by the incremental extraction pipeline."""

import numpy as np

from viking_splatad.dataset.writer import (
    FRAME_FILE_PATTERN,
    INDEX_DIGITS,
    write_frame_file,
)


def test_write_frame_file_names_follow_the_reader_pattern(tmp_path):
    """A lidar frame lands at <root>/lidar/<sensor>/<padded index>_<ts>.npy and round-trips."""
    sweep = np.arange(12, dtype=np.float32).reshape(2, 6)
    frame_index, timestamp_ns = 7, 1_714_571_311_141_675_000
    written_path = write_frame_file(tmp_path, "lidar", "ouster", frame_index, timestamp_ns, sweep)
    # Expected name is built here from the documented layout, not from the implementation.
    expected_name = f"{frame_index:0{INDEX_DIGITS}d}_{timestamp_ns}.npy"
    assert written_path == tmp_path / "lidar" / "ouster" / expected_name
    match = FRAME_FILE_PATTERN.match(written_path.name)
    assert match is not None and int(match.group(1)) == frame_index and int(match.group(2)) == timestamp_ns
    assert np.array_equal(np.load(written_path), sweep)


def test_write_frame_file_writes_decodable_jpeg_for_cameras(tmp_path):
    """A camera frame is stored as a JPEG with the same pixel size."""
    import cv2

    image = np.full((8, 12, 3), 128, dtype=np.uint8)
    written_path = write_frame_file(tmp_path, "camera", "cam", 0, 5, image)
    assert written_path.suffix == ".jpg"
    assert cv2.imread(str(written_path)).shape == image.shape
