"""Contract check ``lidar_no_return_sidecar``: a well-formed sidecar passes, each defect fails it alone."""

import numpy as np
import pytest

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.dataset.schema import NoReturnStream
from viking_splatad.dataset.writer import write_dataset
from viking_splatad.extract.no_return import NO_RETURN_ROW_WIDTH

CHECK_NAME = "lidar_no_return_sidecar"
LIDAR_NAME = "lidar_top"


def dataset_with_valid_sidecar(seed: int = 3):
    """Synthetic dataset plus one valid no-return file per lidar sweep.

    Rows respect the dataset's lidar model: unit directions, |t| < sweep period, channels from the
    elevation mapping, integer columns below 2*pi / azimuth resolution.
    """
    dataset = build_synthetic_dataset()
    rng = np.random.default_rng(seed)
    lidar = dataset.lidar
    columns_per_ring = int(np.rint(2.0 * np.pi / lidar.azimuth_resolution))
    channels = np.array(sorted(lidar.elevation_mapping))
    stream = dataset.streams[LIDAR_NAME]
    frames = []
    for _ in stream.frames:
        row_count = int(rng.integers(1, 6))
        directions = rng.normal(size=(row_count, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        frames.append(np.column_stack([
            directions,
            rng.uniform(-lidar.sweep_period / 2, lidar.sweep_period / 2, row_count),
            rng.choice(channels, row_count),
            rng.integers(0, columns_per_ring, row_count),
        ]).astype(np.float32))
    dataset.no_return_streams = {LIDAR_NAME: NoReturnStream(
        stream.frame_indices.copy(), stream.frame_timestamps_ns.copy(), frames)}
    return dataset


def drop_last_file(dataset):
    """One sweep loses its no-return file."""
    no_return_stream = dataset.no_return_streams[LIDAR_NAME]
    dataset.no_return_streams[LIDAR_NAME] = NoReturnStream(
        no_return_stream.frame_indices[:-1], no_return_stream.frame_timestamps_ns[:-1], no_return_stream.frames[:-1])


def stretch_direction(dataset):
    """A direction 10% longer than unit."""
    dataset.no_return_streams[LIDAR_NAME].frames[0][0, :3] *= np.float32(1.1)


def late_time(dataset):
    """A no-return twice the sweep period after the reference."""
    dataset.no_return_streams[LIDAR_NAME].frames[0][0, 3] = np.float32(2.0 * dataset.lidar.sweep_period)


def unknown_channel(dataset):
    """A channel one past the largest mapping key."""
    dataset.no_return_streams[LIDAR_NAME].frames[0][0, 4] = np.float32(max(dataset.lidar.elevation_mapping) + 1)


def column_past_the_turn(dataset):
    """A column equal to the columns per ring (valid ones are 0..columns-1)."""
    columns_per_ring = int(np.rint(2.0 * np.pi / dataset.lidar.azimuth_resolution))
    dataset.no_return_streams[LIDAR_NAME].frames[0][0, 5] = np.float32(columns_per_ring)


def wrong_width(dataset):
    """Rows with one column too few."""
    dataset.no_return_streams[LIDAR_NAME].frames[0] = dataset.no_return_streams[LIDAR_NAME].frames[0][:, :-1].copy()


def test_valid_sidecar_passes_every_check(tmp_path):
    """A complete, well-formed sidecar keeps the whole contract green."""
    write_dataset(tmp_path, dataset_with_valid_sidecar())
    report = validate_dataset(tmp_path)
    assert report.passed, report.to_markdown()
    assert CHECK_NAME in report.to_markdown()


@pytest.mark.parametrize("corruption", [
    drop_last_file, stretch_direction, late_time, unknown_channel, column_past_the_turn, wrong_width,
])
def test_each_sidecar_defect_fails_only_the_sidecar_check(tmp_path, corruption):
    """Every single defect is caught, and by this check alone."""
    dataset = dataset_with_valid_sidecar()
    corruption(dataset)
    write_dataset(tmp_path, dataset)
    report = validate_dataset(tmp_path)
    assert report.failed_check_names() == [CHECK_NAME], report.to_markdown()


def test_sidecar_rows_have_the_documented_width():
    """The fixture itself follows the documented 6-column layout."""
    frames = dataset_with_valid_sidecar().no_return_streams[LIDAR_NAME].frames
    assert all(frame.shape[1] == NO_RETURN_ROW_WIDTH for frame in frames)
