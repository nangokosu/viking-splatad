"""Contract validator tests: a valid dataset passes; each corruption fails exactly one check."""

import copy

import numpy as np
import pytest

from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.contract.splatad_contract import ContractReport, validate_dataset
from viking_splatad.dataset.writer import write_dataset


def corrupt_pose_nan(dataset):
    """Inject a NaN into one camera pose."""
    dataset.streams["cam_front"].poses_world_sensor[2, 0, 3] = np.nan


def corrupt_float_width(dataset):
    """Declare a non-integer image width."""
    dataset.cameras["cam_front"].width = 32.5


def corrupt_width_mismatch(dataset):
    """Declare a width that differs from the stored JPEGs."""
    dataset.cameras["cam_front"].width = 40


def corrupt_k4(dataset):
    """Non-zero k4 distortion (index 3 in [k1,k2,k3,k4,p1,p2])."""
    dataset.cameras["cam_front"].distortion[3] = 0.1


def corrupt_times(dataset):
    """Repeat a timestamp (not strictly increasing)."""
    dataset.streams["lidar_top"].times[3] = dataset.streams["lidar_top"].times[2]


def corrupt_index_gap(dataset):
    """Skip a frame index."""
    dataset.streams["cam_front"].frame_indices[4:] += 1


def corrupt_lidar_dtype(dataset):
    """Store one sweep as float64."""
    stream = dataset.streams["lidar_top"]
    stream.frames[1] = stream.frames[1].astype(np.float64)


def corrupt_intensity(dataset):
    """Intensity above 1."""
    dataset.streams["lidar_top"].frames[0][0, 3] = 1.5


def corrupt_point_time(dataset):
    """Per-point time far outside +/- sweep period."""
    dataset.streams["lidar_top"].frames[0][0, 4] = 10.0 * dataset.lidar.sweep_period


def corrupt_channel(dataset):
    """Channel id absent from the elevation mapping."""
    dataset.streams["lidar_top"].frames[0][0, 5] = max(dataset.lidar.elevation_mapping) + 100


def corrupt_elevation_length(dataset):
    """Add an unused channel so the mapping length is not a multiple of 8."""
    dataset.lidar.elevation_mapping[max(dataset.lidar.elevation_mapping) + 1] = 0.0


def corrupt_azimuth(dataset):
    """Resolution whose bin count 2*pi/res is not an integer."""
    dataset.lidar.azimuth_resolution = 0.0175


def corrupt_split(dataset):
    """train_fraction 1.0 leaves the eval split empty."""
    dataset.session.train_fraction = 1.0


def corrupt_world_up(dataset):
    """y-up world."""
    dataset.session.world_up = np.array([0.0, 1.0, 0.0])


def corrupt_single_pose(dataset):
    """Fewer than two poses (times kept consistent)."""
    stream = dataset.streams["cam_front"]
    stream.poses_world_sensor = stream.poses_world_sensor[:1]
    stream.times = stream.times[:1]


# check name -> corruption function; each must fail exactly that one check.
CORRUPTIONS = {
    "poses_finite_and_count": corrupt_pose_nan,
    "intrinsics_integer_size": corrupt_float_width,
    "image_shape_matches": corrupt_width_mismatch,
    "distortion_k4_zero": corrupt_k4,
    "times_strictly_increasing": corrupt_times,
    "sensor_index_contiguity": corrupt_index_gap,
    "lidar_array_format": corrupt_lidar_dtype,
    "lidar_intensity_range": corrupt_intensity,
    "lidar_point_time_range": corrupt_point_time,
    "lidar_channel_in_mapping": corrupt_channel,
    "elevation_mapping_multiple_of_8": corrupt_elevation_length,
    "azimuth_tile_integral": corrupt_azimuth,
    "splits_non_empty": corrupt_split,
    "world_up_is_z": corrupt_world_up,
}


def validate_corrupted(tmp_path, corruption):
    """Build a valid dataset, apply ``corruption`` in memory, write, and validate."""
    dataset = build_synthetic_dataset()
    corruption(dataset)
    write_dataset(tmp_path, dataset)
    return validate_dataset(tmp_path)


def test_valid_dataset_passes_every_check(tmp_path):
    """The synthetic dataset built with the writer passes all checks."""
    write_dataset(tmp_path, build_synthetic_dataset())
    report = validate_dataset(tmp_path)
    assert isinstance(report, ContractReport)
    assert report.passed, report.to_markdown()
    # Every corruption target must be a real, executed check.
    assert set(CORRUPTIONS) <= {result.name for result in report.results}


@pytest.mark.parametrize("check_name", sorted(CORRUPTIONS))
def test_minimal_corruption_fails_exactly_that_check(tmp_path, check_name):
    """Each corruption flips exactly the named check to failed."""
    report = validate_corrupted(tmp_path, CORRUPTIONS[check_name])
    assert report.failed_check_names() == [check_name], report.to_markdown()


def test_single_pose_fails_pose_check_only(tmp_path):
    """The >=2 poses requirement is part of the pose check."""
    report = validate_corrupted(tmp_path, corrupt_single_pose)
    assert report.failed_check_names() == ["poses_finite_and_count"]


def test_markdown_report_lists_every_check(tmp_path):
    """to_markdown shows each check name with a PASS/FAIL marker."""
    report = validate_corrupted(tmp_path, corrupt_k4)
    markdown = report.to_markdown()
    for result in report.results:
        assert result.name in markdown
    assert "FAIL" in markdown and "PASS" in markdown


def test_schema_version_mismatch_reported_as_failed_check(tmp_path):
    """An unreadable/foreign dataset yields a failing report instead of an exception."""
    import json

    write_dataset(tmp_path, build_synthetic_dataset(num_frames=3))
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["schema_version"] += 1
    manifest_path.write_text(json.dumps(manifest))
    assert validate_dataset(tmp_path).failed_check_names() == ["dataset_loadable"]
