"""Contract gaps the audit suspected: each test states what the contract SHOULD do and fails today if it does not.

These tests are deliberately left failing until the contract is fixed (no skip, no xfail). Each one
asserts the documented intent: a bad dataset yields a failed check in the report, never a pass and
never an exception, because ``validate_dataset`` is the gate before data goes to the GPU box.
"""

import numpy as np
import pytest

from tests.contract.test_contract_negative_cases import (
    LIDAR_NAME, failed_names, lidar_frame, write_valid,
)
from tests.unit.test_writer import build_synthetic_dataset
from viking_splatad.contract import splatad_contract
from viking_splatad.contract.splatad_contract import validate_dataset
from viking_splatad.dataset.writer import write_dataset


def run_all_checks(dataset):
    """Run every check in memory and return the CheckResult list (exceptions propagate)."""
    return [check(dataset) for check in splatad_contract.CONTRACT_CHECKS]


def test_nan_per_point_time_fails_point_time_check():
    """A NaN per-point time is not a valid offset; the range check must fail, not pass.

    np.abs(nan).max() is nan and ``nan > period`` is False, so NaN slips through today.
    """
    dataset = build_synthetic_dataset()
    lidar_frame(dataset)[0, splatad_contract.TIME_COLUMN] = np.nan
    assert "lidar_point_time_range" in failed_names(dataset)


def test_empty_sweep_is_a_reported_failure_not_an_exception():
    """A sweep with zero points must be REPORTED as a failed check; min()/max() of empty arrays raise today."""
    dataset = build_synthetic_dataset()
    dataset.streams[LIDAR_NAME].frames[0] = np.zeros((0, 6), dtype=np.float32)
    # Must not raise, and the dataset must not be declared valid.
    assert failed_names(dataset) != []


def test_empty_sweep_through_validate_dataset_is_reported(tmp_path):
    """Same through the public entry point: an empty sweep yields a failing report, never an exception."""
    dataset = build_synthetic_dataset()
    dataset.streams[LIDAR_NAME].frames[0] = np.zeros((0, 6), dtype=np.float32)
    write_dataset(tmp_path, dataset)
    assert not validate_dataset(tmp_path).passed


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("axis_column", [0, 1, 2])
def test_non_finite_lidar_xyz_fails_some_check(bad_value, axis_column):
    """A non-finite lidar x/y/z coordinate must fail the contract (no check looks at xyz today)."""
    dataset = build_synthetic_dataset()
    lidar_frame(dataset)[0, axis_column] = bad_value
    assert failed_names(dataset) != []


def scale_rotation(dataset):
    """Pose fault: lidar rotation block scaled by 2 (not orthonormal)."""
    dataset.streams[LIDAR_NAME].poses_world_sensor[2, :3, :3] *= 2.0


def reflect_rotation(dataset):
    """Pose fault: lidar rotation is a reflection (det = -1)."""
    dataset.streams[LIDAR_NAME].poses_world_sensor[2, :3, :3] = np.diag([1.0, 1.0, -1.0])


def shear_rotation(dataset):
    """Pose fault: lidar rotation block sheared (columns not orthogonal)."""
    dataset.streams[LIDAR_NAME].poses_world_sensor[2, 0, 1] = 0.5


@pytest.mark.parametrize("fault", [scale_rotation, reflect_rotation, shear_rotation])
def test_non_rigid_pose_rotation_fails_poses_check(fault):
    """A rotation block that is not in SO(3) is not a rigid pose; poses_finite_and_count must fail it.

    The check only tests the last row and finiteness today.
    """
    dataset = build_synthetic_dataset()
    fault(dataset)
    assert "poses_finite_and_count" in failed_names(dataset)


def test_sweep_with_too_few_columns_is_reported_not_raised():
    """A [n,4] sweep fails lidar_array_format; later checks must not crash with IndexError on column 4."""
    dataset = build_synthetic_dataset()
    dataset.streams[LIDAR_NAME].frames[0] = lidar_frame(dataset)[:, :4].copy()
    results = run_all_checks(dataset)
    assert [result.name for result in results if not result.passed] == ["lidar_array_format"]


def test_truncated_poses_archive_is_reported_as_unloadable(tmp_path):
    """A truncated poses.npz raises zipfile.BadZipFile today, which validate_dataset does not catch."""
    root = write_valid(tmp_path)
    poses_path = root / "poses.npz"
    poses_path.write_bytes(poses_path.read_bytes()[: poses_path.stat().st_size // 2])
    assert validate_dataset(root).failed_check_names() == ["dataset_loadable"]
