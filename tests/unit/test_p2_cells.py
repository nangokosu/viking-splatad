"""Known-answer tests for the per-run P2 cell inputs (eval/p2_cells.py)."""

from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval.p2_cells import (
    IMAGE_FACTORS_MOUNT,
    IMAGE_REPO_MOUNT,
    RASTER_VALUES_PER_RAY,
    accumulate_recorded_evidence,
    factor_export_command,
    p1_hit_cells,
)
from viking_splatad.eval.rendered_pointcloud import RenderedSweep

# Evidence column order of T78 (hidden_geometry.recorded_ray_evidence).
SEEN, BLOCKED, NO_RETURN, PASSED_THROUGH = range(4)


def yaw_pose(yaw_radians: float, translation: np.ndarray) -> np.ndarray:
    """4x4 lidar_to_map pose: rotation about z by yaw, then translation.

    Args:
        yaw_radians: Yaw angle. translation: (3,) map position of the sensor.
    Returns:
        (4, 4) float64 pose.
    """
    pose = np.eye(4)
    pose[:2, :2] = [[np.cos(yaw_radians), -np.sin(yaw_radians)], [np.sin(yaw_radians), np.cos(yaw_radians)]]
    pose[:3, 3] = translation
    return pose


def test_p1_hit_cells_marks_exactly_the_cells_holding_a_mapped_point():
    """A yawed sweep's points land in the cells their map positions floor to; other cells stay empty."""
    voxel_size, grid_origin, grid_shape = 0.5, np.array([-5.0, -5.0, -1.0]), (20, 20, 4)
    pose = yaw_pose(np.pi / 2, np.array([1.0, 2.0, 0.0]))
    # Lidar point (1, 0, 0) yawed by 90 deg is map direction +y: map position (1, 3, 0).
    lidar_points = np.array([[1.0, 0.0, 0.0]])
    expected_map_point = np.array([1.0, 3.0, 0.0])
    expected_index = np.floor((expected_map_point - grid_origin) / voxel_size).astype(int)
    other_index = expected_index + np.array([1, 0, 0])
    sweep = RenderedSweep(stem="0_0", lidar_to_map=pose, gated_points=lidar_points, ungated_points=lidar_points,
                          real_points=lidar_points)
    hit = p1_hit_cells([sweep], np.stack([expected_index, other_index]), grid_origin, voxel_size, grid_shape)
    assert hit.tolist() == [True, False]


def test_p1_hit_cells_refuses_sweeps_without_ungated_points():
    """Binding P1 is the ungated point set; a sweep without it cannot be scored."""
    sweep = RenderedSweep(stem="0_0", lidar_to_map=np.eye(4), gated_points=np.zeros((1, 3)), ungated_points=None,
                          real_points=np.zeros((1, 3)))
    with pytest.raises(ValueError):
        p1_hit_cells([sweep], np.zeros((1, 3), dtype=int), np.zeros(3), 1.0, (2, 2, 2))


def write_sweep(path: Path, rays_az_el_depth: np.ndarray, did_return: np.ndarray, pose: np.ndarray) -> None:
    """Write a minimal rendered-sweep npz holding only the recorded raster and the pose.

    Args:
        path: Output file. rays_az_el_depth: (R, 3) azimuth deg, elevation deg, depth.
        did_return: (R,) bool. pose: (4, 4) lidar_to_map.
    """
    raster = np.zeros((len(rays_az_el_depth), RASTER_VALUES_PER_RAY))
    raster[:, :3] = rays_az_el_depth
    np.savez(path, raster_pts=raster, raster_did_return=did_return, lidar_to_map=pose)


def test_recorded_evidence_follows_one_ray_through_a_row_of_cells(tmp_path):
    """One forward ray returning at 5 m from a sensor at x=10 yawed by 180 deg (looking along -x):
    the cell around the return is seen, nearer cells are passed through, farther cells are blocked,
    and a cell off the ray is unobserved. A second sweep with a no-return beam adds no_return."""
    voxel_size = 1.0
    sensor_position = np.array([10.0, 0.0, 0.0])
    pose = yaw_pose(np.pi, sensor_position)
    return_depth = 5.0
    # Map cells on the ray (the -x axis from the sensor) at depths 2, 5 and 8 m, and one cell 3 m off the ray.
    depths_on_ray = np.array([2.0, return_depth, 8.0])
    centres = np.vstack([sensor_position - np.outer(depths_on_ray, [1.0, 0.0, 0.0]),
                         sensor_position + np.array([-return_depth, 3.0, 0.0])])
    write_sweep(tmp_path / "0_1.npz", np.array([[0.0, 0.0, return_depth]]), np.array([True]), pose)
    evidence = accumulate_recorded_evidence(tmp_path, ["0_1"], centres, voxel_size)
    expected = np.zeros((4, 4), dtype=bool)
    expected[0, PASSED_THROUGH] = True
    expected[1, SEEN] = True
    expected[2, BLOCKED] = True
    assert np.array_equal(evidence, expected)
    # A no-return beam pointing at the off-ray cell (azimuth from the cell's direction in the lidar frame).
    off_ray_lidar = (centres[3] - sensor_position) @ pose[:3, :3]
    azimuth_degrees = np.degrees(np.arctan2(off_ray_lidar[1], off_ray_lidar[0]))
    write_sweep(tmp_path / "1_2.npz", np.array([[azimuth_degrees, 0.0, 1.0]]), np.array([False]), pose)
    evidence = accumulate_recorded_evidence(tmp_path, ["0_1", "1_2"], centres, voxel_size)
    expected[3, NO_RETURN] = True
    assert np.array_equal(evidence, expected)


def test_factor_export_command_mounts_repo_read_only_and_only_the_factor_folder_writable(tmp_path):
    """The run's trainer folder is passed repo-relative, the repo mount is :ro, and the archive goes to the factor mount."""
    project_root = tmp_path / "repo"
    train_dir = project_root / "runs" / "run_a" / "train" / "s" / "m" / "run_a"
    train_dir.mkdir(parents=True)
    factors_dir = tmp_path / "factors"
    factors_dir.mkdir()
    command = factor_export_command(project_root, train_dir, factors_dir, "image:tag", "archive.npz")
    assert command[:2] == ["docker", "run"]
    assert f"{project_root.resolve()}:{IMAGE_REPO_MOUNT}:ro" in command
    assert f"{factors_dir.resolve()}:{IMAGE_FACTORS_MOUNT}" in command
    assert command[command.index("--train-dir") + 1] == str(Path("runs/run_a/train/s/m/run_a"))
    assert command[command.index("--output-path") + 1] == f"{IMAGE_FACTORS_MOUNT}/archive.npz"
    assert command[-1] == "--factorized" and "image:tag" in command


def test_factor_export_command_refuses_a_trainer_folder_outside_the_repo(tmp_path):
    """The container sees only the repo, so a folder elsewhere is refused before docker runs."""
    (tmp_path / "repo").mkdir()
    with pytest.raises(ValueError):
        factor_export_command(tmp_path / "repo", tmp_path / "elsewhere", tmp_path, "image:tag", "archive.npz")
