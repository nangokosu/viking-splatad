"""Known-answer tests for the per-run P2 cell inputs (eval/p2_cells.py)."""

from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval.p2_cells import (
    IMAGE_FACTORS_MOUNT,
    IMAGE_REPO_MOUNT,
    factor_export_command,
    p1_hit_cells,
)
from viking_splatad.eval.rendered_pointcloud import RenderedSweep


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
