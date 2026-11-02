"""Tests of the ripple (ring) view: the ground-facing filter and the refusals of ``--ripples``."""

import importlib.util
import math
from pathlib import Path

import numpy as np

from viking_splatad.viz.pyvista_scene import beam_elevation_degrees, ground_facing_points

VIEWER_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "15_view_pose_pyvista.py"
CONFIG_TEXT = "eval: {render_accumulation_threshold: 0.5, viz_radius_meters: 25.0, scored_class_ids: {tree_trunk: 1}}\n"


def load_viewer_script():
    """Import scripts/15 as a module (PyVista is imported lazily inside it).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("view_pose_pyvista_ripples", VIEWER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def ring_points_on_flat_ground(sensor_height: float, depression_degrees: float, azimuth_count: int) -> np.ndarray:
    """One ring's returns on flat ground below the sensor: a circle of radius h / tan(depression) at z = -h.

    Args:
        sensor_height: Height h (m) of the sensor above the ground.
        depression_degrees: Beam depression below the horizontal (degrees).
        azimuth_count: Number of evenly spaced azimuths on the ring.

    Returns:
        (azimuth_count, 3) lidar-frame points.
    """
    ring_radius = sensor_height / math.tan(math.radians(depression_degrees))
    azimuths = np.linspace(0.0, 2.0 * math.pi, azimuth_count, endpoint=False)
    return np.stack([ring_radius * np.cos(azimuths), ring_radius * np.sin(azimuths), np.full(azimuth_count, -sensor_height)], axis=1)


def test_ground_rings_come_back_at_their_own_depression_angle_and_radius():
    """Flat ground 0.8 m below the sensor, rings at 5 and 10 degrees depression: each kept point's elevation is minus
    its ring's depression and its horizontal range is h / tan(depression), both computed here from first principles."""
    sensor_height = 0.8
    rings = {5.0: ring_points_on_flat_ground(sensor_height, 5.0, 16), 10.0: ring_points_on_flat_ground(sensor_height, 10.0, 16)}
    points, elevations = ground_facing_points(np.vstack(list(rings.values())), radius_meters=1e3)
    assert points.shape[0] == 32
    np.testing.assert_allclose(elevations[:16], -5.0)
    np.testing.assert_allclose(elevations[16:], -10.0)
    np.testing.assert_allclose(np.hypot(points[:16, 0], points[:16, 1]), sensor_height / math.tan(math.radians(5.0)))
    np.testing.assert_allclose(np.hypot(points[16:, 0], points[16:, 1]), sensor_height / math.tan(math.radians(10.0)))


def test_points_at_or_above_the_horizon_and_beyond_the_radius_are_dropped():
    """A point on the horizontal plane (elevation 0), one above it, one far away and one below and near:
    only the last is kept."""
    lidar_points = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 1.0], [50.0, 0.0, -2.0], [3.0, 4.0, -1.0]])
    points, elevations = ground_facing_points(lidar_points, radius_meters=10.0)
    np.testing.assert_allclose(points, [[3.0, 4.0, -1.0]])
    np.testing.assert_allclose(elevations, [-math.degrees(math.atan2(1.0, 5.0))])


def test_an_empty_cloud_gives_empty_arrays():
    """No input points: zero kept points and zero elevations, not an error."""
    points, elevations = ground_facing_points(np.zeros((0, 3)), radius_meters=10.0)
    assert points.shape == (0, 3) and elevations.shape == (0,)


def test_beam_elevation_is_the_angle_above_the_horizontal_from_the_sensor():
    """Points on a 3-4-5 horizontal triangle at heights 0, +5 and -5 m: the horizontal range is 5, so the
    elevations are 0, +45 and -45 degrees (atan(opposite / adjacent), computed here)."""
    points = np.array([[3.0, 4.0, 0.0], [3.0, 4.0, 5.0], [3.0, 4.0, -5.0]])
    np.testing.assert_allclose(beam_elevation_degrees(points), [0.0, math.degrees(math.atan(1.0)), -math.degrees(math.atan(1.0))])


def test_ripples_with_two_runs_is_refused_and_without_a_pose_it_means_pooled(tmp_path, monkeypatch, capsys):
    """--ripples draws one run: two --run folders are refused with exit 1. Without --pose it is allowed (the pooled
    twin of the voxel map): main hands render_ripples the pooled stem."""
    viewer = load_viewer_script()
    config_path = tmp_path / "eval.yaml"
    config_path.write_text(CONFIG_TEXT)
    scene = type("Scene", (), {"sweeps": {"000005_1000": None}, "membership_sha256": "m", "gt_cuboids_sha256": "g"})()
    monkeypatch.setattr(viewer, "load_run_scene", lambda *arguments: scene)
    drawn_stems = []
    monkeypatch.setattr(viewer, "render_ripples", lambda scene_, stem, *rest: drawn_stems.append(stem))
    base = ["--eval-config", str(config_path), "--output-dir", str(tmp_path / "out"), "--ripples"]
    assert viewer.main(["--run", str(tmp_path / "a"), "--run", str(tmp_path / "b")] + base) == 1
    assert "--ripples draws one run" in capsys.readouterr().err
    assert viewer.main(["--run", str(tmp_path / "a")] + base) == 0
    assert drawn_stems == [viewer.POOLED_STEM]
