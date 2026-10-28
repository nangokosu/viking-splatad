"""Tests of P1 semantic occupancy (T45, T46, T67, T68, scripts/13), synthetic known answers."""

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from viking_splatad.eval.detection import (
    cuboid_detection_rate,
    cuboid_point_counts,
    cuboid_voxel_mask,
    median_half_footprint,
    n_min_from_real_counts,
    neighbourhood_point_precision,
    visible_from_any_pose,
)
from viking_splatad.eval.rendered_pointcloud import (
    load_test_lidar_sweeps,
    points_to_map_voxels,
    resolution_at_gate,
    ungated_points_from_raster,
)
from viking_splatad.labels.cuboids import Cuboid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVAL_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "13_eval_semantic_occupancy.py"
# Class ids of the labels file (PLAN §5): 1 trunk, 3 rock.
TRUNK_CLASS_ID, ROCK_CLASS_ID = 1, 3


def load_eval_script():
    """Import scripts/13 as a module (its file name is not a valid module name)."""
    module_spec = importlib.util.spec_from_file_location("eval_semantic_occupancy", EVAL_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def yaw_quaternion_xyzw(yaw_radians: float):
    """Unit quaternion (x, y, z, w) of a rotation by ``yaw_radians`` about +z."""
    return (0.0, 0.0, math.sin(yaw_radians / 2.0), math.cos(yaw_radians / 2.0))


def make_box(class_id, center, dimensions, yaw_radians=0.0, box_id="box"):
    """A Cuboid with the given pose and FULL sizes."""
    return Cuboid(box_id, class_id, f"class{class_id}", tuple(center), tuple(dimensions), yaw_quaternion_xyzw(yaw_radians))


# ---------------------------------------------------------------- T45


@pytest.mark.covers("T45")
def test_detection_rate_counts_points_in_rotated_boxes_exactly():
    """Box A yawed 45° (2 x 0.5 m) gets 3 points along its long axis and one point that only its
    axis-aligned bounding box contains; box B gets 1 point. n_min = 2: A detected, B not -> 0.5.

    Long axis of A after 45° yaw is (1, 1)/√2; points at s·(1,1)/√2 for |s| < 1 are inside.
    (0.6, -0.6) is inside A's AABB (half-width (1 + 0.25)/√2 ≈ 0.88) but 0.6·√2 ≈ 0.85 m along the
    short axis (−1, 1)/√2, whose half-size is 0.25.
    """
    box_a = make_box(TRUNK_CLASS_ID, (0.0, 0.0, 0.0), (2.0, 0.5, 1.0), math.radians(45.0), "a")
    box_b = make_box(TRUNK_CLASS_ID, (10.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0, "b")
    long_axis = np.array([1.0, 1.0, 0.0]) / math.sqrt(2.0)
    points = np.vstack([
        long_axis * -0.9, long_axis * 0.0, long_axis * 0.9,  # inside A
        [0.6, -0.6, 0.0],                                    # A's AABB only
        [10.2, 0.1, 0.0],                                    # inside B
        [20.0, 0.0, 0.0],                                    # nowhere
    ])
    counts = cuboid_point_counts(points, [box_a, box_b])
    np.testing.assert_array_equal(counts, [3, 1])

    result = cuboid_detection_rate(counts, n_min=2)

    assert result.cuboid_count == 2
    assert result.detection_rate == 0.5
    # rate(n) for n = 1, 2, 3: both boxes have >= 1, only A has >= 2 and >= 3.
    assert result.rate_by_min_points == [1.0, 0.5, 0.5]


@pytest.mark.covers("T45")
def test_detection_rate_without_scored_boxes_is_zero_and_empty_curve():
    """No scored cuboid: rate 0.0, empty curve (no division by zero)."""
    result = cuboid_detection_rate(np.zeros(0, dtype=np.int64), n_min=3)
    assert (result.cuboid_count, result.detection_rate, result.rate_by_min_points) == (0, 0.0, [])


def test_n_min_is_an_observed_lower_quartile_of_hit_boxes():
    """Counts [0, 4, 8, 12, 16]: the unhit box is ignored; lower quartile of [4, 8, 12, 16] with
    method "lower" is the element at floor(0.25 · 3) = 0 -> 4. No hit at all -> None."""
    assert n_min_from_real_counts(np.array([0, 4, 8, 12, 16]), 0.25) == 4
    assert n_min_from_real_counts(np.array([0, 0]), 0.25) is None


# ---------------------------------------------------------------- T46


@pytest.mark.covers("T46")
def test_neighbourhood_precision_counts_halo_but_not_far_points():
    """Trunk 1 x 1 x 2 m at the origin, margin 0.5 m: neighbourhood = 2 x 2 x 2 m (same z span).

    2 points inside the trunk, 2 in the halo (0.5 < |y| <= 1), 1 above the box's z span and 1 far
    away: precision = 2 / 4 = 0.5. The point above and the far one are outside the neighbourhood.
    """
    trunk = make_box(TRUNK_CLASS_ID, (0.0, 0.0, 1.0), (1.0, 1.0, 2.0))
    points = np.array([
        [0.1, 0.1, 0.5], [-0.2, 0.3, 1.5],   # inside
        [0.0, 0.8, 1.0], [0.0, -0.9, 0.2],   # halo
        [0.0, 0.0, 2.5],                     # above the z span
        [5.0, 5.0, 1.0],                     # far
    ])

    result = neighbourhood_point_precision(points, [trunk], margin=0.5)

    assert (result.inside_count, result.neighbourhood_count, result.precision) == (2, 4, 0.5)


@pytest.mark.covers("T46")
def test_neighbourhood_precision_is_undefined_without_points():
    """Empty neighbourhood -> precision None (not 0 or 1)."""
    trunk = make_box(TRUNK_CLASS_ID, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    result = neighbourhood_point_precision(np.array([[9.0, 9.0, 9.0]]), [trunk], margin=0.5)
    assert (result.inside_count, result.neighbourhood_count, result.precision) == (0, 0, None)


def test_margin_is_median_half_footprint():
    """Footprints (dx + dy)/2 = 1, 2, 4 -> median 2 -> margin 1."""
    boxes = [make_box(TRUNK_CLASS_ID, (0, 0, 0), (width, width, 1.0)) for width in (1.0, 2.0, 4.0)]
    assert median_half_footprint(boxes) == 1.0


# ---------------------------------------------------------------- T67


def synthetic_render(ray_azimuths_deg, ray_elevations_deg, ray_depths, ray_times, drop_probabilities,
                     accumulations, velocity):
    """Arrays of a one-channel render (C=1, H=1, W=rays) built with SplatAD's point formula.

    pred_points rows = rays with drop probability <= 0.5:
    xyz = depth · (cos el cos az, cos el sin az, sin el) + v · (t − (max t + min t)/2).
    Intensities are 0.1·(ray index + 1), distinct, so rows can be told apart.
    """
    ray_count = len(ray_depths)
    raster = np.zeros((1, 1, ray_count, 5), dtype=np.float32)
    raster[0, 0, :, 0], raster[0, 0, :, 1] = ray_azimuths_deg, ray_elevations_deg
    raster[0, 0, :, 2], raster[0, 0, :, 3] = ray_depths, ray_times
    intensities = (np.arange(ray_count) + 1) * 0.1
    azimuth, elevation = np.radians(ray_azimuths_deg), np.radians(ray_elevations_deg)
    directions = np.stack([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)], 1)
    shift = (max(ray_times) + min(ray_times)) / 2.0
    centred_times = np.asarray(ray_times) - shift
    points = np.asarray(ray_depths)[:, None] * directions + np.asarray(velocity) * centred_times[:, None]
    kept = np.asarray(drop_probabilities) <= 0.5
    pred_points = np.c_[points[kept], intensities[kept], centred_times[kept]].astype(np.float32)

    def raster_channel(values):
        """[1, 1, W, 1] float32 copy of per-ray values."""
        return np.asarray(values, dtype=np.float32).reshape(1, 1, ray_count, 1)

    return {
        "raster_pts": raster, "pred_depth": raster_channel(ray_depths),
        "pred_accumulation": raster_channel(accumulations), "pred_ray_drop_prob": raster_channel(drop_probabilities),
        "pred_intensity": raster_channel(intensities), "pred_points": pred_points,
    }, points


@pytest.mark.covers("T67")
def test_ungated_points_keep_high_accumulation_rays_with_motion_term():
    """5 rays, v = (2, 0, 0) m/s, times 0..0.08 s (shift 0.04 s).

    Rays 0, 1 pass the ray-drop gate; ray 2 is dropped (0.9) but its accumulation 0.8 >= 0.5, so the
    ungated set holds it; ray 3 has accumulation 0.2 (out); ray 4 has depth 0 (no ray, out).
    Expected points come from the formula with the true v, which T67 must recover from rows 0, 1.
    """
    arrays, expected_points = synthetic_render(
        ray_azimuths_deg=[0.0, 90.0, 180.0, -90.0, 45.0], ray_elevations_deg=[0.0, 10.0, -20.0, 0.0, 0.0],
        ray_depths=[10.0, 5.0, 8.0, 3.0, 0.0], ray_times=[0.0, 0.02, 0.04, 0.06, 0.08],
        drop_probabilities=[0.1, 0.2, 0.9, 0.95, 0.99], accumulations=[0.99, 0.9, 0.8, 0.2, 0.0],
        velocity=[2.0, 0.0, 0.0],
    )

    ungated = ungated_points_from_raster(**arrays, accumulation_threshold=0.5)

    np.testing.assert_allclose(ungated, expected_points[[0, 1, 2]], atol=1e-5)


@pytest.mark.covers("T67")
def test_ungated_points_refuse_rows_from_another_render():
    """pred_points whose intensities match no ray cannot be aligned -> ValueError."""
    arrays, _ = synthetic_render([0.0, 90.0], [0.0, 0.0], [10.0, 5.0], [0.0, 0.02], [0.1, 0.2], [0.9, 0.9], [0, 0, 0])
    arrays["pred_points"][:, 3] += 7.0
    with pytest.raises(ValueError, match="assigned"):
        ungated_points_from_raster(**arrays, accumulation_threshold=0.5)


def test_gate_resolution_is_the_gap_around_one_half():
    """Stored values 0.25, 0.498, 0.502, 0.9: the gap bracketing 0.5 is 0.004; one side empty -> 0."""
    assert resolution_at_gate(np.array([0.25, 0.498, 0.502, 0.9]), 0.5) == pytest.approx(0.004)
    assert resolution_at_gate(np.array([0.1, 0.2]), 0.5) == 0.0


# ---------------------------------------------------------------- T68


@pytest.mark.covers("T68")
def test_point_reaches_map_and_voxel_through_yawed_pose():
    """Lidar yawed 90° at (10, 20, 1): lidar +x -> map +y, so (3, 0, 0) lands at (10, 23, 1).

    Grid origin (0, 0, 0), voxel 2 m, shape (8, 16, 2): index ⌊(10, 23, 1)/2⌋ = (5, 11, 0), inside.
    A point 100 m behind lands at y = −80: outside the grid.
    """
    lidar_to_map = np.eye(4)
    lidar_to_map[:3, :3] = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    lidar_to_map[:3, 3] = [10.0, 20.0, 1.0]

    points_map, voxel_indices, inside_grid = points_to_map_voxels(
        np.array([[3.0, 0.0, 0.0], [-100.0, 0.0, 0.0]]), lidar_to_map, np.zeros(3), 2.0, (8, 16, 2)
    )

    np.testing.assert_allclose(points_map[0], [10.0, 23.0, 1.0], atol=1e-12)
    np.testing.assert_array_equal(voxel_indices[0], [5, 11, 0])
    np.testing.assert_array_equal(inside_grid, [True, False])


def test_cuboid_voxel_mask_uses_voxel_centres():
    """Box x, y, z in [1, 3] on a 1 m grid from 0: centres 1.5 and 2.5 per axis are inside -> 2³ = 8."""
    box = make_box(TRUNK_CLASS_ID, (2.0, 2.0, 2.0), (2.0, 2.0, 2.0))
    mask = cuboid_voxel_mask([box], np.zeros(3), 1.0, (5, 5, 5))
    assert mask.sum() == 8 and mask[1:3, 1:3, 1:3].all()


def test_visibility_uses_range_and_vertical_fov():
    """Sensor at the origin, range 10 m, half FOV 45°: (5, 0, 0) in; (5, 0, 6) above 45°; (11, 0, 0) too far."""
    visible = visible_from_any_pose(np.array([[5.0, 0, 0], [5.0, 0, 6.0], [11.0, 0, 0]]), [np.eye(4)], 10.0, math.pi / 4)
    np.testing.assert_array_equal(visible, [True, False, False])


# ---------------------------------------------------------------- scripts/13 end to end


def write_fake_session(project_root: Path, session_name: str, labels_sample_key: str, dataset_name: str):
    """Session YAML, dataset manifest and GT (one trunk 1 x 1 x 2 m at (10, 0, 1), one rock far away).

    Returns:
        sha256 of the manifest (for split.json).
    """
    eval_script = load_eval_script()
    gt_dir = project_root / "data" / "gt" / session_name
    gt_dir.mkdir(parents=True)
    session_yaml = {key: {} for key in ("window", "topics", "frames", "camera", "lidar", "radar", "split", "splatad")}
    session_yaml["session"] = {
        "name": session_name, "extrinsics_variant": "x", "bag_dir": str(project_root), "calibration_dir": str(project_root),
        "labels_json": str(project_root), "labels_sample_key": labels_sample_key,
        "output_dir": str(project_root), "gt_dir": str(gt_dir),
    }
    (project_root / "configs" / "sessions").mkdir(parents=True, exist_ok=True)
    (project_root / "configs" / "sessions" / f"{session_name}.yaml").write_text(yaml.safe_dump(session_yaml))
    boxes = [
        {"box_id": "t", "class_id": TRUNK_CLASS_ID, "class_name": "tree_trunk", "center_xyz": [10.0, 0.0, 1.0],
         "dimensions_xyz": [1.0, 1.0, 2.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
        {"box_id": "r", "class_id": ROCK_CLASS_ID, "class_name": "rock", "center_xyz": [500.0, 0.0, 1.0],
         "dimensions_xyz": [1.0, 1.0, 1.0], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]},
    ]
    (gt_dir / "cuboids.json").write_text(json.dumps(boxes))
    # Grid from (0, -5, -1), voxel 0.5 m: trunk centres x in (9.5, 10.5), |y| < 0.5, z in (0, 2).
    origin, voxel_size, shape = np.array([0.0, -5.0, -1.0]), 0.5, (40, 20, 10)
    class_ids = np.zeros(shape, dtype=np.uint8)
    class_ids[19:21, 9:11, 2:6] = TRUNK_CLASS_ID
    np.savez(gt_dir / "occupancy.npz", class_ids=class_ids, origin=origin, voxel_size=np.float64(voxel_size))
    manifest_dir = project_root / "data" / "intermediate" / dataset_name
    manifest_dir.mkdir(parents=True)
    manifest_path = manifest_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"session": {"name": session_name},
                                         "lidar": {"elevation_mapping": {"0": 0.5, "1": -0.5}}}))
    return eval_script.file_sha256(manifest_path)


def write_fake_run(project_root: Path, run_id: str, dataset_name: str, manifest_sha256: str) -> Path:
    """Run folder with split.json and one test sweep: lidar at (0, 0, 1), rays toward the trunk.

    Rays (az deg, depth): 0° 10 m and 1° 10 m pass the gate; 2° 10 m and −1° 10 m are dropped but
    have accumulation 0.8; atan2(0.8, 10) at hypot(10, 0.8) is a halo point (y = 0.8), gated.
    Real returns: 3 points inside the trunk -> n_min = 3.
    """
    run_dir = project_root / "runs" / run_id
    render_dir = run_dir / "renders" / "test" / "lidar"
    render_dir.mkdir(parents=True)
    stem = "000005_1000"
    halo_azimuth = math.degrees(math.atan2(0.8, 10.0))
    arrays, _ = synthetic_render(
        ray_azimuths_deg=[0.0, 1.0, 2.0, -1.0, halo_azimuth], ray_elevations_deg=[0.0] * 5,
        ray_depths=[10.0, 10.0, 10.0, 10.0, math.hypot(10.0, 0.8)], ray_times=[0.0, 0.01, 0.02, 0.03, 0.04],
        drop_probabilities=[0.1, 0.1, 0.9, 0.9, 0.1], accumulations=[0.9, 0.9, 0.8, 0.8, 0.9], velocity=[0, 0, 0],
    )
    lidar_to_map = np.eye(4)
    lidar_to_map[2, 3] = 1.0
    real_points = np.array([[10.0, 0.0, 0.0, 0.0, 0.0], [10.2, 0.1, 0.0, 0.0, 0.0], [9.8, -0.1, 0.5, 0.0, 0.0]])
    np.savez(render_dir / f"{stem}.npz", **arrays, gt_points=real_points.astype(np.float32), lidar_to_map=lidar_to_map)
    split_record = {
        "run": {"run_id": run_id}, "dataset": {"name": dataset_name, "manifest_sha256": manifest_sha256},
        "render_check": {"lidar/test": {"match": True}},
        "frames": [{"kind": "lidar", "split": "test", "frame_index": 5, "file": f"lidar/ouster/{stem}.npy"},
                   {"kind": "lidar", "split": "train", "frame_index": 4, "file": "lidar/ouster/000004_900.npy"}],
    }
    (run_dir / "split.json").write_text(json.dumps(split_record))
    return run_dir


EVAL_SETTINGS = {"render_accumulation_threshold": 0.5, "scored_class_ids": {"tree_trunk": 1, "rock": 3},
                 "n_min_quantile": 0.25}


def test_script_scores_one_run_against_its_own_session(tmp_path):
    """Trunk: ungated 4 points (>= n_min 3) -> detected; gated 2 points -> not.
    Point precision: ungated 4 / 5, gated 2 / 3 (the halo point counts in both).
    The rock is beyond the farthest real return (≈10 m), so it is not scored.
    """
    eval_script = load_eval_script()
    manifest_sha256 = write_fake_session(tmp_path, "may_x", "2024_05", "may_x_smoke")
    run_dir = write_fake_run(tmp_path, "run-may", "may_x_smoke", manifest_sha256)

    _, report = eval_script.evaluate_run(run_dir, None, EVAL_SETTINGS, None, tmp_path)

    assert (report["session"], report["labels_sample_key"], report["dataset"]) == ("may_x", "2024_05", "may_x_smoke")
    assert report["classes"]["tree_trunk"]["n_min"] == 3 and report["classes"]["rock"]["scored_cuboids"] == 0
    ungated, gated = (report["point_sets"][name]["classes"]["tree_trunk"] for name in ("ungated", "gated"))
    assert ungated["detection"]["detection_rate"] == 1.0 and gated["detection"]["detection_rate"] == 0.0
    assert ungated["point_precision"]["precision"] == pytest.approx(4 / 5)
    assert gated["point_precision"]["precision"] == pytest.approx(2 / 3)


def test_script_refuses_mismatched_renders_and_cross_session_comparison(tmp_path):
    """An extra npz is refused; reports of two sessions cannot be compared."""
    eval_script = load_eval_script()
    may_sha = write_fake_session(tmp_path, "may_x", "2024_05", "may_x")
    june_sha = write_fake_session(tmp_path, "june_x", "2024_06", "june_x")
    may_run = write_fake_run(tmp_path, "run-may", "may_x", may_sha)
    june_run = write_fake_run(tmp_path, "run-june", "june_x", june_sha)
    _, may_report = eval_script.evaluate_run(may_run, None, EVAL_SETTINGS, None, tmp_path)
    _, june_report = eval_script.evaluate_run(june_run, None, EVAL_SETTINGS, None, tmp_path)
    assert june_report["labels_sample_key"] == "2024_06"
    with pytest.raises(ValueError, match="different sessions"):
        eval_script.comparison_markdown([may_report, june_report])

    render_dir = may_run / "renders" / "test" / "lidar"
    (render_dir / "000004_900.npz").write_bytes((render_dir / "000005_1000.npz").read_bytes())
    with pytest.raises(ValueError, match="unexpected"):
        load_test_lidar_sweeps(render_dir, ["000005_1000"], 0.5)
