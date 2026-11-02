"""Tests of the PyVista viewer's scene data on the real smoke run and the real May ground truth.

Uses the fetched smoke run (renders/test/lidar/*.npz) and data/gt/may_short_grass/cuboids.json. Both
folders are gitignored, so on a clone without them the module skips with a reason. Expected values
are computed here from first principles (scipy rotations, numpy norms), not copied from the code.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from viking_splatad.eval.detection import load_cuboids_json
from viking_splatad.eval.rendered_pointcloud import load_rendered_sweep
from viking_splatad.labels.cuboids import DEFAULT_BOUNDARY_TOLERANCE_METERS
from viking_splatad.viz.pyvista_scene import VTK_POINTS_PER_EDGE, cuboid_wireframe_arrays

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIEWER_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "15_view_pose_pyvista.py"
SMOKE_RUN_GLOB = "may_chunks0_1_smoke-*"
MAY_CUBOIDS_PATH = PROJECT_ROOT / "data" / "gt" / "may_short_grass" / "cuboids.json"

smoke_run_dirs = sorted((PROJECT_ROOT / "runs").glob(SMOKE_RUN_GLOB))
smoke_sweep_paths = sorted((smoke_run_dirs[-1] / "renders" / "test" / "lidar").glob("*.npz")) if smoke_run_dirs else []

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not (smoke_sweep_paths and MAY_CUBOIDS_PATH.is_file()),
                       reason="needs the fetched smoke run (runs/) and data/gt/may_short_grass"),
]


def load_viewer_script():
    """Import scripts/15 as a module (PyVista is imported lazily inside it)."""
    module_spec = importlib.util.spec_from_file_location("view_pose_pyvista", VIEWER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def accumulation_threshold() -> float:
    """The T67 threshold the evaluation uses (configs/eval.yaml)."""
    return float(yaml.safe_load((PROJECT_ROOT / "configs" / "eval.yaml").read_text())["eval"]["render_accumulation_threshold"])


def independent_inside_mask(points_map: np.ndarray, cuboids) -> np.ndarray:
    """Point-in-any-box test written from scratch: rotate into each box frame, compare with half sizes."""
    inside_any = np.zeros(points_map.shape[0], dtype=bool)
    for cuboid in cuboids:
        rotation = Rotation.from_quat(cuboid.quaternion_xyzw)
        # p_box = R^T (p - c): scipy's inverse rotation applied to the offsets.
        points_in_box_frame = rotation.inv().apply(points_map - np.array(cuboid.center_xyz))
        half_sizes = np.array(cuboid.dimensions_xyz) / 2.0 + DEFAULT_BOUNDARY_TOLERANCE_METERS
        inside_any |= np.all(np.abs(points_in_box_frame) <= half_sizes, axis=1)
    return inside_any


def test_every_real_cuboid_wireframe_has_its_own_edge_lengths():
    """Each real box: 12 edges whose lengths are its three full sizes, 4 edges per size."""
    cuboids = load_cuboids_json(MAY_CUBOIDS_PATH)
    assert cuboids
    corner_points, line_cells = cuboid_wireframe_arrays(cuboids)
    edge_cells = line_cells.reshape(len(cuboids), -1, VTK_POINTS_PER_EDGE + 1)
    for cuboid_index, cuboid in enumerate(cuboids):
        starts = corner_points[edge_cells[cuboid_index, :, 1]]
        ends = corner_points[edge_cells[cuboid_index, :, 2]]
        edge_lengths = np.linalg.norm(starts - ends, axis=1)
        assert edge_lengths.size == 12
        for full_size in cuboid.dimensions_xyz:
            assert np.count_nonzero(np.isclose(edge_lengths, full_size, atol=1e-9)) >= 4  # a square section repeats a size


def test_smoke_sweeps_colour_exactly_the_points_inside_the_cuboids():
    """On each real smoke sweep, the viewer's inside flags equal an independent box-frame test,
    and the per-box counts add up to at least the number of inside points."""
    viewer = load_viewer_script()
    cuboids = load_cuboids_json(MAY_CUBOIDS_PATH)
    assert smoke_sweep_paths
    for sweep_path in smoke_sweep_paths:
        sweep = load_rendered_sweep(sweep_path, accumulation_threshold())
        arrays = viewer.panel_arrays(sweep, cuboids, use_gated=False)
        np.testing.assert_array_equal(arrays["inside"], independent_inside_mask(arrays["rendered"], cuboids))
        assert arrays["rendered"].shape[0] == (sweep.ungated_points if sweep.ungated_points is not None else sweep.gated_points).shape[0]
        # Each box's count equals an independent single-box test on the same points.
        expected_counts = [int(independent_inside_mask(arrays["rendered"], [cuboid]).sum()) for cuboid in cuboids]
        assert arrays["per_cuboid_counts"].tolist() == expected_counts
        assert arrays["inside"].any()


def test_neighbourhood_crop_keeps_exactly_the_points_in_the_grown_boxes():
    """With the boxes grown by a margin m (full x and y sizes + 2m), the drawn points are exactly those
    an independent box-frame test finds inside the grown boxes, a subset of the uncropped set, and the
    inside flags still refer to the ungrown boxes."""
    from dataclasses import replace

    viewer = load_viewer_script()
    cuboids = load_cuboids_json(MAY_CUBOIDS_PATH)
    # A margin derived from the data: the median of the boxes' own horizontal full sizes.
    margin_meters = float(np.median([size for cuboid in cuboids for size in cuboid.dimensions_xyz[:2]]))
    grown = [replace(cuboid, dimensions_xyz=(cuboid.dimensions_xyz[0] + 2 * margin_meters,
                                             cuboid.dimensions_xyz[1] + 2 * margin_meters,
                                             cuboid.dimensions_xyz[2])) for cuboid in cuboids]
    for sweep_path in smoke_sweep_paths:
        sweep = load_rendered_sweep(sweep_path, accumulation_threshold())
        everything = viewer.panel_arrays(sweep, cuboids, use_gated=False)
        cropped = viewer.panel_arrays(sweep, cuboids, use_gated=False, neighbourhood_cuboids=grown)
        expected_kept = independent_inside_mask(everything["rendered"], grown)
        np.testing.assert_allclose(cropped["rendered"], everything["rendered"][expected_kept])
        assert cropped["rendered_total"] == everything["rendered"].shape[0]
        # Every point inside an ungrown box is inside its grown box, so the inside count is unchanged.
        assert cropped["inside"].sum() == everything["inside"].sum()
        # Non-vacuous: the crop actually drops points here, yet keeps some.
        assert 0 < expected_kept.sum() < expected_kept.size
        assert everything["inside"].any()


def test_voxel_maps_of_each_smoke_sweep_match_an_independent_count():
    """Per smoke test sweep (the run's own returned test points), with every scored class selected and a
    radius larger than the scan: the render voxels (correct + incorrect) are exactly the distinct grid
    voxels hit by that sweep's ungated points (floor((p - origin) / voxel), counted by numpy) that lie in
    the sweep's scoring neighbourhood, and the GT voxels (hit + miss) are the GT voxels of the scored
    classes in that neighbourhood. Both counts must be non-zero, so the check cannot pass vacuously."""
    viewer = load_viewer_script()
    eval_settings = yaml.safe_load((PROJECT_ROOT / "configs" / "eval.yaml").read_text())["eval"]
    scene = viewer.load_run_scene(smoke_run_dirs[-1], None, eval_settings, accumulation_threshold(), PROJECT_ROOT)
    scored_class_ids = list(viewer.resolve_selected_class_ids(eval_settings, None))
    assert len(scene.sweeps) == len(smoke_sweep_paths)
    huge_radius = 1e6
    for stem, sweep in scene.sweeps.items():
        maps = viewer.sweep_voxel_maps(sweep, scene, scored_class_ids, False, huge_radius)
        # The sweep's own region: union over classes of the neighbourhood this pose sees.
        region = np.zeros(scene.ground_truth_class_ids.shape, dtype=bool)
        for class_id in scored_class_ids:
            region |= viewer.restrict_mask_to_visible(scene.class_neighbourhoods[class_id], scene.grid_origin,
                                                      scene.voxel_size, [sweep.lidar_to_map], scene.max_range,
                                                      scene.vertical_half_fov)
        points_map = sweep.ungated_points @ sweep.lidar_to_map[:3, :3].T + sweep.lidar_to_map[:3, 3]
        indices = np.floor((points_map - scene.grid_origin) / scene.voxel_size).astype(int)
        in_grid = np.all((indices >= 0) & (indices < np.array(region.shape)), axis=1)
        occupied_in_region = {tuple(index) for index in indices[in_grid] if region[tuple(index)]}
        gt_in_region = int(np.count_nonzero(region & np.isin(scene.ground_truth_class_ids, scored_class_ids)))
        render_total = maps["render_correct"].shape[0] + maps["render_incorrect"].shape[0]
        assert render_total == len(occupied_in_region) > 0, stem
        assert maps["gt_hit"].shape[0] + maps["gt_miss"].shape[0] == gt_in_region > 0, stem
        assert maps["point_count"] == sweep.ungated_points.shape[0], stem


def test_pooled_maps_reproduce_the_reports_voxel_counts_per_class():
    """The pooled maps of one class are the report's cells: TP/FP/FN of semantic_occupancy.json's ungated voxel
    metrics equal hit GT voxels / incorrect render voxels / missed GT voxels of the same class. Recomputed
    here by the eval script's own pooling (scripts/13 evaluate_run), not copied from the saved report."""
    viewer = load_viewer_script()
    eval_script = viewer.load_eval_script()
    eval_settings = yaml.safe_load((PROJECT_ROOT / "configs" / "eval.yaml").read_text())["eval"]
    run_dir = smoke_run_dirs[-1]
    scene = viewer.load_run_scene(run_dir, None, eval_settings, accumulation_threshold(), PROJECT_ROOT)
    _, report = eval_script.evaluate_run(run_dir, None, eval_settings, None, PROJECT_ROOT)
    for class_name, class_id in dict(eval_settings["scored_class_ids"]).items():
        voxel = report["point_sets"]["ungated"]["classes"][class_name]["voxel"]
        maps = viewer.pooled_voxel_maps(list(scene.sweeps.values()), scene, [class_id], False, 1e6)
        assert maps["totals"]["gt_hit"] == voxel["true_positives"], class_name
        assert maps["totals"]["gt_miss"] == voxel["false_negatives"], class_name
        # With one class selected, incorrect = occupied in the neighbourhood and not this class's GT (the report's FP).
        assert maps["totals"]["render_incorrect"] == voxel["false_positives"], class_name
    assert voxel is not None and maps["sweep_count"] == len(smoke_sweep_paths)
