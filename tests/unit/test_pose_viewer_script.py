"""Tests of the pure logic in scripts/15_view_pose_pyvista.py on hand-built sweeps (no PyVista needed)."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval.rendered_pointcloud import RenderedSweep
from viking_splatad.labels.cuboids import Cuboid

VIEWER_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "15_view_pose_pyvista.py"
# Trunk 1 x 1 x 2 m centred at (10, 0, 1), and the same box grown horizontally by a 1 m margin (3 x 3 x 2).
TRUNK = Cuboid("t", 1, "tree_trunk", (10.0, 0.0, 1.0), (1.0, 1.0, 2.0), (0.0, 0.0, 0.0, 1.0))
GROWN_TRUNK = Cuboid("t", 1, "tree_trunk", (10.0, 0.0, 1.0), (3.0, 3.0, 2.0), (0.0, 0.0, 0.0, 1.0))


def load_viewer_script():
    """Import scripts/15 as a module (PyVista is imported lazily inside it).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("view_pose_pyvista", VIEWER_SCRIPT_PATH)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def make_scene(viewer, stems, membership="m", gt_sha="g"):
    """A RunScene holding only the given sweep stems (empty sweeps) and the trunk box.

    Args:
        viewer: The loaded scripts/15 module.
        stems: Sweep stems the scene contains.
        membership: membership_sha256 of the scene.
        gt_sha: gt_cuboids_sha256 of the scene.

    Returns:
        The scene.
    """
    empty = np.zeros((0, 3))
    sweeps = {stem: RenderedSweep(stem, np.eye(4), empty, empty, empty) for stem in stems}
    return viewer.RunScene("run", sweeps, [TRUNK], [GROWN_TRUNK], membership, gt_sha)


def lifted_sweep(ungated_points):
    """A sweep whose lidar sits 1 m above the map origin (pure z translation, no rotation).

    Args:
        ungated_points: (K, 3) ungated lidar-frame points, or None for a render without them.

    Returns:
        The sweep: gated point (10, 0, 0), real returns (10, 0, 0) and (30, 0, 0), all in the lidar frame.
    """
    lidar_to_map = np.eye(4)
    lidar_to_map[2, 3] = 1.0
    gated = np.array([[10.0, 0.0, 0.0]])
    return RenderedSweep("000005_1000", lidar_to_map, gated, ungated_points, np.array([[10.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))


def test_pose_selection_matches_stem_or_frame_index():
    """Of stems 000005_1000 and 000009_2000: '5', '05' and the full stem pick the first; no request picks both."""
    viewer = load_viewer_script()
    scenes = [make_scene(viewer, ["000009_2000", "000005_1000"])]
    assert viewer.select_pose_stems(scenes, None) == ["000005_1000", "000009_2000"]
    for request in ("5", "05", "000005_1000"):
        assert viewer.select_pose_stems(scenes, [request]) == ["000005_1000"]
    assert viewer.select_pose_stems(scenes, ["9", "5", "9"]) == ["000009_2000", "000005_1000"]


def test_pose_selection_refuses_unknown_poses_and_disjoint_runs():
    """A pose no run has, and runs sharing no sweep, are both refused."""
    viewer = load_viewer_script()
    scenes = [make_scene(viewer, ["000005_1000"])]
    with pytest.raises(ValueError):
        viewer.select_pose_stems(scenes, ["7"])
    with pytest.raises(ValueError):
        viewer.select_pose_stems([scenes[0], make_scene(viewer, ["000009_2000"])], None)


def test_runs_are_comparable_only_with_one_membership_and_one_ground_truth():
    """Equal hashes pass; a different train/test membership or a different cuboids hash is refused."""
    viewer = load_viewer_script()
    viewer.check_scenes_comparable([make_scene(viewer, ["a"]), make_scene(viewer, ["a"])])
    with pytest.raises(ValueError, match="membership"):
        viewer.check_scenes_comparable([make_scene(viewer, ["a"]), make_scene(viewer, ["a"], membership="other")])
    with pytest.raises(ValueError, match="cuboids"):
        viewer.check_scenes_comparable([make_scene(viewer, ["a"]), make_scene(viewer, ["a"], gt_sha="other")])


def test_panel_points_are_in_the_map_frame_and_flagged_by_the_trunk():
    """Ungated lidar points (10, 0, 0), (11, 0, 0), (30, 0, 0) with the lidar 1 m up are at map
    (10, 0, 1), (11, 0, 1), (30, 0, 1). The trunk spans x in [9.5, 10.5]: only the first is inside.
    Without a crop all 3 are kept; with the grown box (x in [8.5, 11.5]) the first two, the second as halo."""
    viewer = load_viewer_script()
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))
    everything = viewer.panel_arrays(sweep, [TRUNK], use_gated=False)
    np.testing.assert_allclose(everything["rendered"], [[10, 0, 1], [11, 0, 1], [30, 0, 1]])
    np.testing.assert_array_equal(everything["inside"], [True, False, False])
    assert everything["point_set"] == "ungated" and everything["per_cuboid_counts"].tolist() == [1]
    cropped = viewer.panel_arrays(sweep, [TRUNK], use_gated=False, neighbourhood_cuboids=[GROWN_TRUNK])
    np.testing.assert_allclose(cropped["rendered"], [[10, 0, 1], [11, 0, 1]])
    np.testing.assert_array_equal(cropped["inside"], [True, False])
    assert cropped["rendered_total"] == 3
    # The real returns (10, 0, 0) and (30, 0, 0) -> map x = 10 stays, x = 30 is cropped away.
    np.testing.assert_allclose(cropped["real"], [[10, 0, 1]])


def test_panel_falls_back_to_the_gated_set_when_asked_or_when_there_is_no_ungated_set():
    """use_gated, or a render without ungated points, draws the gated set and says so."""
    viewer = load_viewer_script()
    for sweep, use_gated in ((lifted_sweep(np.array([[11.0, 0.0, 0.0]])), True), (lifted_sweep(None), False)):
        arrays = viewer.panel_arrays(sweep, [TRUNK], use_gated=use_gated)
        assert arrays["point_set"] == "gated"
        np.testing.assert_allclose(arrays["rendered"], [[10, 0, 1]])


def test_main_refuses_bad_input_with_exit_code_one(tmp_path, capsys):
    """--rerender with two runs, a missing run folder and a malformed eval config are each a
    'refused' message and exit code 1, never a traceback."""
    viewer = load_viewer_script()
    assert viewer.main(["--run", "a", "--run", "b", "--rerender", "x"]) == 1
    assert "exactly one run" in capsys.readouterr().err
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("not_eval: 1\n")
    assert viewer.main(["--run", str(tmp_path / "missing"), "--eval-config", str(config_path)]) == 1
    assert "refused" in capsys.readouterr().err
    config_path.write_text("- just\n- a list\n")
    assert viewer.main(["--run", str(tmp_path / "missing"), "--eval-config", str(config_path)]) == 1
    assert "refused" in capsys.readouterr().err


def test_a_crop_to_a_far_box_leaves_no_points_but_keeps_the_totals():
    """With the neighbourhood far from every point, nothing is drawn, yet the pre-crop total is reported."""
    viewer = load_viewer_script()
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [11.0, 0.0, 0.0]]))
    far_box = Cuboid("far", 1, "tree_trunk", (500.0, 0.0, 1.0), (1.0, 1.0, 2.0), (0.0, 0.0, 0.0, 1.0))
    arrays = viewer.panel_arrays(sweep, [far_box], use_gated=False, neighbourhood_cuboids=[far_box])
    assert arrays["rendered"].shape == (0, 3) and arrays["real"].shape == (0, 3)
    assert arrays["rendered_total"] == 2 and arrays["inside"].size == 0


# ---------------------------------------------------------------- voxel cells (the occupancy picture)

# Grid: origin (0, -5, -1), voxel 0.5 m, shape (40, 20, 10). The trunk's GT voxels are the 2 x 2 x 4 block
# at index [19:21, 9:11, 2:6], i.e. centres x in {9.75, 10.25}, y in {-0.25, 0.25}, z in {0.25 .. 1.75}.
GRID_ORIGIN, VOXEL_SIZE, GRID_SHAPE = np.array([0.0, -5.0, -1.0]), 0.5, (40, 20, 10)
TRUNK_CLASS_ID = 1
ROCK_CLASS_ID = 3
GROUND_TRUTH_VOXEL_COUNT = 2 * 2 * 4


def make_voxel_scene(viewer):
    """A RunScene on the small grid: trunk GT block, a neighbourhood around it, and a view that sees everything.

    Args:
        viewer: The loaded scripts/15 module.

    Returns:
        The scene (one sweep stem, grid origin/size/shape as the module constants, trunk class 1 neighbourhood
        [17:23, 7:13, 0:8], range 100 m, vertical half FOV 90 degrees).
    """
    class_ids = np.zeros(GRID_SHAPE, dtype=np.uint8)
    class_ids[19:21, 9:11, 2:6] = TRUNK_CLASS_ID
    neighbourhood = np.zeros(GRID_SHAPE, dtype=bool)
    neighbourhood[17:23, 7:13, 0:8] = True
    scene = make_scene(viewer, ["000005_1000"])
    scene.grid_origin, scene.voxel_size, scene.ground_truth_class_ids = GRID_ORIGIN, VOXEL_SIZE, class_ids
    scene.class_neighbourhoods = {TRUNK_CLASS_ID: neighbourhood}
    # Everything within 100 m and any elevation (+-90 degrees) counts as visible.
    scene.max_range, scene.vertical_half_fov = 100.0, math.pi / 2
    return scene


def count_of(maps, *map_names):
    """Total voxels over the named entries of a ``sweep_voxel_maps`` result."""
    return sum(maps[map_name].shape[0] for map_name in map_names)


def test_maps_hold_the_gt_hit_miss_and_the_render_correct_incorrect_voxels_of_one_sweep():
    """With the lidar 1 m up, lidar-frame points (10,0,0), (9,0,0), (30,0,0) land at map (10,0,1) -> voxel
    [20,10,4] (inside the GT block: a hit, and a correct render voxel), (9,0,1) -> [18,10,4] (in the
    neighbourhood, not GT: an incorrect render voxel) and (30,0,1) -> x index 60, outside the grid
    (ignored). The 16-voxel GT block has 1 hit, so 15 are missed. Voxel centre = origin + (index + 0.5) * 0.5."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))
    maps = viewer.sweep_voxel_maps(sweep, scene, [TRUNK_CLASS_ID], use_gated=False, radius_meters=25.0)
    np.testing.assert_allclose(maps["gt_hit"], [[10.25, 0.25, 1.25]])
    np.testing.assert_allclose(maps["render_correct"], [[10.25, 0.25, 1.25]])
    np.testing.assert_allclose(maps["render_incorrect"], [[9.25, 0.25, 1.25]])
    assert maps["gt_miss"].shape[0] == GROUND_TRUTH_VOXEL_COUNT - 1
    assert maps["point_set"] == "ungated" and maps["point_count"] == 3


def test_radius_keeps_only_voxels_near_the_lidar():
    """The lidar is at map (0, 0). Within 9.5 m only the incorrect render voxel (centre distance
    hypot(9.25, 0.25) = 9.253 m) stays: the GT block centres are at x >= 9.75 m."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
    maps = viewer.sweep_voxel_maps(sweep, scene, [TRUNK_CLASS_ID], use_gated=False, radius_meters=9.5)
    assert [maps[name].shape[0] for name in ("gt_hit", "gt_miss", "render_correct", "render_incorrect")] == [0, 0, 0, 1]


def test_a_sweep_that_cannot_see_the_trunk_has_no_voxels():
    """Range limit 5 m: the trunk (about 10 m away) is outside the sweep's own scored region, so nothing is drawn."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    scene.max_range = 5.0
    maps = viewer.sweep_voxel_maps(lifted_sweep(np.array([[10.0, 0.0, 0.0]])), scene, [TRUNK_CLASS_ID], False, 25.0)
    assert count_of(maps, "gt_hit", "gt_miss", "render_correct", "render_incorrect") == 0


def test_each_pose_uses_only_its_own_returned_points():
    """Two sweeps of one run: the first returns the trunk voxel, the second returns nothing. The second
    pose's maps must not inherit the first's render voxels (no pooling over sweeps)."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    hitting = lifted_sweep(np.array([[10.0, 0.0, 0.0]]))
    silent = lifted_sweep(np.zeros((0, 3)))
    hit_maps = viewer.sweep_voxel_maps(hitting, scene, [TRUNK_CLASS_ID], False, 25.0)
    silent_maps = viewer.sweep_voxel_maps(silent, scene, [TRUNK_CLASS_ID], False, 25.0)
    assert hit_maps["gt_hit"].shape[0] == 1 and hit_maps["point_count"] == 1
    assert count_of(silent_maps, "gt_hit", "render_correct", "render_incorrect") == 0
    assert silent_maps["gt_miss"].shape[0] == GROUND_TRUTH_VOXEL_COUNT and silent_maps["point_count"] == 0


def test_every_map_shows_the_whole_rendered_cloud_with_its_own_emphasis():
    """Layers per map. Overlay = all GT (more opaque) under all render (more translucent). Recall = faint green
    for every render voxel that is not a hit, missed GT grey, hit GT solid green. Precision = faint red for every
    render voxel not on GT, correct ones solid green, no GT voxel. Scene: 16 GT voxels, one hit, one incorrect."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    maps = viewer.sweep_voxel_maps(lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]])), scene, [TRUNK_CLASS_ID], False, 25.0)
    recall = viewer.map_layers("recall", maps)
    assert [layer[0].shape[0] for layer in recall] == [1, GROUND_TRUTH_VOXEL_COUNT - 1, 1]
    assert [layer[1] for layer in recall] == [viewer.HIT_VOXEL_RGB, viewer.GROUND_TRUTH_VOXEL_RGB, viewer.HIT_VOXEL_RGB]
    assert recall[0][2] < recall[2][2], "the context must be fainter than the solid hits"
    precision = viewer.map_layers("precision", maps)
    assert [layer[0].shape[0] for layer in precision] == [1, 1]
    assert [layer[1] for layer in precision] == [viewer.NON_HIT_VOXEL_RGB, viewer.HIT_VOXEL_RGB]
    ground_truth_layer, render_layer = viewer.map_layers("overlay", maps)
    assert ground_truth_layer[0].shape[0] == GROUND_TRUTH_VOXEL_COUNT and render_layer[0].shape[0] == 2
    assert ground_truth_layer[2] > render_layer[2], "the GT must be the more opaque layer"
    with pytest.raises(ValueError, match="unknown map"):
        viewer.map_layers("nope", maps)


def test_render_voxels_outside_the_scored_boxes_are_drawn_faint_in_all_three_maps_but_never_counted():
    """A return far from every GT box lies outside the scoring neighbourhood: it is in the overlay's render layer,
    the recall map's faint green and the precision map's faint red, while the report-style totals (and so the
    captions' ratios) stay at 1 of 16 and 1 of 2."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    hit_and_incorrect = np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    far_point = np.array([[10.0, 4.0, 0.0]])  # inside the grid (y origin -5), far beyond any class margin of the boxes
    maps = viewer.sweep_voxel_maps(lifted_sweep(np.vstack([hit_and_incorrect, far_point])), scene, [TRUNK_CLASS_ID], False, 25.0)
    assert maps["render_outside"].shape[0] == 1 and maps["outside_total"] == 1 and "render_outside" not in maps["totals"]
    assert maps["totals"]["render_correct"] + maps["totals"]["render_incorrect"] == 2
    assert viewer.map_layers("overlay", maps)[1][0].shape[0] == 3
    assert viewer.map_layers("recall", maps)[0][0].shape[0] == 2  # incorrect + outside
    assert viewer.map_layers("precision", maps)[0][0].shape[0] == 2
    assert "recall 0.06: 1 of 16" in viewer.map_caption("recall", maps)
    assert "precision 0.50: 1 of 2" in viewer.map_caption("precision", maps)


def test_captions_state_recall_and_precision_from_the_voxel_counts():
    """1 of 16 GT voxels hit -> recall 0.06; 1 of 2 render voxels on GT -> precision 0.50."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    maps = viewer.sweep_voxel_maps(lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]])), scene, [TRUNK_CLASS_ID], False, 25.0)
    assert "recall 0.06: 1 of 16" in viewer.map_caption("recall", maps)
    assert "precision 0.50: 1 of 2" in viewer.map_caption("precision", maps)
    assert "16 GT voxels" in viewer.map_caption("overlay", maps) and "2 render voxels" in viewer.map_caption("overlay", maps)
    assert viewer.ratio_text(0, 0) == "n/a"


def test_within_radius_is_horizontal_and_inclusive():
    """(3, 4, z) is 5 m from the origin whatever z is; (3, 4.1) is just outside radius 5."""
    viewer = load_viewer_script()
    points = np.array([[3.0, 4.0, 100.0], [3.0, 4.1, 0.0], [0.0, 0.0, -50.0]])
    np.testing.assert_array_equal(viewer.within_radius(points, np.array([0.0, 0.0]), 5.0), [True, False, True])


def test_main_refuses_a_non_positive_radius(tmp_path, capsys):
    """--radius-meters 0 is refused before any run is read."""
    viewer = load_viewer_script()
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("eval: {render_accumulation_threshold: 0.5}\n")
    assert viewer.main(["--run", str(tmp_path / "x"), "--eval-config", str(config_path), "--radius-meters", "0"]) == 1
    assert "radius must be positive" in capsys.readouterr().err


def make_two_class_scene(viewer):
    """The voxel scene plus a rock (class 3) GT block [30:32, 9:11, 2:4] (8 voxels) whose neighbourhood also covers the trunk."""
    scene = make_voxel_scene(viewer)
    scene.ground_truth_class_ids[30:32, 9:11, 2:4] = ROCK_CLASS_ID
    rock_neighbourhood = np.zeros(GRID_SHAPE, dtype=bool)
    rock_neighbourhood[17:33, 7:13, 0:8] = True
    scene.class_neighbourhoods[ROCK_CLASS_ID] = rock_neighbourhood
    return scene


def test_two_classes_are_scored_per_class_and_a_hit_beats_another_class_false_positive():
    """Class 1 (trunk GT block) and class 3 (rock, 8 voxels) with a rock neighbourhood that also covers
    the trunk. The occupied voxel [20,10,4] is correct for class 1 and incorrect for class 3 (in the
    rock neighbourhood, not rock GT): drawn as correct. [18,10,4] is in both neighbourhoods and GT of
    neither: one incorrect voxel. Missed GT is the trunk's other 15 voxels plus the rock's 8."""
    viewer = load_viewer_script()
    scene = make_two_class_scene(viewer)
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
    maps = viewer.sweep_voxel_maps(sweep, scene, [TRUNK_CLASS_ID, ROCK_CLASS_ID], False, 25.0)
    np.testing.assert_allclose(maps["render_correct"], [[10.25, 0.25, 1.25]])
    np.testing.assert_allclose(maps["render_incorrect"], [[9.25, 0.25, 1.25]])
    assert maps["gt_miss"].shape[0] == (GROUND_TRUTH_VOXEL_COUNT - 1) + 2 * 2 * 2


def test_the_class_filter_removes_the_other_classes_gt_and_render_voxels():
    """Selecting only the rock: the trunk's GT, its hit and the render voxels in the trunk neighbourhood
    that are not in the rock's disappear from every map; the rock's own 8 GT voxels are all missed. The
    rock neighbourhood covers [17:33, 7:13, 0:8], so the two rendered voxels are incorrect rock-class ones."""
    viewer = load_viewer_script()
    scene = make_two_class_scene(viewer)
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
    maps = viewer.sweep_voxel_maps(sweep, scene, [ROCK_CLASS_ID], False, 25.0)
    assert maps["gt_hit"].shape[0] == 0 and maps["gt_miss"].shape[0] == 8
    assert maps["render_correct"].shape[0] == 0 and maps["render_incorrect"].shape[0] == 2
    trunk_only = viewer.sweep_voxel_maps(sweep, scene, [TRUNK_CLASS_ID], False, 25.0)
    assert trunk_only["gt_miss"].shape[0] == GROUND_TRUTH_VOXEL_COUNT - 1


EVAL_SETTINGS = {"scored_class_ids": {"tree_trunk": 1, "rock": 3}}


def test_class_selection_defaults_to_all_scored_then_the_config_then_the_flag():
    """Flag beats config beats all; an unknown name or an empty list is a refusal."""
    viewer = load_viewer_script()
    assert viewer.resolve_selected_class_ids(EVAL_SETTINGS, None) == {1: "tree_trunk", 3: "rock"}
    assert viewer.resolve_selected_class_ids({**EVAL_SETTINGS, "viz_classes": ["rock"]}, None) == {3: "rock"}
    assert viewer.resolve_selected_class_ids({**EVAL_SETTINGS, "viz_classes": ["rock"]}, ["tree_trunk"]) == {1: "tree_trunk"}
    with pytest.raises(ValueError, match="not scored classes"):
        viewer.resolve_selected_class_ids(EVAL_SETTINGS, ["tree_trunk", "bush"])
    with pytest.raises(ValueError, match="not scored classes"):
        viewer.resolve_selected_class_ids(EVAL_SETTINGS, [])


def test_main_refuses_an_unknown_class(tmp_path, capsys):
    """--classes with a name that is not scored is refused before any run is read."""
    viewer = load_viewer_script()
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("eval: {render_accumulation_threshold: 0.5, viz_radius_meters: 25.0, scored_class_ids: {tree_trunk: 1}}\n")
    assert viewer.main(["--run", str(tmp_path / "x"), "--eval-config", str(config_path), "--classes", "oak"]) == 1
    assert "not scored classes" in capsys.readouterr().err


def test_pose_request_with_a_non_ascii_digit_is_a_clean_refusal():
    """'²' is str.isdigit() but not an index: it matches no sweep instead of raising from int()."""
    viewer = load_viewer_script()
    with pytest.raises(ValueError, match="matches no shared sweep"):
        viewer.select_pose_stems([make_scene(viewer, ["000005_1000"])], ["²"])


def test_a_failing_pose_is_reported_and_the_other_poses_are_still_drawn(tmp_path, monkeypatch, capsys):
    """render_pose raising for one pose: main prints 'failed <stem>', still tries the next pose, exits 1."""
    viewer = load_viewer_script()
    scene = make_scene(viewer, ["000005_1000", "000009_2000"])
    monkeypatch.setattr(viewer, "load_run_scene", lambda *arguments: scene)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("eval: {render_accumulation_threshold: 0.5, viz_radius_meters: 25.0, scored_class_ids: {tree_trunk: 1}}\n")
    attempted_stems = []

    def fake_render_pose(scenes, stem, *remaining_arguments):
        attempted_stems.append(stem)
        if stem == "000005_1000":
            raise RuntimeError("no OpenGL")

    monkeypatch.setattr(viewer, "render_pose", fake_render_pose)
    exit_code = viewer.main(["--run", str(tmp_path / "r"), "--eval-config", str(config_path),
                             "--output-dir", str(tmp_path / "out"), "--pose", "5", "--pose", "9"])
    assert exit_code == 1 and attempted_stems == ["000005_1000", "000009_2000"]
    assert "failed 000005_1000" in capsys.readouterr().err


def test_pooled_maps_are_the_union_over_sweeps_like_the_report():
    """One sweep returns the trunk voxel [20,10,4], the other the neighbourhood voxel [18,10,4]. Pooled, both
    are render voxels (1 correct, 1 incorrect) and the GT voxel is hit; each sweep alone sees only its own voxel.
    The point count is the sum over the sweeps."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    on_trunk = lifted_sweep(np.array([[10.0, 0.0, 0.0]]))
    beside_trunk = lifted_sweep(np.array([[9.0, 0.0, 0.0]]))
    scene.sweeps = {"a": on_trunk, "b": beside_trunk}
    pooled = viewer.pooled_voxel_maps([on_trunk, beside_trunk], scene, [TRUNK_CLASS_ID], False, 25.0)
    assert pooled["gt_hit"].shape[0] == 1 and pooled["render_correct"].shape[0] == 1
    assert pooled["render_incorrect"].shape[0] == 1 and pooled["point_count"] == 2 and pooled["sweep_count"] == 2
    assert pooled["totals"] == {"gt_hit": 1, "gt_miss": GROUND_TRUTH_VOXEL_COUNT - 1, "render_correct": 1, "render_incorrect": 1}
    alone = viewer.pooled_voxel_maps([beside_trunk], scene, [TRUNK_CLASS_ID], False, 25.0)
    assert alone["gt_hit"].shape[0] == 0 and alone["render_incorrect"].shape[0] == 1


def test_pooled_maps_never_mix_point_sets():
    """If any pooled sweep lacks the ungated set, all sweeps use the gated one (and the caption says so)."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    with_ungated = lifted_sweep(np.array([[10.0, 0.0, 0.0]]))
    without_ungated = lifted_sweep(None)
    scene.sweeps = {"a": with_ungated, "b": without_ungated}
    maps = viewer.pooled_voxel_maps([with_ungated, without_ungated], scene, [TRUNK_CLASS_ID], False, 25.0)
    assert maps["point_set"] == "gated" and maps["point_count"] == 2


def test_caption_counts_the_whole_region_not_only_the_drawn_radius():
    """Radius 9.5 m leaves only the incorrect voxel drawn, but recall/precision still count every voxel."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
    maps = viewer.sweep_voxel_maps(sweep, scene, [TRUNK_CLASS_ID], False, 9.5)
    assert maps["gt_hit"].shape[0] == 0
    assert "recall 0.06: 1 of 16" in viewer.map_caption("recall", maps)
    assert "precision 0.50: 1 of 2" in viewer.map_caption("precision", maps)


def test_within_radius_of_any_is_the_union_of_discs():
    """A point is kept if it is within the radius of at least one centre."""
    viewer = load_viewer_script()
    points = np.array([[1.0, 0.0, 0.0], [11.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    centres = np.array([[0.0, 0.0], [10.0, 0.0]])
    np.testing.assert_array_equal(viewer.within_radius_of_any(points, centres, 2.0), [True, True, False])


def test_main_draws_the_pooled_maps_by_default_and_refuses_points_without_a_pose(tmp_path, monkeypatch, capsys):
    """No --pose: one render_pose call with the pooled stem. --show-points needs a single sweep: refused."""
    viewer = load_viewer_script()
    scene = make_scene(viewer, ["000005_1000", "000009_2000"])
    monkeypatch.setattr(viewer, "load_run_scene", lambda *arguments: scene)
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("eval: {render_accumulation_threshold: 0.5, viz_radius_meters: 25.0, scored_class_ids: {tree_trunk: 1}}\n")
    drawn_stems = []
    monkeypatch.setattr(viewer, "render_pose", lambda scenes, stem, *remaining: drawn_stems.append(stem))
    base = ["--run", str(tmp_path / "r"), "--eval-config", str(config_path), "--output-dir", str(tmp_path / "out")]
    assert viewer.main(base) == 0 and drawn_stems == [viewer.POOLED_STEM]
    for points_flag in ("--show-points", "--all-points"):
        assert viewer.main(base + [points_flag]) == 1 and "give --pose" in capsys.readouterr().err
