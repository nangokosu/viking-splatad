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


def test_voxel_cells_are_the_metrics_confusion_cells_of_one_sweep():
    """With the lidar 1 m up, lidar-frame points (10,0,0), (9,0,0), (30,0,0) land at map (10,0,1) -> voxel
    [20,10,4] (inside the GT block: true positive), (9,0,1) -> [18,10,4] (in the neighbourhood, not GT: false
    positive) and (30,0,1) -> x index 60, outside the grid (ignored). The 16-voxel GT block has 1 hit,
    so 15 are missed. Voxel centre = origin + (index + 0.5) * 0.5."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0], [30.0, 0.0, 0.0]]))
    cells = viewer.sweep_voxel_cells(sweep, scene, use_gated=False, radius_meters=25.0)
    np.testing.assert_allclose(cells["true_positive"], [[10.25, 0.25, 1.25]])
    np.testing.assert_allclose(cells["false_positive"], [[9.25, 0.25, 1.25]])
    assert cells["false_negative"].shape[0] == GROUND_TRUTH_VOXEL_COUNT - 1
    assert cells["point_set"] == "ungated"
    assert viewer.cell_counts_and_iou(cells) == (1, 1, GROUND_TRUTH_VOXEL_COUNT - 1, pytest.approx(1 / 17))


def test_radius_keeps_only_voxels_near_the_lidar():
    """The lidar is at map (0, 0). Within 9.5 m only the false-positive voxel (centre distance
    hypot(9.25, 0.25) = 9.253 m) stays: the GT block centres are at x >= 9.75 m."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
    cells = viewer.sweep_voxel_cells(sweep, scene, use_gated=False, radius_meters=9.5)
    assert [cells[name].shape[0] for name in ("true_positive", "false_positive", "false_negative")] == [0, 1, 0]


def test_a_sweep_that_cannot_see_the_trunk_has_no_cells():
    """Range limit 5 m: the trunk (about 10 m away) is outside the sweep's own scored region, so nothing is drawn."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    scene.max_range = 5.0
    cells = viewer.sweep_voxel_cells(lifted_sweep(np.array([[10.0, 0.0, 0.0]])), scene, False, 25.0)
    assert all(cells[name].shape[0] == 0 for name in ("true_positive", "false_positive", "false_negative"))
    assert viewer.cell_counts_and_iou(cells) == (0, 0, 0, None)


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


def test_two_classes_are_scored_per_class_and_a_hit_beats_another_class_false_positive():
    """Class 1 (trunk GT block [19:21, 9:11, 2:6]) and class 3 (rock GT block [30:32, 9:11, 2:4], 8 voxels)
    with a class-3 neighbourhood that also covers the trunk. The occupied voxel [20,10,4] is a hit for
    class 1 and a false positive for class 3 (it is in the rock neighbourhood, not rock GT): drawn as a
    hit. [18,10,4] is in both neighbourhoods and GT of neither: one false positive. Missed GT is the
    trunk's other 15 voxels plus the rock's 8."""
    viewer = load_viewer_script()
    scene = make_voxel_scene(viewer)
    rock_class_id = 3
    scene.ground_truth_class_ids[30:32, 9:11, 2:4] = rock_class_id
    rock_neighbourhood = np.zeros(GRID_SHAPE, dtype=bool)
    rock_neighbourhood[17:33, 7:13, 0:8] = True
    scene.class_neighbourhoods[rock_class_id] = rock_neighbourhood
    sweep = lifted_sweep(np.array([[10.0, 0.0, 0.0], [9.0, 0.0, 0.0]]))
    cells = viewer.sweep_voxel_cells(sweep, scene, use_gated=False, radius_meters=25.0)
    np.testing.assert_allclose(cells["true_positive"], [[10.25, 0.25, 1.25]])
    np.testing.assert_allclose(cells["false_positive"], [[9.25, 0.25, 1.25]])
    assert cells["false_negative"].shape[0] == (GROUND_TRUTH_VOXEL_COUNT - 1) + 2 * 2 * 2


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
    config_path.write_text("eval: {render_accumulation_threshold: 0.5, viz_radius_meters: 25.0}\n")
    attempted_stems = []

    def fake_render_pose(scenes, stem, *remaining_arguments):
        attempted_stems.append(stem)
        if stem == "000005_1000":
            raise RuntimeError("no OpenGL")

    monkeypatch.setattr(viewer, "render_pose", fake_render_pose)
    exit_code = viewer.main(["--run", str(tmp_path / "r"), "--eval-config", str(config_path),
                             "--output-dir", str(tmp_path / "out")])
    assert exit_code == 1 and attempted_stems == ["000005_1000", "000009_2000"]
    assert "failed 000005_1000" in capsys.readouterr().err
