"""Tests of the pure logic in scripts/15_view_pose_pyvista.py on hand-built sweeps (no PyVista needed)."""

import importlib.util
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
