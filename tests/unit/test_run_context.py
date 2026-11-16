"""Tests of viking_splatad.eval.run_context (run resolution shared by scripts/13 and scripts/15), synthetic known answers."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from viking_splatad.eval.rendered_pointcloud import RenderedSweep
from viking_splatad.eval.run_context import (
    CUBOIDS_FILE_NAME,
    DATASETS_RELATIVE_DIR,
    JOB_INFO_FILE_NAME,
    LIDAR_RENDER_SUBDIR,
    LIDAR_TEST_RENDER_CHECK_KEY,
    MANIFEST_FILE_NAME,
    MINIMUM_POSSIBLE_POINT_COUNT,
    OCCUPANCY_FILE_NAME,
    RERENDER_CHECK_MATCH_LINE,
    RERENDERS_DIRECTORY_NAME,
    SELECTION_FILE_NAME,
    SESSION_CONFIGS_RELATIVE_DIR,
    SPLIT_RECORD_FILE_NAME,
    build_class_setup,
    file_sha256,
    lidar_test_stems,
    load_ground_truth,
    pooled_map_points,
    resolve_render_target,
    resolve_session,
)
from viking_splatad.labels.cuboids import Cuboid

# Synthetic class ids of the two test classes (any distinct non-zero ids; 0 is the empty voxel).
TRUNK_CLASS_ID, ROCK_CLASS_ID = 1, 3
# Identity quaternion (x, y, z, w): an axis-aligned box.
IDENTITY_QUATERNION_XYZW = (0.0, 0.0, 0.0, 1.0)
# Synthetic split record: two test lidar frames, one train lidar frame, one camera frame.
SPLIT_FRAMES = [
    {"kind": "lidar", "split": "test", "frame_index": 7, "file": "lidar/ouster/000007_700.npy"},
    {"kind": "lidar", "split": "train", "frame_index": 3, "file": "lidar/ouster/000003_300.npy"},
    {"kind": "lidar", "split": "test", "frame_index": 2, "file": "lidar/ouster/000002_200.npy"},
    {"kind": "camera", "split": "test", "frame_index": 2, "file": "images/front/000002_200.jpg"},
]
# FIPS 180-2 Appendix B.1 test vector: SHA-256("abc"). A published constant, the only way to check
# the digest without recomputing it with the same library call under test.
FIPS_ABC_MESSAGE = b"abc"
FIPS_ABC_SHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def make_split_record(render_match: bool = True) -> dict:
    """Synthetic split.json with SPLIT_FRAMES.

    Args:
        render_match: Value of the lidar test render check.

    Returns:
        The parsed record.
    """
    return {
        "run": {"run_id": "run-x"}, "dataset": {"name": "set_x", "manifest_sha256": "unused"},
        "render_check": {LIDAR_TEST_RENDER_CHECK_KEY: {"match": render_match}}, "frames": SPLIT_FRAMES,
    }


def stem_of(frame: dict) -> str:
    """File stem (``<idx>_<ts_ns>``) of a split frame entry.

    Args:
        frame: One split.json frame.

    Returns:
        Stem.
    """
    return Path(frame["file"]).stem


def write_run(run_dir: Path, split_record: dict, with_renders: bool = True) -> None:
    """Write a run folder: split.json and, optionally, an (empty) lidar render folder.

    Args:
        run_dir: Run folder to create.
        split_record: Record to write as split.json.
        with_renders: Create ``renders/test/lidar``.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / SPLIT_RECORD_FILE_NAME).write_text(json.dumps(split_record))
    if with_renders:
        (run_dir / LIDAR_RENDER_SUBDIR).mkdir(parents=True)


def write_ground_truth(gt_dir: Path, cuboid_entries: list, class_ids: np.ndarray, origin: np.ndarray, voxel_size: float) -> None:
    """Write cuboids.json and occupancy.npz as scripts/05 does.

    Args:
        gt_dir: GT folder to create.
        cuboid_entries: cuboids.json entries.
        class_ids: Occupancy class grid.
        origin: (3,) grid minimum corner.
        voxel_size: Voxel edge (m).
    """
    gt_dir.mkdir(parents=True, exist_ok=True)
    (gt_dir / CUBOIDS_FILE_NAME).write_text(json.dumps(cuboid_entries))
    np.savez(gt_dir / OCCUPANCY_FILE_NAME, class_ids=class_ids, origin=origin, voxel_size=np.float64(voxel_size))


# ---------------------------------------------------------------- file_sha256 / lidar_test_stems


def test_file_sha256_matches_the_fips_test_vector(tmp_path):
    """A file holding "abc" hashes to the FIPS 180-2 SHA-256 digest of "abc"."""
    message_path = tmp_path / "abc.bin"
    message_path.write_bytes(FIPS_ABC_MESSAGE)
    assert file_sha256(message_path) == FIPS_ABC_SHA256


def test_lidar_test_stems_pick_the_test_lidar_frames_sorted():
    """Default: the two test lidar frames (not the train lidar or the test camera), sorted by stem."""
    expected_stems = sorted(stem_of(frame) for frame in SPLIT_FRAMES if frame["kind"] == "lidar" and frame["split"] == "test")
    assert lidar_test_stems(make_split_record()) == expected_stems


def test_lidar_test_stems_with_a_selection_ignore_the_split():
    """A render-only selection names lidar frame indices: the train frame is chosen, the camera frame never."""
    selected_indices = [3, 2]
    expected_stems = sorted(stem_of(frame) for frame in SPLIT_FRAMES
                            if frame["kind"] == "lidar" and frame["frame_index"] in selected_indices)
    assert lidar_test_stems(make_split_record(), selected_indices) == expected_stems


# ---------------------------------------------------------------- resolve_render_target


def test_resolve_render_target_uses_the_run_renders_and_the_test_split(tmp_path):
    """Own renders: eval root is the run, render dir its renders/test/lidar, stems the test lidar frames."""
    run_dir = tmp_path / "run"
    split_record = make_split_record()
    write_run(run_dir, split_record)

    eval_root, render_dir, expected_stems, loaded_record = resolve_render_target(run_dir, None)

    assert eval_root == run_dir and render_dir == run_dir / LIDAR_RENDER_SUBDIR
    assert expected_stems == lidar_test_stems(split_record) and loaded_record == split_record


def test_resolve_render_target_reads_a_rerender_selection(tmp_path):
    """Render-only job: eval root rerenders/<label>, stems from its selection.json."""
    run_dir, rerender_label, selected_indices = tmp_path / "run", "render-a", [3]
    split_record = make_split_record()
    write_run(run_dir, split_record, with_renders=False)
    rerender_root = run_dir / RERENDERS_DIRECTORY_NAME / rerender_label
    (rerender_root / LIDAR_RENDER_SUBDIR).mkdir(parents=True)
    (rerender_root / JOB_INFO_FILE_NAME).write_text(f"job: 1\n{RERENDER_CHECK_MATCH_LINE}\n")
    (rerender_root / SELECTION_FILE_NAME).write_text(json.dumps({"lidar": selected_indices}))

    eval_root, render_dir, expected_stems, _ = resolve_render_target(run_dir, rerender_label)

    assert eval_root == rerender_root and render_dir == rerender_root / LIDAR_RENDER_SUBDIR
    assert expected_stems == lidar_test_stems(split_record, selected_indices)


def test_resolve_render_target_refusals(tmp_path):
    """No split.json, a failed render check, a rerender without its match line, and no renders are refused."""
    with pytest.raises(ValueError, match="missing"):
        resolve_render_target(tmp_path / "absent", None)
    mismatched_run = tmp_path / "mismatch"
    write_run(mismatched_run, make_split_record(render_match=False))
    with pytest.raises(ValueError, match="do not match"):
        resolve_render_target(mismatched_run, None)
    unchecked_run, rerender_label = tmp_path / "unchecked", "render-b"
    write_run(unchecked_run, make_split_record(), with_renders=False)
    (unchecked_run / RERENDERS_DIRECTORY_NAME / rerender_label).mkdir(parents=True)
    (unchecked_run / RERENDERS_DIRECTORY_NAME / rerender_label / JOB_INFO_FILE_NAME).write_text("render_check: mismatch\n")
    with pytest.raises(ValueError, match=RERENDER_CHECK_MATCH_LINE):
        resolve_render_target(unchecked_run, rerender_label)
    renderless_run = tmp_path / "renderless"
    write_run(renderless_run, make_split_record(), with_renders=False)
    with pytest.raises(ValueError, match="no lidar renders"):
        resolve_render_target(renderless_run, None)


# ---------------------------------------------------------------- resolve_session


def write_session(project_root: Path, session_name: str, dataset_name: str) -> Path:
    """Minimal session YAML and dataset manifest naming that session.

    Args:
        project_root: Synthetic repo root.
        session_name: Session the manifest names.
        dataset_name: Dataset folder name (differs from the session, like a smoke dataset).

    Returns:
        The manifest path.
    """
    session_yaml = {key: {} for key in ("window", "topics", "frames", "camera", "lidar", "radar", "split", "splatad")}
    session_yaml["session"] = {
        "name": session_name, "extrinsics_variant": "x", "bag_dir": str(project_root), "calibration_dir": str(project_root),
        "labels_json": str(project_root), "labels_sample_key": "sample_x",
        "output_dir": str(project_root), "gt_dir": str(project_root / "data" / "gt" / session_name),
    }
    (project_root / SESSION_CONFIGS_RELATIVE_DIR).mkdir(parents=True)
    (project_root / SESSION_CONFIGS_RELATIVE_DIR / f"{session_name}.yaml").write_text(yaml.safe_dump(session_yaml))
    manifest_path = project_root / DATASETS_RELATIVE_DIR / dataset_name / MANIFEST_FILE_NAME
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps({"session": {"name": session_name}}))
    return manifest_path


def test_resolve_session_follows_the_manifest_session_and_refuses_another_dataset(tmp_path):
    """The manifest's session (not the dataset name) picks the config; a different manifest hash is refused."""
    session_name, dataset_name = "may_x", "may_x_smoke"
    manifest_path = write_session(tmp_path, session_name, dataset_name)
    split_record = {"dataset": {"name": dataset_name,
                                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}}

    session_config, manifest = resolve_session(split_record, tmp_path)

    assert session_config["session"]["name"] == session_name and manifest == json.loads(manifest_path.read_text())
    split_record["dataset"]["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes() + b" ").hexdigest()
    with pytest.raises(ValueError, match="differs"):
        resolve_session(split_record, tmp_path)


# ---------------------------------------------------------------- load_ground_truth


def test_load_ground_truth_round_trips_and_refuses_a_missing_gt(tmp_path):
    """Cuboids and every occupancy array come back as written; an empty GT folder is refused."""
    cuboid_entry = {"box_id": "t", "class_id": TRUNK_CLASS_ID, "class_name": "tree_trunk", "center_xyz": [1.0, 2.0, 3.0],
                    "dimensions_xyz": [0.5, 0.5, 2.0], "quaternion_xyzw": list(IDENTITY_QUATERNION_XYZW)}
    grid_shape, voxel_size, origin = (2, 3, 4), 0.25, np.array([-1.0, -2.0, -3.0])
    class_ids = np.zeros(grid_shape, dtype=np.uint8)
    class_ids[1, 2, 3] = TRUNK_CLASS_ID
    write_ground_truth(tmp_path / "gt", [cuboid_entry], class_ids, origin, voxel_size)

    cuboids, occupancy = load_ground_truth(tmp_path / "gt")

    assert cuboids == [Cuboid("t", TRUNK_CLASS_ID, "tree_trunk", (1.0, 2.0, 3.0), (0.5, 0.5, 2.0), IDENTITY_QUATERNION_XYZW)]
    assert set(occupancy) == {"class_ids", "origin", "voxel_size"}
    np.testing.assert_array_equal(occupancy["class_ids"], class_ids)
    np.testing.assert_array_equal(occupancy["origin"], origin)
    assert float(occupancy["voxel_size"]) == voxel_size
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="GT missing"):
        load_ground_truth(tmp_path / "empty")


# ---------------------------------------------------------------- pooled_map_points


def translation_pose(translation_xyz) -> np.ndarray:
    """4x4 pose with identity rotation and the given translation.

    Args:
        translation_xyz: (3,) translation.

    Returns:
        (4, 4) pose.
    """
    pose = np.eye(4)
    pose[:3, 3] = translation_xyz
    return pose


def test_pooled_map_points_pools_sweeps_in_the_map_frame():
    """Two sweeps translated by (1, 0, 0) and (0, 2, 0), 1 m voxels from the origin, grid 4 x 4 x 2.

    Sweep A real point (0.5, 0.5, 0.5) -> map (1.5, 0.5, 0.5) -> voxel (1, 0, 0).
    Sweep B real points (0.5, 0.5, 0.5) -> (0.5, 2.5, 0.5) -> voxel (0, 2, 0); (10, 0, 0) -> (10, 2, 0): outside.
    Pooled: 3 points in sweep order, 2 occupied voxels, 1 outside the grid. The gated set is pooled
    from the gated attribute (one point per sweep), not the real one.
    """
    grid_shape, voxel_size = (4, 4, 2), 1.0
    grid = {"class_ids": np.zeros(grid_shape, dtype=np.uint8), "origin": np.zeros(3), "voxel_size": np.float64(voxel_size)}
    first_translation, second_translation = np.array([1.0, 0.0, 0.0]), np.array([0.0, 2.0, 0.0])
    first_real = np.array([[0.5, 0.5, 0.5]])
    second_real = np.array([[0.5, 0.5, 0.5], [10.0, 0.0, 0.0]])
    single_point = np.array([[0.5, 0.5, 0.5]])
    sweeps = [
        RenderedSweep("a", translation_pose(first_translation), single_point, None, first_real),
        RenderedSweep("b", translation_pose(second_translation), single_point, None, second_real),
    ]

    pooled = pooled_map_points(sweeps, "real", grid)

    expected_points = np.concatenate([first_real + first_translation, second_real + second_translation])
    np.testing.assert_allclose(pooled["points"], expected_points)
    expected_occupied = np.zeros(grid_shape, dtype=bool)
    expected_occupied[1, 0, 0] = expected_occupied[0, 2, 0] = True
    np.testing.assert_array_equal(pooled["occupied"], expected_occupied)
    assert pooled["outside_grid"] == 1
    assert pooled_map_points(sweeps, "gated", grid)["points"].shape[0] == len(sweeps)


# ---------------------------------------------------------------- build_class_setup


def test_build_class_setup_known_neighbourhood_threshold_and_gt():
    """1 m grid from the origin, 6 x 6 x 4; trunk box centre (3, 3, 2), 2 x 2 x 2 m, margin 1 m.

    Grown box: x, y in [1, 5], z in [1, 3] -> voxel centres 1.5..4.5 (4 per horizontal axis) and
    1.5, 2.5 (2 vertically) -> 4 * 4 * 2 = 32 neighbourhood voxels, all visible from a pose at the box
    centre (range = grid diagonal, half FOV 90°). GT: the box's 2 * 2 * 2 voxels plus one trunk voxel
    at (0, 0, 0) outside the neighbourhood -> 8 GT voxels. Real hits: 3 points in the box -> n_min 3.
    The rock (no box, margin None) gets margin 0, an empty neighbourhood and the minimum n_min.
    """
    grid_shape, voxel_size, origin = (6, 6, 4), 1.0, np.zeros(3)
    box_center, box_dimensions, margin = (3.0, 3.0, 2.0), (2.0, 2.0, 2.0), 1.0
    trunk_box = Cuboid("t", TRUNK_CLASS_ID, "tree_trunk", box_center, box_dimensions, IDENTITY_QUATERNION_XYZW)
    class_ids = np.zeros(grid_shape, dtype=np.uint8)
    class_ids[2:4, 2:4, 1:3] = TRUNK_CLASS_ID
    class_ids[0, 0, 0] = TRUNK_CLASS_ID
    grid = {"class_ids": class_ids, "origin": origin, "voxel_size": np.float64(voxel_size)}
    real_points = np.array([[3.0, 3.0, 2.0], [2.5, 3.5, 1.5], [3.5, 2.5, 2.5], [0.2, 0.2, 0.2]])
    max_range = float(np.linalg.norm(np.asarray(grid_shape) * voxel_size))
    sensor_pose = translation_pose(box_center)

    class_setup = build_class_setup(
        {"tree_trunk": TRUNK_CLASS_ID, "rock": ROCK_CLASS_ID}, {TRUNK_CLASS_ID: [trunk_box], ROCK_CLASS_ID: []},
        {TRUNK_CLASS_ID: margin, ROCK_CLASS_ID: None}, real_points, grid, [sensor_pose], max_range, math.pi / 2, 0.25)

    trunk_setup, rock_setup = class_setup["tree_trunk"], class_setup["rock"]
    expected_neighbourhood = np.zeros(grid_shape, dtype=bool)
    expected_neighbourhood[1:5, 1:5, 1:3] = True
    np.testing.assert_array_equal(trunk_setup["neighbourhood"], expected_neighbourhood)
    expected_ground_truth = np.zeros(grid_shape, dtype=bool)
    expected_ground_truth[2:4, 2:4, 1:3] = True
    np.testing.assert_array_equal(trunk_setup["ground_truth"], expected_ground_truth)
    assert (trunk_setup["class_id"], trunk_setup["margin"], trunk_setup["n_min"]) == (TRUNK_CLASS_ID, margin, 3)
    assert trunk_setup["n_min_source"].startswith("quantile")
    assert (rock_setup["margin"], rock_setup["n_min"]) == (0.0, MINIMUM_POSSIBLE_POINT_COUNT)
    assert not rock_setup["neighbourhood"].any() and not rock_setup["ground_truth"].any()
