"""End-to-end tests of the corridor crop (T87) and the viewer frame (T88) in scripts/22 and scripts/23.

A tiny synthetic run (the checkpoint of ``test_export_supersplat_script``) sits over a tiny synthetic lidar
dataset whose train path and returns are chosen here, so every expected value is computed in the test from
first principles: the corridor from the path and the width, the viewer frame from p' = M (p - o).
"""

import importlib.util
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("torch")
pytest.importorskip("nerfstudio.models.splatad")

from tests.unit.test_export_corridor_crop import build_dataset
from tests.unit.test_export_supersplat_script import (
    CHECKPOINT_FILE_NAME, CONFIG_MIN_OPACITY, DIRECTION_TEST_WEIGHT_STD, FLOAT32_ROUNDOFF_ATOL, GAUSSIAN_COUNT, PLY_RELATIVE_PATH,
    build_synthetic_run, load_script_module, map_means_of, read_ply_coefficients)
from viking_splatad.export.spherical_harmonics import real_sh_basis
from viking_splatad.export.supersplat_ply import read_supersplat_ply
from viking_splatad.export.viewer_frame import (
    MAP_FRAME_NAME, SUPERSPLAT_FROM_MAP_ROTATION, VIEWER_FRAME_FILE_NAME, VIEWER_FRAME_NAME,
    read_viewer_frame_record, viewer_to_map_points, write_viewer_frame_record)

# Sensor-frame offset of every synthetic lidar return from its own sensor origin: a point at (R, 0, z), so
# its horizontal range from the origin is exactly R. R is a multiple of 1/64, exact in float32 and float64.
RANGE_GRID_M = 1.0 / 64.0
# Path points are placed on this grid (exact in float64, so the corridor test has no round-off on the path).
PATH_GRID_M = 0.5
# Half-distance (m) of the two path points from the centre of the Gaussians' horizontal spread.
PATH_HALF_SPAN_M = 2.0
# z margin (m) beyond the Gaussians' z extremes, so the corridor z range keeps every Gaussian in z and the
# test isolates the horizontal distance.
Z_MARGIN_M = 1.0
# Sensor height (m) of the synthetic path: non-zero so that the viewer origin has a z component to remove.
SENSOR_HEIGHT_M = 1.5
# Minimum gap (m) required between the nearest kept and the nearest dropped Gaussian, vs float32 round-off
# of the means (~1e-5 m at magnitude ~60 m).
MIN_CORRIDOR_GAP_M = 0.05
# Round-off budget for the quaternion/covariance and the SH comparisons: 64 float32 epsilons.
COMPARISON_ATOL = 64 * np.finfo(np.float32).eps
# SH degree of the frame test: 1 is the lowest degree with a direction dependence (it spans {1, x, y, z}).
SH_TEST_DEGREE = 1
# Directions drawn to compare the two exports' colour functions, and their seed.
SH_TEST_DIRECTION_COUNT = 16
SH_TEST_SEED = 7
# The wrong frame must be wrong by at least this many tolerances, so the test can tell frames apart.
SH_TEST_MIN_SEPARATION = 10.0


def build_corridor_run(scratch_root: Path, decoder_weight_std=None):
    """A synthetic run over a synthetic dataset with a two-point path and a corridor that cuts the Gaussians.

    The corridor width R is the midpoint between the third and fourth smallest horizontal distance of the
    map-frame means to the path, rounded to a 1/64 m grid: some Gaussians are inside, some outside.

    Args:
        scratch_root: Empty scratch folder.
        decoder_weight_std: Passed to ``build_synthetic_run`` (None = default decoder).

    Returns:
        Dict with ``dataset_dir, run_dir, truth, map_means, path_xy, width_m, z_lower, z_upper,
        origins (2, 3)`` (sensor origins in the map frame).
    """
    # Step 1: probe run -> the map-frame means (they depend only on the seed).
    probe_dir = scratch_root / "probe"
    probe_dir.mkdir()
    map_means = map_means_of(build_synthetic_run(probe_dir, zero_decoder=False))
    # Step 2: two path points either side of the horizontal centre of the means, on the path grid.
    centre_xy = map_means[:, :2].mean(axis=0)
    path_xy = np.stack([np.round((centre_xy + sign * np.array([PATH_HALF_SPAN_M, 0.0])) / PATH_GRID_M)
                        * PATH_GRID_M for sign in (-1.0, 1.0)])
    # Step 3: distance of every mean to its nearest path point, brute force (N x 2 distances).
    nearest_distance = np.linalg.norm(map_means[:, None, :2] - path_xy[None, :, :], axis=2).min(axis=1)
    sorted_distance = np.sort(nearest_distance)
    cut_index = GAUSSIAN_COUNT // 2
    gap = sorted_distance[cut_index] - sorted_distance[cut_index - 1]
    assert gap > 2 * MIN_CORRIDOR_GAP_M, "synthetic means leave no clear corridor cut"
    width_m = np.round(sorted_distance[cut_index - 1:cut_index + 1].mean() / RANGE_GRID_M) * RANGE_GRID_M
    assert (np.abs(sorted_distance - width_m) > MIN_CORRIDOR_GAP_M / 2).all()
    # Step 4: sweeps with returns at (width, 0, z) in the sensor frame: every horizontal range is exactly the
    # width, and the z range covers the means with a margin.
    origins = np.column_stack([path_xy, np.full(len(path_xy), SENSOR_HEIGHT_M)])
    z_lower = float(map_means[:, 2].min() - Z_MARGIN_M)
    z_upper = float(map_means[:, 2].max() + Z_MARGIN_M)
    sweeps = [[(width_m, 0.0, z_lower - SENSOR_HEIGHT_M), (width_m, 0.0, z_upper - SENSOR_HEIGHT_M)]
              for _origin in origins]
    poses = np.tile(np.eye(4), (len(origins), 1, 1))
    poses[:, :3, 3] = origins
    real_root = scratch_root / "real"
    real_root.mkdir()
    dataset_dir, run_dir = build_dataset(real_root, sweeps, poses)
    truth = build_synthetic_run(run_dir, zero_decoder=decoder_weight_std is None,
                                decoder_weight_std=decoder_weight_std)
    return {"dataset_dir": dataset_dir, "run_dir": run_dir, "truth": truth, "map_means": map_means,
            "path_xy": path_xy, "width_m": float(width_m), "z_lower": z_lower, "z_upper": z_upper,
            "origins": origins}


def expected_corridor_keep(run: dict) -> np.ndarray:
    """Keep mask of the corridor crop by its definition: nearest path distance <= width, z in range, opaque.

    Args:
        run: Output of ``build_corridor_run``.

    Returns:
        (N,) bool mask.
    """
    means = run["map_means"]
    nearest_distance = np.linalg.norm(means[:, None, :2] - run["path_xy"][None, :, :], axis=2).min(axis=1)
    activated = 1.0 / (1.0 + np.exp(-run["truth"]["logits"].astype(np.float64)))
    return ((nearest_distance <= run["width_m"]) & (means[:, 2] >= run["z_lower"])
            & (means[:, 2] <= run["z_upper"]) & (activated >= CONFIG_MIN_OPACITY))


def ply_means(ply: dict) -> np.ndarray:
    """(N, 3) positions of a read PLY.

    Args:
        ply: Output of ``read_supersplat_ply``.

    Returns:
        float64 positions.
    """
    return np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float64)


def ply_covariances(ply: dict) -> np.ndarray:
    """(N, 3, 3) covariances R diag(s^2) R^T of a read PLY (quaternions are w, x, y, z; scales are log std).

    Args:
        ply: Output of ``read_supersplat_ply``.

    Returns:
        float64 covariances.
    """
    quaternions_wxyz = np.stack([ply[f"rot_{part}"] for part in range(4)], axis=1).astype(np.float64)
    # scipy wants (x, y, z, w).
    rotations = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    variances = np.exp(2.0 * np.stack([ply[f"scale_{axis}"] for axis in range(3)], axis=1).astype(np.float64))
    return np.einsum("nij,nj,nkj->nik", rotations, variances, rotations)


def expected_map_covariances(run_dir: Path, truth: dict) -> np.ndarray:
    """Map-frame covariances of the synthetic checkpoint's Gaussians, from the stored quaternions and scales.

    In the dataparser frame Sigma_dp = R_q diag(s^2) R_q^T; x_dp = A x_map + t gives Sigma_map = A^T Sigma_dp A.

    Args:
        run_dir: Synthetic run folder.
        truth: Output of ``build_synthetic_run`` (holds A).

    Returns:
        (N, 3, 3) float64 covariances.
    """
    import torch

    train_dir = next((run_dir / "train").glob("*/*/*"))
    state = torch.load(train_dir / "nerfstudio_models" / CHECKPOINT_FILE_NAME)["pipeline"]
    quaternions_wxyz = state["_model.gauss_params.quats"].numpy().astype(np.float64)
    log_scales = state["_model.gauss_params.scales"].numpy().astype(np.float64)
    rotations_dp = Rotation.from_quat(quaternions_wxyz[:, [1, 2, 3, 0]]).as_matrix()
    covariances_dp = np.einsum("nij,nj,nkj->nik", rotations_dp, np.exp(2.0 * log_scales), rotations_dp)
    return truth["rotation"].T @ covariances_dp @ truth["rotation"]


@pytest.mark.covers("T87")
@pytest.mark.covers("T88")
def test_default_export_is_corridor_cropped_and_in_the_viewer_frame(tmp_path):
    """Defaults: kept = corridor mask; means = M (map - origin); covariances = M Sigma M^T; sidecar written."""
    run = build_corridor_run(tmp_path)
    expected_keep = expected_corridor_keep(run)
    assert 0 < expected_keep.sum() < GAUSSIAN_COUNT
    assert load_script_module().main(["--run-dir", str(run["run_dir"]), "--dataset-dir", str(run["dataset_dir"])]) == 0
    ply = read_supersplat_ply(run["run_dir"] / PLY_RELATIVE_PATH)
    assert len(ply["x"]) == expected_keep.sum()
    # Step 1: viewer origin = mean of the train sensor positions; means in the viewer frame.
    origin = run["origins"].mean(axis=0)
    expected_means = (run["map_means"][expected_keep] - origin) @ SUPERSPLAT_FROM_MAP_ROTATION.T
    np.testing.assert_allclose(ply_means(ply), expected_means,
                               atol=FLOAT32_ROUNDOFF_ATOL * np.abs(run["map_means"]).max())
    # Step 2: covariance of a kept Gaussian in the viewer frame = M Sigma_map M^T.
    map_covariances = expected_map_covariances(run["run_dir"], run["truth"])[expected_keep]
    expected_covariances = SUPERSPLAT_FROM_MAP_ROTATION @ map_covariances @ SUPERSPLAT_FROM_MAP_ROTATION.T
    np.testing.assert_allclose(ply_covariances(ply), expected_covariances,
                               atol=COMPARISON_ATOL * np.abs(expected_covariances).max())
    # Step 3: the sidecar describes this export.
    record = read_viewer_frame_record(run["run_dir"] / "renders" / VIEWER_FRAME_FILE_NAME)
    np.testing.assert_allclose(record["origin_map"], origin)
    np.testing.assert_array_equal(record["rotation"], SUPERSPLAT_FROM_MAP_ROTATION)
    assert record["crop"]["mode"] == "corridor"
    assert record["crop"]["width_m"] == pytest.approx(run["width_m"])
    assert record["crop"]["z_lower"] == pytest.approx(run["z_lower"])
    assert record["crop"]["z_upper"] == pytest.approx(run["z_upper"])
    assert record["crop"]["path_points"] == len(run["path_xy"])
    assert "range_percentile" in record["crop"]
    # Step 4: the sidecar and the PLY together give back the map-frame means (what script 23 relies on).
    np.testing.assert_allclose(viewer_to_map_points(ply_means(ply), record["origin_map"], record["rotation"]),
                               run["map_means"][expected_keep],
                               atol=FLOAT32_ROUNDOFF_ATOL * np.abs(run["map_means"]).max())


@pytest.mark.covers("T87")
def test_map_frame_corridor_export_has_unshifted_means_and_a_map_sidecar_replacing_a_stale_viewer_one(tmp_path):
    """--frame map: the same corridor subset in the map frame; the sidecar becomes a map-frame record."""
    run = build_corridor_run(tmp_path)
    expected_keep = expected_corridor_keep(run)
    script_module = load_script_module()
    base_arguments = ["--run-dir", str(run["run_dir"]), "--dataset-dir", str(run["dataset_dir"])]
    sidecar_path = run["run_dir"] / "renders" / VIEWER_FRAME_FILE_NAME
    # Step 1: a viewer export leaves a viewer-frame sidecar.
    assert script_module.main(base_arguments) == 0
    assert read_viewer_frame_record(sidecar_path)["frame"] == VIEWER_FRAME_NAME
    # Step 2: re-exporting in the map frame replaces it by a map-frame record and writes the map-frame means.
    assert script_module.main([*base_arguments, "--frame", "map", "--overwrite"]) == 0
    record = read_viewer_frame_record(sidecar_path)
    assert record["frame"] == MAP_FRAME_NAME
    assert record["crop"]["mode"] == "corridor"
    ply_path = run["run_dir"] / PLY_RELATIVE_PATH
    ply = read_supersplat_ply(ply_path)
    assert record["ply_bytes"] == ply_path.stat().st_size and record["vertex_count"] == expected_keep.sum()
    np.testing.assert_allclose(ply_means(ply), run["map_means"][expected_keep],
                               atol=FLOAT32_ROUNDOFF_ATOL * np.abs(run["map_means"]).max())


@pytest.mark.covers("T88")
def test_map_frame_export_without_dataset_writes_an_identity_sidecar_replacing_a_stale_one(tmp_path):
    """A map-frame PLY gets a map sidecar (origin 0, identity, correct size); a stale viewer one is replaced."""
    build_synthetic_run(tmp_path, zero_decoder=True)
    sidecar_path = tmp_path / "renders" / VIEWER_FRAME_FILE_NAME
    write_viewer_frame_record(sidecar_path, np.ones(3), SUPERSPLAT_FROM_MAP_ROTATION, {"mode": "none"})
    assert load_script_module().main(["--run-dir", str(tmp_path), "--no-crop", "--frame", "map"]) == 0
    record = read_viewer_frame_record(sidecar_path)
    ply_path = tmp_path / PLY_RELATIVE_PATH
    assert record["frame"] == MAP_FRAME_NAME
    np.testing.assert_array_equal(record["origin_map"], np.zeros(3))
    np.testing.assert_array_equal(record["rotation"], np.eye(3))
    assert record["crop"] == {"mode": "none"}
    assert record["ply_bytes"] == ply_path.stat().st_size
    assert record["vertex_count"] == GAUSSIAN_COUNT
    # No staged temp file is left beside the outputs.
    assert sorted(entry.name for entry in (tmp_path / "renders").iterdir()) == sorted(
        [VIEWER_FRAME_FILE_NAME, ply_path.name])


@pytest.mark.covers("T88")
def test_viewer_export_sidecar_records_frame_size_and_count(tmp_path):
    """The viewer sidecar names its frame and holds the exact PLY size and the row count."""
    run = build_corridor_run(tmp_path)
    expected_keep = expected_corridor_keep(run)
    assert load_script_module().main(["--run-dir", str(run["run_dir"]), "--dataset-dir",
                                      str(run["dataset_dir"])]) == 0
    record = read_viewer_frame_record(run["run_dir"] / "renders" / VIEWER_FRAME_FILE_NAME)
    assert record["frame"] == VIEWER_FRAME_NAME
    assert record["ply_bytes"] == (run["run_dir"] / PLY_RELATIVE_PATH).stat().st_size
    assert record["vertex_count"] == expected_keep.sum()


@pytest.mark.covers("T88")
def test_failed_ply_write_keeps_the_old_pair_and_removes_the_staged_sidecar(tmp_path, monkeypatch):
    """The sidecar is staged first and moved last: a failing PLY write leaves the old sidecar and no temp."""
    build_synthetic_run(tmp_path, zero_decoder=True)
    renders_dir = tmp_path / "renders"
    sidecar_path = renders_dir / VIEWER_FRAME_FILE_NAME
    old_bytes = 123
    write_viewer_frame_record(sidecar_path, np.zeros(3), np.eye(3), {"mode": "none"}, frame_name=MAP_FRAME_NAME,
                              ply_bytes=old_bytes, vertex_count=1)
    before = sidecar_path.read_text()

    def failing_write(*_arguments, **_keyword_arguments):
        """Stand-in for the PLY writer that fails like an interrupted export."""
        raise SystemExit("terminated by signal 15")

    monkeypatch.setattr("viking_splatad.export.supersplat_ply.write_supersplat_ply", failing_write)
    with pytest.raises(SystemExit):
        load_script_module().main(["--run-dir", str(tmp_path), "--no-crop", "--frame", "map"])
    assert sidecar_path.read_text() == before
    assert [entry.name for entry in renders_dir.iterdir()] == [VIEWER_FRAME_FILE_NAME]


@pytest.mark.covers("T87")
def test_select_crop_replays_a_recorded_corridor_without_rederiving_it(tmp_path, monkeypatch):
    """Replay from the sidecar's crop = the export's mask, and never calls the width / z derivation."""
    run = build_corridor_run(tmp_path)
    expected_keep = expected_corridor_keep(run)
    script_module = load_script_module()
    assert script_module.main(["--run-dir", str(run["run_dir"]), "--dataset-dir", str(run["dataset_dir"])]) == 0
    record = read_viewer_frame_record(run["run_dir"] / "renders" / VIEWER_FRAME_FILE_NAME)
    activated = 1.0 / (1.0 + np.exp(-run["truth"]["logits"].astype(np.float64)))

    def forbidden(*_arguments, **_keyword_arguments):
        """A replay must use the recorded parameters, not recompute the corridor."""
        raise AssertionError("train_lidar_corridor_map called during a replay")

    monkeypatch.setattr("viking_splatad.export.crop.train_lidar_corridor_map", forbidden)
    keep_mask, replayed_record = script_module.select_crop(
        run["run_dir"], run["dataset_dir"], run["map_means"], activated, "box", CONFIG_MIN_OPACITY,
        recorded_crop=record["crop"])
    np.testing.assert_array_equal(keep_mask, expected_keep)
    assert replayed_record == record["crop"]
    # An unknown mode, an incomplete corridor and a wrong path length are refused.
    with pytest.raises(ValueError, match="unknown crop mode"):
        script_module.select_crop(run["run_dir"], run["dataset_dir"], run["map_means"], activated, None,
                                  CONFIG_MIN_OPACITY, recorded_crop={"mode": "sphere"})
    with pytest.raises(ValueError, match="lacks"):
        script_module.select_crop(run["run_dir"], run["dataset_dir"], run["map_means"], activated, None,
                                  CONFIG_MIN_OPACITY, recorded_crop={"mode": "corridor", "width_m": 1.0})
    with pytest.raises(ValueError, match="path points"):
        script_module.select_crop(run["run_dir"], run["dataset_dir"], run["map_means"], activated, None,
                                  CONFIG_MIN_OPACITY, recorded_crop={**record["crop"], "path_points": 99})


@pytest.mark.covers("T86")
def test_select_crop_box_none_and_recorded_modes(tmp_path):
    """Fresh box = crop_mask on the lidar bounds; recorded box replays its corners; none keeps everything."""
    run = build_corridor_run(tmp_path)
    script_module = load_script_module()
    activated = 1.0 / (1.0 + np.exp(-run["truth"]["logits"].astype(np.float64)))
    box_mask, box_record = script_module.select_crop(
        run["run_dir"], run["dataset_dir"], run["map_means"], activated, "box", CONFIG_MIN_OPACITY)
    assert box_record["mode"] == "box"
    replayed_mask, _record = script_module.select_crop(
        run["run_dir"], run["dataset_dir"], run["map_means"], activated, None, CONFIG_MIN_OPACITY,
        recorded_crop=box_record)
    np.testing.assert_array_equal(replayed_mask, box_mask)
    none_mask, none_record = script_module.select_crop(
        run["run_dir"], run["dataset_dir"], run["map_means"], activated, None, CONFIG_MIN_OPACITY)
    assert none_mask.all() and none_record == {"mode": "none"}


def test_dry_run_writes_neither_ply_nor_sidecar(tmp_path):
    """--dry-run in the viewer frame computes everything but leaves renders/ empty."""
    run = build_corridor_run(tmp_path)
    assert load_script_module().main(["--run-dir", str(run["run_dir"]), "--dataset-dir", str(run["dataset_dir"]),
                                      "--dry-run"]) == 0
    assert not (run["run_dir"] / PLY_RELATIVE_PATH).exists()
    assert not (run["run_dir"] / "renders" / VIEWER_FRAME_FILE_NAME).exists()


def test_viewer_frame_without_a_dataset_raises_and_suggests_the_map_frame(tmp_path):
    """The viewer origin needs the train poses: a missing dataset is a ValueError that names --frame map."""
    build_synthetic_run(tmp_path, zero_decoder=True)
    with pytest.raises(ValueError, match="--frame map"):
        load_script_module().main(["--run-dir", str(tmp_path), "--no-crop",
                                   "--dataset-dir", str(tmp_path / "no_such_dataset")])
    assert not (tmp_path / PLY_RELATIVE_PATH).exists()


@pytest.mark.parametrize("no_crop_value", [False, None])
def test_export_run_accepts_false_and_none_as_no_crop(tmp_path, no_crop_value):
    """The legacy positional ``False`` and ``None`` both mean no crop: every Gaussian is projected."""
    build_synthetic_run(tmp_path, zero_decoder=True)
    result = load_script_module().export_run(tmp_path, None, 0, no_crop_value, False, dry_run=True)
    assert result["kept"] == GAUSSIAN_COUNT and result["crop_mode"] is None


def test_export_run_rejects_unknown_crop_mode_and_frame(tmp_path):
    """Typos must not silently select a default."""
    build_synthetic_run(tmp_path, zero_decoder=True)
    script_module = load_script_module()
    with pytest.raises(ValueError, match="crop_mode"):
        script_module.export_run(tmp_path, None, 0, "boxx", False, dry_run=True)
    with pytest.raises(ValueError, match="frame"):
        script_module.export_run(tmp_path, None, 0, None, False, dry_run=True, frame="world")


@pytest.mark.covers("T88")
def test_sh_degree_one_colour_is_the_same_function_in_both_frames(tmp_path):
    """Viewer SH at M d equals map SH at d (the degree-1 span {1, x, y, z} is rotation invariant).

    Both exports fit the same decoded colours c_m = f(A d_m); the viewer fit uses the basis at M d_m, so
    its colour along M d must equal the map fit's colour along d. Using the map basis at M d (a missing
    rotation of the fit directions) must instead disagree by a clearly larger amount.
    """
    run = build_corridor_run(tmp_path, decoder_weight_std=DIRECTION_TEST_WEIGHT_STD)
    script_module = load_script_module()
    base_arguments = ["--run-dir", str(run["run_dir"]), "--dataset-dir", str(run["dataset_dir"]),
                      "--no-crop", "--sh-degree", str(SH_TEST_DEGREE)]
    assert script_module.main(base_arguments) == 0
    viewer_coefficients = read_ply_coefficients(read_supersplat_ply(run["run_dir"] / PLY_RELATIVE_PATH),
                                                SH_TEST_DEGREE)
    assert script_module.main([*base_arguments, "--frame", "map", "--overwrite"]) == 0
    map_coefficients = read_ply_coefficients(read_supersplat_ply(run["run_dir"] / PLY_RELATIVE_PATH),
                                             SH_TEST_DEGREE)
    # Step 1: query directions in the map frame and their viewer-frame images.
    map_directions = np.random.default_rng(SH_TEST_SEED).normal(size=(SH_TEST_DIRECTION_COUNT, 3))
    map_directions /= np.linalg.norm(map_directions, axis=1, keepdims=True)
    viewer_directions = map_directions @ SUPERSPLAT_FROM_MAP_ROTATION.T
    # Step 2: colour offset 0.5 cancels in the comparison; compare sum_k c_k Y_k.
    colour_in_map = np.einsum("dk,nck->ndc", real_sh_basis(map_directions, SH_TEST_DEGREE), map_coefficients)
    colour_in_viewer = np.einsum("dk,nck->ndc", real_sh_basis(viewer_directions, SH_TEST_DEGREE),
                                 viewer_coefficients)
    np.testing.assert_allclose(colour_in_viewer, colour_in_map, atol=COMPARISON_ATOL)
    # Step 3: the wrong evaluation (viewer coefficients at the map direction) is clearly off.
    colour_wrong = np.einsum("dk,nck->ndc", real_sh_basis(map_directions, SH_TEST_DEGREE), viewer_coefficients)
    assert np.abs(colour_wrong - colour_in_map).max() > SH_TEST_MIN_SEPARATION * COMPARISON_ATOL


@pytest.mark.covers("T88")
def test_check_script_decode_ply_maps_a_viewer_frame_ply_back_to_the_map_frame(tmp_path, monkeypatch):
    """With a sidecar beside the PLY, decode_ply returns p_map = M^T p_viewer + origin and the crop record."""
    spec = importlib.util.spec_from_file_location(
        "check_supersplat_colours_script_for_frame",
        Path(__file__).resolve().parents[2] / "scripts" / "23_check_supersplat_colours.py")
    check_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_module)
    viewer_points = np.array([[1.0, 2.0, 3.0], [-4.0, 0.5, 6.0]])
    fields = {"x": viewer_points[:, 0].astype(np.float32), "y": viewer_points[:, 1].astype(np.float32),
              "z": viewer_points[:, 2].astype(np.float32), "opacity": np.zeros(2, dtype=np.float32),
              **{f"f_dc_{channel}": np.zeros(2, dtype=np.float32) for channel in range(3)}}
    monkeypatch.setattr(check_module, "read_supersplat_ply", lambda path: fields)
    origin_map = np.array([10.0, -20.0, 5.0])
    crop_record = {"mode": "corridor", "width_m": 3.0}
    write_viewer_frame_record(tmp_path / VIEWER_FRAME_FILE_NAME, origin_map, SUPERSPLAT_FROM_MAP_ROTATION,
                              crop_record)
    decoded = check_module.decode_ply(tmp_path / "supersplat.ply")
    # Inverse by hand: M maps (x, y, z)_map to (x, -z, y); so a viewer point (a, b, c) is the map point
    # (a, c, -b), plus the origin.
    expected_map = np.stack([viewer_points[:, 0], viewer_points[:, 2], -viewer_points[:, 1]], axis=1) + origin_map
    np.testing.assert_allclose(decoded["means"], expected_map)
    assert decoded["crop"] == crop_record
    # Without the sidecar the means are the stored ones (a map-frame export), and there is no crop record.
    (tmp_path / VIEWER_FRAME_FILE_NAME).unlink()
    legacy = check_module.decode_ply(tmp_path / "supersplat.ply")
    np.testing.assert_allclose(legacy["means"], viewer_points)
    assert legacy["crop"] is None


@pytest.mark.parametrize("arguments, named_remedy, unnamed_remedy", [
    (["--frame", "map"], "--no-crop", "--frame map"),
    (["--no-crop"], "--frame map", "--no-crop"),
])
def test_missing_dataset_message_names_only_the_option_that_needs_it(tmp_path, arguments, named_remedy,
                                                                     unnamed_remedy):
    """Crop only -> suggests --no-crop (not --frame map); viewer frame only -> suggests --frame map."""
    build_synthetic_run(tmp_path, zero_decoder=True)
    with pytest.raises(ValueError) as error_info:
        load_script_module().main(["--run-dir", str(tmp_path), "--dataset-dir", str(tmp_path / "missing"),
                                   *arguments])
    message = str(error_info.value)
    assert named_remedy in message and unnamed_remedy not in message


def load_check_module():
    """Import scripts/23_check_supersplat_colours.py by path (its name starts with a digit).

    Returns:
        The loaded module.
    """
    spec = importlib.util.spec_from_file_location(
        "check_supersplat_colours_script_for_pairing",
        Path(__file__).resolve().parents[2] / "scripts" / "23_check_supersplat_colours.py")
    check_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check_module)
    return check_module


@pytest.mark.covers("T87")
@pytest.mark.parametrize("frame_arguments", [["--frame", "map"], []])
def test_check_script_pairs_ply_rows_with_checkpoint_rows_of_a_corridor_export(tmp_path, frame_arguments):
    """Script 23's row matching (replayed corridor) finds the export's rows, in the map and viewer frames."""
    run = build_corridor_run(tmp_path)
    expected_keep = expected_corridor_keep(run)
    assert load_script_module().main(["--run-dir", str(run["run_dir"]), "--dataset-dir", str(run["dataset_dir"]),
                                      *frame_arguments]) == 0
    check_module = load_check_module()
    decoded = check_module.decode_ply(run["run_dir"] / PLY_RELATIVE_PATH)
    assert decoded["frame"] == (MAP_FRAME_NAME if frame_arguments else VIEWER_FRAME_NAME)
    checkpoint = check_module.load_checkpoint_rows_matching_ply(
        Namespace(run_dir=run["run_dir"]), run["dataset_dir"], decoded)
    np.testing.assert_array_equal(checkpoint["keep_mask"], expected_keep)
    assert checkpoint["row_match"]["rows_kept"] == expected_keep.sum()


@pytest.mark.covers("T88")
def test_check_script_refuses_a_ply_that_does_not_match_its_sidecar(tmp_path):
    """A sidecar whose ply_bytes differs from the file (interrupted / mixed export) makes decode_ply exit."""
    run = build_corridor_run(tmp_path)
    assert load_script_module().main(["--run-dir", str(run["run_dir"]), "--dataset-dir",
                                      str(run["dataset_dir"])]) == 0
    sidecar_path = run["run_dir"] / "renders" / VIEWER_FRAME_FILE_NAME
    check_module = load_check_module()
    # Sanity: the pair written together decodes.
    check_module.decode_ply(run["run_dir"] / PLY_RELATIVE_PATH)
    record = json.loads(sidecar_path.read_text())
    record["ply_bytes"] += 1
    sidecar_path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="re-export"):
        check_module.decode_ply(run["run_dir"] / PLY_RELATIVE_PATH)
