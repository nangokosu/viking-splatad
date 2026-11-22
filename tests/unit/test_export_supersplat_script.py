"""End-to-end test of scripts/22_export_supersplat_ply.py on a tiny synthetic run (T82-T86 wiring)."""

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

torch = pytest.importorskip("torch")
splatad_module = pytest.importorskip("nerfstudio.models.splatad")

from tests.unit.test_export_crop import build_synthetic_run as build_crop_dataset_and_run
from viking_splatad.export.spherical_harmonics import SH_C0, SH_COLOUR_OFFSET, real_sh_basis
from viking_splatad.export.splatad_colours import (
    bake_gaussian_colours, camera_sensor_index, load_upstream_rgb_decoder, model_settings_from_config_yml)
from viking_splatad.export.supersplat_ply import read_supersplat_ply

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "22_export_supersplat_ply.py"
RANDOM_SEED = 4321
# File name of the synthetic checkpoint (step 5, as nerfstudio zero-pads it); shared with the other export tests.
CHECKPOINT_FILE_NAME = "step-000000005.ckpt"
GAUSSIAN_COUNT = 7
# Model sizes of the real May config (features_dc is 3 wide, features_rest is feature_dim wide).
FEATURE_DIM, APPEARANCE_DIM, HIDDEN_DIM, KERNEL_SIZE, HIDDEN_BLOCKS = 13, 8, 32, 3, 1
ALBEDO_WIDTH = 3
CAMERA_NAMES = ("ids_camera", "second_camera")
SESSION, METHOD, RUN_ID = "toy_session", "splatad", "toy_run"
# Identity of the synthetic run as scripts/slurm/splatad_run.sbatch writes it into job_info.txt.
JOB_INFO_TEXT = f"RUN_ID: {RUN_ID}\nSESSION: {SESSION}\nTRAIN_METHOD: {METHOD}\n"
# Rotation angle (rad) and translation of the synthetic dataparser transform x_dp = A x_map + t.
DATAPARSER_ANGLE_RAD = 0.7
DATAPARSER_TRANSLATION = (-36.4, -52.7, 3.1)
# Albedo range chosen away from the [0, 1] clamp so the zero-decoder colour is exactly the albedo.
ALBEDO_LOW, ALBEDO_HIGH = 0.1, 0.9
# float32 round-off budget of the PLY path: 16 machine epsilons, covering the float32 bake, the
# float32 f_dc and the 0.5 + C0 * f_dc recombination (each worth ~1 eps for values <= 1).
FLOAT32_ROUNDOFF_ATOL = 16 * np.finfo(np.float32).eps


def load_script_module():
    """Import scripts/22_export_supersplat_ply.py (its name starts with a digit).

    Returns:
        The loaded module.
    """
    module_spec = importlib.util.spec_from_file_location("export_supersplat_ply_script", SCRIPT_PATH)
    script_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(script_module)
    return script_module


def build_synthetic_run(run_dir: Path, zero_decoder: bool, decoder_weight_std=None,
                        dataparser_rotation=None) -> dict:
    """Write a minimal run folder (checkpoint, config.yml, dataparser transform) and return its truth.

    Args:
        run_dir: Empty run folder to fill.
        zero_decoder: If true every decoder weight is zero (colour == clamped albedo exactly).
        decoder_weight_std: If given, every decoder parameter is redrawn N(0, std^2), so the colour
            depends strongly on the view direction (the default init scales the last layer by 1e-2).
        dataparser_rotation: (3, 3) rotation A of the dataparser transform (default: z rotation).

    Returns:
        Dict of the stored arrays: means_dp, logits, features_dc, and the rotation / translation.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    train_dir = run_dir / "train" / SESSION / METHOD / RUN_ID
    (train_dir / "nerfstudio_models").mkdir(parents=True)
    # Step 1: Gaussian parameters in the dataparser frame.
    means_dp = rng.normal(size=(GAUSSIAN_COUNT, 3)).astype(np.float32)
    log_scales = rng.normal(scale=0.3, size=(GAUSSIAN_COUNT, 3)).astype(np.float32)
    quats = rng.normal(size=(GAUSSIAN_COUNT, 4)).astype(np.float32)
    logits = rng.normal(scale=2.0, size=(GAUSSIAN_COUNT, 1)).astype(np.float32)
    features_dc = rng.uniform(ALBEDO_LOW, ALBEDO_HIGH, size=(GAUSSIAN_COUNT, ALBEDO_WIDTH)).astype(np.float32)
    features_rest = rng.normal(size=(GAUSSIAN_COUNT, FEATURE_DIM)).astype(np.float32)
    # Step 2: a real upstream decoder, random (or zero) weights, under the checkpoint key prefix.
    decoder = splatad_module.RGBDecoderCNN(
        FEATURE_DIM + APPEARANCE_DIM + 3, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE,
        num_hidden_blocks=HIDDEN_BLOCKS)
    if zero_decoder:
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.zero_()
    if decoder_weight_std is not None:
        with torch.no_grad():
            for parameter in decoder.parameters():
                parameter.copy_(torch.randn_like(parameter) * decoder_weight_std)
    pipeline_state = {f"_model.rgb_decoder.{key}": value for key, value in decoder.state_dict().items()}
    pipeline_state["_model.appearance_embedding.weight"] = torch.randn(len(CAMERA_NAMES) + 1, APPEARANCE_DIM)
    pipeline_state["_model.gauss_params.means"] = torch.from_numpy(means_dp)
    pipeline_state["_model.gauss_params.scales"] = torch.from_numpy(log_scales)
    pipeline_state["_model.gauss_params.quats"] = torch.from_numpy(quats)
    pipeline_state["_model.gauss_params.opacities"] = torch.from_numpy(logits)
    pipeline_state["_model.gauss_params.id"] = torch.zeros(GAUSSIAN_COUNT, 1)
    pipeline_state["_model.gauss_params.features_dc"] = torch.from_numpy(features_dc)
    pipeline_state["_model.gauss_params.features_rest"] = torch.from_numpy(features_rest)
    pipeline_state["_model.dynamic_actors.actor_sizes"] = torch.zeros(0, 3)  # 0 actors
    torch.save({"step": 5, "pipeline": pipeline_state}, train_dir / "nerfstudio_models" / CHECKPOINT_FILE_NAME)
    # Step 3: config.yml with the sections the settings reader walks (python tags as in real runs).
    camera_lines = "".join(f"      - {name}\n" for name in CAMERA_NAMES)
    (train_dir / "config.yml").write_text(
        "pipeline:\n  datamanager:\n    dataparser:\n"
        "      cameras: !!python/tuple\n" + camera_lines +
        "  model:\n"
        f"    appearance_dim: {APPEARANCE_DIM}\n    feature_dim: {FEATURE_DIM}\n"
        "    mcmc_min_opacity: 0.005\n"
        f"    rgb_decoder_hidden_dim: {HIDDEN_DIM}\n    rgb_decoder_kernel_size: {KERNEL_SIZE}\n"
        f"    rgb_decoder_num_hidden_blocks: {HIDDEN_BLOCKS}\n")
    # Step 4: rigid dataparser transform [A | t] with scale 1.
    rotation = (Rotation.from_euler("z", DATAPARSER_ANGLE_RAD).as_matrix()
                if dataparser_rotation is None else np.asarray(dataparser_rotation))
    translation = np.asarray(DATAPARSER_TRANSLATION)
    (train_dir / "dataparser_transforms.json").write_text(
        json.dumps({"transform": np.hstack([rotation, translation[:, None]]).tolist(), "scale": 1.0}))
    (run_dir / "renders").mkdir()
    (run_dir / "job_info.txt").write_text(JOB_INFO_TEXT)
    return {"means_dp": means_dp, "logits": logits.reshape(-1), "features_dc": features_dc,
            "rotation": rotation, "translation": translation}


@pytest.mark.covers("T82")
@pytest.mark.covers("T83")
@pytest.mark.covers("T85")
def test_export_roundtrip_geometry_opacity_and_exact_colour(tmp_path):
    """Zero decoder: colour is the albedo, so 0.5 + C0 * f_dc must equal it; geometry is A^T (mu - t)."""
    truth = build_synthetic_run(tmp_path, zero_decoder=True)
    assert load_script_module().main(["--run-dir", str(tmp_path), "--no-crop", "--frame", "map"]) == 0
    ply = read_supersplat_ply(tmp_path / "renders" / "supersplat.ply")
    # Row count = N (no crop).
    assert len(ply["x"]) == GAUSSIAN_COUNT
    # Means: x_map = A^T (x_dp - t), computed here from the definition x_dp = A x_map + t.
    expected_means = (truth["means_dp"].astype(np.float64) - truth["translation"]) @ truth["rotation"]
    np.testing.assert_allclose(np.stack([ply["x"], ply["y"], ply["z"]], axis=1), expected_means,
                               atol=FLOAT32_ROUNDOFF_ATOL * np.abs(expected_means).max())
    # Opacity: the stored logit round-trips, so sigmoid of the PLY value is the activated opacity.
    expected_opacity = 1.0 / (1.0 + np.exp(-truth["logits"].astype(np.float64)))
    ply_opacity = 1.0 / (1.0 + np.exp(-ply["opacity"].astype(np.float64)))
    np.testing.assert_allclose(ply_opacity, expected_opacity, atol=FLOAT32_ROUNDOFF_ATOL)
    # Colour at degree 0: 0.5 + C0 * f_dc recovers the (clamped) albedo exactly up to float32 round-off.
    ply_colour = SH_COLOUR_OFFSET + SH_C0 * np.stack([ply[f"f_dc_{channel}"] for channel in range(3)], axis=1)
    np.testing.assert_allclose(ply_colour, truth["features_dc"], atol=FLOAT32_ROUNDOFF_ATOL)
    # Quaternions are unit length (T83 output).
    quaternions = np.stack([ply[f"rot_{part}"] for part in range(4)], axis=1)
    np.testing.assert_allclose(np.linalg.norm(quaternions, axis=1), 1.0, atol=FLOAT32_ROUNDOFF_ATOL)


def test_export_random_decoder_degree_one_is_finite_and_refuses_overwrite(tmp_path):
    """Random weights and degree 1 write a finite ply with 3 f_rest per channel; a second run needs --overwrite."""
    build_synthetic_run(tmp_path, zero_decoder=False)
    script_module = load_script_module()
    assert script_module.main(["--run-dir", str(tmp_path), "--no-crop", "--frame", "map", "--sh-degree", "1"]) == 0
    ply = read_supersplat_ply(tmp_path / "renders" / "supersplat.ply")
    # Degree 1: (1+1)^2 - 1 = 3 higher-order coefficients per channel, 9 f_rest columns.
    assert sum(name.startswith("f_rest_") for name in ply) == 3 * ((1 + 1) ** 2 - 1)
    assert all(np.isfinite(column).all() for column in ply.values())
    with pytest.raises(FileExistsError):
        script_module.main(["--run-dir", str(tmp_path), "--no-crop", "--frame", "map", "--sh-degree", "1"])
    assert script_module.main(["--run-dir", str(tmp_path), "--no-crop", "--frame", "map", "--overwrite"]) == 0


# ---------------------------------------------------------------------------------------------
# Added tests: direction dependence, crop path, --all control flow, guards, test_in_image.sh.
# ---------------------------------------------------------------------------------------------
# Std of the redrawn decoder weights: large enough that the colour depends on the view direction by
# far more than the float32 round-off and the SH-fit residual, small enough to stay mostly off the clamp.
DIRECTION_TEST_WEIGHT_STD = 0.08
# Degree-3 fit: the highest degree stored, so the fit residual of the smooth decoder is smallest.
DIRECTION_TEST_SH_DEGREE = 3
# Query directions drawn for the direction test, and the seed of the draw.
DIRECTION_TEST_QUERY_COUNT = 24
DIRECTION_TEST_SEED = 99
# Rotation of the dataparser transform in the direction test (xyz Euler angles, rad): all three axes
# are used, so a missing or transposed A cannot be mistaken for a z-only rotation.
DIRECTION_TEST_EULER_RAD = (0.9, -0.6, 1.3)
# The correct bake must be at least this many times closer to the export than the wrong ones.
DIRECTION_TEST_MIN_SEPARATION = 4.0
# Crop-path test: the box is the data-driven box of the map-frame means, widened by this margin (m),
# except along x where it starts between the two middle values (so some Gaussians are outside).
CROP_TEST_MARGIN = 1.0
# How far outside the data (m) a far-away crop box is put so it contains no Gaussian.
CROP_TEST_FAR_OFFSET = 1000.0
# mcmc_min_opacity written into the synthetic config.yml by build_synthetic_run.
CONFIG_MIN_OPACITY = 0.005
# Sentinel bytes standing in for an old ply, to tell a re-export from a skip.
OLD_PLY_SENTINEL = b"old"
# Names of the runs of the --all tests.
ALL_RUN_NAMES = ("run_a", "run_b")
# The ply file of a run, relative to the run folder.
PLY_RELATIVE_PATH = Path("renders") / "supersplat.ply"


def read_ply_coefficients(ply: dict, sh_degree: int) -> np.ndarray:
    """SH coefficients (N, 3, K + 1) of a read ply: index 0 = f_dc, then f_rest (channel-major).

    Args:
        ply: Output of ``read_supersplat_ply``.
        sh_degree: Degree the ply was written with.

    Returns:
        The coefficient array.
    """
    rest_per_channel = (sh_degree + 1) ** 2 - 1
    coefficients = np.zeros((len(ply["x"]), 3, rest_per_channel + 1))
    for channel in range(3):
        coefficients[:, channel, 0] = ply[f"f_dc_{channel}"]
        for rest_index in range(rest_per_channel):
            coefficients[:, channel, 1 + rest_index] = ply[f"f_rest_{channel * rest_per_channel + rest_index}"]
    return coefficients


@pytest.mark.covers("T85")
def test_exported_sh_is_in_the_map_frame_direction_convention(tmp_path):
    """SH at map direction d must equal a fresh decoder bake at d_dp = A d (not at d, not at A^T d).

    x_dp = A x_map + t, so a direction transforms with A alone. The export fits SH in the map frame;
    a missing rotation would reproduce the bake at d, a transposed one the bake at A^T d. The
    random-weight decoder makes the colour strongly direction dependent, so the three differ.
    """
    rotation = Rotation.from_euler("xyz", DIRECTION_TEST_EULER_RAD).as_matrix()
    build_synthetic_run(tmp_path, zero_decoder=False, decoder_weight_std=DIRECTION_TEST_WEIGHT_STD,
                        dataparser_rotation=rotation)
    assert load_script_module().main(["--run-dir", str(tmp_path), "--no-crop", "--frame", "map",
                                      "--sh-degree", str(DIRECTION_TEST_SH_DEGREE)]) == 0
    coefficients = read_ply_coefficients(read_supersplat_ply(tmp_path / PLY_RELATIVE_PATH),
                                         DIRECTION_TEST_SH_DEGREE)
    # Step 1: the exported colour along random map directions: 0.5 + sum_k c_k Y_k(d).
    query_directions = np.random.default_rng(DIRECTION_TEST_SEED).normal(size=(DIRECTION_TEST_QUERY_COUNT, 3))
    query_directions /= np.linalg.norm(query_directions, axis=1, keepdims=True)
    basis = real_sh_basis(query_directions, DIRECTION_TEST_SH_DEGREE)  # (D, K + 1)
    exported = SH_COLOUR_OFFSET + np.einsum("dk,nck->ndc", basis, coefficients)
    # Step 2: fresh bakes from the checkpoint at the right and the two wrong directions.
    train_dir = tmp_path / "train" / SESSION / METHOD / RUN_ID
    checkpoint = torch.load(train_dir / "nerfstudio_models" / CHECKPOINT_FILE_NAME)["pipeline"]
    settings = model_settings_from_config_yml(train_dir / "config.yml")
    decoder, appearance_weight = load_upstream_rgb_decoder(checkpoint, settings)
    appearance_vector = appearance_weight[camera_sensor_index(settings["cameras"], settings["cameras"][0])].numpy()
    features_dc = checkpoint["_model.gauss_params.features_dc"].numpy()
    features_rest = checkpoint["_model.gauss_params.features_rest"].numpy()
    candidate_directions = {"correct": query_directions @ rotation.T,
                            "no_rotation": query_directions,
                            "transposed": query_directions @ rotation}
    errors = {}
    for name, directions in candidate_directions.items():
        baked = bake_gaussian_colours(decoder, features_dc, features_rest, appearance_vector,
                                      directions.astype(np.float32), GAUSSIAN_COUNT)
        errors[name] = np.abs(exported - baked).max()
    # Step 3: the correct convention is clearly the closest.
    assert errors["correct"] * DIRECTION_TEST_MIN_SEPARATION < min(errors["no_rotation"], errors["transposed"]), errors


def map_means_of(truth: dict) -> np.ndarray:
    """Map-frame means of a synthetic run: x_map = A^T (x_dp - t), from the definition x_dp = A x_map + t.

    Args:
        truth: Output of ``build_synthetic_run``.

    Returns:
        (N, 3) float64 map-frame means.
    """
    return (truth["means_dp"].astype(np.float64) - truth["translation"]) @ truth["rotation"]


def build_crop_run(scratch_root: Path, box_from_means):
    """A synthetic run over a synthetic lidar dataset whose train box is derived from the run's means.

    Two sweeps carry the sensor point (1, 0, 0) with identity rotation, so the map points are
    (1, 0, 0) + t; the translations are chosen so the two points are the box corners.

    Args:
        scratch_root: Empty scratch folder.
        box_from_means: Function (N, 3) map means -> (lower corner, upper corner).

    Returns:
        (dataset_dir, run_dir, truth, lower_corner, upper_corner).
    """
    # Step 1: the means depend only on the seed, so a probe run gives them before the dataset exists.
    probe_dir = scratch_root / "probe"
    probe_dir.mkdir()
    lower_corner, upper_corner = box_from_means(map_means_of(build_synthetic_run(probe_dir, zero_decoder=True)))
    # Step 2: sweeps whose map points are the two corners.
    sensor_point = np.array([1.0, 0.0, 0.0])
    poses = np.tile(np.eye(4), (2, 1, 1))
    poses[0, :3, 3] = lower_corner - sensor_point
    poses[1, :3, 3] = upper_corner - sensor_point
    real_root = scratch_root / "real"
    real_root.mkdir()
    dataset_dir, run_dir = build_crop_dataset_and_run(real_root, poses)
    truth = build_synthetic_run(run_dir, zero_decoder=True)
    return dataset_dir, run_dir, truth, lower_corner, upper_corner


@pytest.mark.covers("T86")
def test_crop_path_keeps_exactly_the_gaussians_inside_the_train_box(tmp_path):
    """The crop path of the script: box from train sweeps, opacity floor, subset order preserved."""
    def half_open_box(means):
        """Box around every mean, except its lower x face cuts the means in two (some kept, some dropped)."""
        lower = means.min(axis=0) - CROP_TEST_MARGIN
        # Midway between the two middle x values: no Gaussian sits on the face (float32 poses would flip it).
        middle_pair = np.sort(means[:, 0])[GAUSSIAN_COUNT // 2 - 1:GAUSSIAN_COUNT // 2 + 1]
        lower[0] = middle_pair.mean()
        return lower, means.max(axis=0) + CROP_TEST_MARGIN

    dataset_dir, run_dir, truth, lower_corner, upper_corner = build_crop_run(tmp_path, half_open_box)
    expected_means = map_means_of(truth)
    activated = 1.0 / (1.0 + np.exp(-truth["logits"].astype(np.float64)))
    expected_keep = (((expected_means >= lower_corner) & (expected_means <= upper_corner)).all(axis=1)
                     & (activated >= CONFIG_MIN_OPACITY))
    assert 0 < expected_keep.sum() < GAUSSIAN_COUNT
    assert load_script_module().main(["--run-dir", str(run_dir), "--dataset-dir", str(dataset_dir),
                                      "--crop-mode", "box", "--frame", "map"]) == 0
    ply = read_supersplat_ply(run_dir / PLY_RELATIVE_PATH)
    assert len(ply["x"]) == expected_keep.sum()
    np.testing.assert_allclose(np.stack([ply["x"], ply["y"], ply["z"]], axis=1), expected_means[expected_keep],
                               atol=FLOAT32_ROUNDOFF_ATOL * np.abs(expected_means).max())


@pytest.mark.covers("T86")
def test_crop_keeping_nothing_raises_instead_of_writing_a_header_only_ply(tmp_path):
    """A box far from every Gaussian keeps 0: ValueError, and no ply is created."""
    def far_box(means):
        """Box CROP_TEST_FAR_OFFSET beyond the largest mean on every axis, so it holds no Gaussian."""
        far_corner = means.max(axis=0) + CROP_TEST_FAR_OFFSET
        return far_corner, far_corner + CROP_TEST_MARGIN

    dataset_dir, run_dir, _, _, _ = build_crop_run(tmp_path, far_box)
    with pytest.raises(ValueError, match="keeps 0"):
        load_script_module().main(["--run-dir", str(run_dir), "--dataset-dir", str(dataset_dir),
                                      "--crop-mode", "box", "--frame", "map"])
    assert not (run_dir / PLY_RELATIVE_PATH).exists()


def build_batch_runs(runs_root: Path) -> list:
    """Create the ``--all`` test runs (same synthetic checkpoint each).

    Args:
        runs_root: Folder that receives one sub-folder per name in ``ALL_RUN_NAMES``.

    Returns:
        The run folders, in ``ALL_RUN_NAMES`` order.
    """
    run_dirs = []
    for run_name in ALL_RUN_NAMES:
        run_dir = runs_root / run_name
        run_dir.mkdir(parents=True)
        build_synthetic_run(run_dir, zero_decoder=True)
        run_dirs.append(run_dir)
    return run_dirs


def test_all_skips_existing_and_overwrite_reexports(tmp_path, capsys):
    """--all skips runs with a ply; --all --overwrite exports them again (the sentinel is replaced)."""
    script_module = load_script_module()
    run_dirs = build_batch_runs(tmp_path)
    ply_paths = [run_dir / PLY_RELATIVE_PATH for run_dir in run_dirs]
    for ply_path in ply_paths:
        ply_path.write_bytes(OLD_PLY_SENTINEL)
    # Step 1: plain --all leaves the old files alone and says why.
    assert script_module.main(["--all", "--runs-root", str(tmp_path), "--no-crop", "--frame", "map"]) == 0
    assert all(ply_path.read_bytes() == OLD_PLY_SENTINEL for ply_path in ply_paths)
    assert "supersplat.ply exists" in capsys.readouterr().out
    # Step 2: --overwrite re-exports both.
    assert script_module.main(["--all", "--runs-root", str(tmp_path), "--no-crop", "--frame", "map", "--overwrite"]) == 0
    assert all(len(read_supersplat_ply(ply_path)["x"]) == GAUSSIAN_COUNT for ply_path in ply_paths)
    assert f"exported {len(ALL_RUN_NAMES)}, skipped 0, failed 0" in capsys.readouterr().out


def test_all_records_a_failing_run_and_continues(tmp_path, capsys):
    """The first run lacks its dataparser transform: it is reported by id, the second is still exported."""
    script_module = load_script_module()
    failing_dir, good_dir = build_batch_runs(tmp_path)
    (failing_dir / "train" / SESSION / METHOD / RUN_ID / "dataparser_transforms.json").unlink()
    assert script_module.main(["--all", "--runs-root", str(tmp_path), "--no-crop", "--frame", "map"]) == 1
    output = capsys.readouterr().out
    assert f"FAILED {failing_dir.name}" in output and "FileNotFoundError" in output
    assert "exported 1, skipped 0, failed 1" in output
    assert (good_dir / PLY_RELATIVE_PATH).is_file()
    assert not (failing_dir / PLY_RELATIVE_PATH).exists()


def test_disk_guard_stops_the_batch_and_counts_the_margin(tmp_path, monkeypatch, capsys):
    """Free space = projected + margin - 1 refuses (batch stops, exit 1); exactly projected + margin passes."""
    script_module = load_script_module()
    first_dir, _ = build_batch_runs(tmp_path)
    projected_bytes = script_module.export_run(first_dir, None, 0, False, False, dry_run=True)["bytes"]
    assert not (first_dir / PLY_RELATIVE_PATH).exists()  # a dry run writes nothing
    needed_bytes = projected_bytes + script_module.DISK_SAFETY_MARGIN_BYTES
    monkeypatch.setattr(script_module, "free_bytes_at", lambda path: needed_bytes - 1)
    assert script_module.main(["--all", "--runs-root", str(tmp_path), "--no-crop", "--frame", "map"]) == 1
    output = capsys.readouterr().out
    assert "stopping" in output and f"not attempted {len(ALL_RUN_NAMES)}" in output
    assert not (first_dir / PLY_RELATIVE_PATH).exists()
    assert script_module.main(["--run-dir", str(first_dir), "--no-crop", "--frame", "map"]) == 1  # --run-dir mode is guarded too
    monkeypatch.setattr(script_module, "free_bytes_at", lambda path: needed_bytes)
    assert script_module.main(["--run-dir", str(first_dir), "--no-crop", "--frame", "map"]) == 0
    # The projection is exact: header + rows.
    assert (first_dir / PLY_RELATIVE_PATH).stat().st_size == projected_bytes


def test_all_rejects_dataset_dir(tmp_path):
    """Each --all run derives its own dataset, so one --dataset-dir is a usage error (exit 2)."""
    with pytest.raises(SystemExit) as exit_info:
        load_script_module().main(["--all", "--runs-root", str(tmp_path), "--dataset-dir", str(tmp_path)])
    assert exit_info.value.code == 2


def make_decoy_train_dir(train_dir: Path) -> None:
    """Create a train dir that has a config.yml and a checkpoint folder but is not the run's own.

    Args:
        train_dir: Folder to create (must not exist).
    """
    train_dir.mkdir(parents=True)
    (train_dir / "config.yml").write_text("")
    (train_dir / "nerfstudio_models").mkdir()


def test_train_dir_is_the_one_named_by_job_info(tmp_path):
    """job_info.txt picks the folder; a decoy session folder is ignored; a wrong or missing job_info raises."""
    script_module = load_script_module()
    build_synthetic_run(tmp_path, zero_decoder=True)
    # A decoy that sorts first and would win a "first glob" rule.
    make_decoy_train_dir(tmp_path / "train" / "a_other_session" / METHOD / RUN_ID)
    assert script_module.find_train_dir(tmp_path) == tmp_path / "train" / SESSION / METHOD / RUN_ID
    (tmp_path / "job_info.txt").write_text(JOB_INFO_TEXT.replace(f"SESSION: {SESSION}", "SESSION: nowhere"))
    with pytest.raises(FileNotFoundError):
        script_module.find_train_dir(tmp_path)
    (tmp_path / "job_info.txt").unlink()
    with pytest.raises(FileNotFoundError):
        script_module.find_train_dir(tmp_path)


def test_train_dir_without_method_must_be_unambiguous(tmp_path):
    """No TRAIN_METHOD recorded: two method folders with the run id are refused as ambiguous."""
    script_module = load_script_module()
    build_synthetic_run(tmp_path, zero_decoder=True)
    make_decoy_train_dir(tmp_path / "train" / SESSION / "other_method" / RUN_ID)
    (tmp_path / "job_info.txt").write_text(f"RUN_ID: {RUN_ID}\nSESSION: {SESSION}\n")
    with pytest.raises(ValueError, match="ambiguous"):
        script_module.find_train_dir(tmp_path)


def test_decoder_batch_size_follows_the_formula():
    """batch = floor(budget / (live * widest * (1 + 2 pad)^2 * 4)), read from the decoder's own layers."""
    script_module = load_script_module()
    input_width = FEATURE_DIM + APPEARANCE_DIM + 3
    decoder = splatad_module.RGBDecoderCNN(input_width, hidden_dim=HIDDEN_DIM, kernel_size=KERNEL_SIZE,
                                           num_hidden_blocks=HIDDEN_BLOCKS)
    padding = KERNEL_SIZE // 2
    widest = max(input_width, HIDDEN_DIM)
    working_bytes = (script_module.DECODER_LIVE_TENSOR_COUNT * widest * (1 + 2 * padding) ** 2
                     * np.dtype(np.float32).itemsize)
    whole_gaussians_in_budget = 10
    budget = whole_gaussians_in_budget * working_bytes + working_bytes // 2
    assert script_module.decoder_batch_size_for_budget(decoder, budget) == whole_gaussians_in_budget
    assert script_module.decoder_batch_size_for_budget(decoder, 1) == 1


# --- scripts/test_in_image.sh: VIKING_WRITABLE_DIR validation (runs before docker) -------------
IN_IMAGE_SCRIPT_PATH = SCRIPT_PATH.parent / "test_in_image.sh"
# Rejected values: parent escape, absolute, repo root (two spellings), embedded '..', ':' and a missing dir.
REJECTED_WRITABLE_DIRS = ("..", "/abs/path", ".", "./", "scripts/../..", "scripts:ro", "does_not_exist_dir")
# A directory that exists in the repo (and in the image), accepted.
ACCEPTED_WRITABLE_DIR = "scripts"


def run_in_image_script_with_stub_docker(tmp_path: Path, writable_dir):
    """Run test_in_image.sh with a fake ``docker`` that records its arguments instead of running.

    Args:
        tmp_path: Scratch folder for the stub and its record.
        writable_dir: Value of VIKING_WRITABLE_DIR, or None for unset.

    Returns:
        (completed process, recorded docker argument lines or None when docker was never called).
    """
    stub_dir = tmp_path / "stub"
    stub_dir.mkdir(exist_ok=True)
    record_path = tmp_path / "docker_args.txt"
    record_path.unlink(missing_ok=True)
    stub_path = stub_dir / "docker"
    stub_path.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{record_path}"\n')
    stub_path.chmod(0o755)
    environment = {**os.environ, "PATH": f"{stub_dir}{os.pathsep}{os.environ['PATH']}"}
    environment.pop("VIKING_WRITABLE_DIR", None)
    if writable_dir is not None:
        environment["VIKING_WRITABLE_DIR"] = writable_dir
    completed = subprocess.run(["sh", str(IN_IMAGE_SCRIPT_PATH), "true"], env=environment,
                               capture_output=True, text=True, check=False)
    return completed, (record_path.read_text().splitlines() if record_path.exists() else None)


@pytest.mark.parametrize("writable_dir", REJECTED_WRITABLE_DIRS)
def test_in_image_script_rejects_bad_writable_dir_before_docker(tmp_path, writable_dir):
    """A bad VIKING_WRITABLE_DIR exits non-zero with a message and never reaches docker."""
    completed, docker_arguments = run_in_image_script_with_stub_docker(tmp_path, writable_dir)
    assert completed.returncode != 0
    assert "VIKING_WRITABLE_DIR" in completed.stderr
    assert docker_arguments is None


def test_in_image_script_mounts_a_valid_writable_dir_and_unset_adds_nothing(tmp_path):
    """A valid dir adds exactly one '-v <src>:<dst>:rw' pair; unset adds no rw mount."""
    completed, docker_arguments = run_in_image_script_with_stub_docker(tmp_path, ACCEPTED_WRITABLE_DIR)
    assert completed.returncode == 0
    read_write_mounts = [argument for argument in docker_arguments if argument.endswith(":rw")]
    assert len(read_write_mounts) == 1
    source_path, target_path, mode = read_write_mounts[0].rsplit(":", 2)
    assert Path(source_path).name == ACCEPTED_WRITABLE_DIR
    assert target_path.endswith("/" + ACCEPTED_WRITABLE_DIR) and mode == "rw"
    completed, docker_arguments = run_in_image_script_with_stub_docker(tmp_path, None)
    assert completed.returncode == 0
    assert not any(argument.endswith(":rw") for argument in docker_arguments)


def test_dry_run_neither_refuses_nor_skips_runs_that_already_have_a_ply(tmp_path, capsys):
    """--run-dir --dry-run works on a run with a ply (no --overwrite); --all --dry-run lists it; file untouched."""
    script_module = load_script_module()
    run_dirs = build_batch_runs(tmp_path)
    ply_paths = [run_dir / PLY_RELATIVE_PATH for run_dir in run_dirs]
    for ply_path in ply_paths:
        ply_path.write_bytes(OLD_PLY_SENTINEL)
    # Step 1: one run, dry: no FileExistsError.
    assert script_module.main(["--run-dir", str(run_dirs[0]), "--no-crop", "--frame", "map", "--dry-run"]) == 0
    # Step 2: --all dry: every run is exported (dry), none skipped.
    assert script_module.main(["--all", "--runs-root", str(tmp_path), "--no-crop", "--frame", "map", "--dry-run"]) == 0
    assert f"exported {len(ALL_RUN_NAMES)}, skipped 0, failed 0" in capsys.readouterr().out
    # Step 3: nothing was written.
    assert all(ply_path.read_bytes() == OLD_PLY_SENTINEL for ply_path in ply_paths)


def test_sigterm_handler_raises_system_exit_and_restores(tmp_path):
    """After installation SIGTERM unwinds as SystemExit (so the writer's temp-file cleanup runs)."""
    import signal

    script_module = load_script_module()
    previous_handler = signal.getsignal(signal.SIGTERM)
    try:
        script_module.raise_system_exit_on_sigterm()
        with pytest.raises(SystemExit):
            os.kill(os.getpid(), signal.SIGTERM)
    finally:
        signal.signal(signal.SIGTERM, previous_handler)


def test_rotate_directions_renormalises_a_scaled_transform():
    """A = 3 * R keeps direction and gives unit length (|A d| = 3 |d| before the division)."""
    script_module = load_script_module()
    rotation = Rotation.from_euler("z", DATAPARSER_ANGLE_RAD).as_matrix()
    directions = np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    rotated = script_module.rotate_directions_to_dataparser(directions, 3.0 * rotation)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), 1.0)
    np.testing.assert_allclose(rotated, directions @ rotation.T)


@pytest.mark.covers("T86")
def test_wrong_dataset_raises_before_the_checkpoint_is_loaded(tmp_path, monkeypatch):
    """A dataset whose hashes differ from split.json is refused before the slow checkpoint load."""
    def around_every_mean(means):
        """Box around every mean (its content is irrelevant here: the run must fail before using it)."""
        return means.min(axis=0) - CROP_TEST_MARGIN, means.max(axis=0) + CROP_TEST_MARGIN

    dataset_dir, run_dir, _, _, _ = build_crop_run(tmp_path, around_every_mean)
    # Step 1: the recorded manifest hash no longer matches the dataset on disk.
    split_path = run_dir / "split.json"
    split_record = json.loads(split_path.read_text())
    split_record["dataset"]["manifest_sha256"] = "0" * len(split_record["dataset"]["manifest_sha256"])
    split_path.write_text(json.dumps(split_record))

    class CheckpointLoadReached(Exception):
        """Sentinel: the script got as far as the checkpoint load."""

    def fail_on_load(_checkpoint_path):
        """Stand-in for load_checkpoint_tensors."""
        raise CheckpointLoadReached

    script_module = load_script_module()
    monkeypatch.setattr(script_module, "load_checkpoint_tensors", fail_on_load)
    # Step 2: the identity error, not the sentinel.
    with pytest.raises(ValueError, match="not the dataset the run trained on"):
        script_module.main(["--run-dir", str(run_dir), "--dataset-dir", str(dataset_dir),
                            "--crop-mode", "box", "--frame", "map"])
