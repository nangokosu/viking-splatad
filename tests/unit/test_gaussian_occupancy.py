"""Tests of viking_splatad.eval.gaussian_occupancy (T56) and its export script, synthetic known answers."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.eval.gaussian_occupancy import (
    ACTOR_SIZES_KEY,
    DDP_KEY_PREFIX,
    GAUSS_PARAM_NAMES,
    GAUSS_PARAMS_KEY_PREFIX,
    GAUSSIANS_NPZ_KEYS,
    checkpoint_gaussians_to_map_frame,
    gauss_params_from_state_dict,
    latest_checkpoint_path,
    save_gaussians_npz,
    stable_sigmoid,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SBATCH_PATH = PROJECT_ROOT / "scripts" / "slurm" / "splatad_run.sbatch"
RENDER_LIDAR_SCRIPT_PATH = PROJECT_ROOT / "scripts" / "08_render_lidar_pointcloud.py"


def rotation_about_z(angle_radians: float) -> np.ndarray:
    """3x3 rotation by ``angle_radians`` about +z (right-handed).

    Args:
        angle_radians: Rotation angle.

    Returns:
        (3, 3) rotation matrix.
    """
    cosine, sine = math.cos(angle_radians), math.sin(angle_radians)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def world_to_mean_transform(yaw_radians: float, translation: np.ndarray) -> np.ndarray:
    """A rigid w2m [A | t] with A a yaw about z.

    Args:
        yaw_radians: Yaw of A.
        translation: (3,) t.

    Returns:
        (3, 4) transform.
    """
    transform_3x4 = np.zeros((3, 4))
    transform_3x4[:, :3] = rotation_about_z(yaw_radians)
    transform_3x4[:, 3] = translation
    return transform_3x4


def one_static_gaussian(**overrides) -> dict:
    """Keyword arguments of checkpoint_gaussians_to_map_frame for one static Gaussian at the origin.

    Identity rotation, unit std, logit 0, identity w2m; ``overrides`` replace any entry.

    Args:
        **overrides: Entries to replace.

    Returns:
        The keyword-argument dict.
    """
    arguments = {
        "means": np.zeros((1, 3)),
        "log_scales": np.zeros((1, 3)),
        "quaternions_wxyz": np.array([[1.0, 0.0, 0.0, 0.0]]),
        "opacity_logits": np.zeros((1, 1)),
        "actor_ids": np.zeros((1, 1)),
        "actor_count": 0,
        "dataparser_transform": world_to_mean_transform(0.0, np.zeros(3)),
        "dataparser_scale": 1.0,
    }
    arguments.update(overrides)
    return arguments


@pytest.mark.covers("T56")
def test_rotated_gaussian_reaches_the_map_frame_with_exact_covariance_and_opacity():
    """Gaussian with std (1, 2, 3), yaw 120° in the dataparser frame; w2m = yaw 30°, shift t.

    Map-frame axes: A^T·R = yaw(120° − 30°) = yaw 90°, which swaps x and y, so
    Σ_map = diag(2², 1², 3²) = diag(4, 1, 9) exactly. The quaternion is the yaw-120° unit
    quaternion times 2 (unnormalised). Opacity logit ln 3: σ = 3 / (1 + 3) = 0.75.
    The dataparser mean is built from a known map mean m as A·m + t, so T56 must return m.
    """
    # Known map-frame mean and the w2m that defines the dataparser frame.
    map_mean = np.array([4.0, -7.0, 1.25])
    w2m_yaw = math.radians(30.0)
    w2m_translation = np.array([-3.0, 5.0, 0.5])
    dataparser_mean = rotation_about_z(w2m_yaw) @ map_mean + w2m_translation
    # Yaw 120° about z as a quaternion: (cos 60°, 0, 0, sin 60°), scaled by 2.
    half_angle = math.radians(120.0) / 2.0
    quaternion = 2.0 * np.array([[math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]])
    axis_std = np.array([1.0, 2.0, 3.0])
    gaussians = checkpoint_gaussians_to_map_frame(**one_static_gaussian(
        means=dataparser_mean[None, :],
        log_scales=np.log(axis_std)[None, :],
        quaternions_wxyz=quaternion,
        opacity_logits=np.array([[math.log(3.0)]]),
        dataparser_transform=world_to_mean_transform(w2m_yaw, w2m_translation),
    ))
    np.testing.assert_allclose(gaussians.means[0], map_mean, atol=1e-12)
    np.testing.assert_allclose(gaussians.covariances[0], np.diag([4.0, 1.0, 9.0]), atol=1e-12)
    np.testing.assert_allclose(gaussians.axis_std[0], axis_std, atol=1e-12)
    np.testing.assert_allclose(gaussians.opacities, [0.75], atol=1e-12)


@pytest.mark.covers("T56")
def test_identity_rotation_gives_diagonal_squared_std():
    """Quaternion (5, 0, 0, 0) normalises to identity, so Σ = diag(std²) = diag(0.25, 1, 16)."""
    gaussians = checkpoint_gaussians_to_map_frame(**one_static_gaussian(
        log_scales=np.log(np.array([[0.5, 1.0, 4.0]])),
        quaternions_wxyz=np.array([[5.0, 0.0, 0.0, 0.0]]),
    ))
    np.testing.assert_allclose(gaussians.covariances[0], np.diag([0.25, 1.0, 16.0]), atol=1e-12)


@pytest.mark.covers("T56")
def test_rotation_about_x_mixes_y_and_z_variances():
    """Yaw-free 90° roll about x maps local y → z and z → −y, so std (1, 2, 3) gives Σ = diag(1, 9, 4)."""
    half_angle = math.radians(90.0) / 2.0
    gaussians = checkpoint_gaussians_to_map_frame(**one_static_gaussian(
        log_scales=np.log(np.array([[1.0, 2.0, 3.0]])),
        quaternions_wxyz=np.array([[math.cos(half_angle), math.sin(half_angle), 0.0, 0.0]]),
    ))
    np.testing.assert_allclose(gaussians.covariances[0], np.diag([1.0, 9.0, 4.0]), atol=1e-12)


@pytest.mark.covers("T56")
def test_roll_gaussian_under_a_yawed_shifted_w2m():
    """Roll 90° about x gives Σ_dp = diag(1, 9, 4); w2m yaw 90°, so A^T = yaw −90° sends dataparser
    x → map −y and y → map x, swapping the first two variances: Σ_map = diag(9, 1, 4).
    The mean is again built as A·m + t from a known map mean m.
    """
    map_mean = np.array([-2.0, 3.0, 0.75])
    w2m_yaw = math.radians(90.0)
    w2m_translation = np.array([1.5, -2.5, 4.0])
    dataparser_mean = rotation_about_z(w2m_yaw) @ map_mean + w2m_translation
    half_angle = math.radians(90.0) / 2.0
    gaussians = checkpoint_gaussians_to_map_frame(**one_static_gaussian(
        means=dataparser_mean[None, :],
        log_scales=np.log(np.array([[1.0, 2.0, 3.0]])),
        quaternions_wxyz=np.array([[math.cos(half_angle), math.sin(half_angle), 0.0, 0.0]]),
        dataparser_transform=world_to_mean_transform(w2m_yaw, w2m_translation),
    ))
    np.testing.assert_allclose(gaussians.means[0], map_mean, atol=1e-12)
    np.testing.assert_allclose(gaussians.covariances[0], np.diag([9.0, 1.0, 4.0]), atol=1e-12)


@pytest.mark.covers("T56")
def test_sigmoid_is_finite_and_exact_at_known_points():
    """σ(0) = 0.5, σ(ln 3) = 0.75, σ(−ln 3) = 0.25; ±1000 give 1 and 0 without overflow."""
    logits = np.array([0.0, math.log(3.0), -math.log(3.0), 1000.0, -1000.0])
    with np.errstate(all="raise"):
        probabilities = stable_sigmoid(logits)
    np.testing.assert_allclose(probabilities, [0.5, 0.75, 0.25, 1.0, 0.0], atol=1e-12)


@pytest.mark.covers("T56")
def test_actor_gaussians_are_refused():
    """One actor: id 0 belongs to it (box frame), id 1 is static; any actor Gaussian raises."""
    with pytest.raises(ValueError, match="dynamic actors"):
        checkpoint_gaussians_to_map_frame(**one_static_gaussian(actor_ids=np.array([[0.0]]), actor_count=1))
    # id = actor count is static and accepted.
    checkpoint_gaussians_to_map_frame(**one_static_gaussian(actor_ids=np.array([[1.0]]), actor_count=1))


@pytest.mark.covers("T56")
def test_non_unit_scale_and_non_rigid_transform_raise():
    """Scale 2 breaks the T50 assumption; a sheared w2m has no rigid inverse."""
    with pytest.raises(ValueError, match="scale"):
        checkpoint_gaussians_to_map_frame(**one_static_gaussian(dataparser_scale=2.0))
    sheared_transform = world_to_mean_transform(0.0, np.zeros(3))
    sheared_transform[0, 1] = 0.5
    with pytest.raises(ValueError, match="rigid"):
        checkpoint_gaussians_to_map_frame(**one_static_gaussian(dataparser_transform=sheared_transform))


def test_zero_norm_quaternion_raises():
    """A zero quaternion defines no rotation."""
    with pytest.raises(ValueError, match="zero-norm"):
        checkpoint_gaussians_to_map_frame(**one_static_gaussian(quaternions_wxyz=np.zeros((1, 4))))


def synthetic_pipeline_state(gaussian_count: int, actor_count: int, key_prefix: str = "") -> dict:
    """A pipeline state dict with the keys T56 reads plus an unrelated one.

    Args:
        gaussian_count: Rows per Gaussian parameter.
        actor_count: Rows of the actor-size buffer.
        key_prefix: Prepended to every key (e.g. the DDP prefix).

    Returns:
        Key -> numpy array.
    """
    # Column count of each parameter, as in SplatAD's gauss_params (opacity and id are (N, 1)).
    column_counts = {"means": 3, "scales": 3, "quats": 4, "opacities": 1, "id": 1}
    state = {key_prefix + GAUSS_PARAMS_KEY_PREFIX + param_name: np.zeros((gaussian_count, column_counts[param_name]))
             for param_name in GAUSS_PARAM_NAMES}
    state[key_prefix + ACTOR_SIZES_KEY] = np.zeros((actor_count, 3))
    state[key_prefix + "_model.lidar_decoder.weight"] = np.zeros((2, 2))
    return state


def test_state_dict_params_are_picked_with_or_without_ddp_prefix():
    """Both key forms give the five parameters with N rows and the actor count from the buffer rows."""
    for key_prefix in ("", DDP_KEY_PREFIX):
        gauss_params = gauss_params_from_state_dict(synthetic_pipeline_state(7, 2, key_prefix))
        assert gauss_params["actor_count"] == 2
        assert {param_name: gauss_params[param_name].shape[0] for param_name in GAUSS_PARAM_NAMES} == \
            {param_name: 7 for param_name in GAUSS_PARAM_NAMES}


def test_state_dict_missing_key_or_row_mismatch_raises():
    """A missing parameter is a format change; disagreeing row counts are a corrupt state."""
    incomplete_state = synthetic_pipeline_state(3, 0)
    del incomplete_state[GAUSS_PARAMS_KEY_PREFIX + "quats"]
    with pytest.raises(KeyError):
        gauss_params_from_state_dict(incomplete_state)
    mismatched_state = synthetic_pipeline_state(3, 0)
    mismatched_state[GAUSS_PARAMS_KEY_PREFIX + "opacities"] = np.zeros((4, 1))
    with pytest.raises(ValueError, match="row count"):
        gauss_params_from_state_dict(mismatched_state)


def test_latest_checkpoint_compares_steps_as_integers(tmp_path):
    """step-10 beats step-9 even unpadded (string order would pick 9); other files are ignored."""
    for file_name in ("step-9.ckpt", "step-10.ckpt", "step-999.ckpt.tmp", "notes.txt"):
        (tmp_path / file_name).write_text("")
    assert latest_checkpoint_path(tmp_path).name == "step-10.ckpt"
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        latest_checkpoint_path(empty_dir)


def test_saved_npz_has_exactly_the_documented_keys(tmp_path):
    """Round trip: keys = GAUSSIANS_NPZ_KEYS, values equal the inputs at float32."""
    gaussians = checkpoint_gaussians_to_map_frame(**one_static_gaussian(opacity_logits=np.array([[math.log(3.0)]])))
    output_path = tmp_path / "renders" / "gaussians.npz"
    w2m = world_to_mean_transform(0.0, np.zeros(3))
    save_gaussians_npz(output_path, gaussians, w2m, checkpoint_step=123)
    with np.load(output_path) as saved:
        assert set(saved.files) == set(GAUSSIANS_NPZ_KEYS)
        assert int(saved["checkpoint_step"]) == 123
        np.testing.assert_allclose(saved["opacities"], [0.75], atol=1e-7)
        np.testing.assert_allclose(saved["dataparser_transform"], w2m, atol=0.0)


def test_glossary_documents_every_gaussians_npz_key():
    """Each gaussians.npz key appears as a `key` table cell in the run-folder glossary."""
    glossary_text = (PROJECT_ROOT / "runs" / "RUN_FOLDER_GLOSSARY.md").read_text()
    assert len(set(GAUSSIANS_NPZ_KEYS)) == len(GAUSSIANS_NPZ_KEYS)
    assert [npz_key for npz_key in GAUSSIANS_NPZ_KEYS if f"| `{npz_key}` |" not in glossary_text] == []


def test_sbatch_exports_gaussians_after_the_lidar_render_and_before_the_split_record():
    """The export runs once, after 08 (same checkpoint), before 09 so a failed export leaves no split.json."""
    sbatch_text = SBATCH_PATH.read_text()
    export_call = "scripts/10_export_gaussians.py"
    assert sbatch_text.count(export_call) == 1
    export_position = sbatch_text.index(export_call)
    assert sbatch_text.index("08_render_lidar_pointcloud.py") < export_position
    assert export_position < sbatch_text.index("09_extract_run_split.py")
    assert '--output-path "$RUN_DIR/renders/gaussians.npz"' in sbatch_text


def test_lidar_render_saves_accumulation():
    """08's npz key list carries the per-ray accumulation the ungated rule (PLAN §5 P1) needs."""
    module_spec = importlib.util.spec_from_file_location("render_lidar_pointcloud", RENDER_LIDAR_SCRIPT_PATH)
    render_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(render_module)
    assert "pred_accumulation" in render_module.LIDAR_NPZ_KEYS
    assert 'outputs["accumulation"]' in RENDER_LIDAR_SCRIPT_PATH.read_text()
