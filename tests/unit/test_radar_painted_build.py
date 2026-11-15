"""Painted radar anchor build (schema /2) and script 07's painting decision and default output path.

Assumption: the session config names the camera under ``camera.name`` (as in
``configs/sessions/*.yaml``); the shared fixture's session config lacks it, so it is added here.
"""

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.unit.test_loss_only_artifact import synthetic_artifact
from viking_splatad.dataset.writer import FRAME_FILE_PATTERN, write_dataset
from viking_splatad.radar_init.artifact import (
    ARTIFACT_SCHEMA, PAINTED_ARTIFACT_SCHEMA, UNPAINTED_CAMERA_STAMP_NS, load_verified_anchors,
)
from viking_splatad.radar_init.build import build_loss_only_artifact, source_digest
from viking_splatad.radar_init.seeds import NEUTRAL_GREY_RGB
from viking_splatad.splatad.viking_conversions import open_dataset_view

CAMERA_NAME = "cam_front"  # camera stream of tests.unit.test_writer.build_synthetic_dataset
IMAGE_SEED = 11
# Static world targets (map frame, x forward of the first camera pose, y, z=optical depth). The
# first lies in front of the camera and inside the image for every train frame the fixture
# uses; the second is the same lateral position behind the camera, so it can never be painted.
VISIBLE_TARGET_MAP = np.array([2.0, 0.5, 10.0])
BEHIND_TARGET_MAP = np.array([2.0, 0.5, -10.0])
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "scripts" / "07_build_radar_seeds.py"
SESSION_NAME = "may_short_grass"  # real session config used by the script-07 tests


def painted_fixture(tmp_path):
    """Synthetic dataset whose radar sees two static targets and whose camera frames hold known pixels.

    The base fixture's single radar point lies along the driving axis (outside the camera's view),
    so its radar sidecar is rewritten. A radar return of a static world point T at time t, with the
    vehicle at x = speed * t, is p = (T_x - speed * t, T_y, T_z) in the radar frame (identity
    extrinsics) and its Doppler is -speed * p_x / |p| (radial velocity of a static point).

    Args:
        tmp_path: Output directory.

    Returns:
        ``(root, settings, session_config, calibration, options, images_rgb_by_name)`` where the
        session config carries ``camera.name`` and images_rgb_by_name maps frame file name -> RGB.
    """
    root, settings, session_config, calibration, options, records = synthetic_artifact(tmp_path)
    view = open_dataset_view(root)
    # Step 1: ego speed from the lidar poses (+x drive), never a literal.
    lidar_times = view.sensor_times["lidar_top"]
    lidar_x = view.sensor_poses["lidar_top"][:, 0, 3]
    speed = (lidar_x[1] - lidar_x[0]) / (lidar_times[1] - lidar_times[0])
    # Step 2: rewrite each radar sweep with both targets.
    for stamp_ns, path in records:
        time_s = stamp_ns / 1e9  # reference_ns is 0 in the fixture
        rows = []
        for target in (VISIBLE_TARGET_MAP, BEHIND_TARGET_MAP):
            point = np.array([target[0] - speed * time_s, target[1], target[2]])
            rows.append([*point, 1.0, -speed * point[0] / np.linalg.norm(point)])
        np.save(path, np.array(rows))
    calibration = dict(calibration, fit_source_sha256=source_digest(records))
    # Step 3: occluder-free lidar sweeps (one far-behind return) so occlusion never hides a target.
    far_behind_row = np.array([[0.0, 0.0, -30.0, 0.5, 0.0, 0.0]], dtype=np.float32)
    for sweep_path in view.sensor_frame_paths["lidar_top"]:
        np.save(sweep_path, far_behind_row)
    # Step 4: exact-pixel camera frames (PNG bytes under the .jpg name; cv2 sniffs content).
    rng = np.random.default_rng(IMAGE_SEED)
    images_rgb_by_name = {}
    for frame_path in view.sensor_frame_paths[CAMERA_NAME]:
        shape = cv2.imread(str(frame_path)).shape
        image_rgb = rng.integers(0, 256, size=shape, dtype=np.uint8)
        encoded_ok, png_bytes = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
        assert encoded_ok
        frame_path.write_bytes(png_bytes.tobytes())
        images_rgb_by_name[frame_path.name] = image_rgb
    session_config = dict(session_config, camera={"name": CAMERA_NAME})
    return root, settings, session_config, calibration, options, images_rgb_by_name


def build_painted(tmp_path):
    """Build the painted artifact on the fixture; return everything the assertions need."""
    from viking_splatad.radar_init.build import build_radar_anchor_artifact

    root, settings, config, calibration, options, images_rgb_by_name = painted_fixture(tmp_path)
    output = tmp_path / "painted_anchors"
    build_radar_anchor_artifact(root, config, settings, options, output, calibration,
                                paint_seed_colours=True)
    arrays, metadata = load_verified_anchors(output, root, "synthetic", settings)
    return root, options, arrays, metadata, images_rgb_by_name


def test_painted_build_writes_schema_two_with_rgb_and_painting_record(tmp_path):
    """Schema /2, uint8 [S,3] rgb, init flag true, and a painting record consistent with its stamps."""
    root, options, arrays, metadata, _ = build_painted(tmp_path)
    anchor_count = len(arrays["means"])
    assert anchor_count == 2  # the visible and the behind-camera target survive persistence
    assert metadata["schema"] == PAINTED_ARTIFACT_SCHEMA
    assert metadata["gaussian_initialization"] is True
    assert arrays["rgb"].dtype == np.uint8 and arrays["rgb"].shape == (anchor_count, 3)
    painting = metadata["painting"]
    stamps = np.array(painting["camera_stamps_ns"], dtype=np.int64)
    assert len(stamps) == anchor_count
    # painted_fraction is the share of anchors that got a camera frame.
    assert painting["painted_fraction"] == pytest.approx(np.mean(stamps != UNPAINTED_CAMERA_STAMP_NS))
    # Exactly the in-view target is painted: strictly between none and all.
    assert 0.0 < painting["painted_fraction"] < 1.0
    # Settings recorded from the call: camera name, margin = range cell, footprint from the cells.
    view = open_dataset_view(root)
    intrinsics = view.manifest["cameras"][CAMERA_NAME]
    assert painting["camera"] == CAMERA_NAME
    assert painting["occlusion_margin_m"] == pytest.approx(options["range_cell_m"])
    expected_radius = max(intrinsics["fx"] * np.tan(np.deg2rad(options["azimuth_cell_degrees"])),
                          intrinsics["fy"] * np.tan(np.deg2rad(options["elevation_cell_degrees"])))
    assert painting["footprint_radius_pixels"] == pytest.approx(expected_radius)


def test_painted_anchor_colour_is_the_pixel_its_mean_projects_to(tmp_path):
    """Each painted anchor holds the recorded frame's pixel under an independent pinhole projection."""
    root, _, arrays, metadata, images_rgb_by_name = build_painted(tmp_path)
    view = open_dataset_view(root)
    intrinsics = view.manifest["cameras"][CAMERA_NAME]
    stamps = metadata["painting"]["camera_stamps_ns"]
    painted_count = 0
    for anchor_index, stamp in enumerate(stamps):
        if stamp == UNPAINTED_CAMERA_STAMP_NS:
            # Unpainted anchors keep the neutral grey colour.
            np.testing.assert_array_equal(arrays["rgb"][anchor_index], NEUTRAL_GREY_RGB)
            continue
        painted_count += 1
        # Locate the recorded camera frame by its timestamp in the file name.
        matches = [index for index, path in enumerate(view.sensor_frame_paths[CAMERA_NAME])
                   if int(FRAME_FILE_PATTERN.match(path.name).group(2)) == stamp]
        assert len(matches) == 1
        frame_index = matches[0]
        # Pinhole: p_cam = R^T (mean - t); u = fx X/Z + cx, v = fy Y/Z + cy; pixel = round(u, v).
        pose = view.sensor_poses[CAMERA_NAME][frame_index]
        point_camera = pose[:3, :3].T @ (arrays["means"][anchor_index] - pose[:3, 3])
        assert point_camera[2] > 0
        column_index = int(np.rint(intrinsics["fx"] * point_camera[0] / point_camera[2] + intrinsics["cx"]))
        row_index = int(np.rint(intrinsics["fy"] * point_camera[1] / point_camera[2] + intrinsics["cy"]))
        image_rgb = images_rgb_by_name[view.sensor_frame_paths[CAMERA_NAME][frame_index].name]
        assert 0 <= row_index < image_rgb.shape[0] and 0 <= column_index < image_rgb.shape[1]
        np.testing.assert_array_equal(arrays["rgb"][anchor_index], image_rgb[row_index, column_index])
    assert painted_count >= 1


def test_unpainted_build_is_schema_one_without_rgb(tmp_path):
    """Regression: build_loss_only_artifact keeps writing Neel's schema /1 artifact, unchanged."""
    root, settings, config, calibration, options, _ = synthetic_artifact(tmp_path)
    output = tmp_path / "anchors"
    build_loss_only_artifact(root, config, settings, options, output, calibration)
    arrays, metadata = load_verified_anchors(output, root, "synthetic", settings)
    assert metadata["schema"] == ARTIFACT_SCHEMA
    assert metadata["gaussian_initialization"] is False
    assert "rgb" not in arrays and "painting" not in metadata


# --- script 07: painting decision and default output path ----------------------------------------


class _Metadata(dict):
    """Fake builder result: any key script 07 prints reads as zero."""

    def __missing__(self, key):
        """Return a JSON-serialisable placeholder for an unrequested key."""
        return 0


def run_script_07(tmp_path, monkeypatch, profile, extra_arguments=()):
    """Run script 07's main() with the builder replaced by a recorder.

    Args:
        tmp_path: Holds the dataset directory ``<tmp>/<session>`` (its manifest names that
            session, so submit_run resolves the real session config).
        monkeypatch: pytest monkeypatch.
        profile: Profile name from configs/slurm.yaml.
        extra_arguments: Extra command-line arguments.

    Returns:
        ``(dataset_directory, recorded)`` where recorded holds the builder's out path and paint flag.
    """
    from tests.unit.test_writer import build_synthetic_dataset

    dataset_directory = tmp_path / SESSION_NAME
    write_dataset(dataset_directory, build_synthetic_dataset(num_frames=6))
    # A real dataset always has its manifest: it names the session (which selects the session YAML)
    # and holds the train fraction that profiles without their own fraction use (T27). Name the
    # synthetic dataset after the real session so script 07 resolves that session's config.
    manifest_path = dataset_directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["session"]["name"] = SESSION_NAME
    manifest_path.write_text(json.dumps(manifest))
    module_spec = importlib.util.spec_from_file_location("build_radar_seeds_script", SCRIPT_PATH)
    script = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(script)
    recorded = {}

    def record_build(dataset, session_config, settings, options, out, calibration=None,
                     paint_seed_colours=False):
        """Capture the arguments script 07 hands the builder."""
        recorded.update(dataset=Path(dataset), out=Path(out), paint=paint_seed_colours)
        # A painted build's metadata carries a painting record (script 07 prints its fraction).
        return _Metadata(painting={"painted_fraction": 0.0}) if paint_seed_colours else _Metadata()

    monkeypatch.setattr(script, "build_radar_anchor_artifact", record_build)
    # The recorder writes no artifact, so the post-build contract check (covered by
    # test_radar_review_fixes.py) is replaced by a passing report here.
    from viking_splatad.contract.splatad_contract import CheckResult, ContractReport

    monkeypatch.setattr(script, "validate_radar_inputs", lambda *arguments, **keywords: ContractReport(
        [CheckResult("radar_anchor_artifact", True, "stubbed: no artifact written")]))
    status = script.main(["--dataset", str(dataset_directory), "--session", SESSION_NAME,
                          "--profile", profile, *extra_arguments])
    assert status == 0
    return dataset_directory, recorded


def test_script_07_init_profile_paints_into_its_own_directory(tmp_path, monkeypatch):
    """radar_variant init paints and defaults --out to <dataset>/radar_inputs/<profile>."""
    dataset_directory, recorded = run_script_07(tmp_path, monkeypatch, "may_scene_a_radar_init")
    assert recorded["paint"] is True
    assert recorded["out"] == dataset_directory / "radar_inputs" / "may_scene_a_radar_init"


def test_script_07_init_loss_profile_uses_its_anchor_profile_directory(tmp_path, monkeypatch):
    """An init_loss profile paints and writes under its radar_anchor_profile's name."""
    dataset_directory, recorded = run_script_07(tmp_path, monkeypatch, "may_scene_a_radar_init_loss")
    assert recorded["paint"] is True
    assert recorded["out"] == dataset_directory / "radar_inputs" / "may_scene_a_radar_init"


def test_script_07_stock_profile_does_not_paint(tmp_path, monkeypatch):
    """A profile without radar_variant builds the unpainted artifact under its own name."""
    dataset_directory, recorded = run_script_07(tmp_path, monkeypatch, "may_scene_a")
    assert recorded["paint"] is False
    assert recorded["out"] == dataset_directory / "radar_inputs" / "may_scene_a"


def test_script_07_explicit_out_overrides_the_default(tmp_path, monkeypatch):
    """--out still wins over the default location."""
    explicit = tmp_path / "elsewhere"
    _, recorded = run_script_07(tmp_path, monkeypatch, "may_scene_a_radar_init",
                                ("--out", str(explicit)))
    assert recorded["out"] == explicit
