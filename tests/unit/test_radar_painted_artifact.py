"""Schema /2 radar anchor artifacts: optional painted seed colours (`rgb`) and their gates."""

import json

import numpy as np
import pytest

from tests.unit.test_loss_only_artifact import synthetic_artifact
from viking_splatad.contract.splatad_contract import validate_radar_inputs
from viking_splatad.dataset.run_split import file_sha256
from viking_splatad.radar_init.artifact import (
    anchor_split, crop_anchor_arrays, load_verified_anchors, validate_anchor_arrays,
)
from viking_splatad.radar_init.build import build_loss_only_artifact
from viking_splatad.splatad.viking_conversions import open_dataset_view

PAINTED_SCHEMA = "viking-radar-loss-anchors/2"
UNPAINTED_SCHEMA = "viking-radar-loss-anchors/1"
# Colour channels per anchor (RGB); a property of the format, not a tunable.
RGB_CHANNELS = 3
# Camera stamp meaning "this anchor was not painted" (contract of the /2 painting record).
UNPAINTED_STAMP_NS = -1


def build_unpainted(tmp_path):
    """Build a real schema /1 artifact with the shared fixture.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        (dataset root, split settings, artifact directory, session name).
    """
    root, settings, config, calibration, options, _ = synthetic_artifact(tmp_path)
    output = tmp_path / "anchors"
    build_loss_only_artifact(root, config, settings, options, output, calibration)
    return root, settings, output, "synthetic"


def make_painted(directory, rgb, camera_stamps_ns):
    """Rewrite an unpainted artifact in place into schema /2.

    Adds `rgb` to radar_seeds.npz, tags the metadata and recomputes the npz checksum
    exactly as the builder does (`file_sha256` of radar_seeds.npz).

    Args:
        directory: Artifact directory holding radar_seeds.npz and radar_inputs.json.
        rgb: uint8 [S,3] colours to store.
        camera_stamps_ns: Per-anchor camera stamps for the painting record.

    Returns:
        The metadata dict that was written.
    """
    # Step 1: re-save the arrays with the extra rgb entry.
    with np.load(directory / "radar_seeds.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["rgb"] = rgb
    np.savez_compressed(directory / "radar_seeds.npz", **arrays)
    # Step 2: retag the metadata and refresh the checksum over the new npz bytes.
    metadata = json.loads((directory / "radar_inputs.json").read_text())
    metadata["schema"] = PAINTED_SCHEMA
    metadata["gaussian_initialization"] = True
    metadata["painting"] = {"camera": "synthetic_camera", "footprint_radius_pixels": 1.0,
                            "occlusion_margin_m": 0.5, "painted_fraction": 1.0,
                            "camera_stamps_ns": [int(stamp) for stamp in camera_stamps_ns]}
    metadata["artifact_sha256"] = file_sha256(directory / "radar_seeds.npz")
    (directory / "radar_inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def rewrite_metadata(directory, **changes):
    """Overwrite top-level keys of radar_inputs.json (npz checksum untouched).

    Args:
        directory: Artifact directory.
        **changes: Keys to set.
    """
    metadata = json.loads((directory / "radar_inputs.json").read_text())
    metadata.update(changes)
    (directory / "radar_inputs.json").write_text(json.dumps(metadata, indent=2) + "\n")


def camera_frames_by_split(root, settings):
    """Return (train camera stamps, non-train camera stamps) of the split.

    Args:
        root: Dataset root; settings: split settings.

    Returns:
        Two lists of camera `timestamp_ns`.
    """
    frames, _, _ = anchor_split(open_dataset_view(root), settings)
    cameras = [frame for frame in frames if frame.kind == "camera"]
    train = [frame.timestamp_ns for frame in cameras if frame.split == "train"]
    other = [frame.timestamp_ns for frame in cameras if frame.split != "train"]
    return train, other


def synthetic_arrays(anchor_count):
    """Return minimal valid anchor arrays with `anchor_count` anchors (no rgb).

    Args:
        anchor_count: Number of anchors.

    Returns:
        Dict with the keys validate_anchor_arrays requires; each anchor has two
        distinct source sweeps.
    """
    sweeps_per_anchor = 2
    return {
        "means": np.arange(anchor_count * 3, dtype=np.float64).reshape(anchor_count, 3),
        "covariances": np.tile(np.eye(3), (anchor_count, 1, 1)),
        "sensor_origins": np.zeros((anchor_count, 3)),
        "time_spans": np.tile([0., 1.], (anchor_count, 1)),
        "support_count": np.full(anchor_count, sweeps_per_anchor, dtype=np.int64),
        "source_offsets": np.arange(anchor_count + 1, dtype=np.int64) * sweeps_per_anchor,
        "source_stamps_ns": np.tile(np.arange(sweeps_per_anchor, dtype=np.int64), anchor_count)
        + np.repeat(np.arange(anchor_count, dtype=np.int64) * sweeps_per_anchor, sweeps_per_anchor),
    }


def test_validate_anchor_arrays_accepts_rgb_and_rejects_bad_rgb():
    """uint8 [S,3] rgb passes; wrong dtype or row count raises; no rgb is unchanged."""
    anchor_count = 4
    arrays = synthetic_arrays(anchor_count)
    validate_anchor_arrays(arrays)
    good = np.zeros((anchor_count, RGB_CHANNELS), dtype=np.uint8)
    validate_anchor_arrays(dict(arrays, rgb=good))
    with pytest.raises(ValueError):
        validate_anchor_arrays(dict(arrays, rgb=good.astype(np.float32)))
    with pytest.raises(ValueError):
        validate_anchor_arrays(dict(arrays, rgb=good[:-1]))
    with pytest.raises(ValueError):
        validate_anchor_arrays(dict(arrays, rgb=good[:, :-1]))


@pytest.mark.covers("T40")
def test_crop_keeps_rgb_rows_aligned_with_means():
    """A crop dropping anchors 0 and 2 keeps exactly rows 1 and 3 of rgb, same dtype."""
    anchor_count = 4
    arrays = synthetic_arrays(anchor_count)
    # Anchors 0 and 2 sit far outside the scene box on x; anchors 1 and 3 sit inside it.
    arrays["means"] = np.array([[-50., 0., 0.], [1., 0., 0.], [-60., 0., 0.], [2., 0., 0.]])
    arrays["rgb"] = np.arange(anchor_count * RGB_CHANNELS, dtype=np.uint8).reshape(anchor_count, RGB_CHANNELS)
    scene_bounds = np.array([[0., -1., -1.], [10., 1., 1.]])
    kept_rows = np.flatnonzero((arrays["means"] >= scene_bounds[0]).all(axis=1)
                               & (arrays["means"] <= scene_bounds[1]).all(axis=1))
    assert kept_rows.tolist() == [1, 3]
    cropped = crop_anchor_arrays(arrays, (0., 1.), scene_bounds, 0.)
    np.testing.assert_array_equal(cropped["rgb"], arrays["rgb"][kept_rows])
    assert cropped["rgb"].dtype == np.uint8
    np.testing.assert_array_equal(cropped["means"], arrays["means"][kept_rows])


@pytest.mark.covers("T40")
def test_crop_without_rgb_has_no_rgb_key():
    """Unpainted anchors never gain an rgb key through the crop."""
    arrays = synthetic_arrays(2)
    arrays["means"] = np.zeros((2, 3))
    cropped = crop_anchor_arrays(arrays, (0., 1.), np.array([[-1.] * 3, [1.] * 3]), 0.)
    assert "rgb" not in cropped


def test_painted_artifact_round_trips(tmp_path):
    """A hand-made schema /2 artifact verifies and returns exactly the rgb that was written."""
    root, settings, output, session = build_unpainted(tmp_path)
    anchor_count = len(load_verified_anchors(output, root, session, settings)[0]["means"])
    rgb = (np.arange(anchor_count * RGB_CHANNELS) % np.iinfo(np.uint8).max).astype(np.uint8)
    rgb = rgb.reshape(anchor_count, RGB_CHANNELS)
    train_stamps, _ = camera_frames_by_split(root, settings)
    make_painted(output, rgb, [train_stamps[0]] * anchor_count)
    arrays, metadata = load_verified_anchors(output, root, session, settings)
    np.testing.assert_array_equal(arrays["rgb"], rgb)
    assert arrays["rgb"].dtype == np.uint8
    assert metadata["schema"] == PAINTED_SCHEMA and metadata["gaussian_initialization"] is True


def test_unpainted_schema_1_artifact_still_loads(tmp_path):
    """Regression: Neel's /1 artifact loads unchanged, with no rgb and the flag false."""
    root, settings, output, session = build_unpainted(tmp_path)
    arrays, metadata = load_verified_anchors(output, root, session, settings)
    assert "rgb" not in arrays
    assert metadata["schema"] == UNPAINTED_SCHEMA and metadata["gaussian_initialization"] is False


def test_flag_without_rgb_is_refused(tmp_path):
    """gaussian_initialization true but no rgb array is refused."""
    root, settings, output, session = build_unpainted(tmp_path)
    rewrite_metadata(output, schema=PAINTED_SCHEMA, gaussian_initialization=True,
                     painting={"camera": "c", "footprint_radius_pixels": 1.0, "occlusion_margin_m": 0.5,
                               "painted_fraction": 0.0, "camera_stamps_ns": [UNPAINTED_STAMP_NS]})
    with pytest.raises(ValueError):
        load_verified_anchors(output, root, session, settings)


def test_rgb_with_flag_false_is_refused(tmp_path):
    """An rgb array under gaussian_initialization false is refused."""
    root, settings, output, session = build_unpainted(tmp_path)
    anchor_count = len(load_verified_anchors(output, root, session, settings)[0]["means"])
    train_stamps, _ = camera_frames_by_split(root, settings)
    make_painted(output, np.zeros((anchor_count, RGB_CHANNELS), dtype=np.uint8),
                 [train_stamps[0]] * anchor_count)
    rewrite_metadata(output, gaussian_initialization=False)
    with pytest.raises(ValueError):
        load_verified_anchors(output, root, session, settings)


def test_heldout_painting_camera_stamp_is_refused(tmp_path):
    """A painting stamp that is a test/guard camera frame leaks held-out data and is refused."""
    root, settings, output, session = build_unpainted(tmp_path)
    anchor_count = len(load_verified_anchors(output, root, session, settings)[0]["means"])
    _, held_out_stamps = camera_frames_by_split(root, settings)
    assert held_out_stamps, "fixture must have a non-train camera frame"
    make_painted(output, np.zeros((anchor_count, RGB_CHANNELS), dtype=np.uint8),
                 [held_out_stamps[0]] * anchor_count)
    with pytest.raises(ValueError):
        load_verified_anchors(output, root, session, settings)


def test_unpainted_marker_stamp_is_allowed(tmp_path):
    """Stamp -1 (unpainted anchor) is not a held-out reference and passes."""
    root, settings, output, session = build_unpainted(tmp_path)
    anchor_count = len(load_verified_anchors(output, root, session, settings)[0]["means"])
    make_painted(output, np.zeros((anchor_count, RGB_CHANNELS), dtype=np.uint8),
                 [UNPAINTED_STAMP_NS] * anchor_count)
    load_verified_anchors(output, root, session, settings)


def test_validate_radar_inputs_colour_requirement(tmp_path):
    """require_seed_colours fails an unpainted artifact, passes a painted one; False passes both."""
    root, settings, output, session = build_unpainted(tmp_path)
    assert validate_radar_inputs(output, session, root, settings, require_seed_colours=False).passed
    unpainted_report = validate_radar_inputs(output, session, root, settings, require_seed_colours=True)
    assert not unpainted_report.passed
    anchor_count = len(load_verified_anchors(output, root, session, settings)[0]["means"])
    train_stamps, _ = camera_frames_by_split(root, settings)
    make_painted(output, np.zeros((anchor_count, RGB_CHANNELS), dtype=np.uint8),
                 [train_stamps[0]] * anchor_count)
    assert validate_radar_inputs(output, session, root, settings, require_seed_colours=True).passed
    assert validate_radar_inputs(output, session, root, settings, require_seed_colours=False).passed
