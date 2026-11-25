"""Tests for the occlusion-section geometry (figure_sections)."""

import json
from pathlib import Path

import numpy as np
import pytest

from viking_splatad.dataset.schema import SCHEMA_VERSION
from viking_splatad.dataset.writer import POSE_KEY_SEPARATOR
from viking_splatad.viz.figure_sections import in_slab, nearest_origin, session_sweep_origins, slice_frame

ABSOLUTE_TOLERANCE = 1e-12
SENSOR_NAME = "ouster"
# Box 3 m east and 4 m north of the sensor: horizontal distance 5, along = (0.6, 0.8, 0).
SENSOR_ORIGIN = np.array([1.0, 2.0, 1.5])
BOX_CENTRE = np.array([4.0, 6.0, 0.7])
HORIZONTAL_DISTANCE = 5.0
SLAB_HALF_WIDTH = 0.5
JUST_INSIDE_OFFSET = 0.49
JUST_OUTSIDE_OFFSET = 0.51


def test_frame_is_orthonormal_right_handed_and_points_to_box():
    """along = (3,4)/5, across = z x along = (-4,3)/5; along x across = up."""
    frame = slice_frame(BOX_CENTRE, SENSOR_ORIGIN)
    np.testing.assert_allclose(frame["along"], [0.6, 0.8, 0.0], atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(frame["across"], [-0.8, 0.6, 0.0], atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(frame["up"], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(frame["origin"], SENSOR_ORIGIN)
    basis = np.stack([frame["along"], frame["across"], frame["up"]])
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=ABSOLUTE_TOLERANCE)
    np.testing.assert_allclose(np.cross(frame["along"], frame["across"]), frame["up"], atol=ABSOLUTE_TOLERANCE)
    # The box centre lies on the plane, at along-distance 5 and its own height.
    mask, coordinates = in_slab(BOX_CENTRE[None], frame, ABSOLUTE_TOLERANCE)
    assert mask.tolist() == [True]
    np.testing.assert_allclose(coordinates, [[HORIZONTAL_DISTANCE, BOX_CENTRE[2]]], atol=ABSOLUTE_TOLERANCE)


def test_in_slab_keeps_points_within_half_width_with_exact_coordinates():
    """Points placed by origin + u*along + w*across + z: kept iff |w| <= half width."""
    frame = slice_frame(BOX_CENTRE, SENSOR_ORIGIN)
    placements = [(2.0, 0.0, 1.0), (3.0, JUST_INSIDE_OFFSET, 0.2), (4.0, -JUST_INSIDE_OFFSET, 0.3),
                  (2.5, JUST_OUTSIDE_OFFSET, 0.4), (2.5, -JUST_OUTSIDE_OFFSET, 0.4)]
    points = np.array([
        (frame["origin"] + along_value * frame["along"] + across_value * frame["across"]
         + np.array([0.0, 0.0, height - frame["origin"][2]]))
        for along_value, across_value, height in placements])
    mask, coordinates = in_slab(points, frame, SLAB_HALF_WIDTH)
    assert mask.tolist() == [True, True, True, False, False]
    np.testing.assert_allclose(coordinates, [[2.0, 1.0], [3.0, 0.2], [4.0, 0.3]], atol=1e-9)


def test_nearest_origin_picks_horizontally_nearest_ignoring_height():
    """Three origins (queries given with z; xy only is used too); the third is nearest in xy despite a huge z offset."""
    origins = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [5.0, 5.0, 100.0]])
    queries = np.array([[1.0, 0.0, 0.0], [9.0, 1.0, 0.0], [5.0, 4.0, 0.0]])
    indices, nearest = nearest_origin(queries, origins)
    assert indices.tolist() == [0, 1, 2]
    np.testing.assert_allclose(nearest, origins[[0, 1, 2]])


def test_coincident_origin_raises_even_with_height_difference():
    """Directly above the sensor the horizontal direction is undefined."""
    with pytest.raises(ValueError):
        slice_frame(np.array([1.0, 2.0, 9.0]), SENSOR_ORIGIN)


def test_session_sweep_origins_reads_pose_translations_without_frames(tmp_path):
    """Poses file only (frame files are empty placeholders, never decoded)."""
    sweep_count = 3
    dataset_dir = tmp_path / "session"
    (dataset_dir / "lidar" / SENSOR_NAME).mkdir(parents=True)
    (dataset_dir / "manifest.json").write_text(json.dumps(
        {"schema_version": SCHEMA_VERSION, "sensors": [{"name": SENSOR_NAME, "kind": "lidar"}]}))
    poses = np.tile(np.eye(4), (sweep_count, 1, 1))
    expected_origins = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    poses[:, :3, 3] = expected_origins
    np.savez(dataset_dir / "poses.npz", **{
        f"{SENSOR_NAME}{POSE_KEY_SEPARATOR}times": np.arange(sweep_count, dtype=np.float64),
        f"{SENSOR_NAME}{POSE_KEY_SEPARATOR}poses_world_sensor": poses})
    for frame_index in range(sweep_count):
        (dataset_dir / "lidar" / SENSOR_NAME / f"{frame_index:06d}_{frame_index + 1}.npy").write_bytes(b"")
    np.testing.assert_allclose(session_sweep_origins(dataset_dir), expected_origins)
