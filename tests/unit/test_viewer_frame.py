"""Tests for the map -> SuperSplat viewer frame transform (T88) and its sidecar record."""

import json
import os
import stat

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viking_splatad.export.viewer_frame import (
    MAP_FRAME_NAME, SUPERSPLAT_FROM_MAP_ROTATION, SUPERSPLAT_LOAD_ROTATION, VIEWER_FRAME_FILE_NAME,
    VIEWER_FRAME_NAME, check_record_matches_ply, map_to_viewer_frame, path_origin, read_viewer_frame_record,
    rotate_directions_to_viewer, stage_viewer_frame_record, viewer_to_map_points, write_viewer_frame_record)

RANDOM_SEED = 20261122
GAUSSIAN_COUNT = 25
# Tolerances for float64 results of a few matrix products.
ABSOLUTE_TOLERANCE = 1e-9
MAP_UP = np.array([0.0, 0.0, 1.0])


@pytest.fixture
def random_generator():
    """Seeded generator, so every test is deterministic."""
    return np.random.default_rng(RANDOM_SEED)


@pytest.mark.covers("T88")
def test_rotation_is_proper_and_maps_up_to_engine_up():
    """M is a read-only proper rotation, and map up ends as engine +y after SuperSplat's Rz(180)."""
    matrix = SUPERSPLAT_FROM_MAP_ROTATION
    assert np.allclose(matrix.T @ matrix, np.eye(3), atol=ABSOLUTE_TOLERANCE)
    assert np.linalg.det(matrix) == pytest.approx(1.0)
    # Map up -> file -y; SuperSplat's load rotation then gives the engine +y (up).
    assert np.allclose(matrix @ MAP_UP, [0.0, -1.0, 0.0])
    assert np.allclose(SUPERSPLAT_LOAD_ROTATION @ (matrix @ MAP_UP), [0.0, 1.0, 0.0])
    assert not matrix.flags.writeable


@pytest.mark.covers("T88")
def test_hand_computed_example():
    """One point through p' = M (p - o), checked against a hand computation."""
    # (11,20,4) - (10,20,1) = (1,0,3); M (1,0,3) = (1, -3, 0).
    means, _ = map_to_viewer_frame(np.array([[11.0, 20.0, 4.0]]), np.eye(3)[None],
                                   np.array([10.0, 20.0, 1.0]))
    assert np.allclose(means, [[1.0, -3.0, 0.0]])


@pytest.mark.covers("T88")
@pytest.mark.covers("T89")
def test_round_trip_points(random_generator):
    """viewer_to_map_points inverts the point part of map_to_viewer_frame."""
    points = random_generator.normal(scale=50.0, size=(GAUSSIAN_COUNT, 3))
    origin = random_generator.normal(size=3)
    rotations = Rotation.random(GAUSSIAN_COUNT, random_state=RANDOM_SEED).as_matrix()
    means_viewer, _ = map_to_viewer_frame(points, rotations, origin)
    assert np.allclose(viewer_to_map_points(means_viewer, origin), points, atol=ABSOLUTE_TOLERANCE)


@pytest.mark.covers("T88")
def test_covariance_transforms_as_m_sigma_mt_and_distances_kept(random_generator):
    """R' = M R gives covariance M Sigma M^T, and the point map preserves pairwise distances."""
    rotations = Rotation.random(GAUSSIAN_COUNT, random_state=RANDOM_SEED).as_matrix()
    scales = random_generator.uniform(0.1, 2.0, size=(GAUSSIAN_COUNT, 3))
    means = random_generator.normal(scale=20.0, size=(GAUSSIAN_COUNT, 3))
    origin = random_generator.normal(size=3)
    matrix = SUPERSPLAT_FROM_MAP_ROTATION
    means_viewer, rotations_viewer = map_to_viewer_frame(means, rotations, origin)
    for gaussian_index in range(GAUSSIAN_COUNT):
        scale_squared = np.diag(scales[gaussian_index] ** 2)
        covariance_map = rotations[gaussian_index] @ scale_squared @ rotations[gaussian_index].T
        covariance_viewer = (rotations_viewer[gaussian_index] @ scale_squared
                             @ rotations_viewer[gaussian_index].T)
        assert np.allclose(covariance_viewer, matrix @ covariance_map @ matrix.T,
                           atol=ABSOLUTE_TOLERANCE)
    # A rotation plus translation is an isometry: pairwise distances are unchanged.
    distances_map = np.linalg.norm(means[:, None] - means[None], axis=2)
    distances_viewer = np.linalg.norm(means_viewer[:, None] - means_viewer[None], axis=2)
    assert np.allclose(distances_map, distances_viewer, atol=ABSOLUTE_TOLERANCE)


@pytest.mark.covers("T88")
@pytest.mark.covers("T90")
def test_rotate_directions_equals_m_d_and_ignores_origin(random_generator):
    """Directions map as unit M d, consistent with the difference of two transformed points."""
    directions = random_generator.normal(size=(GAUSSIAN_COUNT, 3))
    unit_directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    expected = (SUPERSPLAT_FROM_MAP_ROTATION @ unit_directions.T).T
    assert np.allclose(rotate_directions_to_viewer(directions), expected, atol=ABSOLUTE_TOLERANCE)
    # Same result whatever origin a point transform uses: directions take no origin argument, and
    # a difference of two points rotates exactly like a direction.
    origin = random_generator.normal(size=3)
    start, end = np.zeros((1, 3)), directions[:1]
    start_viewer, _ = map_to_viewer_frame(start, np.eye(3)[None], origin)
    end_viewer, _ = map_to_viewer_frame(end, np.eye(3)[None], origin)
    difference = (end_viewer - start_viewer)[0]
    assert np.allclose(difference / np.linalg.norm(difference),
                       rotate_directions_to_viewer(directions[:1])[0], atol=ABSOLUTE_TOLERANCE)


@pytest.mark.covers("T88")
def test_path_origin_is_the_mean():
    """The origin is the arithmetic mean of the positions; empty, NaN and (P, 2) inputs raise."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0], [4.0, 8.0, 3.0]])
    assert np.allclose(path_origin(positions), positions.sum(axis=0) / len(positions))
    with pytest.raises(ValueError):
        path_origin(np.zeros((0, 3)))
    with pytest.raises(ValueError):
        path_origin(np.array([[np.nan, 0.0, 0.0]]))
    with pytest.raises(ValueError):
        path_origin(np.zeros((4, 2)))


@pytest.mark.covers("T88")
@pytest.mark.covers("T89")
@pytest.mark.covers("T90")
def test_validation_errors():
    """Wrong shapes, NaN, improper or scaled frame rotations and zero directions raise ValueError."""
    good_means, good_rotations, good_origin = np.zeros((2, 3)), np.tile(np.eye(3), (2, 1, 1)), np.zeros(3)
    with pytest.raises(ValueError):
        map_to_viewer_frame(np.zeros((2, 2)), good_rotations, good_origin)
    with pytest.raises(ValueError):
        map_to_viewer_frame(good_means, np.zeros((3, 3, 3)), good_origin)
    with pytest.raises(ValueError):
        map_to_viewer_frame(good_means, good_rotations, np.zeros(2))
    with pytest.raises(ValueError):
        map_to_viewer_frame(np.full((2, 3), np.nan), good_rotations, good_origin)
    # A mirror (det -1) and a scaled matrix are both rejected as the frame rotation.
    mirror = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError):
        map_to_viewer_frame(good_means, good_rotations, good_origin, rotation=mirror)
    with pytest.raises(ValueError):
        map_to_viewer_frame(good_means, good_rotations, good_origin, rotation=2.0 * np.eye(3))
    with pytest.raises(ValueError):
        viewer_to_map_points(good_means, good_origin, rotation=mirror)
    with pytest.raises(ValueError):
        rotate_directions_to_viewer(np.zeros((1, 3)))


@pytest.mark.covers("T88")
def test_sidecar_round_trip_and_mode(tmp_path):
    """The sidecar reads back exactly, leaves no temp file and gets mode 0o666 & ~umask."""
    origin = np.array([10.5, -3.25, 1.0])
    crop = {"min": [0.0, 1.0, 2.0], "max": [3.0, 4.0, 5.0], "note": "box"}
    record_path = tmp_path / VIEWER_FRAME_FILE_NAME
    write_viewer_frame_record(record_path, origin, SUPERSPLAT_FROM_MAP_ROTATION, crop)
    record = read_viewer_frame_record(record_path)
    assert np.array_equal(record["origin_map"], origin)
    assert np.array_equal(record["rotation"], SUPERSPLAT_FROM_MAP_ROTATION)
    assert record["crop"] == crop
    # No temp file is left behind and the mode is 0o666 & ~umask.
    assert [entry.name for entry in tmp_path.iterdir()] == [VIEWER_FRAME_FILE_NAME]
    current_umask = os.umask(0)
    os.umask(current_umask)
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o666 & ~current_umask


@pytest.mark.covers("T88")
def test_sidecar_strict_reading(tmp_path):
    """Missing keys, wrong shapes and improper rotations are refused on read and on write."""
    record_path = tmp_path / VIEWER_FRAME_FILE_NAME
    write_viewer_frame_record(record_path, np.zeros(3), SUPERSPLAT_FROM_MAP_ROTATION, {})
    original = json.loads(record_path.read_text())

    def write_variant(changes: dict, dropped_key: str | None = None) -> None:
        """Rewrite the sidecar with ``changes`` applied and ``dropped_key`` removed."""
        variant = {**original, **changes}
        if dropped_key is not None:
            del variant[dropped_key]
        record_path.write_text(json.dumps(variant))

    write_variant({}, dropped_key="origin_map")
    with pytest.raises(ValueError):
        read_viewer_frame_record(record_path)
    write_variant({"origin_map": [0.0, 0.0]})
    with pytest.raises(ValueError):
        read_viewer_frame_record(record_path)
    write_variant({"rotation_viewer_from_map": np.diag([1.0, 1.0, -1.0]).tolist()})
    with pytest.raises(ValueError):
        read_viewer_frame_record(record_path)
    write_variant({"rotation_viewer_from_map": [[1.0, 0.0], [0.0, 1.0]]})
    with pytest.raises(ValueError):
        read_viewer_frame_record(record_path)
    # Writing an improper rotation is refused and leaves no file.
    refused_path = tmp_path / "refused.json"
    with pytest.raises(ValueError):
        write_viewer_frame_record(refused_path, np.zeros(3), np.diag([1.0, 1.0, -1.0]), {})
    assert not refused_path.exists()


@pytest.mark.covers("T88")
def test_sidecar_write_rejects_non_dict_crop_and_nan_crop(tmp_path):
    """A non-dict crop or a NaN inside it is refused and leaves neither the file nor a temp file."""
    record_path = tmp_path / VIEWER_FRAME_FILE_NAME
    with pytest.raises(ValueError, match="crop must be a dict"):
        write_viewer_frame_record(record_path, np.zeros(3), SUPERSPLAT_FROM_MAP_ROTATION, [1, 2])
    with pytest.raises(ValueError):
        write_viewer_frame_record(record_path, np.zeros(3), SUPERSPLAT_FROM_MAP_ROTATION, {"min": float("nan")})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.covers("T88")
def test_sidecar_read_rejects_wrong_frame_name(tmp_path):
    """A record whose 'frame' is not the viewer frame name is refused."""
    record_path = tmp_path / VIEWER_FRAME_FILE_NAME
    write_viewer_frame_record(record_path, np.zeros(3), SUPERSPLAT_FROM_MAP_ROTATION, {})
    record = json.loads(record_path.read_text())
    record["frame"] = "some_other_frame"
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="frame"):
        read_viewer_frame_record(record_path)


@pytest.mark.covers("T88")
def test_map_frame_record_round_trip_and_identity_enforced(tmp_path):
    """A map-frame record reads back with its frame, size and count; a non-identity one is refused."""
    record_path = tmp_path / VIEWER_FRAME_FILE_NAME
    write_viewer_frame_record(record_path, np.zeros(3), np.eye(3), {"mode": "none"}, frame_name=MAP_FRAME_NAME,
                              ply_bytes=1234, vertex_count=7)
    record = read_viewer_frame_record(record_path)
    assert record["frame"] == MAP_FRAME_NAME and record["ply_bytes"] == 1234 and record["vertex_count"] == 7
    np.testing.assert_array_equal(record["rotation"], np.eye(3))
    # Writing a map record with a shift or a rotation is refused, and nothing is left behind.
    other_path = tmp_path / "other.json"
    with pytest.raises(ValueError, match="identity"):
        write_viewer_frame_record(other_path, np.ones(3), np.eye(3), {}, frame_name=MAP_FRAME_NAME)
    with pytest.raises(ValueError, match="identity"):
        write_viewer_frame_record(other_path, np.zeros(3), SUPERSPLAT_FROM_MAP_ROTATION, {},
                                  frame_name=MAP_FRAME_NAME)
    with pytest.raises(ValueError, match="frame_name"):
        write_viewer_frame_record(other_path, np.zeros(3), np.eye(3), {}, frame_name="nonsense")
    assert [entry.name for entry in tmp_path.iterdir()] == [VIEWER_FRAME_FILE_NAME]
    # Reading a hand-edited map record with a shift is refused too.
    edited = json.loads(record_path.read_text())
    edited["origin_map"] = [1.0, 0.0, 0.0]
    record_path.write_text(json.dumps(edited))
    with pytest.raises(ValueError, match="identity"):
        read_viewer_frame_record(record_path)


@pytest.mark.covers("T88")
def test_record_without_counts_reads_as_none_and_bad_counts_are_refused(tmp_path):
    """ply_bytes / vertex_count are optional on read (None); negative, boolean or fractional ones are refused."""
    record_path = tmp_path / VIEWER_FRAME_FILE_NAME
    write_viewer_frame_record(record_path, np.zeros(3), SUPERSPLAT_FROM_MAP_ROTATION, {})
    record = read_viewer_frame_record(record_path)
    assert record["frame"] == VIEWER_FRAME_NAME and record["ply_bytes"] is None and record["vertex_count"] is None
    for bad_value in (-1, True, 1.5):
        with pytest.raises(ValueError):
            write_viewer_frame_record(tmp_path / "bad.json", np.zeros(3), SUPERSPLAT_FROM_MAP_ROTATION, {},
                                      ply_bytes=bad_value)
    assert [entry.name for entry in tmp_path.iterdir()] == [VIEWER_FRAME_FILE_NAME]


@pytest.mark.covers("T88")
def test_check_record_matches_ply_compares_the_byte_size(tmp_path):
    """Equal size passes, a different size raises a re-export message, an unrecorded size is not checked."""
    ply_path = tmp_path / "supersplat.ply"
    ply_path.write_bytes(b"x" * 10)
    check_record_matches_ply({"ply_bytes": 10}, ply_path)
    check_record_matches_ply({"ply_bytes": None}, ply_path)
    with pytest.raises(ValueError, match="re-export"):
        check_record_matches_ply({"ply_bytes": 11}, ply_path)


@pytest.mark.covers("T88")
def test_staged_record_is_not_in_place_until_replaced(tmp_path):
    """stage_viewer_frame_record leaves the target untouched; os.replace publishes it."""
    record_path = tmp_path / VIEWER_FRAME_FILE_NAME
    staged_path = stage_viewer_frame_record(record_path, np.zeros(3), SUPERSPLAT_FROM_MAP_ROTATION, {})
    assert staged_path.is_file() and not record_path.exists()
    os.replace(staged_path, record_path)
    assert read_viewer_frame_record(record_path)["frame"] == VIEWER_FRAME_NAME
    assert [entry.name for entry in tmp_path.iterdir()] == [VIEWER_FRAME_FILE_NAME]


@pytest.mark.covers("T89")
def test_viewer_to_map_points_hand_computed_example():
    """One point through p = M^T p' + o, checked against a hand computation."""
    # M^T = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]; M^T (1, -3, 0) = (1, 0, 3); + (10, 20, 1) = (11, 20, 4).
    points_map = viewer_to_map_points(np.array([[1.0, -3.0, 0.0]]), np.array([10.0, 20.0, 1.0]))
    assert np.allclose(points_map, [[11.0, 20.0, 4.0]])


@pytest.mark.covers("T90")
def test_rotate_directions_to_viewer_hand_computed_example():
    """Directions through d' = M d, renormalised, checked against hand computations."""
    # Map up (0, 0, 2): M d = (0, -2, 0), unit (0, -1, 0). (3, 0, 4): M d = (3, -4, 0), length 5 -> (0.6, -0.8, 0).
    viewer_directions = rotate_directions_to_viewer(np.array([[0.0, 0.0, 2.0], [3.0, 0.0, 4.0]]))
    assert np.allclose(viewer_directions, [[0.0, -1.0, 0.0], [0.6, -0.8, 0.0]])
